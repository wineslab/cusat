/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * ACM controller: measured-waterfall MODCOD selection with per-mode OLLA.
 * Picks the highest-throughput MODCOD supported by the peer SNR, then uses
 * post-BCH outcomes to learn residual OTA/implementation loss independently
 * for each mode.
 */

#ifndef INCLUDED_CUSAT_DSP_ACM_CONTROLLER_H
#define INCLUDED_CUSAT_DSP_ACM_CONTROLLER_H

#include "acm_config.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace cusat_dsp {

class acm_controller
{
public:
    explicit acm_controller(const std::vector<modcod_def_t>& modcods,
                            float guard_db = ACM_DEFAULT_GUARD_DB,
                            float bootstrap_margin_db =
                                ACM_DEFAULT_BOOTSTRAP_MARGIN_DB)
        : d_table(modcods),
          d_guard_db(guard_db),
          d_bootstrap_margin_db(bootstrap_margin_db)
    {
        std::sort(d_table.begin(), d_table.end(),
                  [](const modcod_def_t& a, const modcod_def_t& b) {
                      return a.spectral_eff < b.spectral_eff;
                  });
    }

    /**
     * @brief Select MODCOD based on peer's reported pilot SNR.
     *
     * Picks the highest-spectral-efficiency MODCOD whose measured post-BCH
     * FER=1e-5 threshold is no greater than snr_db.
     * Falls back to the lowest MODCOD if nothing qualifies.
     */
    uint8_t update_snr(float snr_db, float peer_fer = 0.0f)
    {
        if (d_table.empty())
            return 0;

        // The in-band feedback format uses -99 dB while the reporting
        // receiver is not locked.  It is an explicit invalid-sample marker,
        // not an extremely weak channel estimate.  Counting it during
        // startup would consume the three bootstrap samples at the floor and
        // bring back the slow ladder climb that bootstrap is meant to avoid.
        // After startup it is also direct evidence that the selected mode is
        // unusable at the peer, so retreat and collect a fresh bootstrap.
        constexpr float INVALID_FEEDBACK_SNR_DB = -90.0f;
        if (!std::isfinite(snr_db) || snr_db <= INVALID_FEEDBACK_SNR_DB)
            return drop_to_floor();

        // Downshifts protect the measured waterfall with a small guard and
        // happen immediately.  Steady-state upshifts require more headroom and
        // must remain eligible for five consecutive feedback samples. This
        // avoids ping-ponging when the pilot-SNR estimate straddles a threshold while
        // retaining a sub-feedback-period reaction to a deteriorating link.
        // pilotSNR and the measured waterfall table both use Es/N0. The only
        // subtraction is deliberate headroom for estimator variance and the
        // feedback/MODCOD-switch delay.
        constexpr int UP_CONFIRM_SAMPLES = 5;
        // A clean high-SNR report says which modes are nominally admissible,
        // but it does not prove that the complete OTA path matches every
        // measured AWGN waterfall.  Probe the ladder progressively so FER
        // feedback from each newly entered region can arrive before another
        // large step.  Dense neighbouring modes may be skipped; the required
        // SNR of one confirmed probe advances by at most this amount.  If a
        // deliberately sparse ladder has a larger gap, advance by exactly one
        // entry rather than getting stuck forever.
        constexpr float MAX_UPSHIFT_PROBE_DB = 2.0f;
        // Confirmation alone limits how often an upshift can occur, but it
        // does not give the newly selected mode any room to survive ordinary
        // SNR-estimator jitter.  Keep an explicit gap between the downshift
        // boundary and the upshift boundary.  Without it, a confirmed
        // upshift at exactly threshold+guard was commonly undone by the next
        // slightly-negative slope sample, producing persistent MODCOD
        // chatter on an otherwise stationary OTA link.
        constexpr float UPSHIFT_HYSTERESIS_DB = 1.0f;
        // OTA feedback is normally consumed at 2 Hz.  Predict deterioration
        // about two seconds ahead:
        // rising links already have confirmation hysteresis, whereas falling
        // links must account for feedback and MODCOD-switch latency.
        constexpr float SLOPE_EWMA_ALPHA = 0.35f;
        constexpr float MAX_DELTA_DB_PER_SAMPLE = 1.5f;
        constexpr float LOOKAHEAD_SAMPLES = 4.0f; // 2.0 s at 2 Hz
        constexpr float MAX_LOOKAHEAD_BACKOFF_DB = 1.0f;
        constexpr float FER_BACKOFF_THRESHOLD = 1.0e-3f;

        // Pilot-SNR reports occasionally contain an isolated several-dB dip
        // even on a stationary OTA link. Feeding that sample directly to the
        // slope predictor turns one estimator outlier into a multi-rung
        // downshift. A causal three-sample median rejects one such outlier;
        // a real decline reaches the controller after two reports (normally
        // one second), while feedback loss still invokes drop_to_floor().
        d_snr_window[d_snr_window_next] = snr_db;
        d_snr_window_next = (d_snr_window_next + 1) % d_snr_window.size();
        d_snr_window_count = std::min(d_snr_window_count + 1,
                                      d_snr_window.size());
        float filtered_snr_db = snr_db;
        if (d_snr_window_count == d_snr_window.size()) {
            auto sorted = d_snr_window;
            std::sort(sorted.begin(), sorted.end());
            filtered_snr_db = sorted[1];
        }

        if (std::isfinite(d_previous_snr_db)) {
            float delta = filtered_snr_db - d_previous_snr_db;
            delta = std::max(-MAX_DELTA_DB_PER_SAMPLE,
                             std::min(MAX_DELTA_DB_PER_SAMPLE, delta));
            d_snr_delta_ewma = SLOPE_EWMA_ALPHA * delta +
                (1.0f - SLOPE_EWMA_ALPHA) * d_snr_delta_ewma;
        }
        d_previous_snr_db = filtered_snr_db;

        // The measured waterfall plus the dynamic guard is the nominal link
        // model. OLLA learns the residual error of each MODCOD separately.
        // A failed high-order probation must not lower the admissible SNR of
        // every already-proven lower mode; modulation/demapper implementation
        // loss is mode-specific, and a global accumulator poisoned the whole
        // ladder after one fade.
        const float current_olla_db = mode_olla_offset(d_current_idx);
        const float corrected_snr_db = filtered_snr_db - current_olla_db;
        const float falling_delta = std::min(0.0f, d_snr_delta_ewma);
        const float lookahead_backoff_db = std::min(
            MAX_LOOKAHEAD_BACKOFF_DB,
            -LOOKAHEAD_SAMPLES * falling_delta);
        // Keep prediction in raw Es/N0 space. Each candidate's learned OLLA
        // correction is already included by required_snr_db(); subtracting
        // the current mode's correction here would count it twice and would
        // incorrectly penalize every other mode.
        const float predicted_snr_db = filtered_snr_db -
            lookahead_backoff_db;

        int down_safe = 0;
        for (int i = (int)d_table.size() - 1; i >= 0; i--) {
            if (required_snr_db(i) <= predicted_snr_db - d_guard_db) {
                down_safe = i;
                break;
            }
        }

        int next = d_current_idx;
        const char* reason = "hold";
        // Initial link adaptation: after three valid SNR reports, map directly
        // through the measured waterfall instead of validating every 2 dB
        // region on the way up.  An additional startup-only margin makes the
        // first candidate deliberately conservative.  It remains padding-only
        // until the unchanged OTA probation promotes it, so a model mismatch
        // cannot expose application traffic to an unproven MODCOD.
        if (!d_bootstrap_selected) {
            ++d_bootstrap_snr_samples;
            if (d_bootstrap_snr_samples >= BOOTSTRAP_SNR_SAMPLES &&
                peer_fer <= FER_BACKOFF_THRESHOLD) {
                int bootstrap_safe = 0;
                for (int i = (int)d_table.size() - 1; i >= 0; --i) {
                    if (required_snr_db(i) <= filtered_snr_db - d_guard_db -
                            d_bootstrap_margin_db) {
                        bootstrap_safe = i;
                        break;
                    }
                }
                next = bootstrap_safe;
                reason = "snr-bootstrap";
                d_bootstrap_selected = true;
                d_bootstrap_complete = next <= d_validated_idx;
                d_up_candidate = -1;
                d_up_count = 0;
            }
        // A mode-specific NACK is stronger evidence than the pilot-SNR
        // model.  Retreat below the mode that failed exactly once for this
        // newly reported outcome.  OLLA retains the corresponding dB penalty,
        // so a later re-probe is possible if the channel improves, without a
        // stale EWMA FER repeatedly cascading down the ladder.
        } else if (d_fer_backoff_ceiling_idx >= 0 &&
            d_current_idx > d_fer_backoff_ceiling_idx) {
            next = std::min(down_safe, d_fer_backoff_ceiling_idx);
            reason = "fec-nack";
            d_up_candidate = -1;
            d_up_count = 0;
        } else if (down_safe < d_current_idx) {
            next = down_safe;
            reason = "guarded-down";
            d_up_candidate = -1;
            d_up_count = 0;
        } else {
            int up_safe = d_current_idx;
            for (int i = (int)d_table.size() - 1; i > d_current_idx; i--) {
                if (required_snr_db(i) <= filtered_snr_db - d_guard_db -
                                                UPSHIFT_HYSTERESIS_DB) {
                    up_safe = i;
                    break;
                }
            }
            int up_probe = d_current_idx;
            if (up_safe > d_current_idx) {
                const float probe_limit_db =
                    required_snr_db(d_current_idx) + MAX_UPSHIFT_PROBE_DB;
                for (int i = d_current_idx + 1; i <= up_safe; ++i) {
                    if (required_snr_db(i) <= probe_limit_db)
                        up_probe = i;
                }
                if (up_probe == d_current_idx)
                    up_probe = d_current_idx + 1;
            }
            // Never accumulate an upshift confirmation while the decoder is
            // reporting an active FER alarm, even if SNR alone looks safe.
            // Confirmation and the higher guarded threshold are sufficient
            // for a safe upshift.  Requiring a non-negative SNR derivative
            // can starve one direction forever on a stationary noisy link:
            // its slope EWMA may approach zero from below without crossing it.
            // Do not walk farther up the ladder while the current rung is
            // still a padding-only candidate. Delayed feedback must first
            // either validate it or reject it; this keeps exploration
            // sequential and gives every selected mode an interpretable OTA
            // outcome before the next probe.
            if (d_current_idx <= d_validated_idx &&
                up_probe > d_current_idx &&
                peer_fer <= FER_BACKOFF_THRESHOLD) {
                if (up_probe == d_up_candidate)
                    ++d_up_count;
                else {
                    d_up_candidate = up_probe;
                    d_up_count = 1;
                }
                if (d_up_count >= UP_CONFIRM_SAMPLES) {
                    next = up_probe;
                    reason = up_probe < up_safe ? "confirmed-probe"
                                                : "confirmed-up";
                    d_up_candidate = -1;
                    d_up_count = 0;
                }
            } else {
                d_up_candidate = -1;
                d_up_count = 0;
            }
        }

        d_fer_backoff_ceiling_idx = -1;

        if (next != d_current_idx) {
            fprintf(stderr, "ACM: %s → %s (reason=%s peer_snr=%.2f "
                            "filtered_snr=%.2f corrected_snr=%.2f "
                            "olla=%.3f predicted_snr=%.2f "
                            "slope=%.3f peer_fer=%.2g required_snr=%.2f)\n",
                    d_table[d_current_idx].name, d_table[next].name, reason,
                    snr_db, filtered_snr_db, corrected_snr_db,
                    current_olla_db,
                    predicted_snr_db, d_snr_delta_ewma, peer_fer,
                    required_snr_db(next));
            d_current_idx = next;
            d_validation_successes = 0;
            // A downshift from a proven mode is conservative by construction;
            // make it immediately available to application traffic.  An
            // upshift remains a probe until clean OTA outcomes validate it.
            if (d_current_idx < d_validated_idx)
                d_validated_idx = d_current_idx;
            if (d_bootstrap_selected && d_current_idx <= d_validated_idx)
                d_bootstrap_complete = true;
        }

        return d_table[d_current_idx].modcod_idx;
    }

    /**
     * @brief Update OLLA from cumulative receiver outcomes, then select a mode.
     *
     * A failed FECFRAME raises that MODCOD's required-SNR correction by
     * STEP_UP_DB. A successful FECFRAME lowers the same correction by
     * STEP_DOWN_DB, chosen so the expected correction is stationary at
     * TARGET_FER. Cumulative per-MODCOD counters make the update independent
     * of feedback cadence and tolerate missed reports.
     */
    uint8_t update_feedback(float snr_db,
                            float peer_fer,
                            uint64_t fec_frames,
                            uint64_t fec_errors,
                            uint8_t rx_modcod = 0)
    {
        update_olla(fec_frames, fec_errors, rx_modcod);
        return update_snr(snr_db, peer_fer);
    }

    uint8_t drop_to_floor()
    {
        if (d_table.empty())
            return 0;
        if (d_current_idx > 0) {
            fprintf(stderr, "ACM: DROP TO FLOOR %s → %s\n",
                    d_table[d_current_idx].name, d_table[0].name);
            d_current_idx = 0;
            d_validated_idx = 0;
            d_validation_successes = 0;
        }
        // A feedback outage invalidates old SNR samples.  Re-run direct
        // initialization when robust signaling returns, while retaining the
        // learned per-MODCOD OLLA offsets from before the outage.
        d_bootstrap_selected = false;
        d_bootstrap_complete = false;
        d_bootstrap_snr_samples = 0;
        d_up_candidate = -1;
        d_up_count = 0;
        d_previous_snr_db = NAN;
        d_snr_delta_ewma = 0.0f;
        d_snr_window_next = 0;
        d_snr_window_count = 0;
        return d_table[0].modcod_idx;
    }

    uint8_t current_modcod_idx() const
    {
        return d_table.empty() ? 0 : d_table[d_current_idx].modcod_idx;
    }

    // Highest mode that has either already carried a clean OTA probation
    // burst or is a conservative downshift from one that did.  The TX may
    // probe current_modcod_idx() while idle, but should carry application
    // PDUs on this mode until the candidate is validated.
    uint8_t validated_modcod_idx() const
    {
        return d_table.empty() ? 0 : d_table[d_validated_idx].modcod_idx;
    }

    float olla_offset_db() const
    {
        return d_table.empty() ? 0.0f : mode_olla_offset(d_current_idx);
    }

    float olla_offset_db(uint8_t modcod_idx) const
    {
        return d_olla_offset_db[modcod_idx];
    }

    bool bootstrap_complete() const { return d_bootstrap_complete; }
    float bootstrap_margin_db() const { return d_bootstrap_margin_db; }

private:
    float mode_olla_offset(int table_idx) const
    {
        return d_olla_offset_db[d_table[table_idx].modcod_idx];
    }

    float required_snr_db(int table_idx) const
    {
        return d_table[table_idx].min_snr_db + mode_olla_offset(table_idx);
    }

    void update_olla(uint64_t fec_frames,
                     uint64_t fec_errors,
                     uint8_t rx_modcod)
    {
        if (!d_outcomes_initialized[rx_modcod]) {
            // Receiver counters are process-epoch, per-MODCOD totals.  Their
            // first report therefore contains valid outcomes, notably the
            // idle probation burst for a newly selected mode.  Treating that
            // report as a baseline silently discarded the very NACKs meant
            // to keep first application traffic off an unsafe MODCOD.
            d_last_fec_frames[rx_modcod] = 0;
            d_last_fec_errors[rx_modcod] = 0;
            d_outcomes_initialized[rx_modcod] = true;
        }

        // Counters restart after receiver re-acquisition. Rebase without
        // treating the reset as either a huge success burst or a NACK burst;
        // retain the learned link correction across the outage.
        if (fec_frames < d_last_fec_frames[rx_modcod] ||
            fec_errors < d_last_fec_errors[rx_modcod]) {
            d_last_fec_frames[rx_modcod] = fec_frames;
            d_last_fec_errors[rx_modcod] = fec_errors;
            return;
        }

        const uint64_t delta_frames =
            fec_frames - d_last_fec_frames[rx_modcod];
        const uint64_t raw_delta_errors =
            fec_errors - d_last_fec_errors[rx_modcod];
        d_last_fec_frames[rx_modcod] = fec_frames;
        d_last_fec_errors[rx_modcod] = fec_errors;
        if (delta_frames == 0)
            return;

        const uint64_t delta_errors = std::min(raw_delta_errors, delta_frames);
        const uint64_t delta_successes = delta_frames - delta_errors;
        const bool outcome_is_current =
            d_table[d_current_idx].modcod_idx == rx_modcod;

        // Keep application data on the last proven mode while the selected
        // higher mode is exercised with padding.  A clean probation long
        // enough to expose gross OTA/implementation mismatches promotes the
        // candidate.  The 1e-5 residual is still controlled continuously by
        // OLLA; this is only the safe-exploration gate.
        if (outcome_is_current && d_current_idx > d_validated_idx) {
            if (delta_errors > 0) {
                d_validation_successes = 0;
            } else {
                d_validation_successes += delta_successes;
                if (d_validation_successes >=
                        ACM_VALIDATION_SUCCESS_FRAMES) {
                    d_validated_idx = d_current_idx;
                    if (d_bootstrap_selected)
                        d_bootstrap_complete = true;
                    d_validation_successes = 0;
                    fprintf(stderr,
                            "ACM: validated %s after %lu clean OTA frames\n",
                            d_table[d_validated_idx].name,
                            (unsigned long)ACM_VALIDATION_SUCCESS_FRAMES);
                }
            }
        }
        // A GEO feedback report can aggregate hundreds of consecutive NACKs
        // emitted before the transmitter can react. They are one control
        // epoch, not hundreds of independent adaptation opportunities: with
        // immediate feedback the first NACK would already have forced the
        // downshift. Bound their OLLA contribution to one step while retaining
        // the observed count for diagnostics and the mode-specific backoff.
        // Isolated NACKs retain the exact target-FER step ratio below.
        // Late fragments of the same failed candidate can arrive after the
        // controller has already retreated.  Do not turn each fragment into
        // another OLLA step; the outcome still updates that mode's
        // backoff ceiling below and remains visible in diagnostics.
        const uint64_t effective_errors = outcome_is_current
            ? std::min<uint64_t>(delta_errors, 1) : 0;
        const uint64_t effective_successes = outcome_is_current
            ? delta_successes : 0;
        float& mode_offset = d_olla_offset_db[rx_modcod];
        const float old_offset = mode_offset;
        const double update = (double)OLLA_STEP_UP_DB * effective_errors -
            (double)OLLA_STEP_DOWN_DB * effective_successes;
        mode_offset = std::max(-d_guard_db,
            std::min(OLLA_MAX_OFFSET_DB,
                     (float)((double)mode_offset + update)));

        if (delta_errors > 0) {
            for (size_t i = 0; i < d_table.size(); ++i) {
                if (d_table[i].modcod_idx == rx_modcod) {
                    const int ceiling = std::max(0, (int)i - 1);
                    if (d_fer_backoff_ceiling_idx < 0)
                        d_fer_backoff_ceiling_idx = ceiling;
                    else
                        d_fer_backoff_ceiling_idx = std::min(
                            d_fer_backoff_ceiling_idx, ceiling);
                    break;
                }
            }
            fprintf(stderr,
                    "ACM OLLA: mc=%u frames=+%lu errors=+%lu effective=+%lu "
                    "offset=%.3f→%.3f dB "
                    "(target_fer=%.1e)\n",
                    (unsigned)rx_modcod,
                    (unsigned long)delta_frames,
                    (unsigned long)delta_errors,
                    (unsigned long)effective_errors,
                    old_offset, mode_offset, OLLA_TARGET_FER);
        }
    }

    // Cellular-style OLLA, adapted to the post-BCH DVB-S2X FECFRAME target.
    // The asymmetric step sizes make E[Δoffset]=0 at TARGET_FER.
    static constexpr float OLLA_TARGET_FER = 1.0e-5f;
    static constexpr float OLLA_STEP_UP_DB = 0.5f;
    static constexpr float OLLA_STEP_DOWN_DB =
        OLLA_STEP_UP_DB * OLLA_TARGET_FER / (1.0f - OLLA_TARGET_FER);
    // The wide upper bound matters on exceptionally clean/high-SNR test links:
    // implementation losses can still require leaving the top MODCOD even
    // though raw SNR is far above every waterfall threshold.
    static constexpr float OLLA_MAX_OFFSET_DB = 20.0f;
    // Candidate probes are deliberately short so an invalid high-order mode
    // cannot overload an embedded decoder. Accumulate evidence over several
    // feedback epochs before exposing application data. This gate catches
    // gross OTA/model mismatches; the measured waterfall, guard, and
    // continuous OLLA still enforce the long-term 1e-5 target.
    static constexpr uint64_t ACM_VALIDATION_SUCCESS_FRAMES = 64;
    static constexpr int BOOTSTRAP_SNR_SAMPLES = 3;

    std::vector<modcod_def_t> d_table;
    int d_current_idx = 0;
    int d_validated_idx = 0;
    bool d_bootstrap_selected = false;
    bool d_bootstrap_complete = false;
    int d_bootstrap_snr_samples = 0;
    uint64_t d_validation_successes = 0;
    int d_up_candidate = -1;
    int d_up_count = 0;
    float d_previous_snr_db = NAN;
    float d_snr_delta_ewma = 0.0f;
    std::array<float, 3> d_snr_window{};
    size_t d_snr_window_next = 0;
    size_t d_snr_window_count = 0;
    float d_guard_db = ACM_DEFAULT_GUARD_DB;
    float d_bootstrap_margin_db = ACM_DEFAULT_BOOTSTRAP_MARGIN_DB;
    std::array<float, 256> d_olla_offset_db{};
    int d_fer_backoff_ceiling_idx = -1;
    std::array<bool, 256> d_outcomes_initialized{};
    std::array<uint64_t, 256> d_last_fec_frames{};
    std::array<uint64_t, 256> d_last_fec_errors{};
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_ACM_CONTROLLER_H */
