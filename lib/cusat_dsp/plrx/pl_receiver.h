/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL synchronization subsystem.
 *
 * Input: symbol-spaced samples from timing recovery (one complex sample per
 * symbol, unit RMS, carrier counter-rotated by the frontend rotator).
 * Output: descrambled, carrier-corrected XFECFRAMEs (data symbols only) plus
 * per-frame metadata, frame-sync events and link metrics.
 */

#ifndef INCLUDED_CUSAT_DSP_PLRX_PL_RECEIVER_H
#define INCLUDED_CUSAT_DSP_PLRX_PL_RECEIVER_H

#include "framing/plframe.h"
#include "plrx/pl_header_candidates.h"
#include <cusat_dsp/compat.h>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace cusat_dsp {

enum class pl_service_mode : uint8_t {
    ccm,      // one MODCOD / frame size (EN 302 307-1 clause 4.3)
    vcm_acm,  // MODCOD may change frame by frame
};

struct pl_rx_config {
    pl_service_mode mode = pl_service_mode::ccm;
    bool multistream = false;           // MIS: dummy PLFRAMEs expected
    /* PLS codes whose XFECFRAMEs are delivered. Must be
     * non-empty; in ccm mode all entries share one MODCOD and frame size. */
    std::vector<uint8_t> deliver_codes;
    /* Optional: PLFRAME lengths (symbols) considered legitimate in
     * vcm_acm mode. Empty = derive from deliver_codes. */
    std::set<uint32_t> valid_plframe_lens;
    /* Optional: expected PLFRAME length before lock (0 = unknown). */
    uint32_t acq_plframe_hint = 0;
    /* PL scrambling code number n (EN 302 307-1 clause 5.5.4). */
    int scrambling_code = 0;
    /* Worst-case delay, in symbols, between a carrier-frequency change
     * reported by carrier_frequency() and its first effect on the input
     * symbols (frontend batching + GPU pipelining). */
    uint32_t correction_latency_symbols = 0;
    int debug_level = 0;
};

/* One delivered XFECFRAME. Pointers stay valid until the
 * next call into the same pl_receiver. */
struct pl_frame {
    const complexf* data = nullptr; // data symbols, pilots removed
    int n_data = 0;                 // == layout.data_symbols()
    pl_layout layout;
    uint64_t sof_symbol = 0;        // absolute index of the first SOF symbol
    float snr_db = 0.0f;            // per-frame pilot Es/N0 estimate
    bool snr_valid = false;
    float coherence = 0.0f;         // per-frame pilot coherence, [0, 1]
    double residual_cfo = 0.0;      // cycles/symbol measured on this frame
    float pls_confidence = 0.0f;    // PLS decision reliability, [0, 1]
};

enum class pl_sync_event_type : uint8_t {
    header,    // a PLHEADER was accepted at the predicted/searched position
    lost,      // locked -> not locked
    acquired,  // not locked -> locked
};

struct pl_sync_event {
    pl_sync_event_type type;
    uint64_t symbol = 0; // absolute symbol index where the event occurred
};

enum class pl_reset_scope : uint8_t {
    positional,  // input discontinuity: drop buffered symbols and partial
                 // frames; keep carrier/level estimates
    sync,        // also forget frame cadence and PLS history
    full,        // also forget carrier, phase and SNR/coherence history
};

class pl_receiver
{
public:
    explicit pl_receiver(const pl_rx_config& config);
    ~pl_receiver();

    pl_receiver(const pl_receiver&) = delete;
    pl_receiver& operator=(const pl_receiver&) = delete;

    /* Consume all n_syms symbols (always). Writes delivered data symbols to
     * out (capacity max_out) and appends one pl_frame per delivered frame.
     * Frames that do not fit in max_out are kept and emitted on the next
     * call. `candidates` may be null; when present it covers exactly this
     * batch. Events are appended in occurrence order.
     * Returns the number of symbols written to out. */
    int process(const complexf* syms, int n_syms,
                const pl_header_candidates* candidates,
                complexf* out, int max_out,
                std::vector<pl_frame>& frames,
                std::vector<pl_sync_event>& events);

    /* --- frame synchronization ----------------------- */
    /* True while the PLFRAME cadence is confirmed (headers predicted). */
    bool locked() const;
    /* True while a bulk header search would help (not locked). */
    bool wants_header_search() const;
    /* Length of the PLFRAME currently expected, 0 if unknown. */
    uint32_t expected_plframe_len() const;
    /* Consecutive headers whose PLS decision was rejected or overruled. */
    int pls_disagreement_run() const;

    /* --- carrier ---------------------------------------------------- */
    /* Total carrier frequency offset (cycles/symbol) the frontend rotator
     * must remove from the input, valid at any time. */
    double carrier_frequency() const;
    /* True once per change of carrier_frequency(). */
    bool take_carrier_update(double& cycles_per_symbol);

    /* --- metrics -------------------------------------- */
    float snr_db() const;
    bool snr_valid() const;
    float coherence() const;           // smoothed
    float coherence_instant() const;   // last frame
    uint64_t frame_count() const;      // delivered XFECFRAMEs
    uint64_t dummy_count() const;      // dummy PLFRAMEs while locked
    uint64_t rejected_count() const;   // PLFRAMEs not delivered (filtered/invalid)

    /* --- control ---------------------------------------------------- */
    void reset(pl_reset_scope scope);
    void set_valid_plframe_lens(const std::set<uint32_t>& lens);
    void set_acq_plframe_hint(uint32_t plframe_len);
    void set_correction_latency(uint32_t symbols);
    void set_diagnostics(bool on);

private:
    struct impl;
    std::unique_ptr<impl> d_impl;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_PLRX_PL_RECEIVER_H */
