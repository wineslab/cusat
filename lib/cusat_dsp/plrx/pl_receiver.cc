/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL synchronization subsystem.
 *
 * Overview
 * --------
 * The receiver works on the absolute symbol axis: every input symbol has
 * an index (since construction / full reset) and all decisions are taken
 * when the symbols they need are available, independently of how the
 * input was split into batches (R1.10). Symbols are read in place from
 * the caller's batch; only the tail that later decisions still need (an
 * unfinished PLFRAME, a header straddling the batch end) is copied into a
 * carry buffer.
 *
 * States:
 *   SEARCH    candidate PLHEADERs (GPU bulk search, or the CPU detector
 *             below) are verified by a full header test; each verified
 *             header opens a track that predicts the next header from the
 *             PLFRAME length of its PLS decision.
 *   SETTLE    a track confirmed at kLockHeaders consecutive headers: the
 *             carrier estimate of the track is handed to the frontend
 *             rotator; the cadence is followed (no delivery) until the
 *             correction has reached the input (correction latency).
 *   LOCKED    headers are predicted from the cadence and tested at the
 *             predicted position (+-1 symbol); every PLFRAME is processed
 *             once the next header is available (end-phase reference).
 *
 * Header test (all states): for a frequency hypothesis f the 90 header
 * symbols are counter-rotated, the SOF correlation gives the phase, the
 * PLS code is decided with the retained ML decoder (framing/plheader.h)
 * among the recognised codes, and the full 90-symbol correlation C with
 * the decided header gives the detection statistic
 *     s = |C|^2 / (90 sum |r|^2)  in [0, 1],
 * amplitude invariant; s ~ rho / (1 + rho) for Es/N0 = rho and ~1/90 for
 * noise (Massey 1972: coherent correlation with the known sync word).
 *
 * Carrier: three estimators of increasing precision and
 * decreasing range:
 *   acquisition  frequency grid of the header test: SOF-only coarse bins
 *                (26-symbol coherence, spacing 1/65 cycle/symbol over
 *                +-0.10) then full-header fine bins (spacing 0.004),
 *                parabolic refinement; range +-0.10, std ~5e-4 at 0 dB.
 *   intermediate Luise & Reggiannini (1995) on the known blocks of a
 *                PLFRAME (header and 36-symbol pilot blocks), lags 1..16:
 *                range +-1/17 = +-0.059 cycle/symbol.
 *   fine         least-squares slope of the block phases (header, pilot
 *                blocks every 16*90+36 = 1476 symbols, next header),
 *                unwrapped around a predictor: range +-1/(2*1476) =
 *                +-3.4e-4 cycle/symbol (Figure 13 / clause 5.5.3 spacing).
 * The acquisition estimate is applied before any frame is delivered; the
 * intermediate estimate resolves the fine estimator's ambiguity; the fine
 * estimate drives a frequency-locked loop on the absolute carrier with a
 * per-frame step limit of kTrackMargin = 1e-4 cycle/symbol (the
 * acquisition-to-tracking margin of §1.6.4, < 3.4e-4). Commands are only
 * issued when the previous one has reached the input (correction
 * latency), so every frame is measured against a known rotator setting
 * and the loop is stable for any latency.
 *
 * Phase: delivered symbols are counter-rotated by the
 * piecewise-linear interpolation of the unwrapped block phases (header,
 * every pilot block, next header): the trajectory follows phase noise and
 * frequency ramps at the pilot rate. The header fixes the absolute phase,
 * so there is no rotation ambiguity.
 */

#include "plrx/pl_receiver.h"
#include "framing/pl_scrambler.h"
#include "framing/plheader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <deque>
#include <stdexcept>
#include <string>

namespace cusat_dsp {

namespace {

constexpr int HDR = pl::kHeaderLen;
constexpr int SOF = pl::kSofLen;
constexpr int PBL = pl::kPilotBlockLen;
constexpr double TWO_PI = 6.283185307179586;

/* ---- detection thresholds on s = |C|^2 / (90 E) ----------------------
 * Noise: E|C|^2 = E / 90 per hypothesis, s ~ Exp(1/90); P(s > T) per
 * hypothesis = exp(-90 T). Signal: s ~ rho / (1 + rho) (0.33 at -3 dB). */
constexpr float ACQ_THRESHOLD = 0.16f;   // exp(-14.4) = 5.6e-7 per hypothesis
constexpr float TRACK_THRESHOLD = 0.16f; // confirmation at a predicted position
constexpr float LOCK_THRESHOLD = 0.11f;  // header present while locked
constexpr int LOCK_HEADERS = 3;          // headers in cadence before lock
constexpr double SETTLE_STEP_NEGLIGIBLE = 2e-4; // c/s: 90-symbol header loss < 0.1 dB
constexpr float SETTLE_SCORE_MARGIN = 0.05f;
constexpr float ACQ_GPU_GATE = 0.24f;    // GPU candidates below this are not verified
                                         // (noise maxima ~0.20; signal >= 0.5 at 0 dB)
constexpr int LOST_MISSES = 3;           // consecutive missed headers -> lost
constexpr int MAX_TRACKS = 24;

/* PLS reliability (scores normalized to 64 for a noiseless code):
 * a change of the data layout in vcm_acm needs a margin above 0.4 (the
 * nearest competing codeword is orthogonal: margin ~1 +- 0.14 at -1 dB). */
constexpr float PLS_RELIABLE_MARGIN = 0.40f;
constexpr float PLS_RELIABLE_SCORE = 0.50f;

/* ---- carrier ---- */
constexpr double ACQ_RANGE = 0.10;          // cycles/symbol, R1.1 SHOULD range
constexpr int ACQ_COARSE_BINS = 13;         // SOF bins spacing 2*ACQ_RANGE/(N-1)
constexpr double ACQ_FINE_STEP = 0.004;     // full-header bins
constexpr int ACQ_FINE_BINS = 7;            // +-3 steps around the coarse bin
constexpr double TRACK_FREQ_STEP = 0.0015;  // confirmation search around a track
constexpr int LR_MAX_LAG = 16;              // L&R lags -> range 1/17
constexpr double TRACK_MARGIN = 1e-4;       // per-frame loop step limit (§1.6.4)
constexpr double CONVERGE_MARGIN = 2e-3;    // step limit right after lock
constexpr int CONVERGE_FRAMES = 6;
constexpr double LOOP_DRIFT = 1e-9;        // cycles/symbol^2 allowed drift (R1.8)
constexpr double ACQ_FREQ_STD = 8e-4;       // carrier uncertainty at lock
constexpr double PILOT_SPACING = 16.0 * 90.0 + 36.0;
constexpr int ALIAS_SEARCH = 4;             // half-alias predictor offsets
constexpr double PN_RATE = TWO_PI * 2e-6;   // phase-noise rate allowed for, rad^2/symbol
constexpr double ALIAS_ALPHA = 0.05;        // alias watchdog averaging
constexpr double PN_Q_ALPHA = 0.1;          // phase-noise rate averaging over frames
constexpr int ALIAS_MIN_FRAMES = 24;

/* EWMA weights */
constexpr float COHERENCE_ALPHA = 0.2f;
constexpr float GAIN_ALPHA = 0.25f;

constexpr float PLS_NORM = 1.0f / 64.0f;

inline float norm2(const complexf& z) { return z.real() * z.real() + z.imag() * z.imag(); }

inline double wrap_pi(double x) { return std::remainder(x, TWO_PI); }

struct hdr_result {
    bool ok = false;     // above the threshold of the test
    float score = 0.0f;  // s
    int offset = 0;      // timing offset tested (-1, 0, +1)
    double freq = 0.0;   // residual frequency, cycles/symbol
    complexf corr{};     // full-header correlation, phase at the header centre
    float energy = 0.0f; // sum |r|^2 over the header
    pls_decision_t dec{ 0, 0.0f, 0.0f };
};

struct track_t {
    uint64_t prev_pos = 0; // header of the hit before the last one
    uint8_t prev_pls = 0;
    uint64_t pos = 0;   // header of the last hit
    int len = 0;        // PLFRAME length of its decision
    double freq = 0.0;  // residual frequency estimate
    double freq_sum = 0.0;
    int hits = 0;
    uint8_t pls = 0;
    float score = 0.0f;
};

/* Known block of a PLFRAME for carrier/SNR estimation. */
struct block_t {
    int start;     // offset from the header start
    int len;
    double centre; // offset of the centre
    complexf sum;  // sum r conj(ref)
    float energy;  // sum |r|^2
    bool pilot;
};

} // namespace

/* ===================================================================== */

struct pl_receiver::impl {
    /* ---- configuration ---- */
    pl_rx_config cfg;
    std::array<pl_layout, pl::kNumPlsCodes> layouts;
    std::array<bool, pl::kNumPlsCodes> is_delivered{};
    std::array<bool, pl::kNumPlsCodes> is_recognised{};
    std::vector<uint8_t> recognised;
    std::set<uint32_t> valid_lens;     // lengths that may change the cadence
    std::unique_ptr<pls_decoder> decoder;
    std::vector<std::array<complexf, HDR>> hdr_ref; // conj-free references
    std::array<complexf, SOF> sof_ref{};
    std::vector<complexf> descr;       // descrambling factors
    std::vector<complexf> pilot_ref;   // conj(descr * pilot) for the max payload
    pl_layout ccm_layout;              // configured layout (ccm)
    std::vector<std::array<complexf, SOF>> sof_bins; // coarse SOF references

    /* ---- symbol stream ---- */
    uint64_t total_in = 0;             // symbols consumed since reset
    const complexf* batch = nullptr;
    int batch_n = 0;
    uint64_t batch_abs = 0;
    std::vector<complexf> carry;       // symbols [carry_abs, carry_abs + carry_n)
    uint64_t carry_abs = 0;
    int carry_n = 0;
    std::vector<complexf> scratch;     // straddling spans
    std::vector<complexf> sanitized;   // copy of a batch with NaN/Inf
    uint64_t search_from = 0;          // CPU detector progress

    /* ---- sync state ---- */
    enum class st { search, settle, locked } state = st::search;
    std::vector<track_t> tracks;
    uint64_t hdr_pos = 0;              // current PLFRAME header
    pl_layout cur;                     // layout of the current PLFRAME
    bool cur_deliver = false;          // current frame may be delivered
    bool cur_valid_hdr = false;        // header of the current frame detected
    hdr_result cur_hdr;
    pl_layout data_layout;             // last confirmed data layout (vcm)
    bool have_data_layout = false;
    int misses = 0;
    int disagree_run = 0;
    uint64_t settle_until = 0;
    int frames_since_lock = 0;
    bool lost_cadence_frame = true;    // current frame not fully observed in cadence

    /* ---- carrier ---- */
    double f_cmd = 0.0;                // commanded rotator frequency
    uint64_t cmd_eff_seen = 0;         // header where the acquisition command was seen effective
    double f_prev = 0.0;               // previous command
    uint64_t cmd_time = 0;             // absolute index of the last command
    bool cmd_pending = false;          // take_carrier_update() not yet called
    bool cmd_in_flight = false;
    double f_best = 0.0;               // best absolute carrier estimate
    double var_best = 1e-6;            // its variance (cycles/symbol)^2
    double alias_acc = 0.0;            // smoothed intermediate - fine estimate
    double alias_var = 0.0;
    int alias_n = 0;
    double f_int_abs = 0.0;            // smoothed intermediate estimate (absolute)
    bool have_int = false;

    /* ---- metrics ---- */
    float snr_db = 0.0f;
    bool snr_ok = false;
    float coh = 0.0f, coh_inst = 0.0f;
    bool coh_init = false;
    float gain_pow = 1.0f;             // smoothed pilot power (amplitude^2)
    bool gain_init = false;
    uint64_t n_frames = 0, n_dummy = 0, n_rejected = 0, n_dropped = 0;
    bool diag = false;

    /* ---- output ---- */
    struct ready_frame {
        pl_frame meta;
        std::vector<complexf> data;    // preallocated, max XFECFRAME
    };
    static constexpr int OUTBOX_SLOTS = 32;
    std::vector<ready_frame> outbox;   // ring of preallocated slots
    int ob_head = 0, ob_count = 0;
    complexf* call_out = nullptr;      // caller's output of the running call
    int call_max = 0, call_written = 0;
    std::vector<pl_frame>* call_frames = nullptr;
    bool pending_lost = false;         // positional reset while locked
    uint64_t pending_lost_at = 0;
    int alt_len = 0;                   // alternative length of the current frame
    std::vector<float> phase_t;        // block centres
    std::vector<double> phase_v;       // unwrapped block phases
    std::vector<double> pred_v;        // predicted block phases
    std::vector<double> res_v, kf_x, kf_p, kf_xp, kf_pp; // phase smoother
    double pn_q = PN_RATE / 8.0;       // estimated phase random-walk rate
    std::vector<double> dev_v;
    std::vector<block_t> blocks;

    /* ================================================================= */

    explicit impl(const pl_rx_config& c);
    void configure_valid_lens();
    void reset(pl_reset_scope scope);

    // stream access
    const complexf* span(uint64_t abs, int len);
    uint64_t avail_end() const { return batch_abs + (uint64_t)batch_n; }
    void retain(uint64_t keep_from);

    // header tests
    hdr_result test_at(uint64_t pos, double f, bool decode, uint8_t forced_code);
    hdr_result test_acq(uint64_t pos);
    hdr_result test_near(uint64_t pos, const double* freqs, int n_freqs, bool timing_search,
                         float threshold);

    // acquisition
    void acquisition(const pl_header_candidates* cand, std::vector<pl_sync_event>& ev);
    void cpu_candidates(pl_header_candidates& out);
    bool track_step(track_t& t, std::vector<pl_sync_event>& ev);
    void enter_locked(const track_t& t, std::vector<pl_sync_event>& ev);

    // locked
    void locked_loop(std::vector<pl_sync_event>& ev);
    bool frame_step(std::vector<pl_sync_event>& ev);
    void declare_lost(std::vector<pl_sync_event>& ev, uint64_t at);
    bool accept_decision(const hdr_result& h, pl_layout& out_layout, bool& deliver);
    bool length_valid(int len) const;
    void process_payload(const hdr_result& next, bool next_ok);

    // carrier loop
    double rotator_for(uint64_t start, int len, bool& ambiguous) const;
    double residual_at(double abs_time) const;
    void loop_update(double f_abs_meas, double r_meas, int frame_len);
    void command(double f);

    // output
    int flush_outbox(complexf* out, int max_out, int written, std::vector<pl_frame>& frames);
};

/* ===================================================================== */
/*  Construction / configuration                                         */
/* ===================================================================== */

pl_receiver::impl::impl(const pl_rx_config& c) : cfg(c)
{
    if (cfg.deliver_codes.empty())
        throw std::invalid_argument("pl_receiver: deliver_codes is empty");
    for (int code = 0; code < pl::kNumPlsCodes; code++)
        layouts[code] = pl_layout_from_code((uint8_t)code);
    for (uint8_t code : cfg.deliver_codes) {
        const pl_layout& l = layouts[code];
        if (!l.deliverable())
            throw std::invalid_argument("pl_receiver: PLS code " + std::to_string(code) +
                                        " is not deliverable (no supported MODCOD/FEC table)");
        if (!l.pilots)
            throw std::invalid_argument("pl_receiver: PLS code " + std::to_string(code) +
                                        " has no pilots; delivered frames require pilots");
        is_delivered[code] = true;
    }
    if (cfg.mode == pl_service_mode::ccm) {
        const pl_layout& l0 = layouts[cfg.deliver_codes[0]];
        for (uint8_t code : cfg.deliver_codes) {
            const pl_layout& l = layouts[code];
            if (l.modcod != l0.modcod || l.fec_size != l0.fec_size)
                throw std::invalid_argument(
                    "pl_receiver: ccm mode requires one MODCOD and one FECFRAME size");
        }
        ccm_layout = l0;
    }
    if (cfg.scrambling_code < 0 || cfg.scrambling_code >= N_GOLD_CODES)
        throw std::invalid_argument("pl_receiver: invalid scrambling code");

    // recognised set
    const bool all_codes = cfg.mode == pl_service_mode::vcm_acm || cfg.multistream;
    for (int code = 0; code < pl::kNumPlsCodes; code++) {
        const pl_layout& l = layouts[code];
        bool r = is_delivered[code] || l.dummy;
        if (all_codes && l.plframe_symbols() > 0)
            r = true;
        is_recognised[code] = r;
        if (r)
            recognised.push_back((uint8_t)code);
    }
    decoder = std::make_unique<pls_decoder>(recognised);

    hdr_ref.resize(pl::kNumPlsCodes);
    for (int code = 0; code < pl::kNumPlsCodes; code++)
        plheader_symbols((uint8_t)code, hdr_ref[code].data());
    for (int k = 0; k < SOF; k++)
        sof_ref[k] = hdr_ref[0][k];
    sof_bins.resize(ACQ_COARSE_BINS);
    for (int b = 0; b < ACQ_COARSE_BINS; b++) {
        const double f = -ACQ_RANGE + 2.0 * ACQ_RANGE * b / (ACQ_COARSE_BINS - 1);
        for (int k = 0; k < SOF; k++) {
            const double ph = TWO_PI * f * (k - (HDR - 1) / 2.0);
            sof_bins[b][k] = sof_ref[k] * complexf((float)std::cos(ph), (float)std::sin(ph));
        }
    }

    const int max_payload = pl::max_payload_symbols();
    descr.resize(max_payload);
    pl_descrambling_sequence(cfg.scrambling_code, descr.data(), max_payload);
    // pilot reference after descrambling: (1 + j) / sqrt2
    const complexf pilot((float)M_SQRT1_2, (float)M_SQRT1_2);
    pilot_ref.resize(max_payload);
    for (int i = 0; i < max_payload; i++)
        pilot_ref[i] = descr[i] * std::conj(pilot); // multiply received -> r conj(ref)

    const int max_plf = pl::max_plframe_symbols();
    carry.resize((size_t)4 * max_plf + 4 * HDR);
    scratch.resize((size_t)max_plf + 4 * HDR);
    outbox.resize(OUTBOX_SLOTS);
    for (ready_frame& rf : outbox)
        rf.data.resize((size_t)pl::max_payload_symbols());
    blocks.reserve(64);
    phase_t.reserve(64);
    phase_v.reserve(64);
    pred_v.reserve(64);
    res_v.reserve(64);
    kf_x.reserve(64);
    kf_p.reserve(64);
    kf_xp.reserve(64);
    kf_pp.reserve(64);
    dev_v.reserve(64);
    tracks.reserve(MAX_TRACKS);
    configure_valid_lens();
}

void pl_receiver::impl::configure_valid_lens()
{
    valid_lens.clear();
    if (cfg.multistream || cfg.valid_plframe_lens.empty()) {
        // every recognised length (multistream: frames of other streams)
        for (uint8_t code : recognised) {
            const pl_layout& l = layouts[code];
            if (cfg.multistream || is_delivered[code] || l.dummy)
                valid_lens.insert((uint32_t)l.plframe_symbols());
        }
    } else {
        valid_lens = cfg.valid_plframe_lens;
    }
    // dummy PLFRAMEs are recognised in every mode and must keep the cadence
    valid_lens.insert((uint32_t)layouts[0].plframe_symbols());
    valid_lens.erase(0u);
}

bool pl_receiver::impl::length_valid(int len) const
{
    return len > 0 && valid_lens.count((uint32_t)len) > 0;
}

void pl_receiver::impl::reset(pl_reset_scope scope)
{
    // positional: cadence, buffered symbols, partial frames
    carry_n = 0;
    carry_abs = total_in;
    search_from = total_in;
    tracks.clear();
    state = st::search;
    misses = 0;
    cur_valid_hdr = false;
    lost_cadence_frame = true;
    if (scope == pl_reset_scope::positional)
        return;
    // sync: PLS history
    disagree_run = 0;
    have_data_layout = false;
    if (scope == pl_reset_scope::sync)
        return;
    // full: carrier, phase, SNR/coherence history, symbol axis
    total_in = 0;
    carry_abs = 0;
    search_from = 0;
    f_cmd = f_prev = f_best = f_int_abs = 0.0;
    have_int = false;
    cmd_pending = false;
    cmd_in_flight = false;
    cmd_time = 0;
    snr_ok = false;
    snr_db = 0.0f;
    coh = coh_inst = 0.0f;
    coh_init = false;
    gain_init = false;
    gain_pow = 1.0f;
    ob_head = ob_count = 0;
}

/* ===================================================================== */
/*  Symbol access                                                        */
/* ===================================================================== */

const complexf* pl_receiver::impl::span(uint64_t abs, int len)
{
    const uint64_t end = abs + (uint64_t)len;
    if (len <= 0 || abs < carry_abs || end > avail_end())
        return nullptr;
    if (abs >= batch_abs)
        return batch + (abs - batch_abs);
    if (end <= carry_abs + (uint64_t)carry_n)
        return carry.data() + (abs - carry_abs);
    // straddles carry and batch
    if ((size_t)len > scratch.size())
        return nullptr;
    const int n_carry = (int)(carry_abs + carry_n - abs);
    memcpy(scratch.data(), carry.data() + (abs - carry_abs), n_carry * sizeof(complexf));
    memcpy(scratch.data() + n_carry, batch, (len - n_carry) * sizeof(complexf));
    return scratch.data();
}

void pl_receiver::impl::retain(uint64_t keep_from)
{
    const uint64_t end = avail_end();
    keep_from = std::max(keep_from, carry_abs);
    if (keep_from >= end) {
        carry_abs = end;
        carry_n = 0;
        return;
    }
    const uint64_t keep_n = end - keep_from;
    if (keep_n > carry.size()) {
        // never needed in practice (one PLFRAME + a header); keep the newest
        keep_from = end - carry.size();
    }
    // part already in the carry buffer
    int moved = 0;
    if (keep_from < batch_abs) {
        const uint64_t c_end = carry_abs + carry_n;
        const int n_old = (int)(c_end - keep_from);
        if (keep_from != carry_abs)
            memmove(carry.data(), carry.data() + (keep_from - carry_abs), n_old * sizeof(complexf));
        moved = n_old;
        memcpy(carry.data() + moved, batch, batch_n * sizeof(complexf));
        moved += batch_n;
    } else {
        const int n_new = (int)(end - keep_from);
        memcpy(carry.data(), batch + (keep_from - batch_abs), n_new * sizeof(complexf));
        moved = n_new;
    }
    carry_abs = keep_from;
    carry_n = moved;
}

/* ===================================================================== */
/*  Header tests                                                         */
/* ===================================================================== */

/* Full test of the header starting at `pos` for residual frequency f
 * (cycles/symbol). decode = false: use forced_code. */
hdr_result pl_receiver::impl::test_at(uint64_t pos, double f, bool decode, uint8_t forced_code)
{
    hdr_result h;
    const complexf* r = span(pos, HDR);
    if (!r)
        return h;
    complexf z[HDR];
    float e = 0.0f;
    // counter-rotation referenced to the header centre
    const double c0 = -(HDR - 1) / 2.0;
    complexf rot((float)std::cos(-TWO_PI * f * c0), (float)std::sin(-TWO_PI * f * c0));
    const complexf step((float)std::cos(-TWO_PI * f), (float)std::sin(-TWO_PI * f));
    for (int k = 0; k < HDR; k++) {
        z[k] = r[k] * rot;
        rot *= step;
        e += norm2(r[k]);
    }
    if (!(e > 1e-30f) || !std::isfinite(e))
        return h;
    uint8_t code = forced_code;
    if (decode) {
        complexf sof{};
        for (int k = 0; k < SOF; k++)
            sof += z[k] * std::conj(sof_ref[k]);
        const float mag = std::abs(sof);
        if (!(mag > 0.0f))
            return h;
        // carrier phase from the SOF, amplitude normalization from it too
        const complexf derot = std::conj(sof) / (mag * mag / SOF);
        complexf plsc[pl::kPlsCodeLen];
        for (int k = 0; k < pl::kPlsCodeLen; k++)
            plsc[k] = z[SOF + k] * derot;
        h.dec = decoder->decode(plsc);
        code = h.dec.pls;
    }
    const std::array<complexf, HDR>& ref = hdr_ref[code];
    complexf c{};
    for (int k = 0; k < HDR; k++)
        c += z[k] * std::conj(ref[k]);
    h.corr = c;
    h.energy = e;
    h.score = norm2(c) / (HDR * e);
    h.freq = f;
    if (!decode) {
        h.dec.pls = code;
        h.dec.score = 64.0f;
        h.dec.margin = INFINITY;
    }
    return h;
}

/* Acquisition test at a candidate position: SOF-only coarse frequency
 * bins over +-ACQ_RANGE, then full-header bins around the best two. */
hdr_result pl_receiver::impl::test_acq(uint64_t pos)
{
    hdr_result best;
    const complexf* r = span(pos, HDR);
    if (!r)
        return best;
    float mags[ACQ_COARSE_BINS];
    for (int b = 0; b < ACQ_COARSE_BINS; b++) {
        complexf s{};
        const std::array<complexf, SOF>& ref = sof_bins[b];
        for (int k = 0; k < SOF; k++)
            s += r[k] * std::conj(ref[k]);
        mags[b] = norm2(s);
    }
    int b1 = 0;
    for (int b = 1; b < ACQ_COARSE_BINS; b++)
        if (mags[b] > mags[b1])
            b1 = b;
    int b2 = b1 == 0 ? 1 : 0;
    for (int b = 0; b < ACQ_COARSE_BINS; b++)
        if (b != b1 && mags[b] > mags[b2])
            b2 = b;
    const double coarse_step = 2.0 * ACQ_RANGE / (ACQ_COARSE_BINS - 1);
    float fine_scores[ACQ_FINE_BINS];
    for (int pass = 0; pass < 2; pass++) {
        const int b = pass == 0 ? b1 : b2;
        const double fc = -ACQ_RANGE + coarse_step * b;
        hdr_result local;
        int ibest = 0;
        for (int i = 0; i < ACQ_FINE_BINS; i++) {
            const double f = fc + (i - ACQ_FINE_BINS / 2) * ACQ_FINE_STEP;
            hdr_result h = test_at(pos, f, true, 0);
            fine_scores[i] = h.score;
            if (h.score > local.score) {
                local = h;
                ibest = i;
            }
        }
        // parabolic refinement of the frequency on sqrt(score) (~|C|)
        if (ibest > 0 && ibest < ACQ_FINE_BINS - 1) {
            const double ym = std::sqrt(fine_scores[ibest - 1]);
            const double y0 = std::sqrt(fine_scores[ibest]);
            const double yp = std::sqrt(fine_scores[ibest + 1]);
            const double den = ym - 2.0 * y0 + yp;
            if (den < 0.0) {
                const double d = 0.5 * (ym - yp) / den;
                if (std::fabs(d) < 1.0) {
                    hdr_result h = test_at(pos, local.freq + d * ACQ_FINE_STEP, true, 0);
                    if (h.score >= local.score * 0.999f)
                        local = h;
                }
            }
        }
        if (local.score > best.score)
            best = local;
    }
    best.ok = best.score >= ACQ_THRESHOLD;
    return best;
}

/* Test around a predicted position: timing offsets {-1, 0, +1} (optional)
 * and the given frequency hypotheses. */
hdr_result pl_receiver::impl::test_near(uint64_t pos, const double* freqs, int n_freqs,
                                        bool timing_search, float threshold)
{
    hdr_result best;
    for (int d = timing_search ? -1 : 0; d <= (timing_search ? 1 : 0); d++) {
        if (pos == 0 && d < 0)
            continue;
        for (int i = 0; i < n_freqs; i++) {
            hdr_result h = test_at(pos + d, freqs[i], true, 0);
            // prefer the predicted position on near-ties
            const float bias = d == 0 ? 1.0f : 0.97f;
            if (h.score * bias > best.score * (best.offset == 0 ? 1.0f : 0.97f)) {
                best = h;
                best.offset = d;
            }
        }
    }
    best.ok = best.score >= threshold;
    return best;
}

/* ===================================================================== */
/*  Acquisition                                                          */
/* ===================================================================== */

/* CPU candidate detector (no GPU search available): differentially
 * coherent correlation of the SOF (25 products) and of the PLSC pairs (32
 * products, whose sign is the b7 bit; EN 302 307-1 clause 5.5.2.4 note),
 * insensitive to the carrier frequency. Statistic normalized by the
 * header energy. */
void pl_receiver::impl::cpu_candidates(pl_header_candidates& out)
{
    out = pl_header_candidates{};
    const uint64_t end = avail_end();
    uint64_t from = std::max(search_from, carry_abs);
    if (end < (uint64_t)HDR + from) {
        out.searched = batch_n;
        return;
    }
    const uint64_t last = end - HDR; // last start with a complete header
    // known differential patterns
    static thread_local std::array<complexf, SOF - 1> dsof;
    static thread_local std::array<complexf, 32> dpls;
    static thread_local bool init = false;
    if (!init) {
        for (int k = 0; k < SOF - 1; k++)
            dsof[k] = std::conj(sof_ref[k + 1] * std::conj(sof_ref[k]));
        // PLSC pairs: symbol 2i+1 vs 2i of the scrambled codeword with b7 = 0
        complexf h[HDR];
        plheader_symbols(0, h);
        for (int i = 0; i < 32; i++)
            dpls[i] = std::conj(h[SOF + 2 * i + 1] * std::conj(h[SOF + 2 * i]));
        init = true;
    }
    float top_s[pl_header_candidates::kMaxCandidates];
    uint64_t top_p[pl_header_candidates::kMaxCandidates];
    int n_top = 0;
    double pow_sum = 0.0;
    int pow_n = 0;
    for (uint64_t p = from; p <= last; p++) {
        const complexf* r = span(p, HDR);
        if (!r)
            continue;
        complexf a{}, b{};
        float e = 0.0f;
        for (int k = 0; k < SOF - 1; k++)
            a += r[k + 1] * std::conj(r[k]) * dsof[k];
        for (int i = 0; i < 32; i++)
            b += r[SOF + 2 * i + 1] * std::conj(r[SOF + 2 * i]) * dpls[i];
        for (int k = 0; k < HDR; k++)
            e += norm2(r[k]);
        pow_sum += norm2(r[0]);
        pow_n++;
        if (!(e > 1e-30f))
            continue;
        const float s = std::max(std::abs(a + b), std::abs(a - b)) / (e * (57.0f / 90.0f));
        // keep the strongest, with +-3 symbol suppression
        int near = -1;
        for (int i = 0; i < n_top; i++)
            if ((p > top_p[i] ? p - top_p[i] : top_p[i] - p) <= 3)
                near = i;
        if (near >= 0) {
            if (s > top_s[near]) {
                top_s[near] = s;
                top_p[near] = p;
            }
            continue;
        }
        if (n_top < pl_header_candidates::kMaxCandidates) {
            top_s[n_top] = s;
            top_p[n_top] = p;
            n_top++;
        } else {
            int w = 0;
            for (int i = 1; i < n_top; i++)
                if (top_s[i] < top_s[w])
                    w = i;
            if (s > top_s[w]) {
                top_s[w] = s;
                top_p[w] = p;
            }
        }
    }
    search_from = last + 1;
    // strongest first
    for (int i = 0; i < n_top; i++)
        for (int j = i + 1; j < n_top; j++)
            if (top_s[j] > top_s[i]) {
                std::swap(top_s[i], top_s[j]);
                std::swap(top_p[i], top_p[j]);
            }
    out.searched = batch_n;
    out.count = n_top;
    for (int i = 0; i < n_top; i++) {
        out.index[i] = (int)((int64_t)top_p[i] - (int64_t)batch_abs);
        out.score[i] = top_s[i];
    }
    out.mean_power = pow_n ? (float)(pow_sum / pow_n) : 0.0f;
}

/* Advance one track if its next header is available. Returns true when
 * the track progressed (hit or drop). */
bool pl_receiver::impl::track_step(track_t& t, std::vector<pl_sync_event>& ev)
{
    if (t.len <= 0)
        return false;
    const uint64_t pred = t.pos + (uint64_t)t.len;
    if (pred + HDR + 1 > avail_end())
        return false;
    double freqs[5];
    int nf = 0;
    for (int i = -2; i <= 2; i++)
        freqs[nf++] = t.freq + i * TRACK_FREQ_STEP;
    hdr_result h = test_near(pred, freqs, nf, true, TRACK_THRESHOLD);
    if (!h.ok) {
        t.len = 0; // dropped
        return true;
    }
    const uint64_t pos = pred + h.offset;
    const pl_layout& l = layouts[h.dec.pls];
    t.prev_pos = t.pos;
    t.prev_pls = t.pls;
    t.pos = pos;
    t.hits++;
    t.freq_sum += h.freq;
    t.freq = h.freq;
    t.pls = h.dec.pls;
    t.score = h.score;
    // next length: the decided layout if it may hold a cadence, else the
    // expected one (ccm)
    int len = l.plframe_symbols();
    if (!length_valid(len))
        len = cfg.mode == pl_service_mode::ccm && !cfg.multistream ? ccm_layout.plframe_symbols()
                                                                   : 0;
    t.len = len;
    ev.push_back({ pl_sync_event_type::header, pos });
    return true;
}

void pl_receiver::impl::acquisition(const pl_header_candidates* cand, std::vector<pl_sync_event>& ev)
{
    // ---- candidates of this batch ----
    pl_header_candidates local;
    const pl_header_candidates* c = cand;
    if (!c) {
        cpu_candidates(local);
        c = &local;
    }
    // sort candidate positions by time so tracks see headers in order
    uint64_t cpos[pl_header_candidates::kMaxCandidates];
    int nc = 0;
    for (int i = 0; i < c->count && i < pl_header_candidates::kMaxCandidates; i++) {
        if (cand && c->score[i] < ACQ_GPU_GATE)
            continue; // same statistic as the CPU test: cannot pass it
        const int64_t a = (int64_t)batch_abs + c->index[i];
        if (a < 0)
            continue;
        cpos[nc++] = (uint64_t)a;
    }
    std::sort(cpos, cpos + nc);

    int ci = 0;
    for (;;) {
        if (state != st::search)
            return;
        // the earliest pending event: a track prediction or a candidate
        uint64_t t_pred = UINT64_MAX;
        int ti = -1;
        for (int i = 0; i < (int)tracks.size(); i++) {
            if (tracks[i].len <= 0)
                continue;
            const uint64_t p = tracks[i].pos + tracks[i].len;
            if (p + HDR + 1 <= avail_end() && p < t_pred) {
                t_pred = p;
                ti = i;
            }
        }
        const uint64_t t_cand = ci < nc ? cpos[ci] : UINT64_MAX;
        if (ti < 0 && ci >= nc)
            break;
        if (ti >= 0 && (ci >= nc || t_pred <= t_cand + 2)) {
            track_t& t = tracks[ti];
            track_step(t, ev);
            if (t.len > 0 && t.hits >= LOCK_HEADERS) {
                enter_locked(t, ev);
                return;
            }
            tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                        [](const track_t& x) { return x.len <= 0; }),
                         tracks.end());
            continue;
        }
        // a new candidate (skip those coinciding with a live track)
        const uint64_t p = cpos[ci++];
        bool dup = false;
        for (const track_t& t : tracks)
            if ((p > t.pos ? p - t.pos : t.pos - p) <= 2 ||
                (t.len > 0 && std::llabs((long long)p - (long long)(t.pos + t.len)) <= 2))
                dup = true;
        if (dup)
            continue;
        hdr_result h = test_acq(p);
        if (!h.ok)
            continue;
        track_t t;
        t.pos = p;
        t.hits = 1;
        t.freq = h.freq;
        t.freq_sum = h.freq;
        t.pls = h.dec.pls;
        t.score = h.score;
        int len = layouts[h.dec.pls].plframe_symbols();
        if (!length_valid(len))
            len = cfg.mode == pl_service_mode::ccm && !cfg.multistream
                      ? ccm_layout.plframe_symbols() : 0;
        t.len = len;
        if (len <= 0)
            continue;
        if ((int)tracks.size() >= MAX_TRACKS) {
            // replace the weakest single-hit track
            int w = -1;
            for (int i = 0; i < (int)tracks.size(); i++)
                if (tracks[i].hits == 1 && (w < 0 || tracks[i].score < tracks[w].score))
                    w = i;
            if (w < 0 || tracks[w].score >= t.score)
                continue;
            tracks[w] = t;
        } else {
            tracks.push_back(t);
        }
    }
}

void pl_receiver::impl::enter_locked(const track_t& t, std::vector<pl_sync_event>& ev)
{
    (void)ev;
    // carrier from the track (mean over its headers), refined on the last
    // complete PLFRAME of the track (pilots), applied before any frame is
    // delivered
    const double f_res = t.freq_sum / std::max(1, t.hits);
    f_best = f_cmd + f_res;
    var_best = ACQ_FREQ_STD * ACQ_FREQ_STD;
    alias_acc = 0.0;
    alias_n = 0;
    state = st::settle;
    frames_since_lock = 0;
    const pl_layout& lp = layouts[t.prev_pls];
    if (t.hits >= 2 && lp.pilots && !lp.dummy && t.pos == t.prev_pos + (uint64_t)lp.plframe_symbols() &&
        span(t.prev_pos, lp.plframe_symbols() + HDR)) {
        hdr_pos = t.prev_pos;
        cur = lp;
        cur_hdr = test_at(t.prev_pos, f_res, false, t.prev_pls);
        cur_valid_hdr = true;
        const hdr_result last = test_at(t.pos, f_res, false, t.pls);
        process_payload(last, true); // settle: carrier only (issues the command)
    }
    if (!cmd_pending)
        command(f_best);
    f_int_abs = f_cmd;
    have_int = true;
    tracks.clear();
    cmd_eff_seen = 0;
    settle_until = cmd_time + cfg.correction_latency_symbols;
    hdr_pos = t.pos;
    // the header at t.pos becomes the current frame; its decision was taken
    // by the track
    hdr_result h = test_at(t.pos, f_best - f_prev, true, 0);
    h.ok = true;
    cur_hdr = h;
    cur_valid_hdr = true;
    bool deliver = false;
    pl_layout l;
    accept_decision(h, l, deliver);
    cur = l;
    cur_deliver = false;
    misses = 0;
    frames_since_lock = 0;
    lost_cadence_frame = true;
    if (diag)
        fprintf(stderr, "pl_receiver: track confirmed at %llu (pls %u, f=%.6f), settling\n",
                (unsigned long long)t.pos, t.pls, f_cmd);
}

/* ===================================================================== */
/*  PLS acceptance                                         */
/* ===================================================================== */

/* Decide the layout of the frame whose header gave decision h. Returns
 * true if the decision was accepted (false: rejected/overruled). */
bool pl_receiver::impl::accept_decision(const hdr_result& h, pl_layout& out, bool& deliver)
{
    const pl_layout& l = layouts[h.dec.pls];
    alt_len = 0;
    const float sc = h.dec.score * PLS_NORM;
    const float mg = std::isfinite(h.dec.margin) ? h.dec.margin * PLS_NORM : 2.0f;
    const bool reliable = sc >= PLS_RELIABLE_SCORE && mg >= PLS_RELIABLE_MARGIN;
    const bool len_ok = length_valid(l.plframe_symbols());
    deliver = false;

    if (cfg.mode == pl_service_mode::ccm && !cfg.multistream) {
        if (l.dummy && len_ok && reliable) {
            out = l;
            return true;
        }
        if (is_delivered[h.dec.pls]) {
            out = l;
            deliver = true;
            return true;
        }
        // (d) disagreement: the frame is handled with the configured layout
        // (the only non-dummy layout of the stream); the FEC checks decide
        out = ccm_layout;
        deliver = true;
        alt_len = len_ok ? l.plframe_symbols() : 0;
        return false;
    }

    // vcm_acm / multistream
    if (!len_ok) {
        // (a) the cadence keeps the expected layout
        out = have_data_layout ? data_layout : l;
        if (!have_data_layout || !length_valid(out.plframe_symbols()))
            out = layouts[0];
        return false;
    }
    const bool transient = l.dummy || l.modcod == 1;
    const bool same_as_data =
        have_data_layout && l.modcod == data_layout.modcod && l.fec_size == data_layout.fec_size &&
        l.pilots == data_layout.pilots && l.dummy == data_layout.dummy;
    if (reliable || same_as_data || (!have_data_layout && sc >= PLS_RELIABLE_SCORE)) {
        out = l;
        deliver = is_delivered[h.dec.pls];
        if (!transient && !l.reserved && l.modcod != 0) {
            data_layout = l;
            have_data_layout = true;
        }
        return true;
    }
    // (b) unreliable change: keep the last confirmed data layout, no
    // delivery; the decided length stays a candidate for the cadence
    out = have_data_layout ? data_layout : l;
    alt_len = l.plframe_symbols();
    return false;
}

/* ===================================================================== */
/*  Locked operation                                                     */
/* ===================================================================== */

void pl_receiver::impl::declare_lost(std::vector<pl_sync_event>& ev, uint64_t at)
{
    if (state == st::locked)
        ev.push_back({ pl_sync_event_type::lost, at });
    state = st::search;
    tracks.clear();
    misses = 0;
    cur_valid_hdr = false;
    lost_cadence_frame = true;
    search_from = at;
    if (diag)
        fprintf(stderr, "pl_receiver: lost at %llu\n", (unsigned long long)at);
}

void pl_receiver::impl::locked_loop(std::vector<pl_sync_event>& ev)
{
    while ((state == st::locked || state == st::settle) && frame_step(ev)) {
    }
}

/* Process the current PLFRAME once its successor's header is available.
 * Returns false when more input is needed. */
bool pl_receiver::impl::frame_step(std::vector<pl_sync_event>& ev)
{
    const int len = cur.plframe_symbols();
    if (len <= 0) {
        declare_lost(ev, hdr_pos);
        return false;
    }
    const uint64_t next = hdr_pos + (uint64_t)len;
    if (next + HDR + 1 > avail_end())
        return false;
    // an alternative length needs its header too
    if (alt_len > 0 && hdr_pos + (uint64_t)alt_len + HDR + 1 > avail_end())
        return false;
    uint64_t next_override = 0;

    // ---- next header: carrier hypotheses of the current rotator state ----
    bool amb = false;
    const double f_rot = rotator_for(next, HDR, amb);
    double freqs[3];
    int nf = 0;
    freqs[nf++] = f_best - f_rot;
    if (state == st::settle || amb)
        freqs[nf++] = f_best - f_prev; // command not yet effective
    hdr_result h = test_near(next, freqs, nf, true, LOCK_THRESHOLD);
    if (state == st::settle && cmd_eff_seen == 0) {
        // The acquisition command reaches the input after at most the
        // configured latency, usually much sooner (unlocked batches are
        // short and not pipelined): once a header clearly shows the new
        // residual, every later symbol is known to be corrected.
        const double step = std::fabs(f_cmd - f_prev);
        bool seen = step < SETTLE_STEP_NEGLIGIBLE;
        if (!seen) {
            const double fn = f_best - f_cmd, fo = f_best - f_prev;
            const hdr_result hn = test_near(next, &fn, 1, true, LOCK_THRESHOLD);
            const hdr_result ho = test_near(next, &fo, 1, true, LOCK_THRESHOLD);
            seen = hn.ok && hn.score > ho.score + SETTLE_SCORE_MARGIN;
        }
        if (seen) {
            cmd_eff_seen = next;
            settle_until = std::min(settle_until, next);
        }
    }
    if (!h.ok && alt_len > 0 && alt_len != len) {
        // the header decision was not trusted: try the decided length
        const uint64_t alt = hdr_pos + (uint64_t)alt_len;
        if (alt + HDR + 1 <= avail_end()) {
            hdr_result ha = test_near(alt, freqs, nf, true, LOCK_THRESHOLD);
            if (ha.ok) {
                // the frame had the decided layout after all; its layout
                // was not confirmed in time: not delivered
                cur = layouts[cur_hdr.dec.pls];
                cur_deliver = false;
                h = ha;
                next_override = alt;
            }
        }
    }
    const uint64_t next_pos = next_override ? next_override : next;

    // ---- the current frame ----
    if (cur_valid_hdr && !cur.dummy) {
        process_payload(h, h.ok); // settle: carrier estimation only
    } else if (state == st::locked && cur.dummy) {
        n_dummy++;
    }

    // ---- advance the cadence ----
    frames_since_lock++;
    if (h.ok) {
        misses = 0;
        const uint64_t pos = next_pos + h.offset;
        ev.push_back({ pl_sync_event_type::header, pos });
        pl_layout l;
        bool deliver = false;
        const bool accepted = accept_decision(h, l, deliver);
        if (accepted)
            disagree_run = 0;
        else
            disagree_run++;
        hdr_pos = pos;
        cur = l;
        cur_deliver = deliver;
        cur_hdr = h;
        cur_valid_hdr = true;
        lost_cadence_frame = false;
    } else {
        misses++;
        disagree_run++;
        if (diag)
            fprintf(stderr, "pl_receiver: header miss at %llu (score %.3f, run %d)\n",
                    (unsigned long long)next, h.score, misses);
        if (misses >= LOST_MISSES) {
            declare_lost(ev, next);
            return true;
        }
        // keep the cadence with the expected layout; nothing delivered
        alt_len = 0;
        hdr_pos = next;
        if (cfg.mode == pl_service_mode::ccm && !cfg.multistream)
            cur = ccm_layout;
        else if (have_data_layout)
            cur = data_layout;
        cur_deliver = false;
        cur_valid_hdr = false;
    }

    if (state == st::settle && hdr_pos >= settle_until) {
        state = st::locked;
        ev.push_back({ pl_sync_event_type::acquired, hdr_pos });
        if (diag)
            fprintf(stderr, "pl_receiver: locked at %llu\n", (unsigned long long)hdr_pos);
    }
    return true;
}

/* Rotator frequency the frontend applied to [start, start + len). */
double pl_receiver::impl::rotator_for(uint64_t start, int len, bool& ambiguous) const
{
    ambiguous = false;
    if (!cmd_in_flight && cmd_time == 0)
        return f_cmd;
    uint64_t eff = cmd_time + cfg.correction_latency_symbols;
    if (cmd_eff_seen >= cmd_time && cmd_eff_seen < eff)
        eff = cmd_eff_seen; // observed earlier (settle)
    if (start >= eff)
        return f_cmd;
    if (start + (uint64_t)len <= cmd_time)
        return f_prev;
    ambiguous = true;
    return f_cmd;
}

/* Carrier residual expected at an absolute time from the loop estimate
 * and the rotator history (midway inside a command's latency window). */
double pl_receiver::impl::residual_at(double t) const
{
    if (cmd_time == 0)
        return f_best - f_cmd;
    if (t < (double)cmd_time)
        return f_best - f_prev;
    uint64_t eff = cmd_time + cfg.correction_latency_symbols;
    if (cmd_eff_seen >= cmd_time && cmd_eff_seen < eff)
        eff = cmd_eff_seen;
    if (t >= (double)eff)
        return f_best - f_cmd;
    return f_best - 0.5 * (f_prev + f_cmd);
}

void pl_receiver::impl::command(double f)
{
    f_prev = f_cmd;
    f_cmd = f;
    cmd_time = std::max<uint64_t>(avail_end(), 1);
    cmd_eff_seen = 0;
    cmd_pending = true;
    cmd_in_flight = true;
}

void pl_receiver::impl::loop_update(double f_abs, double r_meas, int frame_len)
{
    // scalar Kalman filter on the absolute carrier; the innovation of one
    // frame is limited to the acquisition-to-tracking margin once the
    // estimate has converged
    const double margin = frames_since_lock <= CONVERGE_FRAMES ? CONVERGE_MARGIN : TRACK_MARGIN;
    const double raw = f_abs - f_best;
    const double innov = std::max(-margin, std::min(margin, raw));
    const double k = var_best / (var_best + r_meas);
    f_best += k * innov;
    var_best = (1.0 - k) * var_best;
    // a clamped innovation was not absorbed: keep the uncertainty it shows
    if (std::fabs(raw) > margin) {
        const double left = std::fabs(raw) - k * margin;
        var_best = std::max(var_best, left * left);
    }
    // process noise: frequency drift over one frame (LEO-class ramps)
    const double drift = LOOP_DRIFT * (double)frame_len;
    var_best += drift * drift;
    // issue a command when the previous one has reached the input
    if (cmd_in_flight && avail_end() >= cmd_time + cfg.correction_latency_symbols)
        cmd_in_flight = false;
    if (!cmd_in_flight && std::fabs(f_best - f_cmd) > 1e-9)
        command(f_best);
}

/* Frame processing: known-block correlations, carrier and phase, SNR and
 * coherence, delivery. `next` is the test of the following header. */
namespace {

/* Data symbols of one slot: o[i] = (r[i] * d[i]) * rot * step^i, non-finite
 * results zeroed. The rotation runs in NCH independent phasor chains (chain
 * c handles symbols c, c + NCH, ... and advances by step^NCH), vectorised
 * with NEON where available: a single recurrence rot *= step serialises two
 * FMAs per symbol. */
void rotate_slot(const complexf* __restrict__ r, const complexf* __restrict__ d, complexf rot,
                 complexf step, complexf* __restrict__ o)
{
    constexpr int N = pl::kSlotLen;
    constexpr int NCH = 4;
    float cr[NCH], ci[NCH];
    cr[0] = rot.real();
    ci[0] = rot.imag();
    for (int c = 1; c < NCH; c++) {
        cr[c] = cr[c - 1] * step.real() - ci[c - 1] * step.imag();
        ci[c] = cr[c - 1] * step.imag() + ci[c - 1] * step.real();
    }
    float pr = step.real(), pi = step.imag(); // step^NCH
    for (int c = 1; c < NCH; c++) {
        const float t = pr * step.real() - pi * step.imag();
        pi = pr * step.imag() + pi * step.real();
        pr = t;
    }
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vcr = vld1q_f32(cr), vci = vld1q_f32(ci);
    const float32x4_t vpr = vdupq_n_f32(pr), vpi = vdupq_n_f32(pi);
    const float32x4_t vinf = vdupq_n_f32(INFINITY);
    for (; i + NCH <= N; i += NCH) {
        const float32x4x2_t rv = vld2q_f32(reinterpret_cast<const float*>(r + i));
        const float32x4x2_t dv = vld2q_f32(reinterpret_cast<const float*>(d + i));
        const float32x4_t xr = vsubq_f32(vmulq_f32(rv.val[0], dv.val[0]), vmulq_f32(rv.val[1], dv.val[1]));
        const float32x4_t xi = vaddq_f32(vmulq_f32(rv.val[0], dv.val[1]), vmulq_f32(rv.val[1], dv.val[0]));
        float32x4_t vr = vsubq_f32(vmulq_f32(xr, vcr), vmulq_f32(xi, vci));
        float32x4_t vi = vaddq_f32(vmulq_f32(xr, vci), vmulq_f32(xi, vcr));
        const float32x4_t nr = vsubq_f32(vmulq_f32(vcr, vpr), vmulq_f32(vci, vpi));
        vci = vaddq_f32(vmulq_f32(vcr, vpi), vmulq_f32(vci, vpr));
        vcr = nr;
        // |v| < inf is false for inf and NaN: zero non-finite symbols
        const uint32x4_t ok = vandq_u32(vcltq_f32(vabsq_f32(vr), vinf), vcltq_f32(vabsq_f32(vi), vinf));
        float32x4x2_t ov;
        ov.val[0] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vr), ok));
        ov.val[1] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vi), ok));
        vst2q_f32(reinterpret_cast<float*>(o + i), ov);
    }
    vst1q_f32(cr, vcr);
    vst1q_f32(ci, vci);
#else
    for (; i + NCH <= N; i += NCH) {
        for (int c = 0; c < NCH; c++) {
            const int k = i + c;
            const float ar = r[k].real(), ai = r[k].imag(), br = d[k].real(), bi = d[k].imag();
            const float xr = ar * br - ai * bi, xi = ar * bi + ai * br;
            float vr = xr * cr[c] - xi * ci[c], vi = xr * ci[c] + xi * cr[c];
            const float nr = cr[c] * pr - ci[c] * pi;
            ci[c] = cr[c] * pi + ci[c] * pr;
            cr[c] = nr;
            if (!std::isfinite(vr) || !std::isfinite(vi))
                vr = vi = 0.0f;
            o[k] = complexf(vr, vi);
        }
    }
#endif
    for (int c = 0; i < N; i++, c++) { // remainder: chains continue in order
        const float ar = r[i].real(), ai = r[i].imag(), br = d[i].real(), bi = d[i].imag();
        const float xr = ar * br - ai * bi, xi = ar * bi + ai * br;
        float vr = xr * cr[c] - xi * ci[c], vi = xr * ci[c] + xi * cr[c];
        if (!std::isfinite(vr) || !std::isfinite(vi))
            vr = vi = 0.0f;
        o[i] = complexf(vr, vi);
    }
}

} // namespace

void pl_receiver::impl::process_payload(const hdr_result& next, bool next_ok)
{
    const pl_layout& l = cur;
    const int plen = l.plframe_symbols();
    const int payload = l.payload_symbols();
    const complexf* fr = span(hdr_pos, plen);
    if (!fr)
        return;
    const complexf* pl = fr + HDR;
    const int n_pb = l.pilot_block_count();

    if (!l.pilots || n_pb == 0) {
        // pilotless frames are recognised but never delivered
        if (state == st::locked)
            n_rejected++;
        return;
    }

    bool amb = false;
    const double f_rot = rotator_for(hdr_pos, plen, amb);
    const double f_pred = f_best - f_rot; // residual expected on this frame

    // ---- known blocks ----
    blocks.clear();
    {
        block_t b;
        b.start = 0;
        b.len = HDR;
        b.centre = (HDR - 1) / 2.0;
        // header correlation at the frame's own phase reference (centre)
        b.sum = cur_hdr.corr;
        b.energy = cur_hdr.energy;
        b.pilot = false;
        // cur_hdr.corr was taken with its own frequency hypothesis around
        // the centre: the phase at the centre is unaffected
        blocks.push_back(b);
    }
    for (int k = 0; k < n_pb; k++) {
        block_t b;
        const int off = (k + 1) * pl::kSlotsPerPilotGroup * pl::kSlotLen + k * PBL;
        b.start = HDR + off;
        b.len = PBL;
        b.centre = b.start + (PBL - 1) / 2.0;
        // counter-rotate by the expected residual about the block centre
        // (matters for large residuals, e.g. the refinement at lock)
        const double fr_b = residual_at((double)hdr_pos + b.centre);
        const double c0 = -(PBL - 1) / 2.0;
        complexf rot((float)std::cos(-TWO_PI * fr_b * c0), (float)std::sin(-TWO_PI * fr_b * c0));
        const complexf step((float)std::cos(-TWO_PI * fr_b), (float)std::sin(-TWO_PI * fr_b));
        complexf s{};
        float e = 0.0f;
        const complexf* r = pl + off;
        const complexf* ref = pilot_ref.data() + off;
        for (int i = 0; i < PBL; i++) {
            s += r[i] * ref[i] * rot;
            rot *= step;
            e += norm2(r[i]);
        }
        b.sum = s;
        b.energy = e;
        b.pilot = true;
        blocks.push_back(b);
    }
    if (next_ok) {
        block_t b;
        b.start = plen + next.offset;
        b.len = HDR;
        b.centre = plen + next.offset + (HDR - 1) / 2.0;
        b.sum = next.corr;
        b.energy = next.energy;
        b.pilot = false;
        blocks.push_back(b);
    }

    // ---- intermediate estimate: L&R over the pilot blocks + header ----
    // computed on symbols counter-rotated by the predicted residual, so its
    // +-1/17 range is centred on the prediction; lag m of block b is
    // normalized by its N_b - m products (unbiased R(m), Luise &
    // Reggiannini 1995) and weighted by N_b.
    double f_int = f_pred;
    {
        complexf acc{};
        auto lr_block = [&](const complexf* r, const complexf* ref, bool conj_ref, int n,
                            double t_start) {
            complexf z[HDR];
            const double fr = residual_at((double)hdr_pos + t_start + 0.5 * n);
            complexf rot(1.0f, 0.0f);
            const complexf step((float)std::cos(-TWO_PI * fr), (float)std::sin(-TWO_PI * fr));
            for (int i = 0; i < n; i++) {
                z[i] = r[i] * (conj_ref ? std::conj(ref[i]) : ref[i]) * rot;
                rot *= step;
            }
            for (int m = 1; m <= LR_MAX_LAG; m++) {
                complexf rr{};
                for (int i = 0; i + m < n; i++)
                    rr += z[i + m] * std::conj(z[i]);
                acc += rr * ((float)n / (float)(n - m));
            }
            return fr;
        };
        double fr0 = 0.0;
        for (int k = 0; k < n_pb; k++) {
            const int off = (k + 1) * pl::kSlotsPerPilotGroup * pl::kSlotLen + k * PBL;
            fr0 = lr_block(pl + off, pilot_ref.data() + off, false, PBL, HDR + off);
        }
        fr0 = lr_block(fr, hdr_ref[l.pls_code].data(), true, HDR, 0.0);
        // all blocks of a non-transition frame share the same prediction
        f_int = fr0 + std::arg(acc) / (M_PI * (LR_MAX_LAG + 1));
    }

    // ---- block phases: MAP choice of the 2 pi ambiguity ----
    // Phase noise per block ~ 1 / (2 N_b rho) (rho: pilot Es/N0 of the
    // previous frame). The fine slope is ambiguous by ~1/1476 cycle/symbol;
    // the candidate unwraps (predictor + m / 1476, |m| <= 2) are scored by
    // their LS phase residual plus the Gaussian priors of the loop estimate
    // (variance var_best) and of the intermediate estimate (variance from
    // the per-block Cramer-Rao bound, x1.5 for L&R).
    const double rho = snr_ok ? std::max(0.3, std::pow(10.0, snr_db / 10.0)) : 1.0;
    double var_int = 0.0;
    {
        double info = 0.0;
        for (const block_t& b : blocks)
            info += TWO_PI * TWO_PI * rho * (double)b.len * b.len * b.len / 6.0;
        var_int = info > 0.0 ? 1.5 / info : 1.0;
    }
    // predicted trajectory P (per-step predictors, rotator history)
    pred_v.clear();
    {
        double pv = 0.0;
        pred_v.push_back(0.0);
        for (size_t i = 1; i < blocks.size(); i++) {
            const double dt = blocks[i].centre - blocks[i - 1].centre;
            const double mid = hdr_pos + 0.5 * (blocks[i].centre + blocks[i - 1].centre);
            pv += TWO_PI * residual_at(mid) * dt;
            pred_v.push_back(pv);
        }
    }
    // mean predicted residual over the frame (time-weighted)
    double pred_slope = f_pred;
    if (blocks.size() >= 2) {
        const double span_t = blocks.back().centre - blocks.front().centre;
        if (span_t > 0)
            pred_slope = (pred_v.back() - pred_v.front()) / (TWO_PI * span_t);
    }
    // Increment model (phase random walk + white block noise): the phase
    // step between blocks i-1 and i is 2 pi f dt_i + w_i with
    //   var(w_i) = s_{i-1}^2 + s_i^2 + Q dt_i,  s_i^2 = 1 / (2 N_i rho),
    // Q the phase-noise rate allowed for (2 pi 2e-6 rad^2/symbol, between
    // the PN-A and PN-B test profiles). The ML slope is the
    // weighted mean of the steps; the cost of a candidate unwrap is its
    // step residual plus the two Gaussian priors.
    auto blk_var = [&](size_t i) { return 1.0 / (2.0 * blocks[i].len * rho); };
    double sum_tt = 0.0;
    for (size_t i = 1; i < blocks.size(); i++) {
        const double dt = blocks[i].centre - blocks[i - 1].centre;
        sum_tt += dt * dt / (blk_var(i - 1) + blk_var(i) + PN_RATE * dt);
    }
    const double var_prior = var_best;
    double f_fine = f_pred;
    double best_cost = INFINITY;
    int best_m = 0;
    // predictor offsets on a half-alias grid: every true slope within
    // +-2.25/1476 of the prediction is within a quarter alias of one of
    // them, so its unwrap errs by less than pi/2 per block step
    for (int m = -ALIAS_SEARCH; m <= ALIAS_SEARCH && sum_tt > 0; m++) {
        const double alias = 0.5 * m / PILOT_SPACING;
        // deviation steps from the predicted trajectory
        double num = 0.0;
        dev_v.resize(blocks.size());
        double prev = std::arg(blocks[0].sum);
        dev_v[0] = 0.0;
        for (size_t i = 1; i < blocks.size(); i++) {
            const double dt = blocks[i].centre - blocks[i - 1].centre;
            const double expect = prev + (pred_v[i] - pred_v[i - 1]) + TWO_PI * alias * dt;
            const double v = expect + wrap_pi(std::arg(blocks[i].sum) - expect);
            const double step = (v - prev) - (pred_v[i] - pred_v[i - 1]); // deviation step
            dev_v[i] = step;
            num += step * dt / (blk_var(i - 1) + blk_var(i) + PN_RATE * dt);
            prev = v;
        }
        const double slope = num / sum_tt; // rad/symbol
        double res = 0.0;
        for (size_t i = 1; i < blocks.size(); i++) {
            const double dt = blocks[i].centre - blocks[i - 1].centre;
            const double d = dev_v[i] - slope * dt;
            res += d * d / (blk_var(i - 1) + blk_var(i) + PN_RATE * dt);
        }
        const double e = slope / TWO_PI;
        const double fm = pred_slope + e;
        const double cost = res + e * e / var_prior + (fm - f_int) * (fm - f_int) / var_int;
        if (diag && getenv("CUSAT_PLRX_TRACE"))
            fprintf(stderr, "  m=%d res=%.2f e=%.3e fm=%.6e cost=%.2f\n", m, res, e, fm, cost);
        if (cost < best_cost) {
            best_cost = cost;
            f_fine = amb ? f_pred + e : fm;
            best_m = m;
        }
    }
    const double r_fine = sum_tt > 0 ? 1.0 / (sum_tt * TWO_PI * TWO_PI) : 1e-6;
    // final unwrap with the chosen alias (phase trajectory for the data)
    {
        const double alias = 0.5 * best_m / PILOT_SPACING;
        phase_t.clear();
        phase_v.clear();
        double prev = std::arg(blocks[0].sum);
        phase_t.push_back((float)blocks[0].centre);
        phase_v.push_back(prev);
        for (size_t i = 1; i < blocks.size(); i++) {
            const double dt = blocks[i].centre - blocks[i - 1].centre;
            const double expect = prev + (pred_v[i] - pred_v[i - 1]) + TWO_PI * alias * dt;
            const double v = expect + wrap_pi(std::arg(blocks[i].sum) - expect);
            phase_t.push_back((float)blocks[i].centre);
            phase_v.push_back(v);
            prev = v;
        }
    }
    // ---- phase trajectory: Rauch-Tung-Striebel smoothing ----
    // Model around a line fitted to the unwrapped block phases themselves:
    // residual phase is a random walk (rate pn_q, estimated from the data)
    // observed with the block noise blk_var. In AWGN pn_q -> 0 and the
    // trajectory tends to the line fitted over the whole frame; with phase
    // noise it follows the pilots (Meyr, Moeneclaey & Fechtel 1998, ch. 5;
    // Rauch, Tung & Striebel 1965). The line and pn_q depend only on the
    // block phases, not on the rotator model (command times depend on the
    // batch partition, R1.10).
    if (blocks.size() >= 3) {
        const size_t nb = blocks.size();
        // GLS slope of the phase increments (same weights as sum_tt); for a
        // frame without a command change this equals 2 pi f_fine
        double s_num = 0.0;
        for (size_t i = 1; i < nb; i++) {
            const double dt = blocks[i].centre - blocks[i - 1].centre;
            s_num += (phase_v[i] - phase_v[i - 1]) * dt / (blk_var(i - 1) + blk_var(i) + PN_RATE * dt);
        }
        const double w_line = sum_tt > 0.0 ? s_num / sum_tt : TWO_PI * f_fine; // rad/symbol
        double sw = 0, sv = 0;
        for (size_t i = 0; i < nb; i++) {
            const double w = 1.0 / blk_var(i);
            sw += w;
            sv += w * (phase_v[i] - w_line * blocks[i].centre);
        }
        const double a0 = sv / sw; // offset of the line with slope w_line
        res_v.resize(nb);
        for (size_t i = 0; i < nb; i++)
            res_v[i] = phase_v[i] - (a0 + w_line * blocks[i].centre);
        // random-walk rate from the step innovations
        double num = 0.0, den = 0.0;
        for (size_t i = 1; i < nb; i++) {
            const double d = res_v[i] - res_v[i - 1];
            num += d * d - blk_var(i) - blk_var(i - 1);
            den += blocks[i].centre - blocks[i - 1].centre;
        }
        if (den > 0.0) {
            const double q = std::max(0.0, num / den);
            pn_q = PN_Q_ALPHA * std::min(q, 10.0 * PN_RATE) + (1.0 - PN_Q_ALPHA) * pn_q;
        }
        const double qv = std::max(pn_q, 1e-9);
        // forward filter
        kf_x.resize(nb);
        kf_p.resize(nb);
        kf_xp.resize(nb);
        kf_pp.resize(nb);
        double x = res_v[0], pv = blk_var(0);
        kf_x[0] = x;
        kf_p[0] = pv;
        kf_xp[0] = x;
        kf_pp[0] = pv;
        for (size_t i = 1; i < nb; i++) {
            const double dt = blocks[i].centre - blocks[i - 1].centre;
            const double xp = x, pp = pv + qv * dt;
            const double k = pp / (pp + blk_var(i));
            x = xp + k * (res_v[i] - xp);
            pv = (1.0 - k) * pp;
            kf_xp[i] = xp;
            kf_pp[i] = pp;
            kf_x[i] = x;
            kf_p[i] = pv;
        }
        // backward pass
        for (size_t i = nb - 1; i-- > 0;) {
            const double g = kf_p[i] / kf_pp[i + 1];
            kf_x[i] = kf_x[i] + g * (kf_x[i + 1] - kf_xp[i + 1]);
        }
        for (size_t i = 0; i < nb; i++)
            phase_v[i] = a0 + w_line * blocks[i].centre + kf_x[i];
    }

    if (diag && getenv("CUSAT_PLRX_TRACE")) {
        fprintf(stderr, "frame %llu amb=%d f_pred=%.3e f_int=%.3e f_fine=%.3e m=%d var_best=%.2e var_int=%.2e cmd_time=%llu f_prev=%.6f f_cmd=%.6f f_best=%.6f\n",
                (unsigned long long)hdr_pos, (int)amb, f_pred, f_int, f_fine, best_m, var_best, var_int,
                (unsigned long long)cmd_time, f_prev, f_cmd, f_best);
        for (size_t i = 0; i < blocks.size(); i++)
            fprintf(stderr, "   blk %zu t=%.1f raw=%.3f unw=%.3f\n", i, phase_t[i], std::arg(blocks[i].sum), phase_v[i]);
    }

    // ---- pilot SNR, coherence, amplitude (frequency-corrected) ----
    double sig = 0.0, noise = 0.0, csum = 0.0;
    int n_noise = 0;
    for (int k = 0; k < n_pb; k++) {
        const block_t& b = blocks[1 + k];
        const int off = b.start - HDR;
        complexf z[PBL];
        complexf mean{};
        float e = 0.0f;
        const double c0 = -(PBL - 1) / 2.0;
        complexf rot((float)std::cos(-TWO_PI * f_fine * c0), (float)std::sin(-TWO_PI * f_fine * c0));
        const complexf step((float)std::cos(-TWO_PI * f_fine), (float)std::sin(-TWO_PI * f_fine));
        for (int i = 0; i < PBL; i++) {
            z[i] = pl[off + i] * pilot_ref[off + i] * rot;
            rot *= step;
            mean += z[i];
            e += norm2(z[i]);
        }
        // c_b = |sum r| / sqrt(36 sum |r|^2)
        if (e > 0.0f)
            csum += std::abs(mean) / std::sqrt((float)PBL * e);
        mean /= (float)PBL;
        float ss = 0.0f;
        for (int i = 0; i < PBL; i++)
            ss += norm2(z[i] - mean);
        noise += ss;
        n_noise += PBL - 1;
        sig += norm2(mean);
    }
    const double n_var = n_noise > 0 ? noise / n_noise : 0.0;
    // E|mean|^2 = S + N / 36
    const double s_pow = sig / n_pb - n_var / PBL;
    const float frame_coh = (float)(csum / n_pb);
    bool frame_snr_ok = false;
    float frame_snr_db = 0.0f;
    if (s_pow > 0.0 && n_var > 0.0 && std::isfinite(s_pow) && std::isfinite(n_var)) {
        frame_snr_db = (float)(10.0 * std::log10(s_pow / n_var));
        frame_snr_ok = frame_coh > 0.2f; // phase-tracking anomaly otherwise
    } else if (s_pow > 0.0 && n_var == 0.0) {
        frame_snr_db = 99.0f; // noiseless input
        frame_snr_ok = frame_coh > 0.2f;
    }

    // ---- carrier loop (frames measured against a known rotator) ----
    if (!amb && frame_coh > 0.3f) {
        const double r_meas = r_fine;
        loop_update(f_rot + f_fine, r_meas, plen);
        // alias watchdog: the intermediate estimate is unbiased; a
        // persistent offset of the fine estimate from it by about one alias
        // means the loop settled on a wrong 2 pi ambiguity
        // (exponentially weighted mean and variance of f_int - f_fine over
        // ~40 frames; a correction needs a 5-sigma offset of at least half
        // an alias)
        {
            const double d = f_int - f_fine;
            const double a = ALIAS_ALPHA;
            if (alias_n == 0) {
                alias_acc = d;
                alias_var = 0.0;
            } else {
                const double delta = d - alias_acc;
                alias_acc += a * delta;
                alias_var = (1.0 - a) * (alias_var + a * delta * delta);
            }
            alias_n++;
            const double se = std::sqrt(alias_var * a / (2.0 - a));
            if (alias_n >= ALIAS_MIN_FRAMES && std::fabs(alias_acc) > 0.5 / PILOT_SPACING &&
                std::fabs(alias_acc) > 5.0 * se) {
                const double k = std::round(alias_acc * PILOT_SPACING);
                if (k != 0.0) {
                    f_best += k / PILOT_SPACING;
                    var_best = std::max(var_best, 1e-8);
                    if (diag)
                        fprintf(stderr, "pl_receiver: carrier alias corrected by %+.0f/1476\n", k);
                }
                alias_n = 0;
            }
        }
    }

    if (state != st::locked)
        return; // settling: carrier only

    coh_inst = frame_coh;
    if (!coh_init) {
        coh = coh_inst;
        coh_init = true;
    } else {
        coh = COHERENCE_ALPHA * coh_inst + (1.0f - COHERENCE_ALPHA) * coh;
    }
    if (frame_snr_ok) {
        snr_db = frame_snr_db;
        snr_ok = true;
    }
    if (s_pow > 0.0 && std::isfinite(s_pow) && coh_inst > 0.2f) {
        if (!gain_init) {
            gain_pow = (float)s_pow;
            gain_init = true;
        } else {
            gain_pow = GAIN_ALPHA * (float)s_pow + (1.0f - GAIN_ALPHA) * gain_pow;
        }
    }

    if (!cur_deliver || lost_cadence_frame) {
        n_rejected++;
        if (diag)
            fprintf(stderr, "pl_receiver: frame %llu not delivered (deliver %d, cadence %d, pls %u)\n",
                    (unsigned long long)hdr_pos, (int)cur_deliver, (int)!lost_cadence_frame,
                    cur_hdr.dec.pls);
        return;
    }

    // ---- data: descramble, remove pilots, counter-rotate, scale ----
    const int n_data = l.data_symbols();
    const float inv_amp = gain_pow > 1e-20f ? 1.0f / std::sqrt(gain_pow) : 1.0f;
    // straight into the caller's buffer when it fits, else a pool slot
    const bool direct = ob_count == 0 && call_out && call_written + n_data <= call_max;
    ready_frame* slot = nullptr;
    complexf* dst;
    if (direct) {
        dst = call_out + call_written;
    } else {
        if (ob_count == OUTBOX_SLOTS) {
            // caller not draining: drop the oldest pending frame
            ob_head = (ob_head + 1) % OUTBOX_SLOTS;
            ob_count--;
            n_dropped++;
            if (diag || n_dropped == 1)
                fprintf(stderr, "pl_receiver: output space exhausted, pending frame dropped "
                                "(%llu so far)\n", (unsigned long long)n_dropped);
        }
        slot = &outbox[(ob_head + ob_count) % OUTBOX_SLOTS];
        dst = slot->data.data();
    }
    const int n_slots = l.slots;
    size_t seg = 0; // phase segment: between blocks seg and seg + 1
    int written = 0;
    for (int s = 0; s < n_slots; s++) {
        const int off = s * pl::kSlotLen + std::min(s / pl::kSlotsPerPilotGroup, n_pb) * PBL;
        const int t0 = HDR + off; // offset of the slot's first symbol in the frame
        // advance the segment so that phase_t[seg] <= t0 < phase_t[seg + 1]
        while (seg + 2 < phase_t.size() && phase_t[seg + 1] <= t0)
            seg++;
        double ta = phase_t[seg], va = phase_v[seg];
        double slope;
        if (seg + 1 < phase_t.size())
            slope = (phase_v[seg + 1] - va) / (phase_t[seg + 1] - ta);
        else
            slope = TWO_PI * f_fine;
        // a slot never contains a block centre except possibly the last
        // segment boundary; recompute the rotation per slot
        double ph = va + slope * (t0 - ta);
        complexf rot((float)std::cos(-ph), (float)std::sin(-ph));
        rot *= inv_amp;
        const complexf step((float)std::cos(-slope), (float)std::sin(-slope));
        const complexf* r = pl + off;
        const complexf* d = descr.data() + off;
        rotate_slot(r, d, rot, step, dst + written);
        written += pl::kSlotLen;
    }
    (void)payload;

    pl_frame meta;
    meta.n_data = n_data;
    meta.layout = l;
    meta.sof_symbol = hdr_pos;
    meta.snr_db = frame_snr_db;
    meta.snr_valid = frame_snr_ok;
    meta.coherence = coh_inst;
    meta.residual_cfo = f_fine;
    const float mg = std::isfinite(cur_hdr.dec.margin) ? cur_hdr.dec.margin * PLS_NORM : 2.0f;
    meta.pls_confidence = std::max(0.0f, std::min(1.0f, mg));
    if (direct) {
        meta.data = dst;
        call_frames->push_back(meta);
        call_written += n_data;
    } else {
        slot->meta = meta;
        ob_count++;
    }
    n_frames++;
}

/* ===================================================================== */

int pl_receiver::impl::flush_outbox(complexf* out, int max_out, int written,
                                    std::vector<pl_frame>& frames)
{
    while (ob_count > 0 && out) {
        ready_frame& rf = outbox[ob_head];
        if (written + rf.meta.n_data > max_out)
            break;
        memcpy(out + written, rf.data.data(), rf.meta.n_data * sizeof(complexf));
        rf.meta.data = out + written;
        frames.push_back(rf.meta);
        written += rf.meta.n_data;
        ob_head = (ob_head + 1) % OUTBOX_SLOTS;
        ob_count--;
    }
    return written;
}

/* ===================================================================== */
/*  Public interface                                                     */
/* ===================================================================== */

pl_receiver::pl_receiver(const pl_rx_config& config) : d_impl(std::make_unique<impl>(config)) {}

pl_receiver::~pl_receiver() = default;

namespace {

/* true when none of the n floats is Inf or NaN (|x| < inf fails for both) */
bool all_finite(const float* __restrict__ x, int n)
{
    int i = 0;
#if defined(__ARM_NEON)
    const float32x4_t vinf = vdupq_n_f32(INFINITY);
    uint32x4_t ok = vdupq_n_u32(0xffffffffu);
    for (; i + 16 <= n; i += 16) {
        ok = vandq_u32(ok, vcltq_f32(vabsq_f32(vld1q_f32(x + i)), vinf));
        ok = vandq_u32(ok, vcltq_f32(vabsq_f32(vld1q_f32(x + i + 4)), vinf));
        ok = vandq_u32(ok, vcltq_f32(vabsq_f32(vld1q_f32(x + i + 8)), vinf));
        ok = vandq_u32(ok, vcltq_f32(vabsq_f32(vld1q_f32(x + i + 12)), vinf));
    }
    if (vminvq_u32(ok) == 0)
        return false;
#endif
    for (; i < n; i++)
        if (!std::isfinite(x[i]))
            return false;
    return true;
}

} // namespace

int pl_receiver::process(const complexf* syms, int n_syms, const pl_header_candidates* candidates,
                         complexf* out, int max_out, std::vector<pl_frame>& frames,
                         std::vector<pl_sync_event>& events)
{
    impl& m = *d_impl;
    int written = m.flush_outbox(out, max_out, 0, frames);
    if (m.pending_lost) {
        events.push_back({ pl_sync_event_type::lost, m.pending_lost_at });
        m.pending_lost = false;
    }
    if (n_syms <= 0 || !syms)
        return written;
    m.call_out = out;
    m.call_max = max_out;
    m.call_written = written;
    m.call_frames = &frames;

    // NaN/Inf samples are treated as zero (R1.14)
    const complexf* in = syms;
    {
        const bool bad = !all_finite(reinterpret_cast<const float*>(syms), 2 * n_syms);
        if (bad) {
            if ((int)m.sanitized.size() < n_syms)
                m.sanitized.resize(n_syms);
            for (int i = 0; i < n_syms; i++) {
                const float a = syms[i].real(), b = syms[i].imag();
                m.sanitized[i] = (std::isfinite(a) && std::isfinite(b)) ? syms[i] : complexf(0, 0);
            }
            in = m.sanitized.data();
        }
    }

    m.batch = in;
    m.batch_n = n_syms;
    m.batch_abs = m.total_in;
    m.total_in += (uint64_t)n_syms;
    if (m.carry_n == 0)
        m.carry_abs = m.batch_abs;

    for (int guard = 0; guard < 64; guard++) {
        if (m.state == impl::st::search) {
            m.acquisition(guard == 0 ? candidates : nullptr, events);
            if (m.state == impl::st::search)
                break;
            // a candidate list covers this batch only once; the CPU
            // detector resumes after the lock point if needed
            m.search_from = m.avail_end();
        }
        const bool was_locked = m.state != impl::st::search;
        m.locked_loop(events);
        if (was_locked && m.state == impl::st::search) {
            // lost inside this batch: search the remainder with the CPU
            // detector only when no GPU list was given
            if (candidates)
                break;
            continue;
        }
        break;
    }

    // keep what later decisions need
    uint64_t keep = m.avail_end() > (uint64_t)(HDR + 2) ? m.avail_end() - (HDR + 2) : 0;
    if (m.state != impl::st::search) {
        keep = std::min(keep, m.hdr_pos);
    } else {
        // the frame after the last hit of a track is used to refine the
        // carrier at lock
        for (const track_t& t : m.tracks)
            if (t.len > 0)
                keep = std::min(keep, t.pos);
    }
    m.retain(keep);
    m.batch = nullptr;
    m.batch_n = 0;
    m.batch_abs = m.total_in;
    written = m.call_written;
    m.call_out = nullptr;
    m.call_frames = nullptr;
    return m.flush_outbox(out, max_out, written, frames);
}

bool pl_receiver::locked() const { return d_impl->state == impl::st::locked; }

bool pl_receiver::wants_header_search() const { return d_impl->state == impl::st::search; }

uint32_t pl_receiver::expected_plframe_len() const
{
    const impl& m = *d_impl;
    if (m.state != impl::st::search)
        return (uint32_t)std::max(0, m.cur.plframe_symbols());
    for (const track_t& t : m.tracks)
        if (t.len > 0 && t.hits >= 2)
            return (uint32_t)t.len;
    return m.cfg.acq_plframe_hint;
}

int pl_receiver::pls_disagreement_run() const { return d_impl->disagree_run; }

double pl_receiver::carrier_frequency() const { return d_impl->f_cmd; }

bool pl_receiver::take_carrier_update(double& cycles_per_symbol)
{
    if (!d_impl->cmd_pending)
        return false;
    d_impl->cmd_pending = false;
    cycles_per_symbol = d_impl->f_cmd;
    return true;
}

float pl_receiver::snr_db() const { return d_impl->snr_db; }
bool pl_receiver::snr_valid() const { return d_impl->snr_ok; }
float pl_receiver::coherence() const { return d_impl->coh; }
float pl_receiver::coherence_instant() const { return d_impl->coh_inst; }
uint64_t pl_receiver::frame_count() const { return d_impl->n_frames; }
uint64_t pl_receiver::dummy_count() const { return d_impl->n_dummy; }
uint64_t pl_receiver::rejected_count() const { return d_impl->n_rejected; }

void pl_receiver::reset(pl_reset_scope scope)
{
    impl& m = *d_impl;
    const bool full = scope == pl_reset_scope::full;
    const double f_before = m.f_cmd;
    if (scope == pl_reset_scope::positional && m.state == impl::st::locked) {
        m.pending_lost = true;
        m.pending_lost_at = m.total_in;
    }
    m.reset(scope);
    if (full && f_before != 0.0)
        m.cmd_pending = true; // carrier_frequency() -> 0 must reach the rotator
}

void pl_receiver::set_valid_plframe_lens(const std::set<uint32_t>& lens)
{
    d_impl->cfg.valid_plframe_lens = lens;
    d_impl->configure_valid_lens();
}

void pl_receiver::set_acq_plframe_hint(uint32_t plframe_len) { d_impl->cfg.acq_plframe_hint = plframe_len; }

void pl_receiver::set_correction_latency(uint32_t symbols) { d_impl->cfg.correction_latency_symbols = symbols; }

void pl_receiver::set_diagnostics(bool on) { d_impl->diag = on; }

} // namespace cusat_dsp
