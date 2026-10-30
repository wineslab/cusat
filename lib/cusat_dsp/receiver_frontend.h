/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone DVB-S2/S2X receiver frontend.
 *
 * Consumes oversampled IQ samples and emits descrambled, phase-corrected
 * data symbols (XFECFRAMEs) ready for soft demapping.  The chain is:
 * GPU AGC + inline frequency rotation, GPU feedforward symbol timing
 * recovery (polyphase energy search), GPU bulk PLHEADER search during
 * acquisition, then the PL receiver (plrx/pl_receiver.h).
 *
 * Frequency corrections are applied by an internal rotator phasor that
 * takes effect at the next process() call.
 *
 * PL synchronization itself is done by pl_receiver; this class runs the GPU
 * orchestration, batching/pipelining, lock validation and output gates.
 */

#ifndef INCLUDED_CUSAT_DSP_RECEIVER_FRONTEND_H
#define INCLUDED_CUSAT_DSP_RECEIVER_FRONTEND_H

#include "interpolator.h"
#include "framing/plframe.h"
#include "plrx/pl_receiver.h"
#include <cusat_dsp/compat.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <set>
#include <vector>

namespace cusat_dsp { namespace cuda {
    class plheader_search_cuda;
    class timing_recovery_cuda;
    class agc_rot_cuda;
} }

namespace cusat_dsp {

/**
 * @brief Result from a single process() call indicating a complete XFECFRAME.
 */
struct frontend_frame_t {
    const complexf* payload; /**< Descrambled, phase-corrected data symbols */
    pl_layout pls;           /**< PLFRAME layout of this frame */
    int payload_len;         /**< Number of data symbols (== pls.data_symbols()) */
    float pilot_snr_db = 0.0f;
};

/** Per-call timing breakdown of receiver_frontend::process(). */
struct frontend_perf_t {
    uint32_t agc_us;     /**< memcpy + AGC + frequency rotation (GPU or CPU) */
    uint32_t timing_us;  /**< Symbol timing recovery incl GPU sync */
    uint32_t sync_us;    /**< PL receiver: header search + frame processing */
    uint32_t total_us;   /**< Total process() wall time */
    int n_in;            /**< Input samples this call */
    int n_syms;          /**< Symbols produced by timing recovery */
    int n_frames;        /**< Frames completed this call */
    bool fused;          /**< Whether the fused GPU header search was used */
    /* timing recovery sub-stage breakdown (only when GPU timing recovery active) */
    uint32_t tr_memcpy_us;   /**< memcpy raw IQ to unified memory */
    uint32_t tr_launch_us;   /**< Kernel launch (AGC+OM+interp+search) */
    uint32_t tr_sync_us;     /**< cudaStreamSynchronize wait */
    uint32_t tr_readback_us; /**< Read state + copy symbols */
    /* Host-side sections inside tr_launch_us (timing_recovery process_launch) */
    uint32_t tr_lnch_est_us;    /**< estimator section (energy launch / dead-reckon) */
    uint32_t tr_lnch_interp_us; /**< batch_interp launch + event record */
    /* Per-kernel GPU time (CUDA events, only when GPU timing recovery active) */
    uint32_t gpu_timing_us;      /**< timing_estimate_kernel GPU time */
    uint32_t gpu_interp_us;  /**< batch_interp_kernel GPU time */
    /* Fine-grained GPU pipeline breakdown (CUDA events) */
    uint32_t gpu_h2d_us;     /**< H2D DMA time */
    uint32_t gpu_agc_rot_us; /**< AGC + rotation kernels (total) */
    uint32_t gpu_agc_only_us;/**< AGC RMS kernel only */
    uint32_t gpu_rot_only_us;/**< Rotation+gain kernel only */
};

/* Construction parameters. Grouped by stage. */
struct receiver_frontend_config {
    /* Input and timing recovery */
    float sps = 4.0f;            /**< hardware samples per symbol (fractional ok) */
    float rolloff = 0.2f;
    int rrc_delay = 5;           /**< matched-filter one-sided span, symbols */
    int n_phases = 128;          /**< polyphase subfilters */
    bool prefiltered = false;    /**< RU applied the matched filter (rung 2) */
    bool symbols_in = false;     /**< RU delivers on-time symbols (rung 3) */
    int max_gpu_batch = 0;       /**< max input samples per GPU batch (0 = default) */
    /* AGC */
    float agc_rate = 1e-5f;
    float agc_ref = 1.0f;
    float agc_initial_gain = 1.0f;
    float agc_max_gain = 65536.0f;
    /* PL layer */
    pl_rx_config pl;
    int debug_level = 0;
};

class receiver_frontend
{
public:
    explicit receiver_frontend(const receiver_frontend_config& config);

    ~receiver_frontend();

    /**
     * @brief Process a chunk of input samples.
     *
     * Produces descrambled, phase-corrected data symbols (one or more
     * XFECFRAMEs). The caller must provide output buffer space.
     *
     * @param in Input IQ samples (oversampled at sps).
     * @param n_in Number of input samples available.
     * @param out Output data symbols buffer.
     * @param max_out Maximum number of output symbols.
     * @param n_consumed [out] Number of input samples consumed.
     * @param frames [out] Vector of completed frames with metadata.
     * @return Total number of output symbols produced.
     */
    int process(const complexf* in,
                int n_in,
                complexf* out,
                int max_out,
                int& n_consumed,
                std::vector<frontend_frame_t>& frames);

    // Getters for monitoring
    bool locked() const { return d_frame_locked; }
    /** Force unlock — triggers full re-acquisition (same as natural lock loss). */
    void force_unlock();
    /** Reset every acquisition estimator, including while already unlocked. */
    void force_reacquire();
    double freq_offset() const { return d_total_cfo; }
    uint64_t frame_count() const;
    uint64_t dummy_count() const;
    float pilot_snr_db() const;
    bool has_pilot_snr() const;
    float pilot_coherence() const;
    double get_mu() const { return d_mu; }
    float agc_gain() const;

    // Input history requirement (samples the caller must keep)
    unsigned history() const { return d_ss_history; }

    /** Last process() timing breakdown. */
    const frontend_perf_t& perf() const { return d_perf; }

    /** Sub-stage within process() for stall diagnosis.
     * 30=mutex, 31=launch, 32=sync, 33=pl_sync, 34=perf, 39=done */
    std::atomic<int> process_substage{0};

    /**
     * @brief Set the whitelist of valid PLFRAME lengths for ACM.
     * Forwarded to the PL receiver.
     */
    void set_valid_plframe_lens(const std::set<uint32_t>& lens);

    /**
     * @brief Hint the expected PLFRAME length for PRE-LOCK acquisition sizing.
     *
     * Sizes the pre-lock acquisition batch below one PLFRAME (required for
     * small frames, e.g. short 32APSK = 3402 sym, where the 33282 default
     * would span ~10 frames). Independent of set_valid_plframe_lens().
     * Unset (0) => 33282 default, leaving the normal-frame acquisition path
     * unchanged.
     */
    void set_acq_plframe_hint(uint32_t plframe_len);

    /** Set timing recovery estimation period (1 = every call, default 4). */
    void set_timing_estimation_period(int period);
    /** Set number of polyphase arms for energy search (8..64, default 32). */
    void set_energy_k(int k);
    /** Set max symbols for energy search (0 = no cap, default 0). */
    void set_energy_search_cap(int cap);
    /** Set batch cap in PLFRAMEs for timing recovery (default 2). */
    void set_batch_plframes(int n) { d_batch_plframes = std::max(1, n); update_correction_latency(); }
    /** Hint that process() input buffers are page-locked host memory. */
    void set_input_pinned(bool pinned) { d_input_pinned = pinned; }
    /** Min locked-path batch in samples (0 = off). At high frame rates the
     *  per-batch GPU wait (~300 us under LDPC load) exceeds the real-time
     *  budget of small-PLFRAME batches; a floor of ~rate x 500 us keeps the
     *  budget above the wait. Locked path only — acquisition keeps its
     *  sub-PLFRAME cap. */
    void set_min_batch_samples(int n) { d_min_batch_samples = std::max(0, n); update_correction_latency(); }
    /**
     * @brief Notify the frontend that the input stream is discontinuous.
     *
     * Call after the caller skips samples (e.g. acquisition-mode ring
     * trimming). Invalidates positional state that would otherwise blend
     * across the gap — search overlap buffer, timing NCO phase/base, PL
     * receiver buffers, any partially collected frame — while keeping
     * slowly-varying physical estimates (AGC gain, frequency offset,
     * timing rate correction). Without this, acquisition at rates where
     * the search can't keep up with real time feeds the estimators
     * spliced disjoint chunks and never converges.
     */
    void notify_discontinuity();

    /**
     * @brief Flush deferred symbols from GPU-CPU pipeline.
     *
     * When pipelining is active, the last batch's symbols are deferred until
     * the next process() call. Call flush() after the last process() to
     * run the PL receiver on any remaining symbols.
     */
    int flush(complexf* out, int max_out,
              std::vector<frontend_frame_t>& frames);

private:
    void reset_acquisition_state();
    /* Worst-case delay between a carrier command of the PL receiver and its
     * first effect on the symbols it receives. */
    uint32_t correction_latency_symbols() const;
    void update_correction_latency();

    /* Run the PL receiver on one symbol batch,
     * apply carrier updates to the rotator, then the retained lock
     * validation / output gates below. Replaces the removed frame loop. */
    int run_pl_sync(const complexf* syms, int n_syms,
                    complexf* out, int max_out,
                    const pl_header_candidates* candidates,
                    std::vector<frontend_frame_t>& frames);

    /* Retained cuSAT logic, operating on PL receiver outputs. */
    bool gate_completed_frame(const pl_frame& f, frontend_frame_t& frame,
                              std::vector<frontend_frame_t>& frames);
    void handle_sync_events(const std::vector<pl_sync_event>& events);
    void schedule_validation_reset();

    /* ---- Symbol sync members ---- */
    /** Rung-3 symbols-in mode: input samples are RU-recovered on-time symbols
     *  (1/symbol); AGC/rotation/timing collapse to a CPU scalar gain plus the
     *  inline rotator, and no CUDA timing/AGC objects are created. */
    bool d_symbols_in = false;
    double d_sym_pow = 0.0;  /**< EWMA of mean |x|^2 for symbols-in unit-RMS gain */
    float d_sps;   /**< Hardware samples per symbol (fractional stride) */
    unsigned d_ss_history;
    double d_mu;
    polyphase_filter_bank d_rrc_bank;

    /* ---- Inline frequency rotator ---- */
    complexf d_rot_phase;           /**< Current rotator phase (complex unit vector) */
    float d_phase_inc;                /**< Rotator phase increment per sample */
    double d_total_cfo;             /**< Carrier offset removed by the rotator, cycles/symbol */

    int d_debug_level;
    bool d_pipeline_active = false;  /**< True when GPU-CPU pipelining is in use */

    /* App-visible lock state (PL receiver lock + soft-unlock override). */
    bool d_frame_locked;

    /* ---- PL receiver ---- */
    pl_receiver* d_pl = nullptr;
    std::vector<pl_frame> d_pl_frames;        /**< scratch, per batch */
    std::vector<pl_sync_event> d_pl_events;   /**< scratch, per batch */
    pl_header_candidates d_candidates;        /**< scratch, per batch */
    uint32_t d_acq_plframe_hint = 0;          /**< Pre-lock batch sizing hint (0=default) */

    /* Symbol buffer (filled by GPU timing recovery) */
    std::vector<complexf> d_sym_buf;

    /* GPU-CPU pipeline: deferred symbols from previous batch */
    std::vector<complexf> d_prev_sym_buf;
    int d_prev_n_syms = 0;
    float d_saved_launch_phase_inc = 0.0f;

    int d_n_consumed = 0;  /**< Input samples consumed by timing recovery */

    std::mutex* d_cuda_mtx = nullptr; /**< Shared mutex to serialize CUDA API calls across threads */
public:
    void set_cuda_mutex(std::mutex* mtx) { d_cuda_mtx = mtx; }
    /** Enable carrier diagnostic logging in the PL receiver. */
    void set_diag_carrier(bool on) { d_pl->set_diagnostics(on); }
    /** Enable RX lock/coherence diagnostic logging to /tmp/rxdiag.log. */
    void set_diag_rx(bool on);
private:

    int d_max_gpu_batch = 131072;  /**< Max input samples per GPU batch */
    int d_batch_plframes = 2;      /**< Batch cap in PLFRAMEs for timing recovery */
    int d_min_batch_samples = 0;   /**< Locked-path min batch in samples (0=off) */
    bool d_input_pinned = false;   /**< process() input can be copied direct to GPU */

    /* ---- GPU feedforward timing recovery (sps>=4, or sps=2 via energy search) ---- */
    cuda::timing_recovery_cuda* d_timing_recovery_cuda = nullptr;

    bool d_timing_init = false;
    double d_timing_drift_accum = 0.0;
    double d_timing_nco_base = 0.0; /**< Fractional-sample NCO carry across batches */
    double d_timing_rate_corr = 0.0; /**< Learned TX/RX sample-clock ratio offset
                                          (rate loop). Deliberately NOT reset on
                                          unlock: it is a physical property of the
                                          link and keeping it speeds relock. */
    int d_timing_est_period_cfg = 4; /**< Configured estimation period, restored
                                          after low-coherence freezes */

    bool d_diag_rx = false;
    FILE* d_rx_diag_fp = nullptr;
    int d_rx_diag_frame = 0;
    float d_rx_diag_prev_coh = 1.0f;

    /* ---- Soft/hard unlock tiering ---- */
    int d_soft_unlock_cnt = 0;
    int d_soft_unlock_holdoff = 0;  /**< Blocks remaining before soft unlock counting resumes */
    std::chrono::steady_clock::time_point d_relock_time;
    static constexpr int SOFT_UNLOCK_MAX = 20;

    /* ---- Acquisition timeout: full reset if not locked after N calls ---- */
    int d_acq_calls_since_reset = 0;        /**< process() calls since last full reset */
    static constexpr int ACQ_TIMEOUT_CALLS = 20000; /**< ~3-5s at typical call rates */

    /* ---- Lock validation (coherence + PLS disagreement) ---- */
    bool d_needs_full_reset = false;        /**< Deferred reset after coherence unlock */
    uint64_t d_frames_since_lock = 0;       /**< Frames decoded since last lock acquisition */
    int d_low_coherence_cnt = 0;            /**< Low-coherence frames in validation window */
    int d_low_coherence_streak = 0;         /**< Consecutive low-coherence frames (continuous) */
    int d_good_coherence_run = 0;           /**< Consecutive good-coherence frames after window */
    bool d_coherence_confirmed = false;     /**< True once sustained good coherence observed */
    static constexpr int COHERENCE_VALIDATION_WINDOW = 10; /**< Check first N frames after lock */
    static constexpr int COHERENCE_FAIL_THRESH = 2;        /**< Low-coh frames to unlock (early) */
    static constexpr int PLS_DISAGREE_THRESH = 10;         /**< PLS disagreements to unlock */
    static constexpr float COHERENCE_THRESHOLD = 0.5f;     /**< Below this = false lock */
    static constexpr int COHERENCE_CONFIRM_THRESH = 10;    /**< Good-coh frames needed to confirm */
    static constexpr int COHERENCE_UNLOCK_THRESH = 100;    /**< Force unlock after N consecutive low-coh */
    static constexpr float SNR_GATE_MARGIN = 2.0f;         /**< dB below QEF threshold to suppress frame */

    /* ---- Frame length diversity (garbage detector) ---- */
    static constexpr int PLF_HISTORY_SIZE = 10;            /**< Window size for diversity check */
    static constexpr int PLF_DIVERSITY_MAX = 2;            /**< Max distinct frame lengths (real signal) */
    uint32_t d_plf_history[PLF_HISTORY_SIZE]{};            /**< Circular buffer of recent PLFRAME lengths */
    int d_plf_idx = 0;                                     /**< Write index into d_plf_history */
    int d_plf_count = 0;                                   /**< Frames stored (up to PLF_HISTORY_SIZE) */

    /* ---- GPU fused AGC + frequency rotation ---- */
    cuda::agc_rot_cuda* d_agc_rot_cuda = nullptr;

    /* ---- GPU pipeline profiling events ---- */
    void* d_evt_h2d_start = nullptr;  // cudaEvent_t (void* to avoid cuda header)
    void* d_evt_h2d_done = nullptr;
    void* d_evt_agc_done = nullptr;

    /* ---- GPU bulk header search state ---- */
    cuda::plheader_search_cuda* d_hdr_search = nullptr;
    static constexpr int SEARCH_OVERLAP = pl::kHeaderLen; // symbols saved across batches
    std::vector<complexf> d_gpu_overlap;     // last SEARCH_OVERLAP symbols
    bool d_gpu_overlap_valid = false;
    std::vector<complexf> d_gpu_search_buf;  // overlap + current batch (host path)
    float* d_search_um = nullptr;            // pinned search buffer (fused path)
    /* ---- Per-call performance counters ---- */
    frontend_perf_t d_perf{};
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_RECEIVER_FRONTEND_H */
