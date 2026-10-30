/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU polyphase energy search timing recovery with a fractional-stride NCO.
 *
 * Two-kernel pipeline:
 *   1. Polyphase energy search: evaluates K subfilter phases (arms) spread
 *      over one symbol period to find the optimal timing offset (tau).
 *   2. Batch polyphase interpolation: each output symbol k is interpolated
 *      at the fractional input position base + k*stride, where stride is
 *      the (possibly non-integer) samples-per-symbol ratio. The integer
 *      part of the position selects the basepoint sample; the fractional
 *      part selects the polyphase subfilter.
 *
 * The symbol-timing NCO carries the fractional sample remainder across
 * process() calls: n_consumed is always an integer number of input samples
 * and state_t::nco_base holds the leftover fraction, so the sampling
 * instants stay phase-continuous at batch boundaries for any stride.
 *
 * Input samples, output symbols, and state all use managed (unified) memory.
 * On Grace Hopper, this avoids unnecessary H2D/D2H copies — CPU and GPU
 * share the same physical memory via NVLink-C2C.
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_TIMING_RECOVERY_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_TIMING_RECOVERY_KERNEL_CUH

#include <cuda_runtime.h>
#include <cusat_dsp/compat.h>
#include "polyphase_bank_kernel.cuh"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace cusat_dsp {
namespace cuda {

class timing_recovery_cuda
{
public:
    struct state_t {
        double tau = 0.0;        // last timing phase estimate [0, 1) symbols (diagnostics)
        double nco_base = 0.0;   // NCO carry: sample offset of the next output symbol
                                 // relative to ss_history in the NEXT input buffer
        double drift_accum = 0.0; // accumulated clock drift in symbol periods
        double rate_corr = 0.0;  // relative TX/RX sample-clock offset (symbols of
                                 // drift per symbol, ~1e-6): the NCO runs at
                                 // effective stride = stride * (1 + rate_corr),
                                 // learned from accepted tracking deltas (rate loop)
        bool init = false;       // false until first estimate produced
    };

    /**
     * @param n_phases   Number of polyphase subfilters.
     * @param subfilt_len Length of each subfilter (taps).
     * @param subfilters  Flat array of subfilter taps [n_phases * subfilt_len],
     *                    already reversed (polyphase_filter_bank::flat_taps()).
     * @param sps         Samples per symbol (fractional stride, >= 2.0).
     * @param bank_delay  Exact branch-0 group delay of the subfilter bank in
     *                    input samples (polyphase_filter_bank::group_delay_samples();
     *                    may be fractional).
     * @param max_input   Maximum input sample count. Default 131072.
     */
    timing_recovery_cuda(int n_phases, int subfilt_len,
                   const float* subfilters,
                   double sps,
                   double bank_delay,
                   int max_input = 131072);
    ~timing_recovery_cuda();

    /**
     * @brief Process input samples -> output symbols via energy search + batch interpolation.
     *
     * @param in           Input IQ samples (oversampled at sps).
     * @param n_in         Total number of input samples (including history).
     * @param sym_out      Output symbol buffer (caller-allocated).
     * @param max_syms     Maximum symbols to produce.
     * @param state        [in/out] Timing recovery state.
     * @param n_consumed   [out] Number of input samples consumed (excluding history).
     * @param ss_history   Symbol sync history requirement.
     * @return Number of symbols produced.
     */
    int process(const complexf* in, int n_in,
                complexf* sym_out, int max_syms,
                state_t& state, int& n_consumed,
                unsigned ss_history,
                bool skip_input_copy = false);

    /**
     * @brief Launch energy search + interpolation kernels without synchronizing.
     *
     * Returns the max possible output symbols (for SOF search grid sizing).
     * The caller can launch additional kernels on stream() before calling
     * process_complete() which does the sync and reads results.
     */
    int process_launch(const complexf* in, int n_in,
                       int max_syms, state_t& state,
                       unsigned ss_history,
                       bool skip_input_copy = false);

    /**
     * @brief Synchronize and read results after process_launch().
     *
     * @param sym_out      Output symbol buffer (may be NULL to skip copy).
     * @param max_syms     Same value passed to process_launch().
     * @param state        [in/out] Timing recovery state.
     * @param n_consumed   [out] Number of input samples consumed.
     * @return Number of symbols produced.
     */
    int process_complete(complexf* sym_out, int max_syms,
                         state_t& state, int& n_consumed);

    /**
     * @brief Stage input samples into managed memory.
     *
     * Copies host data to the managed input buffer via CPU memcpy.
     * On Grace Hopper, the GPU reads this coherently — no DMA needed.
     * Returns the managed pointer for use by upstream kernels (e.g. AGC).
     */
    float* stage_input(const float* host_src, int n_samples,
                       bool host_src_pinned = false);

    float* device_input() const { return d_samples_dev; }
    float* symbols_um() const { return d_symbols_um; }
    cudaStream_t stream() const { return d_stream; }
    int max_input() const { return d_max_input; }

    /** Per-kernel GPU time from last process_complete() (microseconds). */
    uint32_t last_timing_us() const { return d_last_timing_us; }
    uint32_t last_interp_us() const { return d_last_interp_us; }

    /** Host-side section times of the last process_launch() (microseconds):
     *  est    — estimator section (energy-search launch or dead-reckoned
     *           interp-param computation),
     *  interp — batch_interp_kernel launch + trailing event record.
     *  Always measured (two steady_clock reads per section). */
    uint32_t last_launch_est_us() const { return d_lnch_est_us; }
    uint32_t last_launch_interp_us() const { return d_lnch_interp_us; }
    /** Whether the last call launched the GPU energy-search kernel. */
    bool last_launched_energy() const { return d_last_launched_energy; }

    /**
     * @brief Set estimation period (1 = every call, N = every Nth call).
     *
     * On skipped calls, the NCO dead-reckons: the carry-adjusted basepoint
     * from the previous batch is reused and only the interpolation params
     * (n_syms, base) are recomputed on the CPU for the current buffer
     * geometry. The batch_interp_kernel still runs every call.
     */
    void set_estimation_period(int period) { d_estimation_period = std::max(1, period); }

    /** Set number of polyphase arms for energy search (8..64, default 32). */
    void set_energy_k(int k) { d_energy_k = std::max(2, std::min(k, (int)ENERGY_K_MAX)); }

    /** Cap the symbols the energy search scores per call (the interpolation
     *  still produces all symbols). The default 2048-symbol cap is enough for
     *  steady-state tracking in the measured file runs while cutting the
     *  near-2-SPS frontend cost. Pass 0 for a full-buffer diagnostic search. */
    void set_energy_search_cap(int cap) { d_energy_search_cap = std::max(0, cap); }

    /** Enable/disable per-kernel profiling events (d_evt_start/mid).
     *  Disabling saves ~20µs per call from eliminated cudaEventRecord +
     *  cudaEventElapsedTime calls. */
    void set_profiling(bool enable) { d_profiling = enable; }

    /** Reset estimation call counter (e.g. after unlock for re-acquisition). */
    void reset_estimation_counter() { d_estimation_call_count = 0; }

private:
    bool d_profiling = true;  // per-kernel timing events
    float* d_samples_dev;     // device memory: input [max_input * 2]
    float* d_samples_pinned;  // pinned host memory: staging [max_input * 2]
    float* d_symbols_um;      // managed memory: output [max_syms * 2]
    float* d_subfilters_dev;  // device memory: subfilter taps [n_phases * subfilt_len]
    double* d_state_um;       // unified memory: timing recovery result + interp params [8 doubles]
                              // [0]=tau, [1]=strength, [2]=n_syms, [3]=base (fractional
                              // sample offset of symbol 0 relative to ss_history),
                              // [4]=frac diag, [5]=drift_accum
    double* d_state_pinned;   // pinned host buffer for CPU→GPU state DMA (ARM64 coherence fix)
    cudaStream_t d_stream;

    cudaEvent_t d_evt_start, d_evt_mid, d_evt_end;
    uint32_t d_last_timing_us = 0, d_last_interp_us = 0;
    uint32_t d_lnch_est_us = 0, d_lnch_interp_us = 0;
    bool d_last_launched_energy = false;

    double d_stride;          // samples per symbol (nominal fractional NCO stride)
    double d_stride_eff = 0.0; // per-batch effective stride = d_stride * (1 + rate_corr),
                               // snapshotted at process_launch and used consistently
                               // for estimation, interpolation and the carry
    double d_bank_delay;      // exact branch-0 group delay (input samples)
    int d_n_phases, d_taps_per_phase;
    int d_max_input, d_max_syms;
    int d_last_n_in = 0, d_last_n_copy = 0, d_last_ss_history = 0;
    int d_last_launch_syms = 0;     // interpolation outputs covered by the last grid
    double d_last_pred_base = 0.0;  // NCO-predicted base at launch (for hysteresis)
    bool d_last_ran_est = false;    // whether a fresh estimate ran this call
    int d_last_track_mode = 0;      // energy search ran in tracking mode (interp
                                    // at predicted phase, measurement in state[6])
    double d_track_resid = 0.0;     // un-corrected fraction of the last tracking
                                    // delta (excluded from the rate innovation)
    double d_rho_innov_sum = 0.0;   // windowed rate-loop accumulators: innovation
    long d_rho_win_syms = 0;        // sum and symbol baseline since last update

    /** Windowed rate-loop integration (see .cu for the noise analysis). */
    void rate_loop_update(state_t& state, double innov_sym, long interval_syms);
    long d_reject_count = 0;        // total rejected estimates (rate-limited log)
    int d_consec_rejects = 0;       // consecutive rejects (wedge-escape snap at 8)
    long d_syms_since_accept = 0;   // symbols emitted since the last accepted
                                    // estimate (rate-loop measurement interval)
    long d_rate_update_count = 0;   // accepted rate-loop updates (for logging)

    int d_estimation_period = 1;        // timing recovery estimation period (1 = every call)
    int d_estimation_call_count = 0;    // monotonic call counter
    int d_energy_k = ENERGY_K_DEFAULT;  // polyphase arms for energy search
    int d_energy_search_cap = 2048;     // max symbols the energy search scores

    // Polyphase energy search buffers
    float* d_partial_energy_dev = nullptr;   // [ENERGY_N_BLOCKS * ENERGY_K_MAX]
    unsigned int* d_energy_done_cnt = nullptr; // atomic completion counter

    /** Output symbol count for a batch starting at fractional offset `base`. */
    int fractional_n_syms(double base, int n_copy, int ss_history,
                          int max_syms_limit) const;

    /** Eye-diagram energy test on the raw samples in d_samples_pinned:
     *  true if sampling at base_meas carries more energy than at the
     *  NCO-predicted base (d_last_pred_base). */
    bool eye_energy_accept(double base_meas) const;

    /** Rate-limited 'estimate rejected' log (first 10, then every 1000th). */
    void log_reject(double base_meas);

    /** CPU-side interp param computation (NCO dead reckoning on skip calls). */
    void compute_interp_params_cpu(const state_t& state, int n_copy,
                                   int ss_history, int max_syms);
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_TIMING_RECOVERY_KERNEL_CUH */
