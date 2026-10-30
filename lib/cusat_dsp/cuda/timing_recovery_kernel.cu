/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU feedforward polyphase timing recovery — fused single-sync pipeline
 * with a fractional-stride symbol-timing NCO.
 *
 * Two-kernel pipeline, ONE cudaStreamSynchronize:
 *   Kernel 1 — timing estimation + interp param computation:
 *     polyphase energy search over K arms → base offset → n_syms.
 *     Thread 0 of the last block writes all params to unified memory.
 *
 *   Kernel 2 — Batch fractional polyphase interpolation:
 *     Reads n_syms and the fractional base offset from unified memory
 *     (written by K1). Output symbol k is interpolated at input position
 *     ss_history + base + k*stride: the integer part selects the basepoint
 *     sample, the fractional part selects the polyphase subfilter.
 *     Launched with max possible grid size; threads with k >= n_syms exit.
 *
 * Consumed-samples accounting: n_consumed = floor(base + n_syms*stride) is
 * an integer; the fractional remainder is carried in state_t::nco_base so
 * sampling instants stay continuous across process() calls at any stride.
 *
 * The CPU never reads GPU state between kernels — single sync at end.
 */

#include "timing_recovery_kernel.cuh"
#include "polyphase_bank_kernel.cuh"
#include "cuda_util.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace cusat_dsp {

namespace cuda {

// The X410's 24.576 Msps clock at 6 Msym/s gives 4.096 samples/symbol.
// Keep that common fractional-rate operating point on the smooth NCO path.
// The old 4.0 cutoff put 4.096 on the direct-measurement path, where an
// occasional ambiguous energy estimate immediately moved the sampling eye.
static constexpr double TRACKING_STRIDE_MAX = 4.5;

/* ========== GPU Kernels ========== */

/**
 * Batch fractional polyphase interpolation with shared memory tiling.
 *
 * Multi-block grid: launched with max possible symbols.
 * Each thread reads n_syms and the fractional base offset from state[]
 * (written by kernel 1 on the same stream) and exits early if k >= n_syms.
 *
 * Per output symbol k the NCO position is p = base + k*stride (in input
 * samples, relative to `start`). The integer part of p selects the
 * basepoint sample m_k = start + floor(p); the fractional part selects
 * the polyphase subfilter round(frac(p) * n_phases) — generalizing the
 * old integer-stride kernel where all symbols shared one (bp, mu) pair.
 *
 * Shared memory tiling: each block cooperatively loads a contiguous tile of
 * input samples into shared memory, then each thread performs its FIR from
 * shared memory. Subfilter taps are read from global memory (the per-thread
 * subfilter index varies with k at fractional stride; the 21 KB tap bank is
 * served from L2/read-only cache).
 *
 * Tile size per block (passed by the host, which must use the same value
 * for the smem allocation): ceil(blockDim.x * stride) + subfilt_len + 8.
 */
__global__ void batch_interp_kernel(
    const float* __restrict__ samples,      // [M*2] interleaved re,im
    float* __restrict__ symbols_out,        // [N_sym*2] interleaved re,im
    const float* __restrict__ subfilters,   // [n_phases * subfilt_len]
    const double* __restrict__ state,       // reads [2]=n_syms, [3]=base
    double stride,
    int start,         // first symbol basepoint origin (ss_history)
    int n_phases,
    int subfilt_len,
    int n_samples,     // total input samples (for bounds checking)
    int tile_len)      // tile capacity in complex samples (host-computed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Read interpolation params from unified memory (written by kernel 1)
    int n_sym = (int)state[2];
    double base = state[3];

    // --- Shared memory tile ---
    // Tile covers the range of samples needed by all threads in this block.
    int k0 = blockIdx.x * blockDim.x;
    int ip0 = (int)floor(base + (double)k0 * stride);
    // newest sample of a window is basepoint or basepoint + 1 (see below),
    // so the tile starts subfilt_len - 1 samples before basepoint ip0
    int tile_base = start + ip0 + 1 - subfilt_len;

    extern __shared__ float s_tile[];

    // Cooperative load: all threads participate (even if k >= n_sym)
    int total_floats = tile_len * 2;
    int global_base = tile_base * 2;
    int buf_limit = n_samples * 2;
    for (int i = threadIdx.x; i < total_floats; i += blockDim.x) {
        int gi = global_base + i;
        s_tile[i] = (gi >= 0 && gi < buf_limit) ? samples[gi] : 0.0f;
    }
    __syncthreads();

    // Early exit for out-of-range threads (after tile load + barrier)
    if (k >= n_sym) return;

    // Fractional NCO position for this symbol
    // Filter-bank convention (interpolator.h): phase ph applied to the
    // window ending at sample n (taps newest-first) yields the filtered
    // signal at n - D - ph / n_phases. With q = round(fr * n_phases) the
    // position base + k * stride = ip + fr is reached with n = ip + (q > 0)
    // and ph = (n_phases - q) mod n_phases.
    double p = base + (double)k * stride;
    int ip = (int)floor(p);
    double fr = p - (double)ip;
    int q = (int)(fr * (double)n_phases + 0.5);
    int newest = ip + (q > 0 ? 1 : 0);
    int ph = (q > 0 && q < n_phases) ? n_phases - q : 0;
    const float* taps = subfilters + ph * subfilt_len;

    // FIR from shared memory (offset relative to tile_base), newest first
    int local_newest = (start + newest - tile_base) * 2;

    float out_re = 0.0f, out_im = 0.0f;
    for (int j = 0; j < subfilt_len; j++) {
        int off = local_newest - j * 2;
        out_re += taps[j] * s_tile[off];
        out_im += taps[j] * s_tile[off + 1];
    }

    symbols_out[2 * k]     = out_re;
    symbols_out[2 * k + 1] = out_im;
}

/* ========== Host-side class ========== */

timing_recovery_cuda::timing_recovery_cuda(int n_phases, int subfilt_len,
                               const float* subfilters,
                               double sps, double bank_delay,
                               int max_input)
    : d_stride(sps),
      d_stride_eff(sps),
      d_bank_delay(bank_delay),
      d_n_phases(n_phases),
      d_taps_per_phase(subfilt_len),
      d_max_input(max_input),
      d_max_syms((int)((double)max_input / sps) + 2)
{
    // Pinned host + device memory for input samples.
    // At 245 Msps, input is ~800 KB — too large for NVLink-C2C coherence.
    // Pinned staging + DMA copy yields non-blocking transfer and HBM reads.
    CUDA_CHECK(cudaMalloc(&d_samples_dev, max_input * 2 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&d_samples_pinned, max_input * 2 * sizeof(float)));

    // Pinned host memory for output symbols and state (GPU writes via
    // NVLink, CPU reads directly after stream sync). Using cudaMallocHost
    // instead of cudaMallocManaged avoids illegal memory access errors
    // on GH200 caused by managed-memory page fault storms when the CUDA
    // context has thousands of managed allocations (ACM flight slots).
    CUDA_CHECK(cudaMallocHost(&d_symbols_um, d_max_syms * 2 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&d_state_um, 8 * sizeof(double)));

    // Pinned host buffer for CPU→GPU state transfer when estimation is skipped.
    // Avoids ARM64 weak-ordering issues with direct CPU writes to managed memory.
    CUDA_CHECK(cudaMallocHost(&d_state_pinned, 8 * sizeof(double)));

    // Device memory for subfilter taps (uploaded once, read-only)
    int taps_size = n_phases * subfilt_len * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_subfilters_dev, taps_size));
    CUDA_CHECK(cudaMemcpy(d_subfilters_dev, subfilters, taps_size,
                          cudaMemcpyHostToDevice));

    // High priority: timing-critical path must preempt LDPC decode work
    int hi_prio;
    cudaDeviceGetStreamPriorityRange(nullptr, &hi_prio);
    CUDA_CHECK(cudaStreamCreateWithPriority(&d_stream, cudaStreamNonBlocking, hi_prio));

    CUDA_CHECK(cudaEventCreate(&d_evt_start));
    CUDA_CHECK(cudaEventCreate(&d_evt_mid));
    CUDA_CHECK(cudaEventCreate(&d_evt_end));

    // Allocate polyphase energy search buffers (sized for max K)
    int partial_count = ENERGY_N_BLOCKS * ENERGY_K_MAX;
    CUDA_CHECK(cudaMalloc(&d_partial_energy_dev, partial_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy_done_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_energy_done_cnt, 0, sizeof(unsigned int)));

    // Opt-in to dynamic shared memory beyond the default 48KB limit for the
    // interp kernel (large strides need bigger tiles).
    {
        int device, max_smem = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cudaFuncSetAttribute(batch_interp_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
    }

    fprintf(stderr, "cuda: timing_recovery_cuda: stride=%.6f, %d subfilters x %d taps, "
            "bank_delay=%.6f (energy search, K=%d, %d blocks)\n",
            d_stride, n_phases, subfilt_len, d_bank_delay,
            d_energy_k, ENERGY_N_BLOCKS);
}

timing_recovery_cuda::~timing_recovery_cuda()
{
    cudaFree(d_samples_dev);
    cudaFreeHost(d_samples_pinned);
    cudaFreeHost(d_symbols_um);
    cudaFreeHost(d_state_um);
    cudaFreeHost(d_state_pinned);
    cudaFree(d_subfilters_dev);
    cudaFree(d_partial_energy_dev);
    cudaFree(d_energy_done_cnt);
    cudaEventDestroy(d_evt_start);
    cudaEventDestroy(d_evt_mid);
    cudaEventDestroy(d_evt_end);
    cudaStreamDestroy(d_stream);
}

float* timing_recovery_cuda::stage_input(const float* host_src, int n_samples,
                                         bool host_src_pinned)
{
    size_t bytes = n_samples * 2 * sizeof(float);
    if (host_src_pinned) {
        CUDA_CHECK(cudaMemcpyAsync(d_samples_dev, host_src, bytes,
                                   cudaMemcpyHostToDevice, d_stream));
    } else {
        // Pageable callers go through an explicit pinned staging buffer; direct
        // cudaMemcpyAsync from pageable memory may block on hidden staging.
        memcpy(d_samples_pinned, host_src, bytes);
        CUDA_CHECK(cudaMemcpyAsync(d_samples_dev, d_samples_pinned, bytes,
                                   cudaMemcpyHostToDevice, d_stream));
    }
    return d_samples_dev;
}

/* Output symbol count for a batch whose first symbol sits at fractional
 * offset `base` (samples, relative to ss_history). Two bounds:
 *   consumption: floor(base + n*stride) <= n_copy - ss_history, so the
 *                retained tail always contains the full filter history;
 *   read safety: the last basepoint (incl. the +1 subfilter rounding
 *                carry) must satisfy m+1 <= n_copy-1.
 */
int timing_recovery_cuda::fractional_n_syms(double base, int n_copy,
                                            int ss_history,
                                            int max_syms_limit) const
{
    double avail = (double)(n_copy - ss_history) - base;
    if (avail < d_stride_eff)
        return 0;
    int n_cons = (int)(avail / d_stride_eff);
    double read_avail = (double)(n_copy - ss_history - 3) - base;
    int n_read = (read_avail >= 0.0) ? (int)(read_avail / d_stride_eff) + 1 : 0;
    int n = std::min(n_cons, n_read);
    n = std::min({n, max_syms_limit, d_max_syms});
    return std::max(n, 0);
}

int timing_recovery_cuda::process_launch(const complexf* in, int n_in,
                                    int max_syms, state_t& state,
                                    unsigned ss_history,
                                    bool skip_input_copy)
{
    int stride_ceil = (int)std::ceil(d_stride);
    if (n_in < (int)ss_history + stride_ceil)
        return 0;

    int n_copy = std::min(n_in, d_max_input);
    d_last_n_in = n_in;
    d_last_n_copy = n_copy;
    d_last_ss_history = (int)ss_history;
    d_last_pred_base = state.init ? state.nco_base : 0.0;

    // Rate loop: run this batch's whole geometry (estimation, interpolation
    // grid, consumption/carry) at the effective stride. rate_corr is the
    // learned TX/RX sample-clock ratio offset; with it the dead-reckoned
    // prediction tracks the true clock instead of growing by drift*symbols
    // between estimates.
    d_stride_eff = d_stride * (1.0 + state.rate_corr);

    if (!skip_input_copy)
        stage_input(reinterpret_cast<const float*>(in), n_copy);

    if (d_profiling)
        CUDA_CHECK(cudaEventRecord(d_evt_start, d_stream));

    // Periodic timing estimation: skip on most calls, dead-reckon the NCO.
    // Always run on first call (!state.init) and every d_estimation_period
    // calls. At low strides, one call can span >100k symbols, so run the
    // tracking-mode energy measurement every call unless the estimator is
    // explicitly frozen.
    d_estimation_call_count++;
    bool low_stride_tracker = (d_stride <= 2.005);
    bool frozen = (d_estimation_period >= 9999);
    bool run_est = (low_stride_tracker && !frozen) ||
                   (d_estimation_period <= 1) || !state.init ||
                   ((d_estimation_call_count % d_estimation_period) == 1);
    d_last_ran_est = run_est;
    d_last_track_mode = 0;
    d_last_launched_energy = false;

    auto t_est0 = std::chrono::steady_clock::now();
    if (run_est) {
        // Tracking mode (stride <= TRACKING_STRIDE_MAX, NCO initialized):
        // the batch is
        // interpolated at the smooth dead-reckoned phase and the energy
        // measurement only disciplines the NCO in process_complete —
        // estimator noise then never modulates the symbols directly.
        // Above TRACKING_STRIDE_MAX, directly using the measured phase
        // remains the validated configuration. First call (!init) always
        // uses the measurement.
        d_last_track_mode =
            (state.init && d_stride <= TRACKING_STRIDE_MAX) ? 1 : 0;
        // Tracking mode uses the configured energy-search cap. The default
        // 2048-symbol window is enough for steady-state tracking in the
        // measured file runs and keeps near-2-SPS frontend cost bounded;
        // pass cap=0 for full-window diagnostics.
        int search_cap = d_energy_search_cap;
        launch_polyphase_energy_search(
            d_samples_dev, d_subfilters_dev, d_state_um,
            d_partial_energy_dev, d_energy_done_cnt,
            n_copy, d_stride_eff, d_n_phases, d_taps_per_phase,
            (int)ss_history, max_syms, d_max_syms,
            d_last_pred_base, state.drift_accum, state.init ? 1 : 0,
            d_last_track_mode, 0.0,
            d_energy_k, search_cap,
            d_stream);
        d_last_launched_energy = true;
    } else {
        // CPU-side: dead-reckon from the NCO carry, compute interp params
        // for the current buffer geometry.
        compute_interp_params_cpu(state, n_copy, (int)ss_history, max_syms);
    }
    auto t_est1 = std::chrono::steady_clock::now();
    d_lnch_est_us = (uint32_t)std::chrono::duration_cast<
        std::chrono::microseconds>(t_est1 - t_est0).count();

    // Grid size: must cover any n_syms the estimator can produce (the base
    // offset can be slightly negative after re-anchoring, hence the +2).
    int max_possible_syms = std::min(
        {(int)((double)(n_copy - (int)ss_history) / d_stride_eff) + 2,
         max_syms, d_max_syms});
    if (max_possible_syms < 0) max_possible_syms = 0;
    d_last_launch_syms = max_possible_syms;
    if (d_profiling)
        CUDA_CHECK(cudaEventRecord(d_evt_mid, d_stream));

    if (max_possible_syms > 0) {
        int block_size = 256;
        int grid_size = (max_possible_syms + block_size - 1) / block_size;
        int tile_len = (int)std::ceil(
                           (double)block_size * std::max(d_stride, d_stride_eff))
                       + d_taps_per_phase + 8;
        int smem_bytes = tile_len * 2 * sizeof(float);

        batch_interp_kernel<<<grid_size, block_size, smem_bytes, d_stream>>>(
            d_samples_dev, d_symbols_um, d_subfilters_dev,
            d_state_um, d_stride_eff, (int)ss_history,
            d_n_phases, d_taps_per_phase,
            n_copy, tile_len);
        CUDA_CHECK_LAUNCH();
    }

    CUDA_CHECK(cudaEventRecord(d_evt_end, d_stream));

    d_lnch_interp_us = (uint32_t)std::chrono::duration_cast<
        std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_est1).count();

    return max_possible_syms;
}

int timing_recovery_cuda::process_complete(complexf* sym_out, int max_syms,
                                      state_t& state, int& n_consumed)
{
    // Check for sticky GPU errors before blocking sync — if a kernel
    // faulted, the sync would deliver SIGSEGV to this thread.
    {
        cudaError_t peek = cudaPeekAtLastError();
        if (peek != cudaSuccess) {
            fprintf(stderr, "cuda: timing_recovery: sticky error before sync: %s (%d)\n",
                    cudaGetErrorString(peek), (int)peek);
            cudaGetLastError(); // clear the sticky error
            n_consumed = 0;
            return 0;
        }
    }

    // Always sync — cudaStreamSynchronize provides the memory fence
    // needed for CPU to see GPU writes to pinned host memory (d_state_um,
    // d_symbols_um) on ARM64's weak memory model. cudaEventQuery does NOT
    // fence, so the old fast-path caused stale reads → zero strength →
    // timing never converges.
    CUDA_CHECK(cudaStreamSynchronize(d_stream));

    // Per-kernel GPU timing (CUDA events) — only when profiling enabled
    if (d_profiling) {
        float ms_om = 0.0f, ms_interp = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_om, d_evt_start, d_evt_mid));
        CUDA_CHECK(cudaEventElapsedTime(&ms_interp, d_evt_mid, d_evt_end));
        d_last_timing_us = (uint32_t)(ms_om * 1000.0f);
        d_last_interp_us = (uint32_t)(ms_interp * 1000.0f);
    }

    int n_syms = (int)d_state_um[2];
    n_syms = std::min({n_syms, max_syms, d_max_syms, d_last_launch_syms});
    double base_used = d_state_um[3];

    if (n_syms <= 0) {
        n_consumed = 0;
        state.tau = d_state_um[0];
        state.drift_accum = d_state_um[5];
        state.init = true;
        return 0;
    }

    if (sym_out)
        memcpy(sym_out, d_symbols_um, n_syms * 2 * sizeof(float));

    // Timing-jump hysteresis for the GPU energy-search estimator (whose
    // base is only known after the sync, so the batch was already
    // interpolated at the measured base): the phase REMEMBERED for future
    // batches is gated — a significant jump is only adopted if the measured
    // sampling instants carry more raw-signal energy than the predicted
    // ones (eye diagram peak test on the pre-AGC samples in
    // d_samples_pinned; AGC gain and rotation do not change relative
    // energies).
    bool accept = true;
    double track_nudge = 0.0;
    if (d_last_ran_est && state.init) {
        if (d_last_track_mode) {
            // Tracking mode: the batch was interpolated at the dead-
            // reckoned phase (base_used == prediction); the measurement is
            // in state[6]. Discipline the NCO with it: a proportional
            // phase nudge (full pull-in for large offsets, gentle for
            // in-noise corrections so estimator noise is low-passed) plus
            // the rate-loop update. Outside the wrap-ambiguity window the
            // measurement is a glitch — ignore it.
            double meas_base = d_state_um[6];
            double delta_sym =
                (meas_base - base_used) / d_stride_eff;
            if (std::fabs(delta_sym) < 0.45) {
                double alpha = (std::fabs(delta_sym) > 0.1) ? 1.0 : 0.3;
                track_nudge = alpha * delta_sym * d_stride_eff;
                // Rate innovation: with partial phase corrections the
                // un-corrected residual (1-alpha)*delta persists into the
                // next measurement; feeding raw deltas to the integrator
                // would count it again every estimate (the loop rings).
                // Only the part beyond the carried residual is new drift.
                double innov = delta_sym - d_track_resid;
                if (d_syms_since_accept > 0)
                    rate_loop_update(state, innov, d_syms_since_accept);
                d_syms_since_accept = 0;
                d_track_resid = (1.0 - alpha) * delta_sym;
                d_consec_rejects = 0;
            } else {
                log_reject(meas_base);
                // Wedge escape: the wrap gate exists to ignore one-off
                // glitches, but when EVERY estimate disagrees by the same
                // ~half-symbol the NCO is parked on the wrong wrap branch
                // (seen in OTA acquisition at stride 2.0: rejects #1..#10
                // with delta ~1.0 sample while unlocked, costing 14+ s of
                // lock time on estimator luck; a true slip in tracking
                // wedges the same way). Dead reckoning can never recover —
                // after 8 consecutive rejects, snap to the measurement and
                // let the downstream sync judge it.
                // The snap is an acquisition escape specifically validated
                // for the dense, <=2-SPS tracker. At 4.096 SPS a long run of
                // rejects can be caused by the dummy PLFRAME's ambiguous
                // energy surface; snapping there converts harmless estimator
                // disagreement into an actual sampling-phase jump.
                if (++d_consec_rejects >= 8 && d_stride <= 2.005) {
                    track_nudge = meas_base - base_used;
                    d_syms_since_accept = 0;
                    d_track_resid = 0.0;
                    d_consec_rejects = 0;
                    fprintf(stderr, "cuda: timing_recovery: 8 consecutive "
                            "rejects — snapping NCO to measured base "
                            "(%+.3f samples)\n", track_nudge);
                }
            }
        } else if (d_stride > TRACKING_STRIDE_MAX) {
            // High stride: the energy search is the proven steady-state
            // tracker (drift per batch is far below 0.03 sym there); keep
            // the eye-energy arbitration for genuine jumps.
            double delta_sym = (base_used - d_last_pred_base) / d_stride_eff;
            if (std::fabs(delta_sym) >= 0.03)
                accept = eye_energy_accept(base_used);
            if (accept)
                d_syms_since_accept = 0;
        }
    }

    // NCO carry: integer samples consumed, fractional remainder carried.
    // Uses the same effective stride the batch was interpolated at, so the
    // carried phase follows the learned clock ratio across batches.
    // On reject, the carry is computed from the predicted base so future
    // batches keep the pre-jump sampling phase (mod stride). The tracking
    // nudge folds the disciplined phase correction into the carry.
    n_consumed = (int)std::floor(base_used + (double)n_syms * d_stride_eff);
    if (n_consumed < 0) n_consumed = 0;
    if (n_consumed > d_last_n_copy) n_consumed = d_last_n_copy;

    double carry_base =
        (accept ? base_used : d_last_pred_base) + track_nudge;
    state.nco_base = carry_base + (double)n_syms * d_stride_eff - (double)n_consumed;
    double tau = carry_base / d_stride_eff;
    state.tau = tau - std::floor(tau);

    d_syms_since_accept += n_syms;
    state.drift_accum = d_state_um[5];
    state.init = true;

    if (!accept)
        log_reject(base_used);

    return n_syms;
}

int timing_recovery_cuda::process(const complexf* in, int n_in,
                             complexf* sym_out, int max_syms,
                             state_t& state, int& n_consumed,
                             unsigned ss_history,
                             bool skip_input_copy)
{
    int max_possible = process_launch(in, n_in, max_syms, state,
                                       ss_history, skip_input_copy);
    if (max_possible == 0 && n_in < (int)ss_history + (int)std::ceil(d_stride)) {
        n_consumed = 0;
        return 0;
    }
    return process_complete(sym_out, max_syms, state, n_consumed);
}

bool timing_recovery_cuda::eye_energy_accept(double base_meas) const
{
    // Eye-diagram peak test on the raw (pre-AGC, pre-rotation) samples in
    // d_samples_pinned: sample |x|² at the predicted vs measured instants;
    // the correct timing phase carries more energy. AGC gain and rotator
    // phase do not change relative energies, so the raw buffer is fine.
    const float* s = d_samples_pinned;
    double e_cur = 0.0, e_new = 0.0;
    double pc = (double)d_last_ss_history + d_last_pred_base;
    double pn = (double)d_last_ss_history + base_meas;
    for (int cnt = 0; cnt < 4000; cnt++) {
        int ic = (int)llround(pc);
        int in_ = (int)llround(pn);
        int hi = ic > in_ ? ic : in_;
        int lo = ic < in_ ? ic : in_;
        if (lo < 0 || hi >= d_last_n_copy)
            break;
        e_cur += (double)s[2 * ic] * s[2 * ic] +
                 (double)s[2 * ic + 1] * s[2 * ic + 1];
        e_new += (double)s[2 * in_] * s[2 * in_] +
                 (double)s[2 * in_ + 1] * s[2 * in_ + 1];
        pc += d_stride_eff;
        pn += d_stride_eff;
    }
    return e_new > e_cur;
}

void timing_recovery_cuda::log_reject(double base_meas)
{
    // Rate-limited: a rejected estimate per estimation period can produce
    // hundreds of thousands of prints in a long run, which itself stalls
    // the real-time path. First 10, then every 1000th (with a counter so
    // the suppressed volume stays visible).
    d_reject_count++;
    if (d_reject_count <= 10 || (d_reject_count % 1000) == 0) {
        fprintf(stderr, "cuda: timing_recovery: estimate rejected #%ld "
                "(base=%.4f pred=%.4f) — keeping NCO phase\n",
                d_reject_count, base_meas, d_last_pred_base);
    }
}

void timing_recovery_cuda::rate_loop_update(state_t& state, double innov_sym,
                                            long interval_syms)
{
    // Windowed rate integrator. The instantaneous rate measurement
    // innov/interval has variance ~ (sigma_meas/interval)^2; at short
    // batches (1-2k symbols) that is several ppm per update, and updating
    // every call turned rate_corr into a random walk. Within-batch timing
    // ramps scale as rate error x symbols-per-batch (up to ~60k), so even
    // 3-5 ppm of walk dirties the eye by ~0.2 symbols. Integrating once
    // per ~1M symbols (every few ms at high symbol rates) keeps the loop
    // fast against real drift (~1e-6, converges in tens of ms) while the
    // per-update noise drops to the 1e-8 level.
    d_rho_innov_sum += innov_sym;
    d_rho_win_syms += interval_syms;
    if (d_rho_win_syms < (1L << 20))
        return;
    state.rate_corr += 0.3 * d_rho_innov_sum / (double)d_rho_win_syms;
    if (state.rate_corr > 50e-6) state.rate_corr = 50e-6;
    if (state.rate_corr < -50e-6) state.rate_corr = -50e-6;
    d_rate_update_count++;
    if (d_rate_update_count <= 4 || (d_rate_update_count % 1000) == 0) {
        fprintf(stderr, "cuda: timing_recovery: rate loop update #%ld: "
                "innov_sum=%+.4f sym over %ld syms -> rate_corr=%+.3f ppm\n",
                d_rate_update_count, d_rho_innov_sum, d_rho_win_syms,
                state.rate_corr * 1e6);
    }
    d_rho_innov_sum = 0.0;
    d_rho_win_syms = 0;
}

void timing_recovery_cuda::compute_interp_params_cpu(
    const state_t& state, int n_copy, int ss_history, int max_syms_limit)
{
    // NCO dead reckoning: reuse the carry-adjusted basepoint from the
    // previous batch (no new estimate). This is what keeps the sampling
    // instants continuous at fractional stride while estimation is skipped
    // or frozen.
    double base = state.nco_base;
    while (base < -2.0)
        base += d_stride_eff;

    int n_syms = fractional_n_syms(base, n_copy, ss_history,
                                   max_syms_limit);

    d_state_pinned[0] = state.tau;
    d_state_pinned[1] = 0.0;   // no strength info on skip
    d_state_pinned[2] = (double)n_syms;
    d_state_pinned[3] = base;
    d_state_pinned[4] = base - std::floor(base);
    d_state_pinned[5] = state.drift_accum;
    d_state_pinned[6] = 0.0;
    d_state_pinned[7] = 0.0;

    // Both d_state_um and d_state_pinned are pinned host memory.
    // Direct memcpy is correct — NVLink-C2C coherence on GH200
    // ensures the GPU sees CPU writes to pinned memory. Using
    // cudaMemcpyAsync for H2H can silently fail on some platforms.
    memcpy(d_state_um, d_state_pinned, 8 * sizeof(double));
}

} // namespace cuda
} // namespace cusat_dsp
