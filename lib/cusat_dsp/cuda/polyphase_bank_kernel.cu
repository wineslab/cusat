/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU polyphase filterbank energy search for timing estimation.
 *
 * Multi-block kernel evaluates K timing arms simultaneously across the
 * input batch at any fractional stride (samples per symbol). Arm a sits
 * at sample offset a * stride / K; the matched-filter output for symbol s
 * of arm a is interpolated at fractional position off_a + s*stride using
 * the same integer-basepoint / polyphase-subfilter mapping as the
 * interpolation kernel, so the estimated offset is directly usable as the
 * NCO base. Each block processes a tile of symbols, computing energy for
 * all arms. The last block (via atomic counter) performs final reduction,
 * parabolic peak interpolation, mod-stride anchoring to the NCO's
 * predicted base, and writes timing parameters to unified memory.
 *
 * K is a runtime parameter (8..64) to allow trading timing resolution
 * against compute cost.
 */

#include "polyphase_bank_kernel.cuh"
#include "cuda_util.cuh"
#include <algorithm>
#include <cstdio>
#include <cmath>

namespace cusat_dsp {
namespace cuda {

/**
 * Polyphase filterbank energy search kernel.
 *
 * Each block processes a tile of symbols for all K arms, accumulating
 * per-arm matched-filter output energy. Subfilter taps are read from
 * global memory (L2/read-only cache serves the whole bank — same proven
 * pattern as batch_interp_kernel). Keeping the bank OUT of shared memory
 * matters twice: a 128x64-tap bank in smem (32 KB) suffered up to 32-way
 * bank conflicts (each lane hits a different subfilter row at fractional
 * stride), and the large per-block footprint prevented co-residency with
 * LDPC decode blocks — measured 20 ms p99 stalls of this kernel behind
 * batch-64 decodes at high prio, vs none for the small-smem interp kernel.
 * The last block to finish combines partial energies, finds the best arm
 * via parabolic interpolation, and writes timing state compatible with
 * batch_interp_kernel.
 */
template <int KT>   // compile-time arm count: keeps arm_energy[] in registers
                    // (a runtime-bound K loop forces the dynamically indexed
                    // local array into local memory — measured ~250 µs/call
                    // vs ~tens of µs unrolled). KT=0 selects the generic
                    // runtime-K fallback for nonstandard arm counts.
__global__ void polyphase_energy_kernel(
    const float* __restrict__ samples,      // [N*2] interleaved re,im
    const float* __restrict__ subfilters,   // [n_phases * subfilt_len]
    double* __restrict__ state_out,         // [8] unified memory
    float* __restrict__ partial_energy,     // [gridDim.x * K] device memory
    unsigned int* __restrict__ done_cnt,    // atomic completion counter
    int N,                    // total input samples
    double stride, int n_phases, int subfilt_len,
    int ss_history,
    int bank_floats,          // n_phases * subfilt_len (smem bank size)
    int tile_cap,             // tile capacity in complex samples (host-computed)
    double prev_base, double prev_drift_accum, int prev_init,
    int track_mode, double base_trim,
    int K_rt,                 // number of timing arms (runtime, 8..64)
    int search_syms,          // symbols evaluated for energy (possibly capped)
    int full_max_syms_limit,  // limit for output n_syms
    int full_max_syms_buf)    // buffer limit for output n_syms
{
    const int K = (KT > 0) ? KT : K_rt;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_blocks = gridDim.x;
    (void)bank_floats;

    // --- Shared memory layout: input IQ tile only (taps come from L2) ---
    extern __shared__ float smem[];
    float* s_tile = smem;

    // --- Compute symbol range for this block ---
    int syms_per_block = (search_syms + n_blocks - 1) / n_blocks;
    int sym_start = bid * syms_per_block;
    int sym_end = sym_start + syms_per_block;
    if (sym_end > search_syms) sym_end = search_syms;
    int n_block_syms = sym_end - sym_start;

    // --- Phase 2: Load input tile into shared memory ---
    // Positions evaluated by this block: p = off_a + s*stride with
    // off_a in [0, stride) and s in [sym_start, sym_end). The FIR for
    // basepoint ip reads the L samples ending at start+ip or start+ip+1
    // (filter-bank convention, interpolator.h).
    int ip_min = (n_block_syms > 0) ? (int)floor((double)sym_start * stride) : 0;
    int tile_base = ss_history + ip_min + 1 - subfilt_len;
    int tile_floats = (n_block_syms > 0) ? tile_cap * 2 : 0;

    for (int i = tid; i < tile_floats; i += blockDim.x) {
        int gi = tile_base * 2 + i;
        s_tile[i] = (gi >= 0 && gi < N * 2) ? samples[gi] : 0.0f;
    }
    __syncthreads();

    // --- Phase 3: Energy search across all K arms ---
    // Each thread processes a subset of symbols for all arms.
    // arm_energy[k] accumulated in local memory (K is runtime).
    // Sized to ENERGY_K_MAX; only K elements are used.
    float arm_energy[ENERGY_K_MAX];
    for (int k = 0; k < K; k++)
        arm_energy[k] = 0.0f;

    for (int idx = tid; idx < n_block_syms; idx += blockDim.x) {
        int sym = sym_start + idx;
        double p_sym = (double)sym * stride;

        for (int k = 0; k < K; k++) {
            double p = p_sym + (double)k * stride / (double)K;
            int ip = (int)floor(p);
            double fr = p - (double)ip;
            int q = (int)(fr * (double)n_phases + 0.5);
            int newest = ip + (q > 0 ? 1 : 0);
            int fi = (q > 0 && q < n_phases) ? n_phases - q : 0;

            // window ending at start+newest, taps newest-first
            int local_newest = (newest - ip_min + subfilt_len - 1) * 2;
            const float* taps = subfilters + fi * subfilt_len;
            float re = 0.0f, im = 0.0f;
            for (int j = 0; j < subfilt_len; j++) {
                int off = local_newest - j * 2;
                re += taps[j] * s_tile[off];
                im += taps[j] * s_tile[off + 1];
            }
            arm_energy[k] += re * re + im * im;
        }
    }

    // --- Phase 4: Reduction ---
    // Warp-level shuffle reduction for all K arms simultaneously
    for (int k = 0; k < K; k++) {
        float val = arm_energy[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        arm_energy[k] = val;  // lane 0 of each warp has warp sum
    }

    // Warp leaders write to shared memory for cross-warp reduction
    const int n_warps = blockDim.x / 32;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    // Reuse tile portion of shared memory for warp_sums (tile no longer needed)
    __syncthreads();
    float* s_warp_sums = s_tile;  // reuse tile smem: [n_warps * K] floats

    if (lane == 0) {
        for (int k = 0; k < K; k++)
            s_warp_sums[warp_id * K + k] = arm_energy[k];
    }
    __syncthreads();

    // Thread 0: cross-warp sum for all K arms, write to global partial_energy
    if (tid == 0) {
        for (int k = 0; k < K; k++) {
            float total = 0.0f;
            for (int w = 0; w < n_warps; w++)
                total += s_warp_sums[w * K + k];
            partial_energy[bid * K + k] = total;
        }
    }

    // --- Phase 5: Last-block finalization ---
    // The whole last block participates: the cross-block sum is one thread
    // PER ARM (deterministic fixed-order b loop), not one thread for all
    // arms — at 512 blocks a single-thread sum would cost tens of µs.
    __shared__ unsigned int s_is_last;
    __shared__ float s_total[ENERGY_K_MAX];
    if (tid == 0) {
        __threadfence();  // ensure partial_energy visible to all blocks

        unsigned int completed = atomicInc(done_cnt, n_blocks - 1);
        // atomicInc wraps: returns old value, increments, resets to 0 when >= n_blocks-1
        s_is_last = (completed == (unsigned int)(n_blocks - 1));
    }
    __syncthreads();
    if (!s_is_last)
        return;

    for (int k = tid; k < K; k += blockDim.x) {
        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++)
            total += partial_energy[b * K + k];
        s_total[k] = total;
    }
    __syncthreads();

    if (tid == 0) {
        {
            const float* total_e = s_total;

            // --- Find best arm ---
            int k_best = 0;
            float e_best = total_e[0];
            for (int k = 1; k < K; k++) {
                if (total_e[k] > e_best) {
                    e_best = total_e[k];
                    k_best = k;
                }
            }

            // --- Parabolic interpolation ---
            float E_m = total_e[(k_best - 1 + K) % K];
            float E_0 = total_e[k_best];
            float E_p = total_e[(k_best + 1) % K];
            float denom = E_m - 2.0f * E_0 + E_p;
            float delta = 0.0f;
            if (fabsf(denom) > 1e-20f)
                delta = 0.5f * (E_m - E_p) / denom;
            if (delta > 0.5f) delta = 0.5f;
            if (delta < -0.5f) delta = -0.5f;

            double tau_raw = fmod((double)(k_best + delta) / K + 1.0, 1.0);
            double off_best = tau_raw * stride;   // sample offset in [0, stride)

            // --- Strength metric: peak-to-average energy ratio ---
            float e_sum = 0.0f;
            for (int k = 0; k < K; k++) e_sum += total_e[k];
            float e_avg = e_sum / K;
            double strength = (e_avg > 1e-30f)
                ? ((double)E_0 - (double)e_avg) / (double)e_avg
                : 0.0;

            // --- Anchor mod-stride to the NCO's predicted base ---
            // The measurement only fixes the phase modulo one symbol; pick
            // the representative nearest the dead-reckoned NCO position so
            // the consumed-sample accounting stays continuous. Clamp the
            // base above -2 (a +stride re-anchor: one-symbol slip) so all
            // FIR reads stay within the retained history.
            double base = off_best;
            double drift_accum = prev_drift_accum;
            if (prev_init) {
                base = off_best + stride * round((prev_base - off_best) / stride);
                while (base < -2.0)
                    base += stride;
                drift_accum += (base - prev_base) / stride;
            } else {
                drift_accum = 0.0;
            }
            // Tracking mode: interpolate this batch at the smooth
            // dead-reckoned phase; export the measurement separately so the
            // host disciplines the NCO (phase nudge + rate loop) without
            // the estimator noise modulating the symbols directly.
            double meas_base = base;
            if (track_mode && prev_init)
                base = prev_base + base_trim;

            // --- Number of output symbols (same bounds as the host's
            //     fractional_n_syms): consumption + read-safety. ---
            double avail = (double)(N - ss_history) - base;
            int n_syms = 0;
            if (avail >= stride) {
                int n_cons = (int)(avail / stride);
                double read_avail = (double)(N - ss_history - 3) - base;
                int n_read = (read_avail >= 0.0)
                    ? (int)(read_avail / stride) + 1 : 0;
                n_syms = n_cons < n_read ? n_cons : n_read;
                if (n_syms > full_max_syms_limit) n_syms = full_max_syms_limit;
                if (n_syms > full_max_syms_buf) n_syms = full_max_syms_buf;
                if (n_syms < 0) n_syms = 0;
            }

            state_out[0] = tau_raw;
            state_out[1] = strength;
            state_out[2] = (double)n_syms;
            state_out[3] = base;
            state_out[4] = base - floor(base);
            state_out[5] = drift_accum;
            state_out[6] = meas_base;
            state_out[7] = 0.0;
        }
    }
}

/* ========== Host-side launcher ========== */

void launch_polyphase_energy_search(
    const float* samples_dev,
    const float* subfilters_dev,
    double* state_um,
    float* partial_energy,
    unsigned int* done_cnt,
    int N, double stride, int n_phases, int subfilt_len,
    int ss_history, int max_syms_limit, int max_syms_buf,
    double prev_base, double prev_drift_accum, int prev_init,
    int track_mode, double base_trim,
    int energy_k,
    int energy_search_cap,
    cudaStream_t stream)
{
    const int K = energy_k;
    const int block_size = 256;

    // Opt-in to dynamic shared memory beyond the default 48KB limit.
    // Query device capability at runtime to support different GPUs.
    static int MAX_SMEM = 0;
    static bool smem_configured = false;
    if (!smem_configured) {
        int device, optin = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&optin,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device);
        // The opt-in limit covers static + dynamic shared memory; the
        // kernel's static usage (s_total etc.) must be subtracted or
        // cudaFuncSetAttribute rejects the value (cudaErrorInvalidValue,
        // which then surfaces as a sticky error at the next launch).
        auto config = [&](const void* func) {
            cudaFuncAttributes fa{};
            cudaFuncGetAttributes(&fa, func);
            int dyn_max = optin - (int)fa.sharedSizeBytes;
            cudaFuncSetAttribute(func,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_max);
            if (MAX_SMEM == 0 || dyn_max < MAX_SMEM)
                MAX_SMEM = dyn_max;
        };
        config((const void*)polyphase_energy_kernel<8>);
        config((const void*)polyphase_energy_kernel<16>);
        config((const void*)polyphase_energy_kernel<32>);
        config((const void*)polyphase_energy_kernel<64>);
        config((const void*)polyphase_energy_kernel<0>);
        smem_configured = true;
    }

    // Shared memory budget (input tile only — taps are served from L2):
    //   smem = tile_cap * 8
    //   tile_cap = ceil(syms_per_block * stride) + subfilt_len + 8
    int bank_floats = n_phases * subfilt_len;
    int max_tile_cap = MAX_SMEM / (2 * (int)sizeof(float));
    int max_spb = (int)((double)(max_tile_cap - subfilt_len - 8) / stride);
    if (max_spb < 1) max_spb = 1;

    int max_possible_syms = (int)((double)(N - ss_history) / stride);
    if (max_possible_syms < 1) max_possible_syms = 1;
    if (max_possible_syms > max_syms_limit) max_possible_syms = max_syms_limit;
    if (max_possible_syms > max_syms_buf) max_possible_syms = max_syms_buf;

    // Optionally cap energy search to a contiguous window of symbols.
    // energy_search_cap=0 means no cap (search full buffer).
    // The full max_syms_limit/buf are passed separately for the output
    // n_syms computation — the interpolation kernel still produces
    // ALL symbols using the estimated base.
    int energy_search_syms = max_possible_syms;
    if (energy_search_cap > 0 && energy_search_syms > energy_search_cap)
        energy_search_syms = energy_search_cap;

    // Adaptive grid: ~256 symbols per 256-thread block keeps every thread
    // of a block on at least one symbol (the thread->symbol mapping idles
    // blockDim - syms_per_block threads otherwise), while the grid grows
    // with the workload. max_spb caps the per-block tile for smem at very
    // large strides.
    int spb_target = std::min(256, max_spb);
    int n_blocks = (energy_search_syms + spb_target - 1) / spb_target;
    if (n_blocks > ENERGY_N_BLOCKS) n_blocks = ENERGY_N_BLOCKS;
    if (n_blocks < 1) n_blocks = 1;
    if (energy_search_syms > n_blocks * max_spb)
        energy_search_syms = n_blocks * max_spb;   // smem-capped

    int syms_per_block = (energy_search_syms + n_blocks - 1) / n_blocks;
    int tile_cap = (int)std::ceil((double)syms_per_block * stride)
                   + subfilt_len + 8;

    int smem_bytes = tile_cap * 2 * (int)sizeof(float);

    // Ensure enough shared memory for warp reduction (n_warps * K floats)
    int warp_reduction_bytes = (block_size / 32) * K * (int)sizeof(float);
    if (smem_bytes < warp_reduction_bytes)
        smem_bytes = warp_reduction_bytes;

    // Reset atomic counter (belt-and-suspenders: atomicInc auto-resets when
    // exactly n_blocks execute, but a prior failed launch could leave it stale)
    cudaMemsetAsync(done_cnt, 0, sizeof(unsigned int), stream);

#define ENERGY_KERNEL_ARGS \
        samples_dev, subfilters_dev, state_um, \
        partial_energy, done_cnt, \
        N, stride, n_phases, subfilt_len, \
        ss_history, \
        bank_floats, tile_cap, \
        prev_base, prev_drift_accum, prev_init, track_mode, base_trim, \
        K, energy_search_syms, max_syms_limit, max_syms_buf

    // Compile-time K for the standard arm counts (keeps the per-arm
    // accumulators in registers); generic fallback otherwise.
    switch (K) {
    case 8:
        polyphase_energy_kernel<8><<<n_blocks, block_size, smem_bytes,
            stream>>>(ENERGY_KERNEL_ARGS);
        break;
    case 16:
        polyphase_energy_kernel<16><<<n_blocks, block_size, smem_bytes,
            stream>>>(ENERGY_KERNEL_ARGS);
        break;
    case 32:
        polyphase_energy_kernel<32><<<n_blocks, block_size, smem_bytes,
            stream>>>(ENERGY_KERNEL_ARGS);
        break;
    case 64:
        polyphase_energy_kernel<64><<<n_blocks, block_size, smem_bytes,
            stream>>>(ENERGY_KERNEL_ARGS);
        break;
    default:
        polyphase_energy_kernel<0><<<n_blocks, block_size, smem_bytes,
            stream>>>(ENERGY_KERNEL_ARGS);
        break;
    }
#undef ENERGY_KERNEL_ARGS
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace cusat_dsp
