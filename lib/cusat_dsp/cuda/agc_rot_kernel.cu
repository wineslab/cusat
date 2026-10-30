/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU fused AGC + frequency rotation kernels.
 *
 * Kernel 1 — agc_rms_kernel <<<1, 256>>>:
 *   Parallel reduction of sum(|x[n]|²) via shared memory. Thread 0 computes
 *   RMS, applies IIR gain update: gain += alpha*(ref/rms - gain), clamps to
 *   [0, max_gain], and writes result to agc_state[0].
 *
 * Kernel 2 — rotate_gain_kernel <<<ceil(N/256), 256>>>:
 *   Each thread computes phase = phase_start + n*phase_inc, rotates the
 *   sample by (cos(phase), sin(phase)), and multiplies by the AGC gain
 *   from agc_state[0]. In-place on the sample buffer.
 *
 * Both kernels are launched on the same CUDA stream (implicit ordering).
 */

#include "agc_rot_kernel.cuh"
#include "cuda_util.cuh"
#include <cstdio>
#include <cmath>

namespace cusat_dsp {
namespace cuda {

/* ========== GPU Kernels ========== */

/**
 * Kernel 1: AGC RMS estimation and IIR gain update (multi-block).
 *
 * Multiple blocks (AGC_NUM_BLOCKS), 256 threads each.  Each block reduces
 * its portion of the input, writes the partial sum to partial_sums[blockIdx.x],
 * and atomically increments a completion counter.  The LAST block to finish
 * combines all partial sums, computes RMS, applies the IIR gain update, and
 * resets the counter for the next call.
 *
 * Using multiple blocks spreads work across SMs, providing enough warps to
 * hide NVLink-C2C coherence latency when reading CPU-written unified memory
 * on GH200 (~300ns per cache line).  1 block = 8 warps (insufficient);
 * 32 blocks = 256 warps (good latency hiding).
 *
 * agc_state layout: [0]=gain, [1]=ref, [2]=max_gain, [3]=rate
 */
static constexpr int AGC_NUM_BLOCKS = 32;

__global__ void agc_rms_kernel(
    const float* __restrict__ samples,   // [N*2] interleaved re,im
    float* __restrict__ agc_state,       // [4]: gain, ref, max_gain, rate
    float* __restrict__ partial_sums,    // [AGC_NUM_BLOCKS]
    unsigned int* __restrict__ done_cnt, // completion counter (init to 0)
    int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Local accumulator for |x[n]|²
    float local_sum = 0.0f;
    for (int n = gid; n < N; n += stride) {
        float re = samples[2 * n];
        float im = samples[2 * n + 1];
        local_sum += re * re + im * im;
    }

    // Shared-memory parallel reduction within block
    __shared__ float s_sum[256];
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        // Store this block's partial sum
        partial_sums[blockIdx.x] = s_sum[0];
        __threadfence();  // ensure partial_sums visible to all blocks

        // Atomically count completed blocks
        unsigned int completed = atomicInc(done_cnt, gridDim.x - 1);
        // atomicInc wraps: returns old value, increments, resets to 0 when >= gridDim.x-1

        if (completed == gridDim.x - 1) {
            // Last block to finish — combine all partial sums
            float total = 0.0f;
            for (int i = 0; i < (int)gridDim.x; i++)
                total += partial_sums[i];

            float rms = sqrtf(total / (float)N);

            float gain = agc_state[0];
            float ref = agc_state[1];
            float max_gain = agc_state[2];
            float rate = agc_state[3];

            if (rms > 1e-20f) {
                float target_gain = ref / rms;
                float alpha = 1.0f - expf(-rate * (float)N);
                gain += alpha * (target_gain - gain);
                if (gain > max_gain) gain = max_gain;
                if (gain < 0.0f) gain = 0.0f;
            }

            agc_state[0] = gain;
        }
    }
}

/**
 * Kernel 2: Fused frequency rotation + AGC gain multiply.
 *
 * Multi-block grid: ceil(N/256) blocks × 256 threads.
 * Each thread computes exact phase = phase_start + n * phase_inc,
 * rotates the sample, and multiplies by gain. In-place.
 *
 * Uses sincosf (full precision) rather than __sincosf (fast, ~2 ULP error)
 * to avoid phase drift at large sample indices (n*phase_inc > 100 rad).
 */
__global__ void rotate_gain_kernel(
    float* __restrict__ samples,         // [N*2] interleaved re,im (in-place)
    const float* __restrict__ agc_state, // [4]: gain at [0]
    int N,
    float phase_start,
    float phase_inc)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float gain = agc_state[0];
    float phase = phase_start + (float)n * phase_inc;

    float s, c;
    sincosf(phase, &s, &c);

    float re = samples[2 * n];
    float im = samples[2 * n + 1];

    // Rotation: (re + j*im) * (c + j*s) = (re*c - im*s) + j*(re*s + im*c)
    samples[2 * n]     = (re * c - im * s) * gain;
    samples[2 * n + 1] = (re * s + im * c) * gain;
}

/* ========== Host-side class ========== */

agc_rot_cuda::agc_rot_cuda(float ref, float gain, float max_gain, float rate)
{
    CUDA_CHECK(cudaMallocHost(&d_agc_state_um, 4 * sizeof(float)));
    d_agc_state_um[0] = gain;
    d_agc_state_um[1] = ref;
    d_agc_state_um[2] = max_gain;
    d_agc_state_um[3] = rate;

    // Multi-block reduction scratch lives on device memory: the last block
    // reads partial sums written by peer blocks, so device-scope fences are
    // sufficient and no C2C host-pinned ordering is involved.
    CUDA_CHECK(cudaMalloc(&d_partial_sums, AGC_NUM_BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_done_cnt, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_done_cnt, 0, sizeof(unsigned int)));

    CUDA_CHECK(cudaEventCreate(&d_evt_start));
    CUDA_CHECK(cudaEventCreate(&d_evt_agc_done));
    CUDA_CHECK(cudaEventCreate(&d_evt_rot_done));

    fprintf(stderr, "cuda: agc_rot_cuda: ref=%.2f gain=%.2f max_gain=%.1f rate=%.2e"
            " (multi-block AGC: %d blocks)\n",
            ref, gain, max_gain, rate, AGC_NUM_BLOCKS);
}

agc_rot_cuda::~agc_rot_cuda()
{
    cudaFreeHost(d_agc_state_um);
    cudaFree(d_partial_sums);
    cudaFree(d_done_cnt);
    if (d_evt_start) cudaEventDestroy(d_evt_start);
    if (d_evt_agc_done) cudaEventDestroy(d_evt_agc_done);
    if (d_evt_rot_done) cudaEventDestroy(d_evt_rot_done);
}

void agc_rot_cuda::process(float* samples_um, int N,
                            float phase_start, float phase_inc,
                            cudaStream_t stream)
{
    if (d_profiling)
        cudaEventRecord(d_evt_start, stream);

    // Kernel 1: AGC RMS estimation + IIR gain update (multi-block)
    CUDA_CHECK(cudaMemsetAsync(d_done_cnt, 0, sizeof(unsigned int), stream));
    agc_rms_kernel<<<AGC_NUM_BLOCKS, 256, 0, stream>>>(
        samples_um, d_agc_state_um, d_partial_sums, d_done_cnt, N);
    CUDA_CHECK_LAUNCH();
    if (d_profiling)
        cudaEventRecord(d_evt_agc_done, stream);

    // Kernel 2: Fused rotation + gain (reads gain written by kernel 1)
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    rotate_gain_kernel<<<grid_size, block_size, 0, stream>>>(
        samples_um, d_agc_state_um, N, phase_start, phase_inc);
    CUDA_CHECK_LAUNCH();
    if (d_profiling)
        cudaEventRecord(d_evt_rot_done, stream);
}

float agc_rot_cuda::gain() const
{
    return d_agc_state_um[0];
}

void agc_rot_cuda::query_events()
{
    float ms_agc = 0.0f, ms_rot = 0.0f;
    cudaEventElapsedTime(&ms_agc, d_evt_start, d_evt_agc_done);
    cudaEventElapsedTime(&ms_rot, d_evt_agc_done, d_evt_rot_done);
    d_last_agc_us = (uint32_t)(ms_agc * 1000.0f);
    d_last_rot_us = (uint32_t)(ms_rot * 1000.0f);
}

void agc_rot_cuda::reset(float gain)
{
    d_agc_state_um[0] = gain;
}

} // namespace cuda
} // namespace cusat_dsp
