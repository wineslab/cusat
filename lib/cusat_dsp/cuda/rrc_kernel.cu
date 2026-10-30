/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA RRC polyphase pulse shaping kernel.
 *
 * Each output sample is independent: thread (n, p) computes the p-th polyphase
 * output for input symbol n by dot-producting sub_taps symbols with sub-filter p.
 * This maps trivially to GPU threads — one thread per output sample.
 */

#include "rrc_kernel.cuh"
#include "cuda_util.cuh"
#include <cstdio>
#include <cstring>

namespace cusat_dsp {
namespace cuda {

/* Taps in constant memory (max 256 floats = 128 taps × 2 phases for sps=2) */
static __constant__ float d_taps_const[256];

/*
 * Kernel: one thread per output sample.
 *
 * symbols: [history_len + n_symbols] complex values (interleaved float pairs)
 * samples: [n_symbols * sps] complex output values
 *
 * Thread index i corresponds to output sample i.
 *   symbol index n = i / sps
 *   phase p = i % sps
 *   output[i] = sum_{k=0}^{sub_taps-1} taps[(sub_taps-1-k)*sps + p] * symbols[n + k]
 *
 * Taps are already stored reversed+interleaved in d_taps_const, so:
 *   output[i] = sum_{k=0}^{sub_taps-1} d_taps_const[k*sps + p] * symbols[n + k]
 */
__global__ void rrc_polyphase_kernel(const float* __restrict__ symbols,
                                     float* __restrict__ samples,
                                     int n_symbols,
                                     int sub_taps,
                                     int sps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_out = n_symbols * sps;
    if (i >= n_out)
        return;

    int n = i / sps;     // symbol index
    int p = i % sps;     // polyphase phase

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    for (int k = 0; k < sub_taps; k++) {
        float t = d_taps_const[k * sps + p];
        float xr = symbols[(n + k) * 2];
        float xi = symbols[(n + k) * 2 + 1];
        acc_re += xr * t;
        acc_im += xi * t;
    }

    samples[i * 2]     = acc_re;
    samples[i * 2 + 1] = acc_im;
}

/* ----- Host API ----- */

rrc_cuda::rrc_cuda(int sps, int sub_taps, int history_len,
                   const float* interleaved_taps)
    : d_sps(sps),
      d_sub_taps(sub_taps),
      d_history_len(history_len),
      d_symbols_dev(nullptr),
      d_samples_dev(nullptr),
      d_alloc_symbols(0),
      d_alloc_samples(0)
{
    /* Upload taps to constant memory */
    int n_taps = sub_taps * sps;
    if (n_taps > 256) {
        fprintf(stderr, "rrc_cuda: tap count %d exceeds constant memory limit 256\n", n_taps);
        return;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_taps_const, interleaved_taps,
                                  n_taps * sizeof(float)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));
}

rrc_cuda::~rrc_cuda()
{
    if (d_symbols_dev) cudaFree(d_symbols_dev);
    if (d_samples_dev) cudaFree(d_samples_dev);
    cudaStreamDestroy(d_stream);
}

void rrc_cuda::ensure_buffers(unsigned int n_symbols)
{
    unsigned int buf_len = d_history_len + n_symbols;
    if (buf_len > d_alloc_symbols) {
        if (d_symbols_dev) cudaFree(d_symbols_dev);
        d_alloc_symbols = buf_len;
        CUDA_CHECK(cudaMalloc(&d_symbols_dev, d_alloc_symbols * 2 * sizeof(float)));
    }

    unsigned int out_len = n_symbols * d_sps;
    if (out_len > d_alloc_samples) {
        if (d_samples_dev) cudaFree(d_samples_dev);
        d_alloc_samples = out_len;
        CUDA_CHECK(cudaMalloc(&d_samples_dev, d_alloc_samples * 2 * sizeof(float)));
    }
}

unsigned int rrc_cuda::pulse_shape(const float* symbol_buf,
                                   unsigned int n_symbols,
                                   float* samples_out)
{
    unsigned int buf_len = d_history_len + n_symbols;
    unsigned int n_out = n_symbols * d_sps;

    ensure_buffers(n_symbols);

    /* Copy symbol buffer (history + current) to device */
    CUDA_CHECK(cudaMemcpyAsync(d_symbols_dev, symbol_buf,
                               buf_len * 2 * sizeof(float),
                               cudaMemcpyHostToDevice, d_stream));

    /* Launch kernel */
    int threads = 256;
    int blocks = (n_out + threads - 1) / threads;
    rrc_polyphase_kernel<<<blocks, threads, 0, d_stream>>>(
        d_symbols_dev, d_samples_dev, n_symbols, d_sub_taps, d_sps);
    CUDA_CHECK_LAUNCH();

    /* Copy output back to host */
    CUDA_CHECK(cudaMemcpyAsync(samples_out, d_samples_dev,
                               n_out * 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, d_stream));
    CUDA_CHECK(cudaStreamSynchronize(d_stream));

    return n_out;
}

} // namespace cuda
} // namespace cusat_dsp
