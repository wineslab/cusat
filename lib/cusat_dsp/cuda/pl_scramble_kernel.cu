/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA PL scrambling kernel.
 *
 * PL scrambling on TX: payload[i] *= conj(descrambling_seq[i])
 * PL descrambling on RX: payload[i] *= descrambling_seq[i]
 *
 * Both are element-wise complex multiplications — embarrassingly parallel.
 */

#include "pl_scramble_kernel.cuh"
#include "cuda_util.cuh"
#include <cstdio>
#include <cstring>

namespace cusat_dsp {
namespace cuda {

/*
 * Complex multiply-by-conjugate: out[i] = a[i] * conj(b[i])
 * conj(b) = (br, -bi), so:
 *   re = ar*br + ai*bi
 *   im = ai*br - ar*bi
 */
__global__ void complex_multiply_conjugate_kernel(float* __restrict__ inout,
                                                  const float* __restrict__ seq,
                                                  int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float ar = inout[i * 2];
    float ai = inout[i * 2 + 1];
    float br = seq[i * 2];
    float bi = seq[i * 2 + 1];

    inout[i * 2]     = ar * br + ai * bi;
    inout[i * 2 + 1] = ai * br - ar * bi;
}

/*
 * Complex multiply: out[i] = a[i] * b[i]
 *   re = ar*br - ai*bi
 *   im = ar*bi + ai*br
 */
__global__ void complex_multiply_kernel(float* __restrict__ inout,
                                        const float* __restrict__ seq,
                                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float ar = inout[i * 2];
    float ai = inout[i * 2 + 1];
    float br = seq[i * 2];
    float bi = seq[i * 2 + 1];

    inout[i * 2]     = ar * br - ai * bi;
    inout[i * 2 + 1] = ar * bi + ai * br;
}

/* ----- Host API ----- */

pl_scramble_cuda::pl_scramble_cuda(const complexf* descrambling_seq, int max_payload_len)
    : d_max_len(max_payload_len),
      d_seq_dev(nullptr),
      d_payload_dev(nullptr)
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));

    /* Upload descrambling sequence to device (constant for the gold code) */
    CUDA_CHECK(cudaMalloc(&d_seq_dev, max_payload_len * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_seq_dev, descrambling_seq,
                          max_payload_len * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Allocate device buffer for payload */
    CUDA_CHECK(cudaMalloc(&d_payload_dev, max_payload_len * 2 * sizeof(float)));
}

pl_scramble_cuda::~pl_scramble_cuda()
{
    if (d_seq_dev) cudaFree(d_seq_dev);
    if (d_payload_dev) cudaFree(d_payload_dev);
    cudaStreamDestroy(d_stream);
}

void pl_scramble_cuda::scramble_inplace(complexf* payload, int payload_len)
{
    /* H2D */
    CUDA_CHECK(cudaMemcpyAsync(d_payload_dev, payload,
                               payload_len * 2 * sizeof(float),
                               cudaMemcpyHostToDevice, d_stream));

    int threads = 256;
    int blocks = (payload_len + threads - 1) / threads;
    complex_multiply_conjugate_kernel<<<blocks, threads, 0, d_stream>>>(
        d_payload_dev, d_seq_dev, payload_len);
    CUDA_CHECK_LAUNCH();

    /* D2H */
    CUDA_CHECK(cudaMemcpyAsync(payload, d_payload_dev,
                               payload_len * 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, d_stream));
    CUDA_CHECK(cudaStreamSynchronize(d_stream));
}

} // namespace cuda
} // namespace cusat_dsp
