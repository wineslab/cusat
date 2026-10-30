/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA mapper: bit interleaver + constellation mapping.
 *
 * One thread per output symbol.  Each thread:
 *   1. Reads n_mod bits from the packed FECFRAME using column offsets
 *      (applying the DVB-S2 bit interleaver permutation as a read pattern).
 *   2. Forms a LUT index from the extracted bits.
 *   3. Writes the constellation point from the precomputed LUT.
 */

#include "mapper_kernel.cuh"
#include "cuda_util.cuh"
#include <cstdio>
#include <cstring>

namespace cusat_dsp {
namespace cuda {

__global__ void mapper_kernel_impl(
    const uint8_t* __restrict__ fecframe_packed,
    float* __restrict__ symbols_out,
    const float* __restrict__ constellation_lut,
    int xfecframe_len,
    int n_mod,
    unsigned int col0, unsigned int col1, unsigned int col2,
    unsigned int col3, unsigned int col4, unsigned int col5,
    unsigned int col6, unsigned int col7,
    int stride)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= xfecframe_len)
        return;

    /* Extract n_mod bits from packed input using column offsets */
    unsigned int cols[8] = { col0, col1, col2, col3, col4, col5, col6, col7 };
    int lut_idx = 0;
    for (int b = 0; b < n_mod; b++) {
        int bit_pos = (int)cols[b] + j * stride;
        int byte_idx = bit_pos >> 3;
        int bit_in_byte = 7 - (bit_pos & 7);
        int bit = (fecframe_packed[byte_idx] >> bit_in_byte) & 1;
        lut_idx |= (bit << (n_mod - 1 - b));
    }

    /* Write constellation point (re, im interleaved) */
    symbols_out[j * 2]     = constellation_lut[lut_idx * 2];
    symbols_out[j * 2 + 1] = constellation_lut[lut_idx * 2 + 1];
}

/* ----- Host API ----- */

mapper_cuda::mapper_cuda(const float* constellation_lut, int n_mod,
                         unsigned int fecframe_len, unsigned int xfecframe_len,
                         const unsigned int* col_offsets, int stride)
    : d_n_mod(n_mod), d_fecframe_len(fecframe_len),
      d_xfecframe_len(xfecframe_len), d_stride(stride)
{
    for (int i = 0; i < 8; i++)
        d_col_offsets[i] = (i < n_mod) ? col_offsets[i] : 0;

    CUDA_CHECK(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));

    /* Upload constellation LUT */
    int lut_entries = 1 << n_mod;
    CUDA_CHECK(cudaMalloc(&d_constellation_dev, lut_entries * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_constellation_dev, constellation_lut,
                          lut_entries * 2 * sizeof(float), cudaMemcpyHostToDevice));

    /* Allocate working buffers */
    CUDA_CHECK(cudaMalloc(&d_fecframe_dev, fecframe_len / 8));
    CUDA_CHECK(cudaMalloc(&d_symbols_dev, xfecframe_len * 2 * sizeof(float)));

    fprintf(stderr, "mapper_cuda: n_mod=%d fecframe=%u xfecframe=%u stride=%d "
            "cols=[%u,%u,%u,%u,%u,%u]\n",
            n_mod, fecframe_len, xfecframe_len, stride,
            d_col_offsets[0], d_col_offsets[1], d_col_offsets[2],
            d_col_offsets[3], d_col_offsets[4], d_col_offsets[5]);
}

mapper_cuda::~mapper_cuda()
{
    cudaFree(d_constellation_dev);
    cudaFree(d_fecframe_dev);
    cudaFree(d_symbols_dev);
    cudaStreamDestroy(d_stream);
}

void mapper_cuda::map(const uint8_t* fecframe_packed, float* symbols_out)
{
    /* Upload packed FECFRAME */
    CUDA_CHECK(cudaMemcpyAsync(d_fecframe_dev, fecframe_packed,
                               d_fecframe_len / 8, cudaMemcpyHostToDevice, d_stream));

    /* Launch mapper kernel */
    int threads = 256;
    int blocks = (d_xfecframe_len + threads - 1) / threads;
    mapper_kernel_impl<<<blocks, threads, 0, d_stream>>>(
        d_fecframe_dev, d_symbols_dev, d_constellation_dev,
        d_xfecframe_len, d_n_mod,
        d_col_offsets[0], d_col_offsets[1], d_col_offsets[2],
        d_col_offsets[3], d_col_offsets[4], d_col_offsets[5],
        d_col_offsets[6], d_col_offsets[7],
        d_stride);
    CUDA_CHECK_LAUNCH();

    /* Download symbols */
    CUDA_CHECK(cudaMemcpyAsync(symbols_out, d_symbols_dev,
                               d_xfecframe_len * 2 * sizeof(float),
                               cudaMemcpyDeviceToHost, d_stream));
    CUDA_CHECK(cudaStreamSynchronize(d_stream));
}

} // namespace cuda
} // namespace cusat_dsp
