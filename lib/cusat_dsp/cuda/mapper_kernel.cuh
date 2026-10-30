/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA mapper: bit interleaver + constellation mapping.
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_MAPPER_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_MAPPER_KERNEL_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace cusat_dsp {
namespace cuda {

class mapper_cuda
{
public:
    /**
     * @brief Construct CUDA mapper for a given constellation + interleave config.
     *
     * @param constellation_lut  Host array of 2^n_mod constellation points (re,im pairs).
     * @param n_mod              Bits per symbol (2, 3, 4, 5, or 6).
     * @param fecframe_len       Total FECFRAME bits (e.g. 64800).
     * @param xfecframe_len      Number of output symbols = fecframe_len / n_mod.
     * @param col_offsets        Array of n_mod column offsets for interleaving.
     * @param stride             Stride between consecutive symbols within a column.
     */
    mapper_cuda(const float* constellation_lut, int n_mod,
                unsigned int fecframe_len, unsigned int xfecframe_len,
                const unsigned int* col_offsets, int stride);

    ~mapper_cuda();

    /**
     * @brief Map packed FECFRAME bits to constellation symbols.
     *
     * @param fecframe_packed  Device or host pointer to packed FECFRAME (fecframe_len/8 bytes).
     * @param symbols_out      Device or host pointer for output symbols (xfecframe_len × 2 floats).
     */
    void map(const uint8_t* fecframe_packed, float* symbols_out);

private:
    int d_n_mod;
    unsigned int d_fecframe_len;
    unsigned int d_xfecframe_len;
    unsigned int d_col_offsets[8];
    int d_stride;

    // Device buffers
    float* d_constellation_dev;    // 2^n_mod × 2 floats (re, im interleaved)
    uint8_t* d_fecframe_dev;       // packed input
    float* d_symbols_dev;          // output symbols (re, im interleaved)
    cudaStream_t d_stream;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_MAPPER_KERNEL_CUH */
