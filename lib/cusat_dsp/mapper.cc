/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tx mapper: bit interleaving (modulation/bit_interleaver.h) and
 * constellation mapping, both carried out on the GPU. The receiver's GPU
 * demapper applies the inverse permutation.
 */

#include "mapper.h"
#include "cuda/mapper_kernel.cuh"
#include <cstring>

namespace cusat_dsp {

mapper::mapper(fecframe_size_t framesize,
               code_rate_t rate,
               modulation_t constellation)
    : d_modulation(constellation),
      d_constellation(make_constellation(constellation, rate, framesize)),
      d_interleaver(make_bit_interleaver(constellation, rate, framesize)),
      d_mapper_cuda(nullptr)
{
    d_fecframe_len = fecframe_bits(framesize);
    d_n_mod = d_constellation.bits_per_sym;
    if (constellation == MOD_128APSK) {
        // 6 stuffing bits in, 12 padding symbols out: 9258 + 12 = 103 slots
        d_interleaved_len = (d_fecframe_len + 6) / d_n_mod;
        d_xfecframe_len = d_interleaved_len + 12;
    } else {
        d_interleaved_len = d_xfecframe_len = d_fecframe_len / d_n_mod;
    }

    const std::vector<float> lut = build_constellation_lut();
    unsigned int col_offsets[8];
    int stride;
    get_interleave_params(col_offsets, stride);

    unsigned int gpu_fecframe_bits = d_fecframe_len;
    if (constellation == MOD_128APSK) {
        gpu_fecframe_bits = (d_fecframe_len + 6 + 7) & ~7u; // byte-aligned, stuffing = 0
        d_padded_fecframe.assign(gpu_fecframe_bits / 8, 0);
        d_padding_point = d_constellation.points[127];     // all-ones label
    }
    d_mapper_cuda = new cuda::mapper_cuda(
        lut.data(), d_n_mod, gpu_fecframe_bits, d_interleaved_len, col_offsets, stride);
}

mapper::~mapper() { delete d_mapper_cuda; }

void mapper::interleave_and_map(const uint8_t* fecframe_in, complexf* symbols_out)
{
    if (d_modulation != MOD_128APSK) {
        d_mapper_cuda->map(fecframe_in, reinterpret_cast<float*>(symbols_out));
        return;
    }
    memcpy(d_padded_fecframe.data(), fecframe_in, d_fecframe_len / 8);
    d_mapper_cuda->map(d_padded_fecframe.data(), reinterpret_cast<float*>(symbols_out));
    for (unsigned int i = d_interleaved_len; i < d_xfecframe_len; i++)
        symbols_out[i] = d_padding_point;
}

std::vector<float> mapper::build_constellation_lut() const
{
    std::vector<float> lut(2 * d_constellation.n_points);
    for (int label = 0; label < d_constellation.n_points; label++) {
        lut[2 * label] = d_constellation.points[label].real();
        lut[2 * label + 1] = d_constellation.points[label].imag();
    }
    return lut;
}

void mapper::get_interleave_params(unsigned int col_offsets[8], int& stride) const
{
    if (d_interleaver.n_cols == 0) {
        // QPSK: consecutive bit pairs, no interleaving
        const unsigned int pairs[8] = { 0, 1 };
        memcpy(col_offsets, pairs, sizeof(pairs));
        stride = 2;
        return;
    }
    memcpy(col_offsets, d_interleaver.col_offset.data(), 8 * sizeof(unsigned int));
    stride = 1;
}

} // namespace cusat_dsp
