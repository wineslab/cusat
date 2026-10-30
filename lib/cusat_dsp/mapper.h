/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tx mapper: bit interleaver (row-to-column) + constellation mapping.
 */

#ifndef INCLUDED_CUSAT_DSP_MAPPER_H
#define INCLUDED_CUSAT_DSP_MAPPER_H

#include "modulation/bit_interleaver.h"
#include "modulation/constellation.h"
#include <cusat_dsp/modcod_types.h>
#include <cusat_dsp/compat.h>
#include <cstdint>
#include <vector>

namespace cusat_dsp {
namespace cuda { class mapper_cuda; }

class mapper
{
public:
    mapper(fecframe_size_t framesize,
           code_rate_t rate,
           modulation_t constellation);

    ~mapper();

    /**
     * @brief Interleave FECFRAME bits and map to constellation symbols.
     *
     * @param fecframe_in Packed FECFRAME bytes (n_bits/8 bytes, MSB first).
     * @param symbols_out Output XFECFRAME symbols (xfecframe_len complex values).
     */
    void interleave_and_map(const uint8_t* fecframe_in, complexf* symbols_out);

    unsigned int fecframe_len() const { return d_fecframe_len; }
    unsigned int xfecframe_len() const { return d_xfecframe_len; }
    int n_mod() const { return d_n_mod; }

    /** Build constellation LUT for GPU pipeline registration.
     *  Returns 2^n_mod entries × 2 floats (re, im interleaved). */
    std::vector<float> build_constellation_lut() const;

    /** Get column offsets and stride for the interleave permutation. */
    void get_interleave_params(unsigned int col_offsets[8], int& stride) const;

private:
    modulation_t d_modulation;
    unsigned int d_fecframe_len;
    unsigned int d_xfecframe_len;
    unsigned int d_interleaved_len; // symbols out of the interleaver (before padding)
    int d_n_mod;
    constellation_t d_constellation;
    bit_interleaver_t d_interleaver;
    std::vector<uint8_t> d_padded_fecframe; // 128APSK: FECFRAME + 6 stuffing bits
    complexf d_padding_point;               // 128APSK: post-interleaver padding symbol
    cuda::mapper_cuda* d_mapper_cuda;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_MAPPER_H */
