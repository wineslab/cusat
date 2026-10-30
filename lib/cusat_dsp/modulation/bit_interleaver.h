/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Block bit interleaver (EN 302 307-1 clause 5.3.3, EN 302 307-2 clause
 * 5.3.3 and Table 9a). The coded bits are written column-wise into
 * bits_per_sym columns of `rows` bits each and read out row-wise; bit c of
 * every symbol comes from column order[c].
 */

#ifndef INCLUDED_CUSAT_DSP_MODULATION_BIT_INTERLEAVER_H
#define INCLUDED_CUSAT_DSP_MODULATION_BIT_INTERLEAVER_H

#include <cusat_dsp/modcod_types.h>
#include <array>

namespace cusat_dsp {

struct bit_interleaver_t {
    unsigned rows = 0;                   // bits per column
    int n_cols = 0;                      // 0: no interleaving (QPSK)
    std::array<unsigned, 8> col_offset{}; // first bit of the column read as bit c
};

bit_interleaver_t make_bit_interleaver(modulation_t modulation,
                                       code_rate_t rate,
                                       fecframe_size_t framesize);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_MODULATION_BIT_INTERLEAVER_H */
