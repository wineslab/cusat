/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "modulation/bit_interleaver.h"
#include <stdexcept>

namespace cusat_dsp {

namespace {

/* Column read-out order, written as in the standard ("210" = bit 0 from
 * column 2, ...). DVB-S2 rates and all unlisted cases read the columns in
 * order. */
const char* column_order(modulation_t modulation, code_rate_t r)
{
    switch (modulation) {
    case MOD_8PSK:
        if (r == RATE_3_5)
            return "210";
        if (r == RATE_25_36 || r == RATE_13_18 || r == RATE_7_15 || r == RATE_8_15 || r == RATE_26_45)
            return "102";
        return "012";
    case MOD_8APSK:
        return "012";
    case MOD_16APSK:
        switch (r) {
        case RATE_26_45: return "3201";
        case RATE_3_5: return "3210";
        case RATE_28_45: return "3012";
        case RATE_23_36: return "3021";
        case RATE_25_36: return "2310";
        case RATE_13_18: return "3021";
        case RATE_7_9:
        case RATE_140_180: return "3210";
        case RATE_154_180: return "0321";
        case RATE_90_180: return "3210";
        case RATE_96_180: return "2310";
        case RATE_100_180: return "2301";
        default: return "0123";
        }
    case MOD_32APSK:
        if (r == RATE_2_3 || r == RATE_32_45)
            return "21430";
        if (r == RATE_128_180 || r == RATE_132_180 || r == RATE_22_30)
            return "40312";
        if (r == RATE_7_9 || r == RATE_140_180)
            return "40213";
        return "01234";
    case MOD_64APSK:
        if (r == RATE_7_9 || r == RATE_140_180)
            return "201543";
        if (r == RATE_4_5)
            return "124053";
        if (r == RATE_5_6)
            return "421053";
        if (r == RATE_128_180)
            return "305214";
        return "520143";
    case MOD_128APSK:
        return (r == RATE_7_9 || r == RATE_140_180) ? "4130256" : "4250316";
    case MOD_256APSK:
        switch (r) {
        case RATE_135_180: return "50743612";
        case RATE_128_180: return "75642301";
        case RATE_116_180: return "40372156";
        case RATE_124_180: return "46320571";
        default: return "01234567";
        }
    default:
        throw std::invalid_argument("make_bit_interleaver: unsupported modulation");
    }
}

} // namespace

bit_interleaver_t make_bit_interleaver(modulation_t modulation,
                                       code_rate_t rate,
                                       fecframe_size_t framesize)
{
    bit_interleaver_t il;
    if (modulation == MOD_QPSK)
        return il;

    const unsigned n_fec = fecframe_bits(framesize);
    const char* order = column_order(modulation, rate);
    il.n_cols = 0;
    while (order[il.n_cols])
        il.n_cols++;
    // 128APSK carries 6 stuffing bits ahead of the interleaver
    const unsigned stuffed = modulation == MOD_128APSK ? n_fec + 6 : n_fec;
    il.rows = stuffed / il.n_cols;
    for (int c = 0; c < il.n_cols; c++)
        il.col_offset[c] = il.rows * (order[c] - '0');
    return il;
}

} // namespace cusat_dsp
