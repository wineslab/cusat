/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "fec/code_params.h"
#include <iterator>
#include <stdexcept>

namespace cusat_dsp {

namespace {

struct bch_row_t {
    code_rate_t rate;
    uint32_t k, n, t;
};

constexpr bch_row_t NORMAL[] = {
    { RATE_1_4, 16008, 16200, 12 },
    { RATE_1_3, 21408, 21600, 12 },
    { RATE_2_5, 25728, 25920, 12 },
    { RATE_1_2, 32208, 32400, 12 },
    { RATE_3_5, 38688, 38880, 12 },
    { RATE_2_3, 43040, 43200, 10 },
    { RATE_3_4, 48408, 48600, 12 },
    { RATE_4_5, 51648, 51840, 12 },
    { RATE_5_6, 53840, 54000, 10 },
    { RATE_8_9, 57472, 57600, 8 },
    { RATE_9_10, 58192, 58320, 8 },
    { RATE_VLSNR_SET1_QPSK_2_9, 14208, 14400, 12 },
    { RATE_13_45, 18528, 18720, 12 },
    { RATE_9_20, 28968, 29160, 12 },
    { RATE_90_180, 32208, 32400, 12 },
    { RATE_96_180, 34368, 34560, 12 },
    { RATE_11_20, 35448, 35640, 12 },
    { RATE_100_180, 35808, 36000, 12 },
    { RATE_104_180, 37248, 37440, 12 },
    { RATE_26_45, 37248, 37440, 12 },
    { RATE_18_30, 38688, 38880, 12 },
    { RATE_28_45, 40128, 40320, 12 },
    { RATE_23_36, 41208, 41400, 12 },
    { RATE_116_180, 41568, 41760, 12 },
    { RATE_20_30, 43008, 43200, 12 },
    { RATE_124_180, 44448, 44640, 12 },
    { RATE_25_36, 44808, 45000, 12 },
    { RATE_128_180, 45888, 46080, 12 },
    { RATE_13_18, 46608, 46800, 12 },
    { RATE_132_180, 47328, 47520, 12 },
    { RATE_22_30, 47328, 47520, 12 },
    { RATE_135_180, 48408, 48600, 12 },
    { RATE_140_180, 50208, 50400, 12 },
    { RATE_7_9, 50208, 50400, 12 },
    { RATE_154_180, 55248, 55440, 12 },
};

constexpr bch_row_t SHORT[] = {
    { RATE_1_4, 3072, 3240, 12 },
    { RATE_1_3, 5232, 5400, 12 },
    { RATE_2_5, 6312, 6480, 12 },
    { RATE_1_2, 7032, 7200, 12 },
    { RATE_3_5, 9552, 9720, 12 },
    { RATE_2_3, 10632, 10800, 12 },
    { RATE_3_4, 11712, 11880, 12 },
    { RATE_4_5, 12432, 12600, 12 },
    { RATE_5_6, 13152, 13320, 12 },
    { RATE_8_9, 14232, 14400, 12 },
    { RATE_11_45, 3792, 3960, 12 },
    { RATE_4_15, 4152, 4320, 12 },
    { RATE_14_45, 4872, 5040, 12 },
    { RATE_7_15, 7392, 7560, 12 },
    { RATE_8_15, 8472, 8640, 12 },
    { RATE_26_45, 9192, 9360, 12 },
    { RATE_32_45, 11352, 11520, 12 },
    { RATE_VLSNR_SET1_BPSKS_1_5, 2512, 2680, 12 },
    { RATE_VLSNR_SET1_BPSKS_11_45, 3792, 3960, 12 },
    { RATE_VLSNR_SET2_BPSK_1_5, 3072, 3240, 12 },
    { RATE_VLSNR_SET2_BPSK_4_15, 4152, 4320, 12 },
    { RATE_VLSNR_SET2_BPSK_1_3, 5232, 5400, 12 },
};

constexpr bch_row_t MEDIUM[] = {
    { RATE_VLSNR_SET1_BPSK_1_5, 5660, 5840, 12 },
    { RATE_VLSNR_SET1_BPSK_11_45, 7740, 7920, 12 },
    { RATE_VLSNR_SET1_BPSK_1_3, 10620, 10800, 12 },
};

const bch_row_t* find_row(fecframe_size_t framesize, code_rate_t rate)
{
    const bch_row_t* begin = framesize == FECFRAME_NORMAL ? std::begin(NORMAL)
                             : framesize == FECFRAME_SHORT ? std::begin(SHORT)
                                                           : std::begin(MEDIUM);
    const bch_row_t* end = framesize == FECFRAME_NORMAL ? std::end(NORMAL)
                           : framesize == FECFRAME_SHORT ? std::end(SHORT)
                                                         : std::end(MEDIUM);
    for (const bch_row_t* r = begin; r != end; r++)
        if (r->rate == rate)
            return r;
    return nullptr;
}

} // namespace

fec_params_t get_fec_params(fecframe_size_t framesize, code_rate_t rate)
{
    const bch_row_t* r = find_row(framesize, rate);
    if (!r)
        throw std::invalid_argument("get_fec_params: code rate not defined for this frame size");
    return { r->k, r->n, r->t, r->n, (uint32_t)fecframe_bits(framesize) };
}

bool has_fec_params(fecframe_size_t framesize, code_rate_t rate) { return find_row(framesize, rate) != nullptr; }

} // namespace cusat_dsp
