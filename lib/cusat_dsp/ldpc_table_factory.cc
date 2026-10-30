/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Maps (standard, framesize, rate) to the matching ETSI LDPC parity-check
 * table. Table identifiers follow the ETSI annex numbering: b* = normal
 * FECFRAME annex B, c* = short/medium FECFRAME annex C.
 */

#include "ldpc_table_factory.h"
#include "ldpc_tables/dvb_s2_tables.h"
#include "ldpc_tables/dvb_s2x_tables.h"

namespace cusat_dsp {

using namespace ldpc_tables;

static const ldpc_code* select_table(fecframe_size_t framesize,
                                     code_rate_t rate)
{
    if (framesize == FECFRAME_NORMAL) {
        switch (rate) {
        case RATE_1_4:  return &s2_b1;
        case RATE_1_3:  return &s2_b2;
        case RATE_2_5:  return &s2_b3;
        case RATE_1_2:  return &s2_b4;
        case RATE_3_5:  return &s2_b5;
        case RATE_2_3:
            return &s2_b6;
        case RATE_3_4:  return &s2_b7;
        case RATE_4_5:  return &s2_b8;
        case RATE_5_6:  return &s2_b9;
        case RATE_8_9:  return &s2_b10;
        case RATE_9_10: return &s2_b11;
        case RATE_VLSNR_SET1_QPSK_2_9: return &s2x_b1;
        case RATE_13_45: return &s2x_b2;
        case RATE_9_20:  return &s2x_b3;
        case RATE_90_180: return &s2x_b11;
        case RATE_96_180: return &s2x_b12;
        case RATE_11_20: return &s2x_b4;
        case RATE_100_180: return &s2x_b13;
        case RATE_104_180: return &s2x_b14;
        case RATE_26_45: return &s2x_b5;
        case RATE_18_30: return &s2x_b22;
        case RATE_28_45: return &s2x_b6;
        case RATE_23_36: return &s2x_b7;
        case RATE_116_180: return &s2x_b15;
        case RATE_20_30: return &s2x_b23;
        case RATE_124_180: return &s2x_b16;
        case RATE_25_36: return &s2x_b8;
        case RATE_128_180: return &s2x_b17;
        case RATE_13_18: return &s2x_b9;
        case RATE_132_180: return &s2x_b18;
        case RATE_22_30: return &s2x_b24;
        case RATE_135_180: return &s2x_b19;
        case RATE_140_180: return &s2x_b20;
        case RATE_7_9:  return &s2x_b10;
        case RATE_154_180: return &s2x_b21;
        default: return nullptr;
        }
    } else if (framesize == FECFRAME_SHORT) {
        switch (rate) {
        case RATE_1_4:  return &s2_c1;
        case RATE_1_3:  return &s2_c2;
        case RATE_2_5:  return &s2_c3;
        case RATE_1_2:  return &s2_c4;
        case RATE_3_5:
            return &s2_c5;
        case RATE_2_3:  return &s2_c6;
        case RATE_3_4:  return &s2_c7;
        case RATE_4_5:  return &s2_c8;
        case RATE_5_6:  return &s2_c9;
        case RATE_8_9:  return &s2_c10;
        case RATE_11_45: return &s2x_c1;
        case RATE_4_15:  return &s2x_c2;
        case RATE_14_45: return &s2x_c3;
        case RATE_7_15:  return &s2x_c4;
        case RATE_8_15:  return &s2x_c5;
        case RATE_26_45: return &s2x_c6;
        case RATE_32_45: return &s2x_c7;
        case RATE_VLSNR_SET1_BPSKS_1_5: return &s2_c1;
        case RATE_VLSNR_SET1_BPSKS_11_45: return &s2x_c1;
        case RATE_VLSNR_SET2_BPSK_1_5: return &s2_c1;
        case RATE_VLSNR_SET2_BPSK_4_15: return &s2x_c2;
        case RATE_VLSNR_SET2_BPSK_1_3: return &s2_c2;
        default: return nullptr;
        }
    } else {
        switch (rate) {
        case RATE_VLSNR_SET1_BPSK_1_5: return &s2x_c8;
        case RATE_VLSNR_SET1_BPSK_11_45: return &s2x_c9;
        case RATE_VLSNR_SET1_BPSK_1_3: return &s2x_c10;
        default: return nullptr;
        }
    }
}

ldpc_code* create_ldpc_table(fecframe_size_t framesize,
                             code_rate_t rate)
{
    const ldpc_code* table = select_table(framesize, rate);
    return table ? new ldpc_code(*table) : nullptr;
}

} // namespace cusat_dsp
