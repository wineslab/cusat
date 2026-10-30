/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Code rates, FECFRAME sizes and modulations of ETSI EN 302 307-1 (DVB-S2)
 * and EN 302 307-2 (DVB-S2X). The numeric values carry no meaning outside
 * this library; never persist or transmit them.
 */

#ifndef INCLUDED_CUSAT_DSP_MODCOD_TYPES_H
#define INCLUDED_CUSAT_DSP_MODCOD_TYPES_H

namespace cusat_dsp {

enum code_rate_t {
    // DVB-S2 code rates (EN 302 307-1 clause 5.3, Table 5a/5b), ascending.
    RATE_1_4,
    RATE_1_3,
    RATE_2_5,
    RATE_1_2,
    RATE_3_5,
    RATE_2_3,
    RATE_3_4,
    RATE_4_5,
    RATE_5_6,
    RATE_8_9,
    RATE_9_10,

    // DVB-S2X code rates (EN 302 307-2 clause 5.3, normal and short
    // FECFRAME), ascending by value; equal values ordered by denominator.
    RATE_11_45,
    RATE_4_15,
    RATE_13_45,
    RATE_14_45,
    RATE_9_20,
    RATE_7_15,
    RATE_90_180,
    RATE_8_15,
    RATE_96_180,
    RATE_11_20,
    RATE_100_180,
    RATE_26_45,
    RATE_104_180,
    RATE_18_30,
    RATE_28_45,
    RATE_23_36,
    RATE_116_180,
    RATE_20_30,
    RATE_124_180,
    RATE_25_36,
    RATE_32_45,
    RATE_128_180,
    RATE_13_18,
    RATE_22_30,
    RATE_132_180,
    RATE_135_180,
    RATE_7_9,
    RATE_140_180,
    RATE_154_180,

    // DVB-S2X VL-SNR codes, named after the canonical MODCOD names of
    // EN 302 307-2 Table 18a (set 1 / set 2). Table keys only: cuSAT does
    // not transmit or receive VL-SNR PLFRAMEs.
    RATE_VLSNR_SET1_QPSK_2_9,
    RATE_VLSNR_SET1_BPSK_1_5,
    RATE_VLSNR_SET1_BPSK_11_45,
    RATE_VLSNR_SET1_BPSK_1_3,
    RATE_VLSNR_SET1_BPSKS_1_5,
    RATE_VLSNR_SET1_BPSKS_11_45,
    RATE_VLSNR_SET2_BPSK_1_5,
    RATE_VLSNR_SET2_BPSK_4_15,
    RATE_VLSNR_SET2_BPSK_1_3,

    RATE_COUNT,                 // number of valid values above
    RATE_UNKNOWN = RATE_COUNT,  // sentinel
};

enum fecframe_size_t {
    FECFRAME_NORMAL, // 64800 bits
    FECFRAME_SHORT,  // 16200 bits
    FECFRAME_MEDIUM, // 32400 bits (VL-SNR)
};

enum modulation_t {
    MOD_QPSK,
    MOD_8PSK,
    MOD_8APSK,
    MOD_16APSK,
    MOD_32APSK,
    MOD_64APSK,
    MOD_128APSK,
    MOD_256APSK,
};

/* Coded bits of one FECFRAME. */
constexpr int fecframe_bits(fecframe_size_t size)
{
    return size == FECFRAME_NORMAL ? 64800 : size == FECFRAME_SHORT ? 16200 : 32400;
}

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_MODCOD_TYPES_H */
