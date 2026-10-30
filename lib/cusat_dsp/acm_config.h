/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * ACM configuration: MODCOD definitions with spectral efficiency and SNR thresholds.
 */

#ifndef INCLUDED_CUSAT_DSP_ACM_CONFIG_H
#define INCLUDED_CUSAT_DSP_ACM_CONFIG_H

#include <cusat_dsp/modcod_types.h>
#include "fec/code_params.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace cusat_dsp {

// Deliberate headroom above the measured FER=1e-5 waterfall thresholds. The
// receiver's pilotSNR estimator already reports Es/N0, so no hidden estimator
// offset is applied here.
inline constexpr float ACM_DEFAULT_GUARD_DB = 2.5f;

// Additional headroom used only for the initial SNR-to-MODCOD jump.  The
// candidate still has to pass the normal padding-only OTA probation before it
// can carry application traffic.  Once it is validated, steady-state ACM uses
// the normal guard above and its learned per-mode OLLA corrections.
inline constexpr float ACM_DEFAULT_BOOTSTRAP_MARGIN_DB = 2.0f;

struct modcod_def_t {
    uint8_t modcod_idx;
    modulation_t constellation;
    code_rate_t code_rate;
    const char* name;
    float spectral_eff;  // bits/symbol
    float min_snr_db;    // Measured Es/N0 for post-BCH FER <= 1e-5
};

/**
 * @brief Get all supported DVB-S2 MODCODs, sorted by spectral efficiency (ascending).
 *
 * The min_snr_db values are implementation thresholds measured with normal
 * FECFRAMEs, the INT16 decoder, 25 maximum LDPC iterations, and AWGN.  Each is
 * the first 0.1 dB waterfall point whose observed post-BCH FER is <= 1e-5
 * (bench-coded-modulation).  These are empirical point
 * estimates, not additional link margin or confidence bounds.
 */
inline const std::vector<modcod_def_t>& get_supported_modcods()
{
    // Sorted by spectral efficiency (ascending)
    static const std::vector<modcod_def_t> table = {
        {  1, MOD_QPSK,   RATE_1_4,  "qpsk1/4",    0.49, -0.05f },
        { 33, MOD_QPSK,   RATE_13_45,"qpsk13/45",  0.57,  0.07f },
        {  2, MOD_QPSK,   RATE_1_3,  "qpsk1/3",    0.66,  0.56f },
        {  3, MOD_QPSK,   RATE_2_5,  "qpsk2/5",    0.79,  1.00f },
        { 43, MOD_QPSK,   RATE_9_20, "qpsk9/20",   0.89,  1.42f },
        {  4, MOD_QPSK,   RATE_1_2,  "qpsk1/2",    0.99,  1.80f },
        { 44, MOD_QPSK,   RATE_11_20,"qpsk11/20",  1.09,  2.25f },
        {  5, MOD_QPSK,   RATE_3_5,  "qpsk3/5",    1.19,  2.73f },
        {  6, MOD_QPSK,   RATE_2_3,  "qpsk2/3",    1.32,  3.50f },
        {  7, MOD_QPSK,   RATE_3_4,  "qpsk3/4",    1.49,  4.43f },
        {  8, MOD_QPSK,   RATE_4_5,  "qpsk4/5",    1.59,  5.08f },
        {  9, MOD_QPSK,   RATE_5_6,  "qpsk5/6",    1.65,  5.48f },
        { 10, MOD_QPSK,   RATE_8_9,  "qpsk8/9",    1.77,  6.50f },
        { 11, MOD_QPSK,   RATE_9_10, "qpsk9/10",   1.79,  6.72f },
        { 56, MOD_8APSK,  RATE_100_180,"8apsk5/9",  1.65,  4.93f },
        { 57, MOD_8APSK,  RATE_104_180,"8apsk26/45",1.71,  5.33f },
        { 12, MOD_8PSK,   RATE_3_5,  "8psk3/5",    1.78,  5.80f },
        { 45, MOD_8PSK,   RATE_23_36,"8psk23/36",  1.90,  7.22f },
        { 13, MOD_8PSK,   RATE_2_3,  "8psk2/3",    1.98,  6.92f },
        { 46, MOD_8PSK,   RATE_25_36,"8psk25/36",  2.06,  7.22f },
        { 47, MOD_8PSK,   RATE_13_18,"8psk13/18",  2.15,  7.69f },
        { 14, MOD_8PSK,   RATE_3_4,  "8psk3/4",    2.23,  8.11f },
        { 15, MOD_8PSK,   RATE_5_6,  "8psk5/6",    2.48,  9.55f },
        { 16, MOD_8PSK,   RATE_8_9,  "8psk8/9",    2.65, 10.79f },
        { 17, MOD_8PSK,   RATE_9_10, "8psk9/10",   2.68, 11.08f },
        // DVB-S2X "-L" 16APSK (8+8APSK constellation, EN 302 307-2 Tab 11c-e)
        { 58, MOD_16APSK, RATE_90_180, "16apsk1/2l",  1.97,  6.27f },
        { 59, MOD_16APSK, RATE_96_180, "16apsk8/15l", 2.10,  6.85f },
        { 60, MOD_16APSK, RATE_100_180,"16apsk5/9l",  2.19,  7.04f },
        { 61, MOD_16APSK, RATE_18_30,  "16apsk3/5l",  2.37,  7.71f },
        { 62, MOD_16APSK, RATE_20_30,  "16apsk2/3l",  2.64,  8.73f },
        // DVB-S2X 4+12APSK 16APSK + DVB-S2 base
        { 48, MOD_16APSK, RATE_26_45, "16apsk26/45", 2.28,  7.81f },
        { 49, MOD_16APSK, RATE_3_5,   "16apsk3/5",   2.37,  8.10f },
        { 50, MOD_16APSK, RATE_28_45, "16apsk28/45", 2.46,  8.30f },
        { 51, MOD_16APSK, RATE_23_36, "16apsk23/36", 2.52,  8.58f },
        { 18, MOD_16APSK, RATE_2_3,  "16apsk2/3",  2.64,  9.17f },
        { 52, MOD_16APSK, RATE_25_36, "16apsk25/36", 2.75,  9.47f },
        { 53, MOD_16APSK, RATE_13_18, "16apsk13/18", 2.86,  9.91f },
        { 19, MOD_16APSK, RATE_3_4,  "16apsk3/4",  2.97, 10.41f },
        { 54, MOD_16APSK, RATE_140_180,"16apsk7/9",  3.08, 10.85f },
        { 20, MOD_16APSK, RATE_4_5,  "16apsk4/5",  3.17, 11.13f },
        { 21, MOD_16APSK, RATE_5_6,  "16apsk5/6",  3.30, 11.71f },
        { 55, MOD_16APSK, RATE_154_180,"16apsk77/90",3.39, 12.19f },
        { 22, MOD_16APSK, RATE_8_9,  "16apsk8/9",  3.52, 12.99f },
        { 23, MOD_16APSK, RATE_9_10, "16apsk9/10", 3.57, 13.23f },
        { 63, MOD_32APSK, RATE_2_3,  "32apsk2/3l", 3.29, 11.30f }, // 4+12+16rbAPSK
        { 64, MOD_32APSK, RATE_128_180,"32apsk32/45",3.51,12.05f }, // 4+8+4+16APSK
        { 65, MOD_32APSK, RATE_132_180,"32apsk11/15",3.62,12.47f }, // 4+8+4+16APSK
        { 24, MOD_32APSK, RATE_3_4,  "32apsk3/4",  3.70, 12.93f },
        { 66, MOD_32APSK, RATE_140_180,"32apsk7/9", 3.84, 13.35f }, // 4+8+4+16APSK
        { 25, MOD_32APSK, RATE_4_5,  "32apsk4/5",  3.95, 13.84f },
        { 26, MOD_32APSK, RATE_5_6,  "32apsk5/6",  4.12, 14.48f },
        { 27, MOD_32APSK, RATE_8_9,  "32apsk8/9",  4.40, 15.79f },
        { 28, MOD_32APSK, RATE_9_10, "32apsk9/10", 4.47, 16.05f },
        { 29, MOD_64APSK, RATE_132_180, "64apsk11/15", 4.40, 15.00f },
        { 30, MOD_64APSK, RATE_7_9,     "64apsk7/9",   4.67, 15.60f },
        { 31, MOD_64APSK, RATE_4_5,     "64apsk4/5",   4.80, 16.10f },
        { 32, MOD_128APSK, RATE_135_180, "128apsk3/4", 5.25, 18.00f },
        { 34, MOD_128APSK, RATE_140_180, "128apsk7/9", 5.44, 18.89f },
        { 35, MOD_64APSK,  RATE_5_6,     "64apsk5/6",   4.98, 17.23f },
        { 36, MOD_64APSK,  RATE_128_180, "64apsk32/45", 4.27, 14.18f },
        { 37, MOD_256APSK, RATE_135_180, "256apsk3/4",  6.00, 23.34f },
        { 38, MOD_256APSK, RATE_128_180, "256apsk32/45", 5.69, 22.40f },
        { 39, MOD_256APSK, RATE_116_180, "256apsk29/45", 5.16, 20.20f },
        { 40, MOD_256APSK, RATE_124_180, "256apsk31/45", 5.51, 21.50f },
        { 41, MOD_256APSK, RATE_20_30,   "256apsk2/3",   5.06, 18.10f },
        { 42, MOD_256APSK, RATE_22_30,   "256apsk11/15", 5.50, 19.80f },
    };
    return table;
}

/**
 * @brief Find a MODCOD definition by its modcod_idx.
 * @return Pointer to the definition, or nullptr if not found.
 */
inline const modcod_def_t* find_modcod(uint8_t modcod_idx)
{
    for (const auto& m : get_supported_modcods()) {
        if (m.modcod_idx == modcod_idx)
            return &m;
    }
    return nullptr;
}

/**
 * @brief True iff this MODCOD may be transmitted with the given frame size.
 *
 * Guards MODCOD selection (esp. ACM) against signalling a short/medium frame
 * with a MODCOD that has no support for that frame size, which would otherwise
 * (a) throw in get_fec_params() for a missing (framesize,rate) entry, or
 * (b) emit a bogus PLS code. Two conditions:
 *   1. FEC parameters exist for (framesize, code_rate).
 *   2. For short/medium frames, only DVB-S2 base MODCODs (idx 1..28) are
 *      allowed — the DVB-S2X MODCODs (idx >= 29) are normal-frame only here
 *      (their short/medium PLS codes are not yet wired in pls_code_for), so
 *      they must never be used for a non-normal frame.
 */
inline bool modcod_supports_framesize(const modcod_def_t& mc, fecframe_size_t fs)
{
    if (!has_fec_params(fs, mc.code_rate))
        return false;
    if (fs != FECFRAME_NORMAL && mc.modcod_idx >= 29)
        return false;
    return true;
}

/**
 * @brief Find a MODCOD definition by its string name (e.g. "qpsk3/5").
 * @return Pointer to the definition, or nullptr if not found.
 */
inline const modcod_def_t* find_modcod_by_name(const char* name)
{
    for (const auto& m : get_supported_modcods()) {
        if (strcmp(m.name, name) == 0)
            return &m;
    }
    return nullptr;
}

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_ACM_CONFIG_H */
