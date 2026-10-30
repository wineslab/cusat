/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLHEADER: PLS codewords, pi/2-BPSK mapping and soft PLS decoding
 * (EN 302 307-1 clause 5.5.2, EN 302 307-2 clause 5.5.2).
 */

#include "cusat_dsp/framing/plheader.h"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <random>

namespace cusat_dsp {

/* The code written out as in the standard: generator rows for b0..b6
 * (b0 = DVB-S2X row, then RM(1,5) rows of EN 302 307-1 figure 13b), each
 * inner bit repeated, the repetition inverted when b7 = 1. */
static uint64_t codeword_from_rows(uint8_t pls)
{
    static const uint32_t rows[7] = { 0x90AC2DDD, 0x55555555, 0x33333333, 0x0F0F0F0F,
                                      0x00FF00FF, 0x0000FFFF, 0xFFFFFFFF };
    uint32_t inner = 0;
    for (int b = 0; b < 7; b++)
        if ((pls >> (7 - b)) & 1)
            inner ^= rows[b];
    uint64_t cw = 0;
    for (int t = 0; t < 32; t++) {
        const uint64_t y = (inner >> (31 - t)) & 1;
        cw = (cw << 2) | (y << 1) | (y ^ (pls & 1));
    }
    return cw;
}

BOOST_AUTO_TEST_CASE(test_codewords_match_generator_rows)
{
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++)
        BOOST_CHECK_EQUAL(pls_codeword(p), codeword_from_rows(p));
}

BOOST_AUTO_TEST_CASE(test_minimum_distance)
{
    // DVB-S2 codes (b0 = 0): distance 32; with the S2X extension: 24.
    int d_s2 = 64, d_s2x = 64;
    for (unsigned a = 0; a < pl::kNumPlsCodes; a++)
        for (unsigned b = a + 1; b < pl::kNumPlsCodes; b++) {
            const int d = __builtin_popcountll(pls_codeword(a) ^ pls_codeword(b));
            d_s2x = std::min(d_s2x, d);
            if (a < 128 && b < 128)
                d_s2 = std::min(d_s2, d);
        }
    BOOST_CHECK_EQUAL(d_s2, 32);
    BOOST_CHECK_EQUAL(d_s2x, 24);
}

BOOST_AUTO_TEST_CASE(test_sof_symbols)
{
    // SOF = 18D2E82 hex: first bits 0, 1, 1, 0
    complexf h[pl::kHeaderLen];
    plheader_symbols(0, h);
    const float s = (float)M_SQRT1_2;
    BOOST_CHECK_EQUAL(h[0], complexf(s, s));   // 0 at an even position: e^{j pi/4}
    BOOST_CHECK_EQUAL(h[1], complexf(s, -s));  // 1 at an odd position: -e^{j 3pi/4}
    BOOST_CHECK_EQUAL(h[2], complexf(-s, -s)); // 1 at an even position
    BOOST_CHECK_EQUAL(h[3], complexf(-s, s));  // 0 at an odd position
    for (const complexf& x : h)
        BOOST_CHECK_CLOSE(std::norm(x), 1.0f, 1e-4);
}

BOOST_AUTO_TEST_CASE(test_plsc_is_scrambled)
{
    // PLS 0 has the all-zero codeword: its PLSC carries the scrambler itself.
    complexf h[pl::kHeaderLen], ref[pl::kPlsCodeLen];
    plheader_symbols(0, h);
    pi2bpsk_map(PLSC_SCRAMBLER, pl::kPlsCodeLen, ref);
    for (int k = 0; k < pl::kPlsCodeLen; k++)
        BOOST_CHECK_EQUAL(h[pl::kSofLen + k], ref[k]);
}

BOOST_AUTO_TEST_CASE(test_modulation_rule)
{
    // EN 302 307-2 clause 5.5.2 with i = 1..45 and bits y_1..y_90: symbols
    // 1..26 as in DVB-S2; from symbol 27 on, b0 = 1 swaps the odd/even rule
    // (a pi/2 phase jump after the SOF).
    const float s = (float)M_SQRT1_2;
    complexf h[pl::kHeaderLen];
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++) {
        plheader_symbols(p, h);
        const uint64_t plsc = pls_codeword(p) ^ PLSC_SCRAMBLER;
        for (int n = 1; n <= pl::kHeaderLen; n++) {
            const int y = n <= pl::kSofLen ? (SOF_BITS >> (pl::kSofLen - n)) & 1
                                        : (plsc >> (pl::kHeaderLen - n)) & 1;
            const float a = s * (1 - 2 * y);
            const bool jump = n > pl::kSofLen && (p & 0x80);
            complexf ref;
            if (n % 2 == 1)
                ref = jump ? complexf(-a, a) : complexf(a, a);
            else
                ref = jump ? complexf(-a, -a) : complexf(-a, a);
            BOOST_CHECK_EQUAL(h[n - 1], ref);
        }
    }
}

static std::vector<uint8_t> all_codes()
{
    std::vector<uint8_t> v;
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++)
        v.push_back(p);
    return v;
}

BOOST_AUTO_TEST_CASE(test_decode_noiseless)
{
    const pls_decoder dec(all_codes());
    complexf h[pl::kHeaderLen];
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++) {
        plheader_symbols(p, h);
        const pls_decision_t d = dec.decode(h + pl::kSofLen);
        BOOST_CHECK_EQUAL(d.pls, p);
        BOOST_CHECK_CLOSE(d.score, 64.0f, 1e-3);
        BOOST_CHECK_GE(d.margin, 2 * 24 - 1e-3); // twice the minimum distance
    }
}

BOOST_AUTO_TEST_CASE(test_decode_awgn)
{
    const pls_decoder dec(all_codes());
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, std::sqrt(0.5f / 4.0f)); // Es/N0 = 6 dB
    complexf h[pl::kHeaderLen];
    int errors = 0;
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++)
        for (int trial = 0; trial < 20; trial++) {
            plheader_symbols(p, h);
            for (complexf& x : h)
                x += complexf(noise(rng), noise(rng));
            errors += dec.decode(h + pl::kSofLen).pls != p;
        }
    BOOST_CHECK_EQUAL(errors, 0);
}

BOOST_AUTO_TEST_CASE(test_decode_restricted_candidates)
{
    const pls_decoder dec({ 13, 187 });
    complexf h[pl::kHeaderLen];
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++) {
        plheader_symbols(p, h);
        const uint8_t got = dec.decode(h + pl::kSofLen).pls;
        BOOST_CHECK(got == 13 || got == 187);
    }

    const pls_decoder single({ 42 });
    plheader_symbols(7, h);
    const pls_decision_t d = single.decode(h + pl::kSofLen);
    BOOST_CHECK_EQUAL(d.pls, 42);
    BOOST_CHECK(std::isinf(d.margin));

    BOOST_CHECK_THROW(pls_decoder(std::vector<uint8_t>{}), std::invalid_argument);
}

} // namespace cusat_dsp
