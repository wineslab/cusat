/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLS code construction (EN 302 307-1 clause 5.5.2.4, EN 302 307-2 clause
 * 5.5.2). Writing the 8-bit code as b0..b7 (b0 = MSB), the 32-bit inner
 * codeword sent at positions t = 0..31 is
 *
 *     y_t = (u . t) ^ b6 ^ (b0 & g_t),    u = b1 + 2 b2 + 4 b3 + 8 b4 + 16 b5,
 *
 * where u . t is the parity of u & t (the rows 0x55555555 ... 0x0000FFFF of
 * the RM(1,5) generator select the bits of t), b6 multiplies the all-ones
 * row and g is the extra row that DVB-S2X adds for b0. Each y_t is then sent
 * twice, the second copy inverted when b7 = 1.
 *
 * The pi/2-BPSK phase carries on from the SOF into the PLSC for b0 = 0; for
 * b0 = 1 the PLSC is advanced by a further pi/2 (EN 302 307-2 clause 5.5.2).
 *
 * Since y depends on u only through a Walsh function, the correlation of
 * the received soft bits with every codeword follows from four 32-point
 * Walsh-Hadamard transforms, one per (b0, b7).
 */

#include "framing/plheader.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace cusat_dsp {

namespace {

constexpr uint32_t S2X_ROW = 0x90AC2DDD;

constexpr bool s2x_row_bit(int t) { return (S2X_ROW >> (31 - t)) & 1; }

constexpr int parity(unsigned x)
{
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}

struct pls_fields_t {
    bool b0;
    unsigned u; // b1..b5
    bool b6;
    bool b7;
};

constexpr pls_fields_t split_pls(uint8_t pls)
{
    pls_fields_t f{};
    f.b0 = (pls >> 7) & 1;
    for (int i = 0; i < 5; i++)
        f.u |= ((pls >> (6 - i)) & 1u) << i;
    f.b6 = (pls >> 1) & 1;
    f.b7 = pls & 1;
    return f;
}

constexpr uint64_t make_codeword(uint8_t pls)
{
    const pls_fields_t f = split_pls(pls);
    uint64_t cw = 0;
    for (int t = 0; t < 32; t++) {
        const bool y = parity(f.u & t) ^ f.b6 ^ (f.b0 && s2x_row_bit(t));
        cw |= uint64_t(y) << (63 - 2 * t);
        cw |= uint64_t(y != f.b7) << (62 - 2 * t);
    }
    return cw;
}

constexpr std::array<uint64_t, pl::kNumPlsCodes> make_codewords()
{
    std::array<uint64_t, pl::kNumPlsCodes> table{};
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++)
        table[p] = make_codeword(p);
    return table;
}

constexpr std::array<uint64_t, pl::kNumPlsCodes> CODEWORDS = make_codewords();

const complexf PI2_ROT[2] = {
    { (float)M_SQRT1_2, (float)M_SQRT1_2 },  // e^{j pi/4}
    { -(float)M_SQRT1_2, (float)M_SQRT1_2 }, // e^{j 3pi/4}
};

void walsh32(float* w)
{
    for (int h = 1; h < 32; h <<= 1)
        for (int i = 0; i < 32; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                const float a = w[j];
                const float b = w[j + h];
                w[j] = a + b;
                w[j + h] = a - b;
            }
}

} // namespace

uint64_t pls_codeword(uint8_t pls) { return CODEWORDS[pls]; }

void pi2bpsk_map(uint64_t bits, int n_bits, complexf* out)
{
    assert(n_bits > 0 && n_bits <= 64);
    for (int k = 0; k < n_bits; k++) {
        const bool one = (bits >> (n_bits - 1 - k)) & 1;
        out[k] = one ? -PI2_ROT[k & 1] : PI2_ROT[k & 1];
    }
}

void plheader_symbols(uint8_t pls, complexf* out)
{
    pi2bpsk_map(SOF_BITS, pl::kSofLen, out);
    complexf* plsc = out + pl::kSofLen;
    pi2bpsk_map(pls_codeword(pls) ^ PLSC_SCRAMBLER, pl::kPlsCodeLen, plsc);
    if (pls & 0x80)
        for (int k = 0; k < pl::kPlsCodeLen; k++)
            plsc[k] = complexf(-plsc[k].imag(), plsc[k].real());
}

pls_decoder::pls_decoder(const std::vector<uint8_t>& candidates)
{
    if (candidates.empty())
        throw std::invalid_argument("pls_decoder: no candidate PLS code");
    for (uint8_t c : candidates)
        d_candidate[c] = true;
}

pls_decision_t pls_decoder::decode(const complexf* plsc) const
{
    /* Soft bits, +1 for a 0: undo the pi/2 rotation and the scrambling. The
     * extra pi/2 of the b0 = 1 codes moves their bits to the imaginary part. */
    float r[2][pl::kPlsCodeLen];
    for (int k = 0; k < pl::kPlsCodeLen; k++) {
        const complexf z = plsc[k] * std::conj(PI2_ROT[k & 1]);
        const float sign = ((PLSC_SCRAMBLER >> (63 - k)) & 1) ? -1.0f : 1.0f;
        r[0][k] = sign * z.real();
        r[1][k] = sign * z.imag();
    }

    /* spectrum[b0][b7][u] = correlation with the codeword of (b0, u, b6 = 0, b7). */
    float spectrum[2][2][32];
    for (int b7 = 0; b7 < 2; b7++) {
        for (int b0 = 0; b0 < 2; b0++) {
            const float* rb = r[b0];
            float* w = spectrum[b0][b7];
            for (int t = 0; t < 32; t++) {
                const float v = b7 ? rb[2 * t] - rb[2 * t + 1] : rb[2 * t] + rb[2 * t + 1];
                w[t] = (b0 && s2x_row_bit(t)) ? -v : v;
            }
            walsh32(w);
        }
    }

    pls_decision_t best{ 0, -INFINITY, INFINITY };
    float runner_up = -INFINITY;
    int n_candidates = 0;
    for (unsigned p = 0; p < pl::kNumPlsCodes; p++) {
        if (!d_candidate[p])
            continue;
        n_candidates++;
        const pls_fields_t f = split_pls(p);
        float corr = spectrum[f.b0][f.b7][f.u];
        if (f.b6)
            corr = -corr;
        if (corr > best.score) {
            runner_up = best.score;
            best.pls = p;
            best.score = corr;
        } else if (corr > runner_up) {
            runner_up = corr;
        }
    }
    best.margin = n_candidates > 1 ? best.score - runner_up : INFINITY;
    return best;
}

} // namespace cusat_dsp
