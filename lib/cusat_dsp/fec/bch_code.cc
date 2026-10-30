/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "fec/bch_code.h"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace cusat_dsp {

namespace {

// g1..g12, bit i = coefficient of x^i
constexpr uint32_t MIN_POLY_NORMAL[12] = { 0x1002D, 0x10173, 0x10FBD, 0x15A55, 0x11F2F, 0x1F7B5,
                                           0x1AF65, 0x17367, 0x10EA1, 0x175A7, 0x13A2D, 0x11AE3 };
constexpr uint32_t MIN_POLY_SHORT[12] = { 0x402B, 0x4941, 0x4647, 0x5591, 0x6B55, 0x6389,
                                          0x6CE5, 0x4F21, 0x460F, 0x5A49, 0x5811, 0x65EF };
constexpr uint32_t MIN_POLY_MEDIUM[12] = { 0x802D, 0x8C93, 0xB555, 0x8D6D, 0x94D7, 0xB0D1,
                                           0xD8B5, 0xCB55, 0xBAB7, 0xBC9F, 0x8A17, 0xED15 };

bool coeff(const poly256_t& p, int i) { return (p[i >> 6] >> (i & 63)) & 1; }

int degree(const poly256_t& p)
{
    for (int i = 255; i >= 0; i--)
        if (coeff(p, i))
            return i;
    return -1;
}

/* p(x) x^s, truncated to degree 255 */
poly256_t shifted(const poly256_t& p, int s)
{
    poly256_t r{};
    const int words = s >> 6, bits = s & 63;
    for (int w = 3; w >= words; w--) {
        uint64_t v = p[w - words] << bits;
        if (bits && w - words - 1 >= 0)
            v |= p[w - words - 1] >> (64 - bits);
        r[w] = v;
    }
    return r;
}

/* Keep only the coefficients of x^0 .. x^(n-1). */
void truncate(poly256_t& p, int n)
{
    for (int w = 0; w < 4; w++) {
        const int lo = 64 * w;
        if (n <= lo)
            p[w] = 0;
        else if (n < lo + 64)
            p[w] &= (1ull << (n - lo)) - 1;
    }
}

void add(poly256_t& a, const poly256_t& b)
{
    for (int w = 0; w < 4; w++)
        a[w] ^= b[w];
}

poly256_t times(const poly256_t& a, uint32_t b)
{
    poly256_t r{};
    for (int i = 0; i < 32; i++)
        if ((b >> i) & 1)
            add(r, shifted(a, i));
    return r;
}

poly256_t modulo(poly256_t a, const poly256_t& g, int deg_g)
{
    for (int i = degree(a); i >= deg_g; i--)
        if (coeff(a, i))
            add(a, shifted(g, i - deg_g));
    return a;
}

poly256_t byte_poly(int b, int shift)
{
    poly256_t p{};
    p[0] = (uint64_t)b;
    return shifted(p, shift);
}

/* Coefficients x^lo .. x^(lo + 7) as a byte, x^(lo + 7) in the MSB. */
uint8_t get_byte(const poly256_t& p, int lo)
{
    const int w = lo >> 6, b = lo & 63;
    uint64_t v = p[w] >> b;
    if (b > 56 && w < 3)
        v |= p[w + 1] << (64 - b);
    return (uint8_t)v;
}

} // namespace

bch_code_t make_bch_code(fecframe_size_t framesize, int n_bits, int t)
{
    if (t < 1 || t > 12)
        throw std::invalid_argument("make_bch_code: t must be 1..12");
    const uint32_t* min_poly = framesize == FECFRAME_NORMAL   ? MIN_POLY_NORMAL
                               : framesize == FECFRAME_SHORT ? MIN_POLY_SHORT
                                                             : MIN_POLY_MEDIUM;
    bch_code_t code;
    code.generator[0] = 1;
    for (int i = 0; i < t; i++)
        code.generator = times(code.generator, min_poly[i]);
    code.parity_bits = degree(code.generator);
    code.n_bits = n_bits;
    code.k_bits = n_bits - code.parity_bits;
    if (code.k_bits <= 0)
        throw std::invalid_argument("make_bch_code: codeword shorter than the generator");

    if (code.n_bits % 8 == 0 && code.k_bits % 8 == 0) {
        for (int b = 0; b < 256; b++) {
            code.encoder_lut[b] = modulo(byte_poly(b, code.parity_bits), code.generator, code.parity_bits);
        }
    }
    return code;
}

void bch_encode(const bch_code_t& code, const uint8_t* msg, uint8_t* codeword)
{
    assert(code.k_bits % 8 == 0 && code.parity_bits % 8 == 0);
    const int p = code.parity_bits;
    const int k_bytes = code.k_bits / 8;

    /* Shift register holding the running remainder of msg(x) x^p mod g(x),
     * fed one message byte at a time. */
    poly256_t reg{};
    for (int i = 0; i < k_bytes; i++) {
        const uint8_t top = get_byte(reg, p - 8) ^ msg[i];
        reg = shifted(reg, 8);
        truncate(reg, p);
        add(reg, code.encoder_lut[top]);
    }

    memcpy(codeword, msg, k_bytes);
    for (int j = 0; j < p / 8; j++)
        codeword[k_bytes + j] = get_byte(reg, p - 8 * (j + 1));
}

} // namespace cusat_dsp
