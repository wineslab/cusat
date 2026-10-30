/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BCH outer code (EN 302 307-1 clause 5.3.1, Tables 5a/5b/6a/6b).
 */

#include "cusat_dsp/fec/bch_code.h"
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

namespace cusat_dsp {

/* Independent bit-serial reference: remainder of a big-endian byte string,
 * read as a polynomial (first bit = highest power), modulo g. */
static std::vector<int> remainder_bits(const std::vector<uint8_t>& bytes, const poly256_t& g, int deg)
{
    std::vector<int> r(deg, 0); // r[i] = coefficient of x^i
    for (uint8_t byte : bytes)
        for (int b = 7; b >= 0; b--) {
            const int top = r[deg - 1];
            for (int i = deg - 1; i > 0; i--)
                r[i] = r[i - 1];
            r[0] = (byte >> b) & 1;
            if (top)
                for (int i = 0; i < deg; i++)
                    r[i] ^= (g[i >> 6] >> (i & 63)) & 1;
        }
    return r;
}

static bool coeff(const poly256_t& p, int i) { return (p[i >> 6] >> (i & 63)) & 1; }

BOOST_AUTO_TEST_CASE(test_generator_degrees)
{
    // parity length m * t: GF(2^16) normal, GF(2^14) short
    BOOST_CHECK_EQUAL(make_bch_code(FECFRAME_NORMAL, 16200, 12).parity_bits, 192);
    BOOST_CHECK_EQUAL(make_bch_code(FECFRAME_NORMAL, 48600, 10).parity_bits, 160);
    BOOST_CHECK_EQUAL(make_bch_code(FECFRAME_NORMAL, 57600, 8).parity_bits, 128);
    BOOST_CHECK_EQUAL(make_bch_code(FECFRAME_SHORT, 3240, 12).parity_bits, 168);
    BOOST_CHECK_EQUAL(make_bch_code(FECFRAME_NORMAL, 16200, 12).k_bits, 16008);
}

BOOST_AUTO_TEST_CASE(test_g1_is_the_field_polynomial)
{
    // g1(x) = 1 + x^2 + x^3 + x^5 + x^16 (Table 6a), 1 + x + x^3 + x^5 + x^14 (Table 6b)
    const bch_code_t n = make_bch_code(FECFRAME_NORMAL, 64800, 1);
    for (int i = 0; i < 256; i++)
        BOOST_CHECK_EQUAL(coeff(n.generator, i), i == 0 || i == 2 || i == 3 || i == 5 || i == 16);
    const bch_code_t s = make_bch_code(FECFRAME_SHORT, 16200, 1);
    for (int i = 0; i < 256; i++)
        BOOST_CHECK_EQUAL(coeff(s.generator, i), i == 0 || i == 1 || i == 3 || i == 5 || i == 14);
}

BOOST_AUTO_TEST_CASE(test_codewords_are_multiples_of_g)
{
    std::mt19937 rng(1);
    const std::tuple<fecframe_size_t, int, int> codes[] = {
        { FECFRAME_NORMAL, 16200, 12 }, { FECFRAME_NORMAL, 58320, 8 }, { FECFRAME_SHORT, 14400, 12 }
    };
    for (const auto& [fs, n, t] : codes) {
        const bch_code_t code = make_bch_code(fs, n, t);
        std::vector<uint8_t> msg(code.k_bits / 8), cw(n / 8);
        for (int trial = 0; trial < 3; trial++) {
            for (auto& x : msg)
                x = rng();
            bch_encode(code, msg.data(), cw.data());
            BOOST_CHECK(std::equal(msg.begin(), msg.end(), cw.begin())); // systematic
            const std::vector<int> r = remainder_bits(cw, code.generator, code.parity_bits);
            BOOST_CHECK(std::all_of(r.begin(), r.end(), [](int v) { return v == 0; }));
        }
    }
}

BOOST_AUTO_TEST_CASE(test_linearity)
{
    const bch_code_t code = make_bch_code(FECFRAME_SHORT, 7200, 12);
    std::mt19937 rng(2);
    std::vector<uint8_t> a(code.k_bits / 8), b(a.size()), ab(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = rng();
        b[i] = rng();
        ab[i] = a[i] ^ b[i];
    }
    std::vector<uint8_t> ca(7200 / 8), cb(ca.size()), cab(ca.size());
    bch_encode(code, a.data(), ca.data());
    bch_encode(code, b.data(), cb.data());
    bch_encode(code, ab.data(), cab.data());
    for (size_t i = 0; i < ca.size(); i++)
        BOOST_CHECK_EQUAL(cab[i], ca[i] ^ cb[i]);
}

BOOST_AUTO_TEST_CASE(test_byte_remainder_lut)
{
    // encoder_lut[b] = b(x) x^parity_bits mod g(x)
    const bch_code_t code = make_bch_code(FECFRAME_NORMAL, 64800, 12);
    for (int b : { 1, 2, 0x80, 0xA5, 0xFF }) {
        std::vector<uint8_t> bytes(code.parity_bits / 8 + 1, 0);
        bytes[0] = b;
        const std::vector<int> r = remainder_bits(bytes, code.generator, code.parity_bits);
        for (int i = 0; i < code.parity_bits; i++)
            BOOST_CHECK_EQUAL(coeff(code.encoder_lut[b], i), r[i]);
        for (int i = code.parity_bits; i < 256; i++)
            BOOST_CHECK(!coeff(code.encoder_lut[b], i));
    }
}

BOOST_AUTO_TEST_CASE(test_invalid_parameters)
{
    BOOST_CHECK_THROW(make_bch_code(FECFRAME_NORMAL, 64800, 0), std::invalid_argument);
    BOOST_CHECK_THROW(make_bch_code(FECFRAME_NORMAL, 100, 12), std::invalid_argument);
}

} // namespace cusat_dsp
