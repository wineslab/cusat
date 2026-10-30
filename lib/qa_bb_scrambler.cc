/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BB scrambler (EN 302 307-1 clause 5.2.2).
 */

#include "cusat_dsp/bb_scrambler.h"
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <random>
#include <vector>

using namespace cusat_dsp;

namespace {

/* Bit-serial PRBS straight from figure 5: stages 1..15, initial content
 * 100101010000000, output = stage 14 XOR stage 15 fed back into stage 1. */
std::vector<int> prbs_bits(size_t n)
{
    int st[16];
    const char* init = "100101010000000";
    for (int i = 1; i <= 15; i++)
        st[i] = init[i - 1] - '0';
    std::vector<int> out(n);
    for (size_t k = 0; k < n; k++) {
        const int b = st[14] ^ st[15];
        for (int i = 15; i > 1; i--)
            st[i] = st[i - 1];
        st[1] = b;
        out[k] = b;
    }
    return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_prbs_matches_register_definition)
{
    const size_t n = BB_SCRAMBLER_MAX_BYTES;
    const std::vector<int> bits = prbs_bits(8 * n);
    // figure 5 shows the first output bits 0 0 0 0 0 0 1 1
    const int first[8] = { 0, 0, 0, 0, 0, 0, 1, 1 };
    for (int i = 0; i < 8; i++)
        BOOST_CHECK_EQUAL(bits[i], first[i]);
    std::vector<uint8_t> zero(n, 0), out(n);
    bb_scramble(zero.data(), out.data(), n); // zero input -> the PRBS itself
    BOOST_CHECK_EQUAL(out[0], 0x03);
    for (size_t i = 0; i < n; i++) {
        uint8_t ref = 0;
        for (int b = 0; b < 8; b++)
            ref = (uint8_t)((ref << 1) | bits[8 * i + b]); // MSB first
        BOOST_REQUIRE_EQUAL(out[i], ref);
    }
    // the sequence restarts with every BBFRAME (period 2^15 - 1 bits)
    bb_scramble(zero.data(), out.data(), 100);
    BOOST_CHECK_EQUAL(out[0], 0x03);
}

BOOST_AUTO_TEST_CASE(test_self_inverse_and_inplace)
{
    std::mt19937 rng(7);
    for (size_t n : { (size_t)0, (size_t)1, (size_t)7, (size_t)8, (size_t)1000, (size_t)2001,
                      BB_SCRAMBLER_MAX_BYTES }) {
        std::vector<uint8_t> a(n), b(n), c(n), d(n);
        for (auto& x : a)
            x = (uint8_t)rng();
        bb_scramble(a.data(), b.data(), n);
        bb_scramble(b.data(), c.data(), n);
        BOOST_CHECK(a == c); // its own inverse
        d = a;
        bb_scramble_inplace(d.data(), n);
        BOOST_CHECK(d == b); // in-place equals out-of-place
        if (n > 16)
            BOOST_CHECK(a != b);
    }
    std::vector<uint8_t> big(BB_SCRAMBLER_MAX_BYTES + 1);
    BOOST_CHECK_THROW(bb_scramble_inplace(big.data(), big.size()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_throughput)
{
    const size_t n = BB_SCRAMBLER_MAX_BYTES;
    std::vector<uint8_t> a(n, 0x5a), b(n);
    const int reps = 20000; // one second of frames at 20 000 frames/s
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        bb_scramble(a.data(), b.data(), n);
        a[r % n] ^= b[(r * 7) % n]; // keep the work observable
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gbs = (double)n * reps / s / 1e9;
    BOOST_TEST_MESSAGE("bb_scramble: " << gbs << " GB/s (" << s * 1e6 / reps << " us per 7274-byte frame)");
    BOOST_CHECK_GE(gbs, 2.0);
}
