/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL scrambling sequence (EN 302 307-1 clause 5.5.4).
 */

#include "cusat_dsp/framing/pl_scrambler.h"
#include "cusat_dsp/framing/plframe.h"
#include <boost/test/unit_test.hpp>
#include <vector>

namespace cusat_dsp {

/* The clause 5.5.4 definition transcribed literally: both m-sequences over a
 * full period, then R_n(i) = 2 z_n((i + 131072) mod P) + z_n(i). */
static std::vector<int> reference_r(int n, int len)
{
    const int P = (1 << 18) - 1;
    std::vector<int> x(P + 18), y(P + 18);
    x[0] = 1;
    for (int i = 0; i < 18; i++)
        y[i] = 1;
    for (int i = 0; i + 18 < P + 18; i++) {
        x[i + 18] = x[i + 7] ^ x[i];
        y[i + 18] = y[i + 10] ^ y[i + 7] ^ y[i + 5] ^ y[i];
    }
    auto z = [&](int i) { return x[(i + n) % P] ^ y[i % P]; };
    std::vector<int> r(len);
    for (int i = 0; i < len; i++)
        r[i] = 2 * z((i + 131072) % P) + z(i);
    return r;
}

BOOST_AUTO_TEST_CASE(test_matches_definition)
{
    const complexf descr[4] = { { 1, 0 }, { 0, -1 }, { -1, 0 }, { 0, 1 } };
    std::vector<complexf> seq(pl::max_payload_symbols());
    for (int n : { 0, 1, 7, 131071, 131072, N_GOLD_CODES - 1 }) {
        pl_descrambling_sequence(n, seq.data(), pl::max_payload_symbols());
        const std::vector<int> r = reference_r(n, pl::max_payload_symbols());
        int mismatches = 0;
        for (int i = 0; i < pl::max_payload_symbols(); i++)
            mismatches += seq[i] != descr[r[i]];
        BOOST_CHECK_EQUAL(mismatches, 0);
    }
}

BOOST_AUTO_TEST_CASE(test_scramble_descramble_round_trip)
{
    const int n = 1000;
    std::vector<complexf> seq(n), payload(n), sym(n);
    pl_descrambling_sequence(42, seq.data(), n);
    for (int i = 0; i < n; i++) {
        payload[i] = std::polar(1.0f, 0.1f * i);
        sym[i] = payload[i] * std::conj(seq[i]);
    }
    pl_descramble(sym.data(), seq.data(), n);
    for (int i = 0; i < n; i++)
        BOOST_CHECK_SMALL(std::abs(sym[i] - payload[i]), 1e-6f);
}

BOOST_AUTO_TEST_CASE(test_invalid_code)
{
    complexf c;
    BOOST_CHECK_THROW(pl_descrambling_sequence(-1, &c, 1), std::out_of_range);
    BOOST_CHECK_THROW(pl_descrambling_sequence(N_GOLD_CODES, &c, 1), std::out_of_range);
}

} // namespace cusat_dsp
