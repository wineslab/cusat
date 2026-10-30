/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bit interleaver (EN 302 307-1 clause 5.3.3, EN 302 307-2 Table 9a).
 */

#include "cusat_dsp/acm_config.h"
#include "cusat_dsp/modulation/bit_interleaver.h"
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include <set>

namespace cusat_dsp {

static std::vector<int> order(const bit_interleaver_t& il)
{
    std::vector<int> o;
    for (int c = 0; c < il.n_cols; c++)
        o.push_back(il.col_offset[c] / il.rows);
    return o;
}

BOOST_AUTO_TEST_CASE(test_qpsk_not_interleaved)
{
    const bit_interleaver_t il = make_bit_interleaver(MOD_QPSK, RATE_1_2, FECFRAME_NORMAL);
    BOOST_CHECK_EQUAL(il.n_cols, 0);
}

BOOST_AUTO_TEST_CASE(test_dvbs2_patterns)
{
    // 8PSK: columns read in order, except rate 3/5 (MSB taken from column 2)
    const bit_interleaver_t a = make_bit_interleaver(MOD_8PSK, RATE_2_3, FECFRAME_NORMAL);
    BOOST_CHECK_EQUAL(a.rows, 64800u / 3);
    BOOST_CHECK((order(a) == std::vector<int>{ 0, 1, 2 }));
    BOOST_CHECK((order(make_bit_interleaver(MOD_8PSK, RATE_3_5, FECFRAME_NORMAL)) == std::vector<int>{ 2, 1, 0 }));
    BOOST_CHECK_EQUAL(make_bit_interleaver(MOD_32APSK, RATE_3_4, FECFRAME_SHORT).rows, 16200u / 5);
}

BOOST_AUTO_TEST_CASE(test_dvbs2x_patterns)
{
    // a few Table 9a entries
    BOOST_CHECK((order(make_bit_interleaver(MOD_32APSK, RATE_2_3, FECFRAME_NORMAL)) == std::vector<int>{ 2, 1, 4, 3, 0 }));
    BOOST_CHECK((order(make_bit_interleaver(MOD_64APSK, RATE_4_5, FECFRAME_NORMAL)) == std::vector<int>{ 1, 2, 4, 0, 5, 3 }));
    BOOST_CHECK((order(make_bit_interleaver(MOD_256APSK, RATE_135_180, FECFRAME_NORMAL)) ==
                 std::vector<int>{ 5, 0, 7, 4, 3, 6, 1, 2 }));

    // 128APSK: 64800 + 6 stuffing bits over 7 columns
    const bit_interleaver_t il = make_bit_interleaver(MOD_128APSK, RATE_135_180, FECFRAME_NORMAL);
    BOOST_CHECK_EQUAL(il.rows, 9258u);
    BOOST_CHECK((order(il) == std::vector<int>{ 4, 2, 5, 0, 3, 1, 6 }));
}

BOOST_AUTO_TEST_CASE(test_every_modcod_is_a_permutation)
{
    for (const modcod_def_t& m : get_supported_modcods()) {
        const bit_interleaver_t il = make_bit_interleaver(m.constellation, m.code_rate, FECFRAME_NORMAL);
        if (m.constellation == MOD_QPSK)
            continue;
        const std::vector<int> o = order(il);
        BOOST_TEST_CONTEXT(m.name)
        {
            BOOST_CHECK_EQUAL((int)std::set<int>(o.begin(), o.end()).size(), il.n_cols);
            BOOST_CHECK_EQUAL(*std::max_element(o.begin(), o.end()), il.n_cols - 1);
        }
    }
}

} // namespace cusat_dsp
