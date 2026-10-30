/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "cuda/ldpc_message_update.h"
#include <boost/test/unit_test.hpp>

namespace cusat_dsp {
namespace cuda {

BOOST_AUTO_TEST_CASE(test_ldpc_bnl_update_uses_stored_clamped_message)
{
    auto high_same = ldpc_make_bnl_update(300, 127);
    BOOST_CHECK_EQUAL((int)high_same.message, 127);
    BOOST_CHECK_EQUAL(high_same.delta, 0);

    auto high_cross = ldpc_make_bnl_update(300, -127);
    BOOST_CHECK_EQUAL((int)high_cross.message, 127);
    BOOST_CHECK_EQUAL(high_cross.delta, 254);

    auto neg_cross = ldpc_make_bnl_update(-300, 127);
    BOOST_CHECK_EQUAL((int)neg_cross.message, -127);
    BOOST_CHECK_EQUAL(neg_cross.delta, -254);

    auto in_range = ldpc_make_bnl_update(42, -5);
    BOOST_CHECK_EQUAL((int)in_range.message, 42);
    BOOST_CHECK_EQUAL(in_range.delta, 47);
}

} // namespace cuda
} // namespace cusat_dsp
