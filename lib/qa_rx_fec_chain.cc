/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "cusat_dsp/rx_fec_chain.h"
#include <boost/test/unit_test.hpp>

namespace cusat_dsp {

BOOST_AUTO_TEST_CASE(test_rx_fec_acceptance_requires_ldpc_and_bch)
{
    BOOST_CHECK(rx_fec_chain::frame_passes_fec_checks(25, true));
    BOOST_CHECK(rx_fec_chain::frame_passes_fec_checks(0, true));

    BOOST_CHECK(!rx_fec_chain::frame_passes_fec_checks(-1, true));
    BOOST_CHECK(!rx_fec_chain::frame_passes_fec_checks(3, false));
    BOOST_CHECK(!rx_fec_chain::frame_passes_fec_checks(-1, false));
}

} // namespace cusat_dsp
