/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "snr_ewma.h"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <limits>

using cusat_dsp::snr_ewma;

BOOST_AUTO_TEST_CASE(initial_sample_is_not_delayed)
{
    snr_ewma tracker(0.1f);
    BOOST_TEST(tracker.update(20.0f));
    BOOST_TEST(tracker.initialized());
    BOOST_TEST(tracker.mean() == 20.0f);
    BOOST_TEST(tracker.stddev() == 0.0f);
}

BOOST_AUTO_TEST_CASE(outlier_is_visible_but_attenuated_for_control)
{
    snr_ewma tracker(0.1f);
    tracker.update(20.0f);
    tracker.update(0.0f);

    // The raw 20 dB excursion contributes to both moments, while the ACM
    // estimate moves by only alpha times the excursion.
    BOOST_TEST(tracker.mean() == 18.0f, boost::test_tools::tolerance(1e-5f));
    BOOST_TEST(tracker.stddev() == 6.0f,
               boost::test_tools::tolerance(1e-5f));
}

BOOST_AUTO_TEST_CASE(sustained_fade_is_followed)
{
    snr_ewma tracker(0.25f);
    tracker.update(20.0f);
    for (int i = 0; i < 12; ++i)
        tracker.update(5.0f);

    BOOST_TEST(tracker.mean() < 5.5f);
    BOOST_TEST(tracker.mean() > 5.0f);
}

BOOST_AUTO_TEST_CASE(non_finite_sample_is_ignored)
{
    snr_ewma tracker(0.1f);
    tracker.update(12.0f);
    BOOST_TEST(!tracker.update(std::numeric_limits<float>::quiet_NaN()));
    BOOST_TEST(tracker.mean() == 12.0f);
}

BOOST_AUTO_TEST_CASE(reset_discards_old_channel_state)
{
    snr_ewma tracker(0.1f);
    tracker.update(20.0f);
    tracker.update(0.0f);
    tracker.reset();
    BOOST_TEST(!tracker.initialized());
    tracker.update(7.0f);
    BOOST_TEST(tracker.mean() == 7.0f);
    BOOST_TEST(tracker.stddev() == 0.0f);
}
