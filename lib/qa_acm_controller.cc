/* SPDX-License-Identifier: Apache-2.0 */

#include "acm_controller.h"
#include <boost/test/unit_test.hpp>

using cusat_dsp::acm_controller;
using cusat_dsp::modcod_def_t;
using namespace cusat_dsp;

static std::vector<modcod_def_t> test_ladder()
{
    return {
        {1, MOD_QPSK, RATE_1_4, "low", 0.5f, 0.0f},
        {2, MOD_QPSK, RATE_1_3, "middle", 1.0f, 3.0f},
        {3, MOD_QPSK, RATE_2_5, "high", 1.5f, 6.0f},
    };
}

static void drive_to_high(acm_controller& controller, float snr_db)
{
    for (int i = 0; i < 3; ++i)
        controller.update_snr(snr_db);
    if (controller.current_modcod_idx() == 2) {
        // A conservative bootstrap can land on the middle rung. Validate it,
        // then exercise the unchanged steady-state confirmation path.
        controller.update_feedback(snr_db, 0.0f, 64, 0, 2);
        BOOST_REQUIRE_EQUAL(controller.validated_modcod_idx(), 2);
        for (int i = 0; i < 5; ++i)
            controller.update_snr(snr_db);
    }
    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(smooth_descent_is_predicted_before_static_threshold)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 10.0f);

    controller.update_snr(9.7f);
    controller.update_snr(9.4f);
    controller.update_snr(9.1f);
    controller.update_snr(8.8f);

    // A static 2.5 dB guard still permits the high mode at 8.8 dB. The
    // falling-SNR lookahead should already have selected the middle mode.
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 2);
}

BOOST_AUTO_TEST_CASE(bootstrap_jumps_directly_after_three_snr_reports)
{
    acm_controller controller(test_ladder());
    for (int i = 0; i < 2; ++i)
        controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 3);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 1);
    BOOST_CHECK(!controller.bootstrap_complete());

    // The direct candidate remains padding-only until the unchanged clean-OTA
    // probation completes.
    controller.update_feedback(14.0f, 0.0f, 64, 0, 3);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 3);
    BOOST_CHECK(controller.bootstrap_complete());
}

BOOST_AUTO_TEST_CASE(unlocked_sentinel_does_not_consume_bootstrap_samples)
{
    acm_controller controller(test_ladder());

    for (int i = 0; i < 8; ++i)
        controller.update_snr(-99.0f);
    controller.update_snr(14.0f);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(unlocked_sentinel_forces_floor_and_fresh_bootstrap)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);

    controller.update_snr(-99.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    BOOST_CHECK(!controller.bootstrap_complete());
    controller.update_snr(14.0f);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(bootstrap_uses_additional_startup_margin)
{
    acm_controller conservative(test_ladder());
    acm_controller no_extra_margin(test_ladder(), 2.5f, 0.0f);

    // At 10 dB, the normal 2.5 dB guard admits the high mode.  The default
    // extra 2 dB bootstrap margin deliberately starts one rung lower.
    for (int i = 0; i < 3; ++i) {
        conservative.update_snr(10.0f);
        no_extra_margin.update_snr(10.0f);
    }
    BOOST_CHECK_EQUAL(conservative.current_modcod_idx(), 2);
    BOOST_CHECK_EQUAL(no_extra_margin.current_modcod_idx(), 3);
    BOOST_CHECK_CLOSE(conservative.bootstrap_margin_db(), 2.0f, 0.01f);
}

BOOST_AUTO_TEST_CASE(configurable_guard_changes_selection_threshold)
{
    acm_controller default_guard(test_ladder());
    acm_controller one_db_guard(test_ladder(), 1.0f);

    // At 8.5 dB, the default 2.5 dB guard cannot admit the 6 dB high mode,
    // while a configured 1 dB guard can after normal upshift confirmation.
    for (int i = 0; i < 5; ++i) {
        default_guard.update_snr(8.5f);
        one_db_guard.update_snr(8.5f);
    }
    one_db_guard.update_feedback(8.5f, 0.01f, 64, 0, 2);
    for (int i = 0; i < 5; ++i) {
        default_guard.update_snr(8.5f);
        one_db_guard.update_snr(8.5f);
    }
    BOOST_CHECK_EQUAL(default_guard.current_modcod_idx(), 2);
    BOOST_CHECK_EQUAL(one_db_guard.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(fer_alarm_does_not_cascade_downshifts)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);

    for (int i = 0; i < 10; ++i)
        controller.update_snr(14.0f, 0.01f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(active_fer_alarm_blocks_upshift)
{
    acm_controller controller(test_ladder());
    for (int i = 0; i < 10; ++i)
        controller.update_snr(10.0f, 0.01f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
}

BOOST_AUTO_TEST_CASE(upshift_hysteresis_prevents_stationary_boundary_chatter)
{
    acm_controller controller(test_ladder());

    // The middle mode's nominal admission boundary is 5.5 dB (3.0 dB
    // threshold + 2.5 dB guard).  The additional 1 dB upshift hysteresis
    // keeps us at the floor while samples merely touch that boundary.
    for (int i = 0; i < 10; ++i)
        controller.update_snr(6.4f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);

    // Once admitted with real headroom, small stationary estimator jitter
    // must not immediately undo the confirmed upshift.
    for (int i = 0; i < 6; ++i)
        controller.update_snr(6.6f);
    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 2);
    controller.update_snr(5.95f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 2);
}

BOOST_AUTO_TEST_CASE(isolated_pilot_snr_dip_does_not_force_downshift)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 10.0f);
    controller.update_snr(10.0f);
    controller.update_snr(10.0f);

    // A single estimator outlier is rejected by the causal median filter.
    controller.update_snr(4.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 3);

    // A sustained deterioration is real and must still cause a downshift.
    controller.update_snr(4.0f);
    BOOST_CHECK_LT(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(olla_nacks_raise_correction_and_force_safe_mode)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 9.6f);

    controller.update_feedback(9.6f, 0.02f, 3, 3, 3);
    // The three NACKs arrived in one delayed feedback report and therefore
    // represent one adaptation epoch: the first would already have caused
    // this downshift if feedback were instantaneous.
    BOOST_CHECK_GT(controller.olla_offset_db(3), 0.48f);
    BOOST_CHECK_LT(controller.olla_offset_db(3), 0.51f);
    BOOST_CHECK_SMALL(controller.olla_offset_db(2), 0.01f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 2);
}

BOOST_AUTO_TEST_CASE(olla_bounds_aggregated_geo_nack_bursts)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);
    controller.update_feedback(14.0f, 1.0f, 261, 261, 3);

    // A complete probation failure must cause one prompt downshift without
    // poisoning every lower mode as a global OLLA correction would.
    BOOST_CHECK_GT(controller.olla_offset_db(3), 0.49f);
    BOOST_CHECK_LT(controller.olla_offset_db(3), 0.51f);
    BOOST_CHECK_SMALL(controller.olla_offset_db(2), 0.01f);
    BOOST_CHECK_LT(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(olla_step_ratio_targets_one_in_one_hundred_thousand)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);
    const float before = controller.olla_offset_db(3);
    controller.update_feedback(14.0f, 0.0f, 100000, 1, 3);

    // One NACK and 99,999 ACKs have zero expected update at FER=1e-5.
    BOOST_CHECK_SMALL(controller.olla_offset_db(3) - before, 1.0e-4f);
}

BOOST_AUTO_TEST_CASE(olla_clean_frames_recover_headroom_and_counter_reset_is_safe)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);
    controller.update_feedback(14.0f, 0.0f, 100, 0, 3);
    controller.update_feedback(14.0f, 0.0f, 101, 1, 3);
    const float after_error = controller.olla_offset_db(3);
    BOOST_REQUIRE_GT(after_error, 0.49f);

    // The NACK conservatively moved the controller to the middle mode.
    // Its clean outcomes do not erase the high mode's learned correction.
    controller.update_feedback(14.0f, 0.0f, 100000, 0, 2);
    BOOST_CHECK_CLOSE(controller.olla_offset_db(3), after_error, 0.001f);

    // Once the higher mode is eligible again, its own clean outcomes recover
    // its headroom according to the target-FER OLLA law.
    for (int i = 0; i < 5; ++i)
        controller.update_snr(14.0f);
    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 3);
    controller.update_feedback(14.0f, 0.0f, 100101, 1, 3);
    BOOST_CHECK_LT(controller.olla_offset_db(3), after_error);

    const float before_reset = controller.olla_offset_db(3);
    controller.update_feedback(14.0f, 0.0f, 0, 0, 2);
    BOOST_CHECK_CLOSE(controller.olla_offset_db(3), before_reset, 0.001f);
}

BOOST_AUTO_TEST_CASE(olla_tracks_each_modcod_counter_independently)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);
    controller.update_feedback(14.0f, 0.0f, 1000, 0, 2);
    controller.update_feedback(14.0f, 0.0f, 90000, 0, 3);
    BOOST_CHECK_LT(controller.olla_offset_db(3), 0.0f);

    // A high-mode NACK makes mode 2 current.
    controller.update_feedback(14.0f, 0.01f, 90001, 1, 3);
    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 2);

    // Returning to mode 2 compares against mode 2's previous snapshot, not
    // mode 3's unrelated (larger) cumulative counter.
    controller.update_feedback(14.0f, 0.0f, 1001, 1, 2);
    BOOST_CHECK_GT(controller.olla_offset_db(2), 0.04f);
    BOOST_CHECK_LT(controller.olla_offset_db(3), 0.1f);
}

BOOST_AUTO_TEST_CASE(upward_candidate_is_not_data_safe_until_clean_probe)
{
    acm_controller controller(test_ladder());
    for (int i = 0; i < 3; ++i)
        controller.update_snr(14.0f);

    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 3);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 1);

    controller.update_feedback(14.0f, 0.0f, 63, 0, 3);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 1);
    controller.update_feedback(14.0f, 0.0f, 64, 0, 3);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(candidate_nack_discards_accumulated_validation)
{
    acm_controller controller(test_ladder());
    for (int i = 0; i < 3; ++i)
        controller.update_snr(10.0f);

    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 2);
    controller.update_feedback(10.0f, 0.0f, 48, 0, 2);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 1);

    controller.update_feedback(10.0f, 0.01f, 49, 1, 2);
    BOOST_CHECK_EQUAL(controller.validated_modcod_idx(), 1);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    BOOST_CHECK(controller.bootstrap_complete());
}

BOOST_AUTO_TEST_CASE(feedback_outage_restarts_direct_bootstrap)
{
    acm_controller controller(test_ladder());
    for (int i = 0; i < 3; ++i)
        controller.update_snr(14.0f);
    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 3);

    controller.drop_to_floor();
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    BOOST_CHECK(!controller.bootstrap_complete());
    controller.update_snr(14.0f);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 1);
    controller.update_snr(14.0f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 3);
}

BOOST_AUTO_TEST_CASE(late_failed_candidate_fragments_do_not_repeat_olla_step)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);

    controller.update_feedback(14.0f, 1.0f, 200, 200, 3);
    const float after_retreat = controller.olla_offset_db(3);
    BOOST_REQUIRE_EQUAL(controller.current_modcod_idx(), 2);

    controller.update_feedback(14.0f, 1.0f, 201, 201, 3);
    BOOST_CHECK_CLOSE(controller.olla_offset_db(3), after_retreat, 0.001f);
}

BOOST_AUTO_TEST_CASE(first_mode_report_is_not_discarded)
{
    acm_controller controller(test_ladder());
    drive_to_high(controller, 14.0f);

    // A first report can contain the complete idle-probation burst. Its NACK
    // must update OLLA and immediately take user traffic off the failed mode.
    controller.update_feedback(14.0f, 0.01f, 100, 1, 3);
    BOOST_CHECK_GT(controller.olla_offset_db(3), 0.49f);
    BOOST_CHECK_EQUAL(controller.current_modcod_idx(), 2);
}
