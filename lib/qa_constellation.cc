/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Constellations (EN 302 307-1 clause 5.4, EN 302 307-2 clause 5.4).
 */

#include "cusat_dsp/acm_config.h"
#include "cusat_dsp/modulation/constellation.h"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <random>
#include <set>

namespace tt = boost::test_tools;

namespace cusat_dsp {

BOOST_AUTO_TEST_CASE(test_qpsk_points)
{
    // clause 5.4.1: first bit -> I sign, second bit -> Q sign, 0 -> positive
    const constellation_t c = make_constellation(MOD_QPSK, RATE_1_2);
    const float a = (float)M_SQRT1_2;
    BOOST_CHECK_EQUAL(c.bits_per_sym, 2);
    BOOST_CHECK_EQUAL(c.points[0], complexf(a, a));
    BOOST_CHECK_EQUAL(c.points[1], complexf(a, -a));
    BOOST_CHECK_EQUAL(c.points[2], complexf(-a, a));
    BOOST_CHECK_EQUAL(c.points[3], complexf(-a, -a));
}

BOOST_AUTO_TEST_CASE(test_8psk_points)
{
    // clause 5.4.2, figure 10: label -> phase in units of pi / 4
    const int phase[8] = { 1, 0, 4, -3, 2, -1, 3, -2 };
    const constellation_t c = make_constellation(MOD_8PSK, RATE_3_5);
    for (int label = 0; label < 8; label++)
        BOOST_CHECK_SMALL(std::abs(c.points[label] - std::polar(1.0f, (float)(phase[label] * M_PI / 4))),
                          1e-6f);
}

static int n_rings(const constellation_t& c)
{
    std::set<long> radii;
    for (int i = 0; i < c.n_points; i++)
        radii.insert(std::lround(std::abs(c.points[i]) * 1e4));
    return radii.size();
}

BOOST_AUTO_TEST_CASE(test_every_modcod_constellation)
{
    for (const modcod_def_t& m : get_supported_modcods()) {
        const constellation_t c = make_constellation(m.constellation, m.code_rate);
        BOOST_TEST_CONTEXT(m.name)
        {
            BOOST_CHECK_EQUAL(c.n_points, 1 << c.bits_per_sym);
            double energy = 0;
            std::set<std::pair<long, long>> distinct;
            for (int i = 0; i < c.n_points; i++) {
                energy += std::norm(c.points[i]);
                distinct.insert({ std::lround(c.points[i].real() * 1e5), std::lround(c.points[i].imag() * 1e5) });
            }
            BOOST_TEST(energy / c.n_points == 1.0, tt::tolerance(1e-3));
            // 256APSK 20/30 (Table 15d) has near-coincident label pairs by design
            if (!(m.constellation == MOD_256APSK && m.code_rate == RATE_20_30))
                BOOST_CHECK_EQUAL((int)distinct.size(), c.n_points);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_ring_structures)
{
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_16APSK, RATE_3_4)), 2);      // 4+12
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_16APSK, RATE_90_180)), 2);   // 8+8
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_32APSK, RATE_3_4)), 3);      // 4+12+16
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_32APSK, RATE_128_180)), 4);  // 4+8+4+16
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_64APSK, RATE_132_180)), 4);  // 4+12+20+28
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_128APSK, RATE_135_180)), 6);
    BOOST_CHECK_EQUAL(n_rings(make_constellation(MOD_256APSK, RATE_135_180)), 8);

    // 4+12APSK: gamma = R2 / R1 of EN 302 307-1 Table 9, e.g. 2.85 at 3/4
    const constellation_t c = make_constellation(MOD_16APSK, RATE_3_4);
    BOOST_TEST(std::abs(c.points[0]) / std::abs(c.points[12]) == 2.85f, tt::tolerance(1e-5f));
}

BOOST_AUTO_TEST_CASE(test_nearest_label)
{
    std::mt19937 rng(9);
    std::normal_distribution<float> noise(0.0f, 0.02f);
    for (modulation_t mod : { MOD_QPSK, MOD_8PSK, MOD_16APSK, MOD_32APSK, MOD_64APSK }) {
        const constellation_t c = make_constellation(mod, mod == MOD_64APSK ? RATE_4_5 : RATE_3_4);
        for (int label = 0; label < c.n_points; label++)
            BOOST_CHECK_EQUAL(nearest_label(c, c.points[label] + complexf(noise(rng), noise(rng))), label);
    }
}

BOOST_AUTO_TEST_CASE(test_unsupported_modulation)
{
    BOOST_CHECK_THROW(make_constellation((modulation_t)99, RATE_1_2), std::invalid_argument);
}

// The data-aided SNR estimator test is in qa_dd_snr.cc.

BOOST_AUTO_TEST_CASE(test_32apsk_4_12_16_radii)
{
    // EN 302 307-1 Table 10: gamma1 = R2/R1, gamma2 = R3/R1; unit energy
    struct { code_rate_t r; double g1, g2; } table[] = {
        { RATE_3_4, 2.84, 5.27 }, { RATE_4_5, 2.72, 4.87 }, { RATE_5_6, 2.64, 4.64 },
        { RATE_8_9, 2.54, 4.33 }, { RATE_9_10, 2.53, 4.30 },
    };
    for (const auto& t : table) {
        const constellation_t c = make_constellation(MOD_32APSK, t.r);
        BOOST_REQUIRE_EQUAL(c.n_points, 32);
        // figure 12 labels: 10001 (17) inner ring, 00000 (0) middle, 01000 (8) outer
        const double r1 = std::abs(c.points[17]), r2 = std::abs(c.points[0]), r3 = std::abs(c.points[8]);
        BOOST_CHECK_SMALL(r2 / r1 - t.g1, 1e-6 * t.g1);
        BOOST_CHECK_SMALL(r3 / r1 - t.g2, 1e-6 * t.g2);
        double e = 0;
        int n1 = 0, n2 = 0, n3 = 0;
        for (int i = 0; i < 32; i++) {
            const double r = std::abs(c.points[i]);
            e += r * r;
            n1 += std::fabs(r - r1) < 1e-5;
            n2 += std::fabs(r - r2) < 1e-5;
            n3 += std::fabs(r - r3) < 1e-5;
        }
        BOOST_CHECK_SMALL(e / 32 - 1.0, 1e-6);
        BOOST_CHECK_EQUAL(n1, 4);
        BOOST_CHECK_EQUAL(n2, 12);
        BOOST_CHECK_EQUAL(n3, 16);
    }
}

} // namespace cusat_dsp
