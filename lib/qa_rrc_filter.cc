/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * TX pulse shaping.
 */

#include "cusat_dsp/interpolator.h"
#include "cusat_dsp/rrc_filter.h"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace cusat_dsp {

using cd = std::complex<double>;

BOOST_AUTO_TEST_CASE(test_impulse_limits)
{
    for (double a : { 0.05, 0.1, 0.2, 0.25, 0.35 }) {
        // removable singularities equal the neighbouring values
        const double t1 = 1.0 / (4 * a);
        BOOST_CHECK_CLOSE(rrc_impulse(t1, a), rrc_impulse(t1 + 1e-6, a), 1e-2);
        BOOST_CHECK_CLOSE(rrc_impulse(-t1, a), rrc_impulse(-t1 - 1e-6, a), 1e-2);
        BOOST_CHECK_CLOSE(rrc_impulse(0.0, a), rrc_impulse(1e-6, a), 1e-4);
        BOOST_CHECK_CLOSE(rrc_impulse(0.0, a), 1 - a + 4 * a / M_PI, 1e-9);
        // symmetric
        BOOST_CHECK_CLOSE(rrc_impulse(0.7, a), rrc_impulse(-0.7, a), 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(test_lengths_and_norms)
{
    for (int sps : { 2, 4, 8 })
        for (int span : { 5, 25 }) {
            auto t = rrc_pulse(sps, 0.2, span, pulse_norm::dc_gain_sps);
            BOOST_CHECK_EQUAL((int)t.size(), 2 * span * sps + 1);
            double s = 0;
            for (float v : t)
                s += v;
            BOOST_CHECK_CLOSE(s, (double)sps, 1e-4);
            for (size_t i = 0; i < t.size(); i++)
                BOOST_CHECK_CLOSE(t[i] + 1e-9, t[t.size() - 1 - i] + 1e-9, 1e-3);
            auto e = rrc_pulse(sps, 0.2, span, pulse_norm::unit_energy);
            double en = 0;
            for (float v : e)
                en += (double)v * v;
            BOOST_CHECK_CLOSE(en, 1.0, 1e-4);
        }
}

/* EN 302 307-1 / -2 Annex A, Table A.1: (f / fN, dB) for upper and lower
 * limits; the mask beyond point S stays at -40 dB. */
struct mask_pt {
    double f;
    double db;
};

static double interp(const std::vector<mask_pt>& m, double f)
{
    if (f <= m.front().f)
        return m.front().db;
    for (size_t i = 1; i < m.size(); i++)
        if (f <= m[i].f) {
            const double r = (f - m[i - 1].f) / (m[i].f - m[i - 1].f);
            return m[i - 1].db + r * (m[i].db - m[i - 1].db);
        }
    return m.back().db;
}

BOOST_AUTO_TEST_CASE(test_spectrum_mask)
{
    struct mask_t {
        double a;
        double G, I, L, P, Q, N, S;
    };
    // frequencies of points G, I, L(=M), P, Q, N, S per roll-off (Table A.1)
    const mask_t masks[] = {
        { 0.35, 0.8, 0.9, 1.2, 1.4, 1.6, 1.8, 2.12 },
        { 0.25, 0.86, 0.93, 1.13, 1.30, 1.45, 1.60, 1.83 },
        { 0.20, 0.89, 0.94, 1.11, 1.23, 1.4, 1.5, 1.7 },
        { 0.15, 0.9175, 0.955, 1.0825, 1.1725, 1.3, 1.375, 1.525 },
        { 0.10, 0.945, 0.97, 1.055, 1.115, 1.2, 1.25, 1.35 },
        { 0.05, 0.9725, 0.985, 1.0275, 1.0575, 1.1, 1.125, 1.175 },
    };
    const int sps = 4, span = 25; // the dvbs2-gse-tx configuration
    for (const mask_t& mk : masks) {
        const std::vector<mask_pt> upper = { { 0, 0.25 },     { 0.4, 0.25 }, { mk.G, 0.15 },
                                             { mk.I, -0.5 },  { 1.0, -2.0 }, { mk.L, -8.0 },
                                             { mk.P, -16.0 }, { mk.Q, -24.0 }, { mk.N, -35.0 },
                                             { mk.S, -40.0 }, { 100, -40.0 } };
        const std::vector<mask_pt> lower = { { 0, -0.25 }, { 0.2, -0.4 }, { 0.4, -0.4 },
                                             { mk.G, -1.1 }, { 1.0, -4.0 }, { mk.L, -11.0 } };
        auto t = rrc_pulse(sps, mk.a, span, pulse_norm::dc_gain_sps);
        auto H = [&](double f_over_fn) { // fN = Rs / 2 -> cycles/sample f/fN / (2 sps)
            const double f = f_over_fn / (2.0 * sps);
            cd acc = 0;
            for (size_t i = 0; i < t.size(); i++)
                acc += (double)t[i] * std::polar(1.0, -2 * M_PI * f * (double)i);
            return std::norm(acc);
        };
        const double h0 = H(0);
        int violations = 0;
        for (int k = 0; k <= 800; k++) {
            const double f = 4.0 * k / 800.0;
            const double db = 10 * std::log10(H(f) / h0 + 1e-30);
            if (db > interp(upper, f) + 1e-9)
                violations++;
            if (f <= mk.L && db < interp(lower, f) - 1e-9)
                violations++;
        }
        BOOST_TEST_CONTEXT("rolloff " << mk.a) { BOOST_CHECK_EQUAL(violations, 0); }
    }
}

/* R3.5: TX pulse convolved with the RX rrc_matched bank (span 5, sps 4):
 * the symbol-spaced response is Nyquist up to the truncations. */
BOOST_AUTO_TEST_CASE(test_isi)
{
    const int sps = 4;
    auto tx = rrc_pulse(sps, 0.2, 25, pulse_norm::dc_gain_sps);
    filter_bank_design d;
    d.samples_per_symbol = sps;
    d.rolloff = 0.2;
    d.span_symbols = 5;
    polyphase_filter_bank b(d);
    // overall response at the symbol instants: phase 0, newest-first taps,
    // delay D; tx centre at index 25 * sps
    const float* rx = b.taps(0);
    const int L = b.taps_per_phase();
    const double D = b.group_delay_samples(); // half-integer: interpolate
    // use the phase that realises an integer total delay
    const int P = b.n_phases();
    const double frac = D - std::floor(D);
    const int ph = (int)std::lround((1.0 - frac) * P) % P; // adds (1-frac) delay
    rx = b.taps(ph);
    const double total = 25.0 * sps + D + (double)ph / P;
    std::vector<double> conv(tx.size() + L, 0.0);
    for (size_t i = 0; i < tx.size(); i++)
        for (int j = 0; j < L; j++)
            conv[i + j] += (double)tx[i] * rx[j];
    const int c = (int)std::lround(total);
    double main = conv[c], isi = 0;
    for (int k = -40; k <= 40; k++) {
        if (k == 0)
            continue;
        const int idx = c + k * sps;
        if (idx >= 0 && idx < (int)conv.size())
            isi += conv[idx] * conv[idx];
    }
    const double mer = 10 * std::log10(main * main / isi);
    BOOST_TEST_MESSAGE("TX(span 25) x RX(span 5) MER " << mer << " dB");
    BOOST_CHECK_GE(mer, 35.0);
}

/* R3.6: TX output level of random QPSK with sum(taps) = sps. */
BOOST_AUTO_TEST_CASE(test_level)
{
    std::mt19937 rng(5);
    for (int sps : { 2, 4 }) {
        auto t = rrc_pulse(sps, 0.2, 25, pulse_norm::dc_gain_sps);
        double e = 0;
        for (float v : t)
            e += (double)v * v;
        const double expect = std::sqrt(e / sps); // per-sample RMS for unit-energy symbols
        const int n = 20000;
        std::vector<cd> sym(n);
        for (auto& s : sym)
            s = cd((rng() & 1) ? -1 : 1, (rng() & 1) ? -1 : 1) / std::sqrt(2.0);
        double p = 0;
        int cnt = 0;
        for (int m = (int)t.size(); m < n * sps - (int)t.size(); m++) {
            cd y = 0;
            for (size_t i = 0; i < t.size(); i++) {
                const int idx = m - (int)i;
                if (idx % sps == 0)
                    y += (double)t[i] * sym[idx / sps];
            }
            p += std::norm(y);
            cnt++;
        }
        const double rms = std::sqrt(p / cnt);
        BOOST_CHECK_SMALL(20 * std::log10(rms / expect), 0.1);
        BOOST_CHECK_SMALL(20 * std::log10(rms), 0.2); // ~unit RMS
    }
}

} // namespace cusat_dsp
