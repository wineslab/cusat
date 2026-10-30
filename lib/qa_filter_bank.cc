/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Polyphase filter bank.
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

/* Output of the bank for NCO position tau (input samples), exactly as the
 * GPU kernels evaluate it (timing_recovery_kernel.cu). */
static cd bank_output(const polyphase_filter_bank& b, const std::vector<cd>& x, double tau)
{
    const int P = b.n_phases(), L = b.taps_per_phase();
    const int ip = (int)std::floor(tau);
    const int q = (int)((tau - ip) * P + 0.5);
    const int n = ip + (q > 0 ? 1 : 0);
    const int ph = (q > 0 && q < P) ? P - q : 0;
    const float* t = b.taps(ph);
    cd acc = 0;
    for (int i = 0; i < L; i++)
        acc += (double)t[i] * x[n - i];
    return acc;
}

static const double SPS_LIST[] = { 1.2288, 1.536, 2.0, 2.4576, 4.096 };

BOOST_AUTO_TEST_CASE(test_structure_and_dc_gain)
{
    for (double sps : SPS_LIST)
        for (bank_prototype proto : { bank_prototype::rrc_matched, bank_prototype::band_limited }) {
            filter_bank_design d;
            d.samples_per_symbol = sps;
            d.rolloff = 0.2;
            d.span_symbols = sps < 2.5 ? 16 : 5;
            d.n_phases = 128;
            d.prototype = proto;
            polyphase_filter_bank b(d);
            BOOST_CHECK_EQUAL(b.taps_per_phase() % 2, 1);
            BOOST_CHECK_EQUAL((int)b.history(), b.taps_per_phase() - 1);
            BOOST_CHECK_EQUAL((int)b.flat_taps().size(), b.n_phases() * b.taps_per_phase());
            for (int p = 0; p < b.n_phases(); p++) {
                double s = 0;
                for (int i = 0; i < b.taps_per_phase(); i++)
                    s += b.taps(p)[i];
                BOOST_CHECK_SMALL(s - 1.0, 1e-4);
            }
        }
}

/* R3.1: RRC-shaped random QPSK, interpolated at 1000 random instants,
 * against the ideal matched-filter output (the symbol itself, up to a
 * complex gain) at the true instant. */
BOOST_AUTO_TEST_CASE(test_interpolation_accuracy)
{
    const double alpha = 0.2;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (double sps : SPS_LIST) {
        const int span = sps < 2.5 ? 16 : 5;
        filter_bank_design d;
        d.samples_per_symbol = sps;
        d.rolloff = alpha;
        d.span_symbols = span;
        d.n_phases = 128;
        polyphase_filter_bank b(d);
        const int tx_span = 40; // near-ideal transmit pulse
        const int n_sym = 2 * tx_span + 4 * span + 20;
        double err = 0, sig = 0;
        cd gain_num = 0;
        double gain_den = 0;
        std::vector<cd> outs, refs;
        for (int trial = 0; trial < 1000; trial++) {
            // symbol 0 at continuous sample time t0
            std::vector<cd> a(n_sym);
            for (auto& s : a)
                s = cd(u(rng) < 0.5 ? -1 : 1, u(rng) < 0.5 ? -1 : 1) / std::sqrt(2.0);
            const int k0 = n_sym / 2;
            const double t0 = 200.0 + u(rng);
            const int n_samp = (int)(t0 + (n_sym - k0) * sps) + 200;
            std::vector<cd> x(n_samp, 0.0);
            for (int n = 0; n < n_samp; n++) {
                cd acc = 0;
                const double ts = (n - t0) / sps + k0; // time in symbols
                const int kc = (int)std::floor(ts);
                for (int k = std::max(0, kc - tx_span); k <= std::min(n_sym - 1, kc + tx_span); k++)
                    acc += a[k] * rrc_impulse(ts - k, alpha);
                x[n] = acc;
            }
            const double tau = t0 + b.group_delay_samples();
            outs.push_back(bank_output(b, x, tau));
            refs.push_back(a[k0]);
        }
        for (size_t i = 0; i < outs.size(); i++) {
            gain_num += outs[i] * std::conj(refs[i]);
            gain_den += std::norm(refs[i]);
        }
        const cd g = gain_num / gain_den;
        for (size_t i = 0; i < outs.size(); i++) {
            err += std::norm(outs[i] - g * refs[i]);
            sig += std::norm(g * refs[i]);
        }
        const double mer = 10 * std::log10(sig / err);
        BOOST_TEST_MESSAGE("sps " << sps << " span " << span << " MER " << mer << " dB");
        BOOST_CHECK_GE(mer, span >= 16 ? 40.0 : 30.0);
    }
}

/* R3.2: delay of every phase (phase slope in the signal band, equivalent
 * to the cross-correlation peak of a band-limited test signal). */
BOOST_AUTO_TEST_CASE(test_group_delay)
{
    for (double sps : SPS_LIST)
        for (bank_prototype proto : { bank_prototype::rrc_matched, bank_prototype::band_limited }) {
            filter_bank_design d;
            d.samples_per_symbol = sps;
            d.rolloff = 0.2;
            d.span_symbols = sps < 2.5 ? 16 : 5;
            d.prototype = proto;
            polyphase_filter_bank b(d);
            double worst = 0;
            for (int p = 0; p < b.n_phases(); p++) {
                // delay = -d arg H / d omega near DC, measured relative to
                // the nominal delay (no phase wrapping)
                const double nominal = b.group_delay_samples() + (double)p / b.n_phases();
                const double w = 2 * M_PI * 0.02 / sps;
                cd hp = 0, hm = 0;
                for (int i = 0; i < b.taps_per_phase(); i++) {
                    hp += (double)b.taps(p)[i] * std::polar(1.0, -w * (i - nominal));
                    hm += (double)b.taps(p)[i] * std::polar(1.0, w * (i - nominal));
                }
                const double delay = -(std::arg(hp) - std::arg(hm)) / (2 * w);
                worst = std::max(worst, std::fabs(delay));
            }
            BOOST_TEST_MESSAGE("sps " << sps << " proto " << (int)proto << " worst delay error " << worst);
            BOOST_CHECK_LT(worst, 0.002);
        }
}

/* band_limited: flat passband (+-0.1 dB) up to (1+a)/(2 sps), >= 40 dB
 * attenuation of the first image starting at 1 - (1+a)/(2 sps). */
BOOST_AUTO_TEST_CASE(test_band_limited_response)
{
    for (double sps : SPS_LIST) {
        const double alpha = 0.2;
        filter_bank_design d;
        d.samples_per_symbol = sps;
        d.rolloff = alpha;
        d.span_symbols = sps < 2.5 ? 16 : 5;
        d.prototype = bank_prototype::band_limited;
        polyphase_filter_bank b(d);
        const int P = b.n_phases(), L = b.taps_per_phase();
        // continuous response of the interleaved prototype (rate P)
        auto H = [&](double f) {
            cd acc = 0;
            for (int p = 0; p < P; p++)
                for (int i = 0; i < L; i++)
                    acc += (double)b.taps(p)[i] * std::polar(1.0, -2 * M_PI * f * (i - (double)p / P));
            return std::abs(acc) / P;
        };
        const double fp = (1 + alpha) / (2 * sps);
        double ripple = 0, image = 0;
        for (int k = 0; k <= 40; k++) {
            const double f = fp * k / 40.0;
            ripple = std::max(ripple, std::fabs(20 * std::log10(H(f))));
            image = std::max(image, H(1.0 - fp + 2 * fp * k / 40.0));
        }
        BOOST_TEST_MESSAGE("sps " << sps << " ripple " << ripple << " dB image " << 20 * std::log10(image) << " dB");
        BOOST_CHECK_LE(ripple, 0.1);
        BOOST_CHECK_LE(20 * std::log10(image), -40.0);
    }
}

} // namespace cusat_dsp
