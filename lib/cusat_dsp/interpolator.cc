/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Polyphase fractional-delay filter bank.
 *
 * Tap convention:
 *
 *   taps(p)[i] multiplies the i-th newest sample of the window, i = 0..L-1,
 *   L = taps_per_phase(). With h the continuous prototype (centred on 0,
 *   in input samples) and D = group_delay_samples(),
 *
 *       taps(p)[i] = h(i - D - p / P) / sum_i h(i - D - p / P)
 *
 *   so the dot product of phase p with the window ending at sample n is
 *   the prototype-filtered signal f(t) = sum_m h(t - m) x[m] at
 *   t = n - D - p / P: phase p adds p / P samples of delay to phase 0, and
 *   every phase has exactly unit DC gain.
 *
 *   For an NCO position tau = m + fr (basepoint m, fr in [0, 1)) the
 *   kernels round q = round(fr P) and use the window ending at
 *   n = m + (q > 0) with phase (P - q) mod P, which evaluates f(tau - D)
 *   up to the 1 / (2P) phase quantization.
 *
 *   L is odd and D = (L - 2) / 2: the window [n - L + 1, n] then covers the
 *   prototype support [-D, D] around t for every phase.
 *
 * Prototypes (both with a cosine taper on the outer quarter of the span):
 *   rrc_matched   h(x) = rrc_impulse(x / sps, rolloff) on |x| <= span * sps
 *                 (the TX pulse of rrc_filter.h).
 *   band_limited  Kaiser-windowed ideal low-pass, cut-off 1/2 cycle/sample
 *                 (midway between the signal edge (1 + a) / (2 sps) and the
 *                 first image 1 - (1 + a) / (2 sps)). The length follows
 *                 Kaiser's formula N = (A - 8) / (2.285 * 2 pi * df) for
 *                 A = 46 dB (6 dB above the 40 dB image-rejection target,
 *                 passband ripple 20 log10(1 + 10^(-46/20)) = 0.04 dB < 0.1 dB),
 *                 df = 1 - (1 + a) / sps; beta = 0.5842 (A-21)^0.4 +
 *                 0.07886 (A-21) (Kaiser 1974; harris 2004 ch. 3). The
 *                 caller's span is a lower bound.
 */

#include "interpolator.h"
#include "rrc_filter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cusat_dsp {

namespace {

/* Modified Bessel function of the first kind, order 0 (power series). */
double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 64; k++) {
        term *= q / ((double)k * k);
        sum += term;
        if (term < 1e-17 * sum)
            break;
    }
    return sum;
}

constexpr double KAISER_ATTEN_DB = 46.0;
constexpr double TAPER = 0.25; // fraction of the half-span with a cosine taper

} // namespace

polyphase_filter_bank::polyphase_filter_bank(const filter_bank_design& d)
{
    if (d.n_phases < 1 || d.samples_per_symbol <= 0.0 || d.span_symbols < 1)
        throw std::invalid_argument("polyphase_filter_bank: invalid design");
    d_n_phases = d.n_phases;
    const double sps = d.samples_per_symbol;

    double half_width = d.span_symbols * sps; // prototype support, samples
    double kaiser_beta = 0.0;
    if (d.prototype == bank_prototype::band_limited) {
        const double df = std::max(1.0 - (1.0 + d.rolloff) / sps, 0.02);
        const double n_kaiser = (KAISER_ATTEN_DB - 8.0) / (2.285 * 2.0 * M_PI * df);
        half_width = std::max(half_width, std::ceil(n_kaiser / 2.0) + 1.0);
        const double a = KAISER_ATTEN_DB - 21.0;
        kaiser_beta = 0.5842 * std::pow(a, 0.4) + 0.07886 * a;
    }

    // smallest odd L with D = (L - 2) / 2 >= half_width
    int len = 2 * (int)std::ceil(half_width) + 2;
    if (len % 2 == 0)
        len++;
    d_taps_per_phase = len;
    d_group_delay = (len - 2) / 2.0;

    // Cosine (Tukey) taper over the outer TAPER of the support: the
    // truncation edge of the prototype would otherwise fall between the
    // samples of each phase asymmetrically and bias the phase delays (up
    // to 0.08 samples at span 5); with the taper every phase meets the
    // p / n_phases delay law to < 1e-3 sample (qa_filter_bank).
    auto taper = [&](double x) {
        const double r = std::fabs(x) / half_width;
        if (r <= 1.0 - TAPER)
            return 1.0;
        return 0.5 * (1.0 + std::cos(M_PI * (r - (1.0 - TAPER)) / TAPER));
    };
    auto prototype = [&](double x) -> double {
        if (std::fabs(x) > half_width)
            return 0.0;
        if (d.prototype == bank_prototype::rrc_matched)
            return rrc_impulse(x / sps, d.rolloff) * taper(x);
        // windowed sinc, cut-off 0.5 cycle/sample: sin(pi x) / (pi x)
        const double s = std::fabs(x) < 1e-12 ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
        const double r = x / half_width;
        const double w = bessel_i0(kaiser_beta * std::sqrt(std::max(0.0, 1.0 - r * r))) /
                         bessel_i0(kaiser_beta);
        return s * w * taper(x);
    };

    d_taps.assign((size_t)d_n_phases * len, 0.0f);
    std::vector<double> h(len);
    for (int p = 0; p < d_n_phases; p++) {
        const double shift = d_group_delay + (double)p / d_n_phases;
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            h[i] = prototype(i - shift);
            sum += h[i];
        }
        if (sum == 0.0)
            throw std::invalid_argument("polyphase_filter_bank: zero-gain phase");
        float* out = &d_taps[(size_t)p * len];
        for (int i = 0; i < len; i++)
            out[i] = (float)(h[i] / sum);
    }
}

int polyphase_filter_bank::n_phases() const { return d_n_phases; }

int polyphase_filter_bank::taps_per_phase() const { return d_taps_per_phase; }

unsigned polyphase_filter_bank::history() const { return (unsigned)(d_taps_per_phase - 1); }

const float* polyphase_filter_bank::taps(int phase) const
{
    return &d_taps[(size_t)phase * d_taps_per_phase];
}

std::vector<float> polyphase_filter_bank::flat_taps() const { return d_taps; }

double polyphase_filter_bank::group_delay_samples() const { return d_group_delay; }

} // namespace cusat_dsp
