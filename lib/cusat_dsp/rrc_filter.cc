/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Root-raised-cosine pulse. Closed form of the square-root
 * raised-cosine impulse response for unit symbol period (e.g. Proakis &
 * Salehi, Digital Communications, or Rice 2009 appendix A):
 *
 *            sin(pi t (1 - a)) + 4 a t cos(pi t (1 + a))
 *   h(t) = ---------------------------------------------
 *                     pi t (1 - (4 a t)^2)
 *
 * normalized so that the pulse has unit energy over a continuous symbol
 * period. The two removable singularities are evaluated by their limits:
 *   t = 0:            h = 1 - a + 4 a / pi
 *   |t| = 1 / (4 a):  h = a / sqrt(2) [(1 + 2/pi) sin(pi / 4a) + (1 - 2/pi) cos(pi / 4a)]
 * Roll-off a per EN 302 307-1 clause 5.6 / EN 302 307-2 clause 5.6.
 */

#include "rrc_filter.h"
#include <cmath>
#include <stdexcept>

namespace cusat_dsp {

double rrc_impulse(double t, double a)
{
    if (a <= 0.0) {
        // a = 0: ideal sinc (no excess bandwidth)
        return std::fabs(t) < 1e-12 ? 1.0 : std::sin(M_PI * t) / (M_PI * t);
    }
    const double eps = 1e-9;
    if (std::fabs(t) < eps)
        return 1.0 - a + 4.0 * a / M_PI;
    const double t_sing = 1.0 / (4.0 * a);
    if (std::fabs(std::fabs(t) - t_sing) < eps) {
        const double x = M_PI / (4.0 * a);
        return a / M_SQRT2 *
               ((1.0 + 2.0 / M_PI) * std::sin(x) + (1.0 - 2.0 / M_PI) * std::cos(x));
    }
    const double num = std::sin(M_PI * t * (1.0 - a)) + 4.0 * a * t * std::cos(M_PI * t * (1.0 + a));
    const double den = M_PI * t * (1.0 - 16.0 * a * a * t * t);
    return num / den;
}

std::vector<float> rrc_pulse(int sps, double rolloff, int span_symbols, pulse_norm norm)
{
    if (sps < 1 || span_symbols < 0 || rolloff < 0.0 || rolloff > 1.0)
        throw std::invalid_argument("rrc_pulse: invalid parameters");
    const int half = span_symbols * sps;
    std::vector<double> h(2 * half + 1);
    double sum = 0.0, energy = 0.0;
    for (int n = -half; n <= half; n++) {
        const double v = rrc_impulse((double)n / sps, rolloff);
        h[n + half] = v;
        sum += v;
        energy += v * v;
    }
    double scale = 1.0;
    switch (norm) {
    case pulse_norm::unit_energy: scale = energy > 0.0 ? 1.0 / std::sqrt(energy) : 1.0; break;
    case pulse_norm::dc_gain_sps: scale = sum != 0.0 ? (double)sps / sum : 1.0; break;
    }
    std::vector<float> taps(h.size());
    for (size_t i = 0; i < h.size(); i++)
        taps[i] = (float)(h[i] * scale);
    return taps;
}

} // namespace cusat_dsp
