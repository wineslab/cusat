/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Cubic Lagrange Farrow resampler (Farrow 1988; Erup, Gardner
 * & Harris 1993; Rice 2009, 8.4.2).
 *
 * For a position t = n + mu (n integer, mu in [0, 1)) the output is the
 * cubic through x[n-1], x[n], x[n+1], x[n+2] evaluated at mu, in Farrow
 * (Horner) form y = ((c3 mu + c2) mu + c1) mu + c0 with
 *   c0 = x0
 *   c1 = x1 - x-1 / 3 - x0 / 2 - x2 / 6
 *   c2 = (x-1 + x1) / 2 - x0
 *   c3 = (x2 - x-1) / 6 + (x0 - x1) / 2
 * (y(-1) = x-1, y(0) = x0, y(1) = x1, y(2) = x2).
 *
 * Output k is at t_k = k * stride. Within a call the positions are computed
 * in closed form from the last output of the previous call,
 * t = mu_last + j * stride with one exactly rounded fma, so the result does
 * not depend on how the stream is split into calls beyond the rounding of
 * the per-call base (and the GPU class uses the same expression).
 */

#include "farrow_interpolator.h"
#include <algorithm>
#include <cmath>

namespace cusat_dsp {

namespace {

inline complexf farrow_cubic(complexf xm1, complexf x0, complexf x1, complexf x2, float mu)
{
    const float third = 1.0f / 3.0f, sixth = 1.0f / 6.0f;
    complexf c1 = x1 - xm1 * third - x0 * 0.5f - x2 * sixth;
    complexf c2 = (xm1 + x1) * 0.5f - x0;
    complexf c3 = (x2 - xm1) * sixth + (x0 - x1) * 0.5f;
    return ((c3 * mu + c2) * mu + c1) * mu + x0;
}

} // namespace

farrow_interpolator::farrow_interpolator(double stride) : d_stride(stride) { reset(); }

void farrow_interpolator::reset()
{
    d_last_n = 0;
    d_last_mu = 0.0;
    d_have_last = false;
    d_in_abs = 0;
    for (int i = 0; i < HISTORY; i++)
        d_hist[i] = complexf(0.0f, 0.0f);
}

int farrow_interpolator::estimate_output(int n_in) const
{
    return (int)std::ceil((double)n_in / d_stride) + 4;
}

int farrow_interpolator::estimate_input(int n_out) const
{
    return (int)std::ceil((double)n_out * d_stride) + HISTORY + 1;
}

int farrow_interpolator::resample(const complexf* in, int n_in, complexf* out, int max_out,
                                  int& n_consumed)
{
    n_consumed = 0;
    if (n_in < 0)
        n_in = 0;
    if (max_out < 0)
        max_out = 0;
    const double s = d_stride;
    const long long nb = d_have_last ? d_last_n : 0;
    const double mub = d_have_last ? d_last_mu : 0.0;
    const double off = d_have_last ? 1.0 : 0.0;
    const long long a0 = d_in_abs;          // absolute index of in[0]
    const long long end = a0 + n_in;        // one past the last available sample
    // sample at absolute index a (a >= a0 - HISTORY)
    auto x = [&](long long a) -> complexf {
        return a >= a0 ? in[a - a0] : d_hist[a - (a0 - HISTORY)];
    };

    int n_out = 0;
    long long n = 0;
    double mu = 0.0;
    while (true) {
        const double t = std::fma((double)n_out + off, s, mub);
        const double fl = std::floor(t);
        n = nb + (long long)fl;
        mu = t - fl;
        if (n + 2 >= end || n_out >= max_out)
            break;
        out[n_out] = farrow_cubic(x(n - 1), x(n), x(n + 1), x(n + 2), (float)mu);
        n_out++;
    }
    // (n, mu) is now the position of the next output, not yet computed
    const bool clipped = n + 2 < end; // could have produced more
    long long u; // first input sample not consumed
    if (!clipped)
        u = end;
    else
        u = std::min(end, std::max(a0, n + 2)); // history [u-3, u) holds x[n-1 .. n+1]
    // new history: samples u-3 .. u-1
    complexf h[HISTORY];
    for (int i = 0; i < HISTORY; i++)
        h[i] = x(u - HISTORY + i);
    for (int i = 0; i < HISTORY; i++)
        d_hist[i] = h[i];
    n_consumed = (int)(u - a0);
    d_in_abs = u;
    if (n_out > 0) {
        const double t = std::fma((double)(n_out - 1) + off, s, mub);
        const double fl = std::floor(t);
        d_last_n = nb + (long long)fl;
        d_last_mu = t - fl;
        d_have_last = true;
    }
    return n_out;
}

} // namespace cusat_dsp
