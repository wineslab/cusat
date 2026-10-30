/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Helpers shared by the Farrow tests.
 */

#ifndef INCLUDED_FARROW_TEST_UTIL_H
#define INCLUDED_FARROW_TEST_UTIL_H

#include "cusat_dsp/rrc_filter.h"
#include <cusat_dsp/compat.h>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace farrow_test {

using ::complexf;

inline std::vector<complexf> random_signal(size_t n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<complexf> x(n);
    for (auto& v : x)
        v = complexf(g(rng), g(rng));
    return x;
}

/* random call sizes 1..max_call */
inline std::vector<int> random_partition(size_t n, int max_call, unsigned seed)
{
    std::mt19937 rng(seed);
    std::vector<int> p;
    size_t done = 0;
    while (done < n) {
        // mostly small, some large calls
        int c = (rng() % 4 == 0) ? 1 + (int)(rng() % max_call) : 1 + (int)(rng() % 64);
        c = (int)std::min<size_t>(c, n - done);
        p.push_back(c);
        done += c;
    }
    return p;
}

/* Max |a - b| / rms(b) over the common length. */
inline double rel_error(const std::vector<complexf>& a, const std::vector<complexf>& b, size_t n)
{
    double e = 0.0, p = 0.0;
    for (size_t i = 0; i < n; i++) {
        e = std::max(e, (double)std::abs(a[i] - b[i]));
        p += std::norm(b[i]);
    }
    return n ? e / std::sqrt(p / n) : 0.0;
}

/* RRC (alpha) QPSK at 2 samples/symbol, unit-energy pulse */
inline std::vector<complexf> rrc_qpsk(size_t n_sym, double alpha, unsigned seed,
                                      std::vector<complexf>& syms)
{
    std::mt19937 rng(seed);
    const float a = (float)M_SQRT1_2;
    syms.resize(n_sym);
    for (auto& s : syms)
        s = complexf((rng() & 1) ? a : -a, (rng() & 1) ? a : -a);
    const std::vector<float> h = cusat_dsp::rrc_pulse(2, alpha, 16, cusat_dsp::pulse_norm::unit_energy);
    std::vector<complexf> x(2 * n_sym + h.size(), complexf(0, 0));
    for (size_t k = 0; k < n_sym; k++)
        for (size_t i = 0; i < h.size(); i++)
            x[2 * k + i] += syms[k] * h[i];
    return x;
}

/* Matched filter (unit-energy RRC at 2 sps) sampled at symbol instants */
inline std::vector<complexf> matched_symbols(const std::vector<complexf>& x, double alpha, size_t n_sym)
{
    const std::vector<float> h = cusat_dsp::rrc_pulse(2, alpha, 16, cusat_dsp::pulse_norm::unit_energy);
    const size_t d = h.size() - 1; // TX + RX delay, samples
    std::vector<complexf> y(n_sym);
    for (size_t k = 0; k < n_sym; k++) {
        complexf acc(0, 0);
        const size_t c = 2 * k + d;
        for (size_t i = 0; i < h.size(); i++)
            if (c >= i && c - i < x.size())
                acc += x[c - i] * h[i];
        y[k] = acc;
    }
    return y;
}

} // namespace farrow_test

#endif
