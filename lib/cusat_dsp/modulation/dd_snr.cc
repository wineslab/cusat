/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "modulation/dd_snr.h"
#include <cmath>

namespace cusat_dsp {

namespace {

/* Error-free input is reported as 60 dB; non-finite input as 0 (invalid),
 * which callers reject as "no estimate". */
float snr_ratio(float signal, float error)
{
    if (!std::isfinite(signal) || !std::isfinite(error))
        return 0.0f;
    constexpr float MAX_SNR = 1e6f;
    return error * MAX_SNR <= signal ? MAX_SNR : signal / error;
}

} // namespace

float qpsk_snr_dd(const complexf* sym, int n_sym)
{
    const float a = (float)M_SQRT1_2;
    const float* x = reinterpret_cast<const float*>(sym);
    float signal = 0.0f;
    float error = 0.0f;
    for (int i = 0; i < n_sym; i++) {
        const float d_re = x[2 * i] < 0 ? -a : a;
        const float d_im = x[2 * i + 1] < 0 ? -a : a;
        signal += d_re * d_re + d_im * d_im;
        const float e_re = x[2 * i] - d_re;
        const float e_im = x[2 * i + 1] - d_im;
        error += e_re * e_re;
        error += e_im * e_im;
    }
    return snr_ratio(signal, error);
}

float constellation_snr_dd(const constellation_t& c, const complexf* sym, int n_sym, int stride)
{
    float signal = 0.0f;
    float error = 0.0f;
    for (int i = 0; i < n_sym; i += stride) {
        const complexf decision = c.points[nearest_label(c, sym[i])];
        signal += std::norm(decision);
        error += std::norm(sym[i] - decision);
    }
    return snr_ratio(signal, error);
}

} // namespace cusat_dsp
