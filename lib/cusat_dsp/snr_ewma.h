/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>

namespace cusat_dsp {

/**
 * Exponentially weighted SNR statistics in the dB domain.
 *
 * The mean is used as the slowly varying ACM channel estimate.  The standard
 * deviation is deliberately computed from the same unfiltered input samples,
 * rather than from an already-clipped signal, so abrupt fades and estimator
 * excursions remain visible in telemetry.
 */
class snr_ewma
{
public:
    explicit snr_ewma(float alpha = 0.1f) : d_alpha(alpha) {}

    void reset()
    {
        d_initialized = false;
        d_mean = 0.0f;
        d_second_moment = 0.0f;
    }

    bool update(float sample)
    {
        if (!std::isfinite(sample))
            return false;

        if (!d_initialized) {
            d_mean = sample;
            d_second_moment = sample * sample;
            d_initialized = true;
        } else {
            d_mean += d_alpha * (sample - d_mean);
            d_second_moment +=
                d_alpha * (sample * sample - d_second_moment);
        }
        return true;
    }

    bool initialized() const { return d_initialized; }
    float mean() const { return d_mean; }

    float stddev() const
    {
        const float variance = d_second_moment - d_mean * d_mean;
        return variance > 0.0f ? std::sqrt(variance) : 0.0f;
    }

private:
    float d_alpha;
    bool d_initialized = false;
    float d_mean = 0.0f;
    float d_second_moment = 0.0f;
};

} // namespace cusat_dsp
