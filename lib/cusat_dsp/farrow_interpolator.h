/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Farrow cubic interpolator with NCO-driven arbitrary rate conversion.
 * Supports both decimation (stride > 1) and interpolation (stride < 1).
 *
 * Output k (counted from the first sample ever fed) is the cubic Lagrange
 * interpolation of the input at position t_k = k * stride (input samples,
 * the first input sample being position 0): constant delay 0, so at integer
 * positions the output equals the input sample. Positions before the first
 * input sample read zeros.
 */

#ifndef INCLUDED_CUSAT_DSP_FARROW_INTERPOLATOR_H
#define INCLUDED_CUSAT_DSP_FARROW_INTERPOLATOR_H

#include <cusat_dsp/compat.h>
#include <vector>

namespace cusat_dsp {

class farrow_interpolator {
public:
    /**
     * @brief Construct a Farrow cubic interpolator with NCO.
     *
     * @param stride NCO stride = input_rate / output_rate.
     *               > 1 for decimation (RX: hw_sps / internal_sps).
     *               < 1 for interpolation (TX: internal_sps / hw_sps).
     */
    farrow_interpolator(double stride);

    /**
     * @brief Resample a block of input samples.
     *
     * Consumes input samples and produces output samples at the rate
     * determined by the stride. The NCO state persists across calls.
     *
     * Unless max_out clips the run, the ENTIRE buffer is consumed
     * (n_consumed == n_in): the tail is retained in an internal prefix so
     * back-to-back buffers resample seamlessly with no waveform loss.
     * If max_out clips, n_consumed < n_in and the caller must re-feed the
     * unconsumed remainder.
     *
     * @param in         Input sample buffer.
     * @param n_in       Number of input samples available.
     * @param out        Output sample buffer (caller-allocated).
     * @param max_out    Maximum number of output samples to produce.
     * @param n_consumed Set to the number of input samples consumed.
     * @return Number of output samples written.
     */
    int resample(const complexf* in, int n_in,
                 complexf* out, int max_out,
                 int& n_consumed);

    /**
     * @brief Estimate how many output samples n_in input samples will produce.
     */
    int estimate_output(int n_in) const;

    /**
     * @brief Estimate how many input samples are needed for n_out outputs.
     */
    int estimate_input(int n_out) const;

    /**
     * @brief Reset the NCO accumulator and history.
     */
    void reset();

    /**
     * @brief Get the current stride.
     */
    double stride() const { return d_stride; }

    /**
     * @brief Update the stride (e.g., for dynamic rate adjustment).
     */
    void set_stride(double stride) { d_stride = stride; }

    /// Input samples retained between calls (lookback of the cubic).
    static constexpr int HISTORY = 3;

private:
    double d_stride;
    // position of the last output produced: integer part relative to the
    // absolute input index, fractional part in [0, 1)
    long long d_last_n = 0;
    double d_last_mu = 0.0;
    bool d_have_last = false;
    long long d_in_abs = 0;            // absolute index of the next input sample
    complexf d_hist[HISTORY];          // samples d_in_abs - 3 .. d_in_abs - 1
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FARROW_INTERPOLATOR_H */
