/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU Farrow cubic interpolator with NCO-driven arbitrary rate conversion.
 *
 * Each output sample is computed by one CUDA thread: compute basepoint and mu
 * from the NCO stride, fetch 4 input samples, evaluate the cubic Horner
 * polynomial. Embarrassingly parallel — no inter-thread dependencies.
 *
 * Same semantics, positions and polynomial as the CPU farrow_interpolator
 * (output k at input position k * stride, delay 0). The output count and
 * the consumed count are computed on the host; resample() enqueues the
 * kernel and the update of the retained history on `stream` and then
 * synchronises the stream, so `out` and the returned count are valid when
 * it returns.
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_FARROW_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_FARROW_KERNEL_CUH

#include <cuda_runtime.h>

namespace cusat_dsp {
namespace cuda {

class farrow_cuda {
public:
    /**
     * @brief Construct a GPU Farrow interpolator.
     *
     * @param stride  NCO stride = input_rate / output_rate.
     *                > 1 for decimation (RX), < 1 for interpolation (TX).
     * @param max_out Maximum output samples per resample() call.
     *                Used to pre-allocate device workspace.
     */
    farrow_cuda(double stride, int max_out);
    ~farrow_cuda();

    /**
     * @brief Resample a block of input samples on the GPU.
     *
     * The kernel is launched on the given stream. After cudaStreamSynchronize,
     * `out` contains the resampled data and `n_consumed` is set.
     *
     * Unless max_out clips the run, the ENTIRE buffer is consumed
     * (n_consumed == n_in): the tail is retained in an internal prefix so
     * back-to-back buffers resample seamlessly with no waveform loss.
     * If max_out clips, n_consumed < n_in and the caller must re-feed the
     * unconsumed remainder.
     *
     * Input and output buffers can be unified memory, pinned memory, or
     * device memory.
     *
     * @param in         Input samples.
     * @param n_in       Number of input samples.
     * @param out        Output samples (caller-allocated, >= max_out).
     * @param max_out    Max output samples to produce.
     * @param n_consumed [out] Number of input samples consumed (valid after sync).
     * @param stream     CUDA stream.
     * @return Number of output samples written (valid after sync; also
     *         written to unified memory for immediate CPU read after sync).
     */
    int resample(const float* in, int n_in,
                 float* out, int max_out,
                 int& n_consumed, cudaStream_t stream);

    /** Get/set stride. */
    double stride() const { return d_stride; }
    void set_stride(double s) { d_stride = s; }

    /// Prefix length: last PREFIX consumed input samples are retained so
    /// the whole input can always be consumed. The carried accumulator is
    /// > -(2 + stride) and the cubic needs 2 samples of lookback below the
    /// basepoint, so 16 covers strides up to ~12. (Public: the kernel
    /// needs it for index mapping.)
    static constexpr int PREFIX = 16;

    /// Input samples retained between calls (lookback of the cubic).
    static constexpr int HISTORY = 3;

private:
    double d_stride;
    int d_max_out;
    long long d_last_n = 0;   // position of the last output (integer part)
    double d_last_mu = 0.0;   // ... and fractional part in [0, 1)
    bool d_have_last = false;
    long long d_in_abs = 0;   // absolute index of the next input sample
    float* d_hist_dev[2] = { nullptr, nullptr }; // [2 * HISTORY] floats each
    int d_hist_cur = 0;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_FARROW_KERNEL_CUH */
