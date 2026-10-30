/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU fused AGC + frequency rotation.
 *
 * Two-kernel pipeline on the caller's CUDA stream:
 *   1. agc_rms_kernel: parallel reduction of |x[n]|², IIR gain update
 *   2. rotate_gain_kernel: per-sample rotation + gain multiply (in-place)
 *
 * Key insight: |rotated|² = |original|², so AGC RMS is computed on raw
 * (pre-rotation) data. The gain is written by kernel 1 and read by kernel 2
 * on the same stream.
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_AGC_ROT_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_AGC_ROT_KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace cusat_dsp {
namespace cuda {

class agc_rot_cuda
{
public:
    /**
     * @param ref       Target RMS amplitude (typically 1.0).
     * @param gain      Initial gain value.
     * @param max_gain  Maximum allowed gain.
     * @param rate      Per-sample IIR rate (time constant ~ 1/rate samples).
     */
    agc_rot_cuda(float ref, float gain, float max_gain, float rate);
    ~agc_rot_cuda();

    /**
     * @brief Launch AGC + rotation kernels on the caller's stream.
     *
     * Operates in-place on samples_um (must be unified/managed memory).
     * After return (and stream sync by caller), the buffer contains
     * rotated + gain-adjusted samples.
     *
     * @param samples_um   Unified memory buffer [N * 2 floats], interleaved re/im.
     * @param N            Number of complex samples.
     * @param phase_start  Starting rotation phase (radians).
     * @param phase_inc    Per-sample phase increment (radians).
     * @param stream       CUDA stream for kernel launches.
     */
    void process(float* samples_um, int N,
                 float phase_start, float phase_inc,
                 cudaStream_t stream);

    /** Read current gain (valid after cudaStreamSynchronize). */
    float gain() const;

    /** Reset gain to a specific value. */
    void reset(float gain);

    /** Query split GPU times (call after stream sync). */
    void query_events();
    /** GPU time for AGC RMS kernel only (microseconds, valid after query_events). */
    uint32_t last_agc_us() const { return d_last_agc_us; }
    /** GPU time for rotation+gain kernel only (microseconds, valid after query_events). */
    uint32_t last_rot_us() const { return d_last_rot_us; }

    /** Enable/disable per-kernel profiling events.
     *  Disabling saves ~15µs per call from eliminated cudaEventRecord calls. */
    void set_profiling(bool enable) { d_profiling = enable; }

private:
    bool d_profiling = true;
    float* d_agc_state_um;    // pinned host memory [4]: {gain, ref, max_gain, rate}
    float* d_partial_sums;    // device memory [AGC_NUM_BLOCKS]: partial reduction sums
    unsigned int* d_done_cnt; // device memory: atomic completion counter
    cudaEvent_t d_evt_start, d_evt_agc_done, d_evt_rot_done;
    uint32_t d_last_agc_us = 0, d_last_rot_us = 0;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_AGC_ROT_KERNEL_CUH */
