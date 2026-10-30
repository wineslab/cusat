/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INCLUDED_CUSAT_DSP_PL_SCRAMBLE_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_PL_SCRAMBLE_KERNEL_CUH

#include <cusat_dsp/compat.h>
#include <cuda_runtime.h>

namespace cusat_dsp {
namespace cuda {

/**
 * GPU-accelerated PL scramble/descramble.
 *
 * TX scrambling multiplies each payload symbol by the conjugate of the
 * descrambling sequence; RX descrambling multiplies by the sequence itself.
 */
class pl_scramble_cuda
{
public:
    /**
     * @param descrambling_seq  Host pointer to the descrambling sequence
     *                          (from pl_descrambling_sequence()).
     * @param max_payload_len   Maximum payload length in complex samples.
     */
    pl_scramble_cuda(const complexf* descrambling_seq, int max_payload_len);
    ~pl_scramble_cuda();

    /** TX: payload[i] *= conj(seq[i]) */
    void scramble_inplace(complexf* payload, int payload_len);

private:
    int d_max_len;
    cudaStream_t d_stream;
    float* d_seq_dev;
    float* d_payload_dev;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_PL_SCRAMBLE_KERNEL_CUH */
