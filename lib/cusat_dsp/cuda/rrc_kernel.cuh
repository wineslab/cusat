/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INCLUDED_CUSAT_DSP_RRC_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_RRC_KERNEL_CUH

#include <cuda_runtime.h>

namespace cusat_dsp {
namespace cuda {

/**
 * GPU-accelerated RRC polyphase pulse shaper.
 *
 * The caller is responsible for building the symbol buffer with history
 * prepended (same as the CPU path). This class manages device memory
 * and kernel launches.
 */
class rrc_cuda
{
public:
    /**
     * @param sps          Samples per symbol.
     * @param sub_taps     Taps per polyphase sub-filter.
     * @param history_len  Number of history symbols (sub_taps - 1).
     * @param interleaved_taps  Host pointer to reversed+interleaved taps
     *                          (sub_taps * sps floats).
     */
    rrc_cuda(int sps, int sub_taps, int history_len,
             const float* interleaved_taps);
    ~rrc_cuda();

    /**
     * @brief Pulse-shape symbols on GPU.
     *
     * @param symbol_buf   Host pointer to [history_len + n_symbols] complex
     *                     values (interleaved re/im floats).
     * @param n_symbols    Number of new input symbols (excludes history).
     * @param samples_out  Host pointer for output (n_symbols * sps complex values).
     * @return Number of output samples.
     */
    unsigned int pulse_shape(const float* symbol_buf,
                             unsigned int n_symbols,
                             float* samples_out);

private:
    void ensure_buffers(unsigned int n_symbols);

    int d_sps;
    int d_sub_taps;
    int d_history_len;
    cudaStream_t d_stream;

    float* d_symbols_dev;
    float* d_samples_dev;
    unsigned int d_alloc_symbols;
    unsigned int d_alloc_samples;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_RRC_KERNEL_CUH */
