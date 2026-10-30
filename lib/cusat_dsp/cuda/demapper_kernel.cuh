/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA demapper: fused soft demap + column-row deinterleave.
 *
 * One thread per symbol. Each thread reads one complex symbol, computes soft
 * LLRs, and writes them to deinterleaved positions in the output buffer.
 * Supports QPSK (formula), 8PSK (formula), 16APSK, 32APSK, and 64APSK
 * (min-distance search against constellation LUT in __constant__ memory).
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_DEMAPPER_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_DEMAPPER_KERNEL_CUH

#include <cusat_dsp/modcod_types.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cusat_dsp {
namespace cuda {

class demapper_cuda
{
public:
    /**
     * @param constellation  MOD_QPSK, MOD_8PSK, MOD_16APSK, MOD_32APSK, or MOD_64APSK.
     * @param rate           Code rate (affects constellation geometry and interleave).
     * @param framesize      FECFRAME_NORMAL, FECFRAME_SHORT, or FECFRAME_MEDIUM.
     * @param max_batch      Maximum batch size (pre-allocates device buffers).
     */
    demapper_cuda(modulation_t constellation,
                  code_rate_t rate,
                  fecframe_size_t framesize,
                  int max_batch = 64);

    ~demapper_cuda();

    /**
     * @brief Demap + deinterleave a batch of XFECFRAME symbols on GPU.
     *
     * H2D copies symbols, launches fused kernel, synchronizes.
     * After return, soft_out_dev() points to valid int16 LLRs on device.
     *
     * @param sym_in_host  Host pointer: batch × xfecframe_len × 2 floats (re,im pairs).
     * @param batch_size   Number of codewords (1..max_batch).
     * @param noise_var           Noise energy (used for QPSK scaling).
     * @param llr_gain    LLR gain (used for 8PSK/APSK, typically 4/noise_var).
     */
    void demap_batch(const float* sym_in_host, int batch_size,
                     float noise_var, float llr_gain);

    /**
     * @brief Demap + deinterleave using an external CUDA stream, NO sync.
     *
     * Same as demap_batch but uses the caller's stream and does not
     * synchronize. The caller must keep sym_in_host stable until the stream
     * reaches the copy, and must ensure completion before reading soft_out_dev().
     */
    void demap_batch_async(const float* sym_in_host, int batch_size,
                           float noise_var, float llr_gain, cudaStream_t stream);

    /** Effective (possibly capped) llr_gain actually applied for a requested
     *  llr_gain — exposed for calibration/diagnostics. */
    float capped_llr_gain(float llr_gain) const;

    /** Device pointer to deinterleaved soft LLRs [max_batch * fecframe_len]. */
    int16_t* soft_out_dev() const { return d_soft_out_dev; }

    int fecframe_len() const { return d_fecframe_len; }
    int xfecframe_len() const { return d_xfecframe_len; }
    int n_mod() const { return d_n_mod; }

private:
    int d_n_mod;
    int d_fecframe_len, d_xfecframe_len, d_interleaved_len;
    int d_max_batch;
    int d_n_points;              // constellation size (4, 8, 16, 32, 64, 128)
    unsigned int d_bit_index[8];   // deinterleave column offsets
    float d_llr_dmax = 0.0f;     // max over (point,bit) nearest opposite-bit d^2
    float d_llr_target = 0.0f;   // target max |LLR| in counts; 0 = cap disabled

    // Device buffers
    float*  d_sym_dev;           // [max_batch * xfecframe_len * 2] (device)
    int16_t* d_soft_out_dev;     // [max_batch * fecframe_len]
    float*  d_const_re_dev;      // [n_points] per-instance constellation I
    float*  d_const_im_dev;      // [n_points] per-instance constellation Q
    cudaStream_t d_stream;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_DEMAPPER_KERNEL_CUH */
