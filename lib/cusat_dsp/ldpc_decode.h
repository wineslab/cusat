/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone LDPC decoder (CUDA GPU only).
 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_DECODE_H
#define INCLUDED_CUSAT_DSP_LDPC_DECODE_H

#include "fec/code_params.h"
#include "ldpc_tables/ldpc_code.h"
#include <cusat_dsp/modcod_types.h>
#include <cstdint>

namespace cusat_dsp { namespace cuda {
class ldpc_decode_cuda;
}}
struct CUstream_st; // forward for cudaStream_t

namespace cusat_dsp {

class ldpc_decode
{
public:
    ldpc_decode(fecframe_size_t framesize,
                code_rate_t rate,
                int max_trials = 25,
                int cuda_max_batch = 64);

    ~ldpc_decode();

    /**
     * @brief Decode a variable-size batch on GPU.
     *
     * @param soft_in    Host pointer: batch_size × n_bits int16_t soft LLRs.
     * @param hard_out   Host pointer: batch_size × k_bytes packed bytes.
     * @param trials_out Host pointer: batch_size ints (remaining trials, <0 = failed).
     * @param batch_size Number of codewords (1..cuda_max_batch).
     */
    void decode_batch_cuda(const int16_t* soft_in, uint8_t* hard_out,
                           int* trials_out, int batch_size);

    /**
     * @brief Decode a batch — device soft input, host hard output.
     *
     * Takes soft LLRs already on device (e.g. from GPU demapper), decodes
     * on GPU, and copies hard decisions + trial counts back to host.
     */
    void decode_batch_cuda_d2h(const int16_t* soft_in_dev, uint8_t* hard_out,
                               int* trials_out, int batch_size);

    /** Get the internal CUDA stream handle (for external event recording). */
    CUstream_st* cuda_stream() const { return d_cuda_stream; }

    unsigned int n_bits() const { return d_n_bits; }
    unsigned int k_bits() const { return d_k_bits; }
    unsigned int n_bytes() const { return d_n_bytes; }
    unsigned int k_bytes() const { return d_k_bytes; }

private:
    unsigned int d_n_bits;
    unsigned int d_n_bytes;
    unsigned int d_k_bits;
    unsigned int d_k_bytes;
    ldpc_code* d_ldpc;

    cuda::ldpc_decode_cuda* d_ldpc_cuda = nullptr;
    CUstream_st* d_cuda_stream = nullptr;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_LDPC_DECODE_H */
