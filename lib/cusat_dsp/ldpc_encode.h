/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * IRA (Irregular Repeat-Accumulate) LDPC encoder per ETSI EN 302 307 Annex C.
 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_ENCODE_H
#define INCLUDED_CUSAT_DSP_LDPC_ENCODE_H

#include "fec/code_params.h"
#include "ldpc_tables/ldpc_code.h"
#include <cusat_dsp/modcod_types.h>
#include <cstdint>

namespace cusat_dsp {
namespace cuda { class ldpc_encode_cuda; }

class ldpc_encode
{
public:
    ldpc_encode(fecframe_size_t framesize,
                code_rate_t rate);
    ~ldpc_encode();

    /**
     * @brief Encode one LDPC codeword.
     *
     * Input: k_bytes of packed info bits (MSB first).
     * Output: n_bytes of packed codeword bits (systematic: info + parity, MSB first).
     */
    void encode(const uint8_t* info_in, uint8_t* codeword_out);

    /**
     * @brief Encode to a device-resident FECFRAME buffer.
     *
     * Uses the caller's CUDA stream and does not synchronize.
     *
     * @param info_packed_host  Host pointer to packed info bits (K/8 bytes).
     * @param fecframe_device   Device pointer for output (N/8 bytes).
     * @param cuda_stream       CUDA stream (cudaStream_t cast to void*).
     */
    void encode_to_device(const uint8_t* info_packed_host,
                          uint8_t* fecframe_device,
                          void* cuda_stream);

    /**
     * @brief Encode N codewords in a single batched GPU launch.
     *
     * All codewords must use the same LDPC code.
     * Uses the caller's CUDA stream and does not synchronize.
     *
     * @param info_packed_host  Host pointer to batch_size × K/8 contiguous bytes.
     * @param fecframes_device  Device pointer to batch_size × N/8 contiguous bytes.
     * @param batch_size        Number of codewords.
     * @param cuda_stream       CUDA stream (cudaStream_t cast to void*).
     */
    void encode_batch_to_device(const uint8_t* info_packed_host,
                                uint8_t* fecframes_device,
                                int batch_size,
                                void* cuda_stream);

    /**
     * @brief Encode N codewords from device-resident info buffers (D2D).
     *
     * Same as encode_batch_to_device but reads from device memory.
     * Use after GPU BCH where BCH output is already on device.
     */
    void encode_batch_d2d(const uint8_t* info_packed_device,
                          uint8_t* fecframes_device,
                          int batch_size,
                          void* cuda_stream);

    unsigned int n_bits() const { return d_n_bits; }
    unsigned int k_bits() const { return d_k_bits; }
    unsigned int n_bytes() const { return d_n_bytes; }
    unsigned int k_bytes() const { return d_k_bytes; }

private:
    unsigned int d_n_bits;
    unsigned int d_k_bits;
    unsigned int d_n_bytes;
    unsigned int d_k_bytes;
    unsigned int d_parity_bits; // R = N - K
    ldpc_code* d_ldpc;
    cuda::ldpc_encode_cuda* d_ldpc_cuda;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_LDPC_ENCODE_H */
