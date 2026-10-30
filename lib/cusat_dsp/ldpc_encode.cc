/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * IRA LDPC encoder per ETSI EN 302 307 Annex C.
 *
 * Algorithm:
 *   1. p[0..R-1] = 0
 *   2. For each info bit i (0..K-1):
 *        For each accumulator address a[j] from table:
 *          p[a[j]] ^= info[i]
 *      (addresses computed via circulant: a[j] = (base[j] + (i%M)*q) % R)
 *   3. p[i] ^= p[i-1] for i = 1..R-1  (accumulate)
 *
 * Uses the same flat ldpc_code tables as the decoder.
 */

#include "ldpc_encode.h"
#include "ldpc_table_factory.h"
#include <cassert>
#include <cstring>

#include "cuda/ldpc_encode_kernel.cuh"

namespace cusat_dsp {

ldpc_encode::ldpc_encode(fecframe_size_t framesize,
                         code_rate_t rate)
    : d_ldpc_cuda(nullptr)
{
    const fec_params_t fec = get_fec_params(framesize, rate);
    d_k_bits = fec.k_ldpc;
    d_n_bits = fec.n_ldpc;
    d_k_bytes = d_k_bits / 8;
    d_n_bytes = d_n_bits / 8;
    d_parity_bits = d_n_bits - d_k_bits;
    d_ldpc = create_ldpc_table(framesize, rate);
    assert(d_ldpc != nullptr);

    d_ldpc_cuda = new cuda::ldpc_encode_cuda(d_ldpc, d_k_bits, d_parity_bits);
}

ldpc_encode::~ldpc_encode()
{
    delete d_ldpc_cuda;
    delete d_ldpc;
}

void ldpc_encode::encode(const uint8_t* info_in, uint8_t* codeword_out)
{
    d_ldpc_cuda->encode(info_in, codeword_out);
}

void ldpc_encode::encode_to_device(const uint8_t* info_packed_host,
                                    uint8_t* fecframe_device,
                                    void* cuda_stream)
{
    d_ldpc_cuda->encode_to_device(
        info_packed_host, fecframe_device,
        static_cast<cudaStream_t>(cuda_stream));
}

void ldpc_encode::encode_batch_to_device(const uint8_t* info_packed_host,
                                          uint8_t* fecframes_device,
                                          int batch_size,
                                          void* cuda_stream)
{
    d_ldpc_cuda->encode_batch_to_device(
        info_packed_host, fecframes_device, batch_size,
        static_cast<cudaStream_t>(cuda_stream));
}

void ldpc_encode::encode_batch_d2d(const uint8_t* info_packed_device,
                                    uint8_t* fecframes_device,
                                    int batch_size,
                                    void* cuda_stream)
{
    d_ldpc_cuda->encode_batch_d2d(
        info_packed_device, fecframes_device, batch_size,
        static_cast<cudaStream_t>(cuda_stream));
}

} // namespace cusat_dsp
