/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone LDPC decoder wrapper (CUDA GPU only).
 */

#include "ldpc_decode.h"
#include "ldpc_table_factory.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda/ldpc_decode_kernel.cuh"
#include <cuda_runtime.h>

namespace cusat_dsp {

ldpc_decode::ldpc_decode(fecframe_size_t framesize,
                         code_rate_t rate,
                         int max_trials,
                         int cuda_max_batch)
    : d_ldpc(nullptr)
{
    const fec_params_t fec = get_fec_params(framesize, rate);
    d_k_bits = fec.k_ldpc;
    d_n_bits = fec.n_ldpc;
    d_k_bytes = d_k_bits / 8;
    d_n_bytes = d_n_bits / 8;

    d_ldpc = create_ldpc_table(framesize, rate);

    // Verify CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "FATAL: No CUDA-capable GPU found (err=%d, devices=%d). "
                "A CUDA GPU is required.\n", (int)err, device_count);
        abort();
    }

    d_ldpc_cuda = new cuda::ldpc_decode_cuda(d_ldpc, rate, max_trials, cuda_max_batch);
    cudaStreamCreate((cudaStream_t*)&d_cuda_stream);
    fprintf(stderr, "ldpc_decode: CUDA GPU decoder enabled (max_batch=%d)\n",
            cuda_max_batch);
}

ldpc_decode::~ldpc_decode()
{
    delete d_ldpc;
    delete d_ldpc_cuda;
    if (d_cuda_stream)
        cudaStreamDestroy((cudaStream_t)d_cuda_stream);
}

void ldpc_decode::decode_batch_cuda(const int16_t* soft_in, uint8_t* hard_out,
                                     int* trials_out, int batch_size)
{
    d_ldpc_cuda->decode_batch(soft_in, hard_out, trials_out, batch_size,
                              (cudaStream_t)d_cuda_stream);
}

void ldpc_decode::decode_batch_cuda_d2h(const int16_t* soft_in_dev, uint8_t* hard_out,
                                         int* trials_out, int batch_size)
{
    d_ldpc_cuda->decode_batch_d2h(soft_in_dev, hard_out, trials_out, batch_size,
                                   (cudaStream_t)d_cuda_stream);
}

} // namespace cusat_dsp
