// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace cusat_dsp
{
namespace cuda
{
// Calls on one instance must be sequenced, including calls on different CUDA streams.
// Independent instances may run concurrently. Growing input capacity can synchronize
// through CUDA allocation; subsequent calls reuse the existing capacity.
class bch_bb_scramble_cuda
{
  public:
    bch_bb_scramble_cuda(const uint64_t* flat_lut, const uint64_t* gen_poly, int gen_degree,
                         const uint8_t* scramble_seq, int scramble_len, unsigned int k_bytes,
                         unsigned int n_bytes, unsigned int parity_bytes);
    ~bch_bb_scramble_cuda();
    void encode_batch(const uint8_t* bbframes_host, uint8_t* bch_out_device, int batch_size,
                      cudaStream_t stream);
    unsigned int k_bytes() const;
    unsigned int n_bytes() const;
    bch_bb_scramble_cuda(const bch_bb_scramble_cuda&) = delete;
    bch_bb_scramble_cuda& operator=(const bch_bb_scramble_cuda&) = delete;

  private:
    struct impl;
    impl* state_;
};
} // namespace cuda
} // namespace cusat_dsp
