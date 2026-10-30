// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace cusat_dsp
{
namespace cuda
{
class bch_check_cuda
{
  public:
    bch_check_cuda(const uint64_t* flat_lut, const uint64_t* gen_poly, int gen_degree,
                   unsigned int n_bytes);
    ~bch_check_cuda();
    void check_async(const uint8_t* codewords_dev, int* valid_out_dev, int batch_size,
                     cudaStream_t stream);
    unsigned int n_bytes() const;
    bch_check_cuda(const bch_check_cuda&) = delete;
    bch_check_cuda& operator=(const bch_check_cuda&) = delete;

  private:
    struct impl;
    impl* state_;
};
} // namespace cuda
} // namespace cusat_dsp
