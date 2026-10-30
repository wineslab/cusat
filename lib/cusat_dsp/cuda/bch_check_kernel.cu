// SPDX-License-Identifier: Apache-2.0
#include "bch_check_kernel.cuh"
#include "bch_common.cuh"
#include <memory>
namespace cusat_dsp
{
namespace cuda
{
struct bch_check_cuda::impl {
    bch_detail::storage tables;
    unsigned n;
};
bch_check_cuda::bch_check_cuda(const uint64_t* flat_lut, const uint64_t* gen_poly, int gen_degree,
                               unsigned int n_bytes)
    : state_(nullptr)
{
    if (gen_degree < 8 || (gen_degree & 7) || n_bytes <= unsigned(gen_degree / 8))
        throw std::invalid_argument("invalid BCH checker geometry");
    auto s = std::make_unique<impl>();
    s->n = n_bytes;
    s->tables.init(flat_lut, gen_poly, gen_degree, n_bytes);
    state_ = s.release();
}
bch_check_cuda::~bch_check_cuda() { delete state_; }
unsigned int bch_check_cuda::n_bytes() const { return state_->n; }
void bch_check_cuda::check_async(const uint8_t* src, int* dst, int count, cudaStream_t stream)
{
    if (count < 0)
        throw std::invalid_argument("negative BCH batch");
    if (!count)
        return;
    if (!src || !dst)
        throw std::invalid_argument("null BCH checker buffer");
    bch_detail::launch<false>(state_->tables.degree, src, nullptr, dst, nullptr, state_->tables,
                              state_->n, count, stream);
}
} // namespace cuda
} // namespace cusat_dsp
