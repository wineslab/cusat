// SPDX-License-Identifier: Apache-2.0
#include "bch_common.cuh"
#include "bch_encode_kernel.cuh"
#include <memory>
namespace cusat_dsp
{
namespace cuda
{
struct bch_bb_scramble_cuda::impl {
    bch_detail::storage tables;
    uint8_t* scramble = nullptr;
    uint8_t* staging = nullptr;
    size_t capacity = 0;
    unsigned k = 0, n = 0;
    ~impl()
    {
        if (scramble)
            cudaFree(scramble);
        if (staging)
            cudaFree(staging);
    }
};
bch_bb_scramble_cuda::bch_bb_scramble_cuda(const uint64_t* flat_lut, const uint64_t* gen_poly,
                                           int gen_degree, const uint8_t* scramble_seq,
                                           int scramble_len, unsigned int k_bytes,
                                           unsigned int n_bytes, unsigned int parity_bytes)
    : state_(nullptr)
{
    if (!k_bytes || !scramble_seq || scramble_len < 0 || unsigned(scramble_len) < k_bytes ||
        gen_degree < 8 || (gen_degree & 7) || parity_bytes != unsigned(gen_degree / 8) ||
        uint64_t(k_bytes) + parity_bytes != n_bytes)
        throw std::invalid_argument("invalid BCH encoder geometry");
    auto s = std::make_unique<impl>();
    s->k = k_bytes;
    s->n = n_bytes;
    s->tables.init(flat_lut, gen_poly, gen_degree, k_bytes);
    bch_detail::checked(cudaMalloc(reinterpret_cast<void**>(&s->scramble), k_bytes));
    bch_detail::checked(cudaMemcpy(s->scramble, scramble_seq, k_bytes, cudaMemcpyHostToDevice));
    state_ = s.release();
}
bch_bb_scramble_cuda::~bch_bb_scramble_cuda() { delete state_; }
unsigned int bch_bb_scramble_cuda::k_bytes() const { return state_->k; }
unsigned int bch_bb_scramble_cuda::n_bytes() const { return state_->n; }
void bch_bb_scramble_cuda::encode_batch(const uint8_t* src, uint8_t* dst, int count,
                                        cudaStream_t stream)
{
    if (count < 0)
        throw std::invalid_argument("negative BCH batch");
    if (!count)
        return;
    if (!src || !dst)
        throw std::invalid_argument("null BCH encoder buffer");
    size_t needed = size_t(count) * state_->k;
    if (needed > state_->capacity) {
        uint8_t* fresh = nullptr;
        bch_detail::checked(cudaMalloc(reinterpret_cast<void**>(&fresh), needed));
        if (state_->staging) {
            auto e = cudaFree(state_->staging);
            if (e != cudaSuccess) {
                cudaFree(fresh);
                bch_detail::checked(e);
            }
        }
        state_->staging = fresh;
        state_->capacity = needed;
    }
    bch_detail::checked(
        cudaMemcpyAsync(state_->staging, src, needed, cudaMemcpyHostToDevice, stream));
    bch_detail::launch<true>(state_->tables.degree, state_->staging, dst, nullptr, state_->scramble,
                             state_->tables, state_->n, count, stream);
}
} // namespace cuda
} // namespace cusat_dsp
