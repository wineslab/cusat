// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "cuda_util.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace cusat_dsp
{
namespace cuda
{
namespace bch_detail
{
inline void checked(cudaError_t e) { CUDA_CHECK(e); }
struct storage {
    uint64_t* lut = nullptr;
    uint64_t* shifts = nullptr;
    int degree = 0;
    unsigned bytes = 0, length = 0, chunk = 0;
    storage() = default;
    storage(const storage&) = delete;
    storage& operator=(const storage&) = delete;
    ~storage()
    {
        if (lut)
            cudaFree(lut);
        if (shifts)
            cudaFree(shifts);
    }
    void init(const uint64_t* flat, const uint64_t* gen, int p, unsigned len)
    {
        if (!flat || !gen || p < 8 || p > 248 || (p & 7) || !len ||
            len > std::numeric_limits<unsigned>::max() - 31u || !(gen[0] & 1))
            throw std::invalid_argument("invalid BCH generator/geometry");
        if ((gen[p / 64] & (uint64_t(1) << (p % 64))) == 0)
            throw std::invalid_argument("BCH generator must be monic");
        degree = p;
        bytes = p / 8;
        length = len;
        chunk = (len + 31u) / 32u;
        const int words = (p + 63) / 64;
        std::vector<uint64_t> compact(256 * words);
        for (int b = 0; b < 256; b++)
            for (int w = 0; w < words; w++)
                compact[b * words + w] = flat[b * 4 + w];
        checked(cudaMalloc(reinterpret_cast<void**>(&lut), compact.size() * sizeof(uint64_t)));
        checked(cudaMemcpy(lut, compact.data(), compact.size() * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));
        // powers[e] represents x^e mod g, with the lowest coefficient in bit zero.
        using poly = std::array<uint64_t, 4>;
        size_t max_power = size_t(len) * 8 + p;
        std::vector<poly> powers(max_power + 1);
        powers[0][0] = 1;
        uint64_t mask = (p % 64) ? ((uint64_t(1) << (p % 64)) - 1) : ~uint64_t(0);
        for (size_t e = 1; e <= max_power; e++) {
            poly a = powers[e - 1];
            bool carry = (a[(p - 1) / 64] >> ((p - 1) % 64)) & 1;
            for (int w = words - 1; w >= 0; w--)
                a[w] = (a[w] << 1) | (w ? a[w - 1] >> 63 : 0);
            a[words - 1] &= mask;
            if (carry)
                for (int w = 0; w < words; w++)
                    a[w] ^= gen[w];
            a[words - 1] &= mask;
            powers[e] = a;
        }
        // A chunk remainder must be shifted past all subsequent message bytes.
        // Linearity lets us tabulate that shift independently for each remainder byte.
        std::vector<uint64_t> transform(size_t(32) * bytes * 256 * words);
        for (unsigned lane = 0; lane < 32; lane++) {
            unsigned end = std::min(len, (lane + 1) * chunk);
            size_t shift = size_t(len - end) * 8;
            for (unsigned j = 0; j < bytes; j++)
                for (unsigned b = 1; b < 256; b++) {
                    auto dest = &transform[((size_t(lane) * bytes + j) * 256 + b) * words];
                    for (unsigned bit = 0; bit < 8; bit++)
                        if (b & (1u << bit))
                            for (int w = 0; w < words; w++)
                                dest[w] ^= powers[shift + j * 8 + bit][w];
                }
        }
        checked(cudaMalloc(reinterpret_cast<void**>(&shifts), transform.size() * sizeof(uint64_t)));
        checked(cudaMemcpy(shifts, transform.data(), transform.size() * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));
    }
};
template <int P, bool Encode>
__global__ void process(const uint8_t* input, uint8_t* output, int* valid, const uint8_t* scramble,
                        const uint64_t* table, const uint64_t* transforms, unsigned length,
                        unsigned stride, unsigned chunk, int batch)
{
    constexpr int W = (P + 63) / 64;
    constexpr int B = P / 8;
    __shared__ uint64_t tab[256 * W];
    for (unsigned j = threadIdx.x; j < 256 * W; j += blockDim.x)
        tab[j] = table[j];
    __syncthreads();
    unsigned lane = threadIdx.x & 31;
    unsigned frame = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (frame >= unsigned(batch))
        return;
    // Each lane evaluates a contiguous polynomial chunk.
    uint64_t r[W] = {};
    const uint8_t* src = input + size_t(frame) * length;
    unsigned begin = lane * chunk, end = min(length, begin + chunk);
    for (unsigned i = begin; i < end; i++) {
        unsigned b = src[i];
        if constexpr (Encode) {
            b ^= scramble[i];
            output[size_t(frame) * stride + i] = uint8_t(b);
        }
        unsigned top = unsigned((r[(P - 8) / 64] >> ((P - 8) % 64)) & 255);
        unsigned idx = b ^ top;
#pragma unroll
        for (int w = W - 1; w >= 0; w--)
            r[w] = (r[w] << 8) | (w ? r[w - 1] >> 56 : 0);
        if constexpr (P % 64)
            r[W - 1] &= (uint64_t(1) << (P % 64)) - 1;
#pragma unroll
        for (int w = 0; w < W; w++)
            r[w] ^= tab[idx * W + w];
    }
    // For chunks A and B: rem(A || B) = rem(A * x^(8*|B|)) XOR rem(B).
    // Transform every lane remainder to its final position before the warp XOR.
    uint64_t combined[W] = {};
    const uint64_t* tr = transforms + size_t(lane) * B * 256 * W;
#pragma unroll
    for (int j = 0; j < B; j++) {
        unsigned b = unsigned((r[j / 8] >> ((j % 8) * 8)) & 255);
#pragma unroll
        for (int w = 0; w < W; w++)
            combined[w] ^= tr[(j * 256 + b) * W + w];
    }
#pragma unroll
    for (int w = 0; w < W; w++) {
#pragma unroll
        for (int d = 16; d; d /= 2)
            combined[w] ^= __shfl_xor_sync(0xffffffff, combined[w], d);
    }
    if (!lane) {
        if constexpr (Encode) {
#pragma unroll
            for (int j = 0; j < B; j++)
                output[size_t(frame) * stride + length + j] =
                    uint8_t(combined[(B - 1 - j) / 8] >> (((B - 1 - j) % 8) * 8));
        } else {
            uint64_t any = 0;
#pragma unroll
            for (int w = 0; w < W; w++)
                any |= combined[w];
            valid[frame] = (any == 0);
        }
    }
}
template <bool E>
inline void launch(int p, const uint8_t* src, uint8_t* out, int* valid, const uint8_t* scramble,
                   const storage& s, unsigned stride, int count, cudaStream_t stream)
{
    unsigned blocks = (unsigned(count) + 3) / 4;
#define BCH_CASE(P)                                                                                \
    case P:                                                                                        \
        process<P, E><<<blocks, 128, 0, stream>>>(src, out, valid, scramble, s.lut, s.shifts,      \
                                                  s.length, stride, s.chunk, count);               \
        break
    switch (p) {
        BCH_CASE(8);
        BCH_CASE(16);
        BCH_CASE(24);
        BCH_CASE(32);
        BCH_CASE(40);
        BCH_CASE(48);
        BCH_CASE(56);
        BCH_CASE(64);
        BCH_CASE(72);
        BCH_CASE(80);
        BCH_CASE(88);
        BCH_CASE(96);
        BCH_CASE(104);
        BCH_CASE(112);
        BCH_CASE(120);
        BCH_CASE(128);
        BCH_CASE(136);
        BCH_CASE(144);
        BCH_CASE(152);
        BCH_CASE(160);
        BCH_CASE(168);
        BCH_CASE(176);
        BCH_CASE(184);
        BCH_CASE(192);
        BCH_CASE(200);
        BCH_CASE(208);
        BCH_CASE(216);
        BCH_CASE(224);
        BCH_CASE(232);
        BCH_CASE(240);
        BCH_CASE(248);
    default:
        throw std::invalid_argument("unsupported BCH degree");
    }
#undef BCH_CASE
    checked(cudaGetLastError());
}
} // namespace bch_detail
} // namespace cuda
} // namespace cusat_dsp
