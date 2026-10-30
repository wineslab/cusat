// SPDX-License-Identifier: Apache-2.0
// GPU BCH correctness and steady-state timing, including the encoder H2D copy.
#include "cuda/bch_check_kernel.cuh"
#include "cuda/bch_encode_kernel.cuh"
#include "fec/bch_code.h"
#include "fec/code_params.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace cusat_dsp;
static void ck(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void require(bool ok, const char* what)
{
    if (!ok)
        throw std::runtime_error(what);
}
struct buffers {
    uint8_t* input = nullptr;
    uint8_t* encoded = nullptr;
    int* valid = nullptr;
    cudaStream_t stream{};
    buffers(size_t k, size_t n, int batch)
    {
        ck(cudaMallocHost(&input, k * batch));
        ck(cudaMalloc(&encoded, n * batch));
        ck(cudaMalloc(&valid, sizeof(int) * batch));
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~buffers()
    {
        cudaStreamSynchronize(stream);
        cudaFreeHost(input);
        cudaFree(encoded);
        cudaFree(valid);
        cudaStreamDestroy(stream);
    }
};
static const uint64_t* table(const bch_code_t& code) { return code.encoder_lut.data()->data(); }
static void verify(const bch_code_t& code, int batch, int pattern)
{
    const size_t k = code.k_bits / 8, n = code.n_bits / 8;
    buffers mem(k, n, batch);
    std::mt19937 random(unsigned(code.n_bits + batch + pattern));
    std::vector<uint8_t> scramble(k), expected(n * batch), scrambled(k), actual(n * batch);
    for (auto& b : scramble)
        b = random();
    for (size_t i = 0; i < k * batch; ++i)
        mem.input[i] = pattern == 0   ? 0
                       : pattern == 1 ? 255
                       : pattern == 2 ? uint8_t(0x80 >> (i % 8))
                                      : uint8_t(random());
    cuda::bch_bb_scramble_cuda encoder(table(code), code.generator.data(), code.parity_bits,
                                       scramble.data(), int(k), k, n, n - k);
    cuda::bch_check_cuda checker(table(code), code.generator.data(), code.parity_bits, n);
    for (int f = 0; f < batch; ++f) {
        for (size_t j = 0; j < k; ++j)
            scrambled[j] = mem.input[size_t(f) * k + j] ^ scramble[j];
        bch_encode(code, scrambled.data(), expected.data() + size_t(f) * n);
    }
    // Queue both operations without a device synchronization between them.
    encoder.encode_batch(mem.input, mem.encoded, 1, mem.stream);
    encoder.encode_batch(mem.input, mem.encoded, batch, mem.stream);
    checker.check_async(mem.encoded, mem.valid, batch, mem.stream);
    std::vector<int> flags(batch);
    ck(cudaMemcpyAsync(actual.data(), mem.encoded, actual.size(), cudaMemcpyDeviceToHost,
                       mem.stream));
    ck(cudaMemcpyAsync(flags.data(), mem.valid, sizeof(int) * batch, cudaMemcpyDeviceToHost,
                       mem.stream));
    ck(cudaStreamSynchronize(mem.stream));
    require(actual == expected, "GPU BCH differs from CPU systematic codeword");
    require(std::all_of(flags.begin(), flags.end(), [](int v) { return v == 1; }),
            "valid BCH codeword rejected");
    // A single-bit corruption is always detectable, across message and parity.
    const size_t positions[] = {0, k - 1, k, n - 1};
    for (size_t pos : positions) {
        actual = expected;
        for (int f = 0; f < batch; ++f)
            if (f % 3 != 1)
                actual[size_t(f) * n + pos] ^= uint8_t(1u << (f % 8));
        ck(cudaMemcpyAsync(mem.encoded, actual.data(), actual.size(), cudaMemcpyHostToDevice,
                           mem.stream));
        checker.check_async(mem.encoded, mem.valid, batch, mem.stream);
        ck(cudaMemcpyAsync(flags.data(), mem.valid, sizeof(int) * batch, cudaMemcpyDeviceToHost,
                           mem.stream));
        ck(cudaStreamSynchronize(mem.stream));
        for (int f = 0; f < batch; ++f)
            require(flags[f] == (f % 3 == 1), "BCH corruption check incorrect");
    }
    encoder.encode_batch(nullptr, nullptr, 0, mem.stream);
    checker.check_async(nullptr, nullptr, 0, mem.stream);
}
template <class F> static double median_us(F call, cudaStream_t stream, int repeats)
{
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin));
    ck(cudaEventCreate(&end));
    for (int i = 0; i < 5; ++i)
        call();
    ck(cudaStreamSynchronize(stream));
    std::vector<double> samples;
    for (int sample = 0; sample < 5; ++sample) {
        ck(cudaEventRecord(begin, stream));
        for (int i = 0; i < repeats; ++i)
            call();
        ck(cudaEventRecord(end, stream));
        ck(cudaEventSynchronize(end));
        float ms;
        ck(cudaEventElapsedTime(&ms, begin, end));
        samples.push_back(ms * 1000.0 / repeats);
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}
static void benchmark(const bch_code_t& code, int batch)
{
    const size_t k = code.k_bits / 8, n = code.n_bits / 8;
    buffers mem(k, n, batch);
    std::mt19937 random(4242);
    std::vector<uint8_t> scramble(k);
    for (auto& b : scramble)
        b = random();
    for (size_t j = 0; j < k * batch; ++j)
        mem.input[j] = random();
    cuda::bch_bb_scramble_cuda encoder(table(code), code.generator.data(), code.parity_bits,
                                       scramble.data(), int(k), k, n, n - k);
    cuda::bch_check_cuda checker(table(code), code.generator.data(), code.parity_bits, n);
    encoder.encode_batch(mem.input, mem.encoded, batch, mem.stream);
    ck(cudaStreamSynchronize(mem.stream));
    const double encode = median_us(
        [&] { encoder.encode_batch(mem.input, mem.encoded, batch, mem.stream); }, mem.stream, 30);
    const double check = median_us(
        [&] { checker.check_async(mem.encoded, mem.valid, batch, mem.stream); }, mem.stream, 30);
    printf("%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n", code.parity_bits, code.n_bits, batch, encode, check,
           batch * double(code.k_bits) / (encode * 1000),
           batch * double(code.n_bits) / (check * 1000));
    fflush(stdout);
}
int main(int argc, char** argv)
{
    try {
        const bool verify_only = argc > 1 && !strcmp(argv[1], "--verify-only");
        const bool bench_only = argc > 1 && !strcmp(argv[1], "--bench-only");
        const bool quick = argc > 1 && !strcmp(argv[1], "--quick");
        cudaDeviceProp prop;
        ck(cudaGetDeviceProperties(&prop, 0));
        fprintf(stderr, "GPU %s sm_%d%d\n", prop.name, prop.major, prop.minor);
        if (!bench_only) {
            std::set<std::tuple<int, int, int>> seen;
            int cases = 0;
            for (fecframe_size_t fs : {FECFRAME_NORMAL, FECFRAME_SHORT}) {
                for (int r = 0; r < RATE_COUNT; ++r) {
                    if (!has_fec_params(fs, code_rate_t(r)))
                        continue;
                    const auto params = get_fec_params(fs, code_rate_t(r));
                    if ((params.k_bch % 8) || (params.n_bch % 8))
                        continue;
                    if (!seen.emplace(fs, params.n_bch, params.t_bch).second)
                        continue;
                    auto code = make_bch_code(fs, params.n_bch, params.t_bch);
                    for (int pattern = 0; pattern < (quick ? 1 : 4); ++pattern)
                        verify(code, 33, pattern);
                    ++cases;
                }
            }
            const auto code = make_bch_code(FECFRAME_NORMAL, 58320, 8);
            for (int batch : {1, 2, 3, 4, 7, 16, 31, 32, 64, 129})
                verify(code, batch, 3);
            fprintf(stderr,
                    "PASS: %d unique byte-aligned BCH geometries, patterns, batch boundaries, "
                    "systematic parity and corruption checks\n",
                    cases);
        }
        if (!verify_only && !quick) {
            puts("parity_bits,n_bits,batch,encode_us,check_us,encode_Gbps,check_Gbps");
            for (const auto& [fs, n, t] :
                 std::vector<std::tuple<fecframe_size_t, int, int>>{{FECFRAME_NORMAL, 58320, 8},
                                                                    {FECFRAME_NORMAL, 54000, 10},
                                                                    {FECFRAME_NORMAL, 48600, 12},
                                                                    {FECFRAME_SHORT, 14400, 12}})
                for (int batch : {1, 4, 16, 32, 64, 256})
                    benchmark(make_bch_code(fs, n, t), batch);
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
