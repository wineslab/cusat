/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU Farrow resampler.
 */

#include "cusat_dsp/cuda/farrow_kernel.cuh"
#include "cusat_dsp/farrow_interpolator.h"
#include "farrow_test_util.h"
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>

using namespace cusat_dsp;
using namespace farrow_test;

namespace {

std::vector<complexf> run_gpu(double stride, const std::vector<complexf>& x, const std::vector<int>& part,
                              bool& all_consumed, bool device_input)
{
    cuda::farrow_cuda f(stride, (int)(65536 / stride) + 16);
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    float *din = nullptr, *dout = nullptr;
    const size_t max_out = (size_t)(65536 / stride) + 16;
    if (device_input) {
        cudaMalloc(&din, x.size() * sizeof(complexf));
        cudaMemcpy(din, x.data(), x.size() * sizeof(complexf), cudaMemcpyHostToDevice);
    } else {
        cudaMallocHost(&din, x.size() * sizeof(complexf));
        memcpy(din, x.data(), x.size() * sizeof(complexf));
    }
    cudaMallocHost(&dout, max_out * sizeof(complexf));
    std::vector<complexf> y;
    size_t pos = 0;
    all_consumed = true;
    for (int c : part) {
        const int mo = (int)std::ceil(c / stride) + 4;
        int consumed = 0;
        const int n = f.resample(din + 2 * pos, c, dout, mo, consumed, st);
        all_consumed = all_consumed && consumed == c;
        const complexf* o = reinterpret_cast<const complexf*>(dout);
        y.insert(y.end(), o, o + n);
        pos += c;
    }
    device_input ? cudaFree(din) : cudaFreeHost(din);
    cudaFreeHost(dout);
    cudaStreamDestroy(st);
    return y;
}

std::vector<complexf> run_cpu(double stride, const std::vector<complexf>& x, const std::vector<int>& part)
{
    farrow_interpolator f(stride);
    std::vector<complexf> y, buf;
    size_t pos = 0;
    for (int c : part) {
        buf.resize((size_t)std::ceil(c / stride) + 4);
        int consumed = 0;
        const int n = f.resample(x.data() + pos, c, buf.data(), (int)buf.size(), consumed);
        y.insert(y.end(), buf.begin(), buf.begin() + n);
        pos += c;
    }
    return y;
}

const double STRIDES[] = { 2.0 / 4.9152, 2.0 / 2.4576, 2.0 / 1.3333, 1.0, 2.7, 0.731731 };

} // namespace

/* R4.1 (GPU), R4.2, R4.5 (GPU) */
BOOST_AUTO_TEST_CASE(test_gpu_matches_cpu_any_partition)
{
    const size_t n = 300000;
    const auto x = random_signal(n, 11);
    for (double s : STRIDES) {
        bool ok;
        const auto g1 = run_gpu(s, x, { 65536, 65536, 65536, 65536, (int)(n - 4 * 65536) }, ok, true);
        BOOST_CHECK(ok);
        double worst_part = 0.0, worst_cpu = 0.0;
        for (unsigned seed = 0; seed < 3; seed++) {
            const auto part = random_partition(n, 65536, 20 + seed);
            const auto g = run_gpu(s, x, part, ok, seed & 1);
            BOOST_CHECK(ok);
            const auto c = run_cpu(s, x, part);
            BOOST_REQUIRE_EQUAL(g.size(), g1.size());
            BOOST_REQUIRE_EQUAL(c.size(), g.size());
            worst_part = std::max(worst_part, rel_error(g, g1, g.size()));
            worst_cpu = std::max(worst_cpu, rel_error(g, c, g.size()));
        }
        BOOST_TEST_MESSAGE("GPU stride " << s << ": partition error " << worst_part << ", GPU vs CPU "
                                         << worst_cpu);
        BOOST_CHECK_LE(worst_part, 1e-6);
        BOOST_CHECK_LE(worst_cpu, 1e-5);
    }
}

/* R4.6 (GPU): >= 1 Gsample/s output for 65536-sample calls (device input
 * and output; the call synchronises its stream) */
BOOST_AUTO_TEST_CASE(test_gpu_throughput)
{
    const size_t n = 65536;
    const auto x = random_signal(n * 16, 12);
    for (double s : { 2.0 / 2.4576, 2.0 / 4.9152, 2.0 / 1.3333 }) {
        cuda::farrow_cuda f(s, (int)(n / s) + 16);
        cudaStream_t st;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        float *din, *dout;
        cudaMalloc(&din, x.size() * sizeof(complexf));
        cudaMemcpy(din, x.data(), x.size() * sizeof(complexf), cudaMemcpyHostToDevice);
        cudaMalloc(&dout, ((size_t)(n / s) + 16) * sizeof(complexf));
        int consumed;
        f.resample(din, (int)n, dout, (int)(n / s) + 16, consumed, st); // warm-up
        size_t total = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 200; rep++)
            total += f.resample(din + 2 * n * (rep % 16), (int)n, dout, (int)(n / s) + 16, consumed, st);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double gsps = total / sec / 1e9;
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        BOOST_TEST_MESSAGE("GPU stride " << s << ": " << gsps << " Gsample/s output (" << sec / 200 * 1e6
                                         << " us per 65536-sample call, " << sms << " SMs)");
        BOOST_CHECK_GE(gsps, 1.0);
        cudaFree(din);
        cudaFree(dout);
        cudaStreamDestroy(st);
    }
}
