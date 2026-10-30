/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU bulk PLHEADER search.
 */

#include "cusat_dsp/cuda/plheader_search_kernel.cuh"
#include "cusat_dsp/framing/plframe.h"
#include "cusat_dsp/framing/plheader.h"
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace cusat_dsp {

namespace {

/* raised cosine (RRC x RRC) at t symbols */
double rc(double t, double a)
{
    if (std::fabs(t) < 1e-9)
        return 1.0;
    if (std::fabs(std::fabs(t) - 1.0 / (2 * a)) < 1e-9)
        return M_PI / 4 * std::sin(M_PI / (2 * a)) / (M_PI / (2 * a));
    return std::sin(M_PI * t) / (M_PI * t) * std::cos(M_PI * a * t) / (1 - 4 * a * a * t * t);
}

/* Symbol-spaced matched-filter output of random QPSK data with one PLHEADER
 * at `pos`, timing error `dt` (symbols), CFO, phase and AWGN. */
std::vector<complexf> make_buffer(int n, int pos, uint8_t pls, double cfo, double phase, double dt,
                                  double esn0_db, std::mt19937& rng)
{
    std::vector<std::complex<double>> a(n + 40);
    std::uniform_int_distribution<int> q(0, 3);
    const double s = M_SQRT1_2;
    for (auto& v : a)
        v = { (q(rng) & 1) ? -s : s, (q(rng) & 2) ? -s : s };
    complexf h[90];
    plheader_symbols(pls, h);
    for (int k = 0; k < 90; k++)
        if (pos + k >= 0 && pos + k < n)
            a[pos + k + 20] = h[k];
    std::normal_distribution<double> g(0.0, std::sqrt(0.5 * std::pow(10.0, -esn0_db / 10.0)));
    std::vector<complexf> y(n);
    for (int k = 0; k < n; k++) {
        std::complex<double> acc = 0;
        for (int j = -12; j <= 12; j++)
            acc += a[k + 20 + j] * rc(dt - j, 0.2);
        acc *= std::polar(1.0, phase + 2 * M_PI * cfo * k);
        acc += std::complex<double>(g(rng), g(rng));
        y[k] = complexf((float)acc.real(), (float)acc.imag());
    }
    return y;
}

bool found(const pl_header_candidates& c, int idx)
{
    for (int i = 0; i < c.count; i++)
        if (std::abs(c.index[i] - idx) <= 1)
            return true;
    return false;
}

} // namespace

/* Detection probability at Es/N0 = -3 dB, random position, phase, CFO in
 * +-0.05, timing error <= 0.1 symbol, 32768-symbol buffers. */
BOOST_AUTO_TEST_CASE(test_detection_minus_3db)
{
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    cuda::plheader_search_cuda s(32768 + 90);
    const char* e = getenv("CUSAT_QA_SCALE");
    const int trials = std::max(100, (int)(2000 * (e ? atof(e) : 1.0)));
    int miss = 0, miss1 = 0, miss2 = 0;
    const uint8_t codes[] = { 17, 0, 5, 101, 157, 201, 211, 3 };
    for (int t = 0; t < trials; t++) {
        const int pos = (int)(u(rng) * (32768 - 90));
        const double cfo = -0.05 + 0.1 * u(rng);
        const double dt = -0.1 + 0.2 * u(rng);
        const auto y = make_buffer(32768, pos, codes[t % 8], cfo, 2 * M_PI * u(rng), dt, -3.0, rng);
        const pl_header_candidates c = s.search(y.data(), (int)y.size(), 0);
        BOOST_REQUIRE_LE(c.count, pl_header_candidates::kMaxCandidates);
        for (int i = 1; i < c.count; i++)
            BOOST_CHECK_GE(c.score[i - 1], c.score[i]);
        miss += !found(c, pos);
        // survival of the per-block top-K (cheap) stage
        bool s1 = false;
        for (int q : s.stage1_positions())
            s1 |= std::abs(q - pos) <= 1;
        miss1 += !s1;
        bool s2 = false;
        for (int q : s.stage2_positions())
            s2 |= std::abs(q - pos) <= 1;
        miss2 += !s2;
        if (!s1 && getenv("QA_DEBUG"))
            printf("stage-1 miss: pos %d code %d cfo %.4f dt %.3f\n", pos, codes[t % 8], cfo, dt);
        if (!found(c, pos) && getenv("QA_DEBUG"))
            printf("miss: pos %d code %d cfo %.4f dt %.3f top %.3f last %.3f\n", pos, codes[t % 8], cfo, dt,
                   c.count ? c.score[0] : 0.f, c.count ? c.score[c.count - 1] : 0.f);
    }
    const double p = 1.0 - (double)miss / trials;
    const double p1 = 1.0 - (double)miss1 / trials;
    BOOST_TEST_MESSAGE("P(detect) at -3 dB = " << p << " (" << miss << " misses / " << trials
                                               << "); top-K stage survival " << p1 << " (" << miss1
                                               << " misses), after the mid statistic "
                                               << 1.0 - (double)miss2 / trials << " (" << miss2 << ")");
    BOOST_CHECK_GE(p1, 0.999);
    BOOST_CHECK_GE(p, 0.999);
}

/* Amplitude invariance: x0.1 and x10 change no candidate, no score by more
 * than 1e-3 relative. */
BOOST_AUTO_TEST_CASE(test_amplitude_invariance)
{
    std::mt19937 rng(2);
    cuda::plheader_search_cuda s(20000);
    for (int t = 0; t < 20; t++) {
        auto y = make_buffer(16384, 5000 + 31 * t, 21, 0.02, 1.0, 0.05, 2.0, rng);
        const pl_header_candidates ref = s.search(y.data(), (int)y.size(), 0);
        for (float k : { 0.1f, 10.0f }) {
            std::vector<complexf> z(y);
            for (auto& v : z)
                v *= k;
            const pl_header_candidates c = s.search(z.data(), (int)z.size(), 0);
            BOOST_REQUIRE_EQUAL(c.count, ref.count);
            for (int i = 0; i < c.count; i++) {
                BOOST_CHECK_EQUAL(c.index[i], ref.index[i]);
                BOOST_CHECK_SMALL(c.score[i] - ref.score[i], 1e-3f * ref.score[i] + 1e-7f);
            }
        }
        BOOST_CHECK(found(ref, 5000 + 31 * t));
        BOOST_CHECK_GT(ref.score[0], 0.4f); // s ~ rho / (1 + rho) = 0.61 at 2 dB, minus grid/timing loss
    }
}

/* Overlap: a header starting in the last 89 symbols of the previous batch
 * is reported with a negative index; one fully inside the overlap is not. */
BOOST_AUTO_TEST_CASE(test_overlap_negative_index)
{
    std::mt19937 rng(3);
    cuda::plheader_search_cuda s(8192 + 90);
    for (int o : { 1, 10, 45, 89 }) {
        // buffer = [90 overlap | 8192 batch]; header starts at overlap index o
        auto y = make_buffer(8192 + 90, o, 18, -0.01, 0.4, 0.0, 6.0, rng);
        const pl_header_candidates c = s.search(y.data(), (int)y.size(), pl::kHeaderLen);
        BOOST_CHECK_EQUAL(c.searched, 8192);
        BOOST_CHECK(found(c, o - 90));
        for (int i = 0; i < c.count; i++)
            BOOST_CHECK_GE(c.index[i], -(pl::kHeaderLen - 1));
    }
    // a header at overlap index 0 lay entirely in the previous batch
    auto y = make_buffer(8192 + 90, 0, 18, 0.0, 0.0, 0.0, 10.0, rng);
    const pl_header_candidates c = s.search(y.data(), (int)y.size(), pl::kHeaderLen);
    for (int i = 0; i < c.count; i++)
        BOOST_CHECK_NE(c.index[i], -90);
}

/* Async contract and GPU time for a 32768-symbol buffer. */
BOOST_AUTO_TEST_CASE(test_async_and_timing)
{
    std::mt19937 rng(4);
    cuda::plheader_search_cuda s(32768 + 90);
    auto y = make_buffer(32768 + 90, 12345, 17, 0.03, 0.2, 0.0, 0.0, rng);
    float* pinned = nullptr;
    cudaMallocHost(&pinned, y.size() * sizeof(complexf));
    memcpy(pinned, y.data(), y.size() * sizeof(complexf));
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    s.launch(pinned, (int)y.size(), 90, st); // warm-up
    cudaStreamSynchronize(st);
    float best = 1e9f;
    for (int r = 0; r < 20; r++) {
        cudaEventRecord(e0, st);
        s.launch(pinned, (int)y.size(), 90, st);
        cudaEventRecord(e1, st);
        cudaStreamSynchronize(st);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, ms);
    }
    const pl_header_candidates c = s.result();
    BOOST_CHECK(found(c, 12345 - 90));
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    BOOST_TEST_MESSAGE("GPU time for 32768 symbols: " << best * 1000 << " us on " << sms << " SMs");
    BOOST_CHECK_EQUAL(cudaGetLastError(), cudaSuccess);
    cudaFreeHost(pinned);
    cudaStreamDestroy(st);
}

/* All-zero, NaN and tiny buffers. */
BOOST_AUTO_TEST_CASE(test_degenerate_inputs)
{
    cuda::plheader_search_cuda s(4096);
    std::vector<complexf> z(4096, complexf(0, 0));
    pl_header_candidates c = s.search(z.data(), 4096, 0);
    BOOST_CHECK_EQUAL(c.count, 0);
    for (auto& v : z)
        v = complexf(NAN, INFINITY);
    c = s.search(z.data(), 4096, 0);
    BOOST_CHECK_EQUAL(c.count, 0);
    c = s.search(z.data(), 10, 0);
    BOOST_CHECK_EQUAL(c.count, 0);
    c = s.search(z.data(), 0, 0);
    BOOST_CHECK_EQUAL(c.count, 0);
}

} // namespace cusat_dsp
