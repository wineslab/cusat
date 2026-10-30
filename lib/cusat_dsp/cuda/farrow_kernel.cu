/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU cubic Lagrange Farrow resampler. See
 * farrow_interpolator.cc for the polynomial and the position convention;
 * the host-side bookkeeping below mirrors it exactly (same fma position
 * expression), so CPU and GPU agree on the output count and positions.
 */

#include "farrow_kernel.cuh"
#include "cuda_util.cuh"
#include <algorithm>
#include <cmath>

namespace cusat_dsp {
namespace cuda {

namespace {

__global__ void farrow_kernel(const float2* __restrict__ in, const float2* __restrict__ hist,
                              float2* __restrict__ out, int n_out, long long a0, long long nb,
                              double mub, double off, double stride)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out)
        return;
    const double t = fma((double)j + off, stride, mub);
    const double fl = floor(t);
    const long long n = nb + (long long)fl;
    const float mu = (float)(t - fl);
    float2 x[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const long long a = n - 1 + i;
        x[i] = a >= a0 ? in[a - a0] : hist[a - (a0 - farrow_cuda::HISTORY)];
    }
    const float third = 1.0f / 3.0f, sixth = 1.0f / 6.0f;
    float2 c1, c2, c3, y;
    c1.x = x[2].x - x[0].x * third - x[1].x * 0.5f - x[3].x * sixth;
    c1.y = x[2].y - x[0].y * third - x[1].y * 0.5f - x[3].y * sixth;
    c2.x = (x[0].x + x[2].x) * 0.5f - x[1].x;
    c2.y = (x[0].y + x[2].y) * 0.5f - x[1].y;
    c3.x = (x[3].x - x[0].x) * sixth + (x[1].x - x[2].x) * 0.5f;
    c3.y = (x[3].y - x[0].y) * sixth + (x[1].y - x[2].y) * 0.5f;
    y.x = ((c3.x * mu + c2.x) * mu + c1.x) * mu + x[1].x;
    y.y = ((c3.y * mu + c2.y) * mu + c1.y) * mu + x[1].y;
    out[j] = y;
}

} // namespace

farrow_cuda::farrow_cuda(double stride, int max_out) : d_stride(stride), d_max_out(max_out)
{
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d_hist_dev[i], 2 * HISTORY * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_hist_dev[i], 0, 2 * HISTORY * sizeof(float)));
    }
}

farrow_cuda::~farrow_cuda()
{
    for (int i = 0; i < 2; i++)
        cudaFree(d_hist_dev[i]);
}

int farrow_cuda::resample(const float* in, int n_in, float* out, int max_out, int& n_consumed,
                          cudaStream_t stream)
{
    n_consumed = 0;
    if (n_in < 0)
        n_in = 0;
    if (max_out < 0)
        max_out = 0;
    const double s = d_stride;
    const long long nb = d_have_last ? d_last_n : 0;
    const double mub = d_have_last ? d_last_mu : 0.0;
    const double off = d_have_last ? 1.0 : 0.0;
    const long long a0 = d_in_abs;
    const long long end = a0 + n_in;

    auto pos = [&](long long j, long long& n, double& mu) {
        const double t = std::fma((double)j + off, s, mub);
        const double fl = std::floor(t);
        n = nb + (long long)fl;
        mu = t - fl;
    };
    // number of outputs with n + 2 < end (estimate, then settle exactly with
    // the same expression the kernel uses)
    long long n = 0;
    double mu = 0.0;
    long long cnt = (long long)std::floor(((double)(end - 3 - nb) + 1.0 - mub) / s - off) + 1;
    cnt = std::max(0LL, cnt);
    while (cnt > 0) {
        pos(cnt - 1, n, mu);
        if (n + 2 < end)
            break;
        cnt--;
    }
    while (true) {
        pos(cnt, n, mu);
        if (n + 2 >= end)
            break;
        cnt++;
    }
    const long long natural = cnt;
    const int n_out = (int)std::min<long long>(natural, max_out);
    pos(n_out, n, mu); // next output position
    const bool clipped = n_out < natural;
    const long long u = clipped ? std::min(end, std::max(a0, n + 2)) : end;

    const float2* hist = reinterpret_cast<const float2*>(d_hist_dev[d_hist_cur]);
    if (n_out > 0) {
        const int threads = 256;
        const int blocks = (n_out + threads - 1) / threads;
        farrow_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const float2*>(in), hist,
                                                      reinterpret_cast<float2*>(out), n_out, a0, nb,
                                                      mub, off, s);
        CUDA_CHECK_LAUNCH();
    }
    // new history: samples u-3 .. u-1, from the old history and/or the input
    float* nh = d_hist_dev[d_hist_cur ^ 1];
    const long long h0 = u - HISTORY;
    const int from_old = (int)std::max(0LL, std::min<long long>(HISTORY, a0 - h0));
    if (from_old > 0)
        CUDA_CHECK(cudaMemcpyAsync(nh, d_hist_dev[d_hist_cur] + 2 * (h0 - (a0 - HISTORY)),
                                   from_old * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    if (from_old < HISTORY)
        CUDA_CHECK(cudaMemcpyAsync(nh + 2 * from_old, in + 2 * (h0 + from_old - a0),
                                   (HISTORY - from_old) * 2 * sizeof(float), cudaMemcpyDefault, stream));
    d_hist_cur ^= 1;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    n_consumed = (int)(u - a0);
    d_in_abs = u;
    if (n_out > 0) {
        pos(n_out - 1, n, mu);
        d_last_n = n;
        d_last_mu = mu;
        d_have_last = true;
    }
    (void)d_max_out;
    return n_out;
}

} // namespace cuda
} // namespace cusat_dsp
