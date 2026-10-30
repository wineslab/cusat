/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA demapper: fused soft demap + column-row deinterleave.
 *
 * Grid: dim3(ceil(xfecframe_len/256), batch_size), Block: 256.
 * Each thread handles one symbol of one codeword.
 *
 * QPSK:   formula-based (2 LLRs, no deinterleave)
 * 8PSK:   formula-based after -π/8 rotation (3 LLRs, deinterleaved)
 * 16APSK: min-distance over 16 constellation points (per-instance device memory)
 * 32APSK: min-distance over 32 constellation points (per-instance device memory)
 * 64APSK: min-distance over 64 constellation points (per-instance device memory)
 */

#include "demapper_kernel.cuh"
#include "modulation/bit_interleaver.h"
#include "modulation/constellation.h"
#include "cuda_util.cuh"
#include <cusat_dsp/modcod_types.h>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace cusat_dsp {
namespace cuda {

/* ---------- Fused demap + deinterleave kernel ---------- */

__global__ void demap_deinterleave_kernel(
    const float* __restrict__ sym_in,     // [B * xfecframe_len * 2] (re,im pairs)
    int16_t* __restrict__ soft_out,       // [B * fecframe_len]
    const float* __restrict__ const_re,   // [n_points] constellation I (per-instance)
    const float* __restrict__ const_im,   // [n_points] constellation Q (per-instance)
    int xfecframe_len, int interleaved_len, int fecframe_len,
    int n_mod, int n_points,
    unsigned int ra0, unsigned int ra1, unsigned int ra2,
    unsigned int ra3, unsigned int ra4, unsigned int ra5,
    unsigned int ra6, unsigned int ra7,
    float noise_var, float llr_gain)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int cw = blockIdx.y;

    if (j >= interleaved_len)
        return;

    float re = sym_in[(cw * xfecframe_len + j) * 2];
    float im = sym_in[(cw * xfecframe_len + j) * 2 + 1];
    int base = cw * fecframe_len;

    if (n_mod == 2) {
        /* QPSK: LLR = clamp(round(val * 2*sqrt(2) / noise_var))
         * Scale: 1x true LLR (exact AWGN QPSK LLR). The decoder's
         * offset min-sum is nominally calibrated for FACTOR=2 (beta=1 =
         * 0.5 LLR) and the AWGN bench shows 2x would buy ~0.15 dB at the
         * waterfall — but on the high-SNR OTA capture gate, doubling the
         * channel-LLR confidence against the fixed int8 message clamp
         * (+-127) measurably hurt (112 -> 114 errored frames of 766):
         * impaired stretches need extrinsic messages to out-vote the
         * channel. Revisit with an noise_var-floor policy if ACM ever operates
         * near threshold (bench-ldpc-decode --llr-scale, 2026-06-12). */
        float scalar = 2.0f * 1.4142135623730951f / noise_var;
        float llr_re = re * scalar;
        float llr_im = im * scalar;
        llr_re = fminf(fmaxf(nearbyintf(llr_re), -32768.0f), 32767.0f);
        llr_im = fminf(fmaxf(nearbyintf(llr_im), -32768.0f), 32767.0f);
        /* QPSK has no interleaving — sequential write */
        int pos = j * 2;
        if (pos + 1 < fecframe_len) {
            soft_out[base + pos]     = (int16_t)llr_re;
            soft_out[base + pos + 1] = (int16_t)llr_im;
        }
    }
    else if (n_mod == 3 && const_re == nullptr) {
        /* 8PSK: rotate by exp(-jπ/8), then formula-based LLRs.
         * The pi/8 derotation turns the eight EN 302 307-1 figure 10 decision
         * regions into octants bounded by the axes and diagonals. (2+4+2 8APSK
         * has n_mod==3 too, but supplies a constellation LUT, so it falls through to the
         * generic APSK branch below.) */
        const float cos_pi_8 = 0.92387953251128676f;
        const float sin_pi_8 = 0.38268343236508977f;
        const float rcp_sqrt_2 = 0.70710678118654752f;
        const float DIST = 2.0f * sin_pi_8;

        /* rot = exp(-jπ/8) = cos(π/8) - j*sin(π/8)
         * c = (re + j*im) * rot */
        float cr =  re * cos_pi_8 + im * sin_pi_8;
        float ci = -re * sin_pi_8 + im * cos_pi_8;

        /* DIST * 4/noise_var = 2x the true max-log LLR slope (true = 2*DIST/2N0):
         * already at the decoder's FACTOR=2 design scale. */
        float scale = DIST * llr_gain;
        float llr0 = rcp_sqrt_2 * (fabsf(cr) - fabsf(ci)) * scale;
        float llr1 = cr * scale;
        float llr2 = ci * scale;

        llr0 = fminf(fmaxf(nearbyintf(llr0), -32768.0f), 32767.0f);
        llr1 = fminf(fmaxf(nearbyintf(llr1), -32768.0f), 32767.0f);
        llr2 = fminf(fmaxf(nearbyintf(llr2), -32768.0f), 32767.0f);

        /* Deinterleaved write: bit b goes to rowaddr[b] + j */
        soft_out[base + ra0 + j] = (int16_t)llr0;
        soft_out[base + ra1 + j] = (int16_t)llr1;
        soft_out[base + ra2 + j] = (int16_t)llr2;
    }
    else {
        /* Higher-order APSK: 16/32/64/128 points, 4-8 bits.
         * Log-MAP LLR via two-pass search with cached distances.
         * Pass 1: compute all distances + find min per bit polarity.
         * Pass 2: accumulate exp corrections using cached distances.
         * Uses fast math intrinsics (__expf/__logf) for throughput. */
        float min_d0[8], min_d1[8];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            min_d0[b] = 1e30f;
            min_d1[b] = 1e30f;
        }

        /* Pass 1: distances + min tracking */
        for (int i = 0; i < n_points; i++) {
            float dr = re - const_re[i];
            float di = im - const_im[i];
            float d = dr * dr + di * di;

            #pragma unroll
            for (int b = 0; b < 8; b++) {
                if (b >= n_mod) break;
                if (i & (1 << (n_mod - 1 - b))) {
                    if (d < min_d1[b]) min_d1[b] = d;
                } else {
                    if (d < min_d0[b]) min_d0[b] = d;
                }
            }
        }

        /* Pass 2: log-MAP correction. Recompute distances instead of caching
         * them in a per-thread array; the cache spills to local memory for
         * high-order APSK and costs more than the extra FMAs. */
        float scale = llr_gain * 0.25f;
        float sum0[8], sum1[8];
        #pragma unroll
        for (int b = 0; b < 8; b++) { sum0[b] = 0.0f; sum1[b] = 0.0f; }

        for (int i = 0; i < n_points; i++) {
            float dr = re - const_re[i];
            float di = im - const_im[i];
            float d = dr * dr + di * di;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                if (b >= n_mod) break;
                if (i & (1 << (n_mod - 1 - b))) {
                    float delta = (d - min_d1[b]) * scale;
                    if (delta < 16.0f)
                        sum1[b] += __expf(-delta);
                } else {
                    float delta = (d - min_d0[b]) * scale;
                    if (delta < 16.0f)
                        sum0[b] += __expf(-delta);
                }
            }
        }

        /* LLR = max-log base + log-MAP correction, at the decoder's
         * FACTOR=2 design scale: true LLR is dd^2/noise_var + log-corr;
         * llr_gain = 4/noise_var, so llr_gain*0.5 (and 2x the correction)
         * yields 2x true LLR. The previous 4x-true scale cost ~0.25 dB
         * on the AWGN waterfall vs FACTOR=2 (offset beta mismatch +
         * int8 message saturation; bench --llr-scale sweep 2026-06-12). */
        unsigned int rowaddr[8] = {ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7};
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            if (b >= n_mod) break;
            float llr = (min_d1[b] - min_d0[b]) * (llr_gain * 0.5f)
                        + 2.0f * (__logf(fmaxf(sum0[b], 1e-30f))
                                - __logf(fmaxf(sum1[b], 1e-30f)));
            llr = fminf(fmaxf(nearbyintf(llr), -32768.0f), 32767.0f);
            unsigned int pos = rowaddr[b] + j;
            if (pos < (unsigned int)fecframe_len)
                soft_out[base + pos] = (int16_t)llr;
        }
    }
}

/* ---------- Host API ---------- */

demapper_cuda::demapper_cuda(modulation_t constellation,
                             code_rate_t rate,
                             fecframe_size_t framesize,
                             int max_batch)
    : d_max_batch(max_batch), d_sym_dev(nullptr), d_soft_out_dev(nullptr)
{
    /* Frame length */
    d_fecframe_len = fecframe_bits(framesize);

    /* Modulation order and constellation LUT (QPSK and 8PSK are demapped
     * by formula; the table is only uploaded for the APSK families) */
    const constellation_t points = make_constellation(constellation, rate, framesize);
    d_n_mod = points.bits_per_sym;
    d_n_points = points.n_points;
    float const_re[256] = {0}, const_im[256] = {0};
    for (int i = 0; i < d_n_points; i++) {
        const_re[i] = points.points[i].real();
        const_im[i] = points.points[i].imag();
    }

    /* "Loudest bit" geometric constant: max over (point, bit) of the squared
     * distance to the nearest opposite-bit point. Together with the llr_gain
     * scale this sets the max channel LLR (~ dmax * llr_gain / 2). Capping
     * llr_gain so that max LLR stays near d_llr_target keeps the decoder's
     * int8 check messages (ldpc_message_update.h, clamp +-127) from railing —
     * uncapped, mid-SNR dense-constellation LLRs flatten all message
     * magnitudes and min-sum degrades toward hard-decision voting (measured
     * as non-monotone error floors, 2026-08-01 waterfall campaign). */
    if (constellation == MOD_QPSK) {
        d_llr_dmax = 2.0f;   /* unit-Es QPSK: opposite bit at d^2 = 2 */
    } else if (constellation == MOD_8PSK) {
        d_llr_dmax = 4.0f;   /* circle diameter bound (Gray MSB flip) */
    } else {
        float dmax = 0.0f;
        for (int i = 0; i < d_n_points; i++) {
            for (int b = 0; b < d_n_mod; b++) {
                const int mask = 1 << (d_n_mod - 1 - b);
                float dmin = 1e30f;
                for (int j = 0; j < d_n_points; j++) {
                    if (((i ^ j) & mask) == 0) continue;
                    float dr = const_re[i] - const_re[j];
                    float di = const_im[i] - const_im[j];
                    float d = dr * dr + di * di;
                    if (d < dmin) dmin = d;
                }
                if (dmin > dmax) dmax = dmin;
            }
        }
        d_llr_dmax = dmax;
    }
    const char* llr_tgt = getenv("CUSAT_DEMAP_LLR_TARGET");
    d_llr_target = llr_tgt ? (float)atof(llr_tgt) : 320.0f;

    if (constellation == MOD_128APSK) {
        d_interleaved_len = (d_fecframe_len + 6) / d_n_mod; // 9258
        d_xfecframe_len = d_interleaved_len + 12;           // 9270
    } else {
        d_xfecframe_len = d_fecframe_len / d_n_mod;
        d_interleaved_len = d_xfecframe_len;
    }

    /* Deinterleave column offsets (QPSK: not interleaved, all zero) */
    const bit_interleaver_t il = make_bit_interleaver(constellation, rate, framesize);
    for (int i = 0; i < 8; i++)
        d_bit_index[i] = il.col_offset[i];

    /* Allocate per-instance constellation on device (APSK constellations) */
    d_const_re_dev = nullptr;
    d_const_im_dev = nullptr;
    if (d_n_mod >= 4 || constellation == MOD_8APSK) {
        CUDA_CHECK(cudaMalloc(&d_const_re_dev, d_n_points * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_const_im_dev, d_n_points * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_const_re_dev, const_re,
                              d_n_points * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_const_im_dev, const_im,
                              d_n_points * sizeof(float), cudaMemcpyHostToDevice));
    }

    /* Allocate device buffers */
    size_t sym_size = (size_t)max_batch * d_xfecframe_len * 2 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_sym_dev, sym_size));
    CUDA_CHECK(cudaMalloc(&d_soft_out_dev,
        (size_t)max_batch * d_fecframe_len * sizeof(int16_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));

    size_t sym_kb = (size_t)max_batch * d_xfecframe_len * 2 * sizeof(float) / 1024;
    size_t soft_kb = (size_t)max_batch * d_fecframe_len * sizeof(int16_t) / 1024;
    fprintf(stderr, "demapper_cuda: n_mod=%d fecframe=%d xfecframe=%d "
            "rowaddr=[%u,%u,%u,%u,%u,%u] (%zu+%zu KB device)\n",
            d_n_mod, d_fecframe_len, d_xfecframe_len,
            d_bit_index[0], d_bit_index[1], d_bit_index[2], d_bit_index[3], d_bit_index[4], d_bit_index[5],
            sym_kb, soft_kb);
}

float demapper_cuda::capped_llr_gain(float llr_gain) const
{
    if (d_llr_target <= 0.0f || d_llr_dmax <= 0.0f)
        return llr_gain;
    /* Kernel max-log LLR ~= dd^2 * llr_gain / 2, so max |LLR| ~=
     * d_llr_dmax * llr_gain / 2. Only ever lowers llr_gain. */
    const float cap = 2.0f * d_llr_target / d_llr_dmax;
    return llr_gain < cap ? llr_gain : cap;
}

demapper_cuda::~demapper_cuda()
{
    cudaFree(d_sym_dev);
    cudaFree(d_soft_out_dev);
    if (d_const_re_dev) cudaFree(d_const_re_dev);
    if (d_const_im_dev) cudaFree(d_const_im_dev);
    cudaStreamDestroy(d_stream);
}

void demapper_cuda::demap_batch(const float* sym_in_host, int batch_size,
                                float noise_var, float llr_gain)
{
    if (batch_size <= 0 || batch_size > d_max_batch)
        return;
    llr_gain = capped_llr_gain(llr_gain);

    size_t sym_bytes = (size_t)batch_size * d_xfecframe_len * 2 * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_sym_dev, sym_in_host, sym_bytes,
                               cudaMemcpyHostToDevice, d_stream));

    /* Launch fused demap + deinterleave kernel.
     * For 128APSK, only process d_interleaved_len (9258) symbols — the 12
     * post-padding symbols must NOT be demapped (their LLRs would overwrite
     * valid FECFRAME positions through the deinterleave permutation). */
    int threads = 256;
    dim3 grid(((unsigned)d_interleaved_len + threads - 1) / threads,
              (unsigned)batch_size);
    demap_deinterleave_kernel<<<grid, threads, 0, d_stream>>>(
        d_sym_dev, d_soft_out_dev,
        d_const_re_dev, d_const_im_dev,
        d_xfecframe_len, d_interleaved_len, d_fecframe_len,
        d_n_mod, d_n_points,
        d_bit_index[0], d_bit_index[1], d_bit_index[2],
        d_bit_index[3], d_bit_index[4], d_bit_index[5],
        d_bit_index[6], d_bit_index[7],
        noise_var, llr_gain);
    CUDA_CHECK_LAUNCH();

    /* Synchronize — soft output must be ready before LDPC reads it */
    CUDA_CHECK(cudaStreamSynchronize(d_stream));
}

void demapper_cuda::demap_batch_async(const float* sym_in_host, int batch_size,
                                      float noise_var, float llr_gain,
                                      cudaStream_t stream)
{
    if (batch_size <= 0 || batch_size > d_max_batch)
        return;
    llr_gain = capped_llr_gain(llr_gain);

    size_t sym_bytes = (size_t)batch_size * d_xfecframe_len * 2 * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_sym_dev, sym_in_host, sym_bytes,
                               cudaMemcpyHostToDevice, stream));

    /* Launch fused demap + deinterleave kernel (see demap_batch comment) */
    int threads = 256;
    dim3 grid(((unsigned)d_interleaved_len + threads - 1) / threads,
              (unsigned)batch_size);
    demap_deinterleave_kernel<<<grid, threads, 0, stream>>>(
        d_sym_dev, d_soft_out_dev,
        d_const_re_dev, d_const_im_dev,
        d_xfecframe_len, d_interleaved_len, d_fecframe_len,
        d_n_mod, d_n_points,
        d_bit_index[0], d_bit_index[1], d_bit_index[2],
        d_bit_index[3], d_bit_index[4], d_bit_index[5],
        d_bit_index[6], d_bit_index[7],
        noise_var, llr_gain);
    CUDA_CHECK_LAUNCH();
    /* No sync — caller tracks completion via cudaEvent */
}

} // namespace cuda
} // namespace cusat_dsp
