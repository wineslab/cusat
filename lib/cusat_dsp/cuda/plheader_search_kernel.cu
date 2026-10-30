/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU bulk PLHEADER search: two launches on the caller's
 * stream, no allocation, no host round trip.
 *
 * Notation. For a start position p, derotate the 90 received symbols by the
 * known pi/2-BPSK structure (EN 302 307-1 clause 5.5.2): SOF symbol k is
 * multiplied by its sign and by (-j)^(k&1); PLSC symbol k by its scrambler
 * sign and (-j)^(k&1) (the common factor e^{-j pi/4} is dropped). PLSC pair
 * t then carries e_t = a_t, o_t = +-a_t (sign b7) with a_t = +-1 the inner
 * Reed-Muller codeword bit y_t = (u.t) ^ b6 ^ (b0 & g_t) (EN 302 307-2
 * clause 5.5.2.4), all multiplied by the carrier.
 *
 * Kernel 1 (every position; 64 positions per block, two positions per
 * thread in packed half2 and two threads per position pair). The tile plus
 * its 89-symbol halo is read once from the caller's buffer, normalised by
 * its RMS (scale-covariant, so half precision does not break amplitude
 * invariance) and staged in shared memory with two copies counter-rotated
 * by -+0.025 cycle/symbol:
 *   cheap score c(p) = (1.3 max_b |S_b| + max_{f,b7,b0,u} sum_q |W_q(u)|)
 *                      / sqrt(sum_k |r_{p+k}|^2)
 *   - S_b: SOF coherent sum on 7 bins spaced 1/52 cycle/symbol (from 13
 *     two-symbol partial sums);
 *   - W_q: 4-point Walsh-Hadamard transform of the pair values
 *     v_t = e_t +- o_t (b7) over segment q of 4 pairs (8 symbols), with or
 *     without the b0 row g, on the two rotated copies; the 8 segments are
 *     combined non-coherently (sum of magnitudes), which tolerates
 *     +-0.025 cycle/symbol around each copy. Where g restricted to a
 *     segment is a Walsh function the b0 hypothesis permutes the same
 *     magnitudes; elsewhere g differs from one in a single position (a
 *     4-term correction). |z| ~ max(|x|,|y|) + 0.4 min(|x|,|y|).
 *   (Non-coherent / segmented correlation of a known preamble under a
 *   carrier offset: textbook technique, cf. ETSI TR 102 376 and Meyr,
 *   Moeneclaey & Fechtel 1998, ch. 8.)
 * The block keeps its K1 = 8 best local maxima (+-3 symbols) and, on those,
 * exports each survivor window to the verification stage.
 *
 * Kernel 2 (one block per kernel-1 block, launched with
 * programmatic stream serialisation) ranks survivors with a segmented mid
 * statistic, then evaluates two distinct positions with the full statistic,
 *   s(p) = max_{f, code} |sum_k r_{p+k} conj(h_code,k) e^{-j2pi f k}|^2
 *          / (90 sum_k |r_{p+k}|^2),
 * maximised over the 256 codes through four 32-point WHTs (8-point WHT per
 * lane, the remaining two stages across 4 lanes) and over 16 bins spaced
 * 0.004 around the candidate's mid copy. The last block (atomic ticket)
 * suppresses +-3-symbol neighbours, ranks the survivors and writes the 64
 * best to the pinned host result in one go.
 *
 * Every score is invariant to the input amplitude.
 */

#include "plheader_search_kernel.cuh"
#include "cuda_util.cuh"
#include "framing/plframe.h"
#include "framing/plheader.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cusat_dsp {
namespace cuda {

namespace {

constexpr int HDR = 90;
constexpr int SOFL = 26;
constexpr int NPAIR = 32;
constexpr int NMS = 3;

// ---- kernel 1 ----
constexpr int BLK = 64;              // positions per block
constexpr int TILE = BLK + HDR - 1;   // staged symbols
// Preserve more spatial candidates before the coherent verification stage.
constexpr int K1 = 8;                 // survivors per block
constexpr float PF = 0.025f;          // frequency hypotheses 0, +-PF
constexpr float W_SOF = 1.3f;

// ---- kernel 2 ----
constexpr int VERIFY_PER_GROUP = 2;   // distinct positions reaching the coherent statistic
constexpr int G2 = 1;                 // kernel-1 blocks per verify block
constexpr int MC = 4;                 // mid-statistic frequency copies
constexpr float MSTEP = 0.025f;       // (cp - 1.5) MSTEP: +-0.0125, +-0.0375
constexpr int NBF = 16;               // full-statistic bins around the mid copy
constexpr float VSTEP = 0.004f;       // bins fc + (b - 7.5) VSTEP: fc +- 0.03
constexpr int V_THREADS = 4 * NBF * 4; // 4 lanes per (hyp, bin)
static_assert(NBF % 8 == 0, "hypothesis uniform per warp");
constexpr int SEL_SMEM = 1024;        // final selection from shared memory up to this (multiple of 8)
constexpr int V_SMEM_ELEMS = G2 * K1 * MC * HDR > SEL_SMEM
                                ? G2 * K1 * MC * HDR : SEL_SMEM;
constexpr size_t V_SMEM_BYTES = V_SMEM_ELEMS * sizeof(float2);

// PL constants (checked against framing/plheader.h)
constexpr uint32_t K_SOF = 0x18D2E82u;
constexpr unsigned long long K_SCR = 0x719D83C953422DFAull;
constexpr uint32_t K_G = 0x90AC2DDDu; // EN 302 307-2 clause 5.5.2.4
static_assert(K_SOF == SOF_BITS, "SOF");
static_assert(K_SCR == PLSC_SCRAMBLER, "PLSC scrambler");

__host__ __device__ constexpr float sof_s(int k) { return ((K_SOF >> (25 - k)) & 1u) ? -1.0f : 1.0f; }
__host__ __device__ constexpr float scr_s(int k) { return ((K_SCR >> (63 - k)) & 1ull) ? -1.0f : 1.0f; }
__host__ __device__ constexpr float g_s(int t) { return ((K_G >> (31 - t)) & 1u) ? -1.0f : 1.0f; }
__host__ __device__ constexpr float walsh2(int m, int t) { return (((m & t) ^ ((m & t) >> 1)) & 1) ? -1.0f : 1.0f; }
/* segment q (pairs 4q..4q+3): g = c w_m(t) (Walsh), returns m, or -1 */
__host__ __device__ constexpr int seg_walsh_m(int q)
{
    for (int m = 0; m < 4; m++) {
        bool ok = true;
        for (int t = 0; t < 4; t++)
            ok = ok && g_s(4 * q + t) == g_s(4 * q) * walsh2(m, t);
        if (ok)
            return m;
    }
    return -1;
}
__host__ __device__ constexpr bool seg_walsh(int q) { return seg_walsh_m(q) >= 0; }
/* otherwise g = c w_m(t) (1 - 2 [t == j]); returns m | (j << 2), or -1 */
__host__ __device__ constexpr int seg_fit(int q)
{
    for (int m = 0; m < 4; m++)
        for (int j = 0; j < 4; j++)
            for (int ci = 0; ci < 2; ci++) {
                const float c = ci ? -1.0f : 1.0f;
                bool ok = true;
                for (int t = 0; t < 4; t++)
                    ok = ok && g_s(4 * q + t) == c * walsh2(m, t) * (t == j ? -1.0f : 1.0f);
                if (ok)
                    return m | (j << 2);
            }
    return -1;
}

/* Phase timestamps for profiling builds (-DHS_PROBE); no code otherwise. */
#ifdef HS_PROBE
__device__ unsigned long long g_probe[2][4096][8];
#define HS_T(k, i)                                                                         \
    do {                                                                                   \
        if (threadIdx.x == 0) {                                                            \
            unsigned long long t_;                                                         \
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));                         \
            g_probe[k][blockIdx.x][i] = t_;                                                \
        }                                                                                  \
    } while (0)
#else
#define HS_T(k, i) \
    do {           \
    } while (0)
#endif

__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
/* r * sign * (-j)^odd */
__device__ __forceinline__ float2 fold(float2 r, float s, bool odd)
{
    return odd ? make_float2(s * r.y, -s * r.x) : make_float2(s * r.x, s * r.y);
}
__device__ __forceinline__ bool finite2(float2 v) { return isfinite(v.x) && isfinite(v.y); }

/* ---- kernel 1 arithmetic: two positions per thread in packed half2 ----
 * Inputs are normalised by the tile RMS (a scale-covariant factor), so the
 * half-precision values, and every cheap score, do not depend on the input
 * amplitude. */
struct c2 {
    __half2 re, im; // (position p, position p + 1)
};
__device__ __forceinline__ __half2 sgn(__half2 x, float s) { return s < 0.0f ? __hneg2(x) : x; }
__device__ __forceinline__ c2 operator+(c2 a, c2 b) { return { __hadd2(a.re, b.re), __hadd2(a.im, b.im) }; }
__device__ __forceinline__ c2 operator-(c2 a, c2 b) { return { __hsub2(a.re, b.re), __hsub2(a.im, b.im) }; }
/* r * sign * (-j)^odd */
__device__ __forceinline__ c2 fold2(c2 r, float s, bool odd)
{
    return odd ? c2{ sgn(r.im, s), sgn(r.re, -s) } : c2{ sgn(r.re, s), sgn(r.im, s) };
}
__device__ __forceinline__ __half2 amag2(c2 z)
{
    const __half2 a = __habs2(z.re), b = __habs2(z.im);
    return __hfma2(__hmin2(a, b), __float2half2_rn(0.4f), __hmax2(a, b));
}
__device__ __forceinline__ void wht4(const c2 (&v)[4], c2 (&w)[4])
{
    const c2 s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    w[0] = s0 + s1;
    w[1] = d0 + d1;
    w[2] = s0 - s1;
    w[3] = d0 - d1;
}

__constant__ __half2 c_sof_c[4][13]; // cos / sin of -2 pi b/52 * 2m, both halves
__constant__ __half2 c_sof_s[4][13];

/* SOF coherent sum on bins spaced 1/52 cycle/symbol (x = unrotated tile):
 * bin pairs +-b/52 for b = 2 part, 2 part + 1 (b = 0 is the centre bin) */
__device__ __forceinline__ __half2 sof_bins2(const __half2* __restrict__ xr, const __half2* __restrict__ xi, int part)
{
    c2 sg[13];
    #pragma unroll
    for (int m = 0; m < 13; m++)
        sg[m] = fold2(c2{ xr[2 * m], xi[2 * m] }, sof_s(2 * m), false) +
                fold2(c2{ xr[2 * m + 1], xi[2 * m + 1] }, sof_s(2 * m + 1), true);
    __half2 best = __float2half2_rn(0.0f);
    #pragma unroll
    for (int bb = 0; bb < 2; bb++) {
        const int b = 2 * part + bb;
        const __half2 z = __float2half2_rn(0.0f);
        c2 sp{ z, z }, sm{ z, z };
        #pragma unroll
        for (int m = 0; m < 13; m++) {
            const __half2 c = c_sof_c[b][m], s = c_sof_s[b][m];
            // sp = sg e^{-j phi}, sm = sg e^{+j phi}; (c, s) = (cos, sin)(-phi)
            sp.re = __hfma2(sg[m].re, c, sp.re);
            sp.re = __hfma2(__hneg2(sg[m].im), s, sp.re);
            sp.im = __hfma2(sg[m].im, c, sp.im);
            sp.im = __hfma2(sg[m].re, s, sp.im);
            sm.re = __hfma2(sg[m].re, c, sm.re);
            sm.re = __hfma2(sg[m].im, s, sm.re);
            sm.im = __hfma2(sg[m].im, c, sm.im);
            sm.im = __hfma2(__hneg2(sg[m].re), s, sm.im);
        }
        best = __hmax2(best, __hmax2(amag2(sp), amag2(sm)));
    }
    return best;
}

/* max over (b7, b0, u) of the segmented PLSC statistic on one rotated copy */
__device__ __forceinline__ __half2 plsc_segmented2(const __half2* __restrict__ xr, const __half2* __restrict__ xi)
{
    const __half2 z = __float2half2_rn(0.0f);
    __half2 a[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int u = 0; u < 4; u++)
                a[i][j][u] = z;
    #pragma unroll
    for (int q = 0; q < 8; q++) {
        c2 v[2][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = 4 * q + j;
            const int k = SOFL + 2 * t;
            const c2 e = fold2(c2{ xr[k], xi[k] }, scr_s(2 * t), false);
            const c2 o = fold2(c2{ xr[k + 1], xi[k + 1] }, scr_s(2 * t + 1), true);
            v[0][j] = e + o;
            v[1][j] = e - o;
        }
        #pragma unroll
        for (int b7 = 0; b7 < 2; b7++) {
            c2 w[4];
            wht4(v[b7], w);
            __half2 m[4];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                m[u] = amag2(w[u]);
                a[b7][0][u] = __hadd2(a[b7][0][u], m[u]);
            }
            if (seg_walsh(q)) {
                #pragma unroll
                for (int u = 0; u < 4; u++)
                    a[b7][1][u] = __hadd2(a[b7][1][u], m[u ^ seg_walsh_m(q)]);
            } else {
                // W_g(u) = c [W(u^m) - 2 w_{u^m}(j) v_j]
                const int fit = seg_fit(q);
                const int mm = fit & 3, jj = fit >> 2;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const __half2 sg2 = __float2half2_rn(-2.0f * walsh2(u ^ mm, jj));
                    const c2 zz{ __hfma2(sg2, v[b7][jj].re, w[u ^ mm].re), __hfma2(sg2, v[b7][jj].im, w[u ^ mm].im) };
                    a[b7][1][u] = __hadd2(a[b7][1][u], amag2(zz));
                }
            }
        }
    }
    __half2 best = a[0][0][0];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int u = 0; u < 4; u++)
                best = __hmax2(best, a[i][j][u]);
    return best;
}

__device__ __forceinline__ bool better(float a, int ia, float b, int ib)
{
    return a > b || (a == b && ia >= 0 && (ib < 0 || ia < ib));
}

__device__ __forceinline__ float amag(float2 z)
{
    const float ax = fabsf(z.x), ay = fabsf(z.y);
    return fmaxf(ax, ay) + 0.4f * fminf(ax, ay);
}
__device__ __forceinline__ void wht8(float2 (&v)[8])
{
    #pragma unroll
    for (int h = 1; h < 8; h <<= 1)
        #pragma unroll
        for (int i = 0; i < 8; i++)
            if ((i & h) == 0) {
                const float2 a = v[i], d = v[i + h];
                v[i] = a + d;
                v[i + h] = a - d;
            }
}
/* pair values of pairs 8q .. 8q+7 for hypothesis hyp = 2 b0 + b7 (runtime,
 * uniform per warp); x = PLSC symbols */
__device__ __forceinline__ void pair_vals(const float2* __restrict__ x, int q, int hyp, float2 (&v)[8])
{
    const float s7 = (hyp & 1) ? -1.0f : 1.0f;
    const uint32_t gbits = (hyp & 2) ? (K_G >> (24 - 8 * q)) & 0xffu : 0u;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const float2 e = x[16 * q + 2 * j], o = x[16 * q + 2 * j + 1];
        const float g = ((gbits >> (7 - j)) & 1u) ? -1.0f : 1.0f;
        v[j] = make_float2(g * fmaf(s7, o.x, e.x), g * fmaf(s7, o.y, e.y));
    }
}

/* mid statistic, one lane per 8-pair segment q (4 lanes per task):
 * sum_q max-free |W8_q(u)|, max over u (lanes 0..3 return the same value) */
__device__ __forceinline__ float mid_lane(const float2* __restrict__ x, int q, int hyp)
{
    float2 v[8];
    pair_vals(x, q, hyp, v);
    wht8(v);
    float m[8];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        m[u] = amag(v[u]);
        m[u] += __shfl_xor_sync(0xffffffffu, m[u], 1);
        m[u] += __shfl_xor_sync(0xffffffffu, m[u], 2);
    }
    float best = m[0];
    #pragma unroll
    for (int u = 1; u < 8; u++)
        best = fmaxf(best, m[u]);
    return best;
}

/* full statistic, one lane per 8 pairs: 8-point WHT in registers, the two
 * remaining stages across the 4 lanes; returns max over this lane's u of
 * |W|^2 + 2 |Re or Im (S* W)| */
__device__ __forceinline__ float full_lane(const float2* __restrict__ x, int q, int hyp, float2 S)
{
    float2 v[8];
    pair_vals(x, q, hyp, v);
    const bool b0 = hyp & 2;
    wht8(v);
    #pragma unroll
    for (int lb = 1; lb < 4; lb <<= 1) {
        const bool hi = q & lb;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const float px = __shfl_xor_sync(0xffffffffu, v[i].x, lb);
            const float py = __shfl_xor_sync(0xffffffffu, v[i].y, lb);
            v[i] = hi ? make_float2(px - v[i].x, py - v[i].y) : make_float2(v[i].x + px, v[i].y + py);
        }
    }
    float best = 0.0f;
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        const float w2 = v[u].x * v[u].x + v[u].y * v[u].y;
        const float cr = b0 ? S.x * v[u].y - S.y * v[u].x : S.x * v[u].x + S.y * v[u].y;
        best = fmaxf(best, w2 + 2.0f * fabsf(cr));
    }
    return best;
}

constexpr int H_PAIRS = BLK / 2;      // position pairs per block
constexpr int H_THREADS = 2 * H_PAIRS; // two roles per pair (SOF bins + one PLSC copy each)
constexpr int HT = TILE;              // half2 pairs (j, j+1), j < TILE

__global__ void __launch_bounds__(H_THREADS, 8)
hs_cheap_kernel(const float2* __restrict__ sym, int n, int first_pos, const float2* __restrict__ rot_pf,
                int* __restrict__ cand_pos, float2* __restrict__ cand_win, float* __restrict__ blk_pow,
                unsigned* __restrict__ pair_done)
{
    constexpr int U_BYTES = 3 * 2 * HT * 4;
    __shared__ float4 s_raw[(U_BYTES + 15) / 16];
    __shared__ float2 s_f0[TILE + 1];        // normalised tile, unrotated
    __shared__ float s_pre[TILE + 1];
    __shared__ float s_red[H_THREADS / 32];
    __shared__ float s_score[BLK];
    __shared__ __half2 s_sof[2][H_PAIRS], s_pl[2][H_PAIRS];
    __half2 (*s_h)[2][HT] = reinterpret_cast<__half2 (*)[2][HT]>(s_raw); // [copy][re/im][j]

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int p0 = first_pos + blockIdx.x * BLK;
    constexpr int PER = (TILE + 1 + H_THREADS - 1) / H_THREADS;

    HS_T(0, 0);
#if __CUDA_ARCH__ >= 900
    // kernel 2 may be scheduled as soon as SM resources free up; it waits for
    // the completion of this grid in cudaGridDependencySynchronize()
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    // ---- load (+ halo), block power. Each thread owns symbols PER t .. PER t
    // + PER - 1 and also loads the next one, so it can pack the half2 pairs
    // (j, j+1) of its symbols without a second pass ----
    float2 v[PER + 1];
    float tot = 0.0f;
    #pragma unroll
    for (int h = 0; h <= PER; h++) {
        const int j = PER * tid + h;
        const int g = p0 + j;
        float2 x = (j <= TILE && g < n) ? sym[g] : make_float2(0.0f, 0.0f);
        if (!finite2(x))
            x = make_float2(0.0f, 0.0f);
        v[h] = x;
        if (h < PER)
            tot += x.x * x.x + x.y * x.y;
    }
    float2 rp[PER + 1];
    #pragma unroll
    for (int h = 0; h <= PER; h++)
        rp[h] = __ldg(rot_pf + min(PER * tid + h, TILE)); // e^{+j 2 pi PF j}
    float inc = tot;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const float o = __shfl_up_sync(0xffffffffu, inc, off);
        if (lane >= off)
            inc += o;
    }
    if (lane == 31)
        s_red[wid] = inc;
    __syncthreads();
    float wbase = 0.0f, all = 0.0f;
    #pragma unroll
    for (int w = 0; w < H_THREADS / 32; w++) {
        if (w < wid)
            wbase += s_red[w];
        all += s_red[w];
    }
    // scale-covariant normalisation: x / rms(tile)
    const float scale = all > 1e-30f ? rsqrtf(all / (TILE + 1)) : 0.0f;
    {
        float run = wbase + inc - tot;
        float2 x[PER + 1], xp[PER + 1], xm[PER + 1];
        #pragma unroll
        for (int h = 0; h <= PER; h++) {
            x[h] = make_float2(v[h].x * scale, v[h].y * scale);
            const float2 r = rp[h];
            xp[h] = make_float2(x[h].x * r.x + x[h].y * r.y, x[h].y * r.x - x[h].x * r.y); // x e^{-j..}
            xm[h] = make_float2(x[h].x * r.x - x[h].y * r.y, x[h].y * r.x + x[h].x * r.y); // x e^{+j..}
        }
        #pragma unroll
        for (int h = 0; h < PER; h++) {
            const int j = PER * tid + h;
            if (j <= TILE) {
                s_pre[j] = run * scale * scale; // exclusive prefix at j
                s_f0[j] = x[h];
            }
            if (j < HT) {
                s_h[0][0][j] = __floats2half2_rn(x[h].x, x[h + 1].x);
                s_h[0][1][j] = __floats2half2_rn(x[h].y, x[h + 1].y);
                s_h[1][0][j] = __floats2half2_rn(xp[h].x, xp[h + 1].x);
                s_h[1][1][j] = __floats2half2_rn(xp[h].y, xp[h + 1].y);
                s_h[2][0][j] = __floats2half2_rn(xm[h].x, xm[h + 1].x);
                s_h[2][1][j] = __floats2half2_rn(xm[h].y, xm[h + 1].y);
            }
            run += v[h].x * v[h].x + v[h].y * v[h].y;
        }
    }
    __syncthreads();

    HS_T(0, 1);
    // ---- cheap scores of positions 2 pr, 2 pr + 1 (two roles, same code) ----
    {
        const int pr = tid % H_PAIRS, role = tid / H_PAIRS;
        const int i0 = 2 * pr;
        s_sof[role][pr] = sof_bins2(s_h[0][0] + i0, s_h[0][1] + i0, role);
        s_pl[role][pr] = plsc_segmented2(s_h[1 + role][0] + i0, s_h[1 + role][1] + i0);
    }
    __syncthreads();
    for (int i = tid; i < BLK; i += H_THREADS) {
        const int pr = i >> 1, h = i & 1;
        const int p = p0 + i;
        float score = -1.0f;
        const float e = s_pre[i + HDR] - s_pre[i];
        if (p + HDR <= n && e > 1e-20f) {
            const float2 sf = __half22float2(__hmax2(s_sof[0][pr], s_sof[1][pr]));
            const float2 pm = __half22float2(s_pl[0][pr]);
            const float2 pp = __half22float2(s_pl[1][pr]);
            score = (W_SOF * (h ? sf.y : sf.x) + fmaxf(h ? pm.y : pm.x, h ? pp.y : pp.x)) * rsqrtf(e);
            if (!isfinite(score))
                score = -1.0f;
        }
        s_score[i] = score;
    }
    __syncthreads();

    HS_T(0, 2);
    // ---- per-block top-K1: local maxima (+-NMS within the block), compacted, ranked ----
    __shared__ float s_lmv[BLK];
    __shared__ int s_lmi[BLK];
    __shared__ int s_nlm;
    __shared__ int s_cidx[K1];
    if (tid < K1)
        s_cidx[tid] = -1;
    if (tid == 0)
        s_nlm = 0;
    __syncthreads();
    static_assert(BLK == H_THREADS, "one position per thread");
    {
        const int i = tid;
        const float vv = s_score[i];
        bool lm = vv > 0.0f;
        #pragma unroll
        for (int d = -NMS; d <= NMS; d++) {
            const int j = i + d;
            if (d != 0 && j >= 0 && j < BLK) {
                const float w = s_score[j];
                lm = lm && (vv > w || (vv == w && i < j));
            }
        }
        const unsigned m = __ballot_sync(0xffffffffu, lm);
        int base = 0;
        if (lane == 0 && m)
            base = atomicAdd(&s_nlm, __popc(m));
        base = __shfl_sync(0xffffffffu, base, 0);
        if (lm) {
            const int slot = base + __popc(m & ((1u << lane) - 1u));
            s_lmv[slot] = vv;
            s_lmi[slot] = i;
        }
    }
    __syncthreads();
    {
        const int nlm = s_nlm;
        if (tid < nlm) {
            const float vv = s_lmv[tid];
            const int i = s_lmi[tid];
            int rank = 0;
            for (int j0 = 0; j0 < nlm; j0 += 8) {
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    const int j = j0 + d;
                    const float w = j < nlm ? s_lmv[j] : -2.0f;
                    const int li = j < nlm ? s_lmi[j] : BLK;
                    rank += (int)(w > vv) | ((int)(w == vv) & (int)(li < i)); // branch-free
                }
            }
            if (rank < K1)
                s_cidx[rank] = i;
        }
    }
    __syncthreads();
    if (tid < K1)
        cand_pos[blockIdx.x * K1 + tid] = s_cidx[tid] >= 0 ? p0 + s_cidx[tid] : -1;

    HS_T(0, 3);
    // survivors' windows (normalised, derotated by the code-0 pattern)
    for (int k = tid; k < HDR; k += H_THREADS) {
        const float sg = k < SOFL ? sof_s(k) : scr_s(k - SOFL);
        #pragma unroll
        for (int c = 0; c < K1; c++) {
            const int ci = s_cidx[c];
            const float2 x = ci >= 0 ? s_f0[ci + k] : make_float2(0.0f, 0.0f);
            cand_win[(size_t)(blockIdx.x * K1 + c) * HDR + k] = fold(x, sg, k & 1);
        }
    }
    if (tid == 0) {
        // power over the block's share of the searched span (input units)
        const bool last = blockIdx.x == gridDim.x - 1;
        const int m = max(0, min(last ? TILE : BLK, n - p0));
        blk_pow[blockIdx.x] = scale > 0.0f ? s_pre[m] / (scale * scale) : 0.0f;
    }
    // publish this block's survivors to the verify block of its pair
    __threadfence();
    __syncthreads();
    if (tid == 0)
        atomicAdd(pair_done + blockIdx.x / G2, 1u);
    HS_T(0, 4);
}

/* Kernel 2: one block per kernel-1 block (8 survivors). It is
 * launched with programmatic stream serialisation and waits only for its own
 * group (fence + counter), so it runs while kernel 1 drains. Mid statistic on
 * the 8 survivors x MC copies, full statistic of the best two distinct positions on NBF bins around
 * each copy; the last block selects the global top 64. */
__global__ void __launch_bounds__(V_THREADS, 4)
hs_verify_kernel(const int* __restrict__ cand_pos, const float2* __restrict__ cand_win, int n_blocks1,
                 const float* __restrict__ blk_pow, int n, int first_pos, int index_offset,
                 const float2* __restrict__ rot_mid, const float2* __restrict__ rot_fine,
                 float* __restrict__ vscore, int* __restrict__ vpos, int* __restrict__ mid_pos, int n_groups,
                 unsigned* __restrict__ done, unsigned* __restrict__ pair_done,
                 pl_header_candidates* __restrict__ out)
{
    constexpr int NC = G2 * K1;               // survivors per group
    constexpr int NS = NC * MC;               // mid slots
    constexpr int U_MID = NS * HDR;           // float2
    static_assert(NBF * HDR <= U_MID, "full bins fit the mid buffer");
    // Keep the static allocation below CUDA's portable 48 KiB limit. The
    // combined workspace exceeds that limit on sm_121 and needs an explicit
    // dynamic-shared-memory opt-in, even though the device has enough capacity.
    extern __shared__ float2 s_u[];          // mid copies, then full bins, then selection
    __shared__ float2 s_win[NC][HDR];        // derotated windows
    __shared__ float2 s_rm[MC * HDR];        // mid copy rotations
    __shared__ float2 s_dt[NBF * HDR];       // fine offsets around a copy
    __shared__ float2 s_xw[VERIFY_PER_GROUP][HDR];
    __shared__ float2 s_S[NBF];
    __shared__ float s_e[NC];
    __shared__ int s_pos[NC];
    __shared__ float s_msof[NS];
    __shared__ float s_mpl[NS][4];
    __shared__ int s_sel[VERIFY_PER_GROUP];
    __shared__ unsigned s_best;
    __shared__ bool s_last;
    __shared__ float s_red[V_THREADS / 32];
    __shared__ int s_ri[V_THREADS / 32];
    float2 (*s_mid)[HDR] = reinterpret_cast<float2 (*)[HDR]>(s_u);  // [NS][HDR]
    float2 (*s_full)[HDR] = reinterpret_cast<float2 (*)[HDR]>(s_u); // [NBF][HDR]
    float* s_sel_sc = reinterpret_cast<float*>(s_u);
    int* s_sel_pos = reinterpret_cast<int*>(s_u) + SEL_SMEM;

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    HS_T(1, 0);
    // tables (independent of kernel 1) while kernel 1 is still running
    for (int i = tid; i < MC * HDR; i += V_THREADS)
        s_rm[i] = __ldg(rot_mid + i);
    for (int i = tid; i < NBF * HDR; i += V_THREADS)
        s_dt[i] = __ldg(rot_fine + i);
    // Wait only for the kernel-1 blocks of this group (fence + counter). With
    // programmatic dependent launch this block started after every kernel-1
    // block had started (they trigger at entry), so waiting cannot block them.
    if (tid == 0) {
        const unsigned need = (unsigned)min(G2, n_blocks1 - G2 * (int)blockIdx.x);
        volatile unsigned* f = pair_done + blockIdx.x;
        while (*f < need)
            __nanosleep(32);
        __threadfence();
        *f = 0u; // reset for the next launch (stream-ordered after this kernel)
        s_best = 0u;
    }
    __syncthreads();
    HS_T(1, 1);
    const int c0 = blockIdx.x * NC;
    const int nvalid = min(NC, (n_blocks1 - G2 * (int)blockIdx.x) * K1);
    if (tid < NC)
        s_pos[tid] = tid < nvalid ? __ldcg(cand_pos + c0 + tid) : -1;
    for (int i = tid; i < NC * HDR; i += V_THREADS)
        s_win[i / HDR][i % HDR] = i < nvalid * HDR ? __ldcg(cand_win + (size_t)c0 * HDR + i)
                                                    : make_float2(0.0f, 0.0f);
    __syncthreads();
    static_assert(NC == V_THREADS / 32, "one warp per survivor");
    {
        float e = 0.0f;
        for (int k = lane; k < HDR; k += 32)
            e += s_win[wid][k].x * s_win[wid][k].x + s_win[wid][k].y * s_win[wid][k].y;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            e += __shfl_xor_sync(0xffffffffu, e, off);
        if (lane == 0)
            s_e[wid] = e;
    }
    HS_T(1, 6);
    // mid copies: thread (k, half of the slots)
    static_assert(2 * HDR <= V_THREADS && NS % 2 == 0, "copy layout");
    if (tid < 2 * HDR) {
        const int k = tid % HDR, h = tid / HDR;
        #pragma unroll
        for (int j = 0; j < NS / 2; j++) {
            const int slot = h * (NS / 2) + j;
            s_mid[slot][k] = cmul(s_win[slot / MC][k], s_rm[(slot % MC) * HDR + k]);
        }
    }
    __syncthreads();
    static_assert(2 * NS <= V_THREADS, "SOF halves");
    if (wid < (2 * NS + 31) / 32) { // SOF halves per slot, whole warps
        const int slot = min(tid >> 1, NS - 1), h = tid & 1;
        float2 a = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int k = 0; k < 13; k++)
            a = a + s_mid[slot][13 * h + k];
        const float m = amag(a);
        const float o = __shfl_xor_sync(0xffffffffu, m, 1);
        if (h == 0 && tid < 2 * NS)
            s_msof[slot] = W_SOF * (m + o);
    }
    // mid PLSC: (hyp, slot) tasks x 4 lanes; NS multiple of 8: hyp uniform per warp
    static_assert(NS % 8 == 0 && (4 * NS * 4) % V_THREADS == 0, "mid rounds");
    #pragma unroll
    for (int base = 0; base < 4 * NS * 4; base += V_THREADS) {
        const int t = base + tid;
        const int task = t >> 2, q = t & 3;
        const int hyp = task / NS, slot = task % NS;
        const float pl = mid_lane(s_mid[slot] + SOFL, q, hyp);
        if (q == 0)
            s_mpl[slot][hyp] = pl;
    }
    __syncthreads();
    if (wid == 0) {
        float score = -1.0f;
        static_assert(NS == 32, "one lane per slot");
        const int candidate = lane / MC;
        if (s_pos[candidate] >= 0 && s_e[candidate] > 1e-30f)
            score = (s_msof[lane] + fmaxf(fmaxf(s_mpl[lane][0], s_mpl[lane][1]),
                                         fmaxf(s_mpl[lane][2], s_mpl[lane][3]))) * rsqrtf(s_e[candidate]);
        // Preserve separate spatial hypotheses, rather than two frequency
        // copies of the same position, until the coherent score is known.
        for (int rank = 0; rank < VERIFY_PER_GROUP; rank++) {
            float bs = score;
            int bi = score >= 0.0f ? lane : -1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float os = __shfl_down_sync(0xffffffffu, bs, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (better(os, oi, bs, bi)) {
                    bs = os;
                    bi = oi;
                }
            }
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (lane == 0) {
                s_sel[rank] = bi;
                mid_pos[blockIdx.x * VERIFY_PER_GROUP + rank] = bi >= 0 ? s_pos[bi / MC] : -1;
            }
            if (bi >= 0) {
                for (int k = lane; k < HDR; k += 32)
                    s_xw[rank][k] = s_mid[bi][k];
                if (candidate == bi / MC)
                    score = -1.0f;
            }
        }
    }
    __syncthreads();
    for (int rank = 0; rank < VERIFY_PER_GROUP; rank++) {
        const int sel = s_sel[rank];
        if (tid == 0)
            s_best = 0u;
        __syncthreads();
        HS_T(1, 7);
        if (sel >= 0) {
            if (tid < HDR) {
                const float2 x = s_xw[rank][tid];
                #pragma unroll
                for (int bb = 0; bb < NBF; bb++)
                    s_full[bb][tid] = cmul(x, s_dt[bb * HDR + tid]);
            }
            __syncthreads();
            if (wid == 0) { // NBF * 2 == 32: SOF sums per bin
                const int bb = lane >> 1, h = lane & 1;
                float2 a = make_float2(0.0f, 0.0f);
                #pragma unroll
                for (int k = 0; k < 13; k++)
                    a = a + s_full[bb][13 * h + k];
                a.x += __shfl_xor_sync(0xffffffffu, a.x, 1);
                a.y += __shfl_xor_sync(0xffffffffu, a.y, 1);
                if (h == 0)
                    s_S[bb] = a;
            }
            __syncthreads();
            static_assert(V_THREADS == 4 * NBF * 4, "one round");
            const int task = tid >> 2, q = tid & 3;
            const int hyp = task / NBF, bb = task % NBF; // NBF multiple of 8: uniform per warp
            const float2 S = s_S[bb];
            float best = full_lane(s_full[bb] + SOFL, q, hyp, S) + S.x * S.x + S.y * S.y;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, off));
            if (lane == 0)
                atomicMax(&s_best, __float_as_uint(best)); // non-negative floats
            __syncthreads();
        }
        if (tid == 0) {
            float sc = -1.0f;
            const int c = sel >= 0 ? sel / MC : 0;
            if (sel >= 0 && s_e[c] > 1e-30f) {
                sc = __uint_as_float(s_best) / (HDR * s_e[c]);
                if (!(sc >= 0.0f) || !isfinite(sc))
                    sc = -1.0f;
            }
            vscore[blockIdx.x * VERIFY_PER_GROUP + rank] = sc;
            vpos[blockIdx.x * VERIFY_PER_GROUP + rank] = sc >= 0.0f ? s_pos[c] : -1;
        }
        __syncthreads();
    }
    HS_T(1, 2);

    // ---- last block: global top-64 with suppression ----
    __threadfence();
    __syncthreads();
    if (tid == 0)
        s_last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last)
        return;
    __threadfence();
    HS_T(1, 4);
    // the result is assembled in shared memory and written to the pinned host
    // structure once (no fence may wait on a host write inside the loop)
    __shared__ int s_oidx[pl_header_candidates::kMaxCandidates];
    __shared__ float s_osc[pl_header_candidates::kMaxCandidates];
    __shared__ float s_opow;
    const int n_sel = n_groups * VERIFY_PER_GROUP;
    const bool in_smem = n_sel <= SEL_SMEM;
    if (in_smem)
        for (int i = tid; i < n_sel; i += V_THREADS) {
            s_sel_sc[i] = __ldcg(vscore + i);
            s_sel_pos[i] = __ldcg(vpos + i);
        }
    {
        // mean power: all threads load at once, then reduce
        float t = 0.0f;
        for (int i = tid; i < n_blocks1; i += V_THREADS)
            t += __ldcg(blk_pow + i);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0)
            s_red[wid] = t;
        if (tid == 0)
            *done = 0u;
    }
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < V_THREADS / 32; w++)
            t += s_red[w];
        s_opow = t;
    }
    __syncthreads();
    HS_T(1, 5);
    int count = 0;
    if (in_smem) {
        // Parallel selection. Suppression: a group is dropped when another
        // group within +-NMS symbols scores better (stage-1 survivors of one
        // block are already NMS apart, so this only acts across groups);
        // survivors are ranked by counting the better ones.
        __shared__ int s_count;
        if (tid == 0)
            s_count = 0;
        constexpr int PERT = (SEL_SMEM + V_THREADS - 1) / V_THREADS;
        bool keep[PERT];
        #pragma unroll
        for (int r = 0; r < PERT; r++) {
            const int i = tid + r * V_THREADS;
            keep[r] = false;
            if (i < n_sel && s_sel_sc[i] > 0.0f) {
                bool k = true;
                const float vi = s_sel_sc[i];
                const int pi = s_sel_pos[i];
                #pragma unroll
                for (int d = -2 * VERIFY_PER_GROUP; d <= 2 * VERIFY_PER_GROUP; d++) { // neighbouring groups
                    const int j = i + d;
                    if (d != 0 && j >= 0 && j < n_sel) {
                        const float vj = s_sel_sc[j];
                        const int pj = s_sel_pos[j];
                        k = k && !(vj > 0.0f && abs(pj - pi) <= NMS && better(vj, j, vi, i));
                    }
                }
                keep[r] = k;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < PERT; r++) {
            const int i = tid + r * V_THREADS;
            if (i < n_sel && !keep[r])
                s_sel_sc[i] = -1.0f;
        }
        __syncthreads();
        // Each warp contributes its Kth-largest score. The minimum of those
        // scores is a lower bound for the global candidate cutoff, so pruning
        // cannot remove any of the requested top candidates.
        constexpr int WARPS = V_THREADS / 32;
        constexpr int BOUND_K = (pl_header_candidates::kMaxCandidates + WARPS - 1) / WARPS;
        static_assert(BOUND_K <= 32, "warp-local cutoff fits");
        __shared__ float s_cutoff[WARPS];
        __shared__ int s_nl;
        __shared__ float s_lv[SEL_SMEM];
        __shared__ int s_li[SEL_SMEM];
        float best[BOUND_K];
        #pragma unroll
        for (int j = 0; j < BOUND_K; j++)
            best[j] = -1.0f;
        #pragma unroll
        for (int r = 0; r < PERT; r++) {
            const int i = tid + r * V_THREADS;
            float value = i < n_sel ? s_sel_sc[i] : -1.0f;
            #pragma unroll
            for (int j = 0; j < BOUND_K; j++) {
                const float hi = fmaxf(best[j], value);
                value = fminf(best[j], value);
                best[j] = hi;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other[BOUND_K];
            #pragma unroll
            for (int j = 0; j < BOUND_K; j++)
                other[j] = __shfl_xor_sync(0xffffffffu, best[j], off);
            #pragma unroll
            for (int j = 0; j < BOUND_K; j++) {
                float value = other[j];
                #pragma unroll
                for (int k = 0; k < BOUND_K; k++) {
                    const float hi = fmaxf(best[k], value);
                    value = fminf(best[k], value);
                    best[k] = hi;
                }
            }
        }
        if (lane == 0)
            s_cutoff[wid] = best[BOUND_K - 1];
        if (tid == 0)
            s_nl = 0;
        __syncthreads();
        float bound = s_cutoff[0];
        #pragma unroll
        for (int w = 1; w < WARPS; w++)
            bound = fminf(bound, s_cutoff[w]);
        #pragma unroll
        for (int r = 0; r < PERT; r++) {
            const int i = tid + r * V_THREADS;
            if (i < n_sel && keep[r] && s_sel_sc[i] >= bound) {
                const int k = atomicAdd(&s_nl, 1);
                s_lv[k] = s_sel_sc[i];
                s_li[k] = i;
            }
        }
        __syncthreads();
        const int nl = s_nl;
        const int nl8 = (nl + 7) & ~7;
        for (int k = nl + tid; k < nl8; k += V_THREADS) {
            s_lv[k] = -2.0f; // padding below every score
            s_li[k] = 0x7fffffff;
        }
        __syncthreads();
        for (int k = tid; k < nl; k += V_THREADS) {
            const float v = s_lv[k];
            const int i = s_li[k];
            int rank = 0;
            for (int j0 = 0; j0 < nl8; j0 += 8) {
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    const float w = s_lv[j0 + d];
                    const int li = s_li[j0 + d];
                    rank += (int)(w > v) | ((int)(w == v) & (int)(li < i)); // branch-free
                }
            }
            if (rank < pl_header_candidates::kMaxCandidates) {
                s_oidx[rank] = s_sel_pos[i] - index_offset;
                s_osc[rank] = v;
                atomicAdd(&s_count, 1);
            }
        }
        __syncthreads();
        count = s_count;
    } else {
    for (int c = 0; c < pl_header_candidates::kMaxCandidates; c++) {
            float bs = -1.0f;
            int bi = -1;
            for (int i = tid; i < n_sel; i += V_THREADS) {
                const float v = in_smem ? s_sel_sc[i] : __ldcg(vscore + i);
                if (v > bs) {
                    bs = v;
                    bi = i;
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float os = __shfl_down_sync(0xffffffffu, bs, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (better(os, oi, bs, bi)) {
                    bs = os;
                    bi = oi;
                }
            }
            if (lane == 0) {
                s_red[wid] = bs;
                s_ri[wid] = bi;
            }
            __syncthreads();
            float b = s_red[0];
            int ii = s_ri[0];
            #pragma unroll
            for (int w = 1; w < V_THREADS / 32; w++)
                if (better(s_red[w], s_ri[w], b, ii)) {
                    b = s_red[w];
                    ii = s_ri[w];
                }
            if (!(b > 0.0f) || ii < 0)
                break; // uniform
            const int sel_pos = in_smem ? s_sel_pos[ii] : __ldcg(vpos + ii);
            if (tid == 0) {
                s_oidx[count] = sel_pos - index_offset;
                s_osc[count] = b;
            }
            count++;
            __syncthreads(); // everyone has read s_red / s_ri
            for (int i = tid; i < n_sel; i += V_THREADS) {
                const int q = in_smem ? s_sel_pos[i] : __ldcg(vpos + i);
                if (abs(q - sel_pos) <= NMS) {
                    if (in_smem)
                        s_sel_sc[i] = -1.0f;
                    else
                        __stcg(vscore + i, -1.0f);
                }
            }
            __syncthreads();
        }
    }
    if (tid < pl_header_candidates::kMaxCandidates) {
        out->index[tid] = tid < count ? s_oidx[tid] : 0;
        out->score[tid] = tid < count ? s_osc[tid] : 0.0f;
    }
    HS_T(1, 3);
    if (tid == 0) {
        out->count = count;
        out->mean_power = n > first_pos ? s_opow / (n - first_pos) : 0.0f;
        out->searched = n - index_offset;
    }
}

} // namespace

plheader_search_cuda::plheader_search_cuda(int max_symbols)
{
    if (max_symbols < HDR)
        max_symbols = HDR;
    d_max_symbols = max_symbols;
    d_max_blocks = (max_symbols + BLK - 1) / BLK;

    // host-side check of the device constants against the PL definitions
    {
        complexf h[HDR];
        plheader_symbols(0x80 | 0x01, h); // b0 = 1, b7 = 1
        for (int k = 0; k < SOFL; k++) {
            const float s = sof_s(k);
            const complexf ref = (float)M_SQRT1_2 * (k & 1 ? complexf(-s, s) : complexf(s, s));
            if (std::abs(ref - h[k]) > 1e-5f)
                throw std::logic_error("plheader_search_cuda: SOF constant mismatch");
        }
        // pair 0 of PLS code 0x81: y_0 = g_0 (b0 = 1), second copy inverted
        const uint64_t cw = pls_codeword(0x81);
        for (int t = 0; t < NPAIR; t++)
            if ((((cw >> (63 - 2 * t)) & 1) != 0) != (g_s(t) < 0))
                throw std::logic_error("plheader_search_cuda: S2X row mismatch");
        for (int q = 0; q < 8; q++)
            if (!seg_walsh(q) && seg_fit(q) < 0)
                throw std::logic_error("plheader_search_cuda: segment fit");
    }

    {
        __half2 hc[4][13], hs[4][13];
        for (int b = 0; b < 4; b++)
            for (int m = 0; m < 13; m++) {
                const double ph = -2.0 * M_PI * b / 52.0 * 2 * m;
                hc[b][m] = __float2half2_rn((float)std::cos(ph));
                hs[b][m] = __float2half2_rn((float)std::sin(ph));
            }
        CUDA_CHECK(cudaMemcpyToSymbol(c_sof_c, hc, sizeof(hc)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_sof_s, hs, sizeof(hs)));
    }
    {
        int dev = 0, major = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
        d_pdl = major >= 9;
    }
    CUDA_CHECK(cudaFuncSetAttribute(hs_cheap_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    (int)cudaSharedmemCarveoutMaxShared));
    CUDA_CHECK(cudaFuncSetAttribute(hs_verify_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    (int)cudaSharedmemCarveoutMaxShared));
    CUDA_CHECK(cudaFuncSetAttribute(hs_verify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)V_SMEM_BYTES));
    CUDA_CHECK(cudaMalloc(&d_sym_dev, (size_t)max_symbols * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_cand_pos, (size_t)d_max_blocks * K1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_pow, (size_t)d_max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vscore, (size_t)(d_max_blocks + G2) * VERIFY_PER_GROUP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vpos, (size_t)(d_max_blocks + G2) * VERIFY_PER_GROUP * sizeof(int)));
    {
        // e^{-j 2 pi f k}: mid copies f = (cp - 1.5) MSTEP
        std::vector<float2> rm(MC * HDR), rf(NBF * HDR);
        for (int cp = 0; cp < MC; cp++)
            for (int k = 0; k < HDR; k++) {
                const double ph = -2.0 * M_PI * (cp - 0.5 * (MC - 1)) * (double)MSTEP * k;
                rm[cp * HDR + k] = make_float2((float)std::cos(ph), (float)std::sin(ph));
            }
        // full statistic: offsets (b - 7.5) VSTEP from the mid copy
        for (int b = 0; b < NBF; b++)
            for (int k = 0; k < HDR; k++) {
                const double ph = -2.0 * M_PI * (b - 0.5 * (NBF - 1)) * (double)VSTEP * k;
                rf[b * HDR + k] = make_float2((float)std::cos(ph), (float)std::sin(ph));
            }
        std::vector<float2> rp(TILE + 1);
        for (int j = 0; j <= TILE; j++) {
            const double ph = 2.0 * M_PI * (double)PF * j;
            rp[j] = make_float2((float)std::cos(ph), (float)std::sin(ph));
        }
        CUDA_CHECK(cudaMalloc(&d_rot_pf, rp.size() * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_rot_pf, rp.data(), rp.size() * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_cand_win, (size_t)d_max_blocks * K1 * HDR * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_mid_pos, (size_t)d_max_blocks * VERIFY_PER_GROUP * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rot_mid, rm.size() * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_rot_fine, rf.size() * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_rot_mid, rm.data(), rm.size() * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rot_fine, rf.data(), rf.size() * sizeof(float2), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_done, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_done, 0, sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_pair_done, (size_t)(d_max_blocks / G2 + 1) * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_pair_done, 0, (size_t)(d_max_blocks / G2 + 1) * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocHost(&d_result_host, sizeof(pl_header_candidates)));
    *d_result_host = pl_header_candidates{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));
}

plheader_search_cuda::~plheader_search_cuda()
{
    cudaFree(d_sym_dev);
    cudaFree(d_cand_pos);
    cudaFree(d_block_pow);
    cudaFree(d_vscore);
    cudaFree(d_vpos);
    cudaFree(d_rot_mid);
    cudaFree(d_rot_pf);
    cudaFree(d_cand_win);
    cudaFree(d_mid_pos);
    cudaFree(d_rot_fine);
    cudaFree(d_done);
    cudaFree(d_pair_done);
    if (d_result_host)
        cudaFreeHost(d_result_host);
    if (d_stream)
        cudaStreamDestroy(d_stream);
}

void plheader_search_cuda::launch(const float* sym, int n_symbols, int index_offset,
                                  cudaStream_t stream)
{
    if (n_symbols > d_max_symbols) {
        fprintf(stderr, "plheader_search_cuda: %d symbols > max %d (truncated)\n", n_symbols,
                d_max_symbols);
        n_symbols = d_max_symbols;
    }
    if (n_symbols < 0)
        n_symbols = 0;
    if (index_offset < 0)
        index_offset = 0;
    // headers fully inside the previous buffer were reported there
    const int first_pos = std::max(0, index_offset - (HDR - 1));
    // positions first_pos .. n - HDR, BLK per kernel-1 block
    const int n_pos = n_symbols - (HDR - 1) - first_pos;
    const int n_blocks = std::max(1, (n_pos + BLK - 1) / BLK);
    const int n_cand = n_blocks * K1;
    const float2* s = reinterpret_cast<const float2*>(sym);
    hs_cheap_kernel<<<n_blocks, H_THREADS, 0, stream>>>(s, n_symbols, first_pos, d_rot_pf, d_cand_pos,
                                                         d_cand_win, d_block_pow, d_pair_done);
    CUDA_CHECK_LAUNCH();
    const int n_groups = (n_blocks + G2 - 1) / G2;
    {
        // programmatic dependent launch (sm_90): kernel 2 blocks take the SM
        // slots that kernel-1 blocks free and wait for their own group only
        cudaLaunchConfig_t lc = {};
        lc.gridDim = dim3(n_groups);
        lc.blockDim = dim3(V_THREADS);
        lc.dynamicSmemBytes = V_SMEM_BYTES;
        lc.stream = stream;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = d_pdl ? 1 : 0;
        lc.attrs = at;
        lc.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&lc, hs_verify_kernel, (const int*)d_cand_pos, (const float2*)d_cand_win,
                                      n_blocks, (const float*)d_block_pow, n_symbols, first_pos, index_offset,
                                      (const float2*)d_rot_mid, (const float2*)d_rot_fine, d_vscore, d_vpos,
                                      d_mid_pos, n_groups, d_done, d_pair_done, d_result_host));
    }
    CUDA_CHECK_LAUNCH();
    d_last_cands = n_cand;
    d_last_groups = n_groups * VERIFY_PER_GROUP;
}

std::vector<int> plheader_search_cuda::stage2_positions() const
{
    std::vector<int> pos(d_last_groups);
    if (d_last_groups > 0)
        CUDA_CHECK(cudaMemcpy(pos.data(), d_mid_pos, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> out;
    for (int p : pos)
        if (p >= 0)
            out.push_back(p);
    return out;
}

pl_header_candidates plheader_search_cuda::result() const { return *d_result_host; }

pl_header_candidates plheader_search_cuda::search(const complexf* sym_host, int n_symbols,
                                                  int index_offset)
{
    if (n_symbols > d_max_symbols)
        n_symbols = d_max_symbols;
    if (n_symbols > 0)
        CUDA_CHECK(cudaMemcpyAsync(d_sym_dev, sym_host, (size_t)n_symbols * sizeof(float2),
                                   cudaMemcpyHostToDevice, d_stream));
    launch(reinterpret_cast<const float*>(d_sym_dev), n_symbols, index_offset, d_stream);
    CUDA_CHECK(cudaStreamSynchronize(d_stream));
    return result();
}

std::vector<int> plheader_search_cuda::stage1_positions() const
{
    std::vector<int> pos(d_last_cands);
    if (d_last_cands > 0)
        CUDA_CHECK(cudaMemcpy(pos.data(), d_cand_pos, pos.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> out;
    for (int p : pos)
        if (p >= 0)
            out.push_back(p);
    return out;
}

} // namespace cuda
} // namespace cusat_dsp
