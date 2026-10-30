/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA LDPC decoder — layered offset min-sum (FACTOR=2, NormalUpdate).
 *
 * Architecture: one CUDA block per codeword, 2*M=720 threads per block.
 * Each schedule step processes up to two independent LDPC layers in parallel;
 * batch parallelism fills the GPU grid.
 *
 * Algorithm: layered offset min-sum over the IRA parity structure with:
 *   - int16 or int8 APP storage, FACTOR=2 fixed-point scaling
 *   - Offset beta = nearbyint(0.5 * 2) = 1
 *   - BNL messages stored as int8, clamped to [-127, 127]
 *   - Early termination via syndrome check after each full iteration
 *
 */

#include "ldpc_decode_kernel.cuh"
#include "cuda_util.cuh"
#include "ldpc_message_update.h"
#include "ldpc_tables/ldpc_code.h"
#include "ldpc_schedules/ldpc_schedule_registry.h"
#include <cstdio>
#include <cstring>
#include <vector>

namespace cusat_dsp {
namespace cuda {

/* ---------- Device helper functions ---------- */

/* Saturating absolute value */
__device__ __forceinline__ int16_t qabs(int16_t a)
{
    return (a == -32768) ? 32767 : (a < 0 ? -a : a);
}

/* Saturating unsigned subtract (result >= 0) */
__device__ __forceinline__ uint16_t qsubu(uint16_t a, uint16_t b)
{
    return a > b ? a - b : 0;
}

/* sign(a, b): return b<0 ? -a : b>0 ? a : 0 */
__device__ __forceinline__ int16_t qsign(int16_t a, int16_t b)
{
    return b < 0 ? -a : (b > 0 ? a : 0);
}

/* Saturating int -> int16 cast. The extrinsic tau = APP - old_message can
 * reach +-(32767 + 127) when the APP is railed (large demapper precision at
 * mid/high Es/N0); a raw (int16_t) cast wraps it to the OPPOSITE sign at
 * near-maximal confidence, poisoning the check-node update and driving the
 * decoder to a far codeword (whole-frame error floors on dense modcods). */
__device__ __forceinline__ int16_t qsat16(int v)
{
    return (int16_t)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
}

/* EXPERIMENT (CUSAT_LDPC_BNL_BITS env): quantize stored check-to-variable
 * messages to a coarser uniform grid inside the int8 container — emulates a
 * sub-8-bit BNL message width for FER studies without changing storage.
 * g_bnl_qstep = 0 disables (default). */
/* EXPERIMENT (CUSAT_LDPC_NMS_ALPHA env): normalized min-sum — scale the raw
 * check-node output by alpha = g_nms_num/16 before clamping. Multiplicative
 * correction matches the boxplus error's proportional shape, unlike the fixed
 * offset beta, so it should recover low-rate (small-LLR) threshold loss.
 * 0 = off. Applied identically in per-edge and check-state paths. */
__device__ int g_nms_num = 0;

/* EXPERIMENT (CUSAT_LDPC_OMS_BETA env): runtime offset for offset-min-sum,
 * in FACTOR=2 counts (default 1 = the deployed beta). 0 = plain min-sum. */
__device__ int g_oms_beta = 1;

__device__ __forceinline__ int nms_scale(int v)
{
    const int n = g_nms_num;   /* alpha in 64ths (0 = off) */
    if (n == 0) return v;
    /* Odd-symmetric rounding: scale the MAGNITUDE, reapply the sign. The
     * previous form ((v*n ± 32) >> 6) floors, so negative values rounded
     * away from zero one step earlier than positive ones (+1 -> 1 but
     * -1 -> -2): a systematic negative message bias, and a violation of
     * min-sum's two-magnitude-per-check invariant that the compressed
     * check-state (cst) encoding depends on — under alpha != 1 it made
     * equal-magnitude opposite-sign messages scale to DIFFERENT
     * magnitudes, corrupting every cst reconstruction on alpha-tuned
     * modcods. */
    const int m = ((v < 0 ? -v : v) * n + 32) >> 6;
    return v < 0 ? -m : m;
}

__device__ int g_bnl_qstep = 0;          /* 0=off, else = number of levels */
__constant__ int8_t c_bnl_levels[16];    /* ascending log-spaced magnitudes */

__device__ __forceinline__ int8_t bnl_quantize(int8_t m)
{
    const int n = g_bnl_qstep;
    if (n == 0) return m;
    const int a = m < 0 ? -(int)m : (int)m;
    int best = c_bnl_levels[0], bd = a - (int)c_bnl_levels[0];
    if (bd < 0) bd = -bd;
    for (int k = 1; k < n; k++) {
        int d = a - (int)c_bnl_levels[k];
        if (d < 0) d = -d;
        if (d < bd) { bd = d; best = c_bnl_levels[k]; }
    }
    return (int8_t)(m < 0 ? -best : best);
}

__device__ __forceinline__ cusat_dsp::cuda::ldpc_bnl_update_t
bnl_quant_update(cusat_dsp::cuda::ldpc_bnl_update_t u)
{
    if (g_bnl_qstep) {
        const int8_t mq = bnl_quantize(u.message);
        u.delta -= (int)u.message - (int)mq;
        u.message = mq;
    }
    return u;
}

/* EXPERIMENT (CUSAT_LDPC_CST env): lossless compressed check-state message
 * storage. Min-sum outgoing messages of one check take at most two distinct
 * magnitudes (min1 to every edge except the arg-min edge, which gets min2),
 * so the whole per-check message set packs into {lo, hi, per-edge hi-bit,
 * per-edge sign-bit} = 16 B per check row instead of deg int8 edges (deg up
 * to 30 at rate 9/10). Reconstruction reproduces the per-edge post-clamp
 * int8 values EXACTLY, so decode is bit-identical; only memory traffic
 * changes. Experimental: state pointer is a process-global symbol — single
 * decoder instance only (bench); NOT for the multi-chain ACM pool. */
struct cst_t {
    int8_t lo, hi;
    uint16_t pad;
    uint32_t signs;   /* per-edge sign bits (deg <= 32) */
    uint32_t his;     /* per-edge "magnitude == hi" bits */
    uint32_t pad2;
};
__device__ int g_cst = 0;
__device__ int g_cst_fmt = 16;    /* tuple bytes in memory: 16, 8, or 4 */
__device__ int g_dbg_syn = 0;     /* CUSAT_LDPC_DBG_SYN: dump failing checks */
/* CUSAT_LDPC_DBG_CST: shadow the tuple with the dense per-edge array and
 * report reconstruction mismatches (budgeted). */
__device__ int g_dbg_cst = 0;
__device__ int g_dbg_cst_budget = 0;
__device__ cst_t* g_cst_ptr = nullptr;

/* SLIM tuple formats (CUSAT_LDPC_CST_FMT): min-sum gives the hi magnitude to
 * at most ONE edge (the arg-min edge; ties collapse to lo==hi), so the his
 * BITMAP of the 16 B format reduces to a 5-bit index (sentinel 31/15 =
 * none). Bit-exact same reconstruction, half/quarter the memory traffic.
 *   8 B (deg <= 32): signs[31:0] lo[38:32] hi[45:39] idx[50:46]
 *   4 B (deg <= 12): signs[11:0] lo[18:12] hi[25:19] idx[29:26]
 * cst_t stays the REGISTER representation; only the memory image changes. */
__device__ __forceinline__ void cst_store(void* p, const int16_t* msgs, int deg)
{
    int v1 = msgs[0] < 0 ? -msgs[0] : msgs[0];
    int v2 = v1;
    for (int c = 1; c < deg; c++) {
        int v = msgs[c] < 0 ? -msgs[c] : msgs[c];
        if (v != v1) v2 = v;
    }
    const int lo = v1 < v2 ? v1 : v2;
    const int hi = v1 < v2 ? v2 : v1;
    uint32_t signs = 0, his = 0;
    for (int c = 0; c < deg; c++) {
        int v = msgs[c] < 0 ? -msgs[c] : msgs[c];
        if (msgs[c] < 0) signs |= 1u << c;
        if (hi != lo && v == hi) his |= 1u << c;
    }
    if (g_cst_fmt == 8) {
        const uint64_t idx = his ? (uint64_t)(__ffs(his) - 1) : 31u;
        *(uint64_t*)p = (uint64_t)signs
                      | ((uint64_t)(uint32_t)lo << 32)
                      | ((uint64_t)(uint32_t)hi << 39)
                      | (idx << 46);
    } else if (g_cst_fmt == 4) {
        const uint32_t idx = his ? (uint32_t)(__ffs(his) - 1) : 15u;
        *(uint32_t*)p = (signs & 0xfffu)
                      | ((uint32_t)lo << 12)
                      | ((uint32_t)hi << 19)
                      | (idx << 26);
    } else {
        cst_t t;
        t.lo = (int8_t)lo; t.hi = (int8_t)hi; t.pad = 0;
        t.signs = signs; t.his = his; t.pad2 = 0;
        *(int4*)p = *(const int4*)&t;
    }
}

__device__ __forceinline__ cst_t cst_load_row(const void* p)
{
    cst_t t;
    if (g_cst_fmt == 8) {
        const uint64_t w = *(const uint64_t*)p;
        t.signs = (uint32_t)(w & 0xffffffffu);
        t.lo = (int8_t)((w >> 32) & 0x7f);
        t.hi = (int8_t)((w >> 39) & 0x7f);
        const int idx = (int)((w >> 46) & 0x1f);
        t.his = (idx < 31) ? (1u << idx) : 0u;
    } else if (g_cst_fmt == 4) {
        const uint32_t w = *(const uint32_t*)p;
        t.signs = w & 0xfffu;
        t.lo = (int8_t)((w >> 12) & 0x7f);
        t.hi = (int8_t)((w >> 19) & 0x7f);
        const int idx = (int)((w >> 26) & 0xf);
        t.his = (idx < 15) ? (1u << idx) : 0u;
    } else {
        *(int4*)&t = *(const int4*)p;
    }
    return t;
}

/* Byte address of the tuple for (cw, layer i, check j). */
__device__ __forceinline__ void* cst_addr(int cw, int q, int i, int M, int j)
{
    return (char*)g_cst_ptr
         + (((size_t)cw * q + i) * M + j) * (size_t)g_cst_fmt;
}

__device__ __forceinline__ int16_t cst_load_edge(const cst_t& t, int c)
{
    const int mag = ((t.his >> c) & 1) ? (int)t.hi : (int)t.lo;
    return (int16_t)(((t.signs >> c) & 1) ? -mag : mag);
}

/* ---------- Maximum check-node degree (including 2 parity edges) ---------- */
/* High-rate DVB-S2 normal codes reach CNL = 28 info links per check (e.g.
 * rate 9/10: links_max_cn = 30), plus 2 parity edges. The previous value of
 * 24 (from the "max 22" short-frame figure) silently overflowed the local
 * inp[] array by up to 6 int16 for rates 8/9 and 9/10 — benign by stack
 * layout on sm_90, an illegal access under the sm_75-JIT path on GB10
 * (caught by compute-sanitizer 2026-08-01). Do NOT shrink below 32 without
 * checking max(links_max_cn)+2 over ALL frame sizes and rates. */
#define LDPC_DEC_MAX_DEG 32

/*
 * Offset min-sum finalp for int16, FACTOR=2 (beta=1).
 *
 * Computes: for each edge i, output = sign(other(mags[i], min1, min2),
 *                                          (xor_signs ^ link[i]) | 0x7FFF)
 * where mags[i] = max(qabs(link[i]) - beta, 0)
 */
__device__ __noinline__ void finalp_oms_i16(int16_t* links, int deg)
{
    const uint16_t beta = (uint16_t)g_oms_beta; /* nearbyint(0.5 * FACTOR) = 1 */

    /* Two-pass offset min-sum WITHOUT mags[] array.
     * Pass 1: find min1, min2, xor_signs by computing |link|-beta inline.
     * Pass 2: recompute |link|-beta inline, select other(min1,min2), apply sign.
     * Eliminates 48-byte mags[24] array → reduces stack frame by 33%. */

    /* Pass 1: find two smallest offset magnitudes + sign XOR */
    int16_t min1, min2;
    int16_t xor_signs = links[0];
    {
        uint16_t m0 = qsubu((uint16_t)qabs(links[0]), beta);
        uint16_t m1 = qsubu((uint16_t)qabs(links[1]), beta);
        xor_signs ^= links[1];
        if (m0 <= m1) { min1 = m0; min2 = m1; }
        else          { min1 = m1; min2 = m0; }
    }
    for (int i = 2; i < deg; i++) {
        xor_signs ^= links[i];
        int16_t m = (int16_t)qsubu((uint16_t)qabs(links[i]), beta);
        if (m < min1) { min2 = min1; min1 = m; }
        else if (m < min2) { min2 = m; }
    }

    /* Pass 2: compute output for each edge (recompute mag inline) */
    for (int i = 0; i < deg; i++) {
        int16_t m = (int16_t)qsubu((uint16_t)qabs(links[i]), beta);
        int16_t oth = (m == min1) ? min2 : min1;
        int16_t s = (xor_signs ^ links[i]) | 0x7FFF;
        links[i] = qsign(oth, s);
    }
}

template <int DEG>
__device__ __forceinline__ void finalp_oms_i16_fixed(int16_t* links)
{
    const uint16_t beta = (uint16_t)g_oms_beta;

    int16_t min1, min2;
    int16_t xor_signs = links[0];
    {
        uint16_t m0 = qsubu((uint16_t)qabs(links[0]), beta);
        uint16_t m1 = qsubu((uint16_t)qabs(links[1]), beta);
        xor_signs ^= links[1];
        if (m0 <= m1) { min1 = m0; min2 = m1; }
        else          { min1 = m1; min2 = m0; }
    }

#pragma unroll
    for (int i = 2; i < DEG; i++) {
        xor_signs ^= links[i];
        int16_t m = (int16_t)qsubu((uint16_t)qabs(links[i]), beta);
        if (m < min1) { min2 = min1; min1 = m; }
        else if (m < min2) { min2 = m; }
    }

#pragma unroll
    for (int i = 0; i < DEG; i++) {
        int16_t m = (int16_t)qsubu((uint16_t)qabs(links[i]), beta);
        int16_t oth = (m == min1) ? min2 : min1;
        int16_t s = (xor_signs ^ links[i]) | 0x7FFF;
        links[i] = qsign(oth, s);
    }
}

template <int DEG>
__device__ __forceinline__ void finalp_oms_i16_fixed_auto(int16_t* links)
{
    finalp_oms_i16_fixed<DEG>(links);
}

template <int HOT0 = 0, int HOT1 = 0, int HOT2 = 0>
__device__ __forceinline__ void finalp_oms_i16_dispatch(int16_t* links, int deg)
{
    if constexpr (HOT0 > 0)
        if (deg == HOT0) { finalp_oms_i16_fixed_auto<HOT0>(links); return; }
    if constexpr (HOT1 > 0)
        if (deg == HOT1) { finalp_oms_i16_fixed_auto<HOT1>(links); return; }
    if constexpr (HOT2 > 0)
        if (deg == HOT2) { finalp_oms_i16_fixed_auto<HOT2>(links); return; }
    finalp_oms_i16(links, deg);
}

/* ---------- Parallel-layer schedule ----------
 * The schedule (per-code-rate, from ldpc_schedules/ headers; 2 layers per step,
 * or sequential fallback at 1/step) is held in PER-INSTANCE device memory
 * (ldpc_decode_cuda::d_schedule_dev) and passed to the kernel as an argument.
 * It MUST NOT be a process-global __constant__: rx_fec_pool (ACM) instantiates
 * one decoder per MODCOD, each with a different layer count q. A shared symbol
 * would hold only the last-constructed code's schedule, so a different chain's
 * decode would index pos[]/bnl[] with the wrong code's layer ids. */

/* ---------- Prep kernel: split soft LLRs into data[] + transposed pty[] ---------- */

__global__ void ldpc_prep_kernel(
    const int16_t* __restrict__ soft_in, // [B * N]
    int16_t* data,                       // [B * K]
    int16_t* pty,                        // [B * R]
    int N, int K, int R, int M, int q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cw = blockIdx.y;

    if (idx >= N)
        return;

    const int16_t* src = soft_in + cw * N;

    int16_t v = src[idx];

    if (idx < K) {
        /* Systematic bits: direct copy */
        data[cw * K + idx] = v;
    } else {
        /* Parity bits: transpose from sequential to [layer][M] order.
         * Input parity is in sequential order: p[0], p[1], ..., p[R-1]
         * CPU decoder transposes: pty[M*i + j] = parity[q*j + i]
         * where i = layer (0..q-1), j = thread within layer (0..M-1).
         * So parity index (idx-K) = q*j + i  => i = (idx-K) % q, j = (idx-K) / q */
        int p = idx - K;
        int i = p % q;
        int j = p / q;
        pty[cw * R + M * i + j] = v;
    }
}

/* ---------- Main layered decode kernel ---------- */

/*
 * One block per codeword. M=360 threads per block.
 * Thread j handles check node j within each layer.
 *
 * BNL layout is edge-major within each layer:
 *   bnl_layer_off[i] + edge * M + j
 * where edge covers the cnt information-bit edges plus the two parity edges.
 * Layer 0 has one padded unused byte at edge cnt+1,j=0 because that check node
 * has no wrap-around parity edge:
 *   deg = cnt+1 for i=0,j=0
 *   deg = cnt+2 otherwise
 */

/* Saturating add into the APP storage type (int16 classic, int8 compact).
 * int8 saturates symmetrically at +-127 (like the BNL messages) so qabs
 * never sees -128. */
template <typename APP>
__device__ __forceinline__ APP app_sat(int x)
{
    if (sizeof(APP) == 1)
        return (APP)(x < -127 ? -127 : (x > 127 ? 127 : x));
    return (APP)(x < -32768 ? -32768 : (x > 32767 ? 32767 : x));
}

/* ---- REAL BFP-8 APP mode (set_bfp8): 8-bit APP container with per-frame
 * per-bit-plane exponents from the demapper. All arithmetic runs at full
 * scale in registers: APP loads shift up by the plane exponent, stores round
 * and shift down with saturation. Messages stay full-scale int8 (they mix
 * planes at check nodes). Signs are shift-invariant, so syndrome checks and
 * hard decisions are unchanged. Active only with int8 APP + exponents set;
 * process-global symbols (single decoder instance, bench-first). ---- */
__device__ int g_bfp8_on = 0;
__device__ const uint8_t* g_bfp8_exp = nullptr;  /* [batch * 16] */
__device__ int g_bfp8_rows = 1;

__device__ __forceinline__ int bfp8_ed(int cw, int v)  /* data VN v */
{
    return g_bfp8_on ? (int)g_bfp8_exp[cw * 16 + v / g_bfp8_rows] : 0;
}
__device__ __forceinline__ int bfp8_ep(int cw, int K, int q, int i, int j)
{   /* parity slot (i,j) sits at frame position K + j*q + i */
    return g_bfp8_on ? (int)g_bfp8_exp[cw * 16 + (K + j * q + i) / g_bfp8_rows] : 0;
}
/* Quantize a full-scale message to its destination VN's plane grid
 * (multiples of 2^e). Makes every APP mantissa update EXACT: no per-update
 * rounding accumulation, tau reconstructs exactly next iteration. */
__device__ __forceinline__ int8_t bfp8_msg_grid(int8_t m, int e)
{
    if (e == 0) return m;
    const int half = 1 << (e - 1);
    int v = (int)m;
    int g = ((v >= 0 ? v + half : v - half) >> e) << e;
    const int lim = (127 >> e) << e;
    return (int8_t)(g < -lim ? -lim : (g > lim ? lim : g));
}

/* ---- SPLIT-APP BFP-8 (CUSAT_LDPC_BFP_SPLIT, int8 container): the mutable
 * state is the EXTRINSIC ACCUMULATOR A only (unit scale, exact — messages
 * are int8-native); the channel term stays in the immutable int16 soft
 * buffer and is added on the fly (tau = chan + A - R_old). No exponents,
 * no rounding drift; A saturates at +-127. ---- */
__device__ int g_bfps_on = 0;
/* Retry channel de-weight: >>k (arithmetic, sign-preserving) on every
 * channel read. 0 in normal operation; the host sets it (set_split_shift)
 * for a selective re-decode of syndrome-failed frames — the graded
 * whole-plane rescale that clears the high-SNR capture floor
 * (win31-equivalent), while retried cliff frames were failures anyway. */
__device__ int g_bfps_shift = 0;
__device__ int g_bfps_cur_trial = 0;
/* Channel-read iteration limit: read channel from global only for
 * iterations < g_bfps_chan_iters. Iterations beyond use A-only (int8-flat
 * behavior). 0 = always read (current). Set via CUSAT_LDPC_BFP_CHAN_ITERS. */
__device__ int g_bfps_chan_iters = 0;

__device__ __forceinline__ int bfps_chd(const int16_t* my_soft, int v)
{
    if (!g_bfps_on || !my_soft) return 0;
    if (g_bfps_chan_iters > 0 && g_bfps_cur_trial >= g_bfps_chan_iters) return 0;
    return (int)__ldg(&my_soft[v]) >> g_bfps_shift;
}
__device__ __forceinline__ int bfps_chp(const int16_t* my_soft, int K, int q,
                                        int i, int j)
{
    if (!g_bfps_on || !my_soft) return 0;
    if (g_bfps_chan_iters > 0 && g_bfps_cur_trial >= g_bfps_chan_iters) return 0;
    return (int)__ldg(&my_soft[K + j * q + i]) >> g_bfps_shift;
}

template <typename APP>
__device__ __forceinline__ APP bfp8_st(int full, int e)
{
    if (sizeof(APP) != 1 || e == 0) return app_sat<APP>(full);
    const int m = (full + (full >= 0 ? (1 << (e - 1)) : -(1 << (e - 1)))) >> e;
    return (APP)(m < -127 ? -127 : (m > 127 ? 127 : m));
}

template <typename APP>
__device__ __forceinline__ void pack_hard_from_app(const APP* __restrict__ s_data,
                                                   uint8_t* __restrict__ hard_out,
                                                   int K, int cw,
                                                   const int16_t* __restrict__ chan = nullptr)
{
    if (!hard_out)
        return;

    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int K_bytes = K / 8;
    uint8_t* dst = hard_out + (size_t)cw * K_bytes;

    for (int byte_idx = tid; byte_idx < K_bytes; byte_idx += n_threads) {
        const APP* src = s_data + byte_idx * 8;
        uint8_t byte = 0;
        for (int b = 0; b < 8; b++) {
            int v = (int)src[b]
                    + (chan ? ((int)chan[byte_idx * 8 + b] >> g_bfps_shift) : 0);
            if (v < 0)
                byte |= (1 << (7 - b));
        }
        dst[byte_idx] = byte;
    }
}

template <typename APP, int HOT0 = 0, int HOT1 = 0, int HOT2 = 0>
__device__ __forceinline__ void process_layer_initial_zero_bnl(
    APP* __restrict__ s_data,
    APP* __restrict__ s_pty,
    int8_t* __restrict__ my_bnl,
    const uint16_t* __restrict__ pos,
    const uint8_t* __restrict__ cnc,
    const int* __restrict__ bnl_layer_off,
    int M, int q, int CNL, int i, int j, int cw, int K,
    const int16_t* __restrict__ my_soft,
    void* cst_row = nullptr)
{
    int cnt = cnc[i];
    int deg = cnt + 2 - (i == 0 && j == 0 ? 1 : 0);
    /* split only when this instantiation is the int8 container (see
     * chan_split in the main kernel) */
    const bool bfps_l = (sizeof(APP) == 1) && g_bfps_on;

    int8_t* bl = my_bnl + bnl_layer_off[i] + j;
    int16_t inp[LDPC_DEC_MAX_DEG];

    for (int c = 0; c < cnt; c++) {
        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
        inp[c] = qsat16(bfps_chd(my_soft, v) + (((int)s_data[v]) << bfp8_ed(cw, v)));
    }

    inp[cnt] = qsat16(bfps_chp(my_soft, K, q, i, j) + (((int)s_pty[M * i + j]) << bfp8_ep(cw, K, q, i, j)));

    if (i > 0) {
        inp[cnt + 1] = qsat16(bfps_chp(my_soft, K, q, i - 1, j) + (((int)s_pty[M * (i - 1) + j]) << bfp8_ep(cw, K, q, i - 1, j)));
    } else if (j > 0) {
        inp[cnt + 1] = qsat16(bfps_chp(my_soft, K, q, q - 1, j - 1) + (((int)s_pty[j + (q - 1) * M - 1]) << bfp8_ep(cw, K, q, q - 1, j - 1)));
    }

    int16_t inp_pty0 = inp[cnt];
    int16_t inp_pty1 = (deg > cnt + 1) ? inp[cnt + 1] : 0;

    finalp_oms_i16_dispatch<HOT0, HOT1, HOT2>(inp, deg);

    for (int c = 0; c < cnt; c++) {
        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
        const int ed = bfp8_ed(cw, v);
        const int8_t msg = bfp8_msg_grid(
            bnl_quantize(ldpc_clamp_bnl_message(nms_scale((int)inp[c]))), ed);
        inp[c] = msg;
        s_data[v] = bfp8_st<APP>((((int)s_data[v]) << ed) + (int)msg, ed);
    }

    const int ep0 = bfp8_ep(cw, K, q, i, j);
    const int8_t msg_pty0 = bfp8_msg_grid(
        bnl_quantize(ldpc_clamp_bnl_message(nms_scale((int)inp[cnt]))), ep0);
    inp[cnt] = msg_pty0;
    {
        const int base0 = bfps_l ? (int)s_pty[M * i + j]
                                 : (int)inp_pty0;
        s_pty[M * i + j] = bfp8_st<APP>(base0 + (int)msg_pty0, ep0);
    }

    if (i > 0) {
        const int ep1 = bfp8_ep(cw, K, q, i - 1, j);
        const int8_t msg_pty1 = bfp8_msg_grid(
            bnl_quantize(ldpc_clamp_bnl_message(nms_scale((int)inp[cnt + 1]))), ep1);
        inp[cnt + 1] = msg_pty1;
        const int base1 = bfps_l ? (int)s_pty[M * (i - 1) + j] : (int)inp_pty1;
        s_pty[M * (i - 1) + j] =
            bfp8_st<APP>(base1 + (int)msg_pty1, ep1);
    } else if (j > 0) {
        const int ep1 = bfp8_ep(cw, K, q, q - 1, j - 1);
        const int8_t msg_pty1 = bfp8_msg_grid(
            bnl_quantize(ldpc_clamp_bnl_message(nms_scale((int)inp[cnt + 1]))), ep1);
        inp[cnt + 1] = msg_pty1;
        const int base1 = bfps_l ? (int)s_pty[j + (q - 1) * M - 1] : (int)inp_pty1;
        s_pty[j + (q - 1) * M - 1] =
            bfp8_st<APP>(base1 + (int)msg_pty1, ep1);
    }

    if (cst_row) {
        cst_store(cst_row, inp, deg);
        if (g_dbg_cst)
            for (int c = 0; c < deg; c++)
                bl[c * M] = (int8_t)inp[c];
    } else {
        for (int c = 0; c < deg; c++) {
            bl[c * M] = (int8_t)inp[c];
        }
    }
}

template <bool UseSharedMem, typename APP, int HOT0 = 0, int HOT1 = 0, int HOT2 = 0>
__device__ __forceinline__ void ldpc_layered_body(
    int16_t* __restrict__ data,                      // [B * K]
    int16_t* __restrict__ pty,                       // [B * R]
    int8_t* __restrict__ bnl,                        // [B * LT] — int8 (OMS messages are small)
    const uint16_t* __restrict__ pos,                // [R * CNL]
    const uint8_t* __restrict__ cnc,                 // [q]
    const int* __restrict__ bnl_layer_off,           // [q+1]
    int* __restrict__ result,                        // [B]
    uint8_t* __restrict__ hard_out,                   // [B * K/8], optional fused output
    int K, int R, int M, int q, int CNL, int LT,
    int max_trials,
    const int16_t* __restrict__ soft_in,              // [B * N] (optional: fused prep)
    int N,                                            // soft input length
    int syndrome_period,                              // trials between checks
    int llr_clamp,                                    // channel-LLR clamp (int8 APP)
    const int16_t* __restrict__ sched,                // [sched_steps * LANES_PER_STEP] per-instance
    int sched_steps,                                  // number of schedule steps
    char* s_mem_raw)                                  // dynamic shared memory
{
    int cw = blockIdx.x;      /* codeword index */
    int j = threadIdx.x % M;    /* check node index within layer (0..M-1) */
    int lane = threadIdx.x / M; /* layer lane within the parallel group (0..1) */

    int16_t* g_data = data + cw * K;
    int16_t* g_pty  = pty  + cw * R;
    int8_t* my_bnl  = bnl  + cw * LT;

    /* Fast path: load K+R LLRs into shared memory (5ns vs 50ns L2).
     * APP=int8 halves the footprint (126 KB -> 63 KB) so two codeword
     * blocks co-reside per SM. Channel LLRs are clamped to +-llr_clamp on
     * load: extrinsic messages keep headroom inside +-127 to out-vote a
     * saturated channel during impaired stretches (high-SNR capture
     * lesson: confidence clipping must hit the CHANNEL, not the
     * extrinsics).
     * Fallback: operate directly on L2-pinned global memory when the device
     * shared memory optin limit is smaller than the K+R working set.
     * __syncthreads() fences global writes between parallel-layer steps,
     * so correctness is preserved in both cases. */
    APP* s_data;
    APP* s_pty;
    int* s_converged_p;
    if constexpr (UseSharedMem) {
        s_data = (APP*)s_mem_raw;
        s_pty = s_data + K;
        s_converged_p = (int*)(s_mem_raw + (size_t)(K + R) * sizeof(APP));
    } else {
        s_data = g_data;
        s_pty = g_pty;
        s_converged_p = (int*)s_mem_raw;  // only sizeof(int) allocated
    }

    /* Input clamp: int16 APP keeps full range (behavior unchanged). */
    const int in_lim = (sizeof(APP) == 1) ? llr_clamp : 32767;

    /* Fused prep: use ALL 720 threads for faster init. */
    int tid = threadIdx.x;
    int n_threads = blockDim.x; /* 720 = 2*M */
    const bool bfp8 = (sizeof(APP) == 1) && g_bfp8_on;
    const bool bfps = (sizeof(APP) == 1) && g_bfps_on;
    /* cst is an int8-container mode; the int16 instance (e.g. the salvage
     * stage) must never take the tuple path even though g_cst is a shared
     * process-global. Compile-time dead for APP=int16. */
    const bool cst_on = (sizeof(APP) == 1) && g_cst;
    const int16_t* my_soft_g = soft_in ? soft_in + cw * N : nullptr;
    /* Channel pointer for SPLIT-mode on-the-fly injection. MUST be nullptr
     * unless this instantiation really is split (int8 APP + g_bfps_on):
     * otherwise an int16 instance sharing the process-global g_bfps_on (e.g.
     * the int16 salvage decoder running inside a split-mode process) would
     * load the channel into the APP at init AND add it again in tau/syndrome
     * /hard-decision -> 2x channel weight, crippling the fallback. */
    const int16_t* chan_split = bfps ? my_soft_g : nullptr;
    if (soft_in) {
        const int16_t* my_soft = my_soft_g;
        for (int idx = tid; idx < K; idx += n_threads) {
            int v = my_soft[idx];
            s_data[idx] = bfps ? (APP)0
                : bfp8 ? bfp8_st<APP>(v, bfp8_ed(cw, idx))
                : (APP)(v < -in_lim ? -in_lim : (v > in_lim ? in_lim : v));
        }
        for (int idx = tid; idx < N - K; idx += n_threads) {
            int i_layer = idx % q;
            int layer_j = idx / q;
            int v = my_soft[K + idx];
            s_pty[M * i_layer + layer_j] = bfps ? (APP)0
                : bfp8 ? bfp8_st<APP>(v, bfp8_ed(cw, K + idx))
                : (APP)(v < -in_lim ? -in_lim : (v > in_lim ? in_lim : v));
        }
    } else {
        for (int idx = tid; idx < K; idx += n_threads) {
            int v = g_data[idx];
            s_data[idx] = bfps ? (APP)0
                : bfp8 ? bfp8_st<APP>(v, bfp8_ed(cw, idx))
                : (APP)(v < -in_lim ? -in_lim : (v > in_lim ? in_lim : v));
        }
        for (int idx = tid; idx < R; idx += n_threads) {
            int v = g_pty[idx];
            s_pty[idx] = bfps ? (APP)0
                : bfp8 ? bfp8_st<APP>(v, bfp8_ep(cw, K, q, idx / M, idx % M))
                : (APP)(v < -in_lim ? -in_lim : (v > in_lim ? in_lim : v));
        }
    }
    /* First iteration treats stale BNL as implicit zero and writes every used
     * message before later iterations read BNL. */
    __syncthreads();

    for (int trial = 0; trial <= max_trials; trial++) {
        if (g_bfps_on && threadIdx.x == 0) g_bfps_cur_trial = trial;
        __syncthreads();

        /* -- Syndrome check (periodic + final) -- */
        bool do_check = (trial % syndrome_period == 0) || (trial == max_trials);
        if (do_check) {
            if (j == 0)
                *s_converged_p = 1;
            __syncthreads();

            /* Parallel syndrome: same 2-layer schedule as decode.
             * Each thread checks its lane's layer. */
            for (int step = 0;
                 step < sched_steps && (*s_converged_p || g_dbg_syn); step++) {
                int i = sched[step * ldpc_schedule::LANES_PER_STEP + lane];
                if (i >= 0) {
                    int cnt = cnc[i];
                    int16_t syn = (int16_t)qsat16(bfps_chp(chan_split, K, q, i, j)
                                                  + (int)s_pty[M * i + j]);
                    if (i > 0)
                        syn ^= (int16_t)qsat16(bfps_chp(chan_split, K, q, i - 1, j)
                                               + (int)s_pty[M * (i - 1) + j]);
                    else if (j > 0)
                        syn ^= (int16_t)qsat16(bfps_chp(chan_split, K, q, q - 1, j - 1)
                                               + (int)s_pty[j + (q - 1) * M - 1]);
                    for (int c = 0; c < cnt; c++) {
                        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
                        syn ^= (int16_t)qsat16(bfps_chd(chan_split, v) + (int)s_data[v]);
                    }
                    if (syn < 0) {
                        *s_converged_p = 0;
                        if (g_dbg_syn && cw == 0 && trial == max_trials)
                            printf("SYNFAIL i=%d j=%d off=%d\n", i, j,
                                   M * i + j);
                    }
                }
                __syncthreads();
            }
            __syncthreads();

            if (*s_converged_p) {
                pack_hard_from_app(s_data, hard_out, K, cw, chan_split);
                /* Write back shared → global before returning */
                for (int idx = tid; idx < K; idx += n_threads)
                    g_data[idx] = (int16_t)qsat16(bfps_chd(chan_split, idx)
                        + (((int)s_data[idx]) << bfp8_ed(cw, idx)));
                for (int idx = tid; idx < R; idx += n_threads)
                    g_pty[idx] = (int16_t)qsat16(bfps_chp(chan_split, K, q, idx / M, idx % M)
                        + (((int)s_pty[idx]) << bfp8_ep(cw, K, q, idx / M, idx % M)));
                __syncthreads();
                if (j == 0 && lane == 0)
                    result[cw] = max_trials - trial;
                return;
            }
            __syncthreads();
        }

        /* No more update iterations left */
        if (trial == max_trials)
            break;

        /* -- Process layers in parallel groups (2-3 independent layers per step) -- */
        for (int step = 0; step < sched_steps; step++) {
            int i = sched[step * ldpc_schedule::LANES_PER_STEP + lane]; /* my layer (-1 if inactive) */

            if (i >= 0) {
                if (trial == 0) {
                    process_layer_initial_zero_bnl<APP, HOT0, HOT1, HOT2>(
                        s_data, s_pty, my_bnl, pos, cnc, bnl_layer_off,
                        M, q, CNL, i, j, cw, K, chan_split,
                        cst_on ? cst_addr(cw, q, i, M, j) : nullptr);
                } else {
                    int cnt = cnc[i];
                    int deg = cnt + 2 - (i == 0 && j == 0 ? 1 : 0);

                    int8_t* bl = my_bnl + bnl_layer_off[i] + j;
                    int16_t inp[LDPC_DEC_MAX_DEG];

                    if (!cst_on) {
                    /* ------- default path: per-edge int8 BNL (original) ------- */
                    for (int c = 0; c < cnt; c++) {
                        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
                        inp[c] = qsat16(bfps_chd(chan_split, v) + (((int)s_data[v]) << bfp8_ed(cw, v)) - (int)bl[c * M]);
                    }

                    inp[cnt] = qsat16(bfps_chp(chan_split, K, q, i, j) + (((int)s_pty[M * i + j]) << bfp8_ep(cw, K, q, i, j)) - (int)bl[cnt * M]);

                    if (i > 0) {
                        inp[cnt + 1] = qsat16(bfps_chp(chan_split, K, q, i - 1, j) + (((int)s_pty[M * (i - 1) + j]) << bfp8_ep(cw, K, q, i - 1, j)) - (int)bl[(cnt + 1) * M]);
                    } else if (j > 0) {
                        inp[cnt + 1] = qsat16(bfps_chp(chan_split, K, q, q - 1, j - 1) + (((int)s_pty[j + (q - 1) * M - 1]) << bfp8_ep(cw, K, q, q - 1, j - 1)) - (int)bl[(cnt + 1) * M]);
                    }

                    int16_t inp_pty0 = inp[cnt];
                    int16_t inp_pty1 = (deg > cnt + 1) ? inp[cnt + 1] : 0;

                    /* Check node update (offset min-sum) — in-place on inp[] */
                    finalp_oms_i16_dispatch<HOT0, HOT1, HOT2>(inp, deg);

                    /* Store exactly the int8 BNL value used for the APP update.
                     * The next iteration subtracts this stored value, so the
                     * APP delta must be based on the clamped message, not the
                     * raw int16 min-sum output. */
                    for (int c = 0; c < cnt; c++) {
                        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
                        const ldpc_bnl_update_t upd =
                            bnl_quant_update(ldpc_make_bnl_update(nms_scale((int)inp[c]), (int)bl[c * M]));
                        const int ed = bfp8_ed(cw, v);
                        const int8_t mq = bfp8_msg_grid(upd.message, ed);
                        inp[c] = mq;
                        s_data[v] = bfp8_st<APP>((((int)s_data[v]) << ed)
                                        + (int)mq - (int)bl[c * M], ed);
                    }

                    const int ep0 = bfp8_ep(cw, K, q, i, j);
                    const ldpc_bnl_update_t upd_pty0 =
                        bnl_quant_update(ldpc_make_bnl_update(nms_scale((int)inp[cnt]), (int)bl[cnt * M]));
                    const int8_t mq0 = bfp8_msg_grid(upd_pty0.message, ep0);
                    inp[cnt] = mq0;
                    s_pty[M * i + j] = bfp8_st<APP>(
                        (bfps ? (int)s_pty[M * i + j] - (int)bl[cnt * M]
                                   : (int)inp_pty0) + (int)mq0, ep0);

                    if (i > 0) {
                        const int ep1 = bfp8_ep(cw, K, q, i - 1, j);
                        const ldpc_bnl_update_t upd_pty1 =
                            bnl_quant_update(ldpc_make_bnl_update(nms_scale((int)inp[cnt + 1]),
                                                 (int)bl[(cnt + 1) * M]));
                        const int8_t mq1 = bfp8_msg_grid(upd_pty1.message, ep1);
                        inp[cnt + 1] = mq1;
                        s_pty[M * (i - 1) + j] = bfp8_st<APP>(
                            (bfps ? (int)s_pty[M * (i - 1) + j] - (int)bl[(cnt + 1) * M]
                                       : (int)inp_pty1) + (int)mq1, ep1);
                    } else if (j > 0) {
                        const int ep1 = bfp8_ep(cw, K, q, q - 1, j - 1);
                        const ldpc_bnl_update_t upd_pty1 =
                            bnl_quant_update(ldpc_make_bnl_update(nms_scale((int)inp[cnt + 1]),
                                                 (int)bl[(cnt + 1) * M]));
                        const int8_t mq1 = bfp8_msg_grid(upd_pty1.message, ep1);
                        inp[cnt + 1] = mq1;
                        s_pty[j + (q - 1) * M - 1] = bfp8_st<APP>(
                            (bfps ? (int)s_pty[j + (q - 1) * M - 1] - (int)bl[(cnt + 1) * M]
                                       : (int)inp_pty1) + (int)mq1, ep1);
                    }

                    for (int c = 0; c < deg; c++) {
                        bl[c * M] = (int8_t)inp[c];
                    }
                    } else {
                    /* ------- compressed check-state path (CUSAT_LDPC_CST) ----- */
                    void* cst_row = cst_addr(cw, q, i, M, j);
                    const cst_t told = cst_load_row(cst_row);
                    if (g_dbg_cst && cw == 0) {
                        /* bl[] holds the dense shadow written on the last
                         * visit of this layer; the tuple must reconstruct
                         * it exactly. */
                        for (int c = 0; c < deg; c++) {
                            const int16_t exp_v = (int16_t)bl[c * M];
                            const int16_t got_v = cst_load_edge(told, c);
                            if (exp_v != got_v &&
                                atomicSub(&g_dbg_cst_budget, 1) > 0)
                                printf("CSTMM t=%d i=%d j=%d c=%d deg=%d "
                                       "exp=%d got=%d lo=%d hi=%d signs=%x "
                                       "his=%x\n",
                                       trial, i, j, c, deg, (int)exp_v,
                                       (int)got_v, (int)told.lo,
                                       (int)told.hi, told.signs, told.his);
                        }
                    }

                    for (int c = 0; c < cnt; c++) {
                        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
                        inp[c] = qsat16(bfps_chd(chan_split, v) + (((int)s_data[v]) << bfp8_ed(cw, v)) - (int)cst_load_edge(told, c));
                    }

                    inp[cnt] = qsat16(bfps_chp(chan_split, K, q, i, j) + (((int)s_pty[M * i + j]) << bfp8_ep(cw, K, q, i, j)) - (int)cst_load_edge(told, cnt));

                    if (i > 0) {
                        inp[cnt + 1] = qsat16(bfps_chp(chan_split, K, q, i - 1, j) + (((int)s_pty[M * (i - 1) + j]) << bfp8_ep(cw, K, q, i - 1, j)) - (int)cst_load_edge(told, cnt + 1));
                    } else if (j > 0) {
                        inp[cnt + 1] = qsat16(bfps_chp(chan_split, K, q, q - 1, j - 1) + (((int)s_pty[j + (q - 1) * M - 1]) << bfp8_ep(cw, K, q, q - 1, j - 1)) - (int)cst_load_edge(told, cnt + 1));
                    }

                    int16_t inp_pty0 = inp[cnt];
                    int16_t inp_pty1 = (deg > cnt + 1) ? inp[cnt + 1] : 0;

                    finalp_oms_i16_dispatch<HOT0, HOT1, HOT2>(inp, deg);

                    for (int c = 0; c < cnt; c++) {
                        uint16_t v = __ldg(&pos[(i * CNL + c) * M + j]);
                        const ldpc_bnl_update_t upd =
                            ldpc_make_bnl_update(nms_scale((int)inp[c]), (int)cst_load_edge(told, c));
                        const int ed = bfp8_ed(cw, v);
                        const int8_t mq = bfp8_msg_grid(upd.message, ed);
                        inp[c] = mq;
                        if (g_dbg_cst) bl[c * M] = mq;
                        s_data[v] = bfp8_st<APP>((((int)s_data[v]) << ed)
                                        + (int)mq - (int)cst_load_edge(told, c), ed);
                    }

                    const int ep0 = bfp8_ep(cw, K, q, i, j);
                    const ldpc_bnl_update_t upd_pty0 =
                        ldpc_make_bnl_update(nms_scale((int)inp[cnt]), (int)cst_load_edge(told, cnt));
                    const int8_t mq0 = bfp8_msg_grid(upd_pty0.message, ep0);
                    inp[cnt] = mq0;
                    if (g_dbg_cst) bl[cnt * M] = mq0;
                    s_pty[M * i + j] = bfp8_st<APP>(
                        (bfps ? (int)s_pty[M * i + j] - (int)cst_load_edge(told, cnt)
                                   : (int)inp_pty0) + (int)mq0, ep0);

                    if (i > 0) {
                        const int ep1 = bfp8_ep(cw, K, q, i - 1, j);
                        const ldpc_bnl_update_t upd_pty1 =
                            ldpc_make_bnl_update(nms_scale((int)inp[cnt + 1]),
                                                 (int)cst_load_edge(told, cnt + 1));
                        const int8_t mq1 = bfp8_msg_grid(upd_pty1.message, ep1);
                        inp[cnt + 1] = mq1;
                        if (g_dbg_cst) bl[(cnt + 1) * M] = mq1;
                        s_pty[M * (i - 1) + j] = bfp8_st<APP>(
                            (bfps ? (int)s_pty[M * (i - 1) + j] - (int)cst_load_edge(told, cnt + 1)
                                       : (int)inp_pty1) + (int)mq1, ep1);
                    } else if (j > 0) {
                        const int ep1 = bfp8_ep(cw, K, q, q - 1, j - 1);
                        const ldpc_bnl_update_t upd_pty1 =
                            ldpc_make_bnl_update(nms_scale((int)inp[cnt + 1]),
                                                 (int)cst_load_edge(told, cnt + 1));
                        const int8_t mq1 = bfp8_msg_grid(upd_pty1.message, ep1);
                        inp[cnt + 1] = mq1;
                        if (g_dbg_cst) bl[(cnt + 1) * M] = mq1;
                        s_pty[j + (q - 1) * M - 1] = bfp8_st<APP>(
                            (bfps ? (int)s_pty[j + (q - 1) * M - 1] - (int)cst_load_edge(told, cnt + 1)
                                       : (int)inp_pty1) + (int)mq1, ep1);
                    }

                    cst_store(cst_row, inp, deg);
                    }
                }
            }

            __syncthreads();
        }
    }

    /* Write back final state to global */
    pack_hard_from_app(s_data, hard_out, K, cw, chan_split);
    for (int idx = tid; idx < K; idx += n_threads)
        g_data[idx] = (int16_t)qsat16(bfps_chd(chan_split, idx)
            + (((int)s_data[idx]) << bfp8_ed(cw, idx)));
    for (int idx = tid; idx < R; idx += n_threads)
        g_pty[idx] = (int16_t)qsat16(bfps_chp(chan_split, K, q, idx / M, idx % M)
            + (((int)s_pty[idx]) << bfp8_ep(cw, K, q, idx / M, idx % M)));
    __syncthreads();

    if (j == 0 && lane == 0)
        result[cw] = -1;
}

/* INT16 APP kernel; shared memory limits residency to one block per SM. */
template <bool UseSharedMem>
__global__ void __launch_bounds__(720, 1) ldpc_decode_layered_kernel(
    int16_t* __restrict__ data, int16_t* __restrict__ pty,
    int8_t* __restrict__ bnl, const uint16_t* __restrict__ pos,
    const uint8_t* __restrict__ cnc, const int* __restrict__ bnl_layer_off,
    int* __restrict__ result, uint8_t* __restrict__ hard_out,
    int K, int R, int M, int q, int CNL, int LT, int max_trials,
    const int16_t* __restrict__ soft_in, int N, int syndrome_period,
    const int16_t* __restrict__ sched, int sched_steps)
{
    extern __shared__ char s_mem_raw[];
    ldpc_layered_body<UseSharedMem, int16_t>(
        data, pty, bnl, pos, cnc, bnl_layer_off, result, hard_out,
        K, R, M, q, CNL, LT, max_trials, soft_in, N, syndrome_period,
        32767, sched, sched_steps, s_mem_raw);
}

/* ---------- Hard decision kernel: int16 signs → packed bytes ---------- */

__global__ void ldpc_hard_decision_kernel(
    const int16_t* __restrict__ data,  // [B * K]
    uint8_t* __restrict__ hard_out,    // [B * K/8]
    int K)
{
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cw = blockIdx.y;
    int K_bytes = K / 8;

    if (byte_idx >= K_bytes)
        return;

    const int16_t* src = data + cw * K + byte_idx * 8;
    uint8_t byte = 0;
    for (int b = 0; b < 8; b++) {
        if (src[b] < 0)
            byte |= (1 << (7 - b));
    }
    hard_out[cw * K_bytes + byte_idx] = byte;
}

/* ---------- Host API ---------- */

ldpc_decode_cuda::ldpc_decode_cuda(const ldpc_code* code, code_rate_t rate,
                                   int max_trials, int max_batch)
    : d_max_trials(max_trials), d_max_batch(max_batch)
{
    /* Extract parameters from the flat table description */
    d_N = code->n;
    d_K = code->k;
    d_M = code->m;
    d_R = d_N - d_K;
    d_q = d_R / d_M;
    d_CNL = code->links_max_cn - 2;
    d_LT = code->links_total;

    /* Build pos[] (bit indices per check node) and cnc[] (check node counts).
     * Info bit j = row*M + jj checks accumulator (addr[c] + jj*q) % R for
     * each base address of its row. */
    std::vector<uint16_t> pos_raw(d_R * d_CNL, 0);
    std::vector<uint8_t> cnc_raw(d_R, 0);

    for (int row = 0; row < code->n_rows; row++) {
        const uint16_t* base = code->addr + code->row_off[row];
        int deg = code->row_deg[row];
        for (int jj = 0; jj < d_M; jj++) {
            int j = row * d_M + jj;
            for (int n = 0; n < deg; n++) {
                int i = ((int)base[n] + jj * d_q) % d_R;
                pos_raw[d_CNL * i + cnc_raw[i]++] = (uint16_t)j;
            }
        }
    }

    /* Transpose pos into edge-major order within each layer:
     * tmp[(i*CNL+c)*M+j] = pos[CNL*(q*j+i)+c].
     *
     * For a fixed layer i and edge c, the 360 check-node threads read
     * consecutive j values. Keeping j contiguous makes each warp's pos[]
     * lookup coalesced. */
    std::vector<uint16_t> pos_trans(d_R * d_CNL);
    for (int i = 0; i < d_q; i++)
        for (int j = 0; j < d_M; j++)
            for (int c = 0; c < d_CNL; c++)
                pos_trans[(i * d_CNL + c) * d_M + j] = pos_raw[d_CNL * (d_q * j + i) + c];

    /* Extract per-layer check node counts.
     * After transposition, cnc[M*i + j] gives the count for check node (i,j).
     * Within a layer, all M check nodes have the same count (quasi-cyclic).
     * We store one count per layer. */
    std::vector<uint8_t> cnc_trans(d_R);
    for (int i = 0; i < d_q; i++)
        for (int j = 0; j < d_M; j++)
            cnc_trans[d_M * i + j] = cnc_raw[d_q * j + i];

    std::vector<uint8_t> cnc_per_layer(d_q);
    for (int i = 0; i < d_q; i++)
        cnc_per_layer[i] = cnc_trans[d_M * i]; /* same for all j in layer */

    /* Compute BNL layer offsets in edge-major order:
     *   layer base + edge*M + check-node j.
     *
     * Layer 0 has one unused padded byte for edge cnt+1,j=0. The padding
     * makes all warp loads/stores for a fixed edge contiguous and removes the
     * special one-byte skew from the old thread-major layout. */
    std::vector<int> bnl_layer_off(d_q + 1);
    bnl_layer_off[0] = 0;
    for (int i = 0; i < d_q; i++) {
        int cnt = cnc_per_layer[i];
        int layer_size = d_M * (cnt + 2);
        bnl_layer_off[i + 1] = bnl_layer_off[i] + layer_size;
    }
    d_LT_alloc = bnl_layer_off[d_q];

    /* Verify LT matches */
    if (d_LT_alloc != d_LT + 1) {
        fprintf(stderr, "ldpc_decode_cuda: LT padded mismatch: computed %d vs expected %d\n",
                d_LT_alloc, d_LT + 1);
    }

    /* Upload tables to device */
    CUDA_CHECK(cudaMalloc(&d_pos_dev, (size_t)d_R * d_CNL * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_pos_dev, pos_trans.data(),
                          (size_t)d_R * d_CNL * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_cnc_dev, d_q * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_cnc_dev, cnc_per_layer.data(),
                          d_q * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_bnl_layer_off_dev, (d_q + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_bnl_layer_off_dev, bnl_layer_off.data(),
                          (d_q + 1) * sizeof(int), cudaMemcpyHostToDevice));

    /* Allocate working buffers for max_batch codewords */
    CUDA_CHECK(cudaMalloc(&d_data_dev, (size_t)max_batch * d_K * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_pty_dev,  (size_t)max_batch * d_R * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_bnl_dev,  (size_t)max_batch * d_LT_alloc * sizeof(int8_t)));

    /* d_soft_dev: LDPC iterative working buffer — read/written d_N × max_trials
     * times per decode.  MUST stay in device HBM (~2 TB/s) not pinned host
     * (~450 GB/s NVLink) or decode throughput drops ~4x.
     * d_hard_dev, d_result_dev: output buffers written once — pinned is fine. */
    CUDA_CHECK(cudaMalloc(&d_soft_dev, (size_t)max_batch * d_N * sizeof(int16_t)));
    CUDA_CHECK(cudaMallocHost(&d_hard_dev, (size_t)max_batch * (d_K / 8) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMallocHost(&d_result_dev, (size_t)max_batch * sizeof(int)));

    /* Opt-in to large shared memory (>48KB) for the decode kernel.
     * Falls back to L2-pinned global memory if the device limit is insufficient. */
    size_t smem_bytes = ((size_t)d_K + d_R) * sizeof(int16_t) + sizeof(int);
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    d_use_smem = (smem_bytes <= (size_t)max_smem_optin);
    /* Experiment knob: the smem-resident path pins occupancy to 1 block/SM
     * (126 KB working set). The L2-pinned global path frees shared memory,
     * letting 2x 720-thread blocks co-reside per SM; on parts with a large
     * L2 (GH200: 50 MB) the batch working set stays cache-resident. */
    if (getenv("CUSAT_LDPC_FORCE_L2")) {
        d_use_smem = false;
        fprintf(stderr, "ldpc_decode_cuda: CUSAT_LDPC_FORCE_L2 — using "
                "L2-pinned global memory path\n");
    }
    if (d_use_smem) {
        CUDA_CHECK(cudaFuncSetAttribute(ldpc_decode_layered_kernel<true>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       smem_bytes));
    } else {
        fprintf(stderr, "ldpc_decode_cuda: smem %zu KB > device optin limit %d KB — using L2-pinned global memory\n",
                smem_bytes / 1024, max_smem_optin / 1024);
    }
    /* Upload parallel-layer schedule from registry (or sequential fallback).
     * CUSAT_LDPC_SEQ=1 forces the sequential fallback — A/B knob for
     * measuring the real (not theoretical) value of a schedule. */
    {
        ldpc_schedule::entry sched = ldpc_schedule::get(rate);
        const char* force_seq = getenv("CUSAT_LDPC_SEQ");
        if (force_seq && atoi(force_seq) != 0)
            sched.pairs = nullptr;
        /* The precomputed MILP schedules are built for the NORMAL-frame code of
         * each rate (q = 90, etc.). Short/medium frames have a different layer
         * count (d_q); the normal schedule would index past the code's per-layer
         * buffers (illegal memory access). Only use the precomputed schedule
         * when its layer count matches this code; else fall back to sequential
         * (correct for any q, just without the parallel speedup). */
        if (sched.pairs && sched.q != d_q) {
            fprintf(stderr, "ldpc_decode_cuda: schedule q=%d != code q=%d "
                    "(non-normal frame) — sequential schedule\n", sched.q, d_q);
            sched.pairs = nullptr;
        }
        /* Per-instance schedule buffer. Allocate the full max so any later
         * code's schedule fits, then upload this code's. */
        CUDA_CHECK(cudaMalloc(&d_schedule_dev,
                              (size_t)ldpc_schedule::MAX_SCHEDULE_STEPS *
                                  ldpc_schedule::LANES_PER_STEP * sizeof(int16_t)));
        /* Zero the full buffer: the kernel's step loop can touch one entry
         * past the uploaded schedule; uninitialized tail bytes there become a
         * garbage layer id (e.g. 0xFFFF) whose pos[] address faults on parts
         * with tight VA layouts (GB10) even though the value is discarded. */
        CUDA_CHECK(cudaMemset(d_schedule_dev, 0,
                              (size_t)ldpc_schedule::MAX_SCHEDULE_STEPS *
                                  ldpc_schedule::LANES_PER_STEP * sizeof(int16_t)));
        if (sched.pairs) {
            d_schedule_n_steps = sched.n_steps;
            CUDA_CHECK(cudaMemcpy(d_schedule_dev, sched.pairs,
                                  (size_t)sched.n_steps * ldpc_schedule::LANES_PER_STEP *
                                      sizeof(int16_t),
                                  cudaMemcpyHostToDevice));
            fprintf(stderr, "ldpc_decode_cuda: parallel schedule %d steps "
                    "(from %d layers, %.2fx)\n",
                    sched.n_steps, sched.q,
                    (float)sched.q / sched.n_steps);
        } else {
            /* No precomputed schedule — sequential fallback */
            int16_t seq_buf[ldpc_schedule::MAX_SCHEDULE_STEPS][2];
            int n = ldpc_schedule::build_sequential(seq_buf, d_q);
            d_schedule_n_steps = n;
            CUDA_CHECK(cudaMemcpy(d_schedule_dev, seq_buf,
                                  (size_t)n * ldpc_schedule::LANES_PER_STEP *
                                      sizeof(int16_t),
                                  cudaMemcpyHostToDevice));
            fprintf(stderr, "ldpc_decode_cuda: sequential fallback %d steps\n", n);
        }
    }

    fprintf(stderr, "ldpc_decode_cuda: shared memory + parallel layers (%zu KB)\n",
            smem_bytes / 1024);

    size_t total_kb = ((size_t)max_batch * ((d_K + d_R) * 2 + d_LT_alloc + d_N * 2 + d_K / 8) +
                       (size_t)d_R * d_CNL * 2 + d_q + (d_q + 1) * 4 +
                       max_batch * 4) / 1024;
    fprintf(stderr, "ldpc_decode_cuda: N=%d K=%d R=%d M=%d q=%d CNL=%d LT=%d "
            "max_batch=%d (%zu KB device)\n",
            d_N, d_K, d_R, d_M, d_q, d_CNL, d_LT, max_batch, total_kb);
}

ldpc_decode_cuda::~ldpc_decode_cuda()
{
    cudaFree(d_pos_dev);
    cudaFree(d_cnc_dev);
    cudaFree(d_bnl_layer_off_dev);
    cudaFree(d_schedule_dev);
    if (d_cst_dev) cudaFree(d_cst_dev);
    cudaFree(d_data_dev);
    cudaFree(d_pty_dev);
    cudaFree(d_bnl_dev);
    cudaFreeHost(d_result_dev);
    cudaFree(d_soft_dev);
    cudaFreeHost(d_hard_dev);
}

void ldpc_decode_cuda::launch_kernels(int batch_size, cudaStream_t stream,
                                      const int16_t* soft_in,
                                      uint8_t* hard_out,
                                      int* result_out)
{
    // Use external buffers if provided, otherwise internal managed buffers
    const int16_t* soft_ptr = soft_in ? soft_in : d_soft_dev;
    uint8_t* hard_ptr = hard_out ? hard_out : d_hard_dev;
    int* result_ptr = result_out ? result_out : d_result_dev;

    /* Pin data+pty in L2 cache for the duration of the decode.
     * Reduces random-access latency from ~200 cycles (DRAM) to ~40 cycles (L2).
     * Working set: batch * (K+R) * 2 bytes. Fits in GH200 L2 (~15MB partition). */
    {
        cudaStreamAttrValue attr = {};
        size_t working_set = (size_t)batch_size * (d_K + d_R) * sizeof(int16_t);
        attr.accessPolicyWindow.base_ptr = d_data_dev;
        attr.accessPolicyWindow.num_bytes = working_set;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    /* Main decode: one block per codeword.
     * 2*M threads = 720: up to 2 independent layers execute per step.
     * Data/pty in shared memory (126KB). */
    int threads_per_block = d_M * ldpc_schedule::LANES_PER_STEP; /* 720 */
    size_t smem_launch = d_use_smem
        ? ((size_t)d_K + d_R) * sizeof(int16_t) + sizeof(int)
        : sizeof(int);
    /* Syndrome-check cadence. Checks are nearly free: the shared
     * convergence flag short-circuits the pass after the first failing
     * layer, so non-converged frames pay ~one layer of sign-XORs.
     * Period 1 (check every trial) exits at TRUE convergence instead of
     * the next multiple: measured +49% realistic throughput at operating
     * SNR (it_mean 9.8 -> 6.2 on RATE_3_5 @2.5dB), identical worst-case
     * floor, FER at threshold equal-or-better (same-seed A/B).
     * CUSAT_LDPC_SYN_PERIOD overrides for A/B without rebuilds. */
    static int syn_period = 0;
    if (syn_period == 0) {
        const char* e = getenv("CUSAT_LDPC_SYN_PERIOD");
        syn_period = (e && atoi(e) > 0) ? atoi(e) : 1;
    }
    static int fused_hard = -1;
    if (fused_hard < 0) {
        const char* e = getenv("CUSAT_LDPC_FUSED_HARD");
        fused_hard = (!e || atoi(e) != 0) ? 1 : 0;
    }
    uint8_t* fused_hard_ptr = fused_hard ? hard_ptr : nullptr;

    if (d_use_smem) {
        ldpc_decode_layered_kernel<true><<<batch_size, threads_per_block, smem_launch, stream>>>(
            d_data_dev, d_pty_dev, d_bnl_dev,
            d_pos_dev, d_cnc_dev, d_bnl_layer_off_dev,
            result_ptr, fused_hard_ptr,
            d_K, d_R, d_M, d_q, d_CNL, d_LT_alloc,
            d_max_trials, soft_ptr, d_N, syn_period,
            d_schedule_dev, d_schedule_n_steps);
    } else {
        ldpc_decode_layered_kernel<false><<<batch_size, threads_per_block, smem_launch, stream>>>(
            d_data_dev, d_pty_dev, d_bnl_dev,
            d_pos_dev, d_cnc_dev, d_bnl_layer_off_dev,
            result_ptr, fused_hard_ptr,
            d_K, d_R, d_M, d_q, d_CNL, d_LT_alloc,
            d_max_trials, soft_ptr, d_N, syn_period,
            d_schedule_dev, d_schedule_n_steps);
    }
    CUDA_CHECK_LAUNCH();

    /* Reset L2 persistence — free cache lines for other kernels */
    {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    /* Hard decision: pack signs into bytes, unless the layered kernel already
     * wrote hard_out from its shared-memory APP state. */
    if (!fused_hard) {
        int K_bytes = d_K / 8;
        int threads = 256;
        dim3 grid(((unsigned)K_bytes + threads - 1) / threads, batch_size);
        ldpc_hard_decision_kernel<<<grid, threads, 0, stream>>>(
            d_data_dev, hard_ptr, d_K);
        CUDA_CHECK_LAUNCH();
    }
}

void ldpc_decode_cuda::decode_batch(const int16_t* soft_in, uint8_t* hard_out,
                                     int* trials_out, int batch_size,
                                     cudaStream_t stream)
{
    if (batch_size <= 0 || batch_size > d_max_batch)
        return;

    /* d_soft_dev is device HBM (cudaMalloc) — host input needs a real H2D
     * copy; a CPU memcpy into it faults (no host mapping). */
    CUDA_CHECK(cudaMemcpyAsync(d_soft_dev, soft_in,
                               (size_t)batch_size * d_N * sizeof(int16_t),
                               cudaMemcpyHostToDevice, stream));
    launch_kernels(batch_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Read results from managed memory */
    memcpy(hard_out, d_hard_dev, (size_t)batch_size * (d_K / 8));
    memcpy(trials_out, d_result_dev, (size_t)batch_size * sizeof(int));
}

void ldpc_decode_cuda::decode_batch_d2h(const int16_t* soft_in_dev,
                                         uint8_t* hard_out_host,
                                         int* trials_out_host,
                                         int batch_size, cudaStream_t stream)
{
    if (batch_size <= 0 || batch_size > d_max_batch)
        return;

    /* Read soft directly, write to internal managed buffers */
    launch_kernels(batch_size, stream, soft_in_dev);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Copy from managed to caller's host buffers */
    memcpy(hard_out_host, d_hard_dev, (size_t)batch_size * (d_K / 8));
    memcpy(trials_out_host, d_result_dev, (size_t)batch_size * sizeof(int));
}

void ldpc_decode_cuda::decode_launch_async(const int16_t* soft_in,
                                           int batch_size,
                                           cudaStream_t stream,
                                           uint8_t* hard_out,
                                           int* result_out)
{
    if (batch_size <= 0 || batch_size > d_max_batch)
        return;

    /* Read soft LLRs directly — no copy needed on GH200.
     * If external output buffers provided, kernels write there directly. */
    launch_kernels(batch_size, stream, soft_in, hard_out, result_out);
}

} // namespace cuda
} // namespace cusat_dsp
