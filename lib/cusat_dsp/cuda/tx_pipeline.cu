/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused TX pipeline: mapper → PL frame+scramble → RRC.
 *
 * All GPU stages run on a single CUDA stream with no intermediate host
 * round-trips. Data flows: FECFRAME packed (H2D) → mapper → PL frame assembly
 * + PL scrambling (fused kernel) → RRC pulse shaping → IQ samples (D2H).
 *
 * The PL framing + scrambling kernel simultaneously:
 *   - Scatters XFECFRAME symbols into slot positions
 *   - Inserts pilot reference symbols
 *   - Applies PL scrambling (multiply by conj of descrambling sequence)
 * This saves one full PLFRAME-sized transfer that would occur between separate
 * PL framing and scrambling stages.
 */

#include "tx_pipeline.cuh"
#include "cuda_util.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace cusat_dsp {
namespace cuda {

/* ========================== KERNELS ========================== */

/*
 * Mapper kernel (same logic as mapper_kernel.cu, inlined to avoid
 * cross-file constant-memory issues).
 *
 * One thread per output symbol. Reads n_mod bits from packed FECFRAME
 * using column offsets, forms LUT index, writes constellation point.
 */
__global__ void pipeline_mapper_kernel(
    const uint8_t* __restrict__ fecframe_packed,
    float* __restrict__ symbols_out,
    const float* __restrict__ constellation_lut,
    int xfecframe_len, int n_mod,
    unsigned int col0, unsigned int col1, unsigned int col2,
    unsigned int col3, unsigned int col4, unsigned int col5,
    unsigned int col6, unsigned int col7, int stride)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= xfecframe_len) return;

    unsigned int cols[8] = { col0, col1, col2, col3, col4, col5, col6, col7 };
    int lut_idx = 0;
    for (int b = 0; b < n_mod; b++) {
        int bit_pos = (int)cols[b] + j * stride;
        int byte_idx = bit_pos >> 3;
        int bit_in_byte = 7 - (bit_pos & 7);
        int bit = (fecframe_packed[byte_idx] >> bit_in_byte) & 1;
        lut_idx |= (bit << (n_mod - 1 - b));
    }
    symbols_out[j * 2]     = constellation_lut[lut_idx * 2];
    symbols_out[j * 2 + 1] = constellation_lut[lut_idx * 2 + 1];
}

/*
 * Fused PL framing + PL scrambling kernel.
 *
 * Simultaneously:
 *   1. Scatters XFECFRAME symbols into PLFRAME slot positions (with pilot insertion)
 *   2. Multiplies by conj(descrambling_seq) for PL scrambling
 *
 * Output is written directly into the symbol_buf at offset (history_len + pl::kHeaderLen),
 * so RRC can read the full [history|PLHEADER|scrambled_payload] contiguously.
 *
 * Without pilots: output[i] = xfecframe[i] * conj(seq[i])
 * With pilots: slot/pilot interleave pattern, then scramble.
 *
 * Pilot pattern: every 16 slots (1440 symbols), one pilot block (36 symbols).
 *   group k (0..n_pilots-1): data[1440] + pilot[36] = 1476 symbols
 *   remaining: data only
 */
__global__ void pl_frame_scramble_kernel(
    const float* __restrict__ xfecframe,
    const float* __restrict__ scramble_seq,
    float* __restrict__ payload_out,
    int payload_len, int n_pilots, bool has_pilots)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= payload_len) return;

    float re, im;

    if (!has_pilots || n_pilots == 0) {
        /* No pilots: identity mapping from xfecframe */
        re = xfecframe[i * 2];
        im = xfecframe[i * 2 + 1];
    } else {
        const int data_per_group = 16 * 90;    /* 1440 */
        const int group_size = data_per_group + 36; /* 1476 */
        int full_groups_end = n_pilots * group_size;

        if (i < full_groups_end) {
            int group = i / group_size;
            int pos = i % group_size;
            if (pos < data_per_group) {
                int xfec_idx = group * data_per_group + pos;
                re = xfecframe[xfec_idx * 2];
                im = xfecframe[xfec_idx * 2 + 1];
            } else {
                /* Pilot reference: (1/√2, 1/√2) */
                re = 0.7071067811865476f;
                im = 0.7071067811865476f;
            }
        } else {
            /* Remaining slots after all pilot groups */
            int remaining_pos = i - full_groups_end;
            int xfec_idx = n_pilots * data_per_group + remaining_pos;
            re = xfecframe[xfec_idx * 2];
            im = xfecframe[xfec_idx * 2 + 1];
        }
    }

    /* PL scramble: multiply by conj(descrambling_seq[i])
     * conj(b) = (br, -bi): re_out = ar*br + ai*bi, im_out = ai*br - ar*bi */
    float sr = scramble_seq[i * 2];
    float si = scramble_seq[i * 2 + 1];
    payload_out[i * 2]     = re * sr + im * si;
    payload_out[i * 2 + 1] = im * sr - re * si;
}

/*
 * RRC polyphase pulse shaping kernel (same logic as rrc_kernel.cu).
 * Taps passed via device pointer instead of __constant__ memory.
 */
__global__ void pipeline_rrc_kernel(
    const float* __restrict__ symbols,
    float* __restrict__ samples,
    const float* __restrict__ taps,
    int n_symbols, int sub_taps, int sps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_out = n_symbols * sps;

    /* Shared memory: [taps (sub_taps*sps)] [symbol tile] */
    extern __shared__ float s_data[];
    float* s_taps = s_data;
    int total_taps = sub_taps * sps;

    /* Cooperatively load taps into shared memory */
    for (int t = threadIdx.x; t < total_taps; t += blockDim.x)
        s_taps[t] = taps[t];

    /* Load symbol tile: threads in this block need symbols
     * from n_start to n_start + (blockDim.x/sps - 1) + sub_taps - 1 */
    int n_start = (blockIdx.x * blockDim.x) / sps;
    int n_tile = blockDim.x / sps + sub_taps;
    float* s_sym = s_data + total_taps;
    int sym_floats = n_tile * 2;
    int sym_limit = (n_symbols + sub_taps - 1) * 2;
    for (int t = threadIdx.x; t < sym_floats; t += blockDim.x) {
        int gi = n_start * 2 + t;
        s_sym[t] = (gi < sym_limit) ? symbols[gi] : 0.0f;
    }
    __syncthreads();

    if (i >= n_out) return;

    int n = i / sps;
    int p = i % sps;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    for (int k = 0; k < sub_taps; k++) {
        float t = s_taps[k * sps + p];
        int local_idx = (n - n_start + k) * 2;
        float xr = s_sym[local_idx];
        float xi = s_sym[local_idx + 1];
        acc_re += xr * t;
        acc_im += xi * t;
    }

    samples[i * 2]     = acc_re;
    samples[i * 2 + 1] = acc_im;
}

/* ========================== BATCHED KERNELS ========================== */

/*
 * Batched mapper: blockIdx.y = frame index.
 * Reads from fecframes[frame * fecframe_bytes], writes to symbols[frame * xfecframe_len * 2].
 */
__global__ void pipeline_mapper_batch_kernel(
    const uint8_t* __restrict__ fecframes,
    float* __restrict__ symbols_out,
    const float* __restrict__ constellation_lut,
    int xfecframe_len, int n_mod,
    unsigned int col0, unsigned int col1, unsigned int col2,
    unsigned int col3, unsigned int col4, unsigned int col5, int stride,
    int fecframe_bytes)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= xfecframe_len) return;
    int frame = blockIdx.y;

    const uint8_t* fec = fecframes + frame * fecframe_bytes;
    float* out = symbols_out + frame * xfecframe_len * 2;

    unsigned int cols[6] = { col0, col1, col2, col3, col4, col5 };
    int lut_idx = 0;
    for (int b = 0; b < n_mod; b++) {
        int bit_pos = (int)cols[b] + j * stride;
        int byte_idx = bit_pos >> 3;
        int bit_in_byte = 7 - (bit_pos & 7);
        int bit = (fec[byte_idx] >> bit_in_byte) & 1;
        lut_idx |= (bit << (n_mod - 1 - b));
    }
    out[j * 2]     = constellation_lut[lut_idx * 2];
    out[j * 2 + 1] = constellation_lut[lut_idx * 2 + 1];
}

/*
 * Scatter N PLHEADERs from contiguous staging into symbol_buf.
 * blockIdx.x = frame index, threadIdx.x = symbol index (0..89).
 */
__global__ void plheader_scatter_kernel(
    const float* __restrict__ plheaders,
    float* __restrict__ symbol_buf,
    int history_len, int plframe_len)
{
    int sym = threadIdx.x;
    int frame = blockIdx.x;

    const float* src = plheaders + frame * 90 * 2;
    float* dst = symbol_buf + (history_len + frame * plframe_len) * 2;

    dst[sym * 2]     = src[sym * 2];
    dst[sym * 2 + 1] = src[sym * 2 + 1];
}

/*
 * Batched PL frame + scramble: blockIdx.y = frame index.
 * Writes payload into symbol_buf at correct per-frame offset.
 */
__global__ void pl_frame_scramble_batch_kernel(
    const float* __restrict__ xfecframe,
    const float* __restrict__ scramble_seq,
    float* __restrict__ symbol_buf,
    int payload_len, int n_pilots, bool has_pilots,
    int xfecframe_len, int plframe_len, int history_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= payload_len) return;
    int frame = blockIdx.y;

    const float* my_xfec = xfecframe + frame * xfecframe_len * 2;
    /* Payload starts after history + frame's PLHEADER (90 symbols) */
    float* my_out = symbol_buf + (history_len + frame * plframe_len + 90) * 2;

    float re, im;

    if (!has_pilots || n_pilots == 0) {
        re = my_xfec[i * 2];
        im = my_xfec[i * 2 + 1];
    } else {
        const int data_per_group = 16 * 90;
        const int group_size = data_per_group + 36;
        int full_groups_end = n_pilots * group_size;

        if (i < full_groups_end) {
            int group = i / group_size;
            int pos = i % group_size;
            if (pos < data_per_group) {
                int xfec_idx = group * data_per_group + pos;
                re = my_xfec[xfec_idx * 2];
                im = my_xfec[xfec_idx * 2 + 1];
            } else {
                re = 0.7071067811865476f;
                im = 0.7071067811865476f;
            }
        } else {
            int remaining_pos = i - full_groups_end;
            int xfec_idx = n_pilots * data_per_group + remaining_pos;
            re = my_xfec[xfec_idx * 2];
            im = my_xfec[xfec_idx * 2 + 1];
        }
    }

    float sr = scramble_seq[i * 2];
    float si = scramble_seq[i * 2 + 1];
    my_out[i * 2]     = re * sr + im * si;
    my_out[i * 2 + 1] = im * sr - re * si;
}

/*
 * Fused mapper + PL frame + scramble: one kernel does bit→constellation→scramble.
 * Eliminates the intermediate XFECFRAME buffer entirely.
 * blockIdx.y = frame index, thread covers payload positions.
 * For pilot positions: writes scrambled pilot reference.
 * For data positions: maps FECFRAME bits to constellation, then scrambles.
 */
__global__ void fused_map_plscr_batch_kernel(
    const uint8_t* __restrict__ fecframes,
    const float* __restrict__ constellation_lut,
    const float* __restrict__ scramble_seq,
    float* __restrict__ symbol_buf,
    int payload_len, int n_pilots, bool has_pilots,
    int xfecframe_len, int plframe_len, int history_len,
    int n_mod,
    unsigned int col0, unsigned int col1, unsigned int col2,
    unsigned int col3, unsigned int col4, unsigned int col5,
    unsigned int col6, unsigned int col7, int stride,
    int fecframe_bytes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= payload_len) return;
    int frame = blockIdx.y;

    const uint8_t* fec = fecframes + frame * fecframe_bytes;
    float* my_out = symbol_buf + (history_len + frame * plframe_len + 90) * 2;

    float re, im;

    if (!has_pilots || n_pilots == 0) {
        /* No pilots: output position i maps to XFECFRAME index i */
        int j = i;
        unsigned int cols[8] = { col0, col1, col2, col3, col4, col5, col6, col7 };
        int lut_idx = 0;
        for (int b = 0; b < n_mod; b++) {
            int bit_pos = (int)cols[b] + j * stride;
            int byte_idx = bit_pos >> 3;
            int bit_in_byte = 7 - (bit_pos & 7);
            // High-order modcods (128APSK n_mod=7, xfecframe 9270 -> 64890 bits)
            // read up to 90 zero PADDING bits past the 64800-bit codeword. The
            // batch buffer is strided at the unpadded fecframe_bytes (= n/8), so
            // those reads fall into the next frame (corruption) or, for the last
            // frame, off the end of the allocation (CUDA illegal access 700).
            // Padding bits are defined 0, matching the single-frame zero-padded
            // buffer; return 0 here instead of reading past the codeword.
            int bit = (byte_idx < fecframe_bytes)
                          ? ((fec[byte_idx] >> bit_in_byte) & 1) : 0;
            lut_idx |= (bit << (n_mod - 1 - b));
        }
        re = constellation_lut[lut_idx * 2];
        im = constellation_lut[lut_idx * 2 + 1];
    } else {
        const int data_per_group = 16 * 90;
        const int group_size = data_per_group + 36;
        int full_groups_end = n_pilots * group_size;

        int xfec_idx = -1;
        if (i < full_groups_end) {
            int group = i / group_size;
            int pos = i % group_size;
            if (pos < data_per_group) {
                xfec_idx = group * data_per_group + pos;
            } else {
                /* Pilot reference */
                re = 0.7071067811865476f;
                im = 0.7071067811865476f;
            }
        } else {
            int remaining_pos = i - full_groups_end;
            xfec_idx = n_pilots * data_per_group + remaining_pos;
        }

        if (xfec_idx >= 0) {
            unsigned int cols[8] = { col0, col1, col2, col3, col4, col5, col6, col7 };
            int lut_idx = 0;
            for (int b = 0; b < n_mod; b++) {
                int bit_pos = (int)cols[b] + xfec_idx * stride;
                int byte_idx = bit_pos >> 3;
                int bit_in_byte = 7 - (bit_pos & 7);
                // Padding bits past the codeword are 0 (see no-pilots branch).
                int bit = (byte_idx < fecframe_bytes)
                              ? ((fec[byte_idx] >> bit_in_byte) & 1) : 0;
                lut_idx |= (bit << (n_mod - 1 - b));
            }
            re = constellation_lut[lut_idx * 2];
            im = constellation_lut[lut_idx * 2 + 1];
        }
    }

    /* PL scramble */
    float sr = scramble_seq[i * 2];
    float si = scramble_seq[i * 2 + 1];
    my_out[i * 2]     = re * sr + im * si;
    my_out[i * 2 + 1] = im * sr - re * si;
}

/* ========================== HOST API ========================== */

tx_pipeline_cuda::tx_pipeline_cuda(int sps, int sub_taps, int history_len,
                                   const float* interleaved_taps,
                                   const float* descrambling_seq,
                                   int max_payload_len, int max_plframe_len)
    : d_sps(sps), d_sub_taps(sub_taps), d_history_len(history_len)
{
    /* Use highest priority so TX is never starved by RX LDPC decode */
    int lo_prio, hi_prio;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&d_stream,
                                            cudaStreamNonBlocking, hi_prio));

    /* Compute max buffer sizes */
    /* 128APSK needs 9270×7=64890 bits accessed by the mapper (including
     * padding). Round up to cover the worst-case read: 64890/8 ≈ 8112 bytes. */
    d_max_fecframe_bytes = (64890 + 7) / 8;  /* 8112 bytes (was 8100) */
    d_max_xfecframe_len = 64800 / 2;   /* QPSK normal */
    d_max_plframe_len = max_plframe_len;
    d_max_iq_len = max_plframe_len * sps;

    /* fecframe: CPU writes, GPU reads once — pinned OK.
     * symbol_buf: GPU working buffer (mapper/PL/RRC) — device HBM for BW.
     * On GH200, device memory is CPU-accessible via NVLink (for PLHEADER writes).
     * iq: GPU writes, CPU reads — pinned for zero-copy readback. */
    CUDA_CHECK(cudaMallocHost(&d_fecframe_dev, d_max_fecframe_bytes));
    CUDA_CHECK(cudaMalloc(&d_xfecframe_dev, d_max_xfecframe_len * 2 * sizeof(float)));

    unsigned int symbol_buf_len = history_len + max_plframe_len;
    CUDA_CHECK(cudaMalloc(&d_symbol_buf_dev, symbol_buf_len * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_symbol_buf_dev, 0, history_len * 2 * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&d_iq_dev, d_max_iq_len * 2 * sizeof(float)));

    /* Upload PL descrambling sequence */
    CUDA_CHECK(cudaMalloc(&d_scramble_seq_dev, max_payload_len * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scramble_seq_dev, descrambling_seq,
                          max_payload_len * 2 * sizeof(float), cudaMemcpyHostToDevice));

    /* Upload RRC taps to device global memory. CUSAT_TX_BACKOFF (linear, <=1)
     * scales the whole TX waveform down so high-PAPR APSK peaks stay inside the
     * DAC full scale instead of clipping (clipping distortion otherwise lowers
     * the SNDR of the entire signal, pilots included, and breaks high-modcod
     * acquisition). Uniform across modcods; default 1.0 (no backoff). */
    // Default 0.45: with unit-average-energy symbols the RRC output RMS is ~1.0
    // (full scale) and APSK peaks reach ~2x (PAPR 5-6 dB), so they hard-clip the
    // DAC. 0.45 keeps even 128APSK's peak (~PAPR 6.4 dB) inside full scale; the
    // RX AGC restores the level so there is no SNR penalty, only less clipping.
    int n_taps = sub_taps * sps;
    const char* bo_env = getenv("CUSAT_TX_BACKOFF");
    float tx_backoff = bo_env ? atof(bo_env) : 0.45f;
    std::vector<float> scaled_taps;
    const float* taps_src = interleaved_taps;
    if (tx_backoff != 1.0f) {
        scaled_taps.assign(interleaved_taps, interleaved_taps + n_taps);
        for (float& t : scaled_taps) t *= tx_backoff;
        taps_src = scaled_taps.data();
        fprintf(stderr, "tx_pipeline_cuda: TX backoff %.3f (%.2f dB)\n",
                tx_backoff, 20.0f * log10f(tx_backoff));
    }
    CUDA_CHECK(cudaMalloc(&d_rrc_taps_dev, n_taps * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rrc_taps_dev, taps_src,
                          n_taps * sizeof(float), cudaMemcpyHostToDevice));

    fprintf(stderr, "tx_pipeline_cuda: sps=%d sub_taps=%d history=%d "
            "max_plframe=%d max_iq=%u\n",
            sps, sub_taps, history_len, max_plframe_len, d_max_iq_len);
}

tx_pipeline_cuda::~tx_pipeline_cuda()
{
    for (auto& [idx, cfg] : d_configs) {
        if (cfg.d_constellation_dev) cudaFree(cfg.d_constellation_dev);
    }
    cudaFreeHost(d_fecframe_dev);
    cudaFree(d_xfecframe_dev);
    cudaFree(d_symbol_buf_dev);
    cudaFreeHost(d_iq_dev);
    cudaFree(d_batch_xfecframe_dev);
    cudaFree(d_batch_symbol_buf_dev);
    cudaFree(d_batch_iq_dev);
    cudaFree(d_batch_plheader_dev);
    cudaFree(d_scramble_seq_dev);
    cudaFree(d_rrc_taps_dev);
    cudaStreamDestroy(d_stream);
}

void tx_pipeline_cuda::reset_state()
{
    CUDA_CHECK(cudaMemsetAsync(d_symbol_buf_dev, 0,
                               d_history_len * 2 * sizeof(float), d_stream));
    if (d_batch_symbol_buf_dev) {
        CUDA_CHECK(cudaMemsetAsync(d_batch_symbol_buf_dev, 0,
                                   d_history_len * 2 * sizeof(float), d_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(d_stream));
}

void tx_pipeline_cuda::register_modcod(
    int modcod_idx,
    const float* constellation_lut_host, int n_mod,
    unsigned int fecframe_len, unsigned int xfecframe_len,
    const unsigned int* col_offsets, int stride,
    unsigned int payload_len, unsigned int plframe_len,
    int n_pilots, bool has_pilots)
{
    tx_pipeline_modcod_config cfg;
    cfg.n_mod = n_mod;
    cfg.fecframe_len = fecframe_len;
    // 128APSK mapper reads beyond fecframe (padding); compute padded byte stride
    cfg.fecframe_bytes_padded = (n_mod == 7)
        ? (fecframe_len + 90 + 7) / 8   // 8112 for 128APSK
        : fecframe_len / 8;
    cfg.xfecframe_len = xfecframe_len;
    cfg.payload_len = payload_len;
    cfg.plframe_len = plframe_len;
    cfg.n_pilots = n_pilots;
    cfg.has_pilots = has_pilots;
    cfg.stride = stride;
    for (int i = 0; i < 8; i++)
        cfg.col_offsets[i] = (i < n_mod) ? col_offsets[i] : 0;

    /* Upload constellation LUT */
    int lut_entries = 1 << n_mod;
    CUDA_CHECK(cudaMalloc(&cfg.d_constellation_dev, lut_entries * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(cfg.d_constellation_dev, constellation_lut_host,
                          lut_entries * 2 * sizeof(float), cudaMemcpyHostToDevice));

    d_configs[modcod_idx] = cfg;

    fprintf(stderr, "tx_pipeline_cuda: registered mc=%d n_mod=%d xfec=%u "
            "payload=%u plframe=%u pilots=%d(%d)\n",
            modcod_idx, n_mod, xfecframe_len, payload_len, plframe_len,
            has_pilots, n_pilots);
}

unsigned int tx_pipeline_cuda::process_device(int modcod_idx,
                                               const uint8_t* fecframe_device,
                                               const float* plheader,
                                               float* iq_out,
                                               cudaStream_t stream)
{
    auto it = d_configs.find(modcod_idx);
    if (it == d_configs.end()) {
        fprintf(stderr, "tx_pipeline_cuda: modcod %d not registered!\n", modcod_idx);
        return 0;
    }
    const auto& cfg = it->second;
    const int plheader_len = 90;
    const unsigned int n_symbols = cfg.plframe_len;
    const unsigned int n_out = n_symbols * d_sps;

    /* 1. No FECFRAME upload needed — already on device */

    /* 2. Mapper kernel: packed FECFRAME → XFECFRAME symbols (device → device) */
    {
        int threads = 256;
        int blocks = (cfg.xfecframe_len + threads - 1) / threads;
        pipeline_mapper_kernel<<<blocks, threads, 0, stream>>>(
            fecframe_device, d_xfecframe_dev, cfg.d_constellation_dev,
            cfg.xfecframe_len, cfg.n_mod,
            cfg.col_offsets[0], cfg.col_offsets[1], cfg.col_offsets[2],
            cfg.col_offsets[3], cfg.col_offsets[4], cfg.col_offsets[5],
            cfg.col_offsets[6], cfg.col_offsets[7], cfg.stride);
        CUDA_CHECK_LAUNCH();
    }

    /* 3. Write PLHEADER to symbol_buf (after history, before payload).
     *    Device memory — must be a stream-ordered H2D copy (see process()). */
    float* plheader_dst = d_symbol_buf_dev + d_history_len * 2;
    CUDA_CHECK(cudaMemcpyAsync(plheader_dst, plheader,
                               plheader_len * 2 * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    /* 4. Fused PL frame + scramble kernel */
    {
        float* payload_dst = d_symbol_buf_dev + (d_history_len + plheader_len) * 2;
        int threads = 256;
        int blocks = (cfg.payload_len + threads - 1) / threads;
        pl_frame_scramble_kernel<<<blocks, threads, 0, stream>>>(
            d_xfecframe_dev, d_scramble_seq_dev, payload_dst,
            cfg.payload_len, cfg.n_pilots, cfg.has_pilots);
        CUDA_CHECK_LAUNCH();
    }

    /* 5. RRC pulse shaping kernel */
    {
        int threads = 256;
        int blocks = (n_out + threads - 1) / threads;
        int smem = (d_sub_taps * d_sps + (threads / d_sps + d_sub_taps) * 2) * sizeof(float);
        pipeline_rrc_kernel<<<blocks, threads, smem, stream>>>(
            d_symbol_buf_dev, d_iq_dev, d_rrc_taps_dev,
            n_symbols, d_sub_taps, d_sps);
        CUDA_CHECK_LAUNCH();
    }

    /* 6. IQ output is in managed d_iq_dev — caller reads via iq_managed() after sync */

    /* 7. Update history */
    CUDA_CHECK(cudaMemcpyAsync(
        d_symbol_buf_dev,
        d_symbol_buf_dev + n_symbols * 2,
        d_history_len * 2 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    /* No sync — caller is responsible */
    return n_out;
}

unsigned int tx_pipeline_cuda::process_symbols(const float* symbols_host,
                                                unsigned int n_symbols,
                                                cudaStream_t stream)
{
    if (!symbols_host || n_symbols == 0 || n_symbols > d_max_plframe_len)
        return 0;
    const unsigned int n_out = n_symbols * d_sps;
    float* dst = d_symbol_buf_dev + d_history_len * 2;
    CUDA_CHECK(cudaMemcpyAsync(dst, symbols_host,
        (size_t)n_symbols * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    int threads = 256;
    int blocks = (n_out + threads - 1) / threads;
    int smem = (d_sub_taps * d_sps +
        (threads / d_sps + d_sub_taps) * 2) * sizeof(float);
    pipeline_rrc_kernel<<<blocks, threads, smem, stream>>>(
        d_symbol_buf_dev, d_iq_dev, d_rrc_taps_dev,
        n_symbols, d_sub_taps, d_sps);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpyAsync(d_symbol_buf_dev,
        d_symbol_buf_dev + n_symbols * 2,
        d_history_len * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    return n_out;
}

void tx_pipeline_cuda::ensure_batch_pipeline_buffers(
    int batch_size, unsigned int plframe_len, unsigned int xfecframe_len)
{
    if (batch_size <= d_batch_pipeline_capacity &&
        plframe_len <= d_batch_plframe_len)
        return;

    cudaFree(d_batch_xfecframe_dev);
    cudaFree(d_batch_symbol_buf_dev);
    cudaFree(d_batch_iq_dev);
    cudaFree(d_batch_plheader_dev);

    size_t xfec_size = (size_t)batch_size * xfecframe_len * 2 * sizeof(float);
    size_t sym_buf_size = (size_t)(d_history_len + batch_size * plframe_len) * 2 * sizeof(float);
    size_t iq_size = (size_t)batch_size * plframe_len * d_sps * 2 * sizeof(float);
    size_t plh_size = (size_t)batch_size * 90 * 2 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_batch_xfecframe_dev, xfec_size));
    CUDA_CHECK(cudaMalloc(&d_batch_symbol_buf_dev, sym_buf_size));
    CUDA_CHECK(cudaMemset(d_batch_symbol_buf_dev, 0,
                          d_history_len * 2 * sizeof(float)));
    // Both buffers are accessed directly by CUDA kernels and by the host.
    // Pinned host allocations are not necessarily device-mapped (notably on
    // GH200), so using them as kernel pointers can fail at launch. Managed
    // allocations provide valid UVA pointers on every supported platform.
    CUDA_CHECK(cudaMallocManaged(&d_batch_iq_dev, iq_size));
    CUDA_CHECK(cudaMallocManaged(&d_batch_plheader_dev, plh_size));

    d_batch_pipeline_capacity = batch_size;
    d_batch_plframe_len = plframe_len;

    size_t total_kb = (xfec_size + sym_buf_size + iq_size + plh_size) / 1024;
    fprintf(stderr, "tx_pipeline_cuda: batch buffers for %d frames "
            "(plframe=%u, %zu KB device)\n", batch_size, plframe_len, total_kb);
}

unsigned int tx_pipeline_cuda::process_batch(
    int modcod_idx,
    const uint8_t* fecframes_device,
    const float* plheaders_host,
    float* iq_out_host,
    int batch_size,
    cudaStream_t stream)
{
    auto it = d_configs.find(modcod_idx);
    if (it == d_configs.end()) {
        fprintf(stderr, "tx_pipeline_cuda: modcod %d not registered!\n", modcod_idx);
        return 0;
    }
    const auto& cfg = it->second;
    const unsigned int plframe_len = cfg.plframe_len;
    const unsigned int total_symbols = batch_size * plframe_len;
    const unsigned int total_iq = total_symbols * d_sps;
    const unsigned int fecframe_bytes = cfg.fecframe_len / 8;

    ensure_batch_pipeline_buffers(batch_size, plframe_len, cfg.xfecframe_len);

    /* The single-frame/dummy path and the batched data path use different
     * work buffers, but they are one continuous RRC stream.  Import the
     * canonical history before filtering.  Without this copy, every
     * dummy-to-data transition spliced the batch onto the tail of the last
     * data batch (possibly seconds old), corrupting complete probe bursts. */
    CUDA_CHECK(cudaMemcpyAsync(
        d_batch_symbol_buf_dev, d_symbol_buf_dev,
        d_history_len * 2 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    /* 1. Copy N PLHEADERs to managed staging buffer */
    memcpy(d_batch_plheader_dev, plheaders_host,
           (size_t)batch_size * 90 * 2 * sizeof(float));

    /* 2. Scatter PLHEADERs into symbol_buf at correct per-frame positions */
    plheader_scatter_kernel<<<batch_size, 90, 0, stream>>>(
        d_batch_plheader_dev, d_batch_symbol_buf_dev,
        d_history_len, plframe_len);
    CUDA_CHECK_LAUNCH();

    /* 3. Fused mapper + PL frame + scramble: FECFRAME bits → scrambled payload
     *    Eliminates intermediate XFECFRAME buffer (saves one kernel launch +
     *    one full-sized device read/write). */
    {
        int threads = 256;
        dim3 grid((cfg.payload_len + threads - 1) / threads, batch_size);
        fused_map_plscr_batch_kernel<<<grid, threads, 0, stream>>>(
            fecframes_device, cfg.d_constellation_dev,
            d_scramble_seq_dev, d_batch_symbol_buf_dev,
            cfg.payload_len, cfg.n_pilots, cfg.has_pilots,
            cfg.xfecframe_len, plframe_len, d_history_len,
            cfg.n_mod,
            cfg.col_offsets[0], cfg.col_offsets[1], cfg.col_offsets[2],
            cfg.col_offsets[3], cfg.col_offsets[4], cfg.col_offsets[5],
            cfg.col_offsets[6], cfg.col_offsets[7], cfg.stride,
            fecframe_bytes);
        CUDA_CHECK_LAUNCH();
    }

    /* 5. RRC pulse shaping over all N frames contiguously.
     *    symbol_buf = [history | PF_0 | PF_1 | ... | PF_{N-1}]
     *    The filter naturally spans frame boundaries. */
    {
        int threads = 256;
        int blocks = (total_iq + threads - 1) / threads;
        int smem = (d_sub_taps * d_sps + (threads / d_sps + d_sub_taps) * 2) * sizeof(float);
        pipeline_rrc_kernel<<<blocks, threads, smem, stream>>>(
            d_batch_symbol_buf_dev, d_batch_iq_dev, d_rrc_taps_dev,
            total_symbols, d_sub_taps, d_sps);
        CUDA_CHECK_LAUNCH();
    }

    /* 6. IQ output is in managed d_batch_iq_dev — caller reads via
     *    batch_iq_managed() after sync. No D2H copy needed on GH200. */

    /* 7. Publish the final history to both work buffers.  The single-frame
     * buffer is canonical so a following dummy frame starts seamlessly; the
     * batch copy keeps consecutive batches seamless too. */
    CUDA_CHECK(cudaMemcpyAsync(
        d_symbol_buf_dev,
        d_batch_symbol_buf_dev + total_symbols * 2,
        d_history_len * 2 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_batch_symbol_buf_dev,
        d_batch_symbol_buf_dev + total_symbols * 2,
        d_history_len * 2 * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    /* No sync — caller is responsible. */
    return total_iq;
}

} // namespace cuda
} // namespace cusat_dsp
