/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA IRA LDPC encoder.
 *
 * Algorithm (per ETSI EN 302 307 Annex C):
 *   1. p[0..R-1] = 0
 *   2. For each info bit i (0..K-1), if bit=1:
 *        For each accumulator address a[j] from table:
 *          p[a[j]] ^= 1
 *      Addresses use circulant: a[j] = (base[j] + (i%M)*q) % R
 *   3. p[i] ^= p[i-1] for i=1..R-1  (accumulate)
 *
 * CUDA parallelization:
 *   - Step 2: one thread per info bit, atomicXor into parity words
 *   - Step 3: one block per codeword, parallel prefix XOR over parity words
 */

#include "ldpc_encode_kernel.cuh"
#include "cuda_util.cuh"
#include "ldpc_tables/ldpc_code.h"
#include <cstdio>
#include <cstring>

namespace cusat_dsp {
namespace cuda {

static constexpr int LDPC_ACCUM_THREADS = 1024;
static constexpr int LDPC_ACCUM_WORDS_PER_BLOCK = 2 * LDPC_ACCUM_THREADS;

/*
 * Kernel: scatter-XOR phase.
 * One thread per info bit. If bit=1, XOR into parity word positions.
 */
__global__ void ldpc_scatter_xor_kernel(
    const uint8_t* __restrict__ info_packed,
    uint32_t* __restrict__ parity_words,
    const uint8_t* __restrict__ degrees,
    const uint16_t* __restrict__ base_pos,
    const uint32_t* __restrict__ row_offset,
    int K, int M, int q, int R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K)
        return;

    /* Extract bit from packed input (MSB first) */
    uint8_t bit = (info_packed[i / 8] >> (7 - (i & 7))) & 1;
    if (!bit)
        return;

    /* Compute positions from circulant table */
    int row = i / M;
    int r = i % M;
    int deg = degrees[row];
    uint32_t offset = row_offset[row];

    for (int j = 0; j < deg; j++) {
        int pos = ((int)base_pos[offset + j] + r * q) % R;
        atomicXor(&parity_words[pos >> 5], 1u << (pos & 31));
    }
}

__device__ __forceinline__ uint32_t prefix_xor_bits(uint32_t x)
{
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    return x;
}

__device__ void ldpc_accumulate_one_codeword(
    const uint32_t* __restrict__ my_pw,
    uint8_t* __restrict__ my_out,
    int R, int pw_count,
    uint32_t* __restrict__ scan,
    uint32_t* __restrict__ tmp)
{
    const int tid = threadIdx.x;
    const int idx0 = tid;
    const int idx1 = tid + blockDim.x;
    const int max_words = 2 * blockDim.x;

    if (pw_count > max_words) {
        if (tid == 0) {
            uint8_t prev = 0;
            for (int i = 0; i < R; i++) {
                uint8_t bit = (my_pw[i >> 5] >> (i & 31)) & 1;
                bit ^= prev;
                my_out[i] = bit;
                prev = bit;
            }
        }
        return;
    }

    scan[idx0] = (idx0 < pw_count) ? (__popc(my_pw[idx0]) & 1u) : 0u;
    scan[idx1] = (idx1 < pw_count) ? (__popc(my_pw[idx1]) & 1u) : 0u;
    __syncthreads();

    for (int offset = 1; offset < max_words; offset <<= 1) {
        uint32_t v0 = scan[idx0];
        uint32_t v1 = scan[idx1];
        if (idx0 >= offset) v0 ^= scan[idx0 - offset];
        if (idx1 >= offset) v1 ^= scan[idx1 - offset];
        tmp[idx0] = v0;
        tmp[idx1] = v1;
        __syncthreads();
        scan[idx0] = tmp[idx0];
        scan[idx1] = tmp[idx1];
        __syncthreads();
    }

    for (int idx = tid; idx < pw_count; idx += blockDim.x) {
        uint32_t prefix = prefix_xor_bits(my_pw[idx]);
        uint32_t carry = (idx == 0) ? 0u : scan[idx - 1];
        if (carry) prefix = ~prefix;

        int bit_base = idx * 32;
        #pragma unroll
        for (int b = 0; b < 32; b++) {
            int bit_idx = bit_base + b;
            if (bit_idx < R)
                my_out[bit_idx] = (uint8_t)((prefix >> b) & 1u);
        }
    }
}

/*
 * Kernel: accumulate + unpack.
 * Inclusive prefix XOR over packed parity words, output unpacked bytes.
 */
__global__ void ldpc_accumulate_kernel(
    const uint32_t* __restrict__ parity_words,
    uint8_t* __restrict__ parity_unpacked,
    int R, int pw_count)
{
    extern __shared__ uint32_t scratch[];
    ldpc_accumulate_one_codeword(parity_words, parity_unpacked, R, pw_count,
                                 scratch, scratch + LDPC_ACCUM_WORDS_PER_BLOCK);
}

/*
 * Kernel: pack unpacked parity bytes into packed MSB-first output.
 * One thread per output byte.
 */
__global__ void ldpc_pack_parity_kernel(
    const uint8_t* __restrict__ parity_unpacked,
    uint8_t* __restrict__ parity_packed,
    int R)
{
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_start = byte_idx * 8;
    if (bit_start >= R)
        return;

    uint8_t byte = 0;
    for (int b = 0; b < 8 && (bit_start + b) < R; b++) {
        byte |= (parity_unpacked[bit_start + b] << (7 - b));
    }
    parity_packed[byte_idx] = byte;
}

/* ----- Batched kernels ----- */

/*
 * Batched scatter-XOR: blockIdx.y selects the codeword.
 * info_packed is batch_size × K/8 contiguous, parity_words is batch_size × pw_count.
 */
__global__ void ldpc_scatter_xor_batch_kernel(
    const uint8_t* __restrict__ info_packed,
    uint32_t* __restrict__ parity_words,
    const uint8_t* __restrict__ degrees,
    const uint16_t* __restrict__ base_pos,
    const uint32_t* __restrict__ row_offset,
    int K, int M, int q, int R, int pw_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K)
        return;
    int cw = blockIdx.y;

    const uint8_t* my_info = info_packed + cw * (K / 8);
    uint32_t* my_parity = parity_words + cw * pw_count;

    uint8_t bit = (my_info[i / 8] >> (7 - (i & 7))) & 1;
    if (!bit)
        return;

    int row = i / M;
    int r = i % M;
    int deg = degrees[row];
    uint32_t offset = row_offset[row];

    for (int j = 0; j < deg; j++) {
        int pos = ((int)base_pos[offset + j] + r * q) % R;
        atomicXor(&my_parity[pos >> 5], 1u << (pos & 31));
    }
}

/* Batched accumulate: one block per codeword. */
__global__ void ldpc_accumulate_batch_kernel(
    const uint32_t* __restrict__ parity_words,
    uint8_t* __restrict__ parity_unpacked,
    int R, int pw_count)
{
    int cw = blockIdx.x;
    const uint32_t* my_pw = parity_words + cw * pw_count;
    uint8_t* my_out = parity_unpacked + cw * R;
    extern __shared__ uint32_t scratch[];
    ldpc_accumulate_one_codeword(my_pw, my_out, R, pw_count,
                                 scratch, scratch + LDPC_ACCUM_WORDS_PER_BLOCK);
}

/*
 * Batched pack: blockIdx.y selects the codeword.
 */
__global__ void ldpc_pack_parity_batch_kernel(
    const uint8_t* __restrict__ parity_unpacked,
    uint8_t* __restrict__ parity_packed,
    int R)
{
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_start = byte_idx * 8;
    if (bit_start >= R)
        return;
    int cw = blockIdx.y;

    const uint8_t* my_in = parity_unpacked + cw * R;
    uint8_t* my_out = parity_packed + cw * (R / 8);

    uint8_t byte = 0;
    for (int b = 0; b < 8 && (bit_start + b) < R; b++) {
        byte |= (my_in[bit_start + b] << (7 - b));
    }
    my_out[byte_idx] = byte;
}

/*
 * Assemble kernel: copy systematic + parity into contiguous FECFRAME slots.
 * blockIdx.y = codeword index, threads cover bytes within one FECFRAME.
 */
__global__ void ldpc_assemble_fecframe_kernel(
    const uint8_t* __restrict__ info_packed,
    const uint8_t* __restrict__ parity_packed,
    uint8_t* __restrict__ fecframes,
    int info_bytes, int parity_bytes, int fecframe_bytes)
{
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= fecframe_bytes)
        return;
    int cw = blockIdx.y;

    uint8_t val;
    if (byte_idx < info_bytes)
        val = info_packed[cw * info_bytes + byte_idx];
    else
        val = parity_packed[cw * parity_bytes + (byte_idx - info_bytes)];

    fecframes[cw * fecframe_bytes + byte_idx] = val;
}

/* ----- Host API ----- */

ldpc_encode_cuda::ldpc_encode_cuda(const ldpc_code* code, unsigned int K, unsigned int R)
    : d_K(K), d_R(R), d_M(code->m), d_q(R / code->m)
{
    d_n_rows = K / d_M;

    CUDA_CHECK(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));

    /* The flat ldpc_code arrays are already in the device layout:
     * per-row degree, prefix offset, and base accumulator addresses. */
    unsigned int total_pos = code->row_off[d_n_rows];

    CUDA_CHECK(cudaMalloc(&d_degrees_dev, d_n_rows * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_degrees_dev, code->row_deg,
                          d_n_rows * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_base_pos_dev, total_pos * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_base_pos_dev, code->addr,
                          total_pos * sizeof(uint16_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_row_offset_dev, d_n_rows * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_row_offset_dev, code->row_off,
                          d_n_rows * sizeof(uint32_t), cudaMemcpyHostToDevice));

    /* Allocate working buffers */
    CUDA_CHECK(cudaMalloc(&d_info_dev, K / 8));
    unsigned int parity_words = (R + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_parity_words_dev, parity_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_parity_unpacked_dev, R));
    CUDA_CHECK(cudaMalloc(&d_parity_packed_dev, R / 8));

    fprintf(stderr, "ldpc_encode_cuda: K=%u R=%u M=%u q=%u rows=%u total_pos=%u\n",
            K, R, d_M, d_q, d_n_rows, total_pos);
}

ldpc_encode_cuda::~ldpc_encode_cuda()
{
    cudaFree(d_info_dev);
    cudaFree(d_parity_words_dev);
    cudaFree(d_parity_unpacked_dev);
    cudaFree(d_parity_packed_dev);
    cudaFree(d_batch_info_dev);
    cudaFree(d_batch_parity_words_dev);
    cudaFree(d_batch_parity_unpacked_dev);
    cudaFree(d_batch_parity_packed_dev);
    cudaFree(d_degrees_dev);
    cudaFree(d_base_pos_dev);
    cudaFree(d_row_offset_dev);
    cudaStreamDestroy(d_stream);
}

void ldpc_encode_cuda::ensure_batch_buffers(int batch_size)
{
    if (batch_size <= d_batch_capacity)
        return;

    /* Free old batch buffers */
    cudaFree(d_batch_info_dev);
    cudaFree(d_batch_parity_words_dev);
    cudaFree(d_batch_parity_unpacked_dev);
    cudaFree(d_batch_parity_packed_dev);

    unsigned int pw_count = (d_R + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_batch_info_dev, (size_t)batch_size * d_K / 8));
    CUDA_CHECK(cudaMalloc(&d_batch_parity_words_dev,
                          (size_t)batch_size * pw_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_batch_parity_unpacked_dev, (size_t)batch_size * d_R));
    CUDA_CHECK(cudaMalloc(&d_batch_parity_packed_dev, (size_t)batch_size * d_R / 8));

    d_batch_capacity = batch_size;
    fprintf(stderr, "ldpc_encode_cuda: batch buffers for %d codewords "
            "(%zu KB device)\n", batch_size,
            (size_t)batch_size * (d_K/8 + pw_count*4 + d_R + d_R/8) / 1024);
}

void ldpc_encode_cuda::encode(const uint8_t* info_packed, uint8_t* codeword_packed)
{
    unsigned int K = d_K;
    unsigned int R = d_R;
    unsigned int parity_word_count = (R + 31) / 32;

    /* Upload packed info bits */
    CUDA_CHECK(cudaMemcpyAsync(d_info_dev, info_packed,
                               K / 8, cudaMemcpyHostToDevice, d_stream));

    /* Clear parity words */
    CUDA_CHECK(cudaMemsetAsync(d_parity_words_dev, 0,
                               parity_word_count * sizeof(uint32_t), d_stream));

    /* Scatter-XOR: one thread per info bit */
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    ldpc_scatter_xor_kernel<<<blocks, threads, 0, d_stream>>>(
        d_info_dev, d_parity_words_dev,
        d_degrees_dev, d_base_pos_dev, d_row_offset_dev,
        K, d_M, d_q, R);
    CUDA_CHECK_LAUNCH();

    size_t accum_smem = 2 * LDPC_ACCUM_WORDS_PER_BLOCK * sizeof(uint32_t);

    /* Accumulate: prefix XOR scan over packed parity words */
    ldpc_accumulate_kernel<<<1, LDPC_ACCUM_THREADS, accum_smem, d_stream>>>(
        d_parity_words_dev, d_parity_unpacked_dev, R, parity_word_count);
    CUDA_CHECK_LAUNCH();

    /* Pack parity bits */
    int pack_threads = 256;
    int pack_blocks = ((R / 8) + pack_threads - 1) / pack_threads;
    ldpc_pack_parity_kernel<<<pack_blocks, pack_threads, 0, d_stream>>>(
        d_parity_unpacked_dev, d_parity_packed_dev, R);
    CUDA_CHECK_LAUNCH();

    /* Copy systematic info bits directly (no encoding needed) */
    memcpy(codeword_packed, info_packed, K / 8);

    /* Copy packed parity from device */
    CUDA_CHECK(cudaMemcpyAsync(codeword_packed + K / 8, d_parity_packed_dev,
                               R / 8, cudaMemcpyDeviceToHost, d_stream));
    CUDA_CHECK(cudaStreamSynchronize(d_stream));
}

void ldpc_encode_cuda::encode_to_device(const uint8_t* info_packed_host,
                                         uint8_t* fecframe_device,
                                         cudaStream_t stream)
{
    unsigned int K = d_K;
    unsigned int R = d_R;
    unsigned int parity_word_count = (R + 31) / 32;

    /* Upload packed info bits to internal buffer */
    CUDA_CHECK(cudaMemcpyAsync(d_info_dev, info_packed_host,
                               K / 8, cudaMemcpyHostToDevice, stream));

    /* Clear parity words */
    CUDA_CHECK(cudaMemsetAsync(d_parity_words_dev, 0,
                               parity_word_count * sizeof(uint32_t), stream));

    /* Scatter-XOR: one thread per info bit */
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    ldpc_scatter_xor_kernel<<<blocks, threads, 0, stream>>>(
        d_info_dev, d_parity_words_dev,
        d_degrees_dev, d_base_pos_dev, d_row_offset_dev,
        K, d_M, d_q, R);
    CUDA_CHECK_LAUNCH();

    size_t accum_smem = 2 * LDPC_ACCUM_WORDS_PER_BLOCK * sizeof(uint32_t);

    /* Accumulate: prefix XOR scan over packed parity words */
    ldpc_accumulate_kernel<<<1, LDPC_ACCUM_THREADS, accum_smem, stream>>>(
        d_parity_words_dev, d_parity_unpacked_dev, R, parity_word_count);
    CUDA_CHECK_LAUNCH();

    /* Pack parity bits */
    int pack_threads = 256;
    int pack_blocks = ((R / 8) + pack_threads - 1) / pack_threads;
    ldpc_pack_parity_kernel<<<pack_blocks, pack_threads, 0, stream>>>(
        d_parity_unpacked_dev, d_parity_packed_dev, R);
    CUDA_CHECK_LAUNCH();

    /* Copy systematic info bits to output FECFRAME (D2D) */
    CUDA_CHECK(cudaMemcpyAsync(fecframe_device, d_info_dev,
                               K / 8, cudaMemcpyDeviceToDevice, stream));

    /* Copy packed parity to output FECFRAME (D2D) */
    CUDA_CHECK(cudaMemcpyAsync(fecframe_device + K / 8, d_parity_packed_dev,
                               R / 8, cudaMemcpyDeviceToDevice, stream));

    /* No sync — caller is responsible for synchronizing */
}

void ldpc_encode_cuda::encode_batch_to_device(const uint8_t* info_packed_host,
                                               uint8_t* fecframes_device,
                                               int batch_size,
                                               cudaStream_t stream)
{
    if (batch_size == 1) {
        encode_to_device(info_packed_host, fecframes_device, stream);
        return;
    }

    unsigned int K = d_K;
    unsigned int R = d_R;
    unsigned int ib = K / 8;          /* info bytes per codeword */
    unsigned int pb = R / 8;          /* parity bytes per codeword */
    unsigned int fb = (K + R) / 8;    /* fecframe bytes per codeword */
    unsigned int pw_count = (R + 31) / 32;

    ensure_batch_buffers(batch_size);

    /* Single H2D transfer: all info blocks */
    CUDA_CHECK(cudaMemcpyAsync(d_batch_info_dev, info_packed_host,
                               (size_t)batch_size * ib,
                               cudaMemcpyHostToDevice, stream));

    /* Clear all parity words */
    CUDA_CHECK(cudaMemsetAsync(d_batch_parity_words_dev, 0,
                               (size_t)batch_size * pw_count * sizeof(uint32_t),
                               stream));

    /* Batched scatter-XOR: grid.y = codeword index */
    int threads = 256;
    dim3 scatter_grid((K + threads - 1) / threads, batch_size);
    ldpc_scatter_xor_batch_kernel<<<scatter_grid, threads, 0, stream>>>(
        d_batch_info_dev, d_batch_parity_words_dev,
        d_degrees_dev, d_base_pos_dev, d_row_offset_dev,
        K, d_M, d_q, R, pw_count);
    CUDA_CHECK_LAUNCH();

    size_t accum_smem = 2 * LDPC_ACCUM_WORDS_PER_BLOCK * sizeof(uint32_t);

    /* Batched accumulate: one block per codeword */
    ldpc_accumulate_batch_kernel<<<batch_size, LDPC_ACCUM_THREADS,
                                   accum_smem, stream>>>(
        d_batch_parity_words_dev, d_batch_parity_unpacked_dev,
        R, pw_count);
    CUDA_CHECK_LAUNCH();

    /* Batched pack parity */
    int pack_threads = 256;
    dim3 pack_grid((pb + pack_threads - 1) / pack_threads, batch_size);
    ldpc_pack_parity_batch_kernel<<<pack_grid, pack_threads, 0, stream>>>(
        d_batch_parity_unpacked_dev, d_batch_parity_packed_dev, R);
    CUDA_CHECK_LAUNCH();

    /* Assemble contiguous FECFRAMEs: systematic + parity in one kernel */
    int asm_threads = 256;
    dim3 asm_grid((fb + asm_threads - 1) / asm_threads, batch_size);
    ldpc_assemble_fecframe_kernel<<<asm_grid, asm_threads, 0, stream>>>(
        d_batch_info_dev, d_batch_parity_packed_dev,
        fecframes_device, ib, pb, fb);
    CUDA_CHECK_LAUNCH();

    /* No sync — caller is responsible */
}

void ldpc_encode_cuda::encode_batch_d2d(const uint8_t* info_packed_device,
                                         uint8_t* fecframes_device,
                                         int batch_size,
                                         cudaStream_t stream)
{
    if (batch_size == 1) {
        /* Single frame: D2D copy to internal buffer, then same pipeline */
        unsigned int K = d_K;
        unsigned int R = d_R;
        unsigned int parity_word_count = (R + 31) / 32;

        CUDA_CHECK(cudaMemcpyAsync(d_info_dev, info_packed_device,
                                   K / 8, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(d_parity_words_dev, 0,
                                   parity_word_count * sizeof(uint32_t), stream));
        int threads = 256;
        int blocks = (K + threads - 1) / threads;
        ldpc_scatter_xor_kernel<<<blocks, threads, 0, stream>>>(
            d_info_dev, d_parity_words_dev,
            d_degrees_dev, d_base_pos_dev, d_row_offset_dev,
            K, d_M, d_q, R);
        CUDA_CHECK_LAUNCH();
        size_t accum_smem = 2 * LDPC_ACCUM_WORDS_PER_BLOCK * sizeof(uint32_t);
        ldpc_accumulate_kernel<<<1, LDPC_ACCUM_THREADS, accum_smem, stream>>>(
            d_parity_words_dev, d_parity_unpacked_dev, R, parity_word_count);
        CUDA_CHECK_LAUNCH();
        int pack_threads = 256;
        int pack_blocks = ((R / 8) + pack_threads - 1) / pack_threads;
        ldpc_pack_parity_kernel<<<pack_blocks, pack_threads, 0, stream>>>(
            d_parity_unpacked_dev, d_parity_packed_dev, R);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpyAsync(fecframes_device, d_info_dev,
                                   K / 8, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(fecframes_device + K / 8, d_parity_packed_dev,
                                   R / 8, cudaMemcpyDeviceToDevice, stream));
        return;
    }

    unsigned int K = d_K;
    unsigned int R = d_R;
    unsigned int ib = K / 8;
    unsigned int pb = R / 8;
    unsigned int fb = (K + R) / 8;
    unsigned int pw_count = (R + 31) / 32;

    ensure_batch_buffers(batch_size);

    /* D2D transfer: all info blocks from device memory */
    CUDA_CHECK(cudaMemcpyAsync(d_batch_info_dev, info_packed_device,
                               (size_t)batch_size * ib,
                               cudaMemcpyDeviceToDevice, stream));

    /* Clear all parity words */
    CUDA_CHECK(cudaMemsetAsync(d_batch_parity_words_dev, 0,
                               (size_t)batch_size * pw_count * sizeof(uint32_t),
                               stream));

    /* Batched scatter-XOR */
    int threads = 256;
    dim3 scatter_grid((K + threads - 1) / threads, batch_size);
    ldpc_scatter_xor_batch_kernel<<<scatter_grid, threads, 0, stream>>>(
        d_batch_info_dev, d_batch_parity_words_dev,
        d_degrees_dev, d_base_pos_dev, d_row_offset_dev,
        K, d_M, d_q, R, pw_count);
    CUDA_CHECK_LAUNCH();

    size_t accum_smem = 2 * LDPC_ACCUM_WORDS_PER_BLOCK * sizeof(uint32_t);

    /* Batched accumulate */
    ldpc_accumulate_batch_kernel<<<batch_size, LDPC_ACCUM_THREADS,
                                   accum_smem, stream>>>(
        d_batch_parity_words_dev, d_batch_parity_unpacked_dev,
        R, pw_count);
    CUDA_CHECK_LAUNCH();

    /* Batched pack parity */
    int pack_threads = 256;
    dim3 pack_grid((pb + pack_threads - 1) / pack_threads, batch_size);
    ldpc_pack_parity_batch_kernel<<<pack_grid, pack_threads, 0, stream>>>(
        d_batch_parity_unpacked_dev, d_batch_parity_packed_dev, R);
    CUDA_CHECK_LAUNCH();

    /* Assemble contiguous FECFRAMEs */
    int asm_threads = 256;
    dim3 asm_grid((fb + asm_threads - 1) / asm_threads, batch_size);
    ldpc_assemble_fecframe_kernel<<<asm_grid, asm_threads, 0, stream>>>(
        d_batch_info_dev, d_batch_parity_packed_dev,
        fecframes_device, ib, pb, fb);
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace cusat_dsp
