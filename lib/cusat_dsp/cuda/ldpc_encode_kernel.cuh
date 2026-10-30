/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA LDPC encoder using the IRA (Irregular Repeat-Accumulate) algorithm.
 *
 * The encoding table uses a circulant structure: each group of M=360 info bits
 * shares base positions, with positions shifted by q per row within the group.
 * The flat ldpc_code arrays (row_deg/row_off/addr) are uploaded as-is.
 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_ENCODE_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_LDPC_ENCODE_KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace cusat_dsp {
struct ldpc_code; // forward

namespace cuda {

class ldpc_encode_cuda
{
public:
    /**
     * @param code      Flat LDPC table description (host-side, static data).
     * @param K         Number of info bits.
     * @param R         Number of parity bits (N - K).
     */
    ldpc_encode_cuda(const ldpc_code* code, unsigned int K, unsigned int R);
    ~ldpc_encode_cuda();

    /**
     * @brief Encode one LDPC codeword on GPU.
     *
     * @param info_packed    Host pointer to K/8 bytes of packed info bits (MSB first).
     * @param codeword_packed Host pointer for output N/8 bytes (systematic: info + parity).
     */
    void encode(const uint8_t* info_packed, uint8_t* codeword_packed);

    /**
     * @brief Encode one LDPC codeword to a device-resident FECFRAME buffer.
     *
     * Uses the caller's stream and does NOT synchronize — the caller must
     * sync after all dependent work is submitted. This enables chaining
     * LDPC → mapper → PL → RRC on a single stream with no host round-trips.
     *
     * @param info_packed_host  Host pointer to K/8 bytes of packed info bits.
     * @param fecframe_device   Device pointer to N/8 bytes output (systematic + parity).
     * @param stream            CUDA stream to use (no internal sync).
     */
    void encode_to_device(const uint8_t* info_packed_host,
                          uint8_t* fecframe_device,
                          cudaStream_t stream);

    /**
     * @brief Encode N LDPC codewords in a single batched GPU launch.
     *
     * All codewords must use the same LDPC code (same K, R, table).
     * Amortizes kernel launch overhead and improves GPU occupancy.
     * Uses the caller's stream and does NOT synchronize.
     *
     * @param info_packed_host  Host pointer to batch_size × K/8 bytes contiguous.
     * @param fecframes_device  Device pointer to batch_size × N/8 bytes contiguous.
     * @param batch_size        Number of codewords to encode.
     * @param stream            CUDA stream to use (no internal sync).
     */
    void encode_batch_to_device(const uint8_t* info_packed_host,
                                uint8_t* fecframes_device,
                                int batch_size,
                                cudaStream_t stream);

    /**
     * @brief Encode N LDPC codewords from device-resident info buffers (D2D).
     *
     * Same as encode_batch_to_device but reads info bits from device memory
     * instead of host memory. Use when BCH output is already on device.
     */
    void encode_batch_d2d(const uint8_t* info_packed_device,
                          uint8_t* fecframes_device,
                          int batch_size,
                          cudaStream_t stream);

    unsigned int fecframe_bytes() const { return (d_K + d_R) / 8; }
    unsigned int info_bytes() const { return d_K / 8; }

private:
    void ensure_batch_buffers(int batch_size);

    unsigned int d_K, d_R, d_M, d_q;
    unsigned int d_n_rows;         /* K / M */

    /* Device buffers (single-frame) */
    uint8_t*  d_info_dev;          /* packed info bits [K/8] */
    uint32_t* d_parity_words_dev;  /* scattered parity [(R+31)/32 uint32_t words] */
    uint8_t*  d_parity_unpacked_dev; /* unpacked parity [R] */
    uint8_t*  d_parity_packed_dev; /* packed parity [R/8] */

    /* Device buffers (batched, allocated on demand) */
    uint8_t*  d_batch_info_dev = nullptr;
    uint32_t* d_batch_parity_words_dev = nullptr;
    uint8_t*  d_batch_parity_unpacked_dev = nullptr;
    uint8_t*  d_batch_parity_packed_dev = nullptr;
    int       d_batch_capacity = 0;

    /* Table on device (uploaded from the flat ldpc_code arrays) */
    uint8_t*  d_degrees_dev;       /* [n_rows] */
    uint16_t* d_base_pos_dev;      /* [total_positions] */
    uint32_t* d_row_offset_dev;    /* [n_rows] cumulative offset into base_pos */

    cudaStream_t d_stream;
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_LDPC_ENCODE_KERNEL_CUH */
