/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA LDPC decoder — layered offset min-sum (FACTOR=2, int16, NormalUpdate).
 *
 * One CUDA block per codeword, 360 threads per block (= M, quasi-cyclic size).
 * Layers are sequential (data dependency), batch parallelism across codewords.
 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_DECODE_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_LDPC_DECODE_KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cusat_dsp/modcod_types.h>

namespace cusat_dsp {
struct ldpc_code; // forward

namespace cuda {

class ldpc_decode_cuda
{
public:
    /**
     * @param code       Flat LDPC table description (host-side, static data).
     * @param rate       Code rate (selects parallel-layer schedule from registry).
     * @param max_trials Maximum decoding iterations (default 25).
     * @param max_batch  Maximum batch size (pre-allocates buffers).
     */
    ldpc_decode_cuda(const ldpc_code* code, code_rate_t rate,
                     int max_trials = 25, int max_batch = 64);
    ~ldpc_decode_cuda();

    /**
     * @brief Decode a batch of LDPC codewords on GPU.
     *
     * @param soft_in    Host pointer: batch_size × N int16_t soft LLRs (contiguous).
     * @param hard_out   Host pointer: batch_size × k_bytes packed bytes (MSB first).
     * @param trials_out Host pointer: batch_size ints (remaining trials, <0 if failed).
     * @param batch_size Number of codewords in this batch (1..max_batch).
     * @param stream     CUDA stream for async ops.
     */
    void decode_batch(const int16_t* soft_in, uint8_t* hard_out,
                      int* trials_out, int batch_size, cudaStream_t stream);

    /**
     * @brief Decode a batch — device soft input, host hard output.
     *
     * D2D copies soft LLRs, launches decode, D2H copies hard decisions + trials.
     */
    void decode_batch_d2h(const int16_t* soft_in_dev, uint8_t* hard_out_host,
                          int* trials_out_host, int batch_size, cudaStream_t stream);

    /**
     * @brief Launch LDPC decode only — no sync, no D2H.
     *
     * Reads soft LLRs directly from soft_in (no internal copy).
     * After completion the internal managed buffers hold the packed hard
     * decisions and the trial counts.
     *
     * Optional: pass external output buffers (hard_out, result_out)
     * to write results directly there instead of internal buffers.
     */
    void decode_launch_async(const int16_t* soft_in, int batch_size,
                             cudaStream_t stream,
                             uint8_t* hard_out = nullptr,
                             int* result_out = nullptr);

    int n_bits() const { return d_N; }
    int k_bits() const { return d_K; }
    int k_bytes() const { return d_K / 8; }


private:
    int d_K, d_R, d_N, d_M, d_q, d_CNL, d_LT, d_LT_alloc, d_max_trials, d_max_batch;
    bool d_use_smem;  // false on devices where K+R working set exceeds smem optin limit

    // Device: code table (read-only, shared across batch)
    uint16_t* d_pos_dev;            // [q * CNL * M] — edge-major pos table
    uint8_t*  d_cnc_dev;            // [q] — check node counts per layer
    int*      d_bnl_layer_off_dev;  // [q+1] — cumulative bnl offsets

    // Device: per-INSTANCE parallel-layer schedule. MUST be per-instance (not a
    // process-global __constant__): in ACM, the rx_fec_pool holds one decoder
    // per MODCOD, each with a different code/q. A shared symbol would be
    // overwritten by the last-constructed decoder, so an earlier chain's kernel
    // would index pos[]/bnl[] with another code's layer indices -> CUDA 700.
    int16_t*  d_schedule_dev = nullptr; // [n_steps * LANES_PER_STEP] layer ids
    void*     d_cst_dev = nullptr;      // CUSAT_LDPC_CST: compressed check-state tuples
    int       d_schedule_n_steps = 0;

    // Device: per-codeword working buffers (max_batch)
    int16_t*  d_data_dev;           // [max_batch * K]
    int16_t*  d_pty_dev;            // [max_batch * R]
    int8_t*   d_bnl_dev;            // [max_batch * d_LT_alloc] — edge-major int8 messages
    int*      d_result_dev;         // [max_batch]

    // Managed: I/O buffers (CPU + GPU accessible on GH200)
    int16_t*  d_soft_dev;           // [max_batch * N]
    uint8_t*  d_hard_dev;           // [max_batch * K/8]

    void launch_kernels(int batch_size, cudaStream_t stream,
                        const int16_t* soft_in = nullptr,
                        uint8_t* hard_out = nullptr,
                        int* result_out = nullptr);
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_LDPC_DECODE_KERNEL_CUH */
