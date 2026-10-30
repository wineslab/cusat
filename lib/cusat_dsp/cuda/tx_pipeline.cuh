/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused TX pipeline: mapper → PL frame+scramble → RRC on a single CUDA stream.
 *
 * Eliminates intermediate host round-trips between GPU stages.
 * On Grace Hopper (unified memory), FECFRAME input and IQ output use
 * managed memory — no H2D/D2H copies needed. All intermediate data stays on device.
 */

#ifndef INCLUDED_CUSAT_DSP_TX_PIPELINE_CUH
#define INCLUDED_CUSAT_DSP_TX_PIPELINE_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <map>

namespace cusat_dsp {
namespace cuda {

/**
 * Per-MODCOD configuration for the fused pipeline.
 */
struct tx_pipeline_modcod_config {
    float* d_constellation_dev;  // 2^n_mod × 2 floats on device
    unsigned int col_offsets[8];
    int stride;
    int n_mod;
    unsigned int fecframe_len;   // total FECFRAME bits
    unsigned int fecframe_bytes_padded; // per-frame stride in batch buffer (≥ fecframe_len/8)
    unsigned int xfecframe_len;  // symbols = fecframe_len / n_mod
    unsigned int payload_len;    // n_slots * 90 + n_pilots * 36
    unsigned int plframe_len;    // pl::kHeaderLen + payload_len
    int n_pilots;
    bool has_pilots;
};

class tx_pipeline_cuda
{
public:
    /**
     * @brief Construct the fused TX pipeline.
     *
     * @param sps          Samples per symbol.
     * @param sub_taps     Number of taps per polyphase sub-filter.
     * @param history_len  Number of history symbols (sub_taps - 1).
     * @param interleaved_taps  Host pointer to interleaved+reversed RRC taps.
     * @param descrambling_seq  Host pointer to PL descrambling sequence (pl::max_payload_symbols()).
     * @param max_payload_len   Maximum PL payload length across all MODCODs.
     * @param max_plframe_len   Maximum PLFRAME length (PLHEADER + payload).
     */
    tx_pipeline_cuda(int sps, int sub_taps, int history_len,
                     const float* interleaved_taps,
                     const float* descrambling_seq,
                     int max_payload_len, int max_plframe_len);
    ~tx_pipeline_cuda();

    /**
     * @brief Register a MODCOD configuration.
     *
     * Must be called before process() for each MODCOD that will be used.
     */
    void register_modcod(int modcod_idx,
                         const float* constellation_lut_host, int n_mod,
                         unsigned int fecframe_len, unsigned int xfecframe_len,
                         const unsigned int* col_offsets, int stride,
                         unsigned int payload_len, unsigned int plframe_len,
                         int n_pilots, bool has_pilots);

    /**
     * @brief Process one frame from a device-resident FECFRAME.
     *
     * Reads the FECFRAME from device memory (no H2D) and uses the caller's
     * CUDA stream (no internal sync). The caller
     * must synchronize the stream after this returns.
     *
     * @param modcod_idx        MODCOD index (must be registered).
     * @param fecframe_device   Device pointer to packed FECFRAME (N/8 bytes).
     * @param plheader          Host pointer to 90 PLHEADER symbols (complexf).
     * @param iq_out            Host pointer for output IQ samples.
     * @param stream            CUDA stream to use.
     * @return Number of output IQ samples.
     */
    unsigned int process_device(int modcod_idx,
                                const uint8_t* fecframe_device,
                                const float* plheader,
                                float* iq_out,
                                cudaStream_t stream);

    /** Pulse-shape an already constructed PLFRAME (used for dummy frames). */
    unsigned int process_symbols(const float* symbols_host,
                                 unsigned int n_symbols,
                                 cudaStream_t stream);

    /**
     * @brief Process N frames through the fused pipeline in a single batch.
     *
     * All N frames must use the same MODCOD (homogeneous batch).
     * FECFRAMEs are contiguous on device (from batched LDPC encoder).
     * PLHEADERs are contiguous on host (N × 90 × 2 floats).
     * Output IQ samples are contiguous on host.
     *
     * Uses the caller's CUDA stream and does not synchronize.
     *
     * @param modcod_idx        MODCOD index (must be registered).
     * @param fecframes_device  Device pointer to N × fecframe_bytes contiguous.
     * @param plheaders_host    Host pointer to N × 90 × 2 floats contiguous.
     * @param iq_out_host       Host pointer for output (N × plframe_len × sps × 2 floats).
     * @param batch_size        Number of frames.
     * @param stream            CUDA stream to use.
     * @return Total number of IQ samples (N × plframe_len × sps).
     */
    unsigned int process_batch(int modcod_idx,
                                const uint8_t* fecframes_device,
                                const float* plheaders_host,
                                float* iq_out_host,
                                int batch_size,
                                cudaStream_t stream);

    /** Expose the pipeline's CUDA stream (for shared-stream operation). */
    cudaStream_t stream() const { return d_stream; }

    /** Clear streaming RRC history after an untimed warm-up launch. */
    void reset_state();

    /** Expose managed FECFRAME buffer (for LDPC → pipeline zero-copy path). */
    uint8_t* fecframe_device() { return d_fecframe_dev; }

    /** Managed IQ output pointer (valid after stream sync). Single-frame path. */
    float* iq_managed() const { return d_iq_dev; }

    /** Managed IQ output pointer (valid after stream sync). Batch path. */
    float* batch_iq_managed() const { return d_batch_iq_dev; }

private:
    void ensure_batch_pipeline_buffers(int batch_size, unsigned int plframe_len,
                                       unsigned int xfecframe_len);

    cudaStream_t d_stream;
    int d_sps, d_sub_taps, d_history_len;

    // Device buffers (sized for max across MODCODs, single-frame)
    uint8_t* d_fecframe_dev;      // packed FECFRAME input
    float* d_xfecframe_dev;       // mapper output (xfecframe symbols)
    float* d_symbol_buf_dev;      // [history | PLHEADER | scrambled_payload] for RRC
    float* d_iq_dev;              // RRC output IQ samples

    // Batch pipeline buffers (allocated on demand)
    float* d_batch_xfecframe_dev = nullptr;   // N × xfecframe_len × 2
    float* d_batch_symbol_buf_dev = nullptr;  // (history + N × plframe_len) × 2
    float* d_batch_iq_dev = nullptr;          // N × plframe_len × sps × 2
    float* d_batch_plheader_dev = nullptr;    // N × 90 × 2
    int d_batch_pipeline_capacity = 0;
    unsigned int d_batch_plframe_len = 0;     // plframe_len used for last alloc

    // Constant data on device
    float* d_scramble_seq_dev;    // PL descrambling sequence
    float* d_rrc_taps_dev;        // interleaved+reversed RRC taps

    // Max sizes
    unsigned int d_max_fecframe_bytes;
    unsigned int d_max_xfecframe_len;
    unsigned int d_max_plframe_len;
    unsigned int d_max_iq_len;

    // Per-MODCOD configs
    std::map<int, tx_pipeline_modcod_config> d_configs;

};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_TX_PIPELINE_CUH */
