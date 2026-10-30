/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * FEC decode chain: demap → deinterleave → LDPC → BCH → BB descramble → BB deheader.
 *
 * GPU path: immediate-submit flight ring with N_STREAMS CUDA streams.
 *           Each frame is launched to the GPU immediately on submit.
 *           Completion is tracked via cudaEvents; output is drained in order.
 */

#ifndef INCLUDED_CUSAT_DSP_RX_FEC_CHAIN_H
#define INCLUDED_CUSAT_DSP_RX_FEC_CHAIN_H

#include "bb_scrambler.h"
#include "bbframe_parser.h"
#include "fec/bch_code.h"
#include "modulation/constellation.h"
#include "fec/code_params.h"
#include "ldpc_decode.h"
#include <cusat_dsp/modcod_types.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
namespace cusat_dsp { namespace cuda {
class demapper_cuda;
class ldpc_decode_cuda;
class bch_check_cuda;
}}

namespace cusat_dsp {

/**
 * Global budget of LDPC decode blocks allowed on the GPU at once.
 *
 * Each decoding codeword occupies one SM-resident thread block for the whole
 * decode; capping the total in-flight blocks at ~1 block/SM leaves per-SM
 * register/smem slack where the latency-critical timing-recovery and TX
 * kernels execute immediately instead of queueing behind block retirement
 * (measured: timing p99 245 us vs 5 ms uncapped under worst-case load, for
 * ~14% peak decode throughput). Shared across all chains of an ACM pool so
 * the cap bounds total GPU residency, not per-modcod residency.
 */
struct gpu_decode_budget {
    std::atomic<int> in_flight{0}; // decode blocks (frames) launched, not drained
    int cap = 0;                   // 0 = unlimited

    /** Default cap: 1 block/SM (device SM count), env override
     *  CUSAT_LDPC_INFLIGHT_CAP (0 disables the cap). */
    static std::shared_ptr<gpu_decode_budget> make_default();
};

class rx_fec_chain
{
public:
    rx_fec_chain(fecframe_size_t framesize,
              code_rate_t rate,
              modulation_t constellation,
              int max_trials = 25,
              int cuda_max_batch = 64,
              int n_gpu_streams = 8,
              int micro_batch = 4);

    ~rx_fec_chain();

    /**
     * @brief Submit one XFECFRAME of symbols to the chain.
     *
     * Launches demap+LDPC on the GPU immediately.
     * Returns true if any previously-submitted frames completed and output is
     * available (call get_gse_output to retrieve).
     *
     * @param symbols Input data symbols (xfecframe_len complex values).
     * @param n_symbols Number of symbols (must equal xfecframe_len).
     * @return true if decoded output is available.
     */
    bool submit_xfecframe(const complexf* symbols, int n_symbols);

    /**
     * @brief Retrieve decoded GSE data field after submit returns true.
     *
     * Returns the raw BBFRAME data fields (one per decoded frame in the batch).
     *
     * @param out Output buffer.
     * @param max_bytes Maximum bytes that can be written.
     * @return Number of GSE data bytes written.
     */
    int get_gse_output(uint8_t* out, int max_bytes);

    /** Per-frame data field sizes (valid after get_gse_output, GSE mode only) */
    const std::vector<int>& gse_dfl_sizes() const { return d_gse_dfl_sizes; }

    /**
     * @brief Reset all state after lock loss.
     *
     * Clears partial batch / flight ring, resets SNR bootstrap, and zeroes
     * counters so the chain can re-acquire cleanly after relock.
     */
    void reset();

    /**
     * @brief Flush any in-flight GPU work.
     *
     * Blocks until all in-flight slots complete.
     * After flush, get_gse_output() returns the remaining data.
     */
    void flush();

    /**
     * @brief Non-blocking drain of completed GPU work.
     *
     * Polls completed flight slots, runs post-processing (BCH → BB → output).
     *
     * @return true if any output was produced (call get_ts/gse_output).
     */
    bool drain_completed();

    // Per-stage timing (GPU path, accumulated over all submit_gpu calls)
    // Stats
    float snr_db() const { return d_snr_db; }
    uint64_t frame_count() const { return d_frame_cnt; }
    uint64_t frame_error_count() const { return d_frame_error_cnt; }
    int batch_size() const { return d_batch_size; }
    /** Maximum bytes that get_gse_output can return per batch */
    int max_output_bytes() const { return d_kbch_bytes * d_batch_size; }
    float data_snr_db() const { return d_snr_db; }

    /** Diagnostic flags (all write to /tmp/, OFF by default). */
    void set_diag_fec_bench(bool on) { d_diag_fec_bench = on; }
    void set_diag_llr_histogram(bool on) { d_diag_llr_histogram = on; }
    void set_diag_ldpc_convergence(bool on) { d_diag_ldpc_convergence = on; }

    /** When the flight ring is full, wait for the oldest decode instead of
     *  dropping the frame. For offline/pipe processing, where there is no
     *  real-time deadline and a drop would only hide decoder throughput. */
    void set_block_when_full(bool on) { d_block_when_full = on; }

    /** Update demapper noise variance / LLR gain from external pilot SNR estimate. */
    void update_snr(float snr_db) {
        if (snr_db > 0.0f) {
            d_pilot_snr_db = snr_db;
            float snr_lin = std::pow(10.0f, snr_db / 10.0f);
            d_snr_db = snr_db;
            d_noise_var = 1.0f / snr_lin;
            d_llr_gain = std::min(4.0f / d_noise_var, 1000.0f);
        }
    }

    float precision() const { return d_llr_gain; }

    /** Number of frames accumulated in current (incomplete) micro-batch */
    int pending_count() const { return d_batch_idx; }

    /** Launch partial micro-batch without blocking (non-blocking flush). */
    void launch_partial();
    /** Exercise the CUDA decode path before real-time streaming begins. */
    void warmup();

    /** Time when the first frame in the current batch was submitted */
    std::chrono::steady_clock::time_point batch_start_time() const
    {
        return d_batch_start_time;
    }

    uint64_t gpu_drop_count() const { return d_gpu_drop_cnt; }

    /** Number of frames currently in-flight on GPU (launched, not drained) */
    int gpu_inflight() const { return d_launch_idx - d_drain_idx; }
    /** Micro-batches ready but deferred by the decode budget */
    int gpu_pending_launch() const { return d_submit_idx - d_launch_idx; }
    uint64_t force_drain_count() const { return d_force_drain_cnt; }

    /** Set shared CUDA mutex (serializes CUDA API calls across threads). */
    void set_cuda_mutex(std::mutex* mtx) { d_cuda_mtx = mtx; }

    /** Share a decode-block budget (ACM pool: one budget across all chains).
     *  The chain creates its own default budget if none is set. */
    void set_decode_budget(std::shared_ptr<gpu_decode_budget> b)
    {
        if (b) d_budget = std::move(b);
    }

    /** A frame is usable only if LDPC converged and BCH validated it. */
    static bool frame_passes_fec_checks(int ldpc_trials_remaining, bool bch_valid)
    {
        return ldpc_trials_remaining >= 0 && bch_valid;
    }

private:
    // Processing stages
    modulation_t d_modulation;
    constellation_t d_constellation; // for decision-directed SNR estimates
    ldpc_decode d_ldpc;
    bch_code_t d_bch;

    // FEC parameters
    unsigned int d_kbch_bytes;
    unsigned int d_nbch_bytes;
    unsigned int d_fecframe_len;
    unsigned int d_xfecframe_len;

    // Batch accumulation
    int d_batch_size;      // = N_FLIGHT * MICRO_BATCH
    int d_batch_idx;       // frames accumulated in current micro-batch

    // Post-LDPC per-frame buffers
    std::vector<uint8_t> d_bch_out;
    std::vector<uint8_t> d_descrambled;

    // Output buffers
    std::vector<uint8_t> d_output;
    int d_output_len;
    std::vector<int> d_gse_dfl_sizes;

    // SNR estimation
    float d_snr_db;
    float d_pilot_snr_db = 0.0f;
    float d_noise_var;
    float d_llr_gain;

    // Stats
    uint64_t d_frame_cnt;
    uint64_t d_frame_error_cnt;
    int d_snr_skip_count = 0;   // periodic SNR estimate (every N frames in GPU path)

    // Diagnostic flags (all OFF by default, write to /tmp/)
    bool d_diag_fec_bench = false;
    bool d_diag_llr_histogram = false;
    bool d_diag_ldpc_convergence = false;
    bool d_block_when_full = false;

    // Batch timing (for flush timeout)
    std::chrono::steady_clock::time_point d_batch_start_time;

    // Per-stage FEC timing

    // ---- GPU immediate-submit flight ring ----
    static constexpr int N_FLIGHT = 128;
    int d_micro_batch;     // frames per GPU launch (configurable)
    int d_n_streams = 8;   // configurable via constructor

    struct flight_slot {
        complexf* syms = nullptr;       // [d_micro_batch * xfecframe_len] pinned host
        uint8_t* hard = nullptr;        // [d_micro_batch * nbch_bytes] managed
        int* trials = nullptr;          // [d_micro_batch] managed
        int* bch_valid = nullptr;       // [d_micro_batch] managed
        cudaEvent_t event = nullptr;    // signaled when GPU work done
        bool in_flight = false;
        int stream_idx = -1;
        int n_frames = 0;               // actual frames in this micro-batch
    };

    struct stream_resources {
        cuda::demapper_cuda* demapper = nullptr;
        cuda::ldpc_decode_cuda* ldpc = nullptr;
        cuda::bch_check_cuda* bch_check = nullptr;
        cudaStream_t stream = nullptr;
    };

    std::mutex* d_cuda_mtx = nullptr;      // shared mutex for CUDA API serialization

    flight_slot* d_flight = nullptr;     // [N_FLIGHT]
    stream_resources* d_streams = nullptr; // [d_n_streams]
    int d_submit_idx = 0;    // next slot to accumulate into (monotonic)
    int d_launch_idx = 0;    // next ready slot to launch (monotonic,
                             // d_drain_idx <= d_launch_idx <= d_submit_idx)
    int d_drain_idx = 0;     // oldest in-flight slot to drain (monotonic)
    int d_next_stream = 0;   // round-robin stream assignment
    uint64_t d_force_drain_cnt = 0; // times force_drain_oldest was called
    uint64_t d_gpu_drop_cnt = 0;    // frames dropped when GPU can't keep up
    std::shared_ptr<gpu_decode_budget> d_budget; // global decode-block cap

    // Per-instance FEC bench diagnostics.
    cudaEvent_t d_fec_bench_ev_start = nullptr;
    cudaEvent_t d_fec_bench_ev_demap = nullptr;
    cudaEvent_t d_fec_bench_ev_ldpc = nullptr;
    cudaEvent_t d_fec_bench_ev_bch = nullptr;
    FILE* d_fec_bench_fp = nullptr;
    int d_fec_bench_cnt = 0;

    // GPU submit/drain helpers
    bool submit_gpu(const complexf* symbols, int n_symbols);
    void launch_slot_gpu(flight_slot& slot, int n_frames);
    void pump_launches(bool bypass_cap = false);
    void release_budget(int n_frames);
    void force_drain_oldest();
    void post_process_slot(flight_slot& slot);
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_RX_FEC_CHAIN_H */
