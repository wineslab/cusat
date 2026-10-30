/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * RX FEC pool: one rx_fec_chain per enabled MODCOD for ACM/VCM reception.
 * Frames are routed to the correct chain based on their PLS info.
 */

#ifndef INCLUDED_CUSAT_DSP_RX_FEC_POOL_H
#define INCLUDED_CUSAT_DSP_RX_FEC_POOL_H

#include "acm_config.h"
#include "rx_fec_chain.h"
#include "framing/plframe.h"
#include "framing/plheader.h"
#include <chrono>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cusat_dsp {

class rx_fec_pool
{
public:
    /**
     * @brief Construct an RX FEC pool with chains for each enabled MODCOD.
     *
     * @param framesize Normal or short FECFRAME.
     * @param enabled_modcods List of modcod_idx values to support.
     * @param max_trials Max LDPC iterations per chain.
     */
    rx_fec_pool(fecframe_size_t framesize,
             const std::vector<uint8_t>& enabled_modcods,
             int max_trials = 25,
             int cuda_max_batch = 64,
             int n_gpu_streams = 8,
             int micro_batch = 4);

    /**
     * @brief Submit a frame to the appropriate chain based on PLS info.
     *
     * @param symbols XFECFRAME symbols.
     * @param n_symbols Number of symbols.
     * @param pls PLS info (contains modcod).
     * @return true if a batch was decoded and output is available.
     */
    bool submit(const complexf* symbols, int n_symbols, const pl_layout& pls);

    /** Get the oldest queued GSE output block from any decoder chain. */
    int get_gse_output(uint8_t* out, int max_bytes);

    /** Per-frame data field sizes for the block returned by get_gse_output(). */
    const std::vector<int>& gse_dfl_sizes() const;

    /** Reset all chains after lock loss. */
    void reset_all();

    /** Exercise every enabled CUDA decoder before real-time streaming. */
    void warmup_all();

    /**
     * @brief Launch any partial chain whose oldest frame exceeds max_age.
     *
     * This is deliberately non-blocking: it closes the sparse-traffic latency
     * hole without synchronizing a CUDA stream from the real-time FEC thread.
     * Completed work is collected later by drain_all_completed(). Under
     * sustained traffic, micro-batches fill before this path is needed.
     *
     * @param max_age Maximum time to hold a partial micro-batch.
     * @return true if at least one partial micro-batch was launched.
     */
    bool launch_stale(std::chrono::milliseconds max_age);

    /** Check if a flushed output block remains to be consumed. */

    /**
     * @brief Non-blocking drain of completed GPU work across all chains.
     *
     * Polls each chain's flight ring for completed frames and runs post-
     * processing. Output is available via get_gse_output()
     * for whichever chain produced data.
     *
     * @return true if any chain produced output.
     */
    bool drain_all_completed();

    /**
     * @brief Block until all in-flight GPU work completes across all chains.
     *
     * Equivalent to calling flush() on every chain. After return, all output
     * is available.
     */
    void finish_all_pending();

    /** Set diagnostic flags on all chains (all OFF by default). */
    void set_diag_fec_bench(bool on);
    void set_diag_llr_histogram(bool on);
    void set_diag_ldpc_convergence(bool on);

    /** Set shared CUDA mutex on all chains (for fine-grained GPU locking). */
    void set_cuda_mutex(std::mutex* mtx);

    /** Update demapper SNR for a specific MODCOD chain. */
    void update_snr(uint8_t modcod, float snr_db);

    /** Max output bytes per batch (across all chains). */
    int max_output_bytes() const { return d_max_output_bytes; }

    // Aggregate stats
    uint64_t total_frame_count() const;
    uint64_t total_error_count() const;
    /** Read cumulative outcomes for one MODCOD chain. */
    bool modcod_counters(uint8_t modcod,
                         uint64_t& frames,
                         uint64_t& errors) const;
    uint64_t total_gpu_drop_count() const;
    float data_snr_db() const;

private:
    struct chain_entry {
        std::unique_ptr<rx_fec_chain> chain;
        uint8_t modcod_idx;
    };

    std::unordered_map<uint8_t, chain_entry> d_chains;
    uint8_t d_last_decoded_modcod = 0;
    uint8_t d_last_submitted_modcod = 0;
    bool d_has_prev_submit = false;
    int d_max_output_bytes = 0;

    // A decoder chain owns its post-FEC output only until its next poll:
    // rx_fec_chain::drain_completed() clears that buffer before collecting
    // newly completed work.  Copy every completed block into this pool-owned
    // FIFO immediately, so a near-simultaneous MODCOD transition cannot make
    // one chain overwrite/erase another chain's valid GSE payload.
    struct ready_output_t {
        std::vector<uint8_t> bytes;
        std::vector<int> dfl_sizes;
        uint8_t modcod = 0;
    };
    std::deque<ready_output_t> d_ready_outputs;
    std::vector<ready_output_t> d_free_outputs;
    std::vector<int> d_last_output_sizes;

    bool harvest_output(uint8_t modcod);
    void clear_ready_outputs();

    // Flush state
    std::vector<uint8_t> d_flush_order;
    int d_flush_chain_idx = 0;

    static const std::vector<int> s_empty_sizes;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_RX_FEC_POOL_H */
