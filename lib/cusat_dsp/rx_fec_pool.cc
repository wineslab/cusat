/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rx_fec_pool.h"
#include <algorithm>
#include <stdexcept>

namespace cusat_dsp {

const std::vector<int> rx_fec_pool::s_empty_sizes;

rx_fec_pool::rx_fec_pool(fecframe_size_t framesize,
                   const std::vector<uint8_t>& enabled_modcods,
                   int max_trials,
                   int cuda_max_batch,
                   int n_gpu_streams,
                   int micro_batch)
{
    // One decode-block budget shared by every chain: the in-flight cap must
    // bound TOTAL GPU decode residency across modcods, not per-chain.
    auto budget = gpu_decode_budget::make_default();

    for (uint8_t mc_idx : enabled_modcods) {
        const modcod_def_t* def = find_modcod(mc_idx);
        if (!def)
            throw std::invalid_argument("Unknown modcod_idx: " + std::to_string(mc_idx));

        chain_entry entry;
        entry.chain = std::make_unique<rx_fec_chain>(
            framesize, def->code_rate, def->constellation,
            max_trials, cuda_max_batch,
            n_gpu_streams, micro_batch);
        entry.chain->set_decode_budget(budget);
        entry.modcod_idx = mc_idx;

        int out_bytes = entry.chain->max_output_bytes();
        if (out_bytes > d_max_output_bytes)
            d_max_output_bytes = out_bytes;

        d_chains[mc_idx] = std::move(entry);
    }
}

bool rx_fec_pool::submit(const complexf* symbols, int n_symbols, const pl_layout& pls)
{
    uint8_t mc = pls.modcod;
    auto it = d_chains.find(mc);
    if (it == d_chains.end())
        return false; // Unknown MODCOD, skip

    // A MODCOD boundary is also an ordering boundary.  Decoder chains own
    // independent CUDA streams; merely launching the old partial batch lets
    // the new chain complete first.  Worse, the pool exposes one chain's GSE
    // output at a time, so two chains completing in the same poll can overwrite
    // the handoff and silently lose otherwise-valid packets.  Finish the old
    // chain before admitting the first frame of the new run.  Transitions are
    // rare (and already require PLS confirmation), while the event wait is in
    // the FEC thread and no longer holds the process-wide CUDA launch mutex.
    bool output_ready = false;
    if (d_has_prev_submit && mc != d_last_submitted_modcod) {
        auto prev = d_chains.find(d_last_submitted_modcod);
        if (prev != d_chains.end()) {
            const bool previous_completed =
                prev->second.chain->pending_count() > 0 ||
                prev->second.chain->gpu_inflight() > 0 ||
                prev->second.chain->gpu_pending_launch() > 0;
            if (previous_completed) {
                prev->second.chain->flush();
                d_last_decoded_modcod = d_last_submitted_modcod;
                output_ready |= harvest_output(d_last_submitted_modcod);
            }
        }
    }
    d_last_submitted_modcod = mc;
    d_has_prev_submit = true;

    bool batch_ready = it->second.chain->submit_xfecframe(symbols, n_symbols);
    if (batch_ready) {
        d_last_decoded_modcod = mc;
        output_ready |= harvest_output(mc);
    }
    return output_ready || !d_ready_outputs.empty();
}

int rx_fec_pool::get_gse_output(uint8_t* out, int max_bytes)
{
    d_last_output_sizes.clear();
    if (d_ready_outputs.empty())
        return 0;

    ready_output_t block = std::move(d_ready_outputs.front());
    d_ready_outputs.pop_front();
    const int to_copy = std::min<int>(block.bytes.size(), max_bytes);
    if (to_copy > 0)
        std::copy_n(block.bytes.data(), to_copy, out);
    d_last_decoded_modcod = block.modcod;
    d_last_output_sizes = std::move(block.dfl_sizes);

    // Keep the potentially large allocation for the next decoder batch.
    block.bytes.clear();
    block.dfl_sizes.clear();
    d_free_outputs.emplace_back(std::move(block));
    return to_copy;
}

const std::vector<int>& rx_fec_pool::gse_dfl_sizes() const
{
    return d_last_output_sizes;
}

bool rx_fec_pool::harvest_output(uint8_t modcod)
{
    auto it = d_chains.find(modcod);
    if (it == d_chains.end())
        return false;

    ready_output_t block;
    if (!d_free_outputs.empty()) {
        block = std::move(d_free_outputs.back());
        d_free_outputs.pop_back();
    }
    block.bytes.resize(d_max_output_bytes);
    const int bytes = it->second.chain->get_gse_output(
        block.bytes.data(), static_cast<int>(block.bytes.size()));
    if (bytes <= 0) {
        block.bytes.clear();
        block.dfl_sizes.clear();
        d_free_outputs.emplace_back(std::move(block));
        return false;
    }
    block.bytes.resize(bytes);
    block.dfl_sizes = it->second.chain->gse_dfl_sizes();
    block.modcod = modcod;
    d_ready_outputs.emplace_back(std::move(block));
    return true;
}

void rx_fec_pool::clear_ready_outputs()
{
    while (!d_ready_outputs.empty()) {
        ready_output_t block = std::move(d_ready_outputs.front());
        d_ready_outputs.pop_front();
        block.bytes.clear();
        block.dfl_sizes.clear();
        d_free_outputs.emplace_back(std::move(block));
    }
    d_last_output_sizes.clear();
}

void rx_fec_pool::reset_all()
{
    for (auto& [mc_idx, entry] : d_chains) {
        entry.chain->reset();
    }
    d_last_decoded_modcod = 0;
    d_last_submitted_modcod = 0;
    d_has_prev_submit = false;
    clear_ready_outputs();
    d_flush_order.clear();
    d_flush_chain_idx = 0;
}

void rx_fec_pool::warmup_all()
{
    for (auto& kv : d_chains)
        kv.second.chain->warmup();
    d_last_decoded_modcod = 0;
    d_last_submitted_modcod = 0;
    d_has_prev_submit = false;
    clear_ready_outputs();
    d_flush_order.clear();
    d_flush_chain_idx = 0;
}

bool rx_fec_pool::launch_stale(std::chrono::milliseconds max_age)
{
    auto now = std::chrono::steady_clock::now();
    bool launched = false;
    for (auto& [mc_idx, entry] : d_chains) {
        if (entry.chain->pending_count() > 0) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - entry.chain->batch_start_time());
            if (age >= max_age) {
                entry.chain->launch_partial();
                launched = true;
            }
        }
    }
    return launched;
}

bool rx_fec_pool::drain_all_completed()
{
    bool any_output = false;
    for (auto& [mc_idx, entry] : d_chains) {
        if (entry.chain->drain_completed()) {
            d_last_decoded_modcod = mc_idx;
            any_output |= harvest_output(mc_idx);
        }
    }
    return any_output || !d_ready_outputs.empty();
}

void rx_fec_pool::finish_all_pending()
{
    for (auto& [mc_idx, entry] : d_chains) {
        entry.chain->flush();
        harvest_output(mc_idx);
    }
}

void rx_fec_pool::set_diag_fec_bench(bool on)
{
    for (auto& [mc_idx, entry] : d_chains)
        entry.chain->set_diag_fec_bench(on);
}

void rx_fec_pool::set_diag_llr_histogram(bool on)
{
    for (auto& [mc_idx, entry] : d_chains)
        entry.chain->set_diag_llr_histogram(on);
}

void rx_fec_pool::set_diag_ldpc_convergence(bool on)
{
    for (auto& [mc_idx, entry] : d_chains)
        entry.chain->set_diag_ldpc_convergence(on);
}

void rx_fec_pool::set_cuda_mutex(std::mutex* mtx)
{
    for (auto& [mc_idx, entry] : d_chains) {
        entry.chain->set_cuda_mutex(mtx);
    }
}

void rx_fec_pool::update_snr(uint8_t modcod, float snr_db)
{
    auto it = d_chains.find(modcod);
    if (it != d_chains.end())
        it->second.chain->update_snr(snr_db);
}

uint64_t rx_fec_pool::total_frame_count() const
{
    uint64_t total = 0;
    for (const auto& [mc_idx, entry] : d_chains) {
        total += entry.chain->frame_count();
    }
    return total;
}

uint64_t rx_fec_pool::total_error_count() const
{
    uint64_t total = 0;
    for (const auto& [mc_idx, entry] : d_chains) {
        total += entry.chain->frame_error_count();
    }
    return total;
}

bool rx_fec_pool::modcod_counters(uint8_t modcod,
                                  uint64_t& frames,
                                  uint64_t& errors) const
{
    auto it = d_chains.find(modcod);
    if (it == d_chains.end())
        return false;
    frames = it->second.chain->frame_count();
    errors = it->second.chain->frame_error_count();
    return true;
}

uint64_t rx_fec_pool::total_gpu_drop_count() const
{
    uint64_t total = 0;
    for (const auto& [mc_idx, entry] : d_chains) {
        total += entry.chain->gpu_drop_count();
    }
    return total;
}

float rx_fec_pool::data_snr_db() const
{
    auto it = d_chains.find(d_last_decoded_modcod);
    if (it == d_chains.end())
        return 0.0f;
    return it->second.chain->data_snr_db();
}

} // namespace cusat_dsp
