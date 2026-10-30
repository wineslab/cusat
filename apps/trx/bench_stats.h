/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark accumulators and GPU BCH helper for dvbs2-uhd-trx.
 */

#pragma once

#include "tx_fec_chain.h"
#include "cuda/bch_encode_kernel.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace cusat_dsp;

// ---------------------------------------------------------------------------
// Benchmark accumulator — tracks min/avg/max for each timing metric
// ---------------------------------------------------------------------------
struct bench_stat {
    double sum = 0;
    double min_val = 1e18;
    double max_val = 0;
    uint64_t count = 0;

    void add(double v) {
        sum += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        ++count;
    }
    double avg() const { return count ? sum / count : 0; }
};

// Return the number of events observed since the previous sample.  Some live
// counters (notably the post-bootstrap RX FEC counter) reset when the receiver
// reacquires.  Treat the new value as the post-reset delta instead of allowing
// unsigned subtraction to wrap to a value near UINT64_MAX.
inline uint64_t reset_aware_counter_delta(uint64_t current, uint64_t previous)
{
    return current >= previous ? current - previous : current;
}

struct bench_accum {
    // TX stages (ms per batch)
    bench_stat tx_loop, tx_gen, tx_gse, tx_bb_bch, tx_ldpc;
    bench_stat tx_gpu;
    bench_stat tx_ring_wait, tx_batch_wait;
    bench_stat tx_batch_size;  // frames per batch
    // RX frontend stages (ms)
    bench_stat rx_loop, fe_total, fe_agc, fe_timing, fe_sync;
    bench_stat tr_stage, tr_launch, tr_sync, tr_readback;
    bench_stat gpu_timing, gpu_interp;
    // RX FEC stages (ms)
    bench_stat fec_submit, fec_loop, gpu_drain, gpu_phase, gpu_submit_xfec;
    // RX ring (ms)
    bench_stat rx_ring;

    // GPU resource utilization (set at shutdown from actual counters)
    double bench_wall_sec = 0;   // wall time during accumulation
    uint64_t total_tx_frames = 0;
    uint64_t total_rx_frames = 0;
    uint64_t total_tx_syncs = 0;
    uint64_t total_rx_syncs = 0;

    void print_summary(FILE* f, double rate) const {
        double avg_batch = tx_batch_size.count ? tx_batch_size.avg() : 1.0;

        fprintf(f, "\n");
        fprintf(f, "╔═══════════════════════════════════════════════════════════════════════════╗\n");
        fprintf(f, "║                    BENCHMARK SUMMARY  (ms)                                ║\n");
        fprintf(f, "║  avg batch = %.1f frames                                                  ║\n", avg_batch);
        fprintf(f, "╠═══════════════════════════════════════════════════════════════════════════╣\n");
        fprintf(f, "║  Stage                      per-batch          per-frame        samples  ║\n");
        fprintf(f, "║                           avg    max         avg    max                   ║\n");
        fprintf(f, "╠═══════════════════════════════════════════════════════════════════════════╣\n");

        auto row = [&](const char* name, const bench_stat& s, bool per_frame) {
            if (s.count == 0) return;
            if (per_frame && avg_batch > 0) {
                fprintf(f, "║  %-22s %7.2f %7.2f     %7.3f %7.3f   %6lu  ║\n",
                        name, s.avg(), s.max_val,
                        s.avg() / avg_batch, s.max_val / avg_batch,
                        (unsigned long)s.count);
            } else {
                fprintf(f, "║  %-22s %7.2f %7.2f                          %6lu  ║\n",
                        name, s.avg(), s.max_val,
                        (unsigned long)s.count);
            }
        };

        fprintf(f, "║  --- TX PIPELINE ---                                                      ║\n");
        row("tx_loop (period)",    tx_loop,       false);
        row("  GSE encap",         tx_gse,        true);
        row("  BB hdr+scr+BCH",   tx_bb_bch,     true);
        row("  LDPC encode (GPU)", tx_ldpc,       true);
        row("  GPU total",        tx_gpu,        true);
        row("  ring wait",        tx_ring_wait,  false);
        row("  batch wait",       tx_batch_wait, false);
        row("  tx_gen total",     tx_gen,        true);

        fprintf(f, "║  --- RX FRONTEND ---                                                      ║\n");
        row("rx_loop (period)",    rx_loop,       false);
        row("  AGC + freq rot",   fe_agc,        false);
        row("  timing recovery",  fe_timing,     false);
        row("    stage memcpy",   tr_stage,      false);
        row("    kernel launch",  tr_launch,     false);
        row("    GPU sync",       tr_sync,       false);
        row("    readback",       tr_readback,   false);
        row("    GPU timing est", gpu_timing,    false);
        row("    GPU interp",     gpu_interp,    false);
        row("  frame sync",      fe_sync,       false);
        row("  frontend total",  fe_total,      false);
        row("  RX ring (ms)",    rx_ring,       false);

        fprintf(f, "║  --- RX FEC ---                                                           ║\n");
        row("fec_loop (period)",   fec_loop,      false);
        row("  GPU drain",        gpu_drain,     false);
        row("  phase correct",    gpu_phase,     false);
        row("  submit xfecframe", gpu_submit_xfec, false);
        row("  FEC submit total", fec_submit,    false);

        // GPU resource utilization section
        fprintf(f, "║  --- GPU RESOURCE UTILIZATION ---                                         ║\n");

        // GPU-ms and CPU-ms per frame (from per-batch averages / avg_batch)
        double gpu_tx_per_frame = 0, cpu_tx_per_frame = 0;
        double gpu_rx_per_frame = 0, cpu_rx_per_frame = 0;
        if (tx_gpu.count > 0 && avg_batch > 0) {
            gpu_tx_per_frame = (tx_gpu.avg() + tx_ldpc.avg()) / avg_batch;
            cpu_tx_per_frame = (tx_gse.avg() + tx_bb_bch.avg()) / avg_batch;
        }
        if (gpu_timing.count > 0) {
            gpu_rx_per_frame = gpu_timing.avg() + gpu_interp.avg() + fe_agc.avg() + gpu_submit_xfec.avg();
            cpu_rx_per_frame = fe_sync.avg() + gpu_drain.avg() + gpu_phase.avg();
        }

        // GPU duty cycle: estimated total GPU time / wall time
        double total_gpu_ms = gpu_tx_per_frame * total_tx_frames +
                              gpu_rx_per_frame * total_rx_frames;
        double duty_pct = (bench_wall_sec > 0) ? total_gpu_ms / (bench_wall_sec * 1000.0) * 100.0 : 0;
        fprintf(f, "║  GPU duty cycle          %5.1f%%                                          ║\n", duty_pct);

        if (total_tx_frames > 0) {
            fprintf(f, "║  GPU-ms per TX frame     %7.3f                                        ║\n", gpu_tx_per_frame);
            fprintf(f, "║  CPU-ms per TX frame     %7.3f  (GSE + BB/BCH)                        ║\n", cpu_tx_per_frame);
        }
        if (total_rx_frames > 0) {
            fprintf(f, "║  GPU-ms per RX frame     %7.3f                                        ║\n", gpu_rx_per_frame);
            fprintf(f, "║  CPU-ms per RX frame     %7.3f  (sync + FEC post)                     ║\n", cpu_rx_per_frame);
        }

        // Kernel launches per frame (deterministic from pipeline structure)
        fprintf(f, "║  Kernel launches/TX frm  ~8     (map+PL+RRC+LDPC+BCH)                   ║\n");
        fprintf(f, "║  Kernel launches/RX frm  ~11    (AGC+TR+SOF+demap+LDPC+BCH)             ║\n");

        // Sync points
        if (total_tx_frames > 0)
            fprintf(f, "║  TX syncs/frame          %5.2f   (%lu syncs / %lu frames)                ║\n",
                    (double)total_tx_syncs / total_tx_frames,
                    (unsigned long)total_tx_syncs, (unsigned long)total_tx_frames);
        if (total_rx_syncs > 0)
            fprintf(f, "║  RX syncs total          %-6lu  (1 per frontend process() call)         ║\n",
                    (unsigned long)total_rx_syncs);

        // System memory
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used_mb = (total_mem - free_mem) / (1024 * 1024);
        size_t total_mb = total_mem / (1024 * 1024);
        fprintf(f, "║  System memory           %zu / %zu MB                              ║\n", used_mb, total_mb);

        fprintf(f, "╚═══════════════════════════════════════════════════════════════════════════╝\n");
    }
};

/**
 * Create a GPU BCH+scramble encoder from a tx_fec_chain's BCH code and scrambler.
 */
inline std::unique_ptr<cuda::bch_bb_scramble_cuda>
make_bch_cuda(const tx_fec_chain& fec)
{
    const auto& bch = fec.get_bch();
    return std::make_unique<cuda::bch_bb_scramble_cuda>(
        bch.encoder_lut.data()->data(), bch.generator.data(), bch.parity_bits,
        bb_scrambler_sequence(), (int)(bch.k_bits / 8),
        bch.k_bits / 8, bch.n_bits / 8, bch.parity_bits / 8);
}

/* Minimum batch size to use GPU BCH. Below this, CPU BCH is faster due to
 * kernel launch overhead. Determined empirically on GH200: crossover ~16. */
inline constexpr int GPU_BCH_MIN_BATCH = 16;
