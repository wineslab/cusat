/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TX benchmark mode for dvbs2-uhd-trx: measure TX throughput without
 * UHD hardware.
 */

#include "trx/bench_tx.h"

#include "trx/bench_stats.h"

#include "acm_config.h"
#include "fec/code_params.h"
#include "gse.h"
#include "ldpc_decode.h"
#include "pl_framer.h"
#include "framing/plframe.h"
#include "framing/plheader.h"
#include "transmitter_frontend.h"
#include "tx_fec_chain.h"
#include <cusat_dsp/modcod_types.h>

#include "cuda/cuda_util.cuh"
#include "cuda/tx_pipeline.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace cusat_dsp;

int run_benchmark_tx(const trx_options& opt)
{
    const modcod_def_t* mc_bm_p = find_modcod_by_name(opt.modcod_str.c_str());
    if (!mc_bm_p) {
        fprintf(stderr, "Error: unknown modcod '%s'\n", opt.modcod_str.c_str());
        return 1;
    }
    const modcod_def_t& mc_bm = *mc_bm_p;
    fecframe_size_t fs_bm = opt.short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL;

    // Create TX pipeline objects
    tx_fec_chain fec(fs_bm, mc_bm.code_rate,
                     mc_bm.constellation);
    fec.set_rolloff(opt.rolloff);
    pl_framer framer((uint8_t)mc_bm.modcod_idx,
                     fs_bm == FECFRAME_SHORT, opt.pilots, opt.gold_code);
    transmitter_frontend frontend(opt.sps, opt.rolloff, 10);

    unsigned int plf_len = framer.plframe_len();
    unsigned int dfl = fec.max_dfl_bytes();
    double sym_rate = opt.rate / opt.sps;
    double frame_period_us = (double)plf_len / sym_rate * 1e6;

    // Compute batch size
    int batch = opt.tx_batch_size;
    if (batch == 0) {
        batch = std::max(1, (int)std::ceil(500.0 / frame_period_us));
        batch = std::min(batch, 64);
    }
    if (batch < 1) batch = 1;

    fprintf(stderr,
        "\n=== TX Benchmark ===\n"
        "  modcod:       %s (idx=%d)\n"
        "  framesize:    %s\n"
        "  sym_rate:     %.3f Msym/s\n"
        "  sps:          %d\n"
        "  plframe_len:  %u symbols\n"
        "  frame_period: %.1f µs\n"
        "  batch_size:   %d\n"
        "  batches:      %d\n"
        "  CUDA:         %s\n"
        "  CPU BCH:      %s\n"
        "====================\n\n",
        opt.modcod_str.c_str(), mc_bm.modcod_idx,
        opt.short_frames ? "SHORT" : "NORMAL",
        sym_rate / 1e6, opt.sps,
        plf_len, frame_period_us, batch, opt.benchmark_batches,
        "ON",
        (batch >= GPU_BCH_MIN_BATCH ? "no (batch >= threshold)" :
         "yes (batch < threshold)"));

    // Allocate buffers
    std::vector<uint8_t> gse_data(dfl, 0);
    std::vector<complexf> xfecframe(fec.xfecframe_len());
    std::vector<complexf> plframe(plf_len);
    std::vector<complexf> iq_out((size_t)batch * plf_len * opt.sps +
                                    frontend.ntaps() * 2);

    // Fill synthetic GSE data (padding)
    gse_encapsulator::write_padding(gse_data.data(), dfl);

    std::unique_ptr<cuda::tx_pipeline_cuda> bm_pipeline;
    uint8_t* bm_fecframes_dev = nullptr;
    uint8_t* bm_bch_out_dev = nullptr;
    std::vector<uint8_t> bm_bbframe_staging;
    std::vector<float> bm_plh_staging;
    std::unique_ptr<cuda::bch_bb_scramble_cuda> bm_bch_cuda;
    unsigned int n_bytes = fec.fecframe_bits() / 8;
    bool bm_gpu_bch = false;

    {
        unsigned int max_payload = plf_len - 90;
        bm_pipeline = std::make_unique<cuda::tx_pipeline_cuda>(
            frontend.sps(), frontend.sub_taps(), frontend.history_len(),
            frontend.interleaved_taps(),
            reinterpret_cast<const float*>(framer.descrambling_seq()),
            max_payload, plf_len);

        // Register MODCOD
        const mapper& m = fec.get_mapper();
        std::vector<float> lut = m.build_constellation_lut();
        unsigned int col_offsets[8];
        int stride;
        m.get_interleave_params(col_offsets, stride);
        const pl_layout pi_bm = pl_layout_for(mc_bm.modcod_idx, fs_bm, opt.pilots);
        bm_pipeline->register_modcod(
            mc_bm.modcod_idx, lut.data(), m.n_mod(),
            m.fecframe_len(), m.xfecframe_len(),
            col_offsets, stride,
            pi_bm.payload_symbols(), pi_bm.plframe_symbols(),
            pi_bm.pilot_block_count(), pi_bm.pilots);

        if (batch > 1) {
            bm_plh_staging.resize((size_t)batch * 90 * 2);
            /* Extra 12 bytes at end: 128APSK mapper reads ~12 bytes
             * past the last FECFRAME (padding symbols). */
            CUDA_CHECK(cudaMalloc(&bm_fecframes_dev,
                                  (size_t)batch * n_bytes + 12));
            CUDA_CHECK(cudaMemset(bm_fecframes_dev, 0,
                                  (size_t)batch * n_bytes + 12));
            bm_gpu_bch = (batch >= GPU_BCH_MIN_BATCH);
            if (!bm_gpu_bch) {
                // CPU BCH: staging holds BCH output (nbch bytes)
                bm_bbframe_staging.resize((size_t)batch * fec.bch_len_bytes());
            } else {
                // GPU BCH: staging holds raw BBFRAMEs (kbch bytes)
                bm_bbframe_staging.resize((size_t)batch * fec.kbch_bytes());
                CUDA_CHECK(cudaMalloc(&bm_bch_out_dev,
                                      (size_t)batch * fec.bch_len_bytes()));
                bm_bch_cuda = make_bch_cuda(fec);
            }
        }
    }

    // Optional concurrent LDPC decode load (worst-case: random LLRs never
    // converge, every codeword burns all trials — same as the legacy load in
    // bench-timing-recovery). Models the RX FEC contending with the TX chain.
    std::atomic<bool> load_stop{false};
    std::atomic<bool> load_ready{false};
    std::atomic<long> load_frames{0};
    std::thread loader;
    if (opt.bench_ldpc_load > 0) {
        loader = std::thread([&] {
            const int lb = opt.bench_ldpc_load;
            ldpc_decode ldpc(FECFRAME_NORMAL, RATE_3_5, 25, lb);
            const int N = ldpc.n_bits();
            std::vector<int16_t> soft((size_t)N * lb);
            std::vector<uint8_t> hard((size_t)ldpc.k_bytes() * lb);
            std::vector<int> trials(lb);
            std::mt19937 r2(7);
            for (auto& s : soft) s = (int16_t)((int)(r2() % 21) - 10);
            load_ready = true;
            while (!load_stop.load(std::memory_order_relaxed)) {
                ldpc.decode_batch_cuda(soft.data(), hard.data(),
                                       trials.data(), lb);
                load_frames += lb;
            }
        });
        while (!load_ready.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    // Warmup: 10 batches
    for (int w = 0; w < 10; w++) {
        if (batch > 1 && bm_pipeline && bm_gpu_bch) {
            // GPU BCH path
            for (int bi = 0; bi < batch; bi++) {
                fec.process_gse_to_bbframe(gse_data.data(), dfl);
                memcpy(bm_bbframe_staging.data() + bi * fec.kbch_bytes(),
                       fec.bbframe(), fec.kbch_bytes());
                memcpy(bm_plh_staging.data() + bi * 90 * 2,
                       reinterpret_cast<const float*>(framer.plheader()),
                       90 * 2 * sizeof(float));
            }
            cudaStream_t s = bm_pipeline->stream();
            bm_bch_cuda->encode_batch(
                bm_bbframe_staging.data(), bm_bch_out_dev, batch, s);
            fec.encode_ldpc_batch_d2d(
                bm_bch_out_dev, bm_fecframes_dev, batch, s);
            bm_pipeline->process_batch(
                mc_bm.modcod_idx, bm_fecframes_dev,
                bm_plh_staging.data(),
                reinterpret_cast<float*>(iq_out.data()), batch, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else if (batch > 1 && bm_pipeline && !bm_gpu_bch) {
            // CPU BCH path (batched)
            for (int bi = 0; bi < batch; bi++) {
                fec.process_gse_to_bch(gse_data.data(), dfl);
                memcpy(bm_bbframe_staging.data() + bi * fec.bch_len_bytes(),
                       fec.bch_output(), fec.bch_len_bytes());
                memcpy(bm_plh_staging.data() + bi * 90 * 2,
                       reinterpret_cast<const float*>(framer.plheader()),
                       90 * 2 * sizeof(float));
            }
            cudaStream_t s = bm_pipeline->stream();
            fec.encode_ldpc_batch_to_device(
                bm_bbframe_staging.data(), bm_fecframes_dev, batch, s);
            bm_pipeline->process_batch(
                mc_bm.modcod_idx, bm_fecframes_dev,
                bm_plh_staging.data(),
                reinterpret_cast<float*>(iq_out.data()), batch, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else {
            fec.process_gse_to_bch(gse_data.data(), dfl);
            cudaStream_t s = bm_pipeline->stream();
            fec.encode_ldpc_to_device(bm_pipeline->fecframe_device(), s);
            bm_pipeline->process_device(
                mc_bm.modcod_idx, bm_pipeline->fecframe_device(),
                reinterpret_cast<const float*>(framer.plheader()),
                reinterpret_cast<float*>(iq_out.data()), s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
    }

    // Benchmark: measure per-batch timing
    double total_cpu_us = 0, total_gpu_us = 0, total_us = 0;
    int total_frames = 0;
    std::vector<double> gpu_us_samples;
    gpu_us_samples.reserve(opt.benchmark_batches);
    auto bench_t0 = std::chrono::steady_clock::now();
    // Count only load frames decoded inside the measurement window —
    // the thread has been running since before warmup.
    const long load_frames_at_start = load_frames.load();

    for (int iter = 0; iter < opt.benchmark_batches; iter++) {
        auto t0 = std::chrono::steady_clock::now();
        auto t_cpu_end = t0;
        int iq_len = 0;

        if (batch > 1 && bm_pipeline && bm_gpu_bch) {
            // GPU BCH path
            for (int bi = 0; bi < batch; bi++) {
                fec.process_gse_to_bbframe(gse_data.data(), dfl);
                memcpy(bm_bbframe_staging.data() + bi * fec.kbch_bytes(),
                       fec.bbframe(), fec.kbch_bytes());
                memcpy(bm_plh_staging.data() + bi * 90 * 2,
                       reinterpret_cast<const float*>(framer.plheader()),
                       90 * 2 * sizeof(float));
            }
            t_cpu_end = std::chrono::steady_clock::now();
            cudaStream_t s = bm_pipeline->stream();
            bm_bch_cuda->encode_batch(
                bm_bbframe_staging.data(), bm_bch_out_dev, batch, s);
            fec.encode_ldpc_batch_d2d(
                bm_bch_out_dev, bm_fecframes_dev, batch, s);
            iq_len = bm_pipeline->process_batch(
                mc_bm.modcod_idx, bm_fecframes_dev,
                bm_plh_staging.data(),
                reinterpret_cast<float*>(iq_out.data()), batch, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else if (batch > 1 && bm_pipeline && !bm_gpu_bch) {
            // CPU BCH path (batched)
            for (int bi = 0; bi < batch; bi++) {
                fec.process_gse_to_bch(gse_data.data(), dfl);
                memcpy(bm_bbframe_staging.data() + bi * fec.bch_len_bytes(),
                       fec.bch_output(), fec.bch_len_bytes());
                memcpy(bm_plh_staging.data() + bi * 90 * 2,
                       reinterpret_cast<const float*>(framer.plheader()),
                       90 * 2 * sizeof(float));
            }
            t_cpu_end = std::chrono::steady_clock::now();
            cudaStream_t s = bm_pipeline->stream();
            fec.encode_ldpc_batch_to_device(
                bm_bbframe_staging.data(), bm_fecframes_dev, batch, s);
            iq_len = bm_pipeline->process_batch(
                mc_bm.modcod_idx, bm_fecframes_dev,
                bm_plh_staging.data(),
                reinterpret_cast<float*>(iq_out.data()), batch, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else {
            fec.process_gse_to_bch(gse_data.data(), dfl);
            t_cpu_end = std::chrono::steady_clock::now();
            cudaStream_t s = bm_pipeline->stream();
            fec.encode_ldpc_to_device(bm_pipeline->fecframe_device(), s);
            iq_len = bm_pipeline->process_device(
                mc_bm.modcod_idx, bm_pipeline->fecframe_device(),
                reinterpret_cast<const float*>(framer.plheader()),
                reinterpret_cast<float*>(iq_out.data()), s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        }

        auto t_end = std::chrono::steady_clock::now();
        double cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_cpu_end - t0).count();
        double gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_cpu_end).count();
        double batch_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t0).count();

        total_cpu_us += cpu_us;
        total_gpu_us += gpu_us;
        total_us += batch_us;
        total_frames += batch;
        gpu_us_samples.push_back(gpu_us);
        (void)iq_len;
    }
    double bench_wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - bench_t0).count();
    const long load_frames_measured = load_frames.load() - load_frames_at_start;

    if (loader.joinable()) {
        load_stop = true;
        loader.join();
    }

    double avg_batch_us = total_us / opt.benchmark_batches;
    double avg_frame_us = total_us / total_frames;
    double avg_cpu_us = total_cpu_us / opt.benchmark_batches;
    double avg_gpu_us = total_gpu_us / opt.benchmark_batches;
    double throughput_sym_s = (double)total_frames * plf_len / (total_us * 1e-6);
    double target_sym_s = sym_rate;
    double utilization = throughput_sym_s / target_sym_s * 100.0;

    fprintf(stderr,
        "\n=== TX Benchmark Results ===\n"
        "  Per-batch (avg over %d batches):\n"
        "    CPU (%s):      %.1f µs (%.1f µs/frame)\n"
        "    GPU (%s): %.1f µs (%.1f µs/frame)\n"
        "    Total:             %.1f µs (%.1f µs/frame)\n"
        "\n"
        "  Throughput:  %.2f Msym/s\n"
        "  Target:      %.2f Msym/s\n"
        "  Utilization: %.0f%%  %s\n"
        "===============================\n",
        opt.benchmark_batches,
        bm_gpu_bch ? "GSE+BB" : "GSE+BB+BCH",
        avg_cpu_us, avg_cpu_us / batch,
        bm_gpu_bch ? "BCH+LDPC+map+PL+RRC" : "LDPC+map+PL+RRC",
        avg_gpu_us, avg_gpu_us / batch,
        avg_batch_us, avg_frame_us,
        throughput_sym_s / 1e6,
        target_sym_s / 1e6,
        utilization,
        utilization >= 100.0 ? "OK" : "** CANNOT KEEP UP **");

    {
        std::vector<double> s = gpu_us_samples;
        std::sort(s.begin(), s.end());
        auto pct = [&](double p) {
            return s[std::min(s.size() - 1, (size_t)(p * s.size()))];
        };
        fprintf(stderr,
                "  GPU us: p50=%.0f p95=%.0f p99=%.0f max=%.0f",
                pct(0.50), pct(0.95), pct(0.99), s.back());
        if (opt.bench_ldpc_load > 0)
            fprintf(stderr, "; ldpc load: %.0f fr/s",
                    (double)load_frames_measured / bench_wall_s);
        fprintf(stderr, "\n");
    }
    if (!opt.bench_dump_samples.empty()) {
        FILE* f = fopen(opt.bench_dump_samples.c_str(), "w");
        if (f) {
            for (double v : gpu_us_samples)
                fprintf(f, "%.1f\n", v);
            fclose(f);
        }
    }

    if (bm_fecframes_dev)
        cudaFree(bm_fecframes_dev);
    if (bm_bch_out_dev)
        cudaFree(bm_bch_out_dev);
    return 0;
}
