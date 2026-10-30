/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rx_fec_chain.h"
#include <cmath>
#include <cstring>

#include "cuda/bch_check_kernel.cuh"
#include "cuda/cuda_util.cuh"
#include "cuda/demapper_kernel.cuh"
#include "cuda/ldpc_decode_kernel.cuh"
#include "ldpc_table_factory.h"
#include "modulation/dd_snr.h"

namespace cusat_dsp {

std::shared_ptr<gpu_decode_budget> gpu_decode_budget::make_default()
{
    auto b = std::make_shared<gpu_decode_budget>();
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
        cudaSuccess)
        sms = 0;
    b->cap = sms; // 1 decode block per SM
    if (const char* e = getenv("CUSAT_LDPC_INFLIGHT_CAP"))
        b->cap = atoi(e);
    static bool printed = false;
    if (!printed) {
        printed = true;
        fprintf(stderr, "gpu_decode_budget: in-flight decode cap = %d blocks%s\n",
                b->cap, b->cap == 0 ? " (unlimited)" : "");
    }
    return b;
}

rx_fec_chain::rx_fec_chain(fecframe_size_t framesize,
                     code_rate_t rate,
                     modulation_t constellation,
                     int max_trials,
                     int cuda_max_batch,
                     int n_gpu_streams,
                     int micro_batch)
    : d_modulation(constellation),
      d_constellation(make_constellation(constellation, rate, framesize)),
      d_ldpc(framesize, rate, max_trials, cuda_max_batch),
      d_batch_idx(0),
      d_output_len(0),
      d_snr_db(0),
      d_noise_var(1.0f),
      d_llr_gain(4.0f),
      d_frame_cnt(0),
      d_frame_error_cnt(0),
      d_micro_batch(micro_batch)
{
    // Get FEC parameters
    const fec_params_t fec = get_fec_params(framesize, rate);
    d_kbch_bytes = fec.k_bch / 8;
    d_nbch_bytes = fec.n_bch / 8;
    d_fecframe_len = fecframe_bits(framesize);
    const int bits = d_constellation.bits_per_sym;
    // 128APSK: 6 stuffing bits before interleaving, 12 padding symbols after
    d_xfecframe_len = constellation == MOD_128APSK ? (d_fecframe_len + 6) / bits + 12
                                                   : d_fecframe_len / bits;

    d_bch = make_bch_code(framesize, fec.n_bch, fec.t_bch);

    // GPU micro-batch path: flight ring + per-stream resources
    d_n_streams = n_gpu_streams;
    d_batch_size = N_FLIGHT * d_micro_batch; // max_output_bytes uses this

    d_flight = new flight_slot[N_FLIGHT];
    for (int i = 0; i < N_FLIGHT; i++) {
        cudaMallocHost((void**)&d_flight[i].syms,
                       d_micro_batch * d_xfecframe_len * sizeof(complexf));
        // Pinned host memory — async copies read symbols from stable flight-slot
        // storage, and GPU writes FEC outputs that CPU reads after event sync.
        // Using cudaMallocHost instead of cudaMallocManaged avoids illegal
        // memory access errors on GH200 caused by managed-memory page fault
        // storms in ACM mode (2600+ allocs).
        cudaMallocHost((void**)&d_flight[i].hard,
                       d_micro_batch * d_ldpc.k_bytes());
        cudaMallocHost((void**)&d_flight[i].trials,
                       d_micro_batch * sizeof(int));
        cudaMallocHost((void**)&d_flight[i].bch_valid,
                       d_micro_batch * sizeof(int));
        cudaEventCreateWithFlags(&d_flight[i].event,
                                 cudaEventDisableTiming);
    }

    // BCH tables for the GPU check: 256 x 4 words and the generator
    const uint64_t* flat_lut = d_bch.encoder_lut.data()->data();
    const uint64_t* gen_poly = d_bch.generator.data();
    const int gen_degree = d_bch.parity_bits;

    // Create per-stream GPU resources
    // Each stream needs its own demapper and LDPC decoder because they
    // have internal device buffers that can't be shared across streams.
    // BCH check shares the same LUT (read-only) but needs per-stream
    // result buffer.
    ldpc_code* ldpc_table = create_ldpc_table(framesize, rate);
    d_streams = new stream_resources[d_n_streams];
    int lo_prio, hi_prio;
    cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio);
    for (int s = 0; s < d_n_streams; s++) {
        cudaStreamCreateWithPriority(&d_streams[s].stream,
                                     cudaStreamNonBlocking, lo_prio);
        d_streams[s].demapper = new cuda::demapper_cuda(
            constellation, rate, framesize, d_micro_batch);
        d_streams[s].ldpc = new cuda::ldpc_decode_cuda(
            ldpc_table, rate, max_trials, d_micro_batch);
        d_streams[s].bch_check = new cuda::bch_check_cuda(
            flat_lut, gen_poly, gen_degree, d_nbch_bytes);
    }
    delete ldpc_table;

    fprintf(stderr, "rx_fec_chain: GPU mode: %d streams, %d flight slots, "
            "micro_batch=%d\n", d_n_streams, N_FLIGHT, d_micro_batch);

    // Per-chain default budget; an ACM pool replaces it with one shared
    // budget via set_decode_budget() so the cap spans all modcod chains.
    d_budget = gpu_decode_budget::make_default();

    // Per-frame buffers
    d_bch_out.resize(d_kbch_bytes);
    d_descrambled.resize(d_kbch_bytes);

    // Output buffer: sized for max possible output
    d_output.resize(d_kbch_bytes * d_batch_size);
}

rx_fec_chain::~rx_fec_chain()
{
    if (d_flight) {
        for (int i = 0; i < N_FLIGHT; i++) {
            if (d_flight[i].syms) cudaFreeHost(d_flight[i].syms);
            if (d_flight[i].hard) cudaFreeHost(d_flight[i].hard);
            if (d_flight[i].trials) cudaFreeHost(d_flight[i].trials);
            if (d_flight[i].bch_valid) cudaFreeHost(d_flight[i].bch_valid);
            if (d_flight[i].event) cudaEventDestroy(d_flight[i].event);
        }
        delete[] d_flight;
    }
    if (d_streams) {
        for (int s = 0; s < d_n_streams; s++) {
            delete d_streams[s].demapper;
            delete d_streams[s].ldpc;
            delete d_streams[s].bch_check;
            if (d_streams[s].stream)
                cudaStreamDestroy(d_streams[s].stream);
        }
        delete[] d_streams;
    }
    if (d_fec_bench_ev_start) cudaEventDestroy(d_fec_bench_ev_start);
    if (d_fec_bench_ev_demap) cudaEventDestroy(d_fec_bench_ev_demap);
    if (d_fec_bench_ev_ldpc) cudaEventDestroy(d_fec_bench_ev_ldpc);
    if (d_fec_bench_ev_bch) cudaEventDestroy(d_fec_bench_ev_bch);
    if (d_fec_bench_fp) fclose(d_fec_bench_fp);
}

void rx_fec_chain::reset()
{
    {
        std::unique_lock<std::mutex> cuda_lock;
        if (d_cuda_mtx)
            cuda_lock = std::unique_lock<std::mutex>(*d_cuda_mtx);
        while (d_drain_idx < d_submit_idx) {
            auto& slot = d_flight[d_drain_idx % N_FLIGHT];
            if (slot.in_flight) {
                CUDA_CHECK(cudaEventSynchronize(slot.event));
                release_budget(slot.n_frames);
            }
            slot.in_flight = false;
            slot.stream_idx = -1;
            slot.n_frames = 0;
            d_drain_idx++;
        }
    }

    d_submit_idx = 0;
    d_launch_idx = 0;
    d_drain_idx = 0;
    d_next_stream = 0;
    d_batch_idx = 0;
    d_output_len = 0;
    d_snr_db = 0;
    d_noise_var = 1.0f;
    d_llr_gain = 4.0f;
    d_frame_cnt = 0;
    d_frame_error_cnt = 0;
    d_gse_dfl_sizes.clear();
}

bool rx_fec_chain::submit_xfecframe(const complexf* symbols, int n_symbols)
{
    return submit_gpu(symbols, n_symbols);
}

void rx_fec_chain::launch_partial()
{
    if (d_batch_idx > 0) {
        auto& slot = d_flight[d_submit_idx % N_FLIGHT];
        slot.n_frames = d_batch_idx;
        d_submit_idx++;
        d_batch_idx = 0;
    }
    pump_launches();
}

void rx_fec_chain::warmup()
{
    // The first demapper/LDPC/BCH launch can spend several milliseconds in
    // lazy CUDA setup and unified-memory placement. If that happens after RF
    // streaming starts, the shared CUDA launch gate also blocks TX generation
    // long enough to underflow a low-latency IQ queue. Prime one partial batch
    // now, when no real-time threads exist, and leave counters/state clean.
    std::vector<complexf> symbols(d_xfecframe_len, complexf(0.0f, 0.0f));
    submit_xfecframe(symbols.data(), (int)symbols.size());
    launch_partial();
    flush();
    reset();
    d_force_drain_cnt = 0;
    d_gpu_drop_cnt = 0;
}

void rx_fec_chain::flush()
{
    // Launch partial micro-batch if any, then drain everything
    d_output_len = 0;
    d_gse_dfl_sizes.clear();
    launch_partial();
    while (d_drain_idx < d_submit_idx)
        force_drain_oldest();
}

bool rx_fec_chain::drain_completed()
{
    // Start fresh — previous output was consumed by get_gse_output
    d_output_len = 0;
    d_gse_dfl_sizes.clear();

    bool any_output = false;
    while (d_drain_idx < d_launch_idx) {
        auto& slot = d_flight[d_drain_idx % N_FLIGHT];
        if (!slot.in_flight) {
            d_drain_idx++;
            continue;
        }

        // cudaEventQuery is thread-safe (CUDA 11+) — no mutex needed.
        if (cudaEventQuery(slot.event) != cudaSuccess)
            break; // not done yet — preserve order

        post_process_slot(slot);
        slot.in_flight = false;
        release_budget(slot.n_frames);
        d_drain_idx++;
        any_output = true;
    }
    // Budget may have freed — launch any deferred micro-batches.
    pump_launches();
    return any_output;
}

int rx_fec_chain::get_gse_output(uint8_t* out, int max_bytes)
{
    int to_copy = std::min(d_output_len, max_bytes);
    memcpy(out, d_output.data(), to_copy);
    d_output_len = 0;
    return to_copy;
}

// =====================================================================
// GPU immediate-submit path
// =====================================================================

bool rx_fec_chain::submit_gpu(const complexf* symbols, int n_symbols)
{
    using clk = std::chrono::steady_clock;

    // 1. Drain completed GPU work at batch boundaries and when the flight
    //    ring is getting full. cudaEventQuery is expensive on ARM64/GH200
    //    (~0.5ms), so calling it on every frame would saturate the CPU at
    //    high frame rates (3100 fps × 0.5ms = 155% CPU).
    bool output_ready = false;
    bool should_drain = (d_batch_idx == 0) ||
                        ((d_submit_idx - d_drain_idx) >= N_FLIGHT / 2);
    if (should_drain) {
        d_output_len = 0;
        d_gse_dfl_sizes.clear();
        output_ready = drain_completed();
    }

    // Offline mode: make room by waiting for the oldest decode.
    while (d_block_when_full && (d_submit_idx - d_drain_idx) >= N_FLIGHT - 2) {
        force_drain_oldest();
        output_ready = true;
    }
    // If flight ring is nearly full, drop this frame instead of blocking.
    if ((d_submit_idx - d_drain_idx) >= N_FLIGHT - 2) {
        d_gpu_drop_cnt++;
        d_frame_cnt++;
        d_frame_error_cnt++;
        return output_ready;
    }
    // 3. Accumulate frame into current micro-batch slot
    auto& slot = d_flight[d_submit_idx % N_FLIGHT];
    if ((unsigned int)n_symbols != d_xfecframe_len) {
        d_frame_cnt++;
        d_frame_error_cnt++;
        d_gpu_drop_cnt++;
        return output_ready;
    }
    complexf* dest = slot.syms + d_batch_idx * d_xfecframe_len;
    memcpy(dest, symbols, n_symbols * sizeof(complexf));

    // Per-frame SNR estimate: for high-order constellations (128APSK+),
    // the data-based hard-decision SNR estimate is unreliable because small
    // residual phase errors cause wrong hard decisions in densely-packed
    // constellations. Use pilot SNR passed from the frontend instead.
    if (++d_snr_skip_count >= 32) {
        d_snr_skip_count = 0;
        float snr_lin;
        if (d_constellation.bits_per_sym >= 7 && d_pilot_snr_db > 5.0f) {
            // Use pilot SNR for 128APSK+ (more reliable than data estimate)
            snr_lin = std::pow(10.0f, d_pilot_snr_db / 10.0f);
        } else {
            // QPSK decisions are sign tests: use every symbol
            snr_lin = d_modulation == MOD_QPSK
                          ? qpsk_snr_dd(dest, n_symbols)
                          : constellation_snr_dd(d_constellation, dest, n_symbols, 8);
        }
        if (snr_lin > 1.0f) {
            d_snr_db = 10.0f * std::log10(snr_lin);
            d_noise_var = 1.0f / snr_lin;
            d_llr_gain = std::min(4.0f / d_noise_var, 1000.0f);
        }
    }
    if (d_batch_idx == 0)
        d_batch_start_time = clk::now();
    d_batch_idx++;

    // 4. Micro-batch full: mark the slot ready and try to launch. The
    // launch is deferred (slot queues in the flight ring) whenever the
    // decode-block budget is exhausted; drain_completed() re-pumps as
    // budget frees.
    if (d_batch_idx >= d_micro_batch) {
        slot.n_frames = d_batch_idx;
        d_submit_idx++;
        d_batch_idx = 0;
        pump_launches();
    }

    return output_ready;
}

void rx_fec_chain::launch_slot_gpu(flight_slot& slot, int n_frames)
{
    using clk = std::chrono::steady_clock;
    int s = d_next_stream++ % d_n_streams;
    cudaStream_t stream = d_streams[s].stream;

        {
            auto mtx_t0 = clk::now();
            std::unique_lock<std::mutex> cuda_lock;
            if (d_cuda_mtx)
                cuda_lock = std::unique_lock<std::mutex>(*d_cuda_mtx);
            auto t5 = clk::now();
            {
                auto mtx_dt = std::chrono::duration<double, std::milli>(t5 - mtx_t0).count();
                if (mtx_dt > 5.0)
                    fprintf(stderr, "dvbs2-rx: cuda_mtx WAIT in fec submit: %.1f ms\n",
                            mtx_dt);
            }

            if (d_diag_fec_bench) {
                if (!d_fec_bench_ev_start) {
                    CUDA_CHECK(cudaEventCreate(&d_fec_bench_ev_start));
                    CUDA_CHECK(cudaEventCreate(&d_fec_bench_ev_demap));
                    CUDA_CHECK(cudaEventCreate(&d_fec_bench_ev_ldpc));
                    CUDA_CHECK(cudaEventCreate(&d_fec_bench_ev_bch));
                    char path[128];
                    snprintf(path, sizeof(path), "/tmp/fec_bench_%p.log", (void*)this);
                    d_fec_bench_fp = fopen(path, "w");
                }
                CUDA_CHECK(cudaEventRecord(d_fec_bench_ev_start, stream));
            }

            /* NOTE (int8 opt-in): the int8 decode path clamps channel LLRs
             * to +-31 on load, but the live demapper gain (LLR gain = 4 / noise variance)
             * reaches ~1000 at OTA SNR and saturates it (+40% FER measured
             * on 16apsk OTA), while a fixed gain cap breaks dense
             * constellations (32apsk loopback FER 1.4e-2 at cap 32). int8
             * needs adaptive LLR normalization in the demapper before it
             * can be the default. */
            d_streams[s].demapper->demap_batch_async(
                reinterpret_cast<const float*>(slot.syms),
                n_frames, d_noise_var, d_llr_gain, stream);

            if (d_diag_fec_bench) CUDA_CHECK(cudaEventRecord(d_fec_bench_ev_demap, stream));

            // One-shot LLR diagnostic: dump histogram to /tmp/llr_diag.log
            if (d_diag_llr_histogram) {
                static int llr_diag_done = 0;
                static int llr_diag_wait = 0;
                if (llr_diag_done < 1 && d_llr_gain > 10.0f && ++llr_diag_wait > 500) {
                    llr_diag_done++;
                    int N = d_streams[s].demapper->fecframe_len();
                    std::vector<int16_t> llr_host(N);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaMemcpy(llr_host.data(),
                                          d_streams[s].demapper->soft_out_dev(),
                                          N * sizeof(int16_t),
                                          cudaMemcpyDeviceToHost));
                    int hist[8] = {0};
                    int zeros = 0;
                    long sum_abs = 0;
                    for (int i = 0; i < N; i++) {
                        int v = llr_host[i];
                        int a = v < 0 ? -v : v;
                        sum_abs += a;
                        if (a == 0) zeros++;
                        else if (a <= 1) hist[0]++;
                        else if (a <= 5) hist[1]++;
                        else if (a <= 15) hist[2]++;
                        else if (a <= 31) hist[3]++;
                        else if (a <= 63) hist[4]++;
                        else if (a <= 127) hist[5]++;
                        else if (a <= 511) hist[6]++;
                        else hist[7]++;
                    }
                    // Fraction of negative LLRs (should be ~50%) and of weak
                    // LLRs (|llr| <= 5).
                    int total_neg = 0, total_weak = 0;
                    for (int i = 0; i < N; i++) {
                        if (llr_host[i] < 0) total_neg++;
                        int a = llr_host[i] < 0 ? -llr_host[i] : llr_host[i];
                        if (a <= 5) total_weak++;
                    }

                    FILE* fp = fopen("/tmp/llr_diag.log", "w");
                    if (fp) {
                        fprintf(fp, "N=%d prec=%.1f n_mod=%d\n"
                                "zeros=%d |1|=%d |2-5|=%d |6-15|=%d |16-31|=%d "
                                "|32-63|=%d |64-127|=%d |128-511|=%d |512+|=%d\n"
                                "mean|llr|=%.1f\n"
                                "negative=%d/%d (%.1f%%)  weak(|llr|<=5)=%d (%.1f%%)\n",
                                N, d_llr_gain, d_constellation.bits_per_sym,
                                zeros, hist[0], hist[1], hist[2], hist[3],
                                hist[4], hist[5], hist[6], hist[7],
                                (double)sum_abs / N,
                                total_neg, N, 100.0*total_neg/N,
                                total_weak, 100.0*total_weak/N);
                        fprintf(fp, "first200:");
                        for (int i = 0; i < 200 && i < N; i++)
                            fprintf(fp, " %d", (int)llr_host[i]);
                        fprintf(fp, "\n");
                        fclose(fp);
                    }
                }
            }

            // LDPC writes directly to slot's managed buffers — no D2H copy
            d_streams[s].ldpc->decode_launch_async(
                d_streams[s].demapper->soft_out_dev(), n_frames, stream,
                slot.hard, slot.trials);

            if (d_diag_fec_bench) CUDA_CHECK(cudaEventRecord(d_fec_bench_ev_ldpc, stream));

            // BCH check reads from slot.hard (where LDPC wrote) and
            // writes directly to slot's managed bch_valid buffer
            d_streams[s].bch_check->check_async(
                slot.hard, slot.bch_valid, n_frames, stream);

            if (d_diag_fec_bench) CUDA_CHECK(cudaEventRecord(d_fec_bench_ev_bch, stream));

            CUDA_CHECK(cudaEventRecord(slot.event, stream));

            // Log FEC stage timing for first 200 batches
            if (d_diag_fec_bench) {
                CUDA_CHECK(cudaEventSynchronize(d_fec_bench_ev_bch));
                float ms_demap = 0, ms_ldpc = 0, ms_bch = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms_demap, d_fec_bench_ev_start,
                                                d_fec_bench_ev_demap));
                CUDA_CHECK(cudaEventElapsedTime(&ms_ldpc, d_fec_bench_ev_demap,
                                                d_fec_bench_ev_ldpc));
                CUDA_CHECK(cudaEventElapsedTime(&ms_bch, d_fec_bench_ev_ldpc,
                                                d_fec_bench_ev_bch));
                if (d_fec_bench_fp) {
                    fprintf(d_fec_bench_fp, "batch=%d demap=%.3f ldpc=%.3f bch=%.3f total=%.3f ms\n",
                            n_frames, ms_demap, ms_ldpc, ms_bch,
                            ms_demap + ms_ldpc + ms_bch);
                    if (d_fec_bench_cnt % 20 == 0) fflush(d_fec_bench_fp);
                }
                d_fec_bench_cnt++;
            }
        }
    slot.in_flight = true;
    slot.stream_idx = s;
    slot.n_frames = n_frames;
}

void rx_fec_chain::pump_launches(bool bypass_cap)
{
    while (d_launch_idx < d_submit_idx) {
        auto& slot = d_flight[d_launch_idx % N_FLIGHT];
        if (!bypass_cap && d_budget && d_budget->cap > 0) {
            int cur = d_budget->in_flight.load(std::memory_order_relaxed);
            // Always admit when the GPU is empty of decode work so an
            // oversized micro-batch cannot deadlock the queue.
            if (cur > 0 && cur + slot.n_frames > d_budget->cap)
                break;
        }
        if (d_budget)
            d_budget->in_flight.fetch_add(slot.n_frames,
                                          std::memory_order_relaxed);
        launch_slot_gpu(slot, slot.n_frames);
        d_launch_idx++;
    }
}

void rx_fec_chain::release_budget(int n_frames)
{
    if (d_budget)
        d_budget->in_flight.fetch_sub(n_frames, std::memory_order_relaxed);
}

void rx_fec_chain::force_drain_oldest()
{
    // Oldest slot may still be deferred by the decode budget (e.g. flush
    // while other chains hold it). Launch it regardless — bypassing the cap
    // here cannot deadlock and force_drain synchronizes right after.
    if (d_drain_idx >= d_launch_idx && d_drain_idx < d_submit_idx)
        pump_launches(/*bypass_cap=*/true);

    auto& slot = d_flight[d_drain_idx % N_FLIGHT];
    if (!slot.in_flight) {
        d_drain_idx++;
        return;
    }
    d_force_drain_cnt++;
    {
        // Event waits are thread-safe and do not mutate launch state.  Do not
        // hold the process-wide CUDA launch mutex while waiting: on a MODCOD
        // ordering barrier that would also stop the latency-critical frontend
        // and TX kernels for the duration of an LDPC batch.
        cudaError_t peek = cudaPeekAtLastError();
        if (peek != cudaSuccess) {
            fprintf(stderr, "rx_fec_chain: CUDA sticky error before force_drain: "
                    "%s (%d)\n", cudaGetErrorString(peek), (int)peek);
            cudaGetLastError(); // clear
        }
        CUDA_CHECK(cudaEventSynchronize(slot.event)); // block until done
    }
    post_process_slot(slot);
    slot.in_flight = false;
    release_budget(slot.n_frames);
    d_drain_idx++;
}

void rx_fec_chain::post_process_slot(flight_slot& slot)
{
    // Log LDPC convergence: trials_remaining distribution to /tmp/ldpc_conv.log
    // trials>0 = converged (higher = fewer iters used), trials<0 = failed
    if (d_diag_ldpc_convergence) {
        static FILE* conv_fp = nullptr;
        static int conv_total = 0, conv_fail = 0;
        static int remaining_hist[32] = {0}; // index = trials_remaining (0-25+)
        if (!conv_fp) conv_fp = fopen("/tmp/ldpc_conv.log", "w");
        for (int i = 0; i < slot.n_frames; i++) {
            int t = slot.trials[i];
            conv_total++;
            if (t < 0) conv_fail++;
            else {
                int idx = (t > 30) ? 31 : t;
                remaining_hist[idx]++;
            }
        }
        if (conv_fp && conv_total > 0 && (conv_total % 5000 < slot.n_frames)) {
            fprintf(conv_fp, "n=%d fail=%d remaining_hist:", conv_total, conv_fail);
            for (int i = 0; i < 32; i++)
                if (remaining_hist[i] > 0) fprintf(conv_fp, " [%d]=%d", i, remaining_hist[i]);
            fprintf(conv_fp, "\n");
            fflush(conv_fp);
        }
    }

    // Per-frame forensics (CUSAT_FEC_FRAMELOG=<path>): one line per frame
    // in arrival order — frame index, LDPC trials remaining (<0 = LDPC
    // fail), BCH verdict. For offline failure-pattern analysis (alternating
    // vs bursty vs periodic) on captures.
    static FILE* flog = nullptr;
    static bool flog_checked = false;
    if (!flog_checked) {
        flog_checked = true;
        const char* p = getenv("CUSAT_FEC_FRAMELOG");
        if (p) flog = fopen(p, "w");
    }

    for (int i = 0; i < slot.n_frames; i++) {
        d_frame_cnt++;
        if (flog)
            fprintf(flog, "%lu %d %d\n", (unsigned long)d_frame_cnt,
                    slot.trials[i], (int)slot.bch_valid[i]);
        if (!frame_passes_fec_checks(slot.trials[i], slot.bch_valid[i])) {
            d_frame_error_cnt++;
            continue; // skip corrupted frame — don't feed garbage to GSE/TS
        }

        // BCH is systematic: first kbch_bytes of LDPC output = BBFRAME data.
        const uint8_t* bbframe_data = slot.hard + i * d_nbch_bytes;

        bb_scramble(bbframe_data, d_descrambled.data(), d_kbch_bytes);

        int out_bytes = extract_gse_datafield(
            d_descrambled.data(), d_kbch_bytes,
            d_output.data() + d_output_len,
            (int)d_output.size() - d_output_len);
        if (out_bytes > 0)
            d_gse_dfl_sizes.push_back(out_bytes);
        d_output_len += out_bytes;
    }
}

} // namespace cusat_dsp
