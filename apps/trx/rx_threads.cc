/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * RX worker threads for dvbs2-uhd-trx: UHD receive, frontend DSP and FEC.
 */

#include "trx/rx_threads.h"

#include "trx/acm_signaling.h"
#include "trx/fec_trace_writer.h"
#include "trx/frontend_trace_writer.h"
#include "trx/thread_util.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <vector>

// ====================================================================
// Thread 3: RX UHD — rx_stream->recv() → RX ring
// ====================================================================
void rx_uhd_thread(trx_context& ctx)
{
    auto& opt = *ctx.opt;
    radio_device& radio = *ctx.radio;
    const double stream_start_time = *ctx.stream_start_time;
    auto& rx_ring = *ctx.rx_ring;
    auto& rx_frontend = *ctx.rx_frontend;
    auto& rx_impairment = *ctx.rx_impairment;
    auto& rx_uhd_overflows = ctx.stats.rx_uhd_overflows;
    auto& rx_ring_drops = ctx.stats.rx_ring_drops;
    auto& fe_stage_atomic = ctx.stats.fe_stage_atomic;

    if (!opt.do_rx) return;
    if (opt.pin_threads) pin_thread("rx_uhd", 2, opt.realtime_threads);

    const size_t samps_per_recv = radio.rx_max_samps();
    std::vector<complexf> buf(samps_per_recv);
    int consecutive_ring_drops = 0;

    // Optional IQ capture to file
    FILE* capture_fp = nullptr;
    size_t capture_remaining = 0;
    if (!opt.capture_iq_file.empty()) {
        capture_fp = fopen(opt.capture_iq_file.c_str(), "wb");
        if (!capture_fp) {
            fprintf(stderr, "dvbs2-uhd-trx: cannot open capture file: %s\n",
                    opt.capture_iq_file.c_str());
        } else {
            capture_remaining = size_t(opt.capture_duration * opt.rate);
            fprintf(stderr, "dvbs2-uhd-trx: capturing %.1fs (%.1f Msamples) to %s\n",
                    opt.capture_duration, capture_remaining / 1e6,
                    opt.capture_iq_file.c_str());
        }
    }

    // Start continuous streaming at the synchronized time
    radio.start_rx_continuous(stream_start_time);

    radio_rx_meta md;
    while (!g_stop.load(std::memory_order_relaxed)) {
        size_t n = radio.recv(buf.data(), samps_per_recv, md, 0.1);
        if (md.error == radio_rx_error::overflow) {
            const uint64_t count =
                rx_uhd_overflows.fetch_add(1, std::memory_order_relaxed) + 1;
            // Do not let a transient transport overflow turn into a permanent
            // receive failure through synchronous Docker logging. The exact
            // count remains available in the once-per-second UHD_OF metric.
            if (count <= 4 || count % 1000 == 0) {
                fprintf(stderr, "dvbs2-uhd-trx: UHD RX OVERFLOW #%lu\n",
                        (unsigned long)count);
            }
        }
        else if (md.error == radio_rx_error::other)
            fprintf(stderr, "dvbs2-uhd-trx: RX error (see radio plugin log)\n");
        if (n > 0) {
            // RX-side emulation lets a remote endpoint own the channel model,
            // keeping the measured Jetson free of emulator work.
            if (rx_impairment) rx_impairment->apply(buf.data(), n);
            // Write to capture file if active
            if (capture_fp && capture_remaining > 0) {
                size_t to_write = std::min(n, capture_remaining);
                fwrite(buf.data(), sizeof(complexf), to_write, capture_fp);
                capture_remaining -= to_write;
                if (capture_remaining == 0) {
                    fclose(capture_fp);
                    capture_fp = nullptr;
                    fprintf(stderr, "dvbs2-uhd-trx: IQ capture complete: %s\n",
                            opt.capture_iq_file.c_str());
                }
            }

            size_t written = rx_ring.write(buf.data(), n);
            if (written < n) {
                rx_ring_drops.fetch_add(n - written);
                consecutive_ring_drops++;
                if (consecutive_ring_drops <= 15 ||
                    consecutive_ring_drops % 10000 == 0) {
                    int sub = rx_frontend ? rx_frontend->process_substage.load(
                        std::memory_order_relaxed) : -1;
                    fprintf(stderr, "dvbs2-uhd-trx: RX ring DROP %zu samples "
                            "(ring full, streak=%d, fe_stage=%d, sub=%d)\n",
                            n - written, consecutive_ring_drops,
                            fe_stage_atomic.load(std::memory_order_relaxed), sub);
                }
                if (consecutive_ring_drops >= 10) {
                    int sub = rx_frontend ? rx_frontend->process_substage.load(
                        std::memory_order_relaxed) : -1;
                    fprintf(stderr, "dvbs2-uhd-trx: 10 consecutive ring drops "
                            "(fe_stage=%d, sub=%d) — exiting for diagnosis\n",
                            fe_stage_atomic.load(std::memory_order_relaxed), sub);
                    g_stop.store(true);
                    break;
                }
            } else {
                consecutive_ring_drops = 0;
            }
        }
    }

    if (capture_fp) fclose(capture_fp);

    // Stop streaming
    radio.stop_rx();
}

// ====================================================================
// Thread 4: RX frontend — RX ring → receiver_frontend → frame_queue
// ====================================================================
void rx_frontend_thread(trx_context& ctx)
{
    auto& opt = *ctx.opt;
    const pl_layout& frame_format = *ctx.frame_format;
    const int max_gpu_batch = ctx.max_gpu_batch;
    const double hw_sps = ctx.hw_sps;
    const int xfecframe_len = ctx.xfecframe_len;
    const auto start_time = ctx.start_time;
    auto& rx_ring = *ctx.rx_ring;
    auto& tx_ring = *ctx.tx_ring;
    auto& frame_queue = *ctx.frame_queue;
    auto& rx_frontend = *ctx.rx_frontend;
    auto& fec_single = *ctx.fec_single;
    auto& fec_multi = *ctx.fec_multi;
    auto& wf_tap = *ctx.wf_tap;
    auto& tx_impairment = *ctx.tx_impairment;
    auto& rx_impairment = *ctx.rx_impairment;
    auto& fe_stage_atomic = ctx.stats.fe_stage_atomic;
    auto& rx_uhd_overflows = ctx.stats.rx_uhd_overflows;
    auto& rx_ring_drops = ctx.stats.rx_ring_drops;
    auto& tx_underflows = ctx.stats.tx_underflows;
    auto& tx_sequence_errors = ctx.stats.tx_sequence_errors;
    auto& tx_time_errors = ctx.stats.tx_time_errors;
    auto& rx_fec_frames = ctx.stats.rx_fec_frames;
    auto& rx_fec_errors = ctx.stats.rx_fec_errors;
    auto& tx_frames = ctx.stats.tx_frames;
    auto& tx_dummy_frames = ctx.stats.tx_dummy_frames;
    auto& rx_pdus = ctx.stats.rx_pdus;
    auto& rx_fec_queue_drops = ctx.stats.rx_fec_queue_drops;
    auto& acm_frames_submitted = ctx.stats.acm_frames_submitted;
    auto& acm_frames_skipped_len = ctx.stats.acm_frames_skipped_len;
    auto& acm_frames_dummy = ctx.stats.acm_frames_dummy;
    auto& acm_mc_hist = ctx.stats.acm_mc_hist;
    auto& acm_pls_guard_corrected = ctx.stats.acm_pls_guard_corrected;
    auto& acm_pls_guard_deferred = ctx.stats.acm_pls_guard_deferred;
    auto& acm_pls_guard_transitions = ctx.stats.acm_pls_guard_transitions;
    auto& acm_pls_guard_outage_frames =
        ctx.stats.acm_pls_guard_outage_frames;
    auto& fec_bootstrapped = ctx.stats.fec_bootstrapped;
    auto& fec_reset_seq = ctx.stats.fec_reset_seq;
    auto& ewma_fer = ctx.stats.ewma_fer;
    auto& sync_count_rx = ctx.stats.sync_count_rx;
    auto& lat_rx_ring_samples = ctx.stats.lat_rx_ring_samples;
    auto& lat_frontend_us = ctx.stats.lat_frontend_us;
    auto& lat_rx_loop_us = ctx.stats.lat_rx_loop_us;
    auto& lat_fe_agc_us = ctx.stats.lat_fe_agc_us;
    auto& lat_fe_timing_us = ctx.stats.lat_fe_timing_us;
    auto& lat_fe_sync_us = ctx.stats.lat_fe_sync_us;
    auto& lat_fe_n_in = ctx.stats.lat_fe_n_in;
    auto& lat_fe_n_syms = ctx.stats.lat_fe_n_syms;
    auto& lat_fe_fused = ctx.stats.lat_fe_fused;
    auto& lat_tr_memcpy_us = ctx.stats.lat_tr_memcpy_us;
    auto& lat_tr_launch_us = ctx.stats.lat_tr_launch_us;
    auto& lat_tr_lnch_est_us = ctx.stats.lat_tr_lnch_est_us;
    auto& lat_tr_lnch_interp_us = ctx.stats.lat_tr_lnch_interp_us;
    auto& lat_tr_sync_us = ctx.stats.lat_tr_sync_us;
    auto& lat_tr_readback_us = ctx.stats.lat_tr_readback_us;
    auto& lat_gpu_timing_us = ctx.stats.lat_gpu_timing_us;
    auto& lat_gpu_interp_us = ctx.stats.lat_gpu_interp_us;
    auto& gpu_inflight_count = ctx.stats.gpu_inflight_count;

    if (!opt.do_rx) return;
    if (opt.pin_threads) pin_thread("rx_fe", 3, opt.realtime_threads);

    iq_source_ring source(rx_ring, g_stop);

    bool rx_was_locked = false;
    auto lock_acquired_time = std::chrono::steady_clock::now();
    auto last_unlock_time = std::chrono::steady_clock::now();
    uint64_t ordinary_frames_since_lock = 0;
    bool fer_watchdog_armed = false;
    auto fer_high_since = std::chrono::steady_clock::now();
    bool link_recovery_active = false;
    channel_impairment* link_impairment = rx_impairment
        ? rx_impairment.get() : tx_impairment.get();
    uint64_t link_up_epoch = link_impairment
        ? link_impairment->link_up_epoch() : 0;

    const int history = (int)rx_frontend->history();
    const int min_chunk = (int)(frame_format.plframe_symbols() * hw_sps) * 2 + 10000;
    const int chunk_size = std::max({50000, min_chunk,
        max_gpu_batch + history + (int)std::ceil(hw_sps) * 2 + 1000});
    const int in_buf_size = chunk_size + history * 2;

    // Input buffer holds hardware-rate samples straight from the RX ring.
    // Prefer pinned host memory so timing recovery can DMA it directly to HBM
    // without an extra pageable->pinned staging memcpy.
    complexf* in_buf_ptr = nullptr;
    bool in_buf_pinned = false;
    cudaError_t alloc_err = cudaMallocHost((void**)&in_buf_ptr,
                                           (size_t)in_buf_size * sizeof(complexf));
    if (alloc_err == cudaSuccess) {
        std::fill_n(in_buf_ptr, in_buf_size, complexf(0.0f, 0.0f));
        in_buf_pinned = true;
    } else {
        cudaGetLastError(); // clear cudaMallocHost failure for the RX thread
        in_buf_ptr = (complexf*)calloc(in_buf_size, sizeof(complexf));
        fprintf(stderr, "dvbs2-uhd-trx: RX input cudaMallocHost failed (%s), "
                "falling back to pageable staging\n",
                cudaGetErrorString(alloc_err));
    }
    rx_frontend->set_input_pinned(in_buf_pinned);
    // Use QPSK xfecframe_len (32400) as minimum — in pipeline mode
    // both previous and current batch symbols are output in one call,
    // which for shorter constellations (8PSK/16APSK) can exceed
    // xfecframe_len * 4 if sized only for the target modcod.
    const int sym_buf_xfec = std::max(xfecframe_len, 32400);
    std::vector<complexf> sym_buf(sym_buf_xfec * 4);

    frontend_trace_writer fe_trace;
    fe_trace.open(opt.frontend_trace_csv, "dvbs2-uhd-trx");

    // Minimum samples before calling frontend — amortizes CUDA sync overhead.
    // Default: 0.5ms at rates > 200MHz, 0 otherwise (no extra buffering).
    int rx_min_batch_samples = 0;
    if (opt.rx_min_batch_ms > 0) {
        rx_min_batch_samples = (int)(opt.rate * opt.rx_min_batch_ms / 1000.0);
    } else if (opt.rate > 200e6) {
        // At >200 MHz, default to 0.15ms to accumulate ~2 PLFRAMEs before
        // each frontend call, reducing the number of cudaStreamSynchronize
        // calls (~150µs each).
        rx_min_batch_samples = (int)(opt.rate * 0.15 / 1000.0);
    }
    // Cap to 60% of max_gpu_batch to avoid deadlock (threshold must be
    // reachable within the input buffer).
    if (rx_min_batch_samples > max_gpu_batch * 6 / 10)
        rx_min_batch_samples = max_gpu_batch * 6 / 10;
    if (rx_min_batch_samples > 0 && opt.log_enabled)
        fprintf(stderr, "dvbs2-uhd-trx: rx_min_batch=%d samples (%.2f ms)\n",
                rx_min_batch_samples, rx_min_batch_samples / opt.rate * 1000.0);

    int n_buffered = 0;
    long acq_burst_remaining = 0; // contiguous samples left in this
                                  // acquisition burst (see trim logic)
    uint64_t rx_fec_frames_at_last_check = 0;
    uint64_t rx_dummy_frames_at_last_check = 0;
    auto last_valid_frame_progress =
        std::chrono::steady_clock::now() - std::chrono::seconds(1);
    auto rx_loop_prev = std::chrono::steady_clock::now();
    auto fe_heartbeat = std::chrono::steady_clock::now();
    int fe_heartbeat_iter = 0;
    int fe_stage = 0;  // 0=top, 1=ring_read, 3=process, 4=post
    auto update_stage = [&](int s) {
        fe_stage = s;
        fe_stage_atomic.store(s, std::memory_order_relaxed);
    };
    while (!g_stop.load(std::memory_order_relaxed)) {
        update_stage(0);
        // LINK OFF/ON is a real stream discontinuity, not merely low SNR.
        // Reset all timing/carrier acquisition state on recovery so a false
        // lock learned from outage samples cannot consume several watchdog
        // periods.  This notification is explicit in the channel controller;
        // it does not infer RF reciprocity from measured SNR.
        if (link_impairment) {
            const uint64_t epoch = link_impairment->link_up_epoch();
            if (epoch != link_up_epoch) {
                link_up_epoch = epoch;
                rx_ring.skip(rx_ring.available_read());
                n_buffered = 0;
                acq_burst_remaining = 0;
                rx_frontend->force_reacquire();
                fec_reset_seq.fetch_add(1, std::memory_order_acq_rel);
                fec_bootstrapped.store(false, std::memory_order_relaxed);
                link_recovery_active = true;
                rx_was_locked = false;
                ordinary_frames_since_lock = 0;
                fer_watchdog_armed = false;
                lock_acquired_time = std::chrono::steady_clock::now();
                last_unlock_time = lock_acquired_time;
                last_valid_frame_progress =
                    lock_acquired_time - std::chrono::seconds(1);
                fprintf(stderr, "dvbs2-uhd-trx: link recovery — RX acquisition state reset\n");
            }
        }
        // Measure RX frontend loop period
        auto rx_loop_now = std::chrono::steady_clock::now();

        // Heartbeat: detect if frontend thread stalls
        fe_heartbeat_iter++;
        auto since_hb = std::chrono::duration<double>(rx_loop_now - fe_heartbeat).count();
        if (since_hb >= 0.5) {
            size_t hb_occ = rx_ring.available_read();
            if (since_hb >= 2.0) {
                int sub = rx_frontend->process_substage.load(std::memory_order_relaxed);
                cudaError_t cuda_err = cudaPeekAtLastError();
                fprintf(stderr, "dvbs2-rx: FE STALL DETECTED: %d iters in %.1fs, "
                        "ring=%.1fms, stage=%d, sub=%d, n_buf=%d, "
                        "cuda=%s(%d), locked=%d\n",
                        fe_heartbeat_iter, since_hb,
                        (double)hb_occ / opt.rate * 1000.0,
                        fe_stage, sub, n_buffered,
                        cudaGetErrorString(cuda_err), (int)cuda_err,
                        rx_frontend->locked() ? 1 : 0);
            } else {
                fprintf(stderr, "dvbs2-rx: FE heartbeat: %d iters in %.1fs, ring=%.1fms, "
                        "n_buf=%d, locked=%d, stage=%d\n",
                        fe_heartbeat_iter, since_hb,
                        (double)hb_occ / opt.rate * 1000.0,
                        n_buffered,
                        rx_frontend->locked() ? 1 : 0,
                        fe_stage);
            }
            fe_heartbeat = rx_loop_now;
            fe_heartbeat_iter = 0;
        }
        lat_rx_loop_us.store(
            std::chrono::duration_cast<std::chrono::microseconds>(
                rx_loop_now - rx_loop_prev).count(),
            std::memory_order_relaxed);
        rx_loop_prev = rx_loop_now;

        // Sample RX ring occupancy before reading
        size_t ring_occ = rx_ring.available_read();
        lat_rx_ring_samples.store(ring_occ, std::memory_order_relaxed);

        // Flush stale data from RX ring.
        // When locked: flush at 1.5s (LDPC death spiral protection).
        // When unlocked: flush aggressively. The frontend can't process
        // samples as fast as they arrive in searching mode (per-symbol
        // SOF correlation + Farrow overhead). Keep only a small window
        // for acquisition — processing more just makes the ring grow.
        // "Receiving" = locked AND actively decoding frames.
        // False lock on noise, or signal loss during soft-unlock holdoff,
        // both have is_locked=true but no FEC output — treat as unlocked
        // for flow control to prevent ring overflow.
        bool is_locked = rx_frontend->locked();
        uint64_t cur_fec = opt.multi_modcod() ? (fec_multi ? fec_multi->total_frame_count() : 0)
                                              : (fec_single ? fec_single->frame_count() : 0);
        const uint64_t cur_dummy = rx_frontend->dummy_count();
        if (cur_fec != rx_fec_frames_at_last_check ||
            cur_dummy != rx_dummy_frames_at_last_check)
            last_valid_frame_progress = rx_loop_now;
        // FEC output is intentionally absent throughout a dummy-idle run.
        // Treat a recently decoded standards-compliant dummy PLFRAME as valid
        // receive progress too, and retain that state between frame
        // boundaries.  The previous one-loop FEC test repeatedly classified
        // healthy dummy idle as a false lock, skipped fresh IQ, and injected
        // discontinuities immediately before normal probe/data bursts.
        constexpr auto RX_PROGRESS_HOLD = std::chrono::milliseconds(100);
        bool is_receiving = is_locked &&
            (rx_loop_now - last_valid_frame_progress < RX_PROGRESS_HOLD);
        rx_fec_frames_at_last_check = cur_fec;
        rx_dummy_frames_at_last_check = cur_dummy;

        const double flush_sec = is_receiving ? 1.5 : 0.005;
        const size_t flush_threshold = (size_t)(opt.rate * flush_sec);
        if (ring_occ > flush_threshold) {
            size_t keep = (size_t)(opt.rate * (is_receiving ? 0.05 : 0.002));
            size_t to_skip = ring_occ - keep;
            rx_ring.skip(to_skip);
            n_buffered = 0;
            rx_frontend->notify_discontinuity();
            if (is_receiving || opt.debug_level > 0)
                fprintf(stderr, "dvbs2-uhd-trx: RX ring flush (%.1fms, %s), "
                        "kept %.1fms\n",
                        (double)ring_occ / opt.rate * 1000.0,
                        is_receiving ? "locked" : "searching",
                        (double)keep / opt.rate * 1000.0);
        }

        // When not receiving (unlocked or false-locked on noise),
        // rate-limit processing to prevent ring overflow. Skip to latest
        // data, keep only enough for one acquisition attempt (~3 PLFRAMEs).
        if (!is_receiving && ring_occ > 0) {
            // Locking needs several CONTIGUOUS frames (timing warmup,
            // frame-sync lock, then validation frames).
            // Trimming on every call guarantees a gap per ~2 PLFRAMEs and
            // acquisition never completes at rates the search can't chase
            // in real time (the 100e6+/245.76 no-lock pathology). Instead:
            // catch up to fresh data once, then process a contiguous burst
            // long enough to lock before the next catch-up trim.
            const long burst_len =
                (long)(frame_format.plframe_symbols() * hw_sps) * 16 + 10000;
            if (acq_burst_remaining <= 0) {
                size_t keep = (size_t)(frame_format.plframe_symbols() * hw_sps * 3 + 10000);
                if (ring_occ > keep) {
                    rx_ring.skip(ring_occ - keep);
                    n_buffered = 0;
                    // The estimators must not blend across the gap.
                    rx_frontend->notify_discontinuity();
                }
                acq_burst_remaining = burst_len;
            }
        } else {
            acq_burst_remaining = 0; // receiving: normal flow control
        }

        int n_in;
        {
            // Direct read of hardware-rate samples; the frontend's
            // fractional-stride timing NCO consumes them as-is.
            int max_read = in_buf_size - n_buffered;
            if (max_read <= 0) {
                n_buffered = 0;
                max_read = chunk_size;
            }
            if (!is_receiving) {
                int acq_limit = (int)(frame_format.plframe_symbols() * hw_sps) * 3 + 10000;
                max_read = std::min(max_read, acq_limit);
            }
            update_stage(1); // ring read
            int n_read = source.read(in_buf_ptr + n_buffered, max_read);
            if (n_read <= 0) {
                if (g_stop.load()) break;
                continue;
            }
            acq_burst_remaining -= n_read;
            n_in = n_buffered + n_read;
        }
        // Feed spectrum tap with raw hardware-rate samples so the
        // dashboard shows the true RF spectrum at full BW.
        if (wf_tap) wf_tap->feed(in_buf_ptr, n_in);

        // Buffer multiple ring reads before calling the frontend to
        // amortize the ~150µs cudaStreamSynchronize overhead on ARM64.
        // At 245.76 Msps, without buffering each call processes ~17K
        // samples (1 PLF) and the sync dominates. With rx_min_batch_ms
        // worth of samples, the sync is amortized over more data.
        if (is_receiving && rx_min_batch_samples > 0) {
            if (n_in < rx_min_batch_samples) {
                n_buffered = n_in;
                continue;  // accumulate more data
            }
        }

        int n_consumed = 0;
        std::vector<frontend_frame_t> frames;

        auto fe_start = std::chrono::steady_clock::now();
        update_stage(3); // process
        rx_frontend->process(in_buf_ptr, n_in, sym_buf.data(),
                             (int)sym_buf.size(), n_consumed, frames);
        sync_count_rx.fetch_add(1, std::memory_order_relaxed);
        update_stage(4); // post-process
        auto fe_end = std::chrono::steady_clock::now();
        if (opt.multi_modcod())
            acm_frames_dummy.store(rx_frontend->dummy_count(),
                                   std::memory_order_relaxed);
        uint64_t fe_total_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                fe_end - fe_start).count();
        lat_frontend_us.store(fe_total_us, std::memory_order_relaxed);

        // Store frontend sub-stage breakdown
        {
            const auto& p = rx_frontend->perf();
            lat_fe_agc_us.store(p.agc_us, std::memory_order_relaxed);
            lat_fe_timing_us.store(p.timing_us, std::memory_order_relaxed);
            lat_fe_sync_us.store(p.sync_us, std::memory_order_relaxed);
            lat_fe_n_in.store(p.n_in, std::memory_order_relaxed);
            lat_fe_n_syms.store(p.n_syms, std::memory_order_relaxed);
            lat_fe_fused.store(p.fused, std::memory_order_relaxed);
            lat_tr_memcpy_us.store(p.tr_memcpy_us, std::memory_order_relaxed);
            lat_tr_launch_us.store(p.tr_launch_us, std::memory_order_relaxed);
            lat_tr_lnch_est_us.store(p.tr_lnch_est_us, std::memory_order_relaxed);
            lat_tr_lnch_interp_us.store(p.tr_lnch_interp_us, std::memory_order_relaxed);
            lat_tr_sync_us.store(p.tr_sync_us, std::memory_order_relaxed);
            lat_tr_readback_us.store(p.tr_readback_us, std::memory_order_relaxed);
            lat_gpu_timing_us.store(p.gpu_timing_us, std::memory_order_relaxed);
            lat_gpu_interp_us.store(p.gpu_interp_us, std::memory_order_relaxed);

            if (fe_trace.active()) {
                size_t rx_occ = rx_ring.available_read();
                fe_trace.write(*rx_frontend, start_time, fe_end, fe_total_us,
                               n_consumed, frames.size(), opt.rate,
                               ewma_fer.load(std::memory_order_relaxed),
                               rx_fec_frames.load(std::memory_order_relaxed),
                               rx_fec_errors.load(std::memory_order_relaxed),
                               frame_queue.available(),
                               gpu_inflight_count.load(std::memory_order_relaxed),
                               rx_occ);
            }
        }

        // RX overload guard (rate-limited, loud): the precursor to the
        // ring-skip frame corruption at very high frame rates is the frontend
        // processing time approaching the batch's real-time budget
        // (n_in / sample_rate). Warn before it tips over so the operator can
        // act (lower the symbol rate, or dedicate more CPU cores to the RX).
        if (rx_frontend->locked() && n_in > 0 && opt.rate > 0.0) {
            double fe_us = std::chrono::duration<double, std::micro>(
                               fe_end - fe_start).count();
            double budget_us = (double)n_in / opt.rate * 1e6;
            static double overload_util_ewma = 0.0;
            static double backlog_ms_ewma = 0.0;
            if (budget_us > 0.0) {
                const double util = fe_us / budget_us;
                overload_util_ewma = overload_util_ewma == 0.0
                    ? util : 0.01 * util + 0.99 * overload_util_ewma;
            }
            // A high per-call utilization is harmless while the consumer is
            // keeping the ring drained: batch sizes and CUDA launch overhead
            // make this ratio noisy, especially at fractional SPS.  It only
            // predicts frame loss once input is also accumulating.  Require
            // 2 ms of queued IQ before calling the condition an overload.
            const size_t rx_queued = rx_ring.available_read();
            const double backlog_ms = 1000.0 * (double)rx_queued / opt.rate;
            backlog_ms_ewma = 0.01 * backlog_ms + 0.99 * backlog_ms_ewma;
            if (budget_us > 0.0 && overload_util_ewma > 0.90 &&
                backlog_ms_ewma >= 2.0) {
                static auto last_warn =
                    std::chrono::steady_clock::now() - std::chrono::seconds(10);
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_warn).count() > 2.0) {
                    last_warn = now;
                    fprintf(stderr,
                        "\n*** RX OVERLOAD WARNING: frontend at %.0f%% of the "
                        "real-time budget with %.1f ms queued "
                        "(%.2f ms work vs %.2f ms for %d samples). "
                        "The RX timing thread is near saturation and frame loss is "
                        "imminent — reduce the symbol rate or dedicate more CPU "
                        "cores to the RX. ***\n",
                        100.0 * overload_util_ewma, backlog_ms_ewma,
                        fe_us / 1000.0,
                        budget_us / 1000.0, n_in);
                }
            }
        }

        // Detect lock loss → signal FEC thread to reset
        {
            bool rx_locked_now = rx_frontend->locked();
            if (rx_was_locked && !rx_locked_now) {
                auto lock_loss_time = std::chrono::steady_clock::now();
                double elapsed_s = std::chrono::duration<double>(
                    lock_loss_time - start_time).count();
                double fe_ms = std::chrono::duration<double, std::milli>(
                    fe_end - fe_start).count();
                size_t rx_ring_occ = rx_ring.available_read();
                size_t tx_ring_occ = tx_ring.available_read();
                double rx_ring_ms = (double)rx_ring_occ / opt.rate * 1000.0;
                double tx_ring_ms = (double)tx_ring_occ / opt.rate * 1000.0;
                fprintf(stderr,
                    "\n===== LOCK LOST at t=%.1fs =====\n"
                    "  frontend: %.2f ms, n_in=%d, n_consumed=%d, frames=%zu\n"
                    "  rx_ring: %zu samples (%.1f ms)\n"
                    "  tx_ring: %zu samples (%.1f ms)\n"
                    "  UHD_OF=%lu, RingDrop=%lu, TxUF=%lu, TxSeq=%lu, TxLate=%lu\n"
                    "  FECFRAMEs=%lu, FER=%lu\n"
                    "  pilotSNR=%.2f, freq_off=%.6f, mu=%.4f, agc=%.4f\n"
                    "  TxFrames=%lu, TxDummy=%lu, PDUs=%lu\n"
                    "===== END LOCK LOST =====\n\n",
                    elapsed_s,
                    fe_ms, n_in, n_consumed, frames.size(),
                    rx_ring_occ, rx_ring_ms,
                    tx_ring_occ, tx_ring_ms,
                    (unsigned long)rx_uhd_overflows.load(),
                    (unsigned long)rx_ring_drops.load(),
                    (unsigned long)tx_underflows.load(),
                    (unsigned long)tx_sequence_errors.load(),
                    (unsigned long)tx_time_errors.load(),
                    (unsigned long)rx_fec_frames.load(),
                    (unsigned long)rx_fec_errors.load(),
                    rx_frontend->pilot_snr_db(),
                    rx_frontend->freq_offset(),
                    rx_frontend->get_mu(),
                    rx_frontend->agc_gain(),
                    (unsigned long)tx_frames.load(),
                    (unsigned long)tx_dummy_frames.load(),
                    (unsigned long)rx_pdus.load());
                if (opt.multi_modcod()) {
                    fprintf(stderr, "  ACM sub=%lu skip_len=%lu dummy=%lu mc_hist=[",
                        (unsigned long)acm_frames_submitted.load(),
                        (unsigned long)acm_frames_skipped_len.load(),
                        (unsigned long)acm_frames_dummy.load());
                    for (int i = 0; i < 256; i++) {
                        uint64_t cnt = acm_mc_hist[i].load();
                        if (cnt > 0) fprintf(stderr, "%d:%lu ", i, (unsigned long)cnt);
                    }
                    fprintf(stderr,
                        "] pls_guard={corrected:%lu deferred:%lu transitions:%lu outage:%lu}\n",
                        (unsigned long)acm_pls_guard_corrected.load(),
                        (unsigned long)acm_pls_guard_deferred.load(),
                        (unsigned long)acm_pls_guard_transitions.load(),
                        (unsigned long)acm_pls_guard_outage_frames.load());
                }
                // Signal FEC thread to reset
                fec_reset_seq.fetch_add(1, std::memory_order_acq_rel);
                fec_bootstrapped.store(false, std::memory_order_relaxed);
                last_unlock_time = std::chrono::steady_clock::now();
                ordinary_frames_since_lock = 0;
                fer_watchdog_armed = false;
            }
            // Track lock acquisition time for FER-gated relock
            if (!rx_was_locked && rx_locked_now) {
                lock_acquired_time = std::chrono::steady_clock::now();
                ordinary_frames_since_lock = frames.size();
            } else if (rx_was_locked && rx_locked_now) {
                ordinary_frames_since_lock += frames.size();
            }
            rx_was_locked = rx_locked_now;
        }

        // A correctly decoded dummy PLS header is positive carrier/frame
        // activity even though dummy PLFRAMEs intentionally bypass FEC.
        // Keep this separate from the signal-loss timer below: both the
        // pre-bootstrap and post-bootstrap FER watchdogs need the same fact.
        static uint64_t fer_watchdog_last_dummy = 0;
        static auto fer_watchdog_last_dummy_time =
            std::chrono::steady_clock::time_point::min();
        const auto watchdog_now = std::chrono::steady_clock::now();
        const uint64_t watchdog_dummy_count = rx_frontend->dummy_count();
        if (watchdog_dummy_count != fer_watchdog_last_dummy) {
            fer_watchdog_last_dummy = watchdog_dummy_count;
            fer_watchdog_last_dummy_time = watchdog_now;
        }
        const bool dummy_activity_recent =
            fer_watchdog_last_dummy_time !=
                std::chrono::steady_clock::time_point::min() &&
            watchdog_now - fer_watchdog_last_dummy_time <
                std::chrono::milliseconds(500);

        // FER-gated relock: if locked but no successful decode,
        // the coarse frequency estimate likely converged wrong. Force
        // unlock to trigger re-acquisition.
        // Fast path: if pilot SNR < 0 dB, clearly locked on noise — relock after 1s.
        // Slow path: 5s timeout for marginal SNR cases.
        // Dummy activity alone is a legitimate idle state, and ACM may need
        // to remain locked long enough to return a downshift report.  Neither
        // condition may suppress recovery forever, however: every peer emits
        // robust ordinary startup/idle probes.  If a full probe burst has
        // reached the FEC path without one successful decode, this is a false
        // lock even when dummy PLS headers continue to decode correctly.
        if (rx_frontend->locked() &&
            !fec_bootstrapped.load(std::memory_order_relaxed))
        {
            float psnr = rx_frontend->pilot_snr_db();
            const uint64_t probe_burst_frames = static_cast<uint64_t>(
                std::max(1, opt.idle_probe_frames));
            const bool probes_exhausted =
                ordinary_frames_since_lock >= probe_burst_frames;
            bool suppress =
                !probes_exhausted && (dummy_activity_recent || (opt.multi_modcod() && psnr > 3.0f));
            if (!suppress) {
                auto since_lock = std::chrono::steady_clock::now() - lock_acquired_time;
                // Once a complete robust probe burst has failed, additional
                // waiting cannot produce a bootstrap decode.  Retry quickly;
                // this also bounds recovery if the first post-outage SOF
                // candidate was a noise correlation.
                int timeout_s = (psnr < 3.0f || probes_exhausted) ? 1 : 5;
                if (since_lock > std::chrono::seconds(timeout_s)) {
                    fprintf(stderr, "dvbs2-uhd-trx: FER-gated relock — locked for "
                            "%.1fs with no successful FEC decode "
                            "(pilotSNR=%.1f ordinary=%lu), "
                            "forcing re-acquire\n",
                            std::chrono::duration<double>(since_lock).count(), psnr,
                            (unsigned long)ordinary_frames_since_lock);
                    rx_frontend->force_unlock();
                    fec_reset_seq.fetch_add(1, std::memory_order_acq_rel);
                    lock_acquired_time = std::chrono::steady_clock::now();
                    last_unlock_time = std::chrono::steady_clock::now();
                    ordinary_frames_since_lock = 0;
                    rx_was_locked = false;
                }
            }
        }

        // Post-bootstrap FER watchdog: if EWMA FER stays above 0.9
        // for >2s (1s if pilotSNR < 0), the receiver is locked on
        // garbage — force re-acquire.
        // ACM exception: in ACM mode with pilot SNR > 3 dB, high FER
        // means the peer's modcod is too high — stay locked so ACM
        // feedback can still flow over the data plane.
        if (rx_frontend->locked() &&
            fec_bootstrapped.load(std::memory_order_relaxed))
        {
            float fer = ewma_fer.load(std::memory_order_relaxed);
            float psnr = rx_frontend->pilot_snr_db();
            bool suppress = dummy_activity_recent || (opt.multi_modcod() && psnr > 3.0f);
            if (fer > 0.9f && !suppress) {
                if (!fer_watchdog_armed) {
                    fer_watchdog_armed = true;
                    fer_high_since = std::chrono::steady_clock::now();
                } else {
                    int timeout_s = (psnr < 0.0f) ? 1 : 2;
                    auto elapsed = std::chrono::steady_clock::now() - fer_high_since;
                    if (elapsed > std::chrono::seconds(timeout_s)) {
                        fprintf(stderr, "dvbs2-uhd-trx: FER watchdog — "
                                "ewma_fer=%.2f for %.1fs (pilotSNR=%.1f), "
                                "forcing re-acquire\n",
                                fer,
                                std::chrono::duration<double>(elapsed).count(),
                                psnr);
                        rx_frontend->force_unlock();
                        fec_reset_seq.fetch_add(1, std::memory_order_acq_rel);
                        fec_bootstrapped.store(false, std::memory_order_relaxed);
                        lock_acquired_time = std::chrono::steady_clock::now();
                        last_unlock_time = std::chrono::steady_clock::now();
                        rx_was_locked = false;
                        fer_watchdog_armed = false;
                    }
                }
            } else {
                fer_watchdog_armed = false;
            }
        }

        // Signal-loss watchdog: if locked but no new PL frames for
        // >3 seconds, the signal is definitely gone. Dummy PLFRAMEs are
        // valid carrier activity during application-idle periods, even
        // though they intentionally bypass FEC.
        // where the FER watchdog is suppressed (ACM mode with stale
        // pilotSNR > 3 dB) and coherence-based detection is too slow
        // (sporadic noise correlations reset the streak counter).
        // Uses raw frame count — immune to SNR/coherence estimation.
        {
            static uint64_t signal_loss_last_fec = 0;
            static uint64_t signal_loss_last_dummy = 0;
            static auto signal_loss_t0 = std::chrono::steady_clock::now();
            if (rx_frontend->locked()) {
                uint64_t cur_fec = opt.multi_modcod()
                                       ? (fec_multi ? fec_multi->total_frame_count() : 0)
                                       : (fec_single ? fec_single->frame_count() : 0);
                uint64_t cur_dummy = rx_frontend->dummy_count();
                if (cur_fec > signal_loss_last_fec ||
                    cur_dummy > signal_loss_last_dummy) {
                    signal_loss_last_fec = cur_fec;
                    signal_loss_last_dummy = cur_dummy;
                    signal_loss_t0 = std::chrono::steady_clock::now();
                } else {
                    auto elapsed = std::chrono::steady_clock::now() - signal_loss_t0;
                    if (elapsed > std::chrono::seconds(3)) {
                        fprintf(stderr, "dvbs2-uhd-trx: signal loss watchdog — "
                                "locked but 0 new PL frames for %.1fs, "
                                "forcing re-acquire\n",
                                std::chrono::duration<double>(elapsed).count());
                        rx_frontend->force_unlock();
                        fec_reset_seq.fetch_add(1, std::memory_order_acq_rel);
                        fec_bootstrapped.store(false, std::memory_order_relaxed);
                        lock_acquired_time = std::chrono::steady_clock::now();
                        last_unlock_time = std::chrono::steady_clock::now();
                        rx_was_locked = false;
                        fer_watchdog_armed = false;
                        signal_loss_last_fec = 0;
                        signal_loss_last_dummy = 0;
                        signal_loss_t0 = std::chrono::steady_clock::now();
                    }
                }
            } else {
                signal_loss_last_fec = 0;
                signal_loss_last_dummy = 0;
                signal_loss_t0 = std::chrono::steady_clock::now();
            }
        }

        // No-lock timeout: if never locked (or lost lock) for >10s,
        // AGC/timing recovery may be stuck. Reset frontend to retry.  During
        // an explicitly signalled link recovery, retry every second until a
        // real FEC decode confirms acquisition; otherwise a bad SOF candidate
        // can consume most of an accelerated visibility window.
        if (fec_bootstrapped.load(std::memory_order_relaxed))
            link_recovery_active = false;
        if (!rx_frontend->locked()) {
            auto since_unlock = std::chrono::steady_clock::now() - last_unlock_time;
            const int timeout_s = link_recovery_active ? 1 : 10;
            if (since_unlock > std::chrono::seconds(timeout_s)) {
                fprintf(stderr, "dvbs2-uhd-trx: no-lock timeout — no lock for "
                        "%.1fs, resetting frontend%s\n",
                        std::chrono::duration<double>(since_unlock).count(),
                        link_recovery_active ? " (link recovery)" : "");
                // force_unlock() intentionally does nothing while already
                // unlocked; this path requires an unconditional estimator
                // reset.
                rx_frontend->force_reacquire();
                fec_reset_seq.fetch_add(1, std::memory_order_acq_rel);
                fec_bootstrapped.store(false, std::memory_order_relaxed);
                last_unlock_time = std::chrono::steady_clock::now();
            }
        }

        // Enqueue completed frames to FEC thread
        for (const auto& frame : frames) {
            if (!rx_frontend->locked())
                continue;
            if (!frame_queue.try_push(frame))
                rx_fec_queue_drops.fetch_add(1);
        }

        n_buffered = n_in - n_consumed;
        if (n_consumed > 0 && n_buffered > 0) {
            memmove(in_buf_ptr, in_buf_ptr + n_consumed,
                    n_buffered * sizeof(complexf));
        }
    }

    // Flush deferred pipeline symbols
    {
        std::vector<frontend_frame_t> flush_frames;
        rx_frontend->flush(sym_buf.data(), (int)sym_buf.size(), flush_frames);
        for (const auto& frame : flush_frames) {
            if (!rx_frontend->locked()) continue;
            if (!frame_queue.try_push(frame))
                rx_fec_queue_drops.fetch_add(1);
        }
    }

    if (in_buf_pinned)
        cudaFreeHost(in_buf_ptr);
    else
        free(in_buf_ptr);
}

// ====================================================================
// Thread 5: RX FEC — frame_queue → phase → demap+LDPC → BCH → GSE → TAP
// ====================================================================
void rx_fec_thread(trx_context& ctx)
{
    auto& opt = *ctx.opt;
    const bool acm_tx_enabled = ctx.acm_tx_enabled;
    const int xfecframe_len = ctx.xfecframe_len;
    auto& tap = *ctx.tap;
    auto& frame_queue = *ctx.frame_queue;
    auto& fec_single = *ctx.fec_single;
    auto& fec_multi = *ctx.fec_multi;
    auto& rx_decap = *ctx.rx_decap;
    auto& acm_inband_snr = ctx.stats.acm_inband_snr;
    auto& acm_inband_fer = ctx.stats.acm_inband_fer;
    auto& acm_inband_fec_frames = ctx.stats.acm_inband_fec_frames;
    auto& acm_inband_fec_errors = ctx.stats.acm_inband_fec_errors;
    auto& acm_inband_rx_modcod = ctx.stats.acm_inband_rx_modcod;
    auto& acm_feedback_rx = ctx.stats.acm_feedback_rx;
    auto& rx_pdus = ctx.stats.rx_pdus;
    auto& rx_pdu_bytes = ctx.stats.rx_pdu_bytes;
    auto& rx_fec_frames = ctx.stats.rx_fec_frames;
    auto& rx_fec_errors = ctx.stats.rx_fec_errors;
    auto& rx_fec_frames_pre = ctx.stats.rx_fec_frames_pre;
    auto& fec_bootstrapped = ctx.stats.fec_bootstrapped;
    auto& fec_reset_seq = ctx.stats.fec_reset_seq;
    auto& ewma_fer = ctx.stats.ewma_fer;
    auto& acm_frames_submitted = ctx.stats.acm_frames_submitted;
    auto& acm_frames_skipped_len = ctx.stats.acm_frames_skipped_len;
    auto& acm_frames_dummy = ctx.stats.acm_frames_dummy;
    auto& acm_mc_hist = ctx.stats.acm_mc_hist;
    auto& acm_mc_fec_frames = ctx.stats.acm_mc_fec_frames;
    auto& acm_mc_fec_errors = ctx.stats.acm_mc_fec_errors;
    auto& acm_pls_guard_corrected = ctx.stats.acm_pls_guard_corrected;
    auto& acm_pls_guard_deferred = ctx.stats.acm_pls_guard_deferred;
    auto& acm_pls_guard_transitions = ctx.stats.acm_pls_guard_transitions;
    auto& acm_pls_guard_outage_frames =
        ctx.stats.acm_pls_guard_outage_frames;
    auto& rx_data_modcod = ctx.stats.rx_data_modcod;
    auto& lat_fec_loop_us = ctx.stats.lat_fec_loop_us;
    auto& lat_gpu_drain_us = ctx.stats.lat_gpu_drain_us;
    auto& lat_gpu_submit_us = ctx.stats.lat_gpu_submit_us;
    auto& lat_fec_decode_us = ctx.stats.lat_fec_decode_us;
    auto& gpu_inflight_count = ctx.stats.gpu_inflight_count;
    auto& gpu_force_drain_count = ctx.stats.gpu_force_drain_count;

    uint64_t ewma_prev_frames = 0;  // only touched from FEC thread
    uint64_t ewma_prev_errors = 0;  // only touched from FEC thread
    // Post-attach FER: after bootstrap (first decode), skip a short settling
    // window before the *reported* FER starts counting, so it reflects the
    // steady-state and not the acquisition transient (timing/phase/AGC still
    // converging right after lock). Reset on relock.
    bool fec_attached = false;
    uint64_t attach_frame_off = 0, attach_error_off = 0;
    std::array<uint64_t, 256> acm_mc_attach_frames{};
    std::array<uint64_t, 256> acm_mc_attach_errors{};
    uint64_t fec_reset_seen = 0;
    constexpr uint64_t FEC_ATTACH_WARMUP_FRAMES = 256;
    static constexpr float EWMA_ALPHA = 0.02f;

    // PLS is intentionally decoded before FEC, so at low SNR an otherwise
    // valid frame can occasionally acquire a plausible but wrong MODCOD. ACM
    // makes this more dangerous than CCM: the wrong label selects a real (but
    // wrong) decoder chain. Orbital ACM transitions persist for many frames;
    // require a short run before changing the accepted data MODCOD. For equal
    // PLFRAME lengths we can recover an isolated bad label by decoding with
    // the accepted mode. For incompatible lengths, defer/drop the candidate
    // until it is confirmed instead of deliberately feeding the wrong chain.
    constexpr unsigned PLS_TRANSITION_CONFIRM_FRAMES = 3;
    bool pls_guard_locked = false;
    uint8_t pls_guard_current = 0;
    uint8_t pls_guard_candidate = 0;
    unsigned pls_guard_candidate_count = 0;
    struct pending_pls_frame_t {
        std::vector<complexf> symbols;
        int payload_len = 0;
        float pilot_snr_db = 0.0f;
    };
    std::vector<pending_pls_frame_t> pls_guard_pending;
    pls_guard_pending.reserve(PLS_TRANSITION_CONFIRM_FRAMES - 1);
    bool pls_guard_outage = false;
    unsigned pls_guard_low_count = 0;
    unsigned pls_guard_recovery_count = 0;
    float pls_guard_snr_ewma = NAN;
    constexpr unsigned PLS_OUTAGE_ENTER_FRAMES = 16;
    constexpr unsigned PLS_OUTAGE_EXIT_FRAMES = 16;
    constexpr float PLS_OUTAGE_ENTER_HEADROOM_DB = 0.5f;
    constexpr float PLS_OUTAGE_EXIT_HEADROOM_DB = 1.0f;
    constexpr float PLS_OUTAGE_SNR_ALPHA = 0.05f;

    if (!opt.do_rx) return;
    if (opt.pin_threads) pin_thread("rx_fec", 5, opt.realtime_threads);

    fec_trace_writer fec_trace;
    fec_trace.open(opt.fec_trace_csv, "dvbs2-uhd-trx");

    // Set TAP to non-blocking so FEC thread never stalls on write
    if (tap) tap->set_nonblocking(true);

    int max_out =
        opt.multi_modcod() ? fec_multi->max_output_bytes() : fec_single->max_output_bytes();
    std::vector<uint8_t> fec_out_buf(max_out);

    // GSE PDU callback — reconstruct Ethernet frame from L3 payload,
    // or extract in-band ACM signaling from QPSK 1/4 frames.
    // Holds a reassembled PDU at a jumbo TAP MTU (matches tap_reader's buf);
    // PDUs larger than this are dropped (see guard below). GSE itself
    // reassembles up to 65535 B, so this buffer is the practical MTU cap.
    std::vector<uint8_t> eth_frame_buf(16384 + ETH_HLEN);
    bool acm_feedback_seq_valid = false;
    uint16_t acm_last_feedback_seq = 0;
    auto pdu_cb = [&](const uint8_t* pdu, int pdu_len, uint16_t proto,
                      const uint8_t* label, int label_len) {
        (void)label; (void)label_len;
        if (proto == ACM_SIG_PROTO &&
            pdu_len >= (int)sizeof(acm_feedback_pkt_t)) {
            acm_feedback_pkt_t fb;
            memcpy(&fb, pdu, sizeof(fb));
            if (fb.magic == ACM_FEEDBACK_MAGIC) {
                // A control opportunity repeats one report several times to
                // survive a correlated MODCOD transition.  Consume the first
                // successfully decoded copy only; otherwise one robust burst
                // would look like several independent channel observations.
                if (acm_feedback_seq_valid &&
                    fb.seq == acm_last_feedback_seq)
                    return;
                acm_feedback_seq_valid = true;
                acm_last_feedback_seq = fb.seq;
                acm_inband_fec_frames.store(fb.fec_frames,
                                            std::memory_order_relaxed);
                acm_inband_fec_errors.store(fb.fec_errors,
                                            std::memory_order_relaxed);
                acm_inband_rx_modcod.store(fb.rx_modcod,
                                           std::memory_order_relaxed);
                acm_inband_fer.store(fb.fer,
                                     std::memory_order_relaxed);
                // Publish SNR last: its release pairs with the main thread's
                // acquire and makes the outcome snapshot visible as one sample.
                acm_inband_snr.store(fb.snr_db,
                                     std::memory_order_release);
                acm_feedback_rx.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (tap && pdu_len > 0) {
            if ((size_t)(ETH_HLEN + pdu_len) > eth_frame_buf.size()) {
                fprintf(stderr, "WARNING: GSE PDU too large (%d bytes), dropping\n", pdu_len);
            } else {
                // GSE carries the L3 packet, not the peer's Ethernet header.
                // Target the TAP's current MAC. Linux drops TCP packets whose
                // skb is marked L2 broadcast even when the L3 address is local;
                // tap_interface refreshes the MAC if a network manager changes it.
                memcpy(eth_frame_buf.data(), tap->mac(), 6);
                memcpy(eth_frame_buf.data() + 6, PEER_MAC, 6);
                eth_frame_buf[12] = (proto >> 8) & 0xFF;
                eth_frame_buf[13] = proto & 0xFF;
                memcpy(eth_frame_buf.data() + ETH_HLEN, pdu, pdu_len);
                tap->write_frame_nonblocking(eth_frame_buf.data(), ETH_HLEN + pdu_len);
            }
            rx_pdu_bytes.fetch_add(pdu_len);
        }
        rx_pdus.fetch_add(1);
    };

    // Drain decoded FEC output (GSE → TAP, or discard if no TAP)
    auto drain_fec_output = [&]() {
        if (tap && rx_decap) {
            int total;
            if (opt.multi_modcod()) {
                total = fec_multi->get_gse_output(
                    fec_out_buf.data(), (int)fec_out_buf.size());
                const auto& sizes = fec_multi->gse_dfl_sizes();
                int offset = 0;
                for (int sz : sizes) {
                    rx_decap->decapsulate(
                        fec_out_buf.data() + offset, sz, pdu_cb);
                    offset += sz;
                }
            } else {
                total = fec_single->get_gse_output(
                    fec_out_buf.data(), (int)fec_out_buf.size());
                if (total > 0) {
                    const auto& sizes = fec_single->gse_dfl_sizes();
                    int offset = 0;
                    for (int sz : sizes) {
                        rx_decap->decapsulate(
                            fec_out_buf.data() + offset, sz, pdu_cb);
                        offset += sz;
                    }
                }
            }
        } else {
            // No TAP: drain decoded output (discard)
            if (opt.multi_modcod())
                fec_multi->get_gse_output(fec_out_buf.data(),
                                           (int)fec_out_buf.size());
            else
                fec_single->get_gse_output(fec_out_buf.data(),
                                            (int)fec_out_buf.size());
        }
        // Read raw aggregate counters from rx_fec_chain (reset on lock loss)
        uint64_t raw_frames =
            opt.multi_modcod() ? fec_multi->total_frame_count() : fec_single->frame_count();
        uint64_t raw_errors =
            opt.multi_modcod() ? fec_multi->total_error_count() : fec_single->frame_error_count();

        // Check FEC bootstrap (first successful decode)
        if (!fec_bootstrapped.load(std::memory_order_relaxed)) {
            if (raw_frames > raw_errors) {
                fec_bootstrapped.store(true, std::memory_order_release);
                // Do not reset the decoder merely to clear statistics here.
                // reset() drains every in-flight CUDA event while holding the
                // process-wide CUDA mutex.  On Jetson that can stop both the
                // RX frontend and TX generator for several milliseconds and
                // underflow a deliberately shallow, low-latency TX IQ ring.
                // Baseline the raw counters instead; real lock-loss resets
                // still use the full reset path below.
                ewma_fer.store(0.0f, std::memory_order_relaxed);
                ewma_prev_frames = raw_frames;
                ewma_prev_errors = raw_errors;
                rx_fec_frames.store(0);
                rx_fec_errors.store(0);
                fec_attached = false;      // start the post-attach warm-up
                attach_frame_off = raw_frames;
                attach_error_off = raw_errors;
                acm_mc_attach_frames.fill(0);
                acm_mc_attach_errors.fill(0);
                if (opt.multi_modcod()) {
                    for (const auto& m : get_supported_modcods()) {
                        uint64_t frames = 0, errors = 0;
                        if (fec_multi->modcod_counters(
                                m.modcod_idx, frames, errors)) {
                            acm_mc_attach_frames[m.modcod_idx] = frames;
                            acm_mc_attach_errors[m.modcod_idx] = errors;
                        }
                        acm_mc_fec_frames[m.modcod_idx].store(
                            0, std::memory_order_relaxed);
                        acm_mc_fec_errors[m.modcod_idx].store(
                            0, std::memory_order_relaxed);
                    }
                }
                fprintf(stderr, "dvbs2-uhd-trx: FEC bootstrapped "
                        "(discarded %lu pre-bootstrap frames)\n",
                        (unsigned long)raw_frames);
            } else {
                rx_fec_frames_pre.store(raw_frames);
            }
        } else {
            // Post-bootstrap: skip a settling window, then report post-attach
            // (steady-state) counts. During the window report raw for
            // visibility; at attach, latch the offset so the reported FER from
            // then on excludes the acquisition transient.
            if (!fec_attached &&
                raw_frames - attach_frame_off >= FEC_ATTACH_WARMUP_FRAMES) {
                fec_attached = true;
                attach_frame_off = raw_frames;
                attach_error_off = raw_errors;
                if (opt.multi_modcod()) {
                    for (const auto& m : get_supported_modcods()) {
                        uint64_t frames = 0, errors = 0;
                        if (fec_multi->modcod_counters(
                                m.modcod_idx, frames, errors)) {
                            acm_mc_attach_frames[m.modcod_idx] = frames;
                            acm_mc_attach_errors[m.modcod_idx] = errors;
                        }
                    }
                }
            }
            rx_fec_frames.store(raw_frames - attach_frame_off);
            rx_fec_errors.store(raw_errors - attach_error_off);

            // Publish post-attach counters for each decoder chain. Matching
            // the aggregate FER's settling window keeps acquisition errors
            // out of OLLA while retaining a causal outcome stream per MODCOD.
            if (opt.multi_modcod() && fec_attached) {
                for (const auto& m : get_supported_modcods()) {
                    uint64_t frames = 0, errors = 0;
                    if (fec_multi->modcod_counters(
                            m.modcod_idx, frames, errors)) {
                        const uint64_t frame_off =
                            acm_mc_attach_frames[m.modcod_idx];
                        const uint64_t error_off =
                            acm_mc_attach_errors[m.modcod_idx];
                        acm_mc_fec_frames[m.modcod_idx].store(
                            frames >= frame_off ? frames - frame_off : 0,
                            std::memory_order_relaxed);
                        acm_mc_fec_errors[m.modcod_idx].store(
                            errors >= error_off ? errors - error_off : 0,
                            std::memory_order_relaxed);
                    }
                }
            }

            // EWMA FER: update per-batch (each output call = 1+ new frames)
            uint64_t df = raw_frames - ewma_prev_frames;
            uint64_t de = raw_errors - ewma_prev_errors;
            if (df > 0) {
                float batch_fer = (float)de / (float)df;
                float prev = ewma_fer.load(std::memory_order_relaxed);
                ewma_fer.store(EWMA_ALPHA * batch_fer + (1.0f - EWMA_ALPHA) * prev,
                               std::memory_order_relaxed);
            }
            ewma_prev_frames = raw_frames;
            ewma_prev_errors = raw_errors;
        }
    };

    // Adaptive batch timeout for CPU path
    const auto batch_timeout_fast = std::chrono::milliseconds(opt.batch_timeout_ms);
    const auto batch_timeout_bootstrap = std::chrono::milliseconds(500);
    auto batch_timeout = batch_timeout_bootstrap;

    // Submit a frame held by the PLS transition guard once the subsequent
    // headers establish whether it belonged to a real MODCOD transition.
    // Previously these frames were immediately relabelled as the old MODCOD;
    // equal-length transitions therefore manufactured two FEC failures before
    // the three-header confirmation completed.
    auto submit_held_acm_frame = [&](const pending_pls_frame_t& held,
                                     const pl_layout& decided_pls,
                                     bool& submitted_any,
                                     int& submitted_count,
                                     bool& batch_ready_this_loop) {
        if (held.payload_len != decided_pls.data_symbols()) {
            acm_frames_skipped_len.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (held.pilot_snr_db > 0.0f)
            fec_multi->update_snr(decided_pls.modcod, held.pilot_snr_db);
        auto fec_t0 = std::chrono::steady_clock::now();
        bool ready = fec_multi->submit(held.symbols.data(), held.payload_len,
                                       decided_pls);
        auto submit_dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - fec_t0).count();
        lat_gpu_submit_us.store(submit_dt, std::memory_order_relaxed);
        if (ready) {
            lat_fec_decode_us.store(submit_dt, std::memory_order_relaxed);
            batch_ready_this_loop = true;
            drain_fec_output();
        }
        acm_frames_submitted.fetch_add(1, std::memory_order_relaxed);
        submitted_any = true;
        ++submitted_count;
    };
    auto replay_held_acm_frames = [&](const pl_layout& decided_pls,
                                      bool& submitted_any,
                                      int& submitted_count,
                                      bool& batch_ready_this_loop) {
        for (const auto& held : pls_guard_pending)
            submit_held_acm_frame(held, decided_pls, submitted_any,
                                  submitted_count, batch_ready_this_loop);
        pls_guard_pending.clear();
    };

    auto fec_loop_prev = std::chrono::steady_clock::now();
    while (!g_stop.load(std::memory_order_relaxed)) {
        auto fec_loop_now = std::chrono::steady_clock::now();
        lat_fec_loop_us.store(
            std::chrono::duration_cast<std::chrono::microseconds>(
                fec_loop_now - fec_loop_prev).count(),
            std::memory_order_relaxed);
        fec_loop_prev = fec_loop_now;
        bool drained_this_loop = false;
        bool batch_ready_this_loop = false;
        bool timeout_flush_this_loop = false;
        int submitted_count = 0;

        // Handle FEC reset request from frontend thread
        uint64_t reset_seq = fec_reset_seq.load(std::memory_order_acquire);
        if (reset_seq != fec_reset_seen) {
            {
                // Reset may call cudaStreamSynchronize internally
                // for in-flight work — locking is handled by rx_fec_chain.
                if (opt.multi_modcod())
                    fec_multi->reset_all();
                else
                    fec_single->reset();
            }
            if (rx_decap)
                rx_decap->reset();
            ewma_fer.store(0.0f, std::memory_order_relaxed);
            ewma_prev_frames = 0;
            ewma_prev_errors = 0;
            rx_fec_frames.store(0);
            rx_fec_errors.store(0);
            fec_attached = false;          // re-run the post-attach warm-up
            attach_frame_off = 0;
            attach_error_off = 0;
            acm_mc_attach_frames.fill(0);
            acm_mc_attach_errors.fill(0);
            if (opt.multi_modcod()) {
                for (const auto& m : get_supported_modcods()) {
                    acm_mc_fec_frames[m.modcod_idx].store(
                        0, std::memory_order_relaxed);
                    acm_mc_fec_errors[m.modcod_idx].store(
                        0, std::memory_order_relaxed);
                }
            }
            batch_timeout = batch_timeout_bootstrap;
            fec_reset_seen = reset_seq;
            pls_guard_locked = false;
            pls_guard_candidate_count = 0;
            pls_guard_pending.clear();
            pls_guard_outage = false;
            pls_guard_low_count = 0;
            pls_guard_recovery_count = 0;
            pls_guard_snr_ewma = NAN;
        }

        // 1. Drain completed GPU work (non-blocking)
        {
            auto drain_t0 = std::chrono::steady_clock::now();
            bool drained = false;
            // Locking is now fine-grained inside rx_fec_chain (only
            // around cudaEventQuery/cudaPeekAtLastError).
            if (opt.multi_modcod())
                drained = fec_multi->drain_all_completed();
            else
                drained = fec_single->drain_completed();
            if (drained) {
                drained_this_loop = true;
                lat_gpu_drain_us.store(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - drain_t0).count(),
                    std::memory_order_relaxed);
                drain_fec_output();
                // Successful GPU work is normally observed by this polling
                // path, not by submit_xfecframe() below.  Once the first
                // decode bootstraps FEC, retire the conservative 500 ms
                // acquisition timeout here as well.  Otherwise a sparse data
                // frame left behind the four-frame warm-up waits 500 ms for a
                // partial launch at each endpoint (~1 s RTT while idle).
                if (fec_bootstrapped.load(std::memory_order_relaxed))
                    batch_timeout = batch_timeout_fast;
            }
        }

        // 2. Submit new frames from the queue.
        bool submitted_any = false;
        while (fec_frame_t* frame = frame_queue.try_pop()) {
            const complexf* fec_input = frame->symbols.data();
            bool batch_ready;
            if (opt.multi_modcod()) {
                uint8_t mc = frame->pls.modcod;
                // QPSK 1/4 is reserved only by the legacy in-band feedback
                // mode.  With UDP feedback it is a normal ACM data MODCOD.
                bool is_sig_frame = (acm_tx_enabled &&
                    opt.acm_feedback_dst.empty() && mc == SIG_MODCOD);
                if (!is_sig_frame)
                    acm_mc_hist[mc].fetch_add(1);
                if (!is_sig_frame)
                    rx_data_modcod.store(mc, std::memory_order_relaxed);

                if (frame->pls.dummy) {
                    acm_frames_dummy.fetch_add(1);
                    frame_queue.pop_commit();
                    continue;
                }

                if (frame->payload_len != frame->pls.data_symbols()) {
                    acm_frames_skipped_len.fetch_add(1);
                    frame_queue.pop_commit();
                    continue;
                }

                pl_layout guarded_pls = frame->pls;
                if (!is_sig_frame) {
                    if (!pls_guard_locked) {
                        pls_guard_current = mc;
                        pls_guard_locked = true;
                    }

                    const float frame_snr = frame->pilot_snr_db;
                    if (frame_snr > 0.0f) {
                        if (!std::isfinite(pls_guard_snr_ewma))
                            pls_guard_snr_ewma = frame_snr;
                        else
                            pls_guard_snr_ewma += PLS_OUTAGE_SNR_ALPHA *
                                (frame_snr - pls_guard_snr_ewma);
                    }
                    const bool at_floor = pls_guard_locked &&
                        pls_guard_current == ctx.acm_floor_modcod;
                    const float enter_snr = ctx.acm_floor_pilot_snr_db +
                        PLS_OUTAGE_ENTER_HEADROOM_DB;
                    const float exit_snr = ctx.acm_floor_pilot_snr_db +
                        PLS_OUTAGE_EXIT_HEADROOM_DB;
                    if (at_floor && std::isfinite(pls_guard_snr_ewma)) {
                        if (pls_guard_snr_ewma < enter_snr) {
                            ++pls_guard_low_count;
                            pls_guard_recovery_count = 0;
                            if (pls_guard_low_count >= PLS_OUTAGE_ENTER_FRAMES)
                                pls_guard_outage = true;
                        } else {
                            pls_guard_low_count = 0;
                            if (pls_guard_outage &&
                                pls_guard_snr_ewma >= exit_snr) {
                                if (++pls_guard_recovery_count >=
                                    PLS_OUTAGE_EXIT_FRAMES) {
                                    pls_guard_outage = false;
                                    pls_guard_recovery_count = 0;
                                }
                            } else if (pls_guard_outage) {
                                pls_guard_recovery_count = 0;
                            }
                        }
                    } else if (!at_floor) {
                        pls_guard_outage = false;
                        pls_guard_low_count = 0;
                        pls_guard_recovery_count = 0;
                        pls_guard_snr_ewma = NAN;
                    }

                    const bool floor_freeze = at_floor &&
                        std::isfinite(pls_guard_snr_ewma) &&
                        (pls_guard_outage ||
                         pls_guard_snr_ewma < enter_snr);
                    if (floor_freeze) {
                        // A candidate that failed confirmation belongs to the
                        // established floor mode when its geometry permits it.
                        pl_layout floor_pls = pl_layout_for(pls_guard_current,
                                        opt.short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL, opt.pilots);
                        replay_held_acm_frames(floor_pls, submitted_any,
                            submitted_count, batch_ready_this_loop);
                        pls_guard_candidate_count = 0;
                        acm_pls_guard_outage_frames.fetch_add(
                            1, std::memory_order_relaxed);
                        if (mc != pls_guard_current) {
                            guarded_pls = pl_layout_for(pls_guard_current, opt.short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL, opt.pilots);
                            mc = pls_guard_current;
                            acm_pls_guard_corrected.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    } else if (mc == pls_guard_current) {
                        // The candidate disappeared: retrospectively treat
                        // buffered equal-geometry frames as the established
                        // MODCOD. Incompatible geometry cannot be recovered.
                        pl_layout current_pls = pl_layout_for(pls_guard_current,
                                          opt.short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL, opt.pilots);
                        replay_held_acm_frames(current_pls, submitted_any,
                            submitted_count, batch_ready_this_loop);
                        pls_guard_candidate_count = 0;
                    } else {
                        if (pls_guard_candidate_count == 0 ||
                            mc != pls_guard_candidate) {
                            // A different candidate invalidates the previous
                            // tentative run. Resolve it against the current
                            // mode before beginning the new confirmation.
                            pl_layout current_pls = pl_layout_for(pls_guard_current,
                                              opt.short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL, opt.pilots);
                            replay_held_acm_frames(current_pls, submitted_any,
                                submitted_count, batch_ready_this_loop);
                            pls_guard_candidate = mc;
                            pls_guard_candidate_count = 1;
                        } else {
                            ++pls_guard_candidate_count;
                        }

                        if (pls_guard_candidate_count >=
                            PLS_TRANSITION_CONFIRM_FRAMES) {
                            pls_guard_current = mc;
                            pls_guard_candidate_count = 0;
                            acm_pls_guard_transitions.fetch_add(
                                1, std::memory_order_relaxed);
                            // Replay the frames withheld during confirmation
                            // through the newly established decoder, in order.
                            replay_held_acm_frames(guarded_pls, submitted_any,
                                submitted_count, batch_ready_this_loop);
                        } else {
                            pending_pls_frame_t held;
                            held.symbols.assign(frame->symbols.begin(),
                                frame->symbols.begin() + frame->payload_len);
                            held.payload_len = frame->payload_len;
                            held.pilot_snr_db = frame->pilot_snr_db;
                            pls_guard_pending.emplace_back(std::move(held));
                            acm_pls_guard_deferred.fetch_add(
                                1, std::memory_order_relaxed);
                            frame_queue.pop_commit();
                            continue;
                        }
                    }
                }
                // Update demapper noise variance / LLR gain from pilot SNR
                if (frame->pilot_snr_db > 0.0f)
                    fec_multi->update_snr(mc, frame->pilot_snr_db);
                auto fec_t0 = std::chrono::steady_clock::now();
                // Locking is now fine-grained inside rx_fec_chain (only
                // around GPU kernel launches, not CPU prep work).
                batch_ready = fec_multi->submit(
                    fec_input, frame->payload_len, guarded_pls);
                auto submit_dt = std::chrono::duration_cast<
                    std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - fec_t0).count();
                lat_gpu_submit_us.store(submit_dt, std::memory_order_relaxed);
                if (batch_ready)
                    lat_fec_decode_us.store(submit_dt, std::memory_order_relaxed);
                acm_frames_submitted.fetch_add(1);
            } else {
                if (frame->payload_len != xfecframe_len) {
                    frame_queue.pop_commit();
                    continue;
                }
                // Update demapper noise variance / LLR gain from pilot SNR
                if (frame->pilot_snr_db > 0.0f)
                    fec_single->update_snr(frame->pilot_snr_db);
                auto fec_t0 = std::chrono::steady_clock::now();
                // Locking is now fine-grained inside rx_fec_chain.
                batch_ready = fec_single->submit_xfecframe(
                    fec_input, frame->payload_len);
                auto submit_dt = std::chrono::duration_cast<
                    std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - fec_t0).count();
                lat_gpu_submit_us.store(submit_dt, std::memory_order_relaxed);
                if (batch_ready)
                    lat_fec_decode_us.store(submit_dt, std::memory_order_relaxed);
                gpu_inflight_count.store(fec_single->gpu_inflight(),
                                         std::memory_order_relaxed);
                gpu_force_drain_count.store(fec_single->force_drain_count(),
                                            std::memory_order_relaxed);
                acm_frames_submitted.fetch_add(1);
            }

            frame_queue.pop_commit();
            submitted_any = true;
            submitted_count++;

            if (batch_ready) {
                batch_ready_this_loop = true;
                drain_fec_output();
                if (fec_bootstrapped.load(std::memory_order_relaxed))
                    batch_timeout = batch_timeout_fast;
            }
        }

        // 3. Launch partial micro-batches if their timeout has elapsed.
        //    With large micro_batch (e.g., 64 for int16 LDPC), the batch
        //    normally fills in ~21ms. If frames stop arriving (lock loss,
        //    end of burst), launch what we have after batch_timeout.
        {
            static auto batch_first_frame_time = std::chrono::steady_clock::now();
            static int prev_pending = 0;
            if (opt.multi_modcod()) {
                // Each MODCOD owns a separate micro-batch. Sparse application
                // traffic may therefore never fill a batch, and previously it
                // was released only when the periodic QPSK ACM feedback frame
                // changed chains. That coupled packet latency to the 500 ms
                // feedback interval at each endpoint. Use each chain's own
                // first-frame timestamp to launch stale work independently.
                if (fec_multi->launch_stale(batch_timeout))
                    timeout_flush_this_loop = true;
            } else {
                int pending = fec_single->pending_count();
                if (pending > 0 && prev_pending == 0)
                    batch_first_frame_time = std::chrono::steady_clock::now();
                prev_pending = pending;

                if (pending > 0 && !submitted_any) {
                    auto now = std::chrono::steady_clock::now();
                    if (now - batch_first_frame_time > batch_timeout) {
                        fec_single->launch_partial();
                        timeout_flush_this_loop = true;
                        prev_pending = 0;
                        drain_fec_output();
                    }
                }
            }
        }

        if (fec_trace.active()) {
            auto now = std::chrono::steady_clock::now();
            if (fec_trace.should_write(now, drained_this_loop,
                                       submitted_count,
                                       timeout_flush_this_loop)) {
                int pending = opt.multi_modcod() ? 0 : fec_single->pending_count();
                int inflight = opt.multi_modcod() ? 0 : fec_single->gpu_inflight();
                uint64_t raw_frames_now =
                    opt.multi_modcod() ? fec_multi->total_frame_count() : fec_single->frame_count();
                uint64_t raw_errors_now = opt.multi_modcod() ? fec_multi->total_error_count()
                                                             : fec_single->frame_error_count();
                fec_trace.write(
                    ctx.start_time, now, frame_queue.available(), pending,
                    inflight, submitted_count, drained_this_loop,
                    batch_ready_this_loop, timeout_flush_this_loop,
                    lat_fec_loop_us.load(std::memory_order_relaxed),
                    lat_gpu_drain_us.load(std::memory_order_relaxed),
                    lat_gpu_submit_us.load(std::memory_order_relaxed),
                    lat_fec_decode_us.load(std::memory_order_relaxed),
                    raw_frames_now, raw_errors_now,
                    rx_fec_frames.load(std::memory_order_relaxed),
                    rx_fec_errors.load(std::memory_order_relaxed),
                    ewma_fer.load(std::memory_order_relaxed),
                    gpu_force_drain_count.load(std::memory_order_relaxed));
            }
        }

        // 4. Brief yield if nothing to do
        if (!submitted_any && frame_queue.available() == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

}
