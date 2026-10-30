/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TX worker threads for dvbs2-uhd-trx: frame generation and UHD send.
 */

#include "trx/tx_threads.h"
#include "trx/plmod_pack.h"

#include "trx/acm_signaling.h"
#include "trx/bench_stats.h"
#include "trx/thread_util.h"
#include "trx/tx_underflow_trace_writer.h"

#include "cuda/bch_encode_kernel.cuh"
#include "cuda/cuda_util.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// ====================================================================
// Thread 1a: TX gen — generate PLFRAME IQ and write to TX ring
// Pipelined: gen runs ahead while send thread drains ring to UHD.
// Ring backpressure (blocking write) provides natural pacing.
// ====================================================================
void tx_gen_thread(trx_context& ctx)
{
    auto& opt = *ctx.opt;
    const fecframe_size_t framesize = ctx.framesize;
    const bool pool_enabled = ctx.acm_tx_enabled;
    const bool acm_tx_enabled = pool_enabled && !opt.manual_modcod;
    const unsigned int acm_max_dfl = ctx.acm_max_dfl;
    const unsigned int acm_max_xfec = ctx.acm_max_xfec;
    const unsigned int acm_max_plf = ctx.acm_max_plf;
    const bool use_farrow = ctx.use_farrow;
    const double hw_sps = ctx.hw_sps;
    auto& tx_fec = *ctx.tx_fec;
    auto& tx_fec_acm = *ctx.tx_fec_acm;
    auto& tx_framer = *ctx.tx_framer;
    auto& tx_frontend = *ctx.tx_frontend;
    auto& tx_enc = *ctx.tx_enc;
    auto& tx_pipeline = *ctx.tx_pipeline;
    auto& tx_farrow = *ctx.tx_farrow;
    auto& tap = *ctx.tap;
    auto& tap_pdus = *ctx.tap_pdus;
    auto& tx_ring = *ctx.tx_ring;
    auto& rx_frontend = *ctx.rx_frontend;
    std::mutex& cuda_mtx = *ctx.cuda_mtx;
    auto& tx_frames = ctx.stats.tx_frames;
    auto& tx_dummy_frames = ctx.stats.tx_dummy_frames;
    auto& ewma_fer = ctx.stats.ewma_fer;
    auto& rx_data_modcod = ctx.stats.rx_data_modcod;
    auto& acm_mc_fec_frames = ctx.stats.acm_mc_fec_frames;
    auto& acm_mc_fec_errors = ctx.stats.acm_mc_fec_errors;
    auto& acm_feedback_tx = ctx.stats.acm_feedback_tx;
    auto& acm_feedback_rx = ctx.stats.acm_feedback_rx;
    auto& acm_feedback_recovery_bursts =
        ctx.stats.acm_feedback_recovery_bursts;
    auto& sync_count_tx = ctx.stats.sync_count_tx;
    auto& acm_tx_modcod = ctx.stats.acm_tx_modcod;
    auto& acm_tx_data_modcod = ctx.stats.acm_tx_data_modcod;
    auto& acm_control_updates = ctx.stats.acm_control_updates;
    auto& lat_tx_ring_wait_us = ctx.stats.lat_tx_ring_wait_us;
    auto& lat_tx_loop_us = ctx.stats.lat_tx_loop_us;
    auto& lat_tx_gse_us = ctx.stats.lat_tx_gse_us;
    auto& lat_tx_bb_bch_us = ctx.stats.lat_tx_bb_bch_us;
    auto& lat_tx_ldpc_us = ctx.stats.lat_tx_ldpc_us;
    auto& lat_tx_batch_size = ctx.stats.lat_tx_batch_size;

    if (!opt.do_tx) return;
    if (opt.pin_threads)
        pin_thread("tx_gen", 0,
                   opt.tx_send_realtime || opt.realtime_threads);

    // Buffer sizes: use max across all ACM MODCODs, or single chain sizes
    const unsigned int max_dfl = pool_enabled ? acm_max_dfl : tx_fec->max_dfl_bytes();
    const unsigned int max_xfec = pool_enabled ? acm_max_xfec : tx_fec->xfecframe_len();
    const unsigned int max_plf = pool_enabled ? acm_max_plf : tx_framer->plframe_len();
    const unsigned int dummy_len = pl_framer::dummy_plframe_len();

    // Determine effective TX batch size
    int effective_batch = opt.tx_batch_size;
    if (tx_pipeline && effective_batch == 0) {
        // Auto: target 2ms of IQ per GPU batch.  Larger batches amortize
        // GPU launch overhead and build ring credit against contention
        // from RX LDPC streams.
        // Use the TRUE symbol rate: with the TX Farrow active the hardware
        // rate is NOT sps * sym_rate, and opt.rate/opt.sps overstates the
        // frame period by hw_sps/sps (half the intended batch at sps 2).
        double sym_rate = (opt.sym_rate_hz > 0.0) ? opt.sym_rate_hz
                                                  : opt.rate / opt.sps;
        double frame_period_us = (double)max_plf / sym_rate * 1e6;
        effective_batch = std::max(1, (int)std::ceil(2000.0 / frame_period_us));
        effective_batch = std::min(effective_batch, 64);
    }
    if (effective_batch < 1) effective_batch = 1;

    std::vector<uint8_t> gse_data(max_dfl);
    std::vector<uint8_t> tx_pending_pdu;
    std::vector<complexf> xfecframe(max_xfec);
    std::vector<complexf> plframe(std::max((unsigned int)dummy_len, max_plf));
    const int dummy_batch_frames = std::max(
        1, std::min(3, (int)(max_plf / dummy_len)));
    std::vector<complexf> dummy_symbols(
        (size_t)dummy_batch_frames * dummy_len);
    const int one_dummy_symbols = tx_framer->generate_dummy_frame(
        dummy_symbols.data());
    for (int i = 1; i < dummy_batch_frames; ++i) {
        std::copy_n(dummy_symbols.data(), one_dummy_symbols,
                    dummy_symbols.data() + (size_t)i * one_dummy_symbols);
    }
    const int dummy_symbols_len = dummy_batch_frames * one_dummy_symbols;
    bool in_dummy_run = false;
    bool first_application_burst = true;
    int burst_warmup_remaining = 0;
    // Exercise an unvalidated MODCOD only in short, data-free epochs.  A long
    // idle probation is unsafe: if the candidate is below its waterfall, its
    // frames consume the maximum LDPC iteration budget and can fill the GPU
    // flight queue before delayed GEO feedback returns.  Eight transmitted
    // frames leave useful outcomes after the PLS transition guard while
    // bounding both decoder work and application airtime.  The controller
    // accumulates clean evidence across epochs and rejects on the first NACK.
    constexpr int ACM_PROBE_EPOCH_FRAMES = 8;
    // The RX transition guard intentionally defers the first repeated PLS
    // headers of a new MODCOD.  Returning from a candidate to the validated
    // mode therefore also needs padding; otherwise the first application
    // BBFRAME becomes that guard and is silently discarded.
    constexpr int ACM_SAFE_SWITCH_WARMUP_FRAMES = 4;
    int acm_transition_warmup_remaining = 0;
    // An isolated robust frame immediately after a high-order PLFRAME is not
    // a robust control channel in practice: the RX transition/acquisition
    // path needs repeated headers before its first useful BBFRAME.  Prime each
    // report with a short padding preamble at the reserved signaling MODCOD,
    // then repeat the same report.  The repetition is intentional: X410 OTA
    // measurements show that the first few low-order frames after a high
    // APSK mode can be lost as one correlated transition event, so isolated
    // reports are not independent trials.
    constexpr int ACM_CONTROL_PREAMBLE_FRAMES = 4;
    constexpr int ACM_CONTROL_REPEAT_FRAMES = 4;
    bool acm_control_pending = false;
    int acm_control_preamble_remaining = 0;
    int acm_control_repeat_remaining = 0;
    uint8_t acm_candidate_mc = 0;
    bool acm_candidate_burst_active = false;
    bool acm_candidate_burst_started = false;
    bool acm_candidate_waiting_feedback = false;
    uint64_t acm_candidate_control_mark = 0;
    // Start with enough ordinary frames for frame-sync acquisition and FEC
    // bootstrap before entering the low-compute dummy-idle steady state.
    int idle_probe_remaining = opt.idle_dummy_frames ? opt.idle_probe_frames : 0;
    auto last_idle_probe = std::chrono::steady_clock::now();
    uint64_t last_feedback_rx = acm_feedback_rx.load(std::memory_order_relaxed);
    auto last_feedback_progress = std::chrono::steady_clock::now();

    // In-band ACM signaling: separate GSE encapsulator + timer
    gse_encapsulator sig_enc(ACM_SIG_PROTO, GSE_LT_BROADCAST);
    // Make the first startup probe carry feedback immediately. Subsequent
    // reports obey the configured cadence.
    auto last_sig_time = std::chrono::steady_clock::now() -
        std::chrono::hours(1);
    uint16_t sig_seq = 0;
    std::array<uint64_t, 256> acm_last_reported_frames{};
    std::array<uint64_t, 256> acm_last_reported_errors{};
    unsigned acm_feedback_scan = 0;
    std::vector<uint8_t> acm_control_frame_cache(max_dfl);
    unsigned int acm_control_frame_cache_dfl = 0;
    std::vector<complexf> iq_out(
        (size_t)effective_batch * max_plf * opt.sps + tx_frontend->ntaps() * 2);

    // Batch staging buffers (CUDA batched path)
    unsigned int max_nldpc_bytes = opt.short_frames ? 16200/8 : 64800/8;
    unsigned int max_kbch_bytes = max_nldpc_bytes;  // safe upper bound
    std::vector<uint8_t> bbframe_staging;
    std::vector<float> plh_staging;
    uint8_t* batch_fecframes_dev = nullptr;
    uint8_t* batch_bch_out_dev = nullptr;
    // Per-MODCOD GPU BCH encoders (lazy init), indexed by the uint8_t
    // modcod_idx (cur_mc). S2X modcods reach idx 33 (qpsk13/45), so a
    // [32] array OOB'd on 128apsk3/4 (idx 32) and qpsk13/45 — a garbage
    // unique_ptr -> SIGSEGV in encode_batch/ensure_batch_buffers, but
    // only on the GPU-BCH path (high frame rate). Size to the full
    // uint8_t index domain so no modcod can overflow it.
    std::unique_ptr<cuda::bch_bb_scramble_cuda> bch_cuda_arr[256];

    bool use_gpu_bch = (effective_batch >= GPU_BCH_MIN_BATCH);
    if (tx_pipeline && effective_batch > 1) {
        plh_staging.resize((size_t)effective_batch * 90 * 2);
        {
            std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
            CUDA_CHECK(cudaMalloc(&batch_fecframes_dev,
                                  (size_t)effective_batch * max_nldpc_bytes));
            if (use_gpu_bch) {
                CUDA_CHECK(cudaMalloc(&batch_bch_out_dev,
                                      (size_t)effective_batch * max_nldpc_bytes));
            }
        }
        if (use_gpu_bch) {
            // GPU BCH: staging = raw BBFRAMEs (kbch bytes)
            bbframe_staging.resize((size_t)effective_batch * max_kbch_bytes);
        } else {
            // CPU BCH: staging = BCH output (nbch bytes ≈ n_bits bytes)
            bbframe_staging.resize((size_t)effective_batch * max_nldpc_bytes);
        }
        fprintf(stderr, "dvbs2-uhd-trx: TX batch=%d (%s BCH)\n",
                effective_batch, use_gpu_bch ? "GPU" : "CPU");
    } else {
        fprintf(stderr, "dvbs2-uhd-trx: TX batch=%d (single-frame)\n",
                effective_batch);
    }

    // Encode one ACM report in a dedicated robust control BBFRAME.  Reports
    // must not ride the currently selected data/candidate MODCOD: that makes
    // the control loop least reliable precisely while it is testing a new
    // operating point.  The scheduler below emits this payload periodically
    // at SIG_MODCOD, including while application traffic is saturated.
    auto append_acm_feedback = [&](uint8_t* gse_buf, unsigned int dfl) {
        if (!acm_tx_enabled || !opt.do_rx || opt.acm_feedback_listen != 0)
            return 0;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(
                now - last_sig_time).count() < opt.acm_feedback_interval_ms)
            return 0;
        last_sig_time = now;

        static float held_snr = -99.0f;
        static int hold_cnt = 0;
        if (rx_frontend && rx_frontend->locked()) {
            const float control_snr = ctx.stats.acm_rx_snr_ewma.load(
                std::memory_order_acquire);
            float raw_coh = rx_frontend->pilot_coherence();
            if (raw_coh > 0.5f && std::isfinite(control_snr)) {
                held_snr = control_snr;
                hold_cnt = 0;
            } else if (hold_cnt < opt.acm_snr_hold_frames) {
                ++hold_cnt;
            } else if (std::isfinite(control_snr)) {
                held_snr = control_snr;
            }
        } else {
            held_snr = -99.0f;
        }

        acm_feedback_pkt_t fb{};
        fb.magic = ACM_FEEDBACK_MAGIC;
        fb.fer = ewma_fer.load(std::memory_order_relaxed);
        fb.snr_db = held_snr;
        const uint8_t data_mc =
            rx_data_modcod.load(std::memory_order_relaxed);
        // A short data-free candidate burst can end before the next 2 Hz
        // feedback opportunity. Report changed per-MODCOD counters rather
        // than blindly reporting only the mode present at this instant;
        // otherwise saturated-link probes are invisible to the controller.
        // Prioritize a new NACK, then a changed non-current mode, then the
        // current mode. Rotating the scan avoids starving simultaneous dirt.
        uint8_t report_mc = data_mc;
        int report_priority = -1;
        for (unsigned off = 0; off < 256; ++off) {
            const unsigned idx = (acm_feedback_scan + off) & 0xffu;
            const uint64_t frames =
                acm_mc_fec_frames[idx].load(std::memory_order_relaxed);
            const uint64_t errors =
                acm_mc_fec_errors[idx].load(std::memory_order_relaxed);
            if (frames < acm_last_reported_frames[idx] ||
                errors < acm_last_reported_errors[idx]) {
                acm_last_reported_frames[idx] = 0;
                acm_last_reported_errors[idx] = 0;
            }
            if (frames == acm_last_reported_frames[idx] &&
                errors == acm_last_reported_errors[idx])
                continue;
            const int priority = errors > acm_last_reported_errors[idx]
                ? 3 : (idx != data_mc ? 2 : 1);
            if (priority > report_priority) {
                report_priority = priority;
                report_mc = static_cast<uint8_t>(idx);
                if (priority == 3) break;
            }
        }
        fb.rx_modcod = report_mc;
        fb.seq = sig_seq++;
        fb.fec_frames =
            acm_mc_fec_frames[report_mc].load(std::memory_order_relaxed);
        fb.fec_errors =
            acm_mc_fec_errors[report_mc].load(std::memory_order_relaxed);
        int written = sig_enc.encapsulate(
            reinterpret_cast<uint8_t*>(&fb), sizeof(fb), gse_buf, dfl);
        if (written > 0) {
            acm_last_reported_frames[report_mc] = fb.fec_frames;
            acm_last_reported_errors[report_mc] = fb.fec_errors;
            acm_feedback_scan =
                (static_cast<unsigned>(report_mc) + 1) & 0xffu;
            acm_feedback_tx.fetch_add(1, std::memory_order_relaxed);
        }
        return written > 0 ? written : 0;
    };

    // Helper: fill one GSE frame's data from TAP queue
    auto fill_gse_frame = [&](uint8_t* gse_buf, unsigned int dfl) {
        int gse_pos = 0;
        if (!tx_enc) {
            if (gse_pos < (int)dfl)
                gse_pos += gse_encapsulator::write_padding(
                    gse_buf + gse_pos, dfl - gse_pos);
            return;
        }
        while (tx_enc->has_pending() && gse_pos + 4 <= (int)dfl) {
            int w = tx_enc->write_pending(gse_buf + gse_pos, dfl - gse_pos);
            if (w == 0) break;
            gse_pos += w;
        }
        if (!tx_pending_pdu.empty() && !tx_enc->has_pending()) {
            int w = tx_enc->encapsulate(tx_pending_pdu.data(),
                                         tx_pending_pdu.size(),
                                         gse_buf + gse_pos, dfl - gse_pos);
            if (w > 0) { gse_pos += w; tx_pending_pdu.clear(); }
            else if (w < 0) { tx_pending_pdu.clear(); }
        }
        while (tx_pending_pdu.empty() && !tx_enc->has_pending() &&
               gse_pos + 4 < (int)dfl) {
            std::vector<uint8_t> pdu;
            if (!tap_pdus.pop(pdu)) break;
            int w = tx_enc->encapsulate(pdu.data(), pdu.size(),
                                         gse_buf + gse_pos, dfl - gse_pos);
            if (w > 0) gse_pos += w;
            else if (w < 0) continue;
            else { tx_pending_pdu = std::move(pdu); break; }
        }
        if (gse_pos < (int)dfl)
            gse_pos += gse_encapsulator::write_padding(
                gse_buf + gse_pos, dfl - gse_pos);
    };
    auto fill_acm_control_frame = [&](uint8_t* gse_buf, unsigned int dfl) {
        int gse_pos = 0;
        if (acm_control_repeat_remaining == ACM_CONTROL_REPEAT_FRAMES) {
            gse_pos = append_acm_feedback(gse_buf, dfl);
            if (gse_pos < (int)dfl)
                gse_pos += gse_encapsulator::write_padding(
                    gse_buf + gse_pos, dfl - gse_pos);
            if (gse_pos <= 0) {
                acm_control_repeat_remaining = 0;
                acm_control_pending = false;
                return;
            }
            std::copy_n(gse_buf, dfl, acm_control_frame_cache.data());
            acm_control_frame_cache_dfl = dfl;
        } else if (acm_control_frame_cache_dfl == dfl) {
            std::copy_n(acm_control_frame_cache.data(), dfl, gse_buf);
            gse_pos = dfl;
        }

        if (acm_control_repeat_remaining > 0)
            --acm_control_repeat_remaining;
        if (acm_control_repeat_remaining == 0)
            acm_control_pending = false;
        if (gse_pos < (int)dfl)
            gse_encapsulator::write_padding(
                gse_buf + gse_pos, dfl - gse_pos);
    };
    auto fill_acm_control_preamble = [&](uint8_t* gse_buf,
                                         unsigned int dfl) {
        gse_encapsulator::write_padding(gse_buf, dfl);
        if (acm_control_preamble_remaining > 0)
            --acm_control_preamble_remaining;
    };
    auto fill_burst_frame = [&](uint8_t* gse_buf, unsigned int dfl) {
        if (acm_transition_warmup_remaining > 0) {
            // Confirm a selected ACM data-mode transition independently of
            // application traffic.  Do not drain TAP packets or control-plane
            // reports through an unvalidated MODCOD.
            gse_encapsulator::write_padding(gse_buf, dfl);
            --acm_transition_warmup_remaining;
        } else if (idle_probe_remaining > 0) {
            // Startup/recovery probes are robust ordinary padding frames.
            // Periodic ACM reports have their own SIG_MODCOD opportunity.
            gse_encapsulator::write_padding(gse_buf, dfl);
            --idle_probe_remaining;
        } else if (burst_warmup_remaining > 0) {
            gse_encapsulator::write_padding(gse_buf, dfl);
            --burst_warmup_remaining;
        } else {
            fill_gse_frame(gse_buf, dfl);
        }
    };

    // DAC buffer for Farrow TX resampling
    complexf* tx_dac_buf = nullptr;
    size_t tx_dac_capacity = 0;
    if (use_farrow && tx_farrow) {
        size_t max_iq = (size_t)effective_batch * max_plf * opt.sps +
                        tx_frontend->ntaps() * 2;
        tx_dac_capacity = (size_t)(max_iq * (hw_sps / opt.sps) + 200);
        // The Farrow kernel writes this buffer directly. Pageable std::vector
        // storage happens to be GPU-addressable through HMM/ATS on GH200 and
        // Spark, but it is not a valid CUDA kernel target on Jetson Orin.
        // Pinned host storage is mapped on every supported platform and lets
        // the CPU feed the TX ring without an extra device-to-host copy.
        CUDA_CHECK(cudaMallocHost(&tx_dac_buf,
                                  tx_dac_capacity * sizeof(complexf)));
    }

    // Helper: write IQ to TX ring with backpressure. If Farrow is active,
    // resample from internal SPS to hardware SPS first.
    auto write_to_ring = [&](const complexf* data, int len) {
        auto t_ring0 = std::chrono::steady_clock::now();
        const complexf* out_data = data;
        int out_len = len;

        if (use_farrow && tx_farrow) {
            int consumed;
            out_len = tx_farrow->resample(
                reinterpret_cast<const float*>(data), len,
                reinterpret_cast<float*>(tx_dac_buf),
                (int)tx_dac_capacity, consumed,
                tx_pipeline->stream());
            out_data = tx_dac_buf;
            // The resampler must consume the whole batch: we stream
            // contiguous buffers and never re-feed a tail. A partial
            // consume means tx_dac_buf is undersized and waveform is
            // being DELETED at batch boundaries (TX timing splices).
            if (consumed != len) {
                static unsigned long drop_count = 0;
                if (++drop_count <= 10 || (drop_count % 1000 == 0))
                    fprintf(stderr, "dvbs2-uhd-trx: TX farrow dropped "
                            "%d/%d samples (#%lu) — DAC buf too small?\n",
                            len - consumed, len, drop_count);
            }
            if (opt.debug_level >= 2) {
                static unsigned long acct_count = 0;
                if ((++acct_count & 1023) == 1)
                    fprintf(stderr, "dvbs2-uhd-trx: TX farrow acct: "
                            "call=%lu in=%d out=%d consumed=%d\n",
                            acct_count, len, out_len, consumed);
            }
        }

        // TX peak diagnostic (CUSAT_TX_PEAKLOG=1): report peak |sample| vs the
        // DAC full scale (1.0 for UHD fc32). peak >= ~1.0 means the high-PAPR
        // waveform is clipping — the SNDR-killing effect behind high-modcod
        // acquisition failures. Read-only; use with CUSAT_TX_BACKOFF to tune.
        static const bool peaklog = getenv("CUSAT_TX_PEAKLOG") != nullptr;
        if (peaklog) {
            float peak = 0.0f; double p2 = 0.0;
            for (int i = 0; i < out_len; i++) {
                float m = std::abs(out_data[i]);
                if (m > peak) peak = m;
                p2 += (double)m * m;
            }
            static unsigned long pk_cnt = 0;
            if ((++pk_cnt & 255) == 1)
                fprintf(stderr, "dvbs2-uhd-trx: TX peak=%.3f rms=%.3f papr=%.1fdB "
                        "(full scale 1.0)\n", peak,
                        out_len ? std::sqrt(p2 / out_len) : 0.0,
                        (peak > 0 && p2 > 0) ? 20.0 * std::log10(peak /
                            std::sqrt(p2 / out_len)) : 0.0);
        }

        int written = 0;
        while (written < out_len && !g_stop.load(std::memory_order_relaxed)) {
            size_t n = tx_ring.write(out_data + written, out_len - written);
            written += n;
            if (written < out_len)
                std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        lat_tx_ring_wait_us.store(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_ring0).count(),
            std::memory_order_relaxed);
    };

    // Keep the deep ring available for application bursts, but do not fill
    // all of it with idle dummy IQ. Two small, batched dummy writes provide a
    // bounded generation/scheduling reserve in CCM. ACM additionally needs
    // enough credit for a MODCOD transition: simultaneous RX decoder work and
    // the first ordinary-frame kernels have produced 2--3 ms stalls OTA. A
    // 5 ms ACM-only floor prevents an RF underflow without penalizing CCM's
    // sparse-packet latency. No samples are skipped, so the RRC and
    // fractional-resampler state remain continuous at dummy/data transitions.
    const size_t dummy_iq_est = (size_t)std::ceil(
        dummy_symbols_len * (use_farrow ? hw_sps : (double)opt.sps)) + 200;
    const size_t acm_transition_reserve = acm_tx_enabled
        ? (size_t)std::ceil(opt.rate * 0.005) : 0;
    const size_t idle_dummy_reserve = std::min(
        ctx.tx_ring_cap,
        std::max(2 * dummy_iq_est, acm_transition_reserve));
    fprintf(stderr, "dvbs2-uhd-trx: TX idle reserve %.2f ms (%zu samples)%s\n",
            1e3 * idle_dummy_reserve / opt.rate, idle_dummy_reserve,
            acm_tx_enabled ? " [ACM transition-safe]" : "");
    auto application_ready = [&]() {
        return tap && tx_enc &&
            (!tap_pdus.empty() || !tx_pending_pdu.empty() ||
             tx_enc->has_pending());
    };

    // Track the waveform MODCOD separately from the selected data MODCOD.
    // Periodic ACM feedback temporarily uses SIG_MODCOD and must not look like
    // a data-mode transition.
    uint8_t cur_mc = acm_tx_modcod.load(std::memory_order_relaxed);
    uint8_t data_mc = cur_mc;
    ctx.stats.manual_applied_modcod.store(cur_mc);

    // Double-buffer state: GPU batch N runs while we write batch N-1
    // to the ring.  The sync for batch N happens at the top of the
    // *next* iteration, so the GPU has the full CPU-side GSE/BCH time
    // plus the ring-write time to finish — typically non-blocking.
    const float* prev_iq_ptr = nullptr; // managed-mem ptr from previous batch
    int prev_iq_len = 0;
    int prev_iq_frames = 0;    // frame count for tx_frames bookkeeping

    // Modulation-compressed fronthaul (CCM): per-frame word buffer + the
    // mapper interleave params (must match the GPU mapper bit-for-bit).
    auto modcomp_next_frame = std::chrono::steady_clock::now();
    std::vector<uint32_t> modcomp_words;
    std::vector<uint8_t> modcomp_fec;
    unsigned int mc_cols[8] = {0};
    int mc_stride = 1;
    if (opt.tx_modcomp) {
        const mapper &mm = (pool_enabled ? tx_fec_acm->get(cur_mc) : *tx_fec).get_mapper();
        mm.get_interleave_params(mc_cols, mc_stride);
        // prologue(3) + PLHEADER(90) + bit words. Worst case is 128APSK:
        // 9270 syms x 7 bits = 64890 bits -> 2028 words (> 2025 for others).
        modcomp_words.resize(3 + 90 + (9270 * 7 + 31) / 32 + 2);
        modcomp_fec.resize(8100);
    }
    bool prev_gpu_pending = false;

    auto tx_loop_prev = std::chrono::steady_clock::now();
    while (!g_stop.load(std::memory_order_relaxed)) {
        // --- Double-buffer: sync previous GPU batch & drain to ring ---
        if (prev_gpu_pending) {
            // Sync without mutex — uses TX pipeline's own NonBlocking
            // stream, safe without serialization. Holding mutex here
            // would block the frontend for the entire GPU wait.
            CUDA_CHECK(cudaStreamSynchronize(tx_pipeline->stream()));
            sync_count_tx.fetch_add(1, std::memory_order_relaxed);
            write_to_ring(reinterpret_cast<const complexf*>(prev_iq_ptr),
                          prev_iq_len);
            tx_frames.fetch_add(prev_iq_frames);
            prev_gpu_pending = false;
        }

        // Measure TX loop period
        auto tx_loop_now = std::chrono::steady_clock::now();
        lat_tx_loop_us.store(
            std::chrono::duration_cast<std::chrono::microseconds>(
                tx_loop_now - tx_loop_prev).count(),
            std::memory_order_relaxed);
        tx_loop_prev = tx_loop_now;

        // Sample TAP readiness before handling a transition.  This lets an
        // idle link spend more frames validating a candidate without adding
        // the same probation delay to an already-active application flow.
        bool app_ready = application_ready();

        // Separate safe data from upward exploration. Candidate bursts carry
        // padding, never application PDUs, and have the same bounded size
        // under idle and saturated traffic.
        // After each burst, return to the validated mode while GEO feedback
        // is in flight. A new control report permits another short burst if
        // the candidate still needs evidence, so a saturated link can learn
        // and upshift instead of waiting forever for an idle interval.
        uint8_t target_mc = acm_tx_modcod.load(std::memory_order_relaxed);
        uint8_t safe_mc = acm_tx_data_modcod.load(
            std::memory_order_relaxed);
        const uint64_t control_epoch = acm_control_updates.load(
            std::memory_order_acquire);
        if (target_mc == safe_mc) {
            acm_candidate_mc = target_mc;
            acm_candidate_burst_active = false;
            acm_candidate_burst_started = false;
            acm_candidate_waiting_feedback = false;
        } else {
            if (target_mc != acm_candidate_mc) {
                acm_candidate_mc = target_mc;
                acm_candidate_burst_active = true;
                acm_candidate_burst_started = false;
                acm_candidate_waiting_feedback = false;
            } else if (acm_candidate_burst_active &&
                       acm_candidate_burst_started &&
                       data_mc == acm_candidate_mc &&
                       acm_transition_warmup_remaining == 0) {
                acm_candidate_burst_active = false;
                acm_candidate_burst_started = false;
                acm_candidate_waiting_feedback = true;
                acm_candidate_control_mark = control_epoch;
            } else if (acm_candidate_waiting_feedback &&
                       control_epoch > acm_candidate_control_mark &&
                       data_mc == safe_mc &&
                       acm_transition_warmup_remaining == 0) {
                // Do not let an unrelated/newly-arrived feedback report
                // preempt the return guard after a candidate epoch.  The RX
                // needs three repeated PLS headers to confirm the safe mode;
                // restarting the candidate after only the first padding
                // frame leaves it logically on the candidate and causes the
                // first subsequent application frames to be discarded.
                acm_candidate_burst_active = true;
                acm_candidate_burst_started = false;
                acm_candidate_waiting_feedback = false;
            }
        }
        uint8_t desired_mc = acm_candidate_burst_active
            ? target_mc : safe_mc;
        if (acm_tx_enabled) {
            const bool candidate_probe = target_mc != safe_mc &&
                desired_mc == target_mc;
            if (desired_mc != data_mc) {
                data_mc = desired_mc;
                acm_transition_warmup_remaining = candidate_probe
                    ? ACM_PROBE_EPOCH_FRAMES
                    : ACM_SAFE_SWITCH_WARMUP_FRAMES;
                if (candidate_probe)
                    acm_candidate_burst_started = true;
            } else if (candidate_probe &&
                       !acm_candidate_burst_started) {
                acm_transition_warmup_remaining = ACM_PROBE_EPOCH_FRAMES;
                acm_candidate_burst_started = true;
            }
            if (cur_mc != data_mc) {
                cur_mc = data_mc;
                tx_framer->set_modcod(
                    cur_mc, framesize == FECFRAME_SHORT, opt.pilots);
            }
        }

        if (opt.manual_modcod) {
            const uint8_t requested = acm_tx_modcod.load(std::memory_order_acquire);
            if (cur_mc != requested) {
                cur_mc = data_mc = requested;
                tx_framer->set_modcod(cur_mc, false, opt.pilots);
                tx_fec_acm->get(cur_mc).get_mapper().get_interleave_params(mc_cols, mc_stride);
            }
            ctx.stats.manual_applied_modcod.store(cur_mc, std::memory_order_release);
        }

        // Application-idle gate. Dummy PLFRAMEs preserve RF timing and
        // synchronization but are rejected from the RX FEC path.
        const bool acm_transition_probe = acm_tx_enabled &&
            acm_transition_warmup_remaining > 0;
        bool release_data = app_ready || idle_probe_remaining > 0 ||
            acm_transition_probe;
        auto sched_now = std::chrono::steady_clock::now();

        // A receiver can start long after its peer and therefore miss that
        // peer's startup frames. Sparse isolated reports preserve an existing
        // lock, but are deliberately not relied upon for cold acquisition.
        // Until feedback is making progress, insert a short consecutive burst
        // of robust ordinary frames every three seconds. The same mechanism
        // repairs a feedback outage without a permanent padding stream.
        if (acm_tx_enabled) {
            const uint64_t feedback_rx =
                acm_feedback_rx.load(std::memory_order_relaxed);
            if (feedback_rx != last_feedback_rx) {
                last_feedback_rx = feedback_rx;
                last_feedback_progress = sched_now;
            }
            const double recovery_ms = std::max(
                3000.0, 6.0 * opt.acm_feedback_interval_ms);
            if (idle_probe_remaining == 0 &&
                std::chrono::duration<double, std::milli>(
                    sched_now - last_feedback_progress).count() >= recovery_ms) {
                idle_probe_remaining = opt.idle_probe_frames;
                release_data = true;
                last_feedback_progress = sched_now;
                acm_feedback_recovery_bursts.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }

        // In ACM mode a report must cross the reverse DVB link independently
        // of application load and of the data MODCOD under test.  Dummy
        // PLFRAMEs have no BBFRAME, while piggybacking on a candidate makes
        // the feedback path disappear when that candidate is marginal.  At
        // each deadline, reserve a short run of robust SIG_MODCOD ordinary
        // frames: a padding preamble establishes the transition, then the
        // same feedback report is repeated.  This applies under both idle
        // and saturated traffic.
        const bool acm_feedback_due = acm_tx_enabled && opt.do_rx &&
            opt.acm_feedback_listen == 0 &&
            std::chrono::duration<double, std::milli>(
                sched_now - last_sig_time).count() >=
                    opt.acm_feedback_interval_ms;
        if (acm_feedback_due && !acm_control_pending) {
            acm_control_pending = true;
            acm_control_preamble_remaining = ACM_CONTROL_PREAMBLE_FRAMES;
            acm_control_repeat_remaining = ACM_CONTROL_REPEAT_FRAMES;
            acm_control_frame_cache_dfl = 0;
        }
        const bool acm_control_preamble = acm_control_pending &&
            acm_control_preamble_remaining > 0;
        const bool acm_control_frame = acm_control_pending &&
            acm_control_preamble_remaining == 0;
        const bool acm_probe_frame =
            acm_tx_enabled && !acm_transition_probe &&
            idle_probe_remaining > 0;
        if (acm_control_preamble || acm_control_frame || acm_probe_frame) {
            release_data = true;
            // Critical return-channel signaling uses the reserved robust
            // QPSK 1/4 mode. The RX treats this as an isolated control frame,
            // so it does not perturb the accepted application MODCOD. The
            // next TX iteration restores the controller-selected MODCOD.
            if (cur_mc != SIG_MODCOD) {
                cur_mc = SIG_MODCOD;
                tx_framer->set_modcod(
                    cur_mc, framesize == FECFRAME_SHORT, opt.pilots);
            }
        }

        // DVB-S2 dummy PLFRAMEs preserve carrier and frame synchronization,
        // but contain neither pilots nor decodable payload.  A sparse normal
        // padding frame keeps channel/ACM estimates fresh during long idle
        // gaps at a tiny fraction of continuous-padding compute cost.
        if (opt.idle_dummy_frames && !acm_tx_enabled && !app_ready &&
            idle_probe_remaining == 0 &&
            opt.idle_probe_ms > 0.0 &&
            std::chrono::duration<double, std::milli>(
                sched_now - last_idle_probe).count() >= opt.idle_probe_ms) {
            idle_probe_remaining = opt.idle_probe_frames;
            release_data = true;
            last_idle_probe = sched_now;
        }

        // An application frame is itself an ordinary acquisition-capable
        // PLFRAME. Do not make it wait behind padding from a probe that was
        // scheduled while the TAP was idle. The configured transition
        // warm-up, if any, remains the explicit reliability guard.
        if (app_ready && idle_probe_remaining > 0)
            idle_probe_remaining = 0;

        if (opt.idle_dummy_frames && !opt.tx_modcomp && !release_data && tx_pipeline) {
            // Recheck TAP arrivals while waiting for room for another dummy.
            // This prevents idle waveform from consuming the burst-oriented
            // ring depth ahead of a newly arrived application packet.
            while (!g_stop.load(std::memory_order_relaxed) &&
                   tx_ring.available_read() + dummy_iq_est >
                       idle_dummy_reserve) {
                if (application_ready()) {
                    app_ready = true;
                    release_data = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }

        if (app_ready && release_data && in_dummy_run) {
            // A peer may have started after our one-time startup burst, and a
            // dummy-only stream cannot bootstrap its FEC decoder. Prime one
            // four-frame micro-batch before the first application packet.
            // Later bursts use only the explicitly configured transition
            // warm-up (normally zero), preserving sparse-traffic latency.
            constexpr int INITIAL_APP_WARMUP_FRAMES = 4;
            burst_warmup_remaining = first_application_burst
                ? std::max(opt.tx_warmup_frames,
                           INITIAL_APP_WARMUP_FRAMES)
                : opt.tx_warmup_frames;
            first_application_burst = false;
        }

        if (opt.idle_dummy_frames && !opt.tx_modcomp && !release_data && tx_pipeline) {
            // The symbols repeat, but their IQ representation generally does
            // not: at fractional SPS the resampler phase advances from one
            // PLFRAME to the next. Replaying a one-frame IQ cache changes the
            // long-term sample rate and leaves the pipeline phase inconsistent
            // when an ordinary frame follows. Keep advancing the stateful TX
            // pipeline; dummy idle still bypasses GSE, BCH, LDPC and mapping.
            int dummy_iq_len;
            {
                std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                dummy_iq_len = tx_pipeline->process_symbols(
                    reinterpret_cast<const float*>(dummy_symbols.data()),
                    dummy_symbols_len, tx_pipeline->stream());
            }
            CUDA_CHECK(cudaStreamSynchronize(tx_pipeline->stream()));
            sync_count_tx.fetch_add(1, std::memory_order_relaxed);
            const complexf* iq = reinterpret_cast<const complexf*>(
                tx_pipeline->iq_managed());
            write_to_ring(iq, dummy_iq_len);
            in_dummy_run = true;
            tx_dummy_frames.fetch_add(
                dummy_batch_frames, std::memory_order_relaxed);
            continue;
        }
        in_dummy_run = false;

        // Get the right FEC chain for current MODCOD
        tx_fec_chain *cur_fec = pool_enabled ? &tx_fec_acm->get(cur_mc) : tx_fec.get();
        unsigned int cur_dfl = cur_fec->max_dfl_bytes();
        // --- Modulation-compressed fronthaul: host packs prologue +
        // PLHEADER + constellation-index bits; the FPGA dvb_plmod block
        // maps/pilots/scrambles. Entirely CPU (no GPU pipeline). ---
        if (opt.tx_modcomp) {
            // Compressed transport can buffer many seconds of RF airtime.
            // Produce at the on-air cadence so padding cannot delay new PDUs.
            const auto now = std::chrono::steady_clock::now();
            const double symbol_rate = opt.sym_rate_hz > 0 ? opt.sym_rate_hz : opt.rate / opt.sps;
            const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(tx_framer->plframe_len() / symbol_rate));
            const auto deadline = modcomp_next_frame - 16 * period;
            if (deadline > now) std::this_thread::sleep_until(deadline);
            modcomp_next_frame = std::max(modcomp_next_frame, now) + period;
            auto tx_gen_start = std::chrono::steady_clock::now();
            fill_gse_frame(gse_data.data(), cur_dfl);
            cur_fec->process_gse_to_bch(gse_data.data(), cur_dfl);
            {
                std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                auto stream = tx_pipeline->stream();
                cur_fec->encode_ldpc_to_device(tx_pipeline->fecframe_device(), stream);
                CUDA_CHECK(cudaMemcpyAsync(modcomp_fec.data(), tx_pipeline->fecframe_device(),
                    cur_fec->fecframe_bits() / 8, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            size_t nw = plmod_build_frame(
                modcomp_words.data(), cur_mc, opt.pilots,
                tx_framer->plheader(), modcomp_fec.data(),
                cur_fec->fecframe_bits(), mc_cols, mc_stride, opt.tx_backoff);
            if (nw & 1)
                modcomp_words[nw++] = 0;   // pad word; plmod resync discards
            write_to_ring(reinterpret_cast<const complexf*>(
                              modcomp_words.data()), (int)(nw / 2));
            tx_frames.fetch_add(1);
            lat_tx_gse_us.store(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tx_gen_start).count(),
                std::memory_order_relaxed);
            continue;
        }

        // --- Generate one PLFRAME ---
        int iq_len = 0;

        if (tap && tx_enc) {
            auto tx_gen_start = std::chrono::steady_clock::now();
            if (tx_pipeline && effective_batch > 1 && use_gpu_bch) {
                // ===== BATCHED GSE + GPU BCH + CUDA PATH =====
                // CPU: GSE→BB header only (no scramble/BCH).
                // GPU: BB scramble + BCH → LDPC → mapper → PL → RRC.
                int frames_in_batch = 0;
                unsigned int kb = cur_fec->kbch_bytes();

                // Lazy-init GPU BCH for this MODCOD
                if (!bch_cuda_arr[cur_mc]) {
                    std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                    bch_cuda_arr[cur_mc] = make_bch_cuda(*cur_fec);
                }

                const int frame_limit = acm_control_frame ? 1 :
                    (acm_control_preamble ? std::min(
                        effective_batch,
                        acm_control_preamble_remaining) :
                    (acm_probe_frame ? std::min(
                        effective_batch, idle_probe_remaining) :
                     (acm_transition_probe ? std::min(
                        effective_batch,
                        acm_transition_warmup_remaining) : effective_batch)));
                for (int bi = 0; bi < frame_limit &&
                     !g_stop.load(std::memory_order_relaxed); bi++) {
                    if (bi > 0 && acm_tx_enabled) {
                        const bool now_ready = application_ready();
                        uint8_t scheduled = now_ready
                            ? acm_tx_data_modcod.load(std::memory_order_relaxed)
                            : acm_tx_modcod.load(std::memory_order_relaxed);
                        if (scheduled != data_mc) break;
                    }

                    if (acm_control_preamble)
                        fill_acm_control_preamble(gse_data.data(), cur_dfl);
                    else if (acm_control_frame)
                        fill_acm_control_frame(gse_data.data(), cur_dfl);
                    else
                        fill_burst_frame(gse_data.data(), cur_dfl);
                    cur_fec->process_gse_to_bbframe(gse_data.data(), cur_dfl);

                    memcpy(bbframe_staging.data() + bi * kb,
                           cur_fec->bbframe(), kb);
                    memcpy(plh_staging.data() + bi * 90 * 2,
                           reinterpret_cast<const float*>(
                               tx_framer->plheader()),
                           90 * 2 * sizeof(float));

                    frames_in_batch++;
                }

                auto t_gse_end = std::chrono::steady_clock::now();

                if (frames_in_batch > 0) {
                    std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                    cudaStream_t s = tx_pipeline->stream();
                    bch_cuda_arr[cur_mc]->encode_batch(
                        bbframe_staging.data(), batch_bch_out_dev,
                        frames_in_batch, s);
                    cur_fec->encode_ldpc_batch_d2d(
                        batch_bch_out_dev, batch_fecframes_dev,
                        frames_in_batch, s);
                    iq_len = tx_pipeline->process_batch(
                        cur_mc, batch_fecframes_dev,
                        plh_staging.data(),
                        reinterpret_cast<float*>(iq_out.data()),
                        frames_in_batch, s);
                    // No sync — deferred to next iteration (double-buffer)
                    lat_tx_batch_size.store(frames_in_batch, std::memory_order_relaxed);
                }

                lat_tx_gse_us.store(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        t_gse_end - tx_gen_start).count(),
                    std::memory_order_relaxed);

                // Defer sync+ring-write to next iteration
                prev_iq_ptr = tx_pipeline->batch_iq_managed();
                prev_iq_len = iq_len;
                prev_iq_frames = frames_in_batch;
                prev_gpu_pending = (frames_in_batch > 0);
                continue;
            } else if (tx_pipeline && effective_batch > 1 && !use_gpu_bch) {
                // ===== BATCHED GSE + CPU BCH + CUDA PATH =====
                // CPU: GSE→BB scramble→BCH encode.
                // GPU: LDPC → mapper → PL → RRC.
                int frames_in_batch = 0;
                unsigned int nb = cur_fec->bch_len_bytes();

                const int frame_limit = acm_control_frame ? 1 :
                    (acm_control_preamble ? std::min(
                        effective_batch,
                        acm_control_preamble_remaining) :
                    (acm_probe_frame ? std::min(
                        effective_batch, idle_probe_remaining) :
                     (acm_transition_probe ? std::min(
                        effective_batch,
                        acm_transition_warmup_remaining) : effective_batch)));
                for (int bi = 0; bi < frame_limit &&
                     !g_stop.load(std::memory_order_relaxed); bi++) {
                    if (bi > 0 && acm_tx_enabled) {
                        const bool now_ready = application_ready();
                        uint8_t scheduled = now_ready
                            ? acm_tx_data_modcod.load(std::memory_order_relaxed)
                            : acm_tx_modcod.load(std::memory_order_relaxed);
                        if (scheduled != data_mc) break;
                    }

                    if (acm_control_preamble)
                        fill_acm_control_preamble(gse_data.data(), cur_dfl);
                    else if (acm_control_frame)
                        fill_acm_control_frame(gse_data.data(), cur_dfl);
                    else
                        fill_burst_frame(gse_data.data(), cur_dfl);
                    cur_fec->process_gse_to_bch(gse_data.data(), cur_dfl);

                    memcpy(bbframe_staging.data() + bi * nb,
                           cur_fec->bch_output(), nb);
                    memcpy(plh_staging.data() + bi * 90 * 2,
                           reinterpret_cast<const float*>(
                               tx_framer->plheader()),
                           90 * 2 * sizeof(float));

                    frames_in_batch++;
                }

                auto t_gse_end = std::chrono::steady_clock::now();

                if (frames_in_batch > 0) {
                    std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                    cudaStream_t s = tx_pipeline->stream();
                    cur_fec->encode_ldpc_batch_to_device(
                        bbframe_staging.data(), batch_fecframes_dev,
                        frames_in_batch, s);
                    iq_len = tx_pipeline->process_batch(
                        cur_mc, batch_fecframes_dev,
                        plh_staging.data(),
                        reinterpret_cast<float*>(iq_out.data()),
                        frames_in_batch, s);
                    // No sync — deferred to next iteration (double-buffer)
                    lat_tx_batch_size.store(frames_in_batch, std::memory_order_relaxed);
                }

                lat_tx_bb_bch_us.store(cur_fec->lat_bb_bch_us(),
                    std::memory_order_relaxed);
                lat_tx_ldpc_us.store(cur_fec->lat_ldpc_us(),
                    std::memory_order_relaxed);
                lat_tx_gse_us.store(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        t_gse_end - tx_gen_start).count(),
                    std::memory_order_relaxed);

                // Defer sync+ring-write to next iteration
                prev_iq_ptr = tx_pipeline->batch_iq_managed();
                prev_iq_len = iq_len;
                prev_iq_frames = frames_in_batch;
                prev_gpu_pending = (frames_in_batch > 0);
                continue;
            } else if (tx_pipeline) {
                // ===== SINGLE-FRAME GSE + CUDA PATH =====
                if (acm_control_preamble)
                    fill_acm_control_preamble(gse_data.data(), cur_dfl);
                else if (acm_control_frame)
                    fill_acm_control_frame(gse_data.data(), cur_dfl);
                else
                    fill_burst_frame(gse_data.data(), cur_dfl);
                auto t_gse_end = std::chrono::steady_clock::now();
                cur_fec->process_gse_to_bch(gse_data.data(), cur_dfl);
                {
                    std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                    cudaStream_t s = tx_pipeline->stream();
                    cur_fec->encode_ldpc_to_device(
                        tx_pipeline->fecframe_device(), s);
                    iq_len = tx_pipeline->process_device(
                        cur_mc, tx_pipeline->fecframe_device(),
                        reinterpret_cast<const float*>(tx_framer->plheader()),
                        reinterpret_cast<float*>(iq_out.data()), s);
                }
                // No sync — deferred to next iteration (double-buffer)
                lat_tx_batch_size.store(1, std::memory_order_relaxed);
                lat_tx_bb_bch_us.store(cur_fec->lat_bb_bch_us(),
                    std::memory_order_relaxed);
                lat_tx_ldpc_us.store(cur_fec->lat_ldpc_us(),
                    std::memory_order_relaxed);
                lat_tx_gse_us.store(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        t_gse_end - tx_gen_start).count(),
                    std::memory_order_relaxed);
            }
            // Defer sync+ring-write to next iteration
            prev_iq_ptr = tx_pipeline->iq_managed();
            prev_iq_len = iq_len;
            prev_iq_frames = 1;
            prev_gpu_pending = true;
        } else {
            // Keep radio-only and offline runs behaviorally identical to
            // TAP-backed links, including the dedicated robust ACM report.
            unsigned int cur_dfl = cur_fec->max_dfl_bytes();
            if (acm_control_preamble)
                fill_acm_control_preamble(gse_data.data(), cur_dfl);
            else if (acm_control_frame)
                fill_acm_control_frame(gse_data.data(), cur_dfl);
            else
                fill_burst_frame(gse_data.data(), cur_dfl);
            cur_fec->process_gse_to_bch(gse_data.data(), cur_dfl);
            {
                std::lock_guard<std::mutex> cuda_lock(cuda_mtx);
                cudaStream_t s = tx_pipeline->stream();
                cur_fec->encode_ldpc_to_device(
                    tx_pipeline->fecframe_device(), s);
                iq_len = tx_pipeline->process_device(
                    cur_mc, tx_pipeline->fecframe_device(),
                    reinterpret_cast<const float*>(tx_framer->plheader()),
                    reinterpret_cast<float*>(iq_out.data()), s);
            }
            // No sync — deferred to next iteration (double-buffer)
            prev_iq_ptr = tx_pipeline->iq_managed();
            prev_iq_len = iq_len;
            prev_iq_frames = 1;
            prev_gpu_pending = true;
        }
    }

    // Flush last pending GPU batch before thread exits
    if (prev_gpu_pending && tx_pipeline) {
        CUDA_CHECK(cudaStreamSynchronize(tx_pipeline->stream()));
        write_to_ring(reinterpret_cast<const complexf*>(prev_iq_ptr),
                      prev_iq_len);
        tx_frames.fetch_add(prev_iq_frames);
    }
    if (tx_dac_buf)
        cudaFreeHost(tx_dac_buf);
}

// ====================================================================
// Thread 1b: TX send — drain TX IQ ring to UHD
// Runs independently from generation. UHD normally provides transport
// backpressure; a radio-clock limiter remains authoritative if it does not.
// ====================================================================
void tx_send_thread(trx_context& ctx)
{
    auto& opt = *ctx.opt;
    radio_device& radio = *ctx.radio;
    const double stream_start_time = *ctx.stream_start_time;
    auto& tx_ring = *ctx.tx_ring;
    const size_t tx_ring_cap = ctx.tx_ring_cap;
    auto& tx_impairment = *ctx.tx_impairment;
    auto& tx_underflows = ctx.stats.tx_underflows;
    auto& tx_sequence_errors = ctx.stats.tx_sequence_errors;
    auto& tx_time_errors = ctx.stats.tx_time_errors;
    auto& txs_send_max_us = ctx.stats.txs_send_max_us;
    auto& txs_gap_max_us = ctx.stats.txs_gap_max_us;
    auto& txs_dur_over200 = ctx.stats.txs_dur_over200;
    auto& txs_gap_over200 = ctx.stats.txs_gap_over200;
    auto& txs_dur_over1ms = ctx.stats.txs_dur_over1ms;
    auto& txs_gap_over1ms = ctx.stats.txs_gap_over1ms;

    if (!opt.do_tx) return;
    if (opt.pin_threads)
        pin_thread("tx_send", opt.tx_send_slot,
                   opt.tx_send_realtime || opt.realtime_threads);

    const size_t spp = radio.tx_max_samps();
    const int chunk_mult = std::max(1, opt.tx_send_chunk);
    const size_t samps_per_send = spp * chunk_mult;
    std::vector<complexf> send_buf(samps_per_send);
    fprintf(stderr, "dvbs2-uhd-trx: tx_send: spp=%zu chunk=%zu (%.1f us/send @ %.2f Msps)\n",
            spp, samps_per_send, samps_per_send / opt.rate * 1e6, opt.rate / 1e6);

    // UHD normally paces the host through transport flow control.  Do not
    // rely on that as the sole rate limiter: an X4xx transport can continue
    // accepting UDP after its TX data path has stopped consuming at the
    // configured sample rate.  In that state the producer runs several times
    // faster than real time, fills the wire with samples the radio cannot
    // transmit, and the peer loses RF lock without a UHD async error.
    //
    // Pace against the radio clock, rather than assuming CLOCK_MONOTONIC and
    // the X4xx sample clock advance at exactly the same rate. Even a few ppm
    // of clock skew drains a fixed host-clock bucket during a long run. Keep
    // a small bounded credit, refresh the clock mapping at 10 Hz outside the
    // real-time sender, and rebuild only that credit after a genuine stall.
    // Other radio plugins retain their own pacing semantics.
    constexpr double PACE_AHEAD_SECONDS = 0.003;
    // X4xx get_time_now() is a network control transaction.  Most replies
    // arrive in well under a millisecond, but an occasional delayed reply
    // must never be treated as a fresh clock sample: pairing an old device
    // time with the post-call host timestamp makes the pacer stop TX for the
    // full RPC delay.  A few milliseconds are then enough to underflow the
    // radio; a longer MPM timeout can collapse both directions of a full-
    // duplex link.  Refreshing twice per second is ample for oscillator drift
    // and avoids needless control-plane traffic alongside the streamers.
    constexpr auto PACE_CLOCK_REFRESH = std::chrono::milliseconds(500);
    constexpr auto PACE_CLOCK_MAX_RTT = std::chrono::milliseconds(5);
    constexpr double PACE_CLOCK_MAX_STEP_SECONDS = 0.002;
    const bool software_pace = opt.radio_driver == "uhd";
    bool pace_armed = false;
    uint64_t pace_samples = 0;
    double pace_radio_epoch = stream_start_time;
    double pace_radio_anchor = 0.0;
    std::chrono::steady_clock::time_point pace_steady_anchor;
    std::chrono::steady_clock::time_point pace_next_refresh;
    std::mutex pace_clock_mtx;
    double sampled_radio_time = 0.0;
    std::chrono::steady_clock::time_point sampled_steady_time;
    std::thread pace_clock_thread;
    if (software_pace) {
        const auto before = std::chrono::steady_clock::now();
        sampled_radio_time = radio.get_time_now();
        const auto after = std::chrono::steady_clock::now();
        sampled_steady_time = before + (after - before) / 2;
        pace_clock_thread = std::thread([&]() {
            unsigned rejected_clock_samples = 0;
            while (!g_stop.load(std::memory_order_relaxed)) {
                try {
                    const auto before = std::chrono::steady_clock::now();
                    const double radio_time = radio.get_time_now();
                    const auto after = std::chrono::steady_clock::now();
                    const auto rtt = after - before;
                    const auto steady_time = before + rtt / 2;

                    bool accept = rtt <= PACE_CLOCK_MAX_RTT;
                    double clock_step = 0.0;
                    {
                        std::lock_guard<std::mutex> lock(pace_clock_mtx);
                        const double predicted = sampled_radio_time +
                            std::chrono::duration<double>(
                                steady_time - sampled_steady_time).count();
                        clock_step = radio_time - predicted;
                        accept = accept &&
                            std::abs(clock_step) <=
                                PACE_CLOCK_MAX_STEP_SECONDS;
                        if (accept) {
                            sampled_radio_time = radio_time;
                            sampled_steady_time = steady_time;
                        }
                    }
                    if (!accept) {
                        ++rejected_clock_samples;
                        if (rejected_clock_samples <= 3 ||
                            rejected_clock_samples % 100 == 0) {
                            fprintf(stderr,
                                    "dvbs2-uhd-trx: rejected delayed radio "
                                    "clock sample (rtt=%.2fms step=%.2fms)\n",
                                    std::chrono::duration<double, std::milli>(
                                        rtt).count(),
                                    clock_step * 1000.0);
                        }
                    }
                } catch (const std::exception& e) {
                    static unsigned clock_errors = 0;
                    if (++clock_errors <= 3)
                        fprintf(stderr,
                                "dvbs2-uhd-trx: radio clock sample failed: %s\n",
                                e.what());
                }
                std::this_thread::sleep_for(PACE_CLOCK_REFRESH);
            }
        });
        fprintf(stderr,
                "dvbs2-uhd-trx: UHD radio-clock pacing enabled (%.1f ms credit)\n",
                PACE_AHEAD_SECONDS * 1000.0);
    }
    auto read_sampled_clock = [&]() {
        std::lock_guard<std::mutex> lock(pace_clock_mtx);
        pace_radio_anchor = sampled_radio_time;
        pace_steady_anchor = sampled_steady_time;
    };

    // Stall diagnostics: ring of recent send timings, dumped at UF.
    // Single-writer; the per-second logger only reads the atomics above.
    struct send_rec { uint64_t t_end_us; uint32_t dur_us; uint32_t gap_us; uint32_t n; };
    constexpr unsigned NREC = 32;
    send_rec recent[NREC] = {};
    unsigned rec_idx = 0;
    const auto t_epoch = std::chrono::steady_clock::now();
    auto us_now = [&t_epoch]() -> uint64_t {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_epoch).count();
    };
    uint64_t t_prev_end_us = 0;
    auto bump_max = [](std::atomic<uint64_t>& a, uint64_t v) {
        if (v > a.load(std::memory_order_relaxed))
            a.store(v, std::memory_order_relaxed);
    };
    auto recent_summary = [&]() {
        struct summary {
            uint32_t max_dur_us = 0;
            uint32_t max_gap_us = 0;
            uint32_t over1ms = 0;
        } out;
        for (const auto& r : recent) {
            if (r.t_end_us == 0)
                continue;
            out.max_dur_us = std::max(out.max_dur_us, r.dur_us);
            out.max_gap_us = std::max(out.max_gap_us, r.gap_us);
            if (r.dur_us > 1000 || r.gap_us > 1000)
                out.over1ms++;
        }
        return out;
    };

    tx_underflow_trace_writer uf_trace;
    if (!uf_trace.open(opt.tx_underflow_trace_csv, "dvbs2-uhd-trx"))
        fprintf(stderr, "dvbs2-uhd-trx: TX underflow tracing disabled\n");

    radio_tx_meta md;
    md.start_of_burst = true;
    md.end_of_burst = false;
    md.has_time_spec = true;
    md.time_spec = stream_start_time;

    while (!g_stop.load(std::memory_order_relaxed)) {
        size_t n = tx_ring.read(send_buf.data(), samps_per_send);
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        if (tx_impairment) tx_impairment->apply(send_buf.data(), n);

        if (software_pace && pace_armed) {
            auto now = std::chrono::steady_clock::now();
            if (now >= pace_next_refresh) {
                read_sampled_clock();
                pace_next_refresh = now + PACE_CLOCK_REFRESH;
            }
            double radio_now = pace_radio_anchor +
                std::chrono::duration<double>(now - pace_steady_anchor).count();
            double end_radio_time = pace_radio_epoch +
                (double)(pace_samples + n) / opt.rate;
            double wait_seconds = end_radio_time - radio_now - PACE_AHEAD_SECONDS;
            if (wait_seconds > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(wait_seconds));
            } else if (wait_seconds < -PACE_AHEAD_SECONDS) {
                // The queued timeline is irrecoverably in the past. Reset the
                // logical epoch and permit only one bucket of immediate refill.
                pace_radio_epoch = radio_now;
                pace_samples = 0;
            }
        }

        const uint64_t t_start_us = us_now();
        size_t off = 0;
        radio_tx_meta send_md = md;
        while (off < n && !g_stop.load(std::memory_order_relaxed)) {
            size_t sent = radio.send(send_buf.data() + off, n - off, send_md, 3.0);
            if (sent == 0 && send_md.has_time_spec && off == 0) {
                // Timed start failed (clock issue) — fall back to immediate
                fprintf(stderr, "dvbs2-uhd-trx: TX timed start timeout, "
                        "falling back to immediate send\n");
                send_md.has_time_spec = false;
                continue;
            }
            if (sent == 0)
                break;
            off += sent;
            send_md.has_time_spec = false;
            send_md.start_of_burst = false;
        }
        if (off < n) {
            fprintf(stderr, "dvbs2-uhd-trx: TX short send dropped %zu/%zu samples\n",
                    n - off, n);
        }
        if (software_pace) {
            if (!pace_armed && off > 0) {
                pace_samples = off;
                // Queue the timed start metadata immediately, then map the
                // scheduled radio epoch onto steady time. The next iteration
                // waits until one credit window before RF actually starts.
                read_sampled_clock();
                pace_next_refresh = pace_steady_anchor + PACE_CLOCK_REFRESH;
                pace_armed = true;
            } else {
                pace_samples += off;
            }
        }
        if (off == 0 && md.has_time_spec) {
            // Timed start failed (clock issue) — fall back to immediate
            md.has_time_spec = false;
        }
        // After first send, switch to continuous (no timed sends)
        md.has_time_spec = false;
        md.start_of_burst = false;

        const uint64_t t_end_us = us_now();
        const uint32_t dur_us = (uint32_t)(t_end_us - t_start_us);
        const uint32_t gap_us = t_prev_end_us
            ? (uint32_t)(t_start_us - t_prev_end_us) : 0;
        t_prev_end_us = t_end_us;
        recent[rec_idx++ % NREC] = {t_end_us, dur_us, gap_us, (uint32_t)n};
        bump_max(txs_send_max_us, dur_us);
        bump_max(txs_gap_max_us, gap_us);
        if (dur_us > 200) {
            txs_dur_over200.fetch_add(1, std::memory_order_relaxed);
            if (dur_us > 1000)
                txs_dur_over1ms.fetch_add(1, std::memory_order_relaxed);
        }
        if (gap_us > 200) {
            txs_gap_over200.fetch_add(1, std::memory_order_relaxed);
            if (gap_us > 1000)
                txs_gap_over1ms.fetch_add(1, std::memory_order_relaxed);
        }

        // Poll for TX async messages (underflow detection).
        radio_async_msg async_md = radio.poll_tx_async();
        auto trace_event = [&](const char* event, uint64_t underflow_count,
                               std::chrono::steady_clock::time_point now) {
            if (!uf_trace.active())
                return;
            const auto rs = recent_summary();
            uf_trace.write(ctx, event, now, async_md, underflow_count,
                           n, dur_us, gap_us,
                           rs.max_dur_us, rs.max_gap_us, rs.over1ms,
                           tx_ring.available_read(), tx_ring_cap);
        };
        if (async_md.present) {
            if (async_md.underflow) {
                auto cnt = tx_underflows.fetch_add(1) + 1;
                trace_event("underflow", cnt, std::chrono::steady_clock::now());
                if (cnt <= 10 || (cnt % 100 == 0)) {
                    fprintf(stderr, "dvbs2-uhd-trx: TX UNDERFLOW #%lu "
                            "(ring=%zu/%zu) t=%.6fs dev_t=%.6fs\n",
                            (unsigned long)cnt,
                            tx_ring.available_read(), tx_ring_cap,
                            t_end_us / 1e6,
                            async_md.time_spec);
                    // Dump recent send timings, oldest first, to locate
                    // the stall (long dur = blocked in send()/flow
                    // control; long gap = thread preempted outside it)
                    fprintf(stderr, "dvbs2-uhd-trx:   recent sends "
                            "(t_end dur gap n):");
                    for (unsigned k = 0; k < NREC; k++) {
                        const auto& r = recent[(rec_idx + k) % NREC];
                        if (r.t_end_us == 0) continue;
                        if (r.dur_us < 150 && r.gap_us < 150) continue;
                        fprintf(stderr, " [%.6f %u %u %u]",
                                r.t_end_us / 1e6, r.dur_us, r.gap_us, r.n);
                    }
                    fprintf(stderr, "\n");
                }
            } else if (async_md.sequence_error) {
                auto cnt = tx_sequence_errors.fetch_add(1) + 1;
                trace_event("sequence_error",
                            tx_underflows.load(std::memory_order_relaxed),
                            std::chrono::steady_clock::now());
                if (cnt <= 10 || (cnt % 100 == 0)) {
                    fprintf(stderr,
                            "dvbs2-uhd-trx: TX SEQUENCE ERROR #%lu "
                            "(event=0x%x): host-to-radio packets were lost; "
                            "check MTU and Ethernet pause flow control\n",
                            (unsigned long)cnt, async_md.event_code);
                }
            } else if (async_md.time_error) {
                auto cnt = tx_time_errors.fetch_add(1) + 1;
                trace_event("time_error",
                            tx_underflows.load(std::memory_order_relaxed),
                            std::chrono::steady_clock::now());
                if (cnt <= 10 || (cnt % 100 == 0)) {
                    fprintf(stderr,
                            "dvbs2-uhd-trx: TX TIME ERROR #%lu "
                            "(event=0x%x): timed samples arrived late\n",
                            (unsigned long)cnt, async_md.event_code);
                }
            } else {
                static unsigned long other_evt = 0;
                trace_event("async", tx_underflows.load(std::memory_order_relaxed),
                            std::chrono::steady_clock::now());
                if (++other_evt <= 10 || (other_evt % 100 == 0))
                    fprintf(stderr, "dvbs2-uhd-trx: TX async event 0x%x "
                            "(#%lu) t=%.6fs\n", async_md.event_code,
                            other_evt, t_end_us / 1e6);
            }
        } else if (uf_trace.should_write(std::chrono::steady_clock::now())) {
            trace_event("periodic", tx_underflows.load(std::memory_order_relaxed),
                        std::chrono::steady_clock::now());
        }
    }

    if (pace_clock_thread.joinable())
        pace_clock_thread.join();

    // End of burst
    md.end_of_burst = true;
    md.has_time_spec = false;
    radio.send(nullptr, 0, md, 0.1);
}
