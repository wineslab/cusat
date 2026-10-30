/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone DVB-S2/S2X receiver frontend (see receiver_frontend.h).
 *
 * GPU orchestration of the receiver: AGC/rotation, timing recovery, batching
 * and GPU-CPU pipelining, bulk header search launch, lock validation, output
 * gates, unlock tiering and resets. Everything between "symbols out of timing
 * recovery" and "frames into the FEC chain" is done by pl_receiver.
 */

#include "receiver_frontend.h"
#include "acm_config.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "cuda/plheader_search_kernel.cuh"
#include "cuda/timing_recovery_kernel.cuh"
#include "cuda/agc_rot_kernel.cuh"

namespace cusat_dsp {

/* ---------- Constructor ---------- */

static filter_bank_design make_bank_design(const receiver_frontend_config& c)
{
    filter_bank_design d;
    d.samples_per_symbol = c.sps;
    d.rolloff = c.rolloff;
    d.span_symbols = c.rrc_delay;
    d.n_phases = c.n_phases;
    d.prototype = c.prefiltered ? bank_prototype::band_limited
                                : bank_prototype::rrc_matched;
    return d;
}

receiver_frontend::receiver_frontend(const receiver_frontend_config& cfg)
    : d_symbols_in(cfg.symbols_in),
      d_sps(cfg.sps),
      d_mu(0),
      d_rrc_bank(make_bank_design(cfg)),
      /* Inline rotator */
      d_rot_phase(1.0f, 0.0f),
      d_phase_inc(0.0f),
      d_total_cfo(0.0),
      d_debug_level(cfg.debug_level),
      d_frame_locked(false)
{
    const float sps = cfg.sps;
    const float rolloff = cfg.rolloff;
    const int debug_level = cfg.debug_level;

    // The oversampling ratio may be fractional (hardware rate / symbol
    // rate). The hard floor is the matched-filter Nyquist bound: the
    // complex signal spans (1 + rolloff) cycles per symbol, plus a small
    // margin for the timing estimator's eye contrast.
    if (!cfg.symbols_in && sps < 1.0f + rolloff + 0.05f) {
        throw std::runtime_error("sps below matched-filter minimum (1 + rolloff + margin)");
    }

    // History requirement: polyphase filter history + buffer margin for the
    // fractional NCO (the re-anchored base offset can reach -2 samples).
    d_ss_history = (unsigned)d_rrc_bank.history() +
                   (unsigned)std::ceil(d_sps / 2.0f) + 2;

    d_max_gpu_batch = (cfg.max_gpu_batch > 0) ? cfg.max_gpu_batch : 131072;
    // Offline A/B hook for the locked-path min-batch floor (gse-rx has no
    // sample clock, so the trx --rx-min-batch-us wiring can't apply there).
    // A set_min_batch_samples() call (trx) takes precedence over the env.
    if (const char* e = getenv("CUSAT_MIN_BATCH_SAMPLES"))
        d_min_batch_samples = std::max(0, atoi(e));

    // PL receiver: configuration errors propagate.
    {
        pl_rx_config pl_cfg = cfg.pl;
        pl_cfg.correction_latency_symbols = correction_latency_symbols();
        if (pl_cfg.debug_level == 0)
            pl_cfg.debug_level = cfg.debug_level;
        d_pl = new pl_receiver(pl_cfg);
        if (cfg.debug_level >= 2)
            d_pl->set_diagnostics(true);
    }
    {
        int max_search_syms = (int)(d_max_gpu_batch / d_sps) + 1000;
        d_hdr_search = new cuda::plheader_search_cuda(max_search_syms);
        d_gpu_overlap.resize(SEARCH_OVERLAP);
        d_gpu_search_buf.resize(max_search_syms);
        cudaMallocHost(&d_search_um, max_search_syms * 2 * sizeof(float));
    }

    // RU symbol timing bypasses host timing recovery and GPU AGC.
    if (!d_symbols_in) {
        int nf = d_rrc_bank.n_phases();
        int fl = d_rrc_bank.taps_per_phase();
        std::vector<float> flat_taps = d_rrc_bank.flat_taps();

        // Polyphase energy search timing recovery (fractional stride)
        d_timing_recovery_cuda = new cuda::timing_recovery_cuda(
            nf, fl, flat_taps.data(), (double)d_sps,
            d_rrc_bank.group_delay_samples(), d_max_gpu_batch);
        d_timing_recovery_cuda->set_estimation_period(4);
        // Disable per-kernel profiling events when not debugging —
        // saves ~20µs per call from eliminated CUDA API overhead.
        if (debug_level < 2)
            d_timing_recovery_cuda->set_profiling(false);

        // GPU fused AGC + frequency rotation (runs before timing kernel)
        d_agc_rot_cuda = new cuda::agc_rot_cuda(
            cfg.agc_ref, cfg.agc_initial_gain, cfg.agc_max_gain, cfg.agc_rate);
        if (debug_level < 2)
            d_agc_rot_cuda->set_profiling(false);

        // Pipeline profiling events
        cudaEventCreate((cudaEvent_t*)&d_evt_h2d_start);
        cudaEventCreate((cudaEvent_t*)&d_evt_h2d_done);
        cudaEventCreate((cudaEvent_t*)&d_evt_agc_done);
    }
}

receiver_frontend::~receiver_frontend()
{
    if (d_rx_diag_fp)
        fclose(d_rx_diag_fp);
    delete d_pl;
    delete d_agc_rot_cuda;
    delete d_timing_recovery_cuda;
    delete d_hdr_search;
    if (d_search_um) cudaFreeHost(d_search_um);
    if (d_evt_h2d_start) cudaEventDestroy((cudaEvent_t)d_evt_h2d_start);
    if (d_evt_h2d_done) cudaEventDestroy((cudaEvent_t)d_evt_h2d_done);
    if (d_evt_agc_done) cudaEventDestroy((cudaEvent_t)d_evt_agc_done);
}

/* ---------- Main process() ---------- */

int receiver_frontend::process(const complexf* in_raw,
                               int n_in,
                               complexf* out,
                               int max_out,
                               int& n_consumed,
                               std::vector<frontend_frame_t>& frames)
{
    int n_produced = 0;
    bool pl_sync_done = false;
    frames.clear();
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto t_agc = t0, t_timing = t0;

    // Deferred full reset after coherence unlock.  Must run BEFORE any
    // CUDA work and OUTSIDE any mutex scope — agc_rot_cuda->reset() may
    // touch the default CUDA stream, which implicitly synchronizes with
    // all other streams and can deadlock on GH200/ARM64 MIG if called
    // while the timing recovery mutex is held.
    if (d_needs_full_reset) {
        d_needs_full_reset = false;
        d_acq_calls_since_reset = 0;
        d_frame_locked = false;
        d_pl->reset(pl_reset_scope::full);
        d_total_cfo = 0.0;
        d_rot_phase = complexf(1.0f, 0.0f);
        d_phase_inc = 0.0f;
        d_mu = 0.0;
        d_gpu_overlap_valid = false;
        d_timing_init = false;
        d_timing_drift_accum = 0.0;
        d_timing_nco_base = 0.0;
        if (d_timing_recovery_cuda)
            d_timing_recovery_cuda->reset_estimation_counter();
        if (d_agc_rot_cuda)
            d_agc_rot_cuda->reset(1.0f);
        d_sym_pow = 0.0;
        d_prev_n_syms = 0;
    }

    // Acquisition timeout: if not locked for too long, full reset.
    // Clears stale AGC gain, frequency offset, and timing state that
    // accumulated on noise before the peer started transmitting.
    if (!d_frame_locked) {
        if (++d_acq_calls_since_reset >= ACQ_TIMEOUT_CALLS) {
            d_acq_calls_since_reset = 0;
            d_needs_full_reset = true;
            fprintf(stderr, "dvbs2-rx: acquisition timeout — full reset\n");
        }
    } else {
        d_acq_calls_since_reset = 0;
    }

    // Tick down soft unlock holdoff (set on lock transition)
    if (d_soft_unlock_holdoff > 0)
        d_soft_unlock_holdoff--;

    // When GPU timing recovery is active, limit input to what it can process
    // in one batch. This avoids wasteful rotation+AGC on samples that won't
    // be consumed (the caller will re-feed unconsumed samples next call).
    if (d_timing_recovery_cuda) {
        // Cap batch size to limit GPU stall per call. d_batch_plframes
        // controls how many PLFRAMEs are processed per call (default 2).
        // Higher values amortize per-call CUDA overhead at high sample
        // rates but increase timing loop update latency.
        //
        // ACQUISITION (not locked): cap to LESS THAN ONE PLFRAME, so each
        // bulk-search buffer holds at most one PLHEADER (observed at
        // 185-200 Msym: buffers spanning two headers never locked while the
        // same capture decoded clean offline where chunking differs). The
        // 2-PLFRAME cap then applies for throughput once locked.
        // Pre-lock there is no confirmed length. If a caller hinted the
        // expected PLFRAME length (set_acq_plframe_hint, used only for small
        // frames), size against it; otherwise keep the 33282 default. Without
        // this a short 32APSK PLFRAME (3402 sym, smallest in the system)
        // sized against 33282 spans ~10 frames per search batch -> no lock.
        // NOTE: deliberately NOT keyed on the valid-length whitelist (the
        // accept-logic list, which the OTA trx sets for ACM) — that would
        // change the normal-frame acquisition batch. The hint is set only in
        // short sessions.
        const uint32_t expected = d_pl->expected_plframe_len();
        int plf = expected > 0 ? (int)expected
                : (d_acq_plframe_hint > 0 ? (int)d_acq_plframe_hint : 33282);
        int batch_cap;
        if (!d_pl->locked()) {
            // The sub-PLFRAME invariant must hold for ALL plframe lengths:
            // a 32768-sample minimum here exceeded one PLFRAME for short-
            // PLFRAME modcods at fractional sps (16apsk@sps1.33 = 1.48
            // plframes -> two headers per search buffer -> lock lottery;
            // 32/64/128apsk marginal at 1.04-1.23). Keep only a small
            // floor against degenerate batches.
            batch_cap = std::max((int)std::ceil((plf - 256) * d_sps), 8192);
        } else {
            batch_cap = std::max((int)std::ceil(plf * d_sps) * d_batch_plframes,
                                 32768);
            // LOCKED: optional min-batch floor (see set_min_batch_samples).
            // Raises the per-batch real-time budget above the GPU sync wait
            // at high frame rates (small PLFRAMEs); clamped so the batch
            // still fits the preallocated GPU input buffer.
            if (d_min_batch_samples > 0)
                batch_cap = std::max(batch_cap,
                                     std::min(d_min_batch_samples,
                                              d_max_gpu_batch));
        }
        int max_useful = batch_cap + (int)d_ss_history + (int)std::ceil(d_sps) * 2;
        n_in = std::min(n_in, max_useful);
    }

    // --- GPU fused AGC + rotation + timing path ---
    const bool gpu_agc_rot_active =
        d_agc_rot_cuda && d_timing_recovery_cuda;

    // --- GPU-CPU pipeline: drain deferred symbols if we can't pipeline ---
    // Pipeline is only active for the locked timing recovery path. If we're
    // unlocked (fused search), drain any deferred symbols now.
    // Pipeline: overlap GPU batch N+1 with CPU PL processing of batch N.
    bool can_pipeline_now = gpu_agc_rot_active &&
        (d_timing_recovery_cuda && d_pl->locked());
    if (!can_pipeline_now && d_prev_n_syms > 0) {
        n_produced += run_pl_sync(
            d_prev_sym_buf.data(), d_prev_n_syms,
            out + n_produced, max_out - n_produced, nullptr, frames);
        d_prev_n_syms = 0;
    }

    const complexf* in = in_raw;
    t_agc = clk::now();

    // --- Symbol recovery ---
    if (n_in < (int)d_ss_history) {
        n_consumed = 0;
        return 0;
    }

    int max_syms = (int)(n_in / d_sps) + 2;
    if ((int)d_sym_buf.size() < max_syms) {
        d_sym_buf.resize(max_syms);
        d_prev_sym_buf.resize(max_syms);
    }

    int n_syms = 0;
    bool fused_search = false;

    // timing recovery sub-stage timing markers
    auto t_tr_memcpy = t_agc, t_tr_launch = t_agc, t_tr_sync = t_agc, t_tr_done = t_agc;
    if (d_symbols_in) {
        // Rung-3 symbols-in: each input sample is an RU-recovered on-time
        // symbol. AGC collapses to a unit-RMS scalar (EWMA of batch power —
        // the RU has no AGC, so the wire scale is arbitrary) and the inline
        // rotator still corrects residual carrier; both run here on the CPU.
        const int n_take = std::min(n_in, max_syms - 2);
        double p = 0.0;
        for (int i = 0; i < n_take; i++)
            p += (double)in_raw[i].real() * in_raw[i].real() +
                 (double)in_raw[i].imag() * in_raw[i].imag();
        p /= std::max(1, n_take);
        d_sym_pow = (d_sym_pow > 0.0) ? 0.9 * d_sym_pow + 0.1 * p : p;
        const float g = (d_sym_pow > 1e-20)
                      ? (float)(1.0 / std::sqrt(d_sym_pow)) : 1.0f;
        if (d_phase_inc == 0.0f &&
            d_rot_phase.real() == 1.0f && d_rot_phase.imag() == 0.0f) {
            for (int i = 0; i < n_take; i++)
                d_sym_buf[i] = in_raw[i] * g;
        } else {
            // Same rotation convention as rotate_gain_kernel:
            // y = x * (cos φ + j sin φ), φ = phase_start + n*phase_inc.
            const float ph = atan2f(d_rot_phase.imag(), d_rot_phase.real());
            for (int i = 0; i < n_take; i++) {
                float s, c;
                sincosf(ph + d_phase_inc * (float)i, &s, &c);
                const complexf& x = in_raw[i];
                d_sym_buf[i] = complexf((x.real() * c - x.imag() * s) * g,
                                        (x.real() * s + x.imag() * c) * g);
            }
            const float advance = d_phase_inc * n_take;
            d_rot_phase *= complexf(cosf(advance), sinf(advance));
            const float mag = std::abs(d_rot_phase);
            if (mag > 0) d_rot_phase /= mag;
        }
        n_syms = n_take;
        d_n_consumed = n_take;
    } else if (d_timing_recovery_cuda) {
        // Take the CUDA mutex for GPU kernel launches only — release before
        // the blocking cudaStreamSynchronize so the FEC thread can issue its
        // own CUDA calls during the ~150µs GPU wait. The sync and event reads
        // operate on the timing recovery's own stream/events and are safe
        // without the mutex.
        process_substage.store(30, std::memory_order_relaxed); // mutex
        std::unique_lock<std::mutex> timing_cuda_lock;
        if (d_cuda_mtx) {
            auto mtx_t0 = std::chrono::steady_clock::now();
            timing_cuda_lock = std::unique_lock<std::mutex>(*d_cuda_mtx);
            auto mtx_dt = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - mtx_t0).count();
            if (mtx_dt > 5.0) {
                fprintf(stderr, "dvbs2-rx: cuda_mtx WAIT in fe process: %.1f ms\n",
                        mtx_dt);
            }
        }

        bool skip_copy = false;
        int tr_n_in = n_in;
        unsigned tr_history = d_ss_history;

        if (gpu_agc_rot_active) {
            cudaStream_t stream = d_timing_recovery_cuda->stream();

            // Unified path for all sps: stage → AGC+rot → timing recovery/energy → interp
            int n_copy = std::min(n_in, d_timing_recovery_cuda->max_input());
            process_substage.store(35, std::memory_order_relaxed); // h2d
            // Profiling events: skip when debug_level < 2 to save ~15µs of
            // CUDA API overhead per call (~5µs × 3 cudaEventRecord).
            if (d_debug_level >= 2)
                cudaEventRecord((cudaEvent_t)d_evt_h2d_start, stream);
            float* samples_dev = d_timing_recovery_cuda->stage_input(
                reinterpret_cast<const float*>(in_raw), n_copy,
                d_input_pinned);
            if (d_debug_level >= 2)
                cudaEventRecord((cudaEvent_t)d_evt_h2d_done, stream);
            t_tr_memcpy = clk::now();

            process_substage.store(36, std::memory_order_relaxed); // agc+rot
            float phase_start = atan2f(d_rot_phase.imag(), d_rot_phase.real());
            d_agc_rot_cuda->process(samples_dev, n_copy,
                                     phase_start, d_phase_inc, stream);
            if (d_debug_level >= 2)
                cudaEventRecord((cudaEvent_t)d_evt_agc_done, stream);

            skip_copy = true;
        }

        // GPU feedforward timing recovery timing estimator
        cuda::timing_recovery_cuda::state_t tr_state;
        tr_state.tau = d_mu;
        tr_state.nco_base = d_timing_nco_base;
        tr_state.drift_accum = d_timing_drift_accum;
        tr_state.rate_corr = d_timing_rate_corr;
        tr_state.init = d_timing_init;

        // Freeze timing estimation during coherence drops to avoid
        // energy search mispicks on ISI-corrupted symbols.
        if (d_timing_recovery_cuda && d_low_coherence_streak > 0) {
            d_timing_recovery_cuda->set_estimation_period(9999);
        } else if (d_timing_recovery_cuda && d_low_coherence_streak == 0) {
            d_timing_recovery_cuda->set_estimation_period(d_timing_est_period_cfg);
        }

        int tr_consumed = 0;

        // Fused path: when not locked, launch the bulk header search on the
        // timing recovery stream before syncing — eliminates a second
        // cudaStreamSynchronize.
        fused_search = d_hdr_search && d_search_um
            && d_pl->wants_header_search();

        process_substage.store(31, std::memory_order_relaxed); // timing_launch
        if (fused_search) {
            d_pipeline_active = false;
            int max_possible = d_timing_recovery_cuda->process_launch(
                in, tr_n_in, max_syms, tr_state, tr_history, skip_copy);

            process_substage.store(37, std::memory_order_relaxed); // search_setup

            // Build search buffer in pinned memory:
            // [overlap | current symbols from symbols_um]
            int n_overlap = d_gpu_overlap_valid ? SEARCH_OVERLAP : 0;
            if (n_overlap > 0)
                memcpy(d_search_um, d_gpu_overlap.data(),
                       n_overlap * 2 * sizeof(float));

            // D2D copy: current symbols from timing recovery output to search buffer
            // (runs after batch_interp on same stream — serialized)
            if (max_possible > 0) {
                process_substage.store(38, std::memory_order_relaxed); // d2d+search
                cudaMemcpyAsync(
                    d_search_um + n_overlap * 2,
                    d_timing_recovery_cuda->symbols_um(),
                    max_possible * 2 * sizeof(float),
                    cudaMemcpyDeviceToDevice, d_timing_recovery_cuda->stream());

                // Launch the search on the timing recovery stream (no extra sync).
                // Candidate indices are relative to the batch start.
                d_hdr_search->launch(
                    d_search_um, n_overlap + max_possible, n_overlap,
                    d_timing_recovery_cuda->stream());
            }

            t_tr_launch = clk::now();

            // Release mutex before blocking sync — launches are enqueued;
            // sync uses our own NonBlocking stream.
            if (timing_cuda_lock.owns_lock()) timing_cuda_lock.unlock();
            process_substage.store(32, std::memory_order_relaxed); // sync
            n_syms = d_timing_recovery_cuda->process_complete(
                nullptr, max_syms, tr_state, tr_consumed);
            t_tr_sync = clk::now();

            // DON'T copy d_sym_buf: the PL receiver reads the pinned
            // symbols_um buffer directly (see below).
        } else if (can_pipeline_now) {
            // Locked path — GPU-CPU pipelined:
            // 1. Launch GPU batch (async)
            // 2. PL processing on PREVIOUS batch symbols (concurrent with GPU)
            // 3. Sync GPU
            // 4. Defer current symbols for next call's PL processing
            d_pipeline_active = true;
            d_saved_launch_phase_inc = d_phase_inc;
            process_substage.store(40, std::memory_order_relaxed); // pipeline launch
            d_timing_recovery_cuda->process_launch(
                in, tr_n_in, max_syms, tr_state, tr_history, skip_copy);
            process_substage.store(41, std::memory_order_relaxed); // pipeline pl_sync
            t_tr_launch = clk::now();

            // Release mutex for CPU PL processing — no CUDA calls here.
            if (timing_cuda_lock.owns_lock()) timing_cuda_lock.unlock();

            // Pipeline: PL processing on previous batch while GPU computes
            if (d_prev_n_syms > 0) {
                n_produced += run_pl_sync(
                    d_prev_sym_buf.data(), d_prev_n_syms,
                    out + n_produced, max_out - n_produced, nullptr, frames);
                d_prev_n_syms = 0;
            }

            // GPU sync — mutex released; sync uses our own NonBlocking stream.
            // (On GH200/ARM64 the CUDA driver hangs when concurrent CUDA API
            // calls enter from different threads, even on separate
            // NonBlocking streams.)
            process_substage.store(42, std::memory_order_relaxed); // pipeline sync
            n_syms = d_timing_recovery_cuda->process_complete(
                d_sym_buf.data(), max_syms, tr_state, tr_consumed);
            t_tr_sync = clk::now();

            d_mu = tr_state.tau;
            d_timing_nco_base = tr_state.nco_base;
            d_timing_drift_accum = tr_state.drift_accum;
            d_timing_rate_corr = tr_state.rate_corr;
            d_timing_init = tr_state.init;
            d_n_consumed = tr_consumed;

            // Advance rotator with SAVED phase_inc to maintain phase continuity
            // with what the GPU actually used for this batch.
            if (gpu_agc_rot_active && tr_consumed > 0) {
                float advance = d_saved_launch_phase_inc * tr_consumed;
                d_rot_phase *= complexf(cosf(advance), sinf(advance));
                float mag = std::abs(d_rot_phase);
                if (mag > 0) d_rot_phase /= mag;
            }

            // Defer current symbols for next call's PL processing
            std::swap(d_sym_buf, d_prev_sym_buf);
            d_prev_n_syms = n_syms;
            n_syms = 0;  // skip regular PL processing below
            pl_sync_done = true;
        } else {
            // Locked path — sequential (no pipelining).
            d_pipeline_active = false;
            d_timing_recovery_cuda->process_launch(
                in, tr_n_in, max_syms, tr_state, tr_history, skip_copy);
            t_tr_launch = clk::now();

            // Release mutex before sync — launches done, sync uses our stream.
            if (timing_cuda_lock.owns_lock()) timing_cuda_lock.unlock();
            n_syms = d_timing_recovery_cuda->process_complete(
                d_sym_buf.data(), max_syms, tr_state, tr_consumed);
            t_tr_sync = clk::now();
        }

        if (!pl_sync_done) {
            t_tr_done = clk::now();

            d_mu = tr_state.tau;
            d_timing_nco_base = tr_state.nco_base;
            d_timing_drift_accum = tr_state.drift_accum;
            d_timing_rate_corr = tr_state.rate_corr;
            d_timing_init = tr_state.init;
            d_n_consumed = tr_consumed;

            if (gpu_agc_rot_active && tr_consumed > 0) {
                float advance = d_phase_inc * tr_consumed;
                d_rot_phase *= complexf(cosf(advance), sinf(advance));
                float mag = std::abs(d_rot_phase);
                if (mag > 0) d_rot_phase /= mag;
            }
        } else {
            t_tr_done = clk::now();
        }
    }

    t_timing = clk::now();

    if (!pl_sync_done && n_syms <= 0) {
        n_consumed = 0;
        auto us = [](clk::time_point a, clk::time_point b) -> uint32_t {
            return (uint32_t)std::chrono::duration_cast<
                std::chrono::microseconds>(b - a).count();
        };
        d_perf = { us(t0, t_agc), us(t_agc, t_timing), 0,
                   us(t0, t_timing), n_in, 0, 0, fused_search };
        return 0;
    }

    // --- Feed symbols to the PL receiver and produce output ---
    process_substage.store(33, std::memory_order_relaxed); // pl_sync
    if (!pl_sync_done && n_syms > 0) {
        const complexf* syms = d_sym_buf.data();
        const pl_header_candidates* cand = nullptr;
        if (fused_search) {
            // Fused path: the search ran on the timing stream; the symbols
            // stay in the pinned symbols_um buffer.
            syms = (const complexf*)d_timing_recovery_cuda->symbols_um();
            d_candidates = d_hdr_search->result();
            cand = &d_candidates;
        } else if (d_hdr_search && d_pl->wants_header_search() &&
                   n_syms > SEARCH_OVERLAP) {
            // Self-contained search (symbols-in / non-fused path).
            int n_overlap = d_gpu_overlap_valid ? SEARCH_OVERLAP : 0;
            int total = n_overlap + n_syms;
            if ((int)d_gpu_search_buf.size() < total)
                d_gpu_search_buf.resize(total);
            if (n_overlap > 0)
                memcpy(d_gpu_search_buf.data(), d_gpu_overlap.data(),
                       n_overlap * sizeof(complexf));
            memcpy(d_gpu_search_buf.data() + n_overlap, syms,
                   n_syms * sizeof(complexf));
            std::unique_lock<std::mutex> cuda_lock;
            bool can_search = true;
            if (d_cuda_mtx) {
                // Try to acquire CUDA mutex without blocking. If the FEC
                // thread holds it (e.g. draining in-flight LDPC), skip the
                // search for this batch rather than stalling the frontend
                // and causing ring drops.
                cuda_lock = std::unique_lock<std::mutex>(*d_cuda_mtx, std::try_to_lock);
                can_search = cuda_lock.owns_lock();
            }
            if (can_search) {
                d_candidates = d_hdr_search->search(d_gpu_search_buf.data(),
                                                    total, n_overlap);
            } else {
                d_candidates = pl_header_candidates{};
                d_candidates.searched = n_syms; // searched-empty: skip batch
            }
            cand = &d_candidates;
        }
        if (cand && n_syms >= SEARCH_OVERLAP) {
            // Save overlap for next batch
            memcpy(d_gpu_overlap.data(), syms + (n_syms - SEARCH_OVERLAP),
                   SEARCH_OVERLAP * sizeof(complexf));
            d_gpu_overlap_valid = true;
        }
        n_produced += run_pl_sync(
            syms, n_syms, out + n_produced, max_out - n_produced,
            cand, frames);
    }

    n_consumed = d_n_consumed;

    // --- Per-call timing/AGC state trace (CUSAT_RX_STATELOG=path) ---
    // Dumps the tracking state on every locked call so periodic limit cycles
    // (e.g. integer-stride timing wander) become visible offline-vs-live.
    // One row/call; columns in the header line below.
    if (const char* slp = getenv("CUSAT_RX_STATELOG")) {
        static FILE* sf = []{
            FILE* f = fopen(getenv("CUSAT_RX_STATELOG"), "w");
            if (f) fprintf(f, "# callno n_in n_syms mu nco_base drift_accum "
                              "rate_corr low_coh_streak coh_raw agc_gain\n");
            return f;
        }();
        static long callno = 0;
        if (sf && d_pl->locked()) {
            fprintf(sf, "%ld %d %d %.6f %.6f %.9f %.4e %d %.4f %.5f\n",
                    callno, n_in, (n_syms > 0 ? n_syms : d_prev_n_syms),
                    d_mu, d_timing_nco_base, d_timing_drift_accum,
                    d_timing_rate_corr, d_low_coherence_streak,
                    d_pl->coherence_instant(),
                    d_agc_rot_cuda ? d_agc_rot_cuda->gain() : 0.0f);
            if ((callno & 0x3f) == 0) fflush(sf);
        }
        callno++;
        (void)slp;
    }

    // --- Fill performance counters ---
    process_substage.store(34, std::memory_order_relaxed); // perf
    {
        auto t_end = clk::now();
        auto us = [](clk::time_point a, clk::time_point b) -> uint32_t {
            return (uint32_t)std::chrono::duration_cast<
                std::chrono::microseconds>(b - a).count();
        };
        d_perf.agc_us = us(t0, t_agc);
        d_perf.timing_us = us(t_agc, t_timing);
        d_perf.sync_us = us(t_timing, t_end);
        d_perf.total_us = us(t0, t_end);
        d_perf.n_in = n_in;
        d_perf.n_syms = n_syms > 0 ? n_syms : d_prev_n_syms;
        d_perf.n_frames = (int)frames.size();
        d_perf.fused = fused_search;
        d_perf.tr_memcpy_us = us(t_agc, t_tr_memcpy);
        d_perf.tr_launch_us = us(t_tr_memcpy, t_tr_launch);
        d_perf.tr_sync_us = us(t_tr_launch, t_tr_sync);
        d_perf.tr_readback_us = us(t_tr_sync, t_tr_done);
        if (d_timing_recovery_cuda) {
            // Plain member reads (no CUDA API) — safe without the CUDA mutex.
            d_perf.tr_lnch_est_us =
                d_timing_recovery_cuda->last_launch_est_us();
            d_perf.tr_lnch_interp_us =
                d_timing_recovery_cuda->last_launch_interp_us();
        }
        // Per-kernel GPU timing: only read when profiling events were recorded
        // (debug_level >= 2). Skipping saves ~30µs per call from eliminated
        // CUDA API overhead (mutex + cudaEventElapsedTime × 4).
        if (d_debug_level >= 2) {
            std::unique_lock<std::mutex> perf_cuda_lock;
            if (d_cuda_mtx)
                perf_cuda_lock = std::unique_lock<std::mutex>(*d_cuda_mtx);
            if (d_timing_recovery_cuda) {
                d_perf.gpu_timing_us = d_timing_recovery_cuda->last_timing_us();
                d_perf.gpu_interp_us = d_timing_recovery_cuda->last_interp_us();
            }
            if (d_evt_h2d_start && d_evt_h2d_done) {
                float ms_h2d = 0, ms_agc = 0;
                cudaEventElapsedTime(&ms_h2d, (cudaEvent_t)d_evt_h2d_start,
                                     (cudaEvent_t)d_evt_h2d_done);
                d_perf.gpu_h2d_us = (uint32_t)(ms_h2d * 1000.0f);
                if (d_evt_agc_done) {
                    cudaEventElapsedTime(&ms_agc, (cudaEvent_t)d_evt_h2d_done,
                                         (cudaEvent_t)d_evt_agc_done);
                    d_perf.gpu_agc_rot_us = (uint32_t)(ms_agc * 1000.0f);
                }
            }
            if (d_agc_rot_cuda && gpu_agc_rot_active) {
                d_agc_rot_cuda->query_events();
                d_perf.gpu_agc_only_us = d_agc_rot_cuda->last_agc_us();
                d_perf.gpu_rot_only_us = d_agc_rot_cuda->last_rot_us();
            }
        }
    }

    process_substage.store(39, std::memory_order_relaxed); // done
    return n_produced;
}

/* ---------- PL receiver integration ---------- */

int receiver_frontend::run_pl_sync(const complexf* syms, int n_syms,
                                   complexf* out, int max_out,
                                   const pl_header_candidates* candidates,
                                   std::vector<frontend_frame_t>& frames)
{

    d_pl_frames.clear();
    d_pl_events.clear();
    const int n_out = d_pl->process(syms, n_syms, candidates, out, std::max(0, max_out),
                                    d_pl_frames, d_pl_events);
    handle_sync_events(d_pl_events);
    for (const pl_frame& f : d_pl_frames) {
        frontend_frame_t frame;
        frame.payload = f.data;
        frame.pls = f.layout;
        frame.payload_len = f.n_data;
        frame.pilot_snr_db = f.snr_valid ? f.snr_db : d_pl->snr_db();
        if (!gate_completed_frame(f, frame, frames))
            break;
    }
    double f_new = 0.0;
    if (d_pl->take_carrier_update(f_new)) {
        // The rotators multiply sample n by exp(j (phase_start + n phase_inc))
        // (agc_rot_kernel.cu, symbols-in branch of process()); removing a
        // carrier of f cycles/symbol at sps samples/symbol needs
        // phase_inc = -2 pi f / sps (symbols-in: one sample per symbol).
        d_total_cfo = f_new;
        const double sps = d_symbols_in ? 1.0 : (double)d_sps;
        d_phase_inc = (float)(-2.0 * M_PI * f_new / sps);
    }
    return n_out;
}

uint32_t receiver_frontend::correction_latency_symbols() const
{
    // Locked batches hold up to d_batch_plframes maximum PLFRAMEs (at least
    // 32768 samples, or the min-batch floor), one batch is pipelined on the
    // GPU, and the PL receiver decides a frame once the next header has
    // arrived: latency <= 2 batches + 1 PLFRAME.
    const double sps = d_symbols_in ? 1.0 : std::max(1.0, (double)d_sps);
    const double plf = (double)pl::max_plframe_symbols();
    double batch = std::max(plf * d_batch_plframes, 32768.0 / sps);
    batch = std::max(batch, d_min_batch_samples / sps);
    if (!d_symbols_in)
        batch = std::min(batch, std::max(plf, d_max_gpu_batch / sps));
    return (uint32_t)std::ceil(2.0 * batch + plf);
}

void receiver_frontend::update_correction_latency()
{
    if (d_pl)
        d_pl->set_correction_latency(correction_latency_symbols());
}

/* ---------- Lock validation and output gates (retained) ---------- */

void receiver_frontend::schedule_validation_reset()
{
    d_pl->reset(pl_reset_scope::sync);
    d_frame_locked = false;
    d_soft_unlock_cnt = 0;
    d_frames_since_lock = 0;
    d_low_coherence_cnt = 0;
    d_low_coherence_streak = 0;
    d_good_coherence_run = 0;
    d_coherence_confirmed = false;
    d_plf_idx = 0;
    d_plf_count = 0;
    d_needs_full_reset = true;
}

/* Returns false when a reset was scheduled: the caller drops the remaining
 * frames of the batch. */
bool receiver_frontend::gate_completed_frame(const pl_frame& f,
                                             frontend_frame_t& frame,
                                             std::vector<frontend_frame_t>& frames)
{
    if (d_diag_rx) {
        if (!d_rx_diag_fp)
            d_rx_diag_fp = fopen("/tmp/rxdiag.log", "w");
        if (d_rx_diag_fp) {
            float coh = d_pl->coherence();
            float snr = d_pl->snr_db();
            bool at_transition = (coh < 0.5f && d_rx_diag_prev_coh >= 0.5f) ||
                                 (coh >= 0.5f && d_rx_diag_prev_coh < 0.5f);
            bool periodic = (d_rx_diag_frame % 200 == 0);
            bool low_streak = (d_low_coherence_streak > 0);
            if (at_transition || periodic || low_streak ||
                d_rx_diag_frame < 100) {
                fprintf(d_rx_diag_fp,
                    "F[%d] sof=%lu coh=%.3f raw=%.3f snr=%.1f "
                    "tau=%.4f rate_ppm=%.4f cum_fo=%.8f pinc=%.9f "
                    "resid=%.8f agc=%.1f "
                    "low_streak=%d pls_dis=%d locked=%d\n",
                    d_rx_diag_frame,
                    (unsigned long)f.sof_symbol,
                    coh, f.coherence, snr,
                    d_mu, d_timing_rate_corr * 1e6,
                    d_total_cfo, d_phase_inc,
                    f.residual_cfo,
                    d_agc_rot_cuda ? d_agc_rot_cuda->gain() : 0.0f,
                    d_low_coherence_streak, d_pl->pls_disagreement_run(),
                    (int)d_frame_locked);
                fflush(d_rx_diag_fp);
            }
            d_rx_diag_prev_coh = coh;
            d_rx_diag_frame++;
        }
    }

    // Lock validation: check pilot coherence on the first
    // COHERENCE_VALIDATION_WINDOW frames after each lock
    // acquisition. If coherence is consistently low, we locked
    // on noise or at a wrong timing offset. Force immediate
    // re-acquisition.
    d_frames_since_lock++;

    // --- PLS disagreement check ---
    // During initial validation, repeated PLS disagreement is a
    // strong false-lock signal. After that window, the data path can
    // remain healthy even if the soft PLS decoder is noisy; keep
    // using the confirmed PLS and let the coherence/signal-loss/FEC
    // watchdogs decide whether to reset the lock.
    if (d_pl->pls_disagreement_run() >= PLS_DISAGREE_THRESH &&
        d_frames_since_lock <= COHERENCE_VALIDATION_WINDOW) {
        fprintf(stderr,
                "dvbs2-rx: lock validation FAILED — %d consecutive "
                "PLS disagreements (frame %lu) — scheduling full reset\n",
                d_pl->pls_disagreement_run(),
                (unsigned long)d_frames_since_lock);
        schedule_validation_reset();
        return false;
    }

    // --- Continuous coherence check ---
    // Check pilot coherence on every frame. During the initial
    // validation window, use the early/majority logic.
    // After the window, track a consecutive low-coherence streak:
    //  - Force unlock after COHERENCE_UNLOCK_THRESH frames
    //    (signal is truly gone)
    // Use raw coherence during lock validation so false locks are
    // rejected quickly. After the window, use EMA-smoothed coherence
    // so single-frame glitches don't trigger unlock cascades.
    float coh = (d_frames_since_lock <= COHERENCE_VALIDATION_WINDOW)
        ? f.coherence
        : pilot_coherence();
    if (d_frames_since_lock <= COHERENCE_VALIDATION_WINDOW) {
        // During validation window: only count for early/majority
        // logic. Don't start the continuous streak yet — tracking
        // is still converging and coherence is naturally low.
        if (coh < COHERENCE_THRESHOLD)
            d_low_coherence_cnt++;
    } else {
        // After validation window: track continuous streak and
        // build up coherence confirmation.
        if (coh < COHERENCE_THRESHOLD) {
            d_low_coherence_streak++;
            d_good_coherence_run = 0;
            // Revoke confirmation if streak grows — signal went bad
            if (d_low_coherence_streak >= COHERENCE_CONFIRM_THRESH)
                d_coherence_confirmed = false;
        } else {
            d_low_coherence_streak = 0;
            if (!d_coherence_confirmed && ++d_good_coherence_run >= COHERENCE_CONFIRM_THRESH)
                d_coherence_confirmed = true;
        }
    }

    // Early validation window: fast unlock on false lock
    if (d_frames_since_lock <= COHERENCE_VALIDATION_WINDOW) {
        bool at_window_end = (d_frames_since_lock == COHERENCE_VALIDATION_WINDOW);
        bool early_coh_fail = (d_low_coherence_cnt >= COHERENCE_FAIL_THRESH &&
                               d_low_coherence_cnt == (int)d_frames_since_lock);
        bool majority_coh_fail = at_window_end &&
            (d_low_coherence_cnt > COHERENCE_VALIDATION_WINDOW / 2);
        if (early_coh_fail || majority_coh_fail) {
            fprintf(stderr,
                    "dvbs2-rx: lock validation FAILED — coh=%.3f "
                    "(%d/%lu low) — scheduling full reset\n",
                    coh, d_low_coherence_cnt,
                    (unsigned long)d_frames_since_lock);
            schedule_validation_reset();
            return false;
        }
    }

    // Continuous coherence: force unlock after sustained low coherence
    if (d_low_coherence_streak >= COHERENCE_UNLOCK_THRESH) {
        fprintf(stderr,
                "dvbs2-rx: coherence watchdog — %d consecutive "
                "low-coh frames (coh=%.3f, frame %lu) — "
                "scheduling full reset\n",
                d_low_coherence_streak, coh,
                (unsigned long)d_frames_since_lock);
        schedule_validation_reset();
        return false;
    }

    // Gate frame output on three conditions:
    // 1. Coherence confirmed (sustained good coherence after lock)
    // 2. Frame length diversity: on garbage, random PLS decodes
    //    produce many different valid PLFRAME lengths. A real signal
    //    has at most 2 distinct lengths in any short window (same
    //    constellation, or transitioning between two during ACM).
    // 3. SNR plausibly sufficient for this MODCOD — if pilotSNR is
    //    more than SNR_GATE_MARGIN below the MODCOD's QEF threshold,
    //    the frame will certainly fail LDPC (max iterations, wastes GPU).
    if (frame.payload_len > 0 && d_coherence_confirmed) {
        // Track frame length diversity
        d_plf_history[d_plf_idx % PLF_HISTORY_SIZE] =
            (uint32_t)frame.pls.plframe_symbols();
        d_plf_idx++;
        if (d_plf_count < PLF_HISTORY_SIZE) d_plf_count++;
        int n_distinct = 0;
        uint32_t seen[PLF_HISTORY_SIZE];
        for (int i = 0; i < d_plf_count; i++) {
            uint32_t v = d_plf_history[i];
            bool found = false;
            for (int j = 0; j < n_distinct; j++) {
                if (seen[j] == v) { found = true; break; }
            }
            if (!found) seen[n_distinct++] = v;
        }

        // SNR gate
        const auto* mc = find_modcod(frame.pls.modcod);
        float snr = frame.pilot_snr_db;
        bool snr_ok = !mc || snr >= mc->min_snr_db - SNR_GATE_MARGIN;

        if (n_distinct > PLF_DIVERSITY_MAX) {
            // Too many distinct frame lengths — garbage
        } else if (!snr_ok) {
            // SNR too low for this MODCOD — suppress
        } else {
            frames.push_back(frame);
        }
        if (d_debug_level >= 2) {
            static long gate_dbg = 0;
            if (++gate_dbg <= 20 || (gate_dbg % 50) == 0)
                fprintf(stderr, "frontend: frame gate #%ld: plen=%d "
                        "snr=%.1f snr_ok=%d n_distinct=%d -> %s\n",
                        gate_dbg, frame.payload_len, snr, (int)snr_ok,
                        n_distinct,
                        (n_distinct <= PLF_DIVERSITY_MAX && snr_ok)
                            ? "OUT" : "SUPPRESSED");
        }
    } else if (d_debug_level >= 2) {
        static long nogate_dbg = 0;
        if (++nogate_dbg <= 20 || (nogate_dbg % 50) == 0)
            fprintf(stderr, "frontend: frame event #%ld NOT gated-in: "
                    "plen=%d coh_confirmed=%d\n",
                    nogate_dbg, frame.payload_len,
                    (int)d_coherence_confirmed);
    }
    return true;
}

/* ---------- Soft/hard unlock tiering (retained) ---------- */

void receiver_frontend::handle_sync_events(const std::vector<pl_sync_event>& events)
{
    for (const pl_sync_event& ev : events) {
        if (ev.type == pl_sync_event_type::lost) {
            // The PL receiver lost the frame cadence, but this may be
            // spurious (e.g., Farrow timing jitter at fractional SPS).
            // Only propagate a real unlock to the app after sustained
            // failure (SOFT_UNLOCK_MAX consecutive events).
            //
            // Soft unlock: keep d_frame_locked=true so the app never sees it.
            // The PL receiver internally goes back to searching and will find
            // the next PLHEADER. No tracking state is reset, no FEC
            // bootstrap.
            //
            // Hard unlock: after SOFT_UNLOCK_MAX consecutive failures,
            // set d_frame_locked=false and do a full reset. This handles
            // genuine signal loss.
            if (d_soft_unlock_holdoff > 0) {
                // Post-acquisition catch-up — always suppress
                d_soft_unlock_holdoff--;
                d_frame_locked = true;  // override PL receiver
            } else {
                d_soft_unlock_cnt++;
                bool escalate = (d_soft_unlock_cnt >= SOFT_UNLOCK_MAX);
                if (escalate) {
                    // Genuine lock loss — full reset
                    d_frame_locked = false;
                    fprintf(stderr,
                            "dvbs2-rx: HARD unlock after %d frame-sync misses "
                            "— full reset\n",
                            d_soft_unlock_cnt);
                    d_pl->reset(pl_reset_scope::full);
                    d_total_cfo = 0.0;
                    d_rot_phase = complexf(1.0f, 0.0f);
                    d_phase_inc = 0.0f;
                    d_soft_unlock_cnt = 0;
                    if (d_agc_rot_cuda) d_agc_rot_cuda->reset(1.0f);
                    d_sym_pow = 0.0;
                    d_mu = 0.0;
                    d_gpu_overlap_valid = false;
                    d_timing_init = false;
                    d_timing_drift_accum = 0.0;
                    d_timing_nco_base = 0.0;
                    d_frames_since_lock = 0;
                    d_low_coherence_cnt = 0;
                    d_low_coherence_streak = 0;
                    d_good_coherence_run = 0;
                    d_coherence_confirmed = false;
                    d_plf_idx = 0;
                    d_plf_count = 0;
                } else {
                    // Spurious miss — keep d_frame_locked=true, no reset.
                    d_frame_locked = true;  // override PL receiver
                }
            }
        } else if (ev.type == pl_sync_event_type::acquired) {
            // Track re-lock transition (after a hard unlock)
            if (!d_frame_locked) {
                d_frame_locked = true;
                d_relock_time = std::chrono::steady_clock::now();
                // Suppress soft unlock counting during post-acquisition
                // catch-up.  20 blocks is generous: catch-up takes ~6
                // blocks at 491.52 Msps, ~3 at 245.76 Msps.
                d_soft_unlock_holdoff = 20;
                d_frames_since_lock = 0;
                d_low_coherence_cnt = 0;
                d_low_coherence_streak = 0;
                d_good_coherence_run = 0;
                d_coherence_confirmed = false;
                d_plf_idx = 0;
                d_plf_count = 0;
            }
        } else {
            // Every accepted PLHEADER proves the signal is good. Reset the
            // soft unlock counter so spurious misses don't accumulate
            // toward a hard unlock.
            if (d_soft_unlock_cnt > 0)
                d_soft_unlock_cnt = 0;
        }
    }
}

/* ---------- Flush deferred pipeline symbols ---------- */

int receiver_frontend::flush(complexf* out, int max_out,
                             std::vector<frontend_frame_t>& frames)
{
    if (d_prev_n_syms <= 0) return 0;
    int n = run_pl_sync(d_prev_sym_buf.data(), d_prev_n_syms,
                        out, max_out, nullptr, frames);
    d_prev_n_syms = 0;
    return n;
}

/* ---------- Additional getters ---------- */

float receiver_frontend::agc_gain() const
{
    if (d_agc_rot_cuda) return d_agc_rot_cuda->gain();
    return d_sym_pow > 1e-20 ? (float)(1.0 / std::sqrt(d_sym_pow)) : 1.0f;
}

uint64_t receiver_frontend::frame_count() const { return d_pl->frame_count(); }
uint64_t receiver_frontend::dummy_count() const { return d_pl->dummy_count(); }
float receiver_frontend::pilot_snr_db() const { return d_pl->snr_db(); }
bool receiver_frontend::has_pilot_snr() const { return d_pl->snr_valid(); }
float receiver_frontend::pilot_coherence() const { return d_pl->coherence(); }

void receiver_frontend::set_valid_plframe_lens(const std::set<uint32_t>& lens)
{
    d_pl->set_valid_plframe_lens(lens);
}

void receiver_frontend::set_acq_plframe_hint(uint32_t plframe_len)
{
    d_acq_plframe_hint = plframe_len;
    d_pl->set_acq_plframe_hint(plframe_len);
}

void receiver_frontend::force_unlock()
{
    if (!d_frame_locked)
        return;
    fprintf(stderr, "dvbs2-rx: FORCE UNLOCK (FER-gated relock)\n");
    reset_acquisition_state();
}

void receiver_frontend::force_reacquire()
{
    fprintf(stderr, "dvbs2-rx: FORCE REACQUIRE (link recovery)\n");
    reset_acquisition_state();
}

void receiver_frontend::reset_acquisition_state()
{
    d_frame_locked = false;
    d_pl->reset(pl_reset_scope::full);
    d_total_cfo = 0.0;
    d_rot_phase = complexf(1.0f, 0.0f);
    d_phase_inc = 0.0f;
    d_mu = 0.0;
    // the valid-length whitelist persists across unlock (PL receiver)
    d_gpu_overlap_valid = false;
    d_timing_init = false;
    d_timing_drift_accum = 0.0;
    d_timing_nco_base = 0.0;
    if (d_agc_rot_cuda)
        d_agc_rot_cuda->reset(1.0f);
    d_sym_pow = 0.0;
    d_soft_unlock_cnt = 0;
    d_frames_since_lock = 0;
    d_low_coherence_cnt = 0;
    d_low_coherence_streak = 0;
    d_good_coherence_run = 0;
    d_coherence_confirmed = false;
    d_plf_idx = 0;
    d_plf_count = 0;
}

void receiver_frontend::notify_discontinuity()
{
    // Positional state is invalid across the gap; physical estimates
    // (AGC gain, rotator frequency, timing rate_corr) remain valid.
    d_gpu_overlap_valid = false;
    d_timing_init = false;
    d_mu = 0.0;
    d_timing_nco_base = 0.0;
    d_timing_drift_accum = 0.0;
    d_prev_n_syms = 0;
    if (d_pl)
        d_pl->reset(pl_reset_scope::positional);
}

void receiver_frontend::set_timing_estimation_period(int period)
{
    d_timing_est_period_cfg = period;
    if (d_timing_recovery_cuda)
        d_timing_recovery_cuda->set_estimation_period(period);
}

void receiver_frontend::set_energy_k(int k)
{
    if (d_timing_recovery_cuda)
        d_timing_recovery_cuda->set_energy_k(k);
}

void receiver_frontend::set_energy_search_cap(int cap)
{
    if (d_timing_recovery_cuda)
        d_timing_recovery_cuda->set_energy_search_cap(cap);
}

void receiver_frontend::set_diag_rx(bool on)
{
    d_diag_rx = on;
    if (!on && d_rx_diag_fp) {
        fclose(d_rx_diag_fp);
        d_rx_diag_fp = nullptr;
    }
    if (on) {
        d_rx_diag_frame = 0;
        d_rx_diag_prev_coh = 1.0f;
    }
}

} // namespace cusat_dsp
