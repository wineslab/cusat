/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared context passed to the dvbs2-uhd-trx worker threads.
 *
 * trx_stats owns the atomic counters / latency probes shared between the
 * worker threads and main()'s logging loop (field names identical to the
 * historical locals in main()).  trx_context points at the objects main()
 * constructs (options, radio streams, rings, DSP pipelines, CUDA mutex)
 * so each thread body can run unmodified against hoisted references.
 */

#pragma once

#include "channel_impairment.h"
#include "gse.h"
#include "pl_framer.h"
#include "framing/plframe.h"
#include "framing/plheader.h"
#include "receiver_frontend.h"
#include "rx_fec_chain.h"
#include "rx_fec_pool.h"
#include "spectrum_tap.h"
#include "spsc_ring.h"
#include "tap_interface.h"
#include "transmitter_frontend.h"
#include "tx_fec_chain.h"
#include "tx_fec_pool.h"
#include "cuda/farrow_kernel.cuh"
#include "cuda/tx_pipeline.cuh"
#include <cusat_dsp/modcod_types.h>

#include "trx/options.h"
#include "trx/queues.h"
#include "trx/radio_device.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>

using namespace cusat_dsp;

/* ---------- Shared atomic counters & latency probes ---------- */
struct trx_stats {
    // --- Counters ---
    std::atomic<uint64_t> tx_frames{0};
    std::atomic<uint64_t> tx_dummy_frames{0};
    std::atomic<uint64_t> rx_fec_frames{0};     // post-bootstrap only
    std::atomic<uint64_t> rx_fec_errors{0};     // post-bootstrap only
    std::atomic<uint64_t> rx_fec_frames_pre{0}; // pre-bootstrap (discarded)
    std::atomic<uint64_t> rx_pdus{0};
    std::atomic<uint64_t> rx_pdu_bytes{0};
    std::atomic<uint64_t> tx_underflows{0};
    std::atomic<uint64_t> tx_sequence_errors{0};
    std::atomic<uint64_t> tx_time_errors{0};
    std::atomic<uint64_t> rx_uhd_overflows{0};
    std::atomic<uint64_t> rx_ring_drops{0};
    std::atomic<uint64_t> tx_pdu_queue_drops{0};
    std::atomic<int> fe_stage_atomic{-1}; // frontend thread stage for stall diagnosis
    std::atomic<uint64_t> rx_fec_queue_drops{0};

    // EWMA FER (α=0.02, ~50-frame time constant)
    // Atomic: written by FEC thread, read by RX frontend thread (FER watchdog).
    std::atomic<float> ewma_fer{0.0f};

    // GPU sync point counters (for benchmark summary)
    std::atomic<uint64_t> sync_count_tx{0};
    std::atomic<uint64_t> sync_count_rx{0};

    // Shared state between frontend and FEC threads
    std::atomic<bool> fec_bootstrapped{false};
    std::atomic<uint64_t> fec_reset_seq{0};

    // ACM diagnostic counters
    std::atomic<uint64_t> acm_frames_submitted{0};
    std::atomic<uint64_t> acm_frames_skipped_len{0};
    std::atomic<uint64_t> acm_frames_skipped_mc{0};
    std::atomic<uint64_t> acm_frames_dummy{0};
    std::atomic<uint64_t> acm_feedback_tx{0};
    std::atomic<uint64_t> acm_feedback_rx{0};
    std::atomic<uint64_t> acm_feedback_recovery_bursts{0};
    // Per-modcod frame counts over the full DVB-S2X index domain; excludes
    // the reserved ACM signaling frames.
    std::atomic<uint64_t> acm_mc_hist[256] = {};
    // Per-chain decoder outcomes. OLLA reads the peer's selected data mode,
    // thereby excluding the reserved robust signaling chain.
    std::atomic<uint64_t> acm_mc_fec_frames[256] = {};
    std::atomic<uint64_t> acm_mc_fec_errors[256] = {};
    std::atomic<uint64_t> acm_pls_guard_corrected{0};
    std::atomic<uint64_t> acm_pls_guard_deferred{0};
    std::atomic<uint64_t> acm_pls_guard_transitions{0};
    std::atomic<uint64_t> acm_pls_guard_outage_frames{0};
    // Last non-signaling RX MODCOD (for dashboard display)
    std::atomic<uint8_t> rx_data_modcod{0};

    // Latency instrumentation
    std::atomic<uint64_t> lat_rx_ring_samples{0};  // RX ring occupancy (sampled periodically)
    std::atomic<uint64_t> lat_fec_decode_us{0};    // Last FEC decode time (µs)
    std::atomic<uint64_t> lat_frontend_us{0};      // Last frontend.process() time (µs)
    std::atomic<uint64_t> lat_tx_gen_us{0};        // Last TX frame generation time (µs)
    std::atomic<uint64_t> lat_tx_gse_us{0};        // TX GSE encap time (µs)
    std::atomic<uint64_t> lat_tx_bb_bch_us{0};      // TX BB header+scramble+BCH (µs)
    std::atomic<uint64_t> lat_tx_ldpc_us{0};        // TX LDPC encode (µs)
    std::atomic<uint64_t> lat_tx_gpu_us{0};         // TX GPU pipeline time (µs)
    std::atomic<uint64_t> lat_batch_wait_us{0};    // Time frame waited in batch before flush (µs)
    std::atomic<uint64_t> lat_tx_loop_us{0};       // TX data thread loop period (µs)
    std::atomic<uint64_t> lat_rx_loop_us{0};       // RX pipeline loop period (µs)
    std::atomic<uint64_t> lat_gpu_drain_us{0};     // GPU drain_completed time (µs)
    std::atomic<uint64_t> lat_gpu_phase_us{0};     // GPU phase correction time (µs)
    std::atomic<uint64_t> lat_gpu_submit_us{0};    // GPU submit_xfecframe time (µs)
    std::atomic<int> gpu_inflight_count{0};        // GPU in-flight slot count
    std::atomic<uint64_t> lat_fec_loop_us{0};      // Full FEC loop iteration time (µs)
    std::atomic<uint64_t> gpu_force_drain_count{0}; // Times force_drain_oldest was called

    // Frontend sub-stage breakdown (from receiver_frontend::perf())
    std::atomic<uint64_t> lat_fe_agc_us{0};
    std::atomic<uint64_t> lat_fe_timing_us{0};
    std::atomic<uint64_t> lat_fe_sync_us{0};
    std::atomic<int>      lat_fe_n_in{0};
    std::atomic<int>      lat_fe_n_syms{0};
    std::atomic<bool>     lat_fe_fused{false};
    // timing recovery sub-stage breakdown
    std::atomic<uint64_t> lat_tr_memcpy_us{0};
    std::atomic<uint64_t> lat_tr_launch_us{0};
    std::atomic<uint64_t> lat_tr_lnch_est_us{0};
    std::atomic<uint64_t> lat_tr_lnch_interp_us{0};
    std::atomic<uint64_t> lat_tr_sync_us{0};
    std::atomic<uint64_t> lat_tr_readback_us{0};
    std::atomic<uint64_t> lat_gpu_timing_us{0};
    std::atomic<uint64_t> lat_gpu_interp_us{0};

    std::atomic<uint64_t> lat_tx_ring_wait_us{0}; // Ring backpressure wait
    std::atomic<int> lat_tx_batch_size{0};        // frames in last batch

    // TX send-path stall diagnostics (written by tx_send thread; the
    // logger exchanges the maxima to 0 each period, counters are cumulative)
    std::atomic<uint64_t> txs_send_max_us{0};  // max time inside send() this period
    std::atomic<uint64_t> txs_gap_max_us{0};   // max gap between sends this period
    std::atomic<uint64_t> txs_dur_over200{0};  // sends with dur > 200µs
    std::atomic<uint64_t> txs_gap_over200{0};  // inter-send gaps > 200µs
    std::atomic<uint64_t> txs_dur_over1ms{0};  // sends with dur > 1ms
    std::atomic<uint64_t> txs_gap_over1ms{0};  // inter-send gaps > 1ms

    // ACM TX control: main thread writes, TX thread reads.  The selected
    // MODCOD may be an upward candidate under OTA probation; application
    // traffic remains on acm_tx_data_modcod until feedback validates it.
    std::atomic<uint8_t> acm_tx_modcod{0};
    std::atomic<uint8_t> manual_applied_modcod{0};
    std::atomic<uint8_t> acm_tx_data_modcod{0};
    std::atomic<uint64_t> acm_control_updates{0};

    // Local receiver's smoothed pilot Es/N0. Main updates this from the raw
    // estimator every 200 ms; both in-band and UDP feedback paths consume it.
    std::atomic<float> acm_rx_snr_ewma{NAN};

    // In-band ACM signaling: RX FEC thread → main thread (peer's reported SNR)
    std::atomic<float> acm_inband_snr{NAN};
    std::atomic<float> acm_inband_fer{0.0f};
    std::atomic<uint64_t> acm_inband_fec_frames{0};
    std::atomic<uint64_t> acm_inband_fec_errors{0};
    std::atomic<uint8_t> acm_inband_rx_modcod{0};


    // Debug: track who holds cuda_mtx and for how long
    std::atomic<int> cuda_mtx_owner{0};       // 0=none, 1=fe, 2=fec, 3=tx
    std::atomic<uint64_t> cuda_mtx_hold_us{0}; // last hold duration
};

/* ---------- Shared context wiring main()'s objects to the threads ---------- */
struct trx_context {
    // --- Options & derived parameters ---
    trx_options* opt = nullptr;
    fecframe_size_t framesize = FECFRAME_NORMAL;
    bool acm_tx_enabled = false;
    // For ACM TX: max buffer sizes across all enabled MODCODs
    unsigned int acm_max_dfl = 0;
    unsigned int acm_max_xfec = 0;
    unsigned int acm_max_plf = 0;
    bool use_farrow = false;
    double hw_sps = 0.0;
    int max_gpu_batch = 0;
    int xfecframe_len = 0;
    uint8_t acm_floor_modcod = 0;
    float acm_floor_pilot_snr_db = -100.0f;
    size_t tx_ring_cap = 1;
    pl_layout* frame_format = nullptr;
    std::chrono::steady_clock::time_point start_time{};

    // --- Radio (hardware-agnostic; concrete driver is a dlopen'd plugin) ---
    radio_device* radio = nullptr;
    double* stream_start_time = nullptr;   // device time, seconds

    // --- Rings & queues ---
    spsc_ring<complexf>* rx_ring = nullptr;
    spsc_ring<complexf>* tx_ring = nullptr;
    spsc_frame_queue* frame_queue = nullptr;
    pdu_queue* tap_pdus = nullptr;

    // --- TAP & demo dashboard ---
    std::unique_ptr<tap_interface>* tap = nullptr;
    std::unique_ptr<spectrum_tap>* wf_tap = nullptr;
    std::unique_ptr<channel_impairment>* tx_impairment = nullptr;
    std::unique_ptr<channel_impairment>* rx_impairment = nullptr;

    // --- TX DSP pipeline ---
    std::unique_ptr<tx_fec_chain>* tx_fec = nullptr;       // CCM mode
    std::unique_ptr<tx_fec_pool>* tx_fec_acm = nullptr;    // ACM mode
    std::unique_ptr<pl_framer>* tx_framer = nullptr;
    std::unique_ptr<transmitter_frontend>* tx_frontend = nullptr;
    std::unique_ptr<gse_encapsulator>* tx_enc = nullptr;
    std::unique_ptr<cuda::tx_pipeline_cuda>* tx_pipeline = nullptr;

    // --- TX Farrow resampler (RX consumes hardware-rate samples directly) ---
    std::unique_ptr<cuda::farrow_cuda>* tx_farrow = nullptr;

    // --- RX DSP pipeline ---
    std::unique_ptr<receiver_frontend>* rx_frontend = nullptr;
    std::unique_ptr<rx_fec_chain>* fec_single = nullptr;
    std::unique_ptr<rx_fec_pool>* fec_multi = nullptr;
    std::unique_ptr<gse_decapsulator>* rx_decap = nullptr;

    // --- CUDA serialization (see comment at definition in main()) ---
    std::mutex* cuda_mtx = nullptr;

    // --- Shared atomic counters & thread state ---
    trx_stats stats;
};
