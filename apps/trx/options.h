/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Configuration options and CLI/YAML parsing for dvbs2-uhd-trx.
 */

#pragma once

#include "acm_config.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

/**
 * All configuration for the transceiver, with defaults identical to the
 * historical locals at the top of main().
 */
struct trx_options {
    // Defaults
    std::string usrp_args;
    double tx_freq = 0, rx_freq = 0;
    double tx_gain = 30, rx_gain = 30;
    double rate = 5.12e6;
    int tx_channel = 0, rx_channel = 0;
    std::string modcod_str = "qpsk3/5";
    static constexpr bool pilots = true; // pilots are mandatory
    const int sps = 4;
    float rolloff = 0.2f;
    int gold_code = 0;
    bool acm_mode = false;
    bool manual_modcod = false; // CCM with operator-selected, frame-boundary changes
    int demo_control_port = 0;  // loopback UDP, disabled by default
    bool multi_modcod() const { return acm_mode || manual_modcod; }
    double acm_guard_db = cusat_dsp::ACM_DEFAULT_GUARD_DB;
    double acm_bootstrap_margin_db =
        cusat_dsp::ACM_DEFAULT_BOOTSTRAP_MARGIN_DB;
    int ldpc_trials = 10;
    std::string tap_name;
    bool tx_only = false, rx_only = false;
    bool log_enabled = false;
    int debug_level = 0;
    int batch_timeout_ms = 1;  // LDPC batch flush timeout in ms
    bool pin_threads = false;    // Pin threads to cores
    bool realtime_threads = false; // Promote pinned workers to SCHED_FIFO
    int acm_max_modcod = 255;    // Optional ACM MODCOD-index ceiling; 255 = uncapped
    double acm_snr_ewma_alpha = 0.1; // ACM SNR EWMA coefficient, sampled every 200 ms
    int acm_snr_hold_frames = 50; // Hold last good SNR for N frames during coherence dips
    double acm_feedback_interval_ms = 500.0; // In-band ACM report cadence
    int acm_feedback_listen = 0; // UDP port for receiver SNR (TX side, 0=in-band)
    std::string acm_feedback_dst; // HOST:PORT for receiver SNR (RX side)
    // Modes dominated on the measured INT16 FER=1e-5 SNR/efficiency envelope:
    // QPSK 8/9, 8PSK 5/6, 8PSK 8/9, 8PSK 9/10, 16APSK 8/9, 16APSK 9/10.
    std::set<int> acm_exclude_modcods = {10, 15, 16, 17, 22, 23};
    bool short_frames = false;   // Use FECFRAME_SHORT (16200 bits)
    int tx_batch_size = 0;       // TX batch size (0 = auto)
    // Production idle is always standards-compliant DVB dummy PLFRAMEs.  This
    // remains mutable only for the hidden --padding-idle diagnostic override.
    bool idle_dummy_frames = true;
    // Dummy PLFRAMEs maintain an existing carrier lock, but the receiver does
    // not cold-acquire from their dummy PLS geometry.  A sparse normal padding
    // frame guarantees initial/re-acquisition without returning to continuous
    // padding and its compute cost.
    double idle_probe_ms = 1000.0; // Internal CCM reacquisition-probe cadence
    int idle_probe_frames = 8; // Internal startup/reacquisition probe burst
    int tx_warmup_frames = 0; // Extra padding frames before each application burst
    bool benchmark_tx = false;   // TX benchmark mode (no UHD/RX)
    int benchmark_batches = 1000; // Number of batches to benchmark
    int bench_ldpc_load = 0;     // Concurrent LDPC decode load (batch size, 0 = off)
    std::string bench_dump_samples; // Path: raw per-batch GPU latencies (us) for CDF
    double gpu_batch_ms = 0;     // GPU RX batch cap in ms (0 = auto 2ms)
    double rx_min_batch_ms = 0;  // Min samples before frontend call (0 = auto)
    int energy_k = 32;           // Polyphase energy search arms (default: 32)
    int energy_search_cap = 2048; // Max symbols for energy search (0 = no cap)
    int timing_est_period = 4;   // Timing recovery estimation period
    int rx_batch_plframes = 2;   // Batch cap in PLFRAMEs for timing recovery
    int rx_min_batch_us = -1;    // Locked-path min batch, us (-1 = auto: 500 when rate>150e6)
    int gpu_ldpc_batch = 16;     // GPU LDPC decode batch size
    int gpu_streams = 8;         // GPU CUDA streams for RX LDPC
    int micro_batch = 4;         // Frames per GPU launch (micro-batch size)
    bool spectrum_enabled = false; // Enable spectrum tap (writes to FIFO for sidecar)
    int noise_port = 5556;       // UDP port to receive noise/pathloss commands (localhost)
    bool noise_enabled = false;  // Enable channel noise injection
    bool pathloss_enabled = false; // Enable path loss channel model
    std::string pathloss_model = "none"; // none or freespace (relative to initial distance)
    std::string channel_emulation_side = "tx"; // Apply emulator at local TX or RX
    bool propagation_delay_enabled = false; // Enable IQ propagation-delay line
    double init_distance_km = 0.0; // Initial per-direction path distance (km)
    std::string capture_iq_file;   // If set, dump raw RX IQ to this file
    double capture_duration = 5.0; // Capture duration in seconds
    double sym_rate_hz = 0.0;     // Symbol rate in Hz (0 = compute from rate/sps)
    std::string clock_source = "internal"; // USRP clock source: "internal", "external", "gpsdo"
    std::string subdev_spec;     // UHD subdev spec for TX+RX (empty = UHD default)
    // UHD transport frame sizes in bytes (0 = UHD/driver automatic).  These
    // are device arguments rather than DVB frame sizes; a standard-MTU value
    // is useful on host NICs whose jumbo UDP path is unreliable.
    int uhd_send_frame_size = 0;
    int uhd_recv_frame_size = 0;

    // Embedded usrpproxy tenant server (shares the USRP with an external OAI
    // softmodem over a POSIX shm ring). Forwarded to the radio plugin, which
    // owns the multi_usrp; the app itself never links UHD.
    // Radio driver plugin: "uhd" (local USRP) or "difi" (IQ-over-UDP per
    // IEEE-ISTO 4900; data plane only, remote RF head tuned out of band).
    bool tx_symbols = false;
    bool tx_modcomp = false;    // modulation-compressed TX fronthaul: send
                                // packed constellation indices; the FPGA
                                // dvb_plmod block maps/pilots/scrambles    // symbol-rate TX fronthaul: hand PL symbols
                                // (pre-RRC) to the radio plugin; the FPGA
                                // rrc_interp block does the pulse shaping
    int  tx_backoff = -1;       // modcomp per-frame PAPR backoff override for
                                // the plmod prologue (W1[31:24], N/256 scale);
                                // -1 = use modcod_table.h default (128APSK 192)
    bool rx_prefiltered = false; // RX functional-split rung 2: the RU (FPGA
                                // rrc_rx block) has already applied the RRC
                                // matched filter + decimated toward Nyquist, so
                                // the frontend must NOT re-apply RRC — it runs
                                // timing recovery with a neutral (windowed-sinc)
                                // fractional interpolator instead. --rate is the
                                // post-decimation host rate (e.g. 61.44e6 = sps2)
    bool rx_symbol_sync = false; // RX functional-split rung 3: the RU (rrc_rx
                                // mode 2, feedforward energy search) recovers
                                // symbol timing and ships exactly one on-time
                                // symbol per symbol. The frontend skips AGC/
                                // timing entirely (symbols-in). --rate must
                                // equal --sym-rate (host receives 1 sps).
    std::string radio_driver = "uhd";

    bool        usrpproxy_enabled = false;
    std::string usrpproxy_shm_name = "/usrpproxy";
    std::string usrpproxy_tx_antenna;
    std::string usrpproxy_rx_antenna;
    std::vector<int> usrpproxy_tx_channels;
    std::vector<int> usrpproxy_rx_channels;
    int         usrpproxy_tx_core = -1;
    int         usrpproxy_rx_core = -1;
    uint64_t    usrpproxy_tx_ring_size = 16ull * 1024 * 1024;
    uint64_t    usrpproxy_rx_ring_size = 16ull * 1024 * 1024;

    // TX underflow experiment knobs (defaults = historical behavior)
    int tx_send_slot = 1;        // pin slot for tx_send thread (with --pin-threads)
    int tx_send_chunk = 1;       // ring-read size per send(), in multiples of spp
    bool tx_send_realtime = true;// SCHED_FIFO only for UHD's paced sender
    int uhd_num_send_frames = 16; // UHD TX stream num_send_frames
    int uhd_send_buff_size = 0;  // UHD TX link SO_SNDBUF (0 = UHD default)
    // GPU sync wait mode: "spin" (CUDA default here; busy-waits at FIFO80
    // and starves ksoftirqd -> delays UHD flow-control packets), or
    // "block" (cudaDeviceScheduleBlockingSync; sleeps in the driver)
    std::string cuda_sync = "spin";

    // Diagnostic flags (all OFF by default)
    bool diag_fec_bench = false;
    bool diag_llr_histogram = false;
    bool diag_ldpc_convergence = false;
    bool diag_carrier = false;
    bool diag_rx = false;
    std::string frontend_trace_csv;
    std::string fec_trace_csv;
    std::string tx_underflow_trace_csv;

    // Derived from tx_only/rx_only by trx_parse_options()
    bool do_tx = true;
    bool do_rx = true;
};

/** Print the usage/help text to stderr (verbatim historical text). */
void trx_usage(const char* prog);

/**
 * Parse YAML config (--config) then CLI flags into @p opt, and run the
 * pure option validation.
 *
 * Returns true to continue; returns false when the process must exit,
 * with @p exit_code set to the historical exit status (0 for --help,
 * 1 for errors).
 */
bool trx_parse_options(int argc, char* argv[], trx_options& opt, int& exit_code);
