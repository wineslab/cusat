/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CLI/YAML option parsing for dvbs2-uhd-trx.
 */

#include "trx/options.h"

#include "config_yaml.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <string>


/* ---------- Modcod parsing ---------- */

static std::string to_lower(const std::string& s)
{
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

static int parse_int_value(const char* s, const char* opt_name)
{
    size_t pos = 0;
    int v = std::stoi(s, &pos, 0);
    if (s[pos] != '\0')
        throw std::invalid_argument(std::string(opt_name) + ": invalid integer");
    return v;
}

static uint64_t parse_u64_value(const std::string& s, const char* opt_name)
{
    size_t pos = 0;
    uint64_t v = std::stoull(s, &pos, 0);
    if (s[pos] != '\0')
        throw std::invalid_argument(std::string(opt_name) + ": invalid unsigned integer");
    return v;
}

static double parse_double_value(const char* s, const char* opt_name)
{
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (s[pos] != '\0' || !std::isfinite(v))
        throw std::invalid_argument(std::string(opt_name) + ": invalid number");
    return v;
}

void trx_usage(const char* prog)
{
    fprintf(
        stderr,
        "Usage: %s [options]\n"
        "\n"
        "UHD FDD transceiver: simultaneous TX+RX on a single USRP.\n"
        "\n"
        "USRP options:\n"
        "  --usrp <args>         UHD device args (e.g. \"addr=192.168.40.24\"), or\n"
        "                        \"difi:addr=<radio>,dest_addr=<host data ip>[,dest_port=7777]\"\n"
        "                        for the DIFI radio plugin (rfnoc-difi image; no usrpproxy)\n"
        "  --tx-freq <Hz>        TX center frequency\n"
        "  --rx-freq <Hz>        RX center frequency\n"
        "  --tx-gain <dB>        TX gain (default: 30)\n"
        "  --rx-gain <dB>        RX gain (default: 30)\n"
        "  --rate <Hz>           Sample rate (default: 5.12e6)\n"
        "  --send-frame-size <B> UHD TX transport frame size (0 = automatic)\n"
        "  --recv-frame-size <B> UHD RX transport frame size (0 = automatic)\n"
        "  --tx-channel <int>    TX channel (default: 0)\n"
        "  --rx-channel <int>    RX channel (default: 0)\n"
        "\n"
        "DVB-S2 options:\n"
        "  --modcod <str>        Modulation and coding (e.g. qpsk3/5)\n"
        "  --sym-rate <Hz>       Symbol rate (enables Farrow resampling when rate/sym_rate != "
        "sps)\n"
        "  --rolloff <float>     RRC rolloff (default: 0.2)\n"
        "  --gold-code <int>     PL scrambling gold code (default: 0)\n"
        "  --manual-modcod       CCM with live operator-selected MODCOD (normal TX modcomp)\n"
        "  --demo-control-port <int> Enable loopback-only demo control/telemetry UDP\n"
        "  --acm                 Enable ACM (adaptive TX+RX via QPSK 1/4 signaling)\n"
        "                        ACM targets measured post-BCH FER <= 1e-5\n"
        "  --acm-max-modcod <int> Optional ACM MODCOD-index ceiling (default: uncapped)\n"
        "  --acm-guard-db <dB>   Headroom above measured waterfall thresholds (default: 2.5)\n"
        "  --acm-bootstrap-margin-db <dB> Extra initial-selection headroom (default: 2.0)\n"
        "  --acm-snr-ewma-alpha <a> ACM SNR smoothing coefficient in (0,1] (default: 0.1)\n"
        "  --acm-feedback-interval-ms <ms> In-band ACM report period (default: 500)\n"
        "  --ldpc-trials <int>   Max LDPC iterations (default: 25)\n"
        "  --gpu-ldpc-batch <n>  GPU LDPC decode batch size (default: 16)\n"
        "  --gpu-streams <n>    GPU CUDA streams for RX LDPC (default: 8)\n"
        "\n"
        "I/O options:\n"
        "  --tap <name>          TAP device name for GSE networking\n"
        "  --tx-only             TX only (no RX)\n"
        "  --rx-only             RX only (no TX)\n"
        "  --log                 Print periodic status to stderr\n"
        "  --debug <int>         Debug level (default: 0)\n"
        "  --gpu-batch-ms <ms>   GPU RX batch cap in ms (default: auto 2ms, scales with rate)\n"
        "  --rx-min-batch-ms <ms> Min buffering before RX frontend call (default: 0.5ms at "
        ">200MHz)\n"
        "  --energy-k <int>     Polyphase energy search arms (default: 32, range 8..64)\n"
        "  --energy-search-cap <int> Max symbols for energy search (default: 2048, 0 = full "
        "buffer)\n"
        "  --timing-est-period <int> Timing estimation period (default: 4, 1=every call)\n"
        "  --rx-batch-plframes <int> Batch cap in PLFRAMEs for timing recovery (default: 2)\n"
        "  --rx-min-batch-us <int>  Locked-path min batch in us; raises the per-batch\n"
        "                           budget above the GPU sync wait at high frame rates\n"
        "                           (-1 = auto: 500 us when rate > 150 Msps; 0 = off)\n"
        "  --tx-batch <int>      TX batch size (default: auto, 0=auto)\n"
        "  --pin-threads         Pin workers to distinct available CPUs\n"
        "  --realtime-threads    Also promote pinned workers to SCHED_FIFO (advanced)\n"
        "  --benchmark-tx        TX benchmark mode (no UHD, measures throughput)\n"
        "  --benchmark-batches <int>  Number of batches (default: 1000)\n"
        "  --radio-driver <s>    Radio plugin: uhd (local USRP, default) or difi\n"
        "                           (rfnoc-difi X410, DIFI v1.1 IQ-over-UDP; configure\n"
        "                           via --usrp difi:... args; no usrpproxy)\n"
        "  --tx-symbols          Symbol-rate TX fronthaul: send PL symbols (pre-RRC)\n"
        "                           over DIFI; the FPGA rrc_interp block pulse-shapes\n"
        "                           (SYMTX image, difi driver + --sym-rate required)\n"
        "  --tx-modcomp          Modulation-compressed TX fronthaul: send packed\n"
        "                           constellation indices; the FPGA dvb_plmod block\n"
        "                           maps/pilots/scrambles (MODCOMP image, DVB-S2\n"
        "                           modcods only, difi driver + --sym-rate)\n"
        "  --rx-prefiltered      RX functional-split rung 2: the RU (FPGA rrc_rx)\n"
        "                           already applied the RRC matched filter and\n"
        "                           decimated toward Nyquist, so the frontend uses a\n"
        "                           neutral (windowed-sinc) interpolator, not RRC.\n"
        "                           --rate is the post-decimation host rate.\n"
        "  --rx-symbol-sync      RX functional-split rung 3: the RU (rrc_rx mode 2)\n"
        "                           recovers symbol timing (feedforward energy\n"
        "                           search) and ships one on-time symbol per symbol;\n"
        "                           the frontend skips AGC/timing (symbols-in).\n"
        "                           --rate must equal --sym-rate (1 sps).\n"

        "  --ldpc-load <int>     Benchmark: concurrent worst-case LDPC decode load\n"
        "                           of this batch size (0 = off)\n"
        "  --dump-samples <file> Benchmark: write raw per-batch GPU latencies (us)\n"
        "  --capture-iq <file>   Dump raw RX IQ samples to file\n"
        "  --capture-duration <s> Capture duration in seconds (default: 5)\n"
        "  --channel-side <tx|rx> Apply channel emulator before TX or after RX\n"
        "  --rx-diag             Write RX lock/coherence diagnostics to /tmp/rxdiag.log\n"
        "  --frontend-trace-csv <file> Write per-call frontend timing trace\n"
        "  --fec-trace-csv <file> Write RX FEC scheduling/latency trace\n"
        "  --tx-warmup-frames <n> Padding data frames before releasing a burst\n"
        "  --tx-underflow-trace-csv <file> Write TX send/underflow trace\n"
        "  --config <file>       Load YAML configuration (CLI flags override)\n"
        "  -h, --help            Show this help\n",
        prog);
}

bool trx_parse_options(int argc, char* argv[], trx_options& opt, int& exit_code)
{
    try {
    // --- Pass 1: scan for --config and load YAML defaults ---
    yaml_config cfg;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            if (!cfg.load(argv[++i])) { exit_code = 1; return false; }
            break;
        }
    }
    // Apply YAML values (these become the defaults; CLI overrides below)
    if (!cfg.entries.empty()) {
        opt.usrp_args        = cfg.get_str("radio.usrp", opt.usrp_args);
        opt.tx_freq          = cfg.get_double("radio.tx_freq", opt.tx_freq);
        opt.rx_freq          = cfg.get_double("radio.rx_freq", opt.rx_freq);
        opt.tx_gain          = cfg.get_double("radio.tx_gain", opt.tx_gain);
        opt.rx_gain          = cfg.get_double("radio.rx_gain", opt.rx_gain);
        opt.rate             = cfg.get_double("radio.rate", opt.rate);
        opt.sym_rate_hz      = cfg.get_double("radio.sym_rate", opt.sym_rate_hz);
        opt.uhd_send_frame_size = cfg.get_int("radio.send_frame_size",
                                              opt.uhd_send_frame_size);
        opt.uhd_recv_frame_size = cfg.get_int("radio.recv_frame_size",
                                              opt.uhd_recv_frame_size);
        opt.tx_send_chunk    = cfg.get_int("radio.tx_send_chunk",
                                           opt.tx_send_chunk);
        opt.tx_channel       = cfg.get_int("radio.tx_channel", opt.tx_channel);
        opt.rx_channel       = cfg.get_int("radio.rx_channel", opt.rx_channel);
        opt.clock_source     = cfg.get_str("radio.clock_source", opt.clock_source);
        opt.subdev_spec      = cfg.get_str("radio.subdev_spec", opt.subdev_spec);

        // Embedded usrpproxy tenant server (handled inside the radio plugin).
        opt.radio_driver     = cfg.get_str("radio.driver", opt.radio_driver);
        opt.tx_symbols       = cfg.get_bool("radio.tx_symbols", opt.tx_symbols);
        opt.tx_modcomp       = cfg.get_bool("radio.tx_modcomp", opt.tx_modcomp);
        opt.tx_backoff       = cfg.get_int("radio.tx_backoff", opt.tx_backoff);
        opt.rx_prefiltered   = cfg.get_bool("radio.rx_prefiltered", opt.rx_prefiltered);
        opt.rx_symbol_sync   = cfg.get_bool("radio.rx_symbol_sync", opt.rx_symbol_sync);
        opt.usrpproxy_enabled  = cfg.get_bool("usrpproxy.enabled",  opt.usrpproxy_enabled);
        opt.usrpproxy_shm_name = cfg.get_str ("usrpproxy.shm_name", opt.usrpproxy_shm_name);
        if (cfg.has("usrpproxy.tx_channels"))
            opt.usrpproxy_tx_channels = cfg.get_int_vector("usrpproxy.tx_channels");
        if (cfg.has("usrpproxy.rx_channels"))
            opt.usrpproxy_rx_channels = cfg.get_int_vector("usrpproxy.rx_channels");
        opt.usrpproxy_tx_core  = cfg.get_int("usrpproxy.tx_core", opt.usrpproxy_tx_core);
        opt.usrpproxy_rx_core  = cfg.get_int("usrpproxy.rx_core", opt.usrpproxy_rx_core);
        if (cfg.has("usrpproxy.tx_ring_size"))
            opt.usrpproxy_tx_ring_size =
                parse_u64_value(cfg.get_str("usrpproxy.tx_ring_size"),
                                "usrpproxy.tx_ring_size");
        if (cfg.has("usrpproxy.rx_ring_size"))
            opt.usrpproxy_rx_ring_size =
                parse_u64_value(cfg.get_str("usrpproxy.rx_ring_size"),
                                "usrpproxy.rx_ring_size");

        opt.usrpproxy_tx_antenna = cfg.get_str("usrpproxy.tx_antenna", opt.usrpproxy_tx_antenna);
        opt.usrpproxy_rx_antenna = cfg.get_str("usrpproxy.rx_antenna", opt.usrpproxy_rx_antenna);

        opt.manual_modcod = cfg.get_bool("modulation.manual", opt.manual_modcod);
        opt.demo_control_port = cfg.get_int("demo.control_port", opt.demo_control_port);
        opt.modcod_str       = cfg.get_str("modulation.modcod", opt.modcod_str);
        opt.rolloff          = cfg.get_float("modulation.rolloff", opt.rolloff);
        opt.gold_code        = cfg.get_int("modulation.gold_code", opt.gold_code);
        opt.short_frames     = cfg.get_bool("modulation.short_frames", opt.short_frames);

        opt.acm_mode         = cfg.get_bool("acm.enabled", opt.acm_mode);
        opt.acm_guard_db     = cfg.get_double("acm.guard_db", opt.acm_guard_db);
        opt.acm_bootstrap_margin_db = cfg.get_double(
            "acm.bootstrap_margin_db", opt.acm_bootstrap_margin_db);
        opt.acm_max_modcod   = cfg.get_int("acm.max_modcod", opt.acm_max_modcod);
        opt.acm_snr_ewma_alpha = cfg.get_double(
            "acm.snr_ewma_alpha", opt.acm_snr_ewma_alpha);
        opt.acm_snr_hold_frames = cfg.get_int("acm.snr_hold_frames", opt.acm_snr_hold_frames);
        opt.acm_feedback_interval_ms = cfg.get_double(
            "acm.feedback_interval_ms", opt.acm_feedback_interval_ms);
        if (cfg.has("acm.exclude_modcods")) {
            const auto configured = cfg.get_int_list("acm.exclude_modcods");
            opt.acm_exclude_modcods.insert(configured.begin(), configured.end());
        }

        opt.ldpc_trials      = cfg.get_int("fec.ldpc_trials", opt.ldpc_trials);
        opt.batch_timeout_ms = cfg.get_int("fec.batch_timeout_ms", opt.batch_timeout_ms);

        opt.energy_k         = cfg.get_int("timing_recovery.energy_k", opt.energy_k);
        opt.energy_search_cap = cfg.get_int("timing_recovery.energy_search_cap", opt.energy_search_cap);
        opt.timing_est_period = cfg.get_int("timing_recovery.estimation_period", opt.timing_est_period);
        opt.rx_batch_plframes = cfg.get_int("timing_recovery.rx_batch_plframes", opt.rx_batch_plframes);
        opt.rx_min_batch_us = cfg.get_int("timing_recovery.rx_min_batch_us", opt.rx_min_batch_us);
        opt.rx_min_batch_ms  = cfg.get_double("timing_recovery.rx_min_batch_ms", opt.rx_min_batch_ms);

        opt.gpu_batch_ms     = cfg.get_double("gpu.batch_ms", opt.gpu_batch_ms);
        opt.gpu_ldpc_batch   = cfg.get_int("gpu.ldpc_batch", opt.gpu_ldpc_batch);
        opt.gpu_streams      = cfg.get_int("gpu.streams", opt.gpu_streams);
        opt.micro_batch      = cfg.get_int("gpu.micro_batch", opt.micro_batch);
        opt.tx_batch_size    = cfg.get_int("gpu.tx_batch", opt.tx_batch_size);
        opt.pin_threads      = cfg.get_bool("gpu.pin_threads", opt.pin_threads);
        opt.realtime_threads = cfg.get_bool("gpu.realtime_threads", opt.realtime_threads);
        opt.cuda_sync        = cfg.get_str("gpu.cuda_sync", opt.cuda_sync);
        opt.tx_send_realtime = cfg.get_bool("radio.tx_send_realtime", opt.tx_send_realtime);

        opt.tap_name         = cfg.get_str("io.tap", opt.tap_name);
        opt.tx_only          = cfg.get_bool("io.tx_only", opt.tx_only);
        opt.rx_only          = cfg.get_bool("io.rx_only", opt.rx_only);
        opt.log_enabled      = cfg.get_bool("io.log", opt.log_enabled);
        opt.debug_level      = cfg.get_int("io.debug", opt.debug_level);
        opt.spectrum_enabled = cfg.get_bool("io.spectrum", opt.spectrum_enabled);
        opt.idle_probe_ms    = cfg.get_double("io.idle_probe_ms", opt.idle_probe_ms);
        opt.idle_probe_frames = cfg.get_int("io.idle_probe_frames", opt.idle_probe_frames);
        opt.tx_warmup_frames = cfg.get_int("io.tx_warmup_frames", opt.tx_warmup_frames);

        opt.capture_iq_file  = cfg.get_str("capture.iq_file", opt.capture_iq_file);
        opt.capture_duration = cfg.get_double("capture.duration_s", opt.capture_duration);

        opt.noise_enabled    = cfg.get_bool("channel.noise", opt.noise_enabled);
        opt.pathloss_enabled = cfg.get_bool("channel.pathloss", opt.pathloss_enabled);
        opt.pathloss_model   = cfg.get_str("channel.pathloss_model", opt.pathloss_model);
        if (opt.pathloss_model == "freespace") opt.pathloss_enabled = true;
        opt.channel_emulation_side = cfg.get_str("channel.side", opt.channel_emulation_side);
        opt.noise_port       = cfg.get_int("channel.noise_port", opt.noise_port);
        opt.propagation_delay_enabled = cfg.get_bool("channel.propagation_delay", opt.propagation_delay_enabled);
        opt.init_distance_km = cfg.get_double("channel.distance_km", opt.init_distance_km);

        opt.diag_fec_bench       = cfg.get_bool("diagnostics.fec_bench", opt.diag_fec_bench);
        opt.diag_llr_histogram   = cfg.get_bool("diagnostics.llr_histogram", opt.diag_llr_histogram);
        opt.diag_ldpc_convergence = cfg.get_bool("diagnostics.ldpc_convergence", opt.diag_ldpc_convergence);
        opt.diag_carrier         = cfg.get_bool("diagnostics.carrier_diag", opt.diag_carrier);
        opt.diag_rx              = cfg.get_bool("diagnostics.rx_diag", opt.diag_rx);
        opt.frontend_trace_csv   = cfg.get_str("diagnostics.frontend_trace_csv", opt.frontend_trace_csv);
        opt.fec_trace_csv        = cfg.get_str("diagnostics.fec_trace_csv", opt.fec_trace_csv);
        opt.tx_underflow_trace_csv =
            cfg.get_str("diagnostics.tx_underflow_trace_csv",
                        opt.tx_underflow_trace_csv);
    }

    // --- Pass 2: CLI flags override YAML/defaults ---
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i+1 < argc) { ++i; continue; }  // already handled
        else if (arg == "--usrp" && i+1 < argc) opt.usrp_args = argv[++i];
        else if (arg == "--tx-freq" && i+1 < argc) opt.tx_freq = parse_double_value(argv[++i], "--tx-freq");
        else if (arg == "--rx-freq" && i+1 < argc) opt.rx_freq = parse_double_value(argv[++i], "--rx-freq");
        else if (arg == "--tx-gain" && i+1 < argc) opt.tx_gain = parse_double_value(argv[++i], "--tx-gain");
        else if (arg == "--rx-gain" && i+1 < argc) opt.rx_gain = parse_double_value(argv[++i], "--rx-gain");
        else if (arg == "--rate" && i+1 < argc) opt.rate = parse_double_value(argv[++i], "--rate");
        else if (arg == "--send-frame-size" && i+1 < argc) opt.uhd_send_frame_size = parse_int_value(argv[++i], "--send-frame-size");
        else if (arg == "--recv-frame-size" && i+1 < argc) opt.uhd_recv_frame_size = parse_int_value(argv[++i], "--recv-frame-size");
        else if (arg == "--radio-driver" && i+1 < argc) opt.radio_driver = argv[++i];
        else if (arg == "--tx-symbols") opt.tx_symbols = true;
        else if (arg == "--tx-modcomp") opt.tx_modcomp = true;
        else if (arg == "--tx-backoff" && i+1 < argc) opt.tx_backoff = parse_int_value(argv[++i], "--tx-backoff");
        else if (arg == "--rx-prefiltered") opt.rx_prefiltered = true;
        else if (arg == "--rx-symbol-sync") opt.rx_symbol_sync = true;
        else if (arg == "--tx-channel" && i+1 < argc) opt.tx_channel = parse_int_value(argv[++i], "--tx-channel");
        else if (arg == "--rx-channel" && i+1 < argc) opt.rx_channel = parse_int_value(argv[++i], "--rx-channel");
        else if (arg == "--modcod" && i+1 < argc) opt.modcod_str = to_lower(argv[++i]);
        else if (arg == "--rolloff" && i+1 < argc) opt.rolloff = (float)parse_double_value(argv[++i], "--rolloff");
        else if (arg == "--gold-code" && i+1 < argc) opt.gold_code = parse_int_value(argv[++i], "--gold-code");
        else if (arg == "--manual-modcod")
            opt.manual_modcod = true;
        else if (arg == "--demo-control-port" && i + 1 < argc)
            opt.demo_control_port = parse_int_value(argv[++i], "--demo-control-port");
        else if (arg == "--acm") opt.acm_mode = true;
        else if (arg == "--acm-guard-db" && i+1 < argc) opt.acm_guard_db = parse_double_value(argv[++i], "--acm-guard-db");
        else if (arg == "--acm-bootstrap-margin-db" && i+1 < argc) opt.acm_bootstrap_margin_db = parse_double_value(argv[++i], "--acm-bootstrap-margin-db");
        else if (arg == "--acm-max-modcod" && i+1 < argc) opt.acm_max_modcod = parse_int_value(argv[++i], "--acm-max-modcod");
        else if (arg == "--acm-snr-ewma-alpha" && i+1 < argc) opt.acm_snr_ewma_alpha = parse_double_value(argv[++i], "--acm-snr-ewma-alpha");
        else if (arg == "--acm-feedback-interval-ms" && i+1 < argc) opt.acm_feedback_interval_ms = parse_double_value(argv[++i], "--acm-feedback-interval-ms");
        else if (arg == "--acm-exclude-modcods" && i+1 < argc) {
            std::string csv = argv[++i];
            std::istringstream ss(csv);
            std::string tok;
            while (std::getline(ss, tok, ','))
                opt.acm_exclude_modcods.insert(parse_int_value(tok.c_str(), "--acm-exclude-modcods"));
        }
        else if (arg == "--acm-feedback-listen" && i+1 < argc) opt.acm_feedback_listen = parse_int_value(argv[++i], "--acm-feedback-listen");
        else if (arg == "--acm-feedback-dst" && i+1 < argc) opt.acm_feedback_dst = argv[++i];
        else if (arg == "--ldpc-trials" && i+1 < argc) opt.ldpc_trials = parse_int_value(argv[++i], "--ldpc-trials");
        else if (arg == "--tap" && i+1 < argc) opt.tap_name = argv[++i];
        else if (arg == "--tx-only") opt.tx_only = true;
        else if (arg == "--rx-only") opt.rx_only = true;
        else if (arg == "--log") opt.log_enabled = true;
        else if (arg == "--debug" && i+1 < argc) opt.debug_level = parse_int_value(argv[++i], "--debug");
        else if (arg == "--batch-timeout-ms" && i+1 < argc) opt.batch_timeout_ms = parse_int_value(argv[++i], "--batch-timeout-ms");
        else if (arg == "--pin-threads") opt.pin_threads = true;
        else if (arg == "--realtime-threads") opt.realtime_threads = true;
        else if (arg == "--short-frames") opt.short_frames = true;
        else if (arg == "--tx-batch" && i+1 < argc) opt.tx_batch_size = parse_int_value(argv[++i], "--tx-batch");
        else if (arg == "--padding-idle") opt.idle_dummy_frames = false;
        else if (arg == "--dummy-idle") opt.idle_dummy_frames = true;
        else if (arg == "--idle-probe-ms" && i+1 < argc) opt.idle_probe_ms = parse_double_value(argv[++i], "--idle-probe-ms");
        else if (arg == "--idle-probe-frames" && i+1 < argc) opt.idle_probe_frames = parse_int_value(argv[++i], "--idle-probe-frames");
        else if (arg == "--tx-warmup-frames" && i+1 < argc) opt.tx_warmup_frames = parse_int_value(argv[++i], "--tx-warmup-frames");
        else if (arg == "--benchmark-tx") opt.benchmark_tx = true;
        else if (arg == "--benchmark-batches" && i+1 < argc) opt.benchmark_batches = parse_int_value(argv[++i], "--benchmark-batches");
        else if (arg == "--ldpc-load" && i+1 < argc) opt.bench_ldpc_load = parse_int_value(argv[++i], "--ldpc-load");
        else if (arg == "--dump-samples" && i+1 < argc) opt.bench_dump_samples = argv[++i];
        else if (arg == "--gpu-batch-ms" && i+1 < argc) opt.gpu_batch_ms = parse_double_value(argv[++i], "--gpu-batch-ms");
        else if (arg == "--rx-min-batch-ms" && i+1 < argc) opt.rx_min_batch_ms = parse_double_value(argv[++i], "--rx-min-batch-ms");
        else if (arg == "--energy-k" && i+1 < argc) opt.energy_k = parse_int_value(argv[++i], "--energy-k");
        else if (arg == "--energy-search-cap" && i+1 < argc) opt.energy_search_cap = parse_int_value(argv[++i], "--energy-search-cap");
        else if (arg == "--timing-est-period" && i+1 < argc) opt.timing_est_period = parse_int_value(argv[++i], "--timing-est-period");
        else if (arg == "--rx-batch-plframes" && i+1 < argc) opt.rx_batch_plframes = parse_int_value(argv[++i], "--rx-batch-plframes");
        else if (arg == "--rx-min-batch-us" && i+1 < argc) opt.rx_min_batch_us = parse_int_value(argv[++i], "--rx-min-batch-us");
        else if (arg == "--gpu-ldpc-batch" && i+1 < argc) opt.gpu_ldpc_batch = parse_int_value(argv[++i], "--gpu-ldpc-batch");
        else if (arg == "--gpu-streams" && i+1 < argc) opt.gpu_streams = parse_int_value(argv[++i], "--gpu-streams");
        else if (arg == "--micro-batch" && i+1 < argc) opt.micro_batch = parse_int_value(argv[++i], "--micro-batch");
        else if (arg == "--spectrum") opt.spectrum_enabled = true;
        else if (arg == "--noise") opt.noise_enabled = true;
        else if (arg == "--pathloss") opt.pathloss_enabled = true;
        else if (arg == "--channel-side" && i+1 < argc) opt.channel_emulation_side = argv[++i];
        else if (arg == "--noise-port" && i+1 < argc) opt.noise_port = parse_int_value(argv[++i], "--noise-port");
        else if (arg == "--propagation-delay") opt.propagation_delay_enabled = true;
        else if (arg == "--distance-km" && i+1 < argc) opt.init_distance_km = parse_double_value(argv[++i], "--distance-km");
        else if (arg == "--capture-iq" && i+1 < argc) opt.capture_iq_file = argv[++i];
        else if (arg == "--capture-duration" && i+1 < argc) opt.capture_duration = parse_double_value(argv[++i], "--capture-duration");
        else if (arg == "--rx-diag") opt.diag_rx = true;
        else if (arg == "--frontend-trace-csv" && i+1 < argc) opt.frontend_trace_csv = argv[++i];
        else if (arg == "--fec-trace-csv" && i+1 < argc) opt.fec_trace_csv = argv[++i];
        else if (arg == "--tx-underflow-trace-csv" && i+1 < argc) opt.tx_underflow_trace_csv = argv[++i];
        else if (arg == "--sym-rate" && i+1 < argc) opt.sym_rate_hz = parse_double_value(argv[++i], "--sym-rate");
        else if (arg == "--clock-source" && i+1 < argc) opt.clock_source = argv[++i];
        else if (arg == "--cuda-sync" && i+1 < argc) opt.cuda_sync = argv[++i];
        else if (arg == "--tx-send-slot" && i+1 < argc) opt.tx_send_slot = parse_int_value(argv[++i], "--tx-send-slot");
        else if (arg == "--tx-send-chunk" && i+1 < argc) opt.tx_send_chunk = parse_int_value(argv[++i], "--tx-send-chunk");
        else if (arg == "--no-tx-send-realtime") opt.tx_send_realtime = false;
        else if (arg == "--uhd-num-send-frames" && i+1 < argc) opt.uhd_num_send_frames = parse_int_value(argv[++i], "--uhd-num-send-frames");
        else if (arg == "--uhd-send-buff-size" && i+1 < argc) opt.uhd_send_buff_size = parse_int_value(argv[++i], "--uhd-send-buff-size");
        else if (arg == "--clock-external") opt.clock_source = "external";
        else if (arg == "-h" || arg == "--help") { trx_usage(argv[0]); exit_code = 0; return false; }
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); exit_code = 1; return false; }
    }

    opt.do_tx = !opt.rx_only;
    opt.do_rx = !opt.tx_only;

    auto invalid = [&](const char* msg) {
        fprintf(stderr, "Invalid option: %s\n", msg);
        exit_code = 1;
        return false;
    };
    if (opt.rate < 1.0) return invalid("rate must be >= 1 sample/s");
    if (opt.radio_driver != "uhd" && opt.radio_driver != "difi" &&
        opt.radio_driver != "tcp" && opt.radio_driver != "compfh")
        return invalid("radio-driver must be 'uhd', 'difi', 'tcp', or 'compfh'");
    if (opt.radio_driver != "uhd" && opt.usrpproxy_enabled)
        return invalid("usrpproxy (OAI proxy) requires the uhd radio driver: "
                       "it shares a local multi_usrp, which does not exist "
                       "over a DIFI stream");
    if (opt.sym_rate_hz < 0.0) return invalid("sym-rate must be >= 0");
    if (opt.tx_channel < 0 || opt.rx_channel < 0) return invalid("radio channels must be >= 0");
    if (opt.rolloff < 0.0f || opt.rolloff > 1.0f) return invalid("rolloff must be in [0,1]");
    if (opt.ldpc_trials <= 0) return invalid("ldpc-trials must be > 0");
    if (opt.demo_control_port < 0 || opt.demo_control_port > 65535)
        return invalid("demo-control-port must be in 0..65535");
    if (opt.manual_modcod && (opt.acm_mode || !opt.tx_modcomp || opt.short_frames))
        return invalid(
            "manual-modcod requires normal-frame TX modulation compression and ACM disabled");
    if (opt.tx_modcomp && (opt.noise_enabled || opt.pathloss_enabled) &&
        opt.channel_emulation_side != "rx")
        return invalid("channel emulation must be on RX when TX carries packed indices");
    if (opt.acm_max_modcod < 0) return invalid("acm-max-modcod must be >= 0");
    if (!std::isfinite(opt.acm_guard_db) || opt.acm_guard_db < 0.0)
        return invalid("acm-guard-db must be a finite value >= 0");
    if (!std::isfinite(opt.acm_bootstrap_margin_db) ||
        opt.acm_bootstrap_margin_db < 0.0)
        return invalid("acm-bootstrap-margin-db must be a finite value >= 0");
    if (!std::isfinite(opt.acm_snr_ewma_alpha) ||
        opt.acm_snr_ewma_alpha <= 0.0 || opt.acm_snr_ewma_alpha > 1.0)
        return invalid("acm-snr-ewma-alpha must be in (0,1]");
    if (opt.acm_feedback_interval_ms <= 0.0)
        return invalid("acm-feedback-interval-ms must be > 0");
    if (opt.acm_feedback_listen < 0 || opt.acm_feedback_listen > 65535)
        return invalid("acm-feedback-listen must be in 0..65535");
    if (opt.batch_timeout_ms < 0) return invalid("batch-timeout-ms must be >= 0");
    if (opt.tx_batch_size < 0) return invalid("tx-batch must be >= 0");
    if (opt.benchmark_batches <= 0) return invalid("benchmark-batches must be > 0");
    if (opt.gpu_batch_ms < 0.0) return invalid("gpu-batch-ms must be >= 0");
    if (opt.rx_min_batch_ms < 0.0) return invalid("rx-min-batch-ms must be >= 0");
    if (opt.energy_k <= 0) return invalid("energy-k must be > 0");
    if (opt.energy_search_cap < 0) return invalid("energy-search-cap must be >= 0");
    if (opt.timing_est_period <= 0) return invalid("timing-est-period must be > 0");
    if (opt.rx_batch_plframes <= 0) return invalid("rx-batch-plframes must be > 0");
    if (opt.gpu_ldpc_batch <= 0) return invalid("gpu-ldpc-batch must be > 0");
    if (opt.gpu_streams <= 0) return invalid("gpu-streams must be > 0");
    if (opt.micro_batch <= 0) return invalid("micro-batch must be > 0");
    if (opt.noise_port <= 0 || opt.noise_port > 65535) return invalid("noise-port must be in 1..65535");
    if (opt.channel_emulation_side != "tx" && opt.channel_emulation_side != "rx")
        return invalid("channel-side must be tx or rx");
    if (opt.pathloss_model != "none" && opt.pathloss_model != "freespace")
        return invalid("pathloss-model must be 'none' or 'freespace'");
    if (opt.init_distance_km < 0.0) return invalid("distance-km must be >= 0");
    if (opt.capture_duration <= 0.0) return invalid("capture-duration must be > 0");
    if (opt.tx_send_chunk <= 0) return invalid("tx-send-chunk must be > 0");
    if (opt.idle_probe_frames < 1) return invalid("idle-probe-frames must be >= 1");
    if (!std::isfinite(opt.idle_probe_ms) || opt.idle_probe_ms < 0.0)
        return invalid("idle-probe-ms must be a finite value >= 0");
    if (opt.tx_warmup_frames < 0) return invalid("tx-warmup-frames must be >= 0");
    if (opt.uhd_num_send_frames <= 0) return invalid("uhd-num-send-frames must be > 0");
    if (opt.uhd_send_buff_size < 0) return invalid("uhd-send-buff-size must be >= 0");
    if (opt.uhd_send_frame_size < 0) return invalid("send-frame-size must be >= 0");
    if (opt.uhd_recv_frame_size < 0) return invalid("recv-frame-size must be >= 0");

    exit_code = 0;
    return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Invalid option: %s\n", e.what());
        exit_code = 1;
        return false;
    }
}
