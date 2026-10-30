/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * UHD FDD transceiver: simultaneous TX+RX on a single USRP.
 *
 * Uses two RF chains on the same (or different) channel(s):
 *   TX on TX/RX port, RX on RX2 port.
 * The X410 ZBX daughterboard supports independent TX/RX LOs on the same
 * channel, enabling FDD without digital frequency separation.
 *
 * Thread architecture (7 threads):
 *   0. TAP read:     TAP fd → strip Ethernet → PDU queue (continuous)
 *   1. TX gen:       PDU queue → GSE → FEC → PL frame → RRC → TX IQ ring
 *   2. TX send:      TX IQ ring → tx_stream->send()
 *   3. RX UHD:       rx_stream->recv() → RX ring
 *   4. RX frontend:  RX ring → receiver_frontend → frame_queue
 *   5. RX FEC:       frame_queue → GPU demap+LDPC (async) → BCH → GSE → TAP
 *   6. Main:         USRP setup, logging, ACM controller
 *
 * TX pipelining: generation and UHD I/O run concurrently via an IQ
 * ring buffer. When no user data, the gen thread emits cheap dummy
 * PLFRAMEs (no FEC) to build buffer credit in the ring.
 */

#include "acm_config.h"
#include "trx/plmod_pack.h"
#include "acm_controller.h"
#include "cuda/farrow_kernel.cuh"
#include "rx_fec_chain.h"
#include "fec/code_params.h"
#include "rx_fec_pool.h"
#include "gse.h"
#include "iq_source.h"
#include "pl_framer.h"
#include "framing/plframe.h"
#include "framing/plheader.h"
#include "receiver_frontend.h"
#include "spsc_ring.h"
#include "tap_interface.h"
#include "transmitter_frontend.h"
#include "tx_fec_chain.h"
#include "tx_fec_pool.h"
#include <cusat_dsp/modcod_types.h>

#include "channel_impairment.h"
#include "spectrum_tap.h"
#include "snr_ewma.h"
#include "crash_handler.h"
#include "config_yaml.h"
#include "cuda/cuda_util.cuh"
#include "cuda/bch_encode_kernel.cuh"
#include "cuda/tx_pipeline.cuh"

#include "trx/acm_signaling.h"
#include "trx/bench_stats.h"
#include "trx/bench_tx.h"
#include "trx/context.h"
#include "trx/options.h"
#include <sstream>
#include "trx/queues.h"
#include "trx/radio_device.h"
#include "trx/radio_loader.h"
#include "trx/rx_threads.h"
#include "trx/tap_thread.h"
#include "trx/thread_util.h"
#include "trx/tx_threads.h"


#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <pthread.h>
#include <sched.h>
#include <sys/select.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace cusat_dsp;

/* ---------- Main ---------- */

int main(int argc, char* argv[])
{
    install_crash_handler();

    trx_options opt;
    int parse_exit = 0;
    if (!trx_parse_options(argc, argv, opt, parse_exit))
        return parse_exit;

    if (opt.pin_threads) detect_available_cores();

    // Must precede CUDA context creation (i.e. init_cuda). "block" makes
    // GPU-waiting threads sleep in the driver instead of busy-spinning.
    // This is required if --realtime-threads is combined with workloads that
    // cannot dedicate the pinned CPUs and their network softirqs completely.
    if (opt.cuda_sync == "block") {
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        fprintf(stderr, "dvbs2-uhd-trx: CUDA sync mode: blocking\n");
    } else if (opt.cuda_sync != "spin") {
        fprintf(stderr, "Error: --cuda-sync must be 'spin' or 'block'\n");
        return 1;
    }

    if (!cusat_dsp::cuda::init_cuda()) {
        fprintf(stderr, "Error: no GPU available\n");
        return 1;
    }

    // --- TX Farrow resampler setup ---
    // When --sym-rate is given, compute the hardware SPS. The TX path still
    // generates at the internal integer SPS and Farrow-resamples to the
    // hardware rate. The RX path processes hardware-rate samples directly:
    // the receiver frontend's fractional-stride timing NCO handles any
    // hw_sps = rate/sym_rate down to the matched-filter Nyquist bound
    // (1 + rolloff) per symbol — no RX resampler.
    double hw_sps = (double)opt.sps;  // default: hardware rate == internal rate
    bool use_farrow = false;
    std::unique_ptr<cuda::farrow_cuda> tx_farrow;

    if (opt.sym_rate_hz > 0.0) {
        hw_sps = opt.rate / opt.sym_rate_hz;
        // RX bound: the complex signal spans (1+rolloff)*sym_rate, so the
        // matched filter needs hw_sps > 1 + rolloff (plus a small margin
        // for the timing estimator's eye contrast).
        // Rung 3 (--rx-symbol-sync): the RU recovers timing, the host gets
        // exactly one symbol per symbol — hw_sps must be 1.0 and the
        // matched-filter minimum below does not apply.
        if (opt.rx_symbol_sync && std::fabs(hw_sps - 1.0) > 1e-6) {
            fprintf(stderr, "Error: --rx-symbol-sync requires --rate == "
                    "--sym-rate (host receives 1 sample/symbol), got "
                    "rate/sym_rate = %.4f\n", hw_sps);
            return 1;
        }
        if (opt.rx_symbol_sync && opt.rx_prefiltered) {
            fprintf(stderr, "Error: --rx-symbol-sync (rung 3) and "
                    "--rx-prefiltered (rung 2) are mutually exclusive\n");
            return 1;
        }
        double min_sps = 1.0 + opt.rolloff + 0.05;
        // Tolerance so an exactly-at-the-bound rate (e.g. 38.4/30.72 = 1.25
        // at rolloff 0.20) isn't rejected by float rounding of the sum.
        if (!opt.rx_symbol_sync && hw_sps < min_sps - 1e-6) {
            fprintf(stderr, "Error: rate/sym_rate = %.4f < %.2f "
                    "(matched-filter minimum at rolloff %.2f)\n",
                    hw_sps, min_sps, opt.rolloff);
            return 1;
        }
        // hw_sps==3 (integer stride via TX Farrow 4->3): the energy-search
        // timing recovery branch-alternates and limit-cycles live (~20% FER).
        // RX path only; TX-only at sps3 is fine. Override: CUSAT_ALLOW_SPS3=1.
        if (opt.do_rx && std::fabs(hw_sps - 3.0) < 0.05 &&
            !getenv("CUSAT_ALLOW_SPS3")) {
            fprintf(stderr,
                    "Error: hw_sps = rate/sym_rate = %.4f (~3) is a known-bad "
                    "live RX timing regime (energy-search branch alternation, "
                    "~20%% FER). Use sps=2 or sps=4, or a well-fractional "
                    "stride. Set CUSAT_ALLOW_SPS3=1 to override.\n", hw_sps);
            return 1;
        }
        double ratio = hw_sps / opt.sps;
        if (opt.tx_symbols || opt.tx_modcomp) {
            // Symbol-rate TX fronthaul: the host emits PL symbols (1 sample
            // per symbol, pre-RRC); the FPGA rrc_interp + DUC produce the
            // DAC rate, so neither the GPU RRC nor the TX Farrow run. RX is
            // untouched (hardware-rate samples, fractional timing NCO).
            fprintf(stderr, "dvbs2-uhd-trx: TX %s fronthaul: %.3f Msym/s "
                    "(FPGA pulse shaping); RX at %.3f Msps (hw_sps=%.4f)\n",
                    opt.tx_modcomp ? "modulation-compressed" : "symbol",
                    opt.sym_rate_hz / 1e6, opt.rate / 1e6, hw_sps);
        } else if (fabs(ratio - 1.0) > 1e-9) {
            // TX Farrow uses tx_pipeline->stream() (same as TX gen GPU work).
            if (opt.do_tx) {
                int tx_max = (int)(66000 * hw_sps * 64 + 1000);
                tx_farrow = std::make_unique<cuda::farrow_cuda>((double)opt.sps / hw_sps, tx_max);
                use_farrow = true;
            }
            fprintf(stderr,
                    "dvbs2-uhd-trx: TX Farrow resampler %s: hw_sps=%.6f, "
                    "internal_sps=%d (rate=%.0f, sym_rate=%.0f); "
                    "RX runs at the hardware rate (fractional timing NCO)\n",
                    opt.do_tx ? "enabled" : "not needed (RX only)",
                    hw_sps, opt.sps, opt.rate, opt.sym_rate_hz);
        }
    }

    // --- Demo dashboard: spectrum tap + channel impairment ---
    std::unique_ptr<spectrum_tap> wf_tap;
    std::unique_ptr<channel_impairment> tx_impairment;
    std::unique_ptr<channel_impairment> rx_impairment;
    if (opt.spectrum_enabled) {
        wf_tap = std::make_unique<spectrum_tap>();
        if (*wf_tap) {
            wf_tap->set_rf_params(opt.rx_freq, opt.rate);
            fprintf(stderr, "dvbs2-uhd-trx: spectrum tap → /tmp/dvbs2_spectrum\n");
        } else
            fprintf(stderr, "dvbs2-uhd-trx: WARNING: spectrum tap FIFO failed\n");
    }
    if (opt.noise_enabled && opt.pathloss_enabled) {
        fprintf(stderr, "Error: --noise and --pathloss are mutually exclusive\n");
        return 1;
    }
    if (opt.noise_enabled || opt.pathloss_enabled || opt.propagation_delay_enabled) {
        auto mode = opt.pathloss_enabled ? channel_impairment::PATHLOSS
                                     : channel_impairment::NOISE;
        auto impairment = std::make_unique<channel_impairment>(
            opt.noise_port, mode, 0.001f, opt.rate,
            opt.propagation_delay_enabled, opt.init_distance_km);
        if (*impairment) {
            if (opt.channel_emulation_side == "rx")
                rx_impairment = std::move(impairment);
            else
                tx_impairment = std::move(impairment);
            const char* ch = opt.pathloss_enabled ? "pathloss"
                           : opt.noise_enabled    ? "noise" : "delay-only";
            channel_impairment* active = tx_impairment ? tx_impairment.get()
                                                       : rx_impairment.get();
            fprintf(stderr, "dvbs2-uhd-trx: channel %s on %s samples listening on UDP :%d\n",
                    ch, opt.channel_emulation_side.c_str(), opt.noise_port);
            if (active->delay_enabled())
                fprintf(stderr, "dvbs2-uhd-trx: propagation delay ON, init %.3f km "
                        "(%.1f ms one-way, %zu integer samples @ %.3f Msps), "
                        "512 MiB ring, cap %.0f km\n",
                        active->current_distance_km(),
                        active->current_distance_km() * 1e3 / 299792.458,
                        active->delay_samples(),
                        opt.rate / 1e6, 2.0 * 35786.0);
        } else
            fprintf(stderr, "dvbs2-uhd-trx: WARNING: channel impairment socket failed\n");
    }

    // ====================================================================
    // Benchmark TX mode: measure TX throughput without UHD hardware
    // ====================================================================
    if (opt.benchmark_tx)
        return run_benchmark_tx(opt);

    // USRP args and tune frequencies only make sense for a local radio;
    // the DIFI driver is an IQ-over-UDP data plane (RF head tuned out of
    // band), so none of them are required there.
    if (opt.radio_driver == "uhd") {
        if (opt.usrp_args.empty()) {
            fprintf(stderr, "Error: --usrp is required\n");
            return 1;
        }
        if (opt.do_tx && opt.tx_freq == 0) {
            fprintf(stderr, "Error: --tx-freq is required\n");
            return 1;
        }
        if (opt.do_rx && opt.rx_freq == 0) {
            fprintf(stderr, "Error: --rx-freq is required\n");
            return 1;
        }
    }

    const modcod_def_t* mc_p = find_modcod_by_name(opt.modcod_str.c_str());
    if (!mc_p) {
        fprintf(stderr, "Error: unknown modcod '%s'\n", opt.modcod_str.c_str());
        return 1;
    }
    const modcod_def_t& mc = *mc_p;
    if (opt.manual_modcod && (mc.modcod_idx > 28 || opt.acm_max_modcod < 28)) {
        fprintf(stderr, "Error: manual MODCOD requires the normal DVB-S2 ladder (indices 1..28)\n");
        return 1;
    }
    fecframe_size_t framesize = opt.short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL;

    // Reject an explicitly-selected MODCOD with no support for this frame size
    // (C9/10 has no short FECFRAME; DVB-S2X MODCODs are normal-frame only here).
    if (!modcod_supports_framesize(mc, framesize)) {
        fprintf(stderr, "Error: %s not available with --short-frames (no short FECFRAME support)\n",
                opt.modcod_str.c_str());
        return 1;
    }

    // Install signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // --- Radio setup (driver supplied by a dlopen'd plugin; see
    // trx/radio_loader — keeps GPLv3 UHD out of this Apache-2.0 binary) ---
    // Radio driver selection: --radio-driver <uhd|difi> (or radio.driver in
    // YAML); "--usrp difi:..." is an equivalent alias for the DIFI plugin
    // (IQ rides DIFI/VITA-49.2 UDP from an rfnoc-difi X410 image; UHD is
    // control-plane only there).
    if (opt.usrp_args.rfind("difi:", 0) == 0)
        opt.radio_driver = "difi";
    if (opt.usrp_args.rfind("compfh:", 0) == 0)
        opt.radio_driver = "compfh";
    const bool compressed_driver =
        opt.radio_driver == "difi" || opt.radio_driver == "compfh";
    if ((opt.tx_symbols || opt.tx_modcomp) &&
        (!compressed_driver || opt.sym_rate_hz <= 0.0)) {
        fprintf(stderr, "Error: --tx-symbols/--tx-modcomp need the difi or "
                "compfh radio driver and --sym-rate (the FPGA does the pulse "
                "shaping)\n");
        return 1;
    }
    if (opt.tx_modcomp && opt.acm_mode) {
        fprintf(stderr, "Error: --tx-modcomp is CCM-only for now (per-frame "
                "ACM prologue exists FPGA-side but is not wired here)\n");
        return 1;
    }
    if (opt.tx_modcomp && (opt.short_frames || opt.gold_code != 0 ||
                           std::abs(opt.rolloff - 0.2f) > 1e-6)) {
        fprintf(stderr, "Error: --tx-modcomp requires normal frames, Gold code 0 "
                        "and rolloff 0.2 for the current FPGA ROM/taps\n");
        return 1;
    }
    if (opt.tx_symbols && opt.tx_modcomp) {
        fprintf(stderr, "Error: --tx-symbols and --tx-modcomp are mutually exclusive\n");
        return 1;
    }
    if (opt.tx_modcomp) {
        int mc_nmod = 0, mc_base = 0;
        if (!plmod_geom(mc.modcod_idx, mc_nmod, mc_base)) {
            fprintf(stderr, "Error: --tx-modcomp: modcod '%s' (idx %d) is not in "
                    "the dvb_plmod ROM table (modcod_table.h)\n",
                    opt.modcod_str.c_str(), mc.modcod_idx);
            return 1;
        }
    }
    // Assert-fatal, not just option validation: the OAI usrpproxy shares a
    // LOCAL multi_usrp inside the UHD plugin; over DIFI there is no local
    // radio, so the proxy would silently expose nothing to the softmodem.
    if (opt.radio_driver != "uhd" && opt.usrpproxy_enabled) {
        fprintf(stderr, "dvbs2-uhd-trx: FATAL: usrpproxy requires "
                "radio-driver 'uhd' (got '%s')\n", opt.radio_driver.c_str());
        abort();
    }
    std::unique_ptr<radio_device> radio_dev = load_radio_device(opt.radio_driver);
    if (!radio_dev) {
        fprintf(stderr, "dvbs2-uhd-trx: failed to load radio plugin '%s'\n",
                opt.radio_driver.c_str());
        return 1;
    }
    radio_config rcfg;
    rcfg.usrp_args           = opt.usrp_args;
    rcfg.clock_source        = opt.clock_source;
    rcfg.do_tx               = opt.do_tx;
    rcfg.do_rx               = opt.do_rx;
    rcfg.rate                = opt.rate;
    rcfg.rx_symbol_sync      = opt.rx_symbol_sync;
    rcfg.demo_control_enabled = opt.demo_control_port != 0;
    rcfg.tx_rate             = (opt.tx_symbols || opt.tx_modcomp)
                                   ? opt.sym_rate_hz : 0.0;
    rcfg.tx_freq             = opt.tx_freq;
    rcfg.rx_freq             = opt.rx_freq;
    rcfg.tx_gain             = opt.tx_gain;
    rcfg.rx_gain             = opt.rx_gain;
    rcfg.tx_channel          = opt.tx_channel;
    rcfg.rx_channel          = opt.rx_channel;
    rcfg.uhd_num_send_frames = opt.uhd_num_send_frames;
    rcfg.uhd_send_buff_size  = opt.uhd_send_buff_size;
    rcfg.uhd_send_frame_size = opt.uhd_send_frame_size;
    rcfg.uhd_recv_frame_size = opt.uhd_recv_frame_size;
    rcfg.subdev_spec         = opt.subdev_spec;
    rcfg.usrpproxy_enabled   = opt.usrpproxy_enabled;
    rcfg.usrpproxy_shm_name  = opt.usrpproxy_shm_name;
    rcfg.usrpproxy_tx_antenna = opt.usrpproxy_tx_antenna;
    rcfg.usrpproxy_rx_antenna = opt.usrpproxy_rx_antenna;
    rcfg.usrpproxy_tx_channels = opt.usrpproxy_tx_channels;
    rcfg.usrpproxy_rx_channels = opt.usrpproxy_rx_channels;
    rcfg.usrpproxy_tx_core   = opt.usrpproxy_tx_core;
    rcfg.usrpproxy_rx_core   = opt.usrpproxy_rx_core;
    rcfg.usrpproxy_tx_ring_size = opt.usrpproxy_tx_ring_size;
    rcfg.usrpproxy_rx_ring_size = opt.usrpproxy_rx_ring_size;
    if (!radio_dev->setup(rcfg)) {
        fprintf(stderr, "dvbs2-uhd-trx: radio setup failed\n");
        return 1;
    }
    radio_device* radio = radio_dev.get();

    // TX and RX will start 2.0s from now (after all setup is done).
    // Computed later, after FEC pool init, right before thread launch.
    double stream_start_time = 0.0;


    // --- RX ring buffer ---
    // 2 seconds capacity: gives 1.5s of headroom before the flush threshold
    // fires, enough to absorb transient stalls (FEC bootstrap, lock search).
    // At 20.48 MHz this is 328 MB — much less than the old 3s @ 10.24 (246 MB).
    const size_t rx_ring_size = (size_t)(opt.rate * 2.0);
    spsc_ring<complexf> rx_ring(rx_ring_size);

    // --- TAP interface (optional) ---
    std::unique_ptr<tap_interface> tap;
    if (!opt.tap_name.empty())
        tap = std::make_unique<tap_interface>(opt.tap_name);

    // --- TX DSP pipeline ---
    // When ACM peer is set, use tx_fec_pool (one chain per MODCOD).
    // Otherwise, use a single tx_fec_chain at the configured MODCOD.
    std::unique_ptr<tx_fec_chain> tx_fec;       // CCM mode
    std::unique_ptr<tx_fec_pool> tx_fec_acm;    // ACM mode
    std::unique_ptr<pl_framer> tx_framer;
    std::unique_ptr<transmitter_frontend> tx_frontend;
    std::unique_ptr<gse_encapsulator> tx_enc;

    // For ACM TX: compute max buffer sizes across all enabled MODCODs
    unsigned int acm_max_dfl = 0;
    unsigned int acm_max_xfec = 0;
    unsigned int acm_max_plf = 0;
    bool acm_tx_enabled = opt.multi_modcod();

    if (opt.do_tx) {
        if (acm_tx_enabled) {
            // Build lazy pool: all DVB-S2 MODCODs (QPSK through 32APSK).
            // Chains are NOT allocated here — only built on first get().
            std::vector<uint8_t> tx_modcods;
            for (const auto& m : get_supported_modcods()) {
                if (!modcod_supports_framesize(m, framesize)) continue;
                if (opt.manual_modcod && m.modcod_idx > 28)
                    continue;
                tx_modcods.push_back(m.modcod_idx);
            }
            tx_fec_acm = std::make_unique<tx_fec_pool>(
                framesize, tx_modcods, opt.rolloff);
            // Compute max buffer sizes from DVB-S2 parameters (no chain build)
            for (uint8_t i : tx_modcods) {
                const modcod_def_t* def = find_modcod(i);
                // PLFRAME and XFECFRAME sizes from PLS info (cheap table lookup)
                const pl_layout pi = pl_layout_for(i, framesize, opt.pilots);
                acm_max_xfec = std::max(acm_max_xfec, (unsigned int)pi.data_symbols());
                acm_max_plf = std::max(acm_max_plf, (unsigned int)pi.plframe_symbols());
                // DFL from FEC parameters (cheap table lookup)
                fec_params_t fi;
                fi = get_fec_params(framesize, def->code_rate);
                unsigned int dfl = (fi.k_bch - 80) / 8;  // kbch - BBHEADER bits
                acm_max_dfl = std::max(acm_max_dfl, dfl);
            }
            // Eagerly build the initial MODCOD chain and the signaling
            // chain (QPSK 1/4) to avoid lazy CUDA allocations (cudaMalloc,
            // cudaMemcpy on default stream) at runtime, which would race
            // with the Farrow's default-stream kernel launch.
            tx_fec_acm->get(mc.modcod_idx);
            if (mc.modcod_idx != SIG_MODCOD)
                tx_fec_acm->get(SIG_MODCOD);
        } else {
            tx_fec = std::make_unique<tx_fec_chain>(
                framesize, mc.code_rate, mc.constellation);
            tx_fec->set_rolloff(opt.rolloff);
        }
        tx_framer = std::make_unique<pl_framer>(
            (uint8_t)mc.modcod_idx, framesize == FECFRAME_SHORT, opt.pilots, opt.gold_code);
        tx_frontend = std::make_unique<transmitter_frontend>(opt.sps, opt.rolloff, 10);
        if (tap)
            tx_enc = std::make_unique<gse_encapsulator>(0x0800, GSE_LT_BROADCAST);
    }

    // --- Fused TX CUDA pipeline (mapper → PL frame+scramble → RRC) ---
    // Replaces individual CUDA stages with a single-stream pipeline that
    // keeps intermediate data on GPU, eliminating 3 host round-trips.
    std::unique_ptr<cuda::tx_pipeline_cuda> tx_pipeline;
    if (opt.do_tx) {
        const unsigned int max_plf = acm_tx_enabled
            ? acm_max_plf : tx_framer->plframe_len();
        // Payload = plframe - PLHEADER(90)
        const unsigned int max_payload = max_plf - 90;

        // Symbol fronthaul: sps=1 with a single identity tap degenerates the
        // polyphase RRC into a 1:1 copy — the fused pipeline then emits the
        // scrambled PL symbols (PLHEADER+pilots+payload, PAPR backoff
        // applied) and the FPGA rrc_interp does the pulse shaping with taps
        // generated from the same rrc_pulse() design (rrc_filter.h).
        static const float identity_tap[1] = { 1.0f };
        tx_pipeline = std::make_unique<cuda::tx_pipeline_cuda>(
            opt.tx_symbols ? 1 : tx_frontend->sps(),
            opt.tx_symbols ? 1 : tx_frontend->sub_taps(),
            opt.tx_symbols ? 0 : tx_frontend->history_len(),
            opt.tx_symbols ? identity_tap : tx_frontend->interleaved_taps(),
            reinterpret_cast<const float*>(tx_framer->descrambling_seq()),
            max_payload, max_plf);


        // Register MODCOD configs
        auto register_mc = [&](uint8_t mc_idx) {
            tx_fec_chain& chain = acm_tx_enabled
                ? tx_fec_acm->get(mc_idx) : *tx_fec;
            const mapper& m = chain.get_mapper();
            std::vector<float> lut = m.build_constellation_lut();
            unsigned int col_offsets[8];
            int stride;
            m.get_interleave_params(col_offsets, stride);
            const pl_layout pi = pl_layout_for(mc_idx, framesize, opt.pilots);
            tx_pipeline->register_modcod(
                mc_idx, lut.data(), m.n_mod(),
                m.fecframe_len(), m.xfecframe_len(),
                col_offsets, stride,
                pi.payload_symbols(), pi.plframe_symbols(),
                pi.pilot_block_count(), pi.pilots);
        };

        if (acm_tx_enabled) {
            // Register all ACM MODCODs (builds chains lazily)
            for (const auto& m : get_supported_modcods()) {
                if (!modcod_supports_framesize(m, framesize)) continue;
                if (opt.manual_modcod && m.modcod_idx > 28)
                    continue;
                register_mc(m.modcod_idx);
            }
        } else {
            register_mc(mc.modcod_idx);
        }

        // Prime the ordinary-frame TX path before the timed RF stream starts.
        // Dummy idle bypasses BCH/LDPC/mapping, so without this the first
        // application packet can pay several milliseconds of lazy CUDA setup
        // while the radio is draining its latency buffer.
        const uint8_t warm_mc = (uint8_t)mc.modcod_idx;
        tx_fec_chain& warm_fec = acm_tx_enabled
            ? tx_fec_acm->get(warm_mc) : *tx_fec;
        std::vector<uint8_t> warm_gse(warm_fec.max_dfl_bytes());
        gse_encapsulator::write_padding(warm_gse.data(), warm_gse.size());
        warm_fec.process_gse_to_bch(warm_gse.data(), warm_gse.size());
        pl_framer warm_framer(warm_mc,
                              framesize == FECFRAME_SHORT,
                              opt.pilots, opt.gold_code);
        cudaStream_t warm_stream = tx_pipeline->stream();
        warm_fec.encode_ldpc_to_device(
            tx_pipeline->fecframe_device(), warm_stream);
        tx_pipeline->process_device(
            warm_mc, tx_pipeline->fecframe_device(),
            reinterpret_cast<const float*>(warm_framer.plheader()),
            nullptr, warm_stream);
        CUDA_CHECK(cudaStreamSynchronize(warm_stream));
        tx_pipeline->reset_state();
    }

    // --- TX IQ ring buffer (decouples generation from UHD send) ---
    // Must hold at least 2× the batch output to absorb GPU contention spikes
    // (RX LDPC streams blocking TX CUDA pipeline).  At higher sample rates
    // the ring needs more absolute capacity since each frame is consumed
    // faster — scale to at least 5ms of samples.
    size_t tx_ring_cap = 1;
    if (opt.do_tx) {
        const unsigned int max_plf_len = acm_tx_enabled
            ? acm_max_plf : tx_framer->plframe_len();
        // When Farrow is active, the ring holds hw-rate samples (larger by hw_sps/sps)
        double ring_sps = use_farrow ? hw_sps : (double)opt.sps;
        size_t three_frames = 3 * (size_t)(max_plf_len * ring_sps + 200);
        size_t five_ms = (size_t)(opt.rate * 0.005);  // 5ms at sample rate
        tx_ring_cap = std::max(three_frames, five_ms);
        if (opt.tx_modcomp) {
            // Ring entries hold two packed words, rather than waveform IQ.
            // Bound queueing to a few PLFRAMEs at the compressed representation.
            const size_t words = 3 + 90 +
                                 ((acm_tx_enabled ? tx_fec_acm->get(mc.modcod_idx).fecframe_bits()
                                                  : tx_fec->fecframe_bits()) +
                                  31) /
                                     32 +
                                 1;
            tx_ring_cap = 4 * ((words + 1) / 2);
        }
        if (opt.log_enabled) {
            fprintf(stderr, "dvbs2-uhd-trx: TX IQ ring=%zu samples "
                    "(%.2f ms)\n", tx_ring_cap,
                    (double)tx_ring_cap / opt.rate * 1000.0);
        }
    }
    spsc_ring<complexf> tx_ring(tx_ring_cap);

    // --- RX DSP pipeline ---
    // Delivered PLS codes
    std::vector<uint8_t> deliver_codes;
    std::vector<uint8_t> acm_modcods;
    uint8_t acm_floor_modcod = 0;
    float acm_floor_eff = INFINITY;
    float acm_floor_pilot_snr_db = -100.0f;
    int xfecframe_len = 0;

    if (opt.do_rx) {
        if (opt.multi_modcod()) {
            const auto& all_mc = get_supported_modcods();
            for (const auto& m : all_mc) {
                if (!modcod_supports_framesize(m, framesize)) continue;
                if (opt.manual_modcod && m.modcod_idx > 28)
                    continue;
                // Only accept MODCODs up to the configured max
                if (m.modcod_idx > opt.acm_max_modcod) continue;
                // every accepted MODCOD, DVB-S2X codes (>= 128) included
                deliver_codes.push_back(pls_code_for(m.modcod_idx, framesize, opt.pilots));
                acm_modcods.push_back(m.modcod_idx);
                const bool reserved_inband_sig = acm_tx_enabled &&
                    opt.acm_feedback_dst.empty() &&
                    m.modcod_idx == SIG_MODCOD;
                if (!reserved_inband_sig &&
                    !opt.acm_exclude_modcods.count(m.modcod_idx) &&
                    m.spectral_eff < acm_floor_eff) {
                    acm_floor_eff = m.spectral_eff;
                    acm_floor_modcod = m.modcod_idx;
                    acm_floor_pilot_snr_db =
                        m.min_snr_db + opt.acm_guard_db;
                }
            }
            // in-band signalling MODCOD (QPSK 1/4) is always delivered
            const uint8_t sig_code = pls_code_for(SIG_MODCOD, framesize, opt.pilots);
            if (std::find(deliver_codes.begin(), deliver_codes.end(), sig_code) ==
                deliver_codes.end())
                deliver_codes.push_back(sig_code);
            xfecframe_len = pl_layout_for((uint8_t)1, framesize, opt.pilots).data_symbols();
        } else {
            uint8_t target_plsc = pls_code_for(mc.modcod_idx, framesize, opt.pilots);
            deliver_codes.push_back(target_plsc);
            xfecframe_len = pl_layout_from_code(target_plsc).data_symbols();
        }
    }

    pl_layout frame_format;
    {
        // Use the PLS code helper so DVB-S2X MODCODs (64/128APSK) get their
        // S2X PLS codes (>= 128). A hand-rolled modcod<<2 encoding mis-maps
        // them into the DVB-S2 range -> wrong PLFRAME length -> no high-modcod
        // acquisition (worse at high symbol rate).
        uint8_t plsc = pls_code_for(mc.modcod_idx, framesize, opt.pilots);
        frame_format = pl_layout_from_code(plsc);
        // A real configured MODCOD must map to a payload constellation; a
        // non-deliverable layout is the classic symptom of a mis-encoded
        // S2X PLS code.
        if (frame_format.bits_per_symbol == 0) {
            fprintf(stderr, "Error: MODCOD %s -> PLS %u has no payload "
                    "constellation; PLS encoding is wrong\n",
                    opt.modcod_str.c_str(), plsc);
            return 1;
        }
    }

    std::unique_ptr<receiver_frontend> rx_frontend;
    std::unique_ptr<rx_fec_chain> fec_single;
    std::unique_ptr<rx_fec_pool> fec_multi;
    std::unique_ptr<gse_decapsulator> rx_decap;

    // GPU batch cap: scale so CPU frame sync fits within the block duration.
    // Frame sync on 65K symbols (sps=2) takes ~0.8ms — safe at all rates.
    // At low rates, use larger batches (fewer kernel launches).
    // At high rates, keep per-block symbol count manageable.
    int max_gpu_batch;
    if (opt.rate <= 20e6)
        max_gpu_batch = 524288;   // 51ms @ 10.24 — plenty of headroom
    else if (opt.rate <= 70e6)
        max_gpu_batch = 262144;   // 4.3ms @ 61.44
    else
        max_gpu_batch = 131072;   // 1.1ms @ 122.88, 0.53ms @ 245.76
    if (opt.gpu_batch_ms > 0) {
        max_gpu_batch = std::max(131072, (int)(opt.rate * opt.gpu_batch_ms / 1000.0));
    }
    if (opt.log_enabled && opt.do_rx) {
        fprintf(stderr, "GPU batch cap: %d samples (%.2f ms at %.2f MHz)\n",
                max_gpu_batch, max_gpu_batch / opt.rate * 1000.0, opt.rate / 1e6);
    }

    // Serialize CUDA API calls between all GPU threads.
    // The CUDA driver on GH200/ARM64 SIGSEGVs when concurrent CUDA API
    // calls enter the driver from different threads.
    std::mutex cuda_mtx;

    // Shared thread context: atomic counters live in ctx.stats; pointers to
    // the objects constructed below are wired in right before thread launch.
    trx_context ctx;

    // LDPC decode batch: at high rates the frame rate outruns the
    // worst-case decode throughput of small batches (GH200: RATE_3_5 ~3.4k
    // fr/s at batch 16 vs ~6.6k at 64; 122.88 Msym qpsk needs 3.8k).
    // Larger batches also drain the lock-transition backlog faster
    // (measured at 245.76: cumulative FER 4.8e-3 -> 1.6e-4/0.0).
    // Keep 16 at low rates (batch fill adds latency to ACM feedback).
    //
    // If decoder throughput wave-quantizes at the SM count (one codeword
    // per SM), a batch that is not a multiple of the SM count wastes a
    // wave. Align the auto batch to the device SM count.
    if (opt.gpu_ldpc_batch == 16 && opt.rate > 150e6) {
        int sms = 64;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        opt.gpu_ldpc_batch = std::max(16, std::min(sms, 132));
        fprintf(stderr, "dvbs2-uhd-trx: LDPC batch auto -> %d (device has "
                "%d SMs; 1 codeword/SM)\n", opt.gpu_ldpc_batch, sms);
    }

    // NOTE on trials: opt.ldpc_trials defaults to 10 — the worst-case
    // decode floor in the field is already ~2.5x the trials-25 bench
    // numbers, at a measured AWGN-waterfall cost of only a few hundredths
    // of a dB (bench-ldpc-decode --ebn0 at --trials 25/15/12).

    if (opt.do_rx) {
        // The frontend processes hardware-rate samples directly: its
        // polyphase matched filter and timing NCO are designed for hw_sps.
        // Below ~2.5 samples/symbol the truncated prototype's residual
        // in-band ripple becomes comparable to the shrinking eye-energy
        // contrast, so use a longer RRC prototype there (the polyphase
        // subfilters grow from ~21 to ~64 taps — negligible GPU cost).
        int rrc_delay = (hw_sps < 2.5) ? 16 : 5;
        if (opt.rx_prefiltered)
            fprintf(stderr, "rx: functional-split rung 2 — RU pre-applied the RRC "
                    "matched filter; frontend uses a neutral windowed-sinc "
                    "interpolator at hw_sps=%.3f (no host RRC)\n", hw_sps);
        if (opt.rx_symbol_sync)
            fprintf(stderr, "rx: functional-split rung 3 — RU recovered symbol "
                    "timing (rrc_rx mode 2); frontend runs symbols-in (no host "
                    "AGC/timing, 1 sample/symbol)\n");
        receiver_frontend_config fe_cfg;
        fe_cfg.sps = (float)hw_sps;
        fe_cfg.rolloff = opt.rolloff;
        fe_cfg.rrc_delay = rrc_delay;
        fe_cfg.n_phases = 128;
        fe_cfg.prefiltered = opt.rx_prefiltered;
        fe_cfg.symbols_in = opt.rx_symbol_sync;
        fe_cfg.max_gpu_batch = max_gpu_batch;
        fe_cfg.agc_rate = 1e-5f;
        fe_cfg.agc_ref = 1.0f;
        fe_cfg.agc_initial_gain = 1.0f;
        fe_cfg.agc_max_gain = 65536.0f;
        fe_cfg.debug_level = opt.debug_level;
        fe_cfg.pl.mode = opt.multi_modcod() ? pl_service_mode::vcm_acm
                                            : pl_service_mode::ccm;
        fe_cfg.pl.deliver_codes = deliver_codes;
        fe_cfg.pl.scrambling_code = opt.gold_code;
        fe_cfg.pl.debug_level = opt.debug_level;
        rx_frontend = std::make_unique<receiver_frontend>(fe_cfg);
        rx_frontend->set_timing_estimation_period(opt.timing_est_period);
        rx_frontend->set_energy_k(opt.energy_k);
        rx_frontend->set_energy_search_cap(opt.energy_search_cap);
        rx_frontend->set_batch_plframes(opt.rx_batch_plframes);
        // Locked-path min batch floor: at high frame rates (small PLFRAMEs,
        // e.g. 16apsk@185 Msym = 11k fr/s) the per-batch GPU sync wait
        // (~300 us under LDPC load) exceeds the 2-PLFRAME real-time budget
        // (180 us) and the frontend falls behind. A 500 us floor keeps the
        // budget above the wait; the timing loop already tracks at 360 us
        // updates (qpsk@185 OTA-validated), so the latency cost is benign.
        {
            int mb_us = opt.rx_min_batch_us;
            if (mb_us < 0) mb_us = (opt.rate > 150e6) ? 500 : 0;
            if (mb_us > 0) {
                int mb = (int)(opt.rate * mb_us * 1e-6);
                rx_frontend->set_min_batch_samples(mb);
                fprintf(stderr, "rx: locked-path min batch %d us = %d samples "
                        "(budget floor vs GPU sync wait)\n", mb_us, mb);
            }
        }
        if (opt.multi_modcod()) {
            fec_multi = std::make_unique<rx_fec_pool>(
                framesize, acm_modcods, opt.ldpc_trials,
                opt.gpu_ldpc_batch, opt.gpu_streams, opt.micro_batch);
        } else {
            fec_single = std::make_unique<rx_fec_chain>(
                framesize, mc.code_rate, mc.constellation,
                opt.ldpc_trials, opt.gpu_ldpc_batch,
                opt.gpu_streams, opt.micro_batch);
        }

        // In ACM mode, build whitelist of valid PLFRAME lengths so the
        // receiver accepts all configured MODCODs + signaling without
        // rejecting legitimate transitions as PLSC misdecodes.
        if (opt.multi_modcod()) {
            std::set<uint32_t> valid_lens;
            for (const auto& m : get_supported_modcods()) {
                if (!modcod_supports_framesize(m, framesize)) continue;
                if (opt.manual_modcod && m.modcod_idx > 28)
                    continue;
                if (m.modcod_idx > opt.acm_max_modcod) continue;
                valid_lens.insert(
                    pl_layout_for(m.modcod_idx, framesize, opt.pilots).plframe_symbols());
            }
            // Also include signaling MODCOD (QPSK 1/4)
            valid_lens.insert(pl_layout_for(SIG_MODCOD, framesize, opt.pilots).plframe_symbols());
            rx_frontend->set_valid_plframe_lens(valid_lens);
            fprintf(stderr, "dvbs2-uhd-trx: ACM valid PLFRAME lengths:");
            for (uint32_t l : valid_lens)
                fprintf(stderr, " %u", l);
            fprintf(stderr, "\n");
            // Short frames only: hint the smallest expected PLFRAME so the
            // pre-lock acquisition batch is sub-PLFRAME (small short frames
            // otherwise never lock). Normal frames leave the hint unset so the
            // acquisition path is unchanged from the high-Msym-validated one.
            if (opt.short_frames && !valid_lens.empty())
                rx_frontend->set_acq_plframe_hint(*valid_lens.begin());
        }

        // CCM short frames: hint the single MODCOD's PLFRAME length (same
        // pre-lock sub-PLFRAME sizing). Normal CCM leaves the hint unset.
        if (!opt.multi_modcod() && opt.short_frames) {
            rx_frontend->set_acq_plframe_hint(
                pl_layout_for(mc.modcod_idx, FECFRAME_SHORT, opt.pilots).plframe_symbols());
        }

        rx_frontend->set_cuda_mutex(&cuda_mtx);
        if (fec_single)
            fec_single->set_cuda_mutex(&cuda_mtx);
        if (fec_multi)
            fec_multi->set_cuda_mutex(&cuda_mtx);

        // Prime lazy CUDA decoder setup before the timed RF stream starts.
        // This is especially important for low-latency TX buffering because
        // RX and TX intentionally serialize CUDA API launches on ARM64.
        if (fec_single)
            fec_single->warmup();
        if (fec_multi)
            fec_multi->warmup_all();

        if (tap)
            rx_decap = std::make_unique<gse_decapsulator>();

        // CPU frontend handles pilot-interpolated phase correction.
        // Pass diagnostic flags to components
        if (fec_single) {
            fec_single->set_diag_fec_bench(opt.diag_fec_bench);
            fec_single->set_diag_llr_histogram(opt.diag_llr_histogram);
            fec_single->set_diag_ldpc_convergence(opt.diag_ldpc_convergence);
        }
        if (fec_multi) {
            fec_multi->set_diag_fec_bench(opt.diag_fec_bench);
            fec_multi->set_diag_llr_histogram(opt.diag_llr_histogram);
            fec_multi->set_diag_ldpc_convergence(opt.diag_ldpc_convergence);
        }
        rx_frontend->set_diag_carrier(opt.diag_carrier);
        rx_frontend->set_diag_rx(opt.diag_rx);
    }

    // --- Counters (live in ctx.stats; hoisted references keep names) ---
    auto& tx_frames = ctx.stats.tx_frames;
    auto& tx_dummy_frames = ctx.stats.tx_dummy_frames;
    auto& rx_fec_frames = ctx.stats.rx_fec_frames;     // post-bootstrap only
    auto& rx_fec_errors = ctx.stats.rx_fec_errors;     // post-bootstrap only
    auto& rx_pdus = ctx.stats.rx_pdus;
    auto& rx_pdu_bytes = ctx.stats.rx_pdu_bytes;
    auto& tx_underflows = ctx.stats.tx_underflows;
    auto& tx_sequence_errors = ctx.stats.tx_sequence_errors;
    auto& tx_time_errors = ctx.stats.tx_time_errors;
    auto& rx_uhd_overflows = ctx.stats.rx_uhd_overflows;
    auto& rx_ring_drops = ctx.stats.rx_ring_drops;
    auto& tx_pdu_queue_drops = ctx.stats.tx_pdu_queue_drops;
    auto& rx_fec_queue_drops = ctx.stats.rx_fec_queue_drops;

    // EWMA FER (α=0.02, ~50-frame time constant)
    // Atomic: written by FEC thread, read by RX frontend thread (FER watchdog).
    auto& ewma_fer = ctx.stats.ewma_fer;

    // GPU sync point counters (for benchmark summary)
    auto& sync_count_tx = ctx.stats.sync_count_tx;
    auto& sync_count_rx = ctx.stats.sync_count_rx;

    // ACM diagnostic counters
    auto& acm_frames_submitted = ctx.stats.acm_frames_submitted;
    auto& acm_frames_skipped_len = ctx.stats.acm_frames_skipped_len;
    auto& acm_frames_dummy = ctx.stats.acm_frames_dummy;
    auto& acm_feedback_tx = ctx.stats.acm_feedback_tx;
    auto& acm_feedback_rx = ctx.stats.acm_feedback_rx;
    auto& acm_feedback_recovery_bursts =
        ctx.stats.acm_feedback_recovery_bursts;
    // Per-modcod frame counts (modcod 0..31), excludes ACM signaling frames
    auto& acm_mc_hist = ctx.stats.acm_mc_hist;
    auto& acm_mc_fec_frames = ctx.stats.acm_mc_fec_frames;
    auto& acm_mc_fec_errors = ctx.stats.acm_mc_fec_errors;
    auto& acm_pls_guard_corrected = ctx.stats.acm_pls_guard_corrected;
    auto& acm_pls_guard_deferred = ctx.stats.acm_pls_guard_deferred;
    auto& acm_pls_guard_transitions = ctx.stats.acm_pls_guard_transitions;
    auto& acm_pls_guard_outage_frames =
        ctx.stats.acm_pls_guard_outage_frames;
    // Last non-signaling RX MODCOD (for dashboard display)
    auto& rx_data_modcod = ctx.stats.rx_data_modcod;
    rx_data_modcod.store((uint8_t)mc.modcod_idx, std::memory_order_relaxed);

    // Latency instrumentation (see trx_stats for field documentation)
    auto& lat_rx_ring_samples = ctx.stats.lat_rx_ring_samples;
    auto& lat_fec_decode_us = ctx.stats.lat_fec_decode_us;
    auto& lat_frontend_us = ctx.stats.lat_frontend_us;
    auto& lat_tx_gen_us = ctx.stats.lat_tx_gen_us;
    auto& lat_tx_gse_us = ctx.stats.lat_tx_gse_us;
    auto& lat_tx_bb_bch_us = ctx.stats.lat_tx_bb_bch_us;
    auto& lat_tx_ldpc_us = ctx.stats.lat_tx_ldpc_us;
    auto& lat_tx_gpu_us = ctx.stats.lat_tx_gpu_us;
    auto& lat_batch_wait_us = ctx.stats.lat_batch_wait_us;
    auto& lat_tx_loop_us = ctx.stats.lat_tx_loop_us;
    auto& lat_rx_loop_us = ctx.stats.lat_rx_loop_us;
    auto& lat_gpu_drain_us = ctx.stats.lat_gpu_drain_us;
    auto& lat_gpu_phase_us = ctx.stats.lat_gpu_phase_us;
    auto& lat_gpu_submit_us = ctx.stats.lat_gpu_submit_us;
    auto& gpu_inflight_count = ctx.stats.gpu_inflight_count;
    auto& lat_fec_loop_us = ctx.stats.lat_fec_loop_us;
    auto& gpu_force_drain_count = ctx.stats.gpu_force_drain_count;

    // Frontend sub-stage breakdown (from receiver_frontend::perf())
    auto& lat_fe_agc_us = ctx.stats.lat_fe_agc_us;
    auto& lat_fe_timing_us = ctx.stats.lat_fe_timing_us;
    auto& lat_fe_sync_us = ctx.stats.lat_fe_sync_us;
    auto& lat_fe_n_in = ctx.stats.lat_fe_n_in;
    auto& lat_fe_n_syms = ctx.stats.lat_fe_n_syms;
    auto& lat_fe_fused = ctx.stats.lat_fe_fused;
    // timing recovery sub-stage breakdown
    auto& lat_tr_memcpy_us = ctx.stats.lat_tr_memcpy_us;
    auto& lat_tr_launch_us = ctx.stats.lat_tr_launch_us;
    auto& lat_tr_sync_us = ctx.stats.lat_tr_sync_us;
    auto& lat_tr_readback_us = ctx.stats.lat_tr_readback_us;
    auto& lat_gpu_timing_us = ctx.stats.lat_gpu_timing_us;
    auto& lat_gpu_interp_us = ctx.stats.lat_gpu_interp_us;

    auto& lat_tx_ring_wait_us = ctx.stats.lat_tx_ring_wait_us;
    auto& lat_tx_batch_size = ctx.stats.lat_tx_batch_size;

    // ACM TX control: main thread writes, TX thread reads
    auto& acm_tx_modcod = ctx.stats.acm_tx_modcod;
    auto& acm_tx_data_modcod = ctx.stats.acm_tx_data_modcod;
    auto& acm_control_updates = ctx.stats.acm_control_updates;
    acm_tx_modcod.store((uint8_t)mc.modcod_idx, std::memory_order_relaxed);
    acm_tx_data_modcod.store((uint8_t)mc.modcod_idx,
                             std::memory_order_relaxed);

    // In-band ACM signaling: RX FEC thread → main thread (peer's reported SNR)
    auto& acm_inband_snr = ctx.stats.acm_inband_snr;
    auto& acm_inband_fer = ctx.stats.acm_inband_fer;
    auto& acm_inband_fec_frames = ctx.stats.acm_inband_fec_frames;
    auto& acm_inband_fec_errors = ctx.stats.acm_inband_fec_errors;
    auto& acm_inband_rx_modcod = ctx.stats.acm_inband_rx_modcod;

    // ACM controller (used by main thread only)
    std::unique_ptr<acm_controller> acm_ctrl;
    if (opt.acm_mode) {
        // Build ACM controller from MODCODs valid for the active frame size
        // (modcod_supports_framesize excludes C9/10 and all DVB-S2X MODCODs
        // under --short-frames, so short ACM never signals a normal-only MODCOD).
        std::vector<modcod_def_t> ctrl_modcods;
        for (const auto& m : get_supported_modcods()) {
            if (!modcod_supports_framesize(m, framesize)) continue;
            if (opt.manual_modcod && m.modcod_idx > 28)
                continue;
            if (m.modcod_idx > opt.acm_max_modcod) continue;
            if (opt.acm_exclude_modcods.count(m.modcod_idx)) continue;
            // QPSK 1/4 carries robust in-band ACM signaling. Reserve it from
            // the data ladder so its decoder outcomes cannot be confused with
            // application-mode outcomes. Out-of-band feedback may use it as
            // an ordinary data MODCOD.
            if (opt.acm_feedback_listen == 0 &&
                m.modcod_idx == SIG_MODCOD) continue;
            ctrl_modcods.push_back(m);
        }
        if (ctrl_modcods.empty()) {
            fprintf(stderr,
                    "dvbs2-uhd-trx: ACM configuration has no usable data "
                    "MODCOD (max_modcod=%d; QPSK 1/4 is reserved for "
                    "in-band feedback)\n",
                    opt.acm_max_modcod);
            return 1;
        }
        acm_ctrl = std::make_unique<acm_controller>(
            ctrl_modcods, static_cast<float>(opt.acm_guard_db),
            static_cast<float>(opt.acm_bootstrap_margin_db));
        acm_tx_data_modcod.store(acm_ctrl->validated_modcod_idx(),
                                 std::memory_order_relaxed);
    }

    // Optional out-of-band ACM feedback for split or radio-free deployments.
    int acm_fb_rx_fd = -1, acm_fb_tx_fd = -1;
    sockaddr_in acm_fb_dst{};
    if (opt.acm_feedback_listen > 0) {
        acm_fb_rx_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons((uint16_t)opt.acm_feedback_listen);
        if (acm_fb_rx_fd < 0 || bind(acm_fb_rx_fd,
                reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            perror("ACM feedback bind");
            return 1;
        }
        fprintf(stderr, "dvbs2-uhd-trx: ACM UDP feedback listening on %d\n",
                opt.acm_feedback_listen);
    }
    if (!opt.acm_feedback_dst.empty()) {
        auto colon = opt.acm_feedback_dst.rfind(':');
        if (colon == std::string::npos) {
            fprintf(stderr, "Error: --acm-feedback-dst requires HOST:PORT\n");
            return 1;
        }
        std::string host = opt.acm_feedback_dst.substr(0, colon);
        int port = std::stoi(opt.acm_feedback_dst.substr(colon + 1));
        acm_fb_tx_fd = socket(AF_INET, SOCK_DGRAM, 0);
        acm_fb_dst.sin_family = AF_INET;
        acm_fb_dst.sin_port = htons((uint16_t)port);
        if (acm_fb_tx_fd < 0 || port <= 0 || port > 65535 ||
            inet_pton(AF_INET, host.c_str(), &acm_fb_dst.sin_addr) != 1) {
            fprintf(stderr, "Error: invalid --acm-feedback-dst '%s'\n",
                    opt.acm_feedback_dst.c_str());
            return 1;
        }
        fprintf(stderr, "dvbs2-uhd-trx: ACM UDP feedback destination %s\n",
                opt.acm_feedback_dst.c_str());
    }

    auto start_time = std::chrono::steady_clock::now();

    // RX frame queue: frontend thread → FEC thread (SPSC, lock-free)
    // Max XFECFRAME len across all MODCODs: QPSK normal = 32400
    // Keep capacity moderate to bound worst-case latency (64 × 1ms = 64ms at 122.88 MHz)
    const int max_xfec_for_queue = 32400;
    spsc_frame_queue frame_queue(256, max_xfec_for_queue);

    // ====================================================================
    // TAP reader thread: see trx/tap_thread.cc (tap_reader_thread)
    // ====================================================================
    pdu_queue tap_pdus;

    // ====================================================================
    // Thread 1a: TX gen + Thread 1b: TX send — see trx/tx_threads.cc
    // (tx_gen_thread, tx_send_thread)
    // Thread 3: RX UHD, Thread 4: RX frontend, Thread 5: RX FEC
    // — see trx/rx_threads.cc (rx_uhd_thread, rx_frontend_thread,
    //   rx_fec_thread)
    // ====================================================================

    // ====================================================================
    // Wire up the shared thread context (everything is constructed by now)
    // ====================================================================
    ctx.opt = &opt;
    ctx.framesize = framesize;
    ctx.acm_tx_enabled = acm_tx_enabled;
    ctx.acm_max_dfl = acm_max_dfl;
    ctx.acm_max_xfec = acm_max_xfec;
    ctx.acm_max_plf = acm_max_plf;
    ctx.use_farrow = use_farrow;
    ctx.hw_sps = hw_sps;
    ctx.max_gpu_batch = max_gpu_batch;
    ctx.xfecframe_len = xfecframe_len;
    ctx.acm_floor_modcod = acm_floor_modcod;
    ctx.acm_floor_pilot_snr_db = acm_floor_pilot_snr_db;
    ctx.tx_ring_cap = tx_ring_cap;
    ctx.frame_format = &frame_format;
    ctx.start_time = start_time;
    ctx.radio = radio;
    ctx.stream_start_time = &stream_start_time;
    ctx.rx_ring = &rx_ring;
    ctx.tx_ring = &tx_ring;
    ctx.frame_queue = &frame_queue;
    ctx.tap_pdus = &tap_pdus;
    ctx.tap = &tap;
    ctx.wf_tap = &wf_tap;
    ctx.tx_impairment = &tx_impairment;
    ctx.rx_impairment = &rx_impairment;
    ctx.tx_fec = &tx_fec;
    ctx.tx_fec_acm = &tx_fec_acm;
    ctx.tx_framer = &tx_framer;
    ctx.tx_frontend = &tx_frontend;
    ctx.tx_enc = &tx_enc;
    ctx.tx_pipeline = &tx_pipeline;
    ctx.tx_farrow = &tx_farrow;
    ctx.rx_frontend = &rx_frontend;
    ctx.fec_single = &fec_single;
    ctx.fec_multi = &fec_multi;
    ctx.rx_decap = &rx_decap;
    ctx.cuda_mtx = &cuda_mtx;

    // ====================================================================
    // Launch threads
    // ====================================================================
    if (opt.sym_rate_hz > 0.0)
        fprintf(stderr, "dvbs2-uhd-trx: starting (modcod=%s sps=%d hw_sps=%.4f sym_rate=%.0f%s%s)\n",
                opt.modcod_str.c_str(), opt.sps, hw_sps, opt.sym_rate_hz,
                opt.do_tx ? " TX" : "", opt.do_rx ? " RX" : "");
    else
        fprintf(stderr, "dvbs2-uhd-trx: starting (modcod=%s sps=%d%s%s)\n",
                opt.modcod_str.c_str(), opt.sps,
                opt.do_tx ? " TX" : "", opt.do_rx ? " RX" : "");

    // Set stream start time to 2s from now (after all setup is done)
    stream_start_time = radio->get_time_now() + 2.0;
    fprintf(stderr, "dvbs2-uhd-trx: stream start at t=%.3f (now=%.3f)\n",
            stream_start_time, radio->get_time_now());

    std::thread t_tap_reader(tap_reader_thread, std::ref(ctx));
    std::thread t_tx_gen(tx_gen_thread, std::ref(ctx));
    std::thread t_tx_send(tx_send_thread, std::ref(ctx));
    std::thread t_rx_uhd(rx_uhd_thread, std::ref(ctx));
    std::thread t_rx_frontend(rx_frontend_thread, std::ref(ctx));
    std::thread t_rx_fec(rx_fec_thread, std::ref(ctx));

    // ====================================================================
    // Main thread: periodic logging + ACM feedback
    // ====================================================================

    if (opt.acm_mode)
        fprintf(stderr, "dvbs2-uhd-trx: ACM SNR-based enabled "
                "(QPSK 1/4 @ 2 Hz, target post-BCH FER=1e-5, "
                "guard=%.2f dB, bootstrap_margin=%.2f dB, "
                "SNR EWMA alpha=%.3f, max_modcod=%d)\n",
                opt.acm_guard_db, opt.acm_bootstrap_margin_db,
                opt.acm_snr_ewma_alpha,
                opt.acm_max_modcod);

    // GPU memory diagnostic
    {
        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);
        fprintf(stderr, "dvbs2-uhd-trx: GPU memory: %.1f MB used / %.1f MB total "
                "(%.1f MB free)\n",
                (gpu_total - gpu_free) / 1e6, gpu_total / 1e6, gpu_free / 1e6);
    }

    auto last_log = start_time;
    bench_accum bench;       // benchmark accumulator (sampled each log cycle)
    bool bench_started = false;  // skip first few seconds to avoid startup transients
    std::chrono::steady_clock::time_point bench_start_time;
    uint64_t bench_prev_tx_frames = 0;
    uint64_t bench_prev_rx_frames = 0;

    // In-band ACM feedback timeout: if no feedback for 3s, peer can't decode us
    auto last_inband_fb = start_time;
    constexpr double ACM_INBAND_TIMEOUT = 3.0; // seconds

    // The estimator remains unsmoothed for observability.  This tracker is
    // the only temporal SNR filter used by ACM, and also derives telemetry
    // variance directly from the unsmoothed input samples.
    snr_ewma snr_tracker(static_cast<float>(opt.acm_snr_ewma_alpha));
    auto& acm_rx_snr_ewma = ctx.stats.acm_rx_snr_ewma;

    // Throughput EWMA (PDU bytes → Mbps)
    float throughput_ewma = 0.0f;
    uint64_t prev_pdu_bytes = 0;
    auto prev_tput_time = start_time;
    constexpr float TPUT_EWMA_ALPHA = 0.3f; // ~3-sample time constant at 1s

    // Degraded-RF detector: consecutive ~1s log cycles locked with
    // pilot coherence < 0.3 (per-init X410 RF flakiness at MCR 245.76)
    int low_coh_secs = 0;

    // Setup is done; from here a CUDA fault is unrecoverable in-process
    // (poisoned context), so exit cleanly (code 42) instead of abort() — a
    // supervisor (scripts/trx-supervised.sh) restarts us and the RX re-acquires.
    cusat_dsp::cuda::set_cuda_fault_graceful(true);

    int demo_fd = -1;
    if (opt.demo_control_port) {
        demo_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons(opt.demo_control_port);
        if (demo_fd < 0 || bind(demo_fd, (sockaddr *)&a, sizeof(a)) < 0) {
            fprintf(stderr, "demo: cannot bind control endpoint\n");
            g_stop.store(true);
        }
    }

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto now = std::chrono::steady_clock::now();
        if (demo_fd >= 0) {
            char b[128];
            sockaddr_in peer{};
            socklen_t plen = sizeof(peer);
            const auto len =
                recvfrom(demo_fd, b, sizeof(b) - 1, MSG_DONTWAIT, (sockaddr *)&peer, &plen);
            if (len > 0) {
                b[len] = 0;
                std::istringstream in(b);
                std::string op, extra;
                in >> op;
                std::string reply = "{\"ok\":false,\"error\":\"unsupported request\"}";
                if (op == "modcod" && opt.manual_modcod) {
                    int requested = -1;
                    in >> requested;
                    if (in && !(in >> extra) && requested >= 1 && requested <= 28) {
                        acm_tx_data_modcod.store(requested);
                        acm_tx_modcod.store(requested, std::memory_order_release);
                        reply = "{\"ok\":true,\"pending\":true}";
                    }
                } else if (op == "stats" && !(in >> extra)) {
                    auto rs = radio->live_stats();
                    const float snr = opt.do_rx ? rx_frontend->pilot_snr_db() : NAN;
                    const float noise = rx_impairment ? rx_impairment->current_db() : 0;
                    std::ostringstream o;
                    o << "{\"ok\":true,\"pid\":" << getpid()
                      << ",\"uptime\":" << std::chrono::duration<double>(now - start_time).count()
                      << ",\"locked\":" << (opt.do_rx && rx_frontend->locked() ? "true" : "false")
                      << ",\"snr_db\":" << (std::isfinite(snr) ? std::to_string(snr) : "null")
                      << ",\"noise_parameter\":" << noise
                      << ",\"tx_modcod\":" << int(ctx.stats.manual_applied_modcod.load())
                      << ",\"requested_modcod\":" << int(acm_tx_modcod.load())
                      << ",\"rx_modcod\":" << int(rx_data_modcod.load())
                      << ",\"fec_frames\":" << rx_fec_frames.load()
                      << ",\"fec_errors\":" << rx_fec_errors.load()
                      << ",\"pdu_bytes\":" << rx_pdu_bytes.load()
                      << ",\"rx_wire_bytes\":" << rs.rx_wire_bytes
                      << ",\"tx_wire_bytes\":" << rs.tx_wire_bytes
                      << ",\"rx_packets\":" << rs.rx_packets << ",\"compression_mode\":" << rs.mode
                      << ",\"shift\":" << rs.shift << ",\"dbits\":" << rs.dbits
                      << ",\"switches\":" << rs.switches << ",\"clips\":" << rs.clips
                      << ",\"underflows\":" << tx_underflows.load()
                      << ",\"overflows\":" << rx_uhd_overflows.load()
                      << ",\"ring_drops\":" << rx_ring_drops.load() << "}";
                    reply = o.str();
                }
                sendto(demo_fd, reply.data(), reply.size(), 0, (sockaddr *)&peer, plen);
            }
        }

        if (acm_fb_rx_fd >= 0) {
            acm_feedback_pkt_t fb{};
            while (recv(acm_fb_rx_fd, &fb, sizeof(fb), 0) == sizeof(fb)) {
                if (fb.magic == ACM_FEEDBACK_MAGIC) {
                    acm_inband_fec_frames.store(fb.fec_frames,
                                                std::memory_order_relaxed);
                    acm_inband_fec_errors.store(fb.fec_errors,
                                                std::memory_order_relaxed);
                    acm_inband_rx_modcod.store(fb.rx_modcod,
                                               std::memory_order_relaxed);
                    acm_inband_fer.store(fb.fer,
                                         std::memory_order_relaxed);
                    acm_inband_snr.store(fb.snr_db, std::memory_order_release);
                }
            }
        }

        // --- Update ACM SNR EWMA every iteration (200 ms) ---
        if (opt.do_rx && rx_frontend->locked() &&
            rx_frontend->has_pilot_snr()) {
            snr_tracker.update(rx_frontend->pilot_snr_db());
            acm_rx_snr_ewma.store(snr_tracker.mean(),
                                  std::memory_order_release);
        } else {
            snr_tracker.reset();
            acm_rx_snr_ewma.store(NAN, std::memory_order_release);
        }

        if (acm_fb_tx_fd >= 0) {
            acm_feedback_pkt_t fb{};
            fb.magic = ACM_FEEDBACK_MAGIC;
            fb.fer = ewma_fer.load(std::memory_order_relaxed);
            const float control_snr = acm_rx_snr_ewma.load(
                std::memory_order_acquire);
            fb.snr_db = std::isfinite(control_snr) ? control_snr : -99.0f;
            const uint8_t data_mc =
                rx_data_modcod.load(std::memory_order_relaxed);
            fb.rx_modcod = data_mc;
            fb.fec_frames =
                acm_mc_fec_frames[data_mc].load(std::memory_order_relaxed);
            fb.fec_errors =
                acm_mc_fec_errors[data_mc].load(std::memory_order_relaxed);
            sendto(acm_fb_tx_fd, &fb, sizeof(fb), 0,
                   reinterpret_cast<sockaddr*>(&acm_fb_dst), sizeof(acm_fb_dst));
        }

        // --- In-band ACM feedback: use peer's reported pilot SNR ---
        if (acm_ctrl) {
            float peer_snr = acm_inband_snr.exchange(NAN, std::memory_order_acquire);
            if (!std::isnan(peer_snr)) {
                last_inband_fb = now;
                float peer_fer = acm_inband_fer.exchange(
                    0.0f, std::memory_order_acquire);
                uint64_t peer_frames = acm_inband_fec_frames.load(
                    std::memory_order_relaxed);
                uint64_t peer_errors = acm_inband_fec_errors.load(
                    std::memory_order_relaxed);
                uint8_t peer_modcod = acm_inband_rx_modcod.load(
                    std::memory_order_relaxed);
                const bool reserved_inband_sig =
                    opt.acm_feedback_listen == 0 &&
                    peer_modcod == SIG_MODCOD;
                uint8_t new_mc = reserved_inband_sig
                    ? acm_ctrl->update_snr(peer_snr, peer_fer)
                    : acm_ctrl->update_feedback(
                        peer_snr, peer_fer, peer_frames, peer_errors,
                        peer_modcod);
                uint8_t old_mc = acm_tx_modcod.load(std::memory_order_relaxed);
                if (new_mc != old_mc) {
                    const modcod_def_t* def = find_modcod(new_mc);
                    fprintf(stderr, "dvbs2-uhd-trx: ACM TX modcod %d→%d (%s) "
                            "(peer_snr=%.1f)\n",
                            old_mc, new_mc,
                            def ? def->name : "?", peer_snr);
                }
                acm_tx_modcod.store(new_mc, std::memory_order_relaxed);
                acm_tx_data_modcod.store(
                    acm_ctrl->validated_modcod_idx(),
                    std::memory_order_relaxed);
                acm_control_updates.fetch_add(1, std::memory_order_release);
            } else {
                // No feedback received this cycle — check for timeout.
                // If peer can't decode our signal (too high MODCOD), the
                // in-band signaling frames won't reach us. Drop to floor
                // so the peer can at least decode QPSK 1/4 and resume feedback.
                double since_fb = std::chrono::duration<double>(
                    now - last_inband_fb).count();
                if (since_fb >= ACM_INBAND_TIMEOUT &&
                    acm_ctrl->current_modcod_idx() > SIG_MODCOD) {
                    uint8_t new_mc = acm_ctrl->drop_to_floor();
                    uint8_t old_mc = acm_tx_modcod.load(std::memory_order_relaxed);
                    if (new_mc != old_mc) {
                        fprintf(stderr, "dvbs2-uhd-trx: ACM feedback timeout "
                                "(%.1fs) — drop to floor %d→%d\n",
                                since_fb, old_mc, new_mc);
                        acm_tx_modcod.store(new_mc, std::memory_order_relaxed);
                        acm_tx_data_modcod.store(
                            acm_ctrl->validated_modcod_idx(),
                            std::memory_order_relaxed);
                    }
                    last_inband_fb = now; // reset so we don't spam
                }
            }
        }

        if (!opt.log_enabled) continue;

        double log_ms = std::chrono::duration<double, std::milli>(now - last_log).count();
        if (log_ms < 1000.0) continue;
        last_log = now;

        double elapsed = std::chrono::duration<double>(now - start_time).count();
        uint64_t fc = rx_fec_frames.load();
        uint64_t fe = rx_fec_errors.load();
        float pilot_snr = opt.do_rx ? rx_frontend->pilot_snr_db() : 0;
        float acm_snr = acm_rx_snr_ewma.load(std::memory_order_acquire);
        float snr_std = snr_tracker.stddev();
        float data_snr = 0;
        if (opt.do_rx) {
            if (opt.multi_modcod()) {
                data_snr = fec_multi->data_snr_db();
            } else {
                data_snr = fec_single->data_snr_db();
            }
        }

        // Update throughput EWMA (once per log cycle, ~1s)
        {
            uint64_t cur_bytes = rx_pdu_bytes.load(std::memory_order_relaxed);
            double dt = std::chrono::duration<double>(now - prev_tput_time).count();
            if (dt > 0.01) {
                double instant_mbps = (double)(cur_bytes - prev_pdu_bytes) * 8.0
                                      / dt / 1e6;
                bool is_locked_now = opt.do_rx && rx_frontend->locked();
                if (!is_locked_now) {
                    throughput_ewma = 0.0f;
                } else {
                    throughput_ewma += TPUT_EWMA_ALPHA
                        * ((float)instant_mbps - throughput_ewma);
                }
                prev_pdu_bytes = cur_bytes;
                prev_tput_time = now;
            }
        }

        // Update spectrum tap KPIs for dashboard
        if (wf_tap) {
            bool is_locked = opt.do_rx && rx_frontend->locked();
            uint8_t tx_mc = acm_tx_enabled
                ? acm_tx_modcod.load(std::memory_order_relaxed)
                : (uint8_t)mc.modcod_idx;
            uint8_t rx_mc = rx_data_modcod.load(std::memory_order_relaxed);
            float fer = ewma_fer.load(std::memory_order_relaxed);
            wf_tap->set_kpis(pilot_snr, snr_std * snr_std,
                             rx_mc, tx_mc, fer, is_locked,
                             throughput_ewma);
        }

        // Degraded-RF detection (per-init X410 flakiness at MCR 245.76:
        // ~50% of radio inits bring one direction up with pilot coherence
        // 0.1-0.15 vs >0.9 healthy; only a full radio reboot fixes it).
        // Expose coherence in the log and print a single grep-able marker
        // once it has been persistently low while locked — harness scripts
        // key a radio-reset-retry off this.
        float coh_now = (opt.do_rx && rx_frontend->locked())
            ? rx_frontend->pilot_coherence() : -1.0f;
        if (opt.do_rx) {
            if (coh_now >= 0.0f && coh_now < 0.3f)
                low_coh_secs++;
            else
                low_coh_secs = 0;
            if (low_coh_secs == 20)
                fprintf(stderr, "log :warn: RF_DEGRADED pilot coherence %.2f"
                        " < 0.30 for %ds while locked — radio reset"
                        " recommended\n", coh_now, low_coh_secs);
        }

        // Rung-3 (--rx-symbol-sync) estimator gear shift: once frame sync is
        // locked with healthy pilot coherence, drop the RU's timing
        // estimator to tracking gear (much rarer estimates → near-zero
        // phase-step jitter, still tracking real clock drift). Back to
        // acquisition gear immediately on unlock.
        if (opt.rx_symbol_sync && opt.do_rx) {
            static bool ru_slow_gear = false;
            const bool locked_now = rx_frontend->locked();
            if (!ru_slow_gear && locked_now && coh_now > 0.9f) {
                radio->rx_sync_gear(true);
                ru_slow_gear = true;
                fprintf(stderr, "log :info: rung-3 RU estimator -> tracking "
                        "gear (locked, coh %.2f)\n", coh_now);
            } else if (ru_slow_gear && !locked_now) {
                radio->rx_sync_gear(false);
                ru_slow_gear = false;
                fprintf(stderr, "log :info: rung-3 RU estimator -> "
                        "acquisition gear (unlock)\n");
            }
        }

        fprintf(stderr,
                "log :info: %.0fs Lock=%s; pilotSNR=%.2f(std=%.2f); acmSNR=%.2f; dataSNR=%.2f; "
                "coh=%.2f; "
                "FECFRAMEs=%lu; FER=%.1e; eFER=%.1e; TxFrames=%lu; TxDummy=%lu; RxDummy=%lu; "
                "PDUs=%lu; UF=%lu; TX_SEQ=%lu; TX_LATE=%lu; UHD_OF=%lu; "
                "RingDrop=%lu; tap_drop=%lu; q_drop=%lu; gpu_drop=%lu",
                elapsed, (opt.do_rx && rx_frontend->locked()) ? "True" : "False", pilot_snr,
                snr_std, std::isfinite(acm_snr) ? acm_snr : -99.0f, data_snr, coh_now,
                (unsigned long)fc, fc > 0 ? (double)fe / fc : 0.0,
                (double)ewma_fer.load(std::memory_order_relaxed), (unsigned long)tx_frames.load(),
                (unsigned long)tx_dummy_frames.load(),
                (unsigned long)(opt.do_rx ? rx_frontend->dummy_count() : 0),
                (unsigned long)rx_pdus.load(), (unsigned long)tx_underflows.load(),
                (unsigned long)tx_sequence_errors.load(), (unsigned long)tx_time_errors.load(),
                (unsigned long)rx_uhd_overflows.load(), (unsigned long)rx_ring_drops.load(),
                (unsigned long)tx_pdu_queue_drops.load(), (unsigned long)rx_fec_queue_drops.load(),
                (unsigned long)(!opt.do_rx           ? 0
                                : opt.multi_modcod() ? fec_multi->total_gpu_drop_count()
                                                     : fec_single->gpu_drop_count()));

        // Latency diagnostics
        {
            double rx_ring_ms = (double)lat_rx_ring_samples.load(std::memory_order_relaxed) / opt.rate * 1000.0;
            double fec_ms = (double)lat_fec_decode_us.load(std::memory_order_relaxed) / 1000.0;
            double fe_ms = (double)lat_frontend_us.load(std::memory_order_relaxed) / 1000.0;
            double fe_agc_ms = (double)lat_fe_agc_us.load(std::memory_order_relaxed) / 1000.0;
            double fe_timing_ms = (double)lat_fe_timing_us.load(std::memory_order_relaxed) / 1000.0;
            double fe_sync_ms = (double)lat_fe_sync_us.load(std::memory_order_relaxed) / 1000.0;
            int fe_n_in = lat_fe_n_in.load(std::memory_order_relaxed);
            int fe_n_syms = lat_fe_n_syms.load(std::memory_order_relaxed);
            bool fe_fused = lat_fe_fused.load(std::memory_order_relaxed);
            double tr_memcpy_ms = (double)lat_tr_memcpy_us.load(std::memory_order_relaxed) / 1000.0;
            double tr_launch_ms = (double)lat_tr_launch_us.load(std::memory_order_relaxed) / 1000.0;
            double tr_lnch_est_ms = (double)ctx.stats.lat_tr_lnch_est_us.load(std::memory_order_relaxed) / 1000.0;
            double tr_lnch_interp_ms = (double)ctx.stats.lat_tr_lnch_interp_us.load(std::memory_order_relaxed) / 1000.0;
            double tr_sync_ms = (double)lat_tr_sync_us.load(std::memory_order_relaxed) / 1000.0;
            double tr_readback_ms = (double)lat_tr_readback_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_gen_ms = (double)lat_tx_gen_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_gse_ms = (double)lat_tx_gse_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_bb_bch_ms = (double)lat_tx_bb_bch_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_ldpc_ms = (double)lat_tx_ldpc_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_gpu_ms = (double)lat_tx_gpu_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_ring_ms = (double)lat_tx_ring_wait_us.load(std::memory_order_relaxed) / 1000.0;
            int tx_batch = lat_tx_batch_size.load(std::memory_order_relaxed);
            double batch_wait_ms = (double)lat_batch_wait_us.load(std::memory_order_relaxed) / 1000.0;
            double tx_loop_ms = (double)lat_tx_loop_us.load(std::memory_order_relaxed) / 1000.0;
            double rx_loop_ms = (double)lat_rx_loop_us.load(std::memory_order_relaxed) / 1000.0;
            double gpu_drain_ms = (double)lat_gpu_drain_us.load(std::memory_order_relaxed) / 1000.0;
            double gpu_phase_ms = (double)lat_gpu_phase_us.load(std::memory_order_relaxed) / 1000.0;
            double gpu_submit_ms = (double)lat_gpu_submit_us.load(std::memory_order_relaxed) / 1000.0;
            double fec_loop_ms = (double)lat_fec_loop_us.load(std::memory_order_relaxed) / 1000.0;
            double gpu_timing_ms = (double)lat_gpu_timing_us.load(std::memory_order_relaxed) / 1000.0;
            double gpu_interp_ms = (double)lat_gpu_interp_us.load(std::memory_order_relaxed) / 1000.0;
            fprintf(
                stderr,
                "; LAT tx_loop=%.1f tx_gen=%.1f(gse=%.2f bb_bch=%.2f ldpc=%.2f gpu=%.2f ring=%.1f "
                "b=%d)"
                " rx_ring=%.1f rx_loop=%.1f fe=%.1f(agc=%.2f tim=%.2f[cp=%.2f lnch=%.2f{est=%.2f "
                "interp=%.2f} sync=%.2f rb=%.2f gpu_est=%.2f gpu_interp=%.2f] sync=%.2f n=%d/%d%s)"
                " batch_wait=%.1f fec=%.1f fq=%d"
                " fec_loop=%.2f gpu_drain=%.2f gpu_phase=%.2f gpu_sub=%.2f gpu_ifl=%d fd=%lu "
                "prec=%.0f",
                tx_loop_ms, tx_gen_ms, tx_gse_ms, tx_bb_bch_ms, tx_ldpc_ms, tx_gpu_ms, tx_ring_ms,
                tx_batch, rx_ring_ms, rx_loop_ms, fe_ms, fe_agc_ms, fe_timing_ms, tr_memcpy_ms,
                tr_launch_ms, tr_lnch_est_ms, tr_lnch_interp_ms, tr_sync_ms, tr_readback_ms,
                gpu_timing_ms, gpu_interp_ms, fe_sync_ms, fe_n_in, fe_n_syms, fe_fused ? "F" : "",
                batch_wait_ms, fec_ms, frame_queue.available(), fec_loop_ms, gpu_drain_ms,
                gpu_phase_ms, gpu_submit_ms, gpu_inflight_count.load(std::memory_order_relaxed),
                (unsigned long)gpu_force_drain_count.load(std::memory_order_relaxed),
                (double)((opt.multi_modcod() || !opt.do_rx) ? 0.0f : fec_single->precision()));

            // TX send-path stall diagnostics: per-period max time inside
            // send() / max inter-send gap, plus cumulative outlier counts
            if (opt.do_tx) {
                double snd_max_ms = (double)ctx.stats.txs_send_max_us.exchange(0) / 1000.0;
                double gap_max_ms = (double)ctx.stats.txs_gap_max_us.exchange(0) / 1000.0;
                fprintf(stderr,
                        " txsnd[max=%.2f gap=%.2f d200=%lu g200=%lu d1k=%lu g1k=%lu]",
                        snd_max_ms, gap_max_ms,
                        (unsigned long)ctx.stats.txs_dur_over200.load(std::memory_order_relaxed),
                        (unsigned long)ctx.stats.txs_gap_over200.load(std::memory_order_relaxed),
                        (unsigned long)ctx.stats.txs_dur_over1ms.load(std::memory_order_relaxed),
                        (unsigned long)ctx.stats.txs_gap_over1ms.load(std::memory_order_relaxed));
            }

            // Accumulate benchmark stats (skip first 5s for startup transients)
            if (!bench_started) {
                double up = std::chrono::duration<double>(now - start_time).count();
                if (up > 5.0 &&
                    (!opt.do_rx || rx_frontend->locked())) {
                    bench_started = true;
                    bench_start_time = now;
                    bench_prev_tx_frames = tx_frames.load();
                    bench_prev_rx_frames = rx_fec_frames.load();
                }
            }
            if (bench_started) {
                bench.tx_loop.add(tx_loop_ms);
                bench.tx_gen.add(tx_gen_ms);
                bench.tx_gse.add(tx_gse_ms);
                bench.tx_bb_bch.add(tx_bb_bch_ms);
                bench.tx_ldpc.add(tx_ldpc_ms);
                bench.tx_gpu.add(tx_gpu_ms);
                bench.tx_ring_wait.add(tx_ring_ms);
                bench.tx_batch_wait.add(batch_wait_ms);
                bench.tx_batch_size.add((double)tx_batch);

                bench.rx_loop.add(rx_loop_ms);
                bench.fe_total.add(fe_ms);
                bench.fe_agc.add(fe_agc_ms);
                bench.fe_timing.add(fe_timing_ms);
                bench.fe_sync.add(fe_sync_ms);
                bench.tr_stage.add(tr_memcpy_ms);
                bench.tr_launch.add(tr_launch_ms);
                bench.tr_sync.add(tr_sync_ms);
                bench.tr_readback.add(tr_readback_ms);
                bench.gpu_timing.add(gpu_timing_ms);
                bench.gpu_interp.add(gpu_interp_ms);

                bench.fec_submit.add(fec_ms);
                bench.fec_loop.add(fec_loop_ms);
                bench.gpu_drain.add(gpu_drain_ms);
                bench.gpu_phase.add(gpu_phase_ms);
                bench.gpu_submit_xfec.add(gpu_submit_ms);

                bench.rx_ring.add(rx_ring_ms);

                // Accumulate sync point counters (reset each period)
                bench.total_tx_syncs += sync_count_tx.exchange(0);
                bench.total_rx_syncs += sync_count_rx.exchange(0);

                const uint64_t current_tx_frames = tx_frames.load();
                const uint64_t current_rx_frames = rx_fec_frames.load();
                bench.total_tx_frames += reset_aware_counter_delta(
                    current_tx_frames, bench_prev_tx_frames);
                bench.total_rx_frames += reset_aware_counter_delta(
                    current_rx_frames, bench_prev_rx_frames);
                bench_prev_tx_frames = current_tx_frames;
                bench_prev_rx_frames = current_rx_frames;
            }
        }

        // ACM status
        if (opt.multi_modcod() && opt.do_rx) {
            fprintf(stderr,
                    "; ACM sub=%lu skip_len=%lu dummy=%lu fb_tx=%lu fb_rx=%lu "
                    "fb_recover=%lu mc_hist=[",
                    (unsigned long)acm_frames_submitted.load(),
                    (unsigned long)acm_frames_skipped_len.load(),
                    (unsigned long)acm_frames_dummy.load(),
                    (unsigned long)acm_feedback_tx.load(),
                    (unsigned long)acm_feedback_rx.load(),
                    (unsigned long)acm_feedback_recovery_bursts.load());
            bool first = true;
            for (int i = 0; i < 256; i++) {
                uint64_t cnt = acm_mc_hist[i].load();
                if (cnt > 0) {
                    fprintf(stderr, "%s%d:%lu", first ? "" : " ", i, (unsigned long)cnt);
                    first = false;
                }
            }
            fprintf(stderr, "] pls_guard={corrected:%lu deferred:%lu transitions:%lu outage:%lu}",
                    (unsigned long)acm_pls_guard_corrected.load(),
                    (unsigned long)acm_pls_guard_deferred.load(),
                    (unsigned long)acm_pls_guard_transitions.load(),
                    (unsigned long)acm_pls_guard_outage_frames.load());
        }

        // TX MODCOD (ACM)
        if (acm_tx_enabled) {
            uint8_t tx_mc = acm_tx_modcod.load(std::memory_order_relaxed);
            const modcod_def_t* def = find_modcod(tx_mc);
            uint8_t data_mc = acm_tx_data_modcod.load(
                std::memory_order_relaxed);
            const modcod_def_t* data_def = find_modcod(data_mc);
            fprintf(stderr, "; TxMC=%s(%d) dataMC=%s(%d) olla=%.3fdB",
                    def ? def->name : "?", tx_mc,
                    data_def ? data_def->name : "?", data_mc,
                    acm_ctrl ? acm_ctrl->olla_offset_db() : 0.0f);
        }

        fprintf(stderr, "\n");
    }

    // --- Shutdown ---
    fprintf(stderr, "dvbs2-uhd-trx: shutting down...\n");
    g_stop.store(true);

    if (demo_fd >= 0) close(demo_fd);
    if (acm_fb_rx_fd >= 0) close(acm_fb_rx_fd);
    if (acm_fb_tx_fd >= 0) close(acm_fb_tx_fd);

    t_tap_reader.join();
    t_tx_gen.join();
    t_tx_send.join();
    t_rx_uhd.join();
    t_rx_frontend.join();
    t_rx_fec.join();

    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    fprintf(stderr,
            "dvbs2-uhd-trx: finished. %.1f sec, TX %lu frames (%lu dummy), "
            "TX errors: %lu underflow, %lu sequence, %lu late; "
            "RX %lu FECFRAMEs %lu errors, %lu PDUs\n",
            total_sec,
            (unsigned long)tx_frames.load(),
            (unsigned long)tx_dummy_frames.load(),
            (unsigned long)tx_underflows.load(),
            (unsigned long)tx_sequence_errors.load(),
            (unsigned long)tx_time_errors.load(),
            (unsigned long)rx_fec_frames.load(),
            (unsigned long)rx_fec_errors.load(),
            (unsigned long)rx_pdus.load());

    // Compute bench wall time and actual frame counts for resource metrics
    if (bench_started) {
        bench.bench_wall_sec = std::chrono::duration<double>(end_time - bench_start_time).count();
        const uint64_t current_tx_frames = tx_frames.load();
        const uint64_t current_rx_frames = rx_fec_frames.load();
        bench.total_tx_frames += reset_aware_counter_delta(
            current_tx_frames, bench_prev_tx_frames);
        bench.total_rx_frames += reset_aware_counter_delta(
            current_rx_frames, bench_prev_rx_frames);
    }

    // Print benchmark summary if we collected data
    if (bench.tx_loop.count > 0 || bench.fe_total.count > 0) {
        bench.print_summary(stderr, opt.rate);
    }

    return 0;
}
