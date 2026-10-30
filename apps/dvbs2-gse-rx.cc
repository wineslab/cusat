/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GSE receiver: DVB-S2 RX → GSE → TAP/pipe pipeline.
 *
 * Usage:
 *   dvbs2-gse-rx --modcod qpsk3/5 --tap cusatrx0 < input.iq
 *   dvbs2-gse-rx --modcod qpsk3/5 --pipe < input.iq > output.framed
 */

#include "acm_config.h"
#include "rx_fec_chain.h"
#include "gse.h"
#include "iq_source_file.h"
#include "receiver_frontend.h"
#include "tap_interface.h"
#include "trx/frontend_trace_writer.h"
#include <cusat_dsp/modcod_types.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using namespace cusat_dsp;

/* ---------- Ethernet L2/L3 helpers ---------- */
static constexpr int ETH_HLEN = 14;
static constexpr int TAP_MTU_BYTES = 16384;
static const uint8_t PEER_MAC[6] = {0x02, 0x00, 0x5a, 0x54, 0x00, 0x01};

static bool write_tap_pdu(tap_interface& tap, std::vector<uint8_t>& eth_frame,
                          const uint8_t* pdu, int pdu_len, uint16_t proto)
{
    if (pdu_len <= 0)
        return true;
    if ((size_t)(ETH_HLEN + pdu_len) > eth_frame.size()) {
        fprintf(stderr, "dvbs2-gse-rx: dropping oversized TAP PDU (%d bytes, cap=%zu)\n",
                pdu_len, eth_frame.size() - ETH_HLEN);
        return false;
    }
    // GSE transports L3 packets and does not preserve an Ethernet
    // destination. Target the TAP's current MAC: Linux deliberately drops
    // TCP packets delivered as L2 broadcast even when the L3 address is local.
    memcpy(eth_frame.data(), tap.mac(), 6);        // dst = local TAP
    memcpy(eth_frame.data() + 6, PEER_MAC, 6);    // src = peer
    eth_frame[12] = (proto >> 8) & 0xFF;
    eth_frame[13] = proto & 0xFF;
    memcpy(eth_frame.data() + ETH_HLEN, pdu, pdu_len);
    tap.write_frame(eth_frame.data(), ETH_HLEN + pdu_len);
    return true;
}

/* ---------- Thread-safe XFECFRAME queue ---------- */

struct queued_frame_t {
    std::vector<complexf> payload;
    pl_layout pls;
    float pilot_snr_db = 0.0f;
};

class FrameQueue
{
    std::queue<queued_frame_t> d_queue;
    std::mutex d_mtx;
    std::condition_variable d_cv;
    bool d_done = false;

public:
    void push(const complexf* data, int len, const pl_layout& pls, float pilot_snr_db)
    {
        {
            std::lock_guard<std::mutex> lk(d_mtx);
            queued_frame_t f;
            f.payload.assign(data, data + len);
            f.pls = pls;
            f.pilot_snr_db = pilot_snr_db;
            d_queue.push(std::move(f));
        }
        d_cv.notify_one();
    }

    bool pop(queued_frame_t& out)
    {
        std::unique_lock<std::mutex> lk(d_mtx);
        d_cv.wait(lk, [&] { return !d_queue.empty() || d_done; });
        if (d_queue.empty()) return false;
        out = std::move(d_queue.front());
        d_queue.pop();
        return true;
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lk(d_mtx);
        return d_queue.size();
    }

    void finish()
    {
        { std::lock_guard<std::mutex> lk(d_mtx); d_done = true; }
        d_cv.notify_all();
    }
};

/* ---------- Modcod parsing ---------- */

static std::string to_lower(const std::string& s)
{
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

/* ---------- PDU output helpers ---------- */

/** Write one PDU to pipe (2-byte big-endian length prefix + data). */
static void write_pdu_pipe(int fd, const uint8_t* pdu, int len)
{
    uint8_t len_buf[2] = { (uint8_t)((len >> 8) & 0xFF), (uint8_t)(len & 0xFF) };
    ssize_t w = write(fd, len_buf, 2);
    w = write(fd, pdu, len);
    (void)w;
}

/* ---------- Usage ---------- */

static void usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s --modcod <modcod> [options]\n"
            "\n"
            "GSE receiver: reads DVB-S2 IQ, outputs PDUs to TAP or pipe.\n"
            "\n"
            "Options:\n"
            "  --modcod <str>          Modulation and coding (e.g. qpsk3/5)\n"
            "  --tap <name>            TAP device name (Linux only)\n"
            "  --pipe                  Write framed PDUs to stdout (2B len + data)\n"
            "  --sps <float>          Samples per symbol, fractional OK (default: 4)\n"
            "  --rolloff <float>       RRC rolloff (default: 0.2)\n"
            "  --gold-code <int>       PL scrambling gold code (default: 0)\n"
            "  --in-file <path>        Input IQ file (default: stdin)\n"
            "  --out-file <path>       Output file for pipe mode (default: stdout)\n"
            "  --ldpc-trials <int>     Max LDPC iterations (default: 25)\n"
            "  --frontend-trace-csv <path> Write per-call frontend timing trace\n"
            "  --trace-sample-rate <Hz> Sample rate for trace period/deadline fields\n"
            "  --timing-est-period <int> Timing estimation period (default: 4)\n"
            "  --energy-k <int>        Polyphase energy search arms (default: 32)\n"
            "  --energy-search-cap <int> Max symbols for energy search (default: 2048, 0 = full)\n"
            "  --rx-batch-plframes <int> Batch cap in PLFRAMEs (default: frontend default)\n"
            "  --rx-min-batch-us <int> Locked-path min batch budget in us (0 = off)\n"
            "  --rx-diag               Write per-frame synchronization diagnostics\n"
            "  --log                   Print periodic status to stderr\n"
            "  --debug <int>           Debug level (default: 0)\n"
            "  -h, --help              Show this help\n",
            prog);
}

/* ---------- Main ---------- */

int main(int argc, char* argv[])
{
    std::string modcod_str;
    std::string tap_name;
    bool pipe_mode = false;
    constexpr bool pilots = true; // pilots are mandatory
    bool short_frames = false;
    float sps = 4.0f;
    float rolloff = 0.2f;
    int gold_code = 0;
    std::string in_file;
    std::string out_file;
    int ldpc_trials = 25;
    std::string frontend_trace_csv;
    double trace_sample_rate = 0.0;
    int timing_est_period = 4;
    int energy_k = 32;
    int energy_search_cap = 2048;
    int rx_batch_plframes = 0;
    int rx_min_batch_us = 0;
    bool log_enabled = false;
    bool rx_diag = false;
    int debug_level = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--modcod" && i + 1 < argc) modcod_str = to_lower(argv[++i]);
        else if (arg == "--tap" && i + 1 < argc) tap_name = argv[++i];
        else if (arg == "--pipe") pipe_mode = true;
        else if (arg == "--short") short_frames = true;
        else if (arg == "--sps" && i + 1 < argc) sps = atof(argv[++i]);
        else if (arg == "--rolloff" && i + 1 < argc) rolloff = atof(argv[++i]);
        else if (arg == "--gold-code" && i + 1 < argc) gold_code = atoi(argv[++i]);
        else if (arg == "--in-file" && i + 1 < argc) in_file = argv[++i];
        else if (arg == "--out-file" && i + 1 < argc) out_file = argv[++i];
        else if (arg == "--ldpc-trials" && i + 1 < argc) ldpc_trials = atoi(argv[++i]);
        else if (arg == "--frontend-trace-csv" && i + 1 < argc) frontend_trace_csv = argv[++i];
        else if (arg == "--trace-sample-rate" && i + 1 < argc) trace_sample_rate = atof(argv[++i]);
        else if (arg == "--timing-est-period" && i + 1 < argc) timing_est_period = atoi(argv[++i]);
        else if (arg == "--energy-k" && i + 1 < argc) energy_k = atoi(argv[++i]);
        else if (arg == "--energy-search-cap" && i + 1 < argc) energy_search_cap = atoi(argv[++i]);
        else if (arg == "--rx-batch-plframes" && i + 1 < argc) rx_batch_plframes = atoi(argv[++i]);
        else if (arg == "--rx-min-batch-us" && i + 1 < argc) rx_min_batch_us = atoi(argv[++i]);
        else if (arg == "--rx-diag") rx_diag = true;
        else if (arg == "--log") log_enabled = true;
        else if (arg == "--debug" && i + 1 < argc) debug_level = atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); return 1; }
    }

    if (modcod_str.empty()) {
        fprintf(stderr, "Error: --modcod is required\n");
        return 1;
    }
    if (tap_name.empty() && !pipe_mode) {
        fprintf(stderr, "Error: --tap or --pipe is required\n");
        return 1;
    }

    const modcod_def_t* mc_p = find_modcod_by_name(modcod_str.c_str());
    if (!mc_p) {
        fprintf(stderr, "Error: unknown modcod '%s'\n", modcod_str.c_str());
        return 1;
    }
    const modcod_def_t& mc = *mc_p;
    // Fractional sps mirrors the trx hw-rate RX (the frontend's polyphase
    // matched filter and timing NCO run at any stride >= 1 + rolloff + 0.05).
    // Lets OTA captures taken at fractional hw_sps decode offline.
    if (sps < 1.0f + rolloff + 0.05f || sps > 8.0f) {
        fprintf(stderr, "Error: --sps must be in [%.2f, 8] (got %g)\n",
                1.0f + rolloff + 0.05f, (double)sps);
        return 1;
    }
    fecframe_size_t framesize = short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL;

    // --- Open I/O ---
    std::unique_ptr<tap_interface> tap;
    if (!tap_name.empty())
        tap = std::make_unique<tap_interface>(tap_name);

    std::unique_ptr<iq_source> source;
    if (in_file.empty())
        source = std::make_unique<iq_source_file>(STDIN_FILENO);
    else
        source = std::make_unique<iq_source_file>(in_file);

    int out_fd = STDOUT_FILENO;
    if (!out_file.empty()) {
        out_fd = open(out_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out_fd < 0) { perror("Cannot open output file"); return 1; }
    }

    // --- PLS setup ---
    // The PLS code helper handles DVB-S2X MODCODs (64/128APSK -> PLS >= 128);
    // a hand-rolled modcod<<2 encoding mis-maps them to the DVB-S2 range.
    if (!modcod_supports_framesize(mc, framesize)) {
        fprintf(stderr, "Error: MODCOD %s is not available with %s frames\n",
                modcod_str.c_str(), short_frames ? "short" : "normal");
        return 1;
    }
    uint8_t target_plsc = pls_code_for(mc.modcod_idx, framesize, pilots);
    pl_layout frame_format = pl_layout_from_code(target_plsc);
    if (frame_format.bits_per_symbol == 0 || !frame_format.deliverable()) {
        fprintf(stderr, "Error: MODCOD %s -> PLS %u has no payload "
                "constellation; PLS encoding is wrong\n",
                modcod_str.c_str(), target_plsc);
        return 1;
    }
    int xfecframe_len = frame_format.data_symbols();

    fprintf(stderr,
            "dvbs2-gse-rx: modcod=%s sps=%g rolloff=%.2f "
            "xfecframe_len=%d output=%s\n",
            modcod_str.c_str(), (double)sps, rolloff,
            xfecframe_len, tap_name.empty() ? "pipe" : tap_name.c_str());

    // --- Create DSP pipeline ---
    // Match dvbs2-uhd-trx: below ~2.5 sps the truncated delay-5 RRC
    // prototype's in-band ripple swamps the timing/pilot metrics.
    const int rrc_delay = (sps < 2.5) ? 16 : 5;
    const int n_phases = 128;
    const float agc_rate = 1e-5f;
    const float agc_ref = 1.0f;
    const float agc_gain = 1.0f;
    const float agc_max_gain = 65536.0f;

    // CCM, one delivered PLS code.
    receiver_frontend_config fe_cfg;
    fe_cfg.sps = sps;
    fe_cfg.rolloff = rolloff;
    fe_cfg.rrc_delay = rrc_delay;
    fe_cfg.n_phases = n_phases;
    fe_cfg.prefiltered = false;
    fe_cfg.symbols_in = false;
    fe_cfg.agc_rate = agc_rate;
    fe_cfg.agc_ref = agc_ref;
    fe_cfg.agc_initial_gain = agc_gain;
    fe_cfg.agc_max_gain = agc_max_gain;
    fe_cfg.debug_level = debug_level;
    fe_cfg.pl.mode = pl_service_mode::ccm;
    fe_cfg.pl.deliver_codes = { target_plsc };
    fe_cfg.pl.scrambling_code = gold_code;
    fe_cfg.pl.debug_level = debug_level;
    receiver_frontend frontend(fe_cfg);
    // Single known MODCOD: whitelist its PLFRAME length so garbage PLSC decodes
    // at wrong offsets are rejected (accept-logic), and hint the same length so
    // the pre-lock acquisition batch is sized sub-PLFRAME (critical for small
    // frames like short 32APSK).
    frontend.set_valid_plframe_lens({ (uint32_t)frame_format.plframe_symbols() });
    frontend.set_acq_plframe_hint((uint32_t)frame_format.plframe_symbols());
    frontend.set_timing_estimation_period(timing_est_period);
    frontend.set_energy_k(energy_k);
    frontend.set_energy_search_cap(energy_search_cap);
    frontend.set_diag_rx(rx_diag);
    if (rx_batch_plframes > 0)
        frontend.set_batch_plframes(rx_batch_plframes);
    if (rx_min_batch_us > 0 && trace_sample_rate > 0.0)
        frontend.set_min_batch_samples((int)(trace_sample_rate * rx_min_batch_us / 1e6));

    rx_fec_chain fec(framesize, mc.code_rate, mc.constellation, ldpc_trials);
    fec.set_block_when_full(true); // offline/pipe input: never drop frames

    // GSE decapsulator
    gse_decapsulator decap;

    // --- Shared state ---
    FrameQueue frame_queue;
    std::atomic<bool> consumer_done{false};
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<uint64_t> total_pdus{0};
    std::atomic<uint64_t> fec_frames_seen{0};
    std::atomic<uint64_t> fec_errors_seen{0};

    frontend_trace_writer fe_trace;
    if (!fe_trace.open(frontend_trace_csv, "dvbs2-gse-rx"))
        return 1;

    // --- Producer thread: frontend ---
    auto producer_fn = [&]() {
        const int min_chunk = (int)(frame_format.plframe_symbols() * sps) * 2 + 10000;
        const int chunk_size = std::max(50000, min_chunk);
        const int history = frontend.history();
        std::vector<complexf> in_buf(chunk_size + history * 2);
        // output space for every frame one call can complete: at most the
        // call's input in symbols (plus a pending frame), never less than
        // one maximum XFECFRAME (receiver_frontend contract)
        const int max_call_syms = (int)((chunk_size + history * 2) / sps) + 2;
        std::vector<complexf> sym_buf(std::max({ xfecframe_len * 4, max_call_syms + 2 * xfecframe_len,
                                                 pl::max_payload_symbols() }));

        int n_buffered = 0;
        while (true) {
            int max_read = (int)in_buf.size() - n_buffered;
            if (max_read <= 0) max_read = chunk_size;
            int n_read = source->read(in_buf.data() + n_buffered, max_read);
            if (n_read <= 0) break;
            int n_in = n_buffered + n_read;
            int n_consumed = 0;
            std::vector<frontend_frame_t> frames;

            auto fe_start = std::chrono::steady_clock::now();
            frontend.process(in_buf.data(), n_in, sym_buf.data(), (int)sym_buf.size(),
                             n_consumed, frames);
            auto fe_end = std::chrono::steady_clock::now();
            uint64_t fe_total_us =
                (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                    fe_end - fe_start).count();
            if (fe_trace.active()) {
                uint64_t ff = fec_frames_seen.load(std::memory_order_relaxed);
                uint64_t ferr = fec_errors_seen.load(std::memory_order_relaxed);
                double fer = ff ? (double)ferr / (double)ff : 0.0;
                fe_trace.write(frontend, start_time, fe_end, fe_total_us,
                               n_consumed, frames.size(), trace_sample_rate,
                               fer, ff, ferr, frame_queue.size(), 0, 0);
            }

            for (const auto& frame : frames) {
                if (frame.payload_len == xfecframe_len) {
                    frame_queue.push(frame.payload, frame.payload_len, frame.pls, frame.pilot_snr_db);
                } else {
                    static long mismatch_cnt = 0;
                    if (++mismatch_cnt <= 5 || (mismatch_cnt % 100) == 0)
                        fprintf(stderr, "dvbs2-gse-rx: frame payload_len=%d"
                                " != xfec=%d (#%ld) — dropped\n",
                                frame.payload_len, xfecframe_len, mismatch_cnt);
                }
            }

            n_buffered = n_in - n_consumed;
            if (n_consumed > 0 && n_buffered > 0) {
                memmove(in_buf.data(), in_buf.data() + n_consumed,
                        n_buffered * sizeof(complexf));
            }
        }
        fe_trace.flush();
        frame_queue.finish();
    };

    // --- Consumer thread: FEC + GSE decapsulation ---
    auto consumer_fn = [&]() {
        // Buffer must hold an entire LDPC batch of GSE data fields
        std::vector<uint8_t> gse_buf(fec.max_output_bytes());
        queued_frame_t qf;

        // PDU output callback — reconstruct Ethernet frame from L3 payload
        std::vector<uint8_t> eth_frame(TAP_MTU_BYTES + ETH_HLEN);
        auto pdu_cb = [&](const uint8_t* pdu, int pdu_len, uint16_t proto,
                          const uint8_t* label, int label_len) {
            (void)label;
            (void)label_len;
            if (tap && pdu_len > 0) {
                write_tap_pdu(*tap, eth_frame, pdu, pdu_len, proto);
            } else {
                write_pdu_pipe(out_fd, pdu, pdu_len);
            }
            total_pdus.fetch_add(1);
        };

        while (frame_queue.pop(qf)) {
            // per-frame pilot SNR for LLR scaling
            fec.update_snr(qf.pilot_snr_db);
            bool batch_ready = fec.submit_xfecframe(qf.payload.data(), (int)qf.payload.size());
            if (batch_ready) {
                int total = fec.get_gse_output(gse_buf.data(), (int)gse_buf.size());
                fec_frames_seen.store(fec.frame_count(), std::memory_order_relaxed);
                fec_errors_seen.store(fec.frame_error_count(), std::memory_order_relaxed);
                if (total > 0) {
                    // Decapsulate each frame's data field separately
                    const auto& sizes = fec.gse_dfl_sizes();
                    int offset = 0;
                    for (int sz : sizes) {
                        decap.decapsulate(gse_buf.data() + offset, sz, pdu_cb);
                        offset += sz;
                    }
                }
            }
        }

        // Flush partial FEC batch
        fec.flush();
        int total = fec.get_gse_output(gse_buf.data(), (int)gse_buf.size());
        fec_frames_seen.store(fec.frame_count(), std::memory_order_relaxed);
        fec_errors_seen.store(fec.frame_error_count(), std::memory_order_relaxed);
        if (total > 0) {
            const auto& sizes = fec.gse_dfl_sizes();
            int offset = 0;
            auto pdu_cb2 = [&](const uint8_t* pdu, int pdu_len, uint16_t proto,
                               const uint8_t* label, int label_len) {
                (void)label; (void)label_len;
                if (tap && pdu_len > 0) {
                    write_tap_pdu(*tap, eth_frame, pdu, pdu_len, proto);
                } else {
                    write_pdu_pipe(out_fd, pdu, pdu_len);
                }
                total_pdus.fetch_add(1);
            };
            for (int sz : sizes) {
                decap.decapsulate(gse_buf.data() + offset, sz, pdu_cb2);
                offset += sz;
            }
        }
        consumer_done.store(true);
    };

    // --- Launch threads ---
    std::thread frontend_thread(producer_fn);
    std::thread fec_thread(consumer_fn);

    // --- Main thread: periodic logging ---
    if (log_enabled) {
        bool was_locked = false;
        auto last_log_time = start_time;
        while (!consumer_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            bool locked = frontend.locked();
            if (!was_locked && locked) {
                fprintf(stderr, "log :info: Lock=True\n");
                was_locked = true;
            }

            auto now = std::chrono::steady_clock::now();
            double elapsed_ms =
                std::chrono::duration<double, std::milli>(now - last_log_time).count();

            if (elapsed_ms >= 1000.0) {
                last_log_time = now;
                fprintf(stderr,
                        "log :info: Lock=%s; pilotSNR=%.2f; dataSNR=%.2f; "
                        "FECFRAMEs=%lu; FER=%.1e; PDUs=%lu; "
                        "GSE_errors=%lu; agc=%.1fdB\n",
                        locked ? "True" : "False",
                        frontend.pilot_snr_db(),
                        fec.data_snr_db(),
                        (unsigned long)fec.frame_count(),
                        fec.frame_count() > 0
                            ? (double)fec.frame_error_count() / fec.frame_count()
                            : 0.0,
                        (unsigned long)total_pdus.load(),
                        (unsigned long)decap.error_count(),
                        20.0f * std::log10(frontend.agc_gain() + 1e-20f));
            }
        }
    }

    frontend_thread.join();
    fec_thread.join();

    if (out_fd != STDOUT_FILENO) close(out_fd);
    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    fprintf(stderr,
            "dvbs2-gse-rx: finished. %.1f sec, %lu PDUs, "
            "%lu FECFRAMEs, %lu FEC errors, %lu GSE errors\n",
            total_sec,
            (unsigned long)total_pdus.load(),
            (unsigned long)fec.frame_count(),
            (unsigned long)fec.frame_error_count(),
            (unsigned long)decap.error_count());

    return 0;
}
