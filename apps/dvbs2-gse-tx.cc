/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GSE transmitter: TAP/pipe → GSE → DVB-S2 TX pipeline.
 *
 * Usage:
 *   dvbs2-gse-tx --modcod qpsk3/5 --tap dvbs2tx0 > output.iq
 *   dvbs2-gse-tx --modcod qpsk3/5 --pipe < input.framed > output.iq
 */

#include "gse.h"
#include "pl_framer.h"
#include "tap_interface.h"
#include "transmitter_frontend.h"
#include "acm_config.h"
#include "tx_fec_chain.h"
#include <cusat_dsp/modcod_types.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace cusat_dsp;

/* ---------- Ethernet L2/L3 helpers ---------- */
static constexpr int ETH_HLEN = 14;
static constexpr uint16_t ETH_P_IP = 0x0800;

/* ---------- Modcod parsing ---------- */

static std::string to_lower(const std::string& s)
{
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

/* ---------- PDU I/O helpers ---------- */

/** Read exactly n bytes (blocking). Returns bytes read, or -1 on error/EOF. */
static int read_exact(int fd, uint8_t* buf, int n)
{
    int total = 0;
    while (total < n) {
        ssize_t r = read(fd, buf + total, n - total);
        if (r <= 0) return (total > 0) ? total : -1;
        total += r;
    }
    return total;
}

/** Read one PDU from pipe (2-byte big-endian length prefix + data). */
static int read_pdu_pipe(int fd, uint8_t* buf, int max_len)
{
    uint8_t len_buf[2];
    if (read_exact(fd, len_buf, 2) != 2) return -1;
    int len = (len_buf[0] << 8) | len_buf[1];
    if (len <= 0 || len > max_len) return -1;
    return (read_exact(fd, buf, len) == len) ? len : -1;
}

/** 0 = timeout, -1 = EOF/error, positive = PDU length. */
static int read_pdu_pipe_timeout(int fd, uint8_t* buf, int max_len, int timeout_ms)
{
    struct pollfd pfd = {fd, POLLIN, 0};
    int ready = poll(&pfd, 1, timeout_ms);
    if (ready == 0) return 0;
    if (ready < 0 || (pfd.revents & (POLLERR | POLLNVAL))) return -1;
    return read_pdu_pipe(fd, buf, max_len);
}

/* ---------- Usage ---------- */

static void usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s --modcod <modcod> [options]\n"
            "\n"
            "GSE transmitter: reads PDUs from TAP or pipe, outputs DVB-S2 IQ.\n"
            "\n"
            "Options:\n"
            "  --modcod <str>      Modulation and coding (e.g. qpsk3/5)\n"
            "  --tap <name>        TAP device name (Linux only)\n"
            "  --pipe              Read framed PDUs from stdin (2B len + data)\n"
            "  --sps <int>        Samples per symbol, 2 or 4 (default: 4)\n"
            "  --rolloff <float>   RRC rolloff (default: 0.2)\n"
            "  --gold-code <int>   PL scrambling gold code (default: 0)\n"
            "  --out-file <path>   Output IQ file (default: stdout)\n"
            "  --n-frames <int>    Max frames to transmit (default: unlimited)\n"
            "  --awgn-snr-db <dB>  Add fixed AWGN at this sample-SNR\n"
            "  --noise-seed <int>  AWGN RNG seed (default: 1234)\n"
            "  --sym-rate          Output at symbol rate (no RRC)\n"
            "  --pace-sample-rate <Hz> Pace IQ output in real time\n"
            "  --frame-trace-csv <path> Log application/GSE load per PLFRAME\n"
            "  --gse-flush-ms <ms> Flush a partial BBFRAME after timeout (default: 2)\n"
            "  --log               Print periodic status to stderr\n"
            "  -h, --help          Show this help\n",
            prog);
}

static void apply_awgn(complexf* samples, unsigned int n_samples,
                       float snr_db, std::mt19937& rng)
{
    if (n_samples == 0)
        return;

    double sig_power = 0.0;
    for (unsigned int i = 0; i < n_samples; i++)
        sig_power += std::norm(samples[i]);
    sig_power /= (double)n_samples;
    if (sig_power <= 1e-30)
        return;

    double snr_lin = std::pow(10.0, (double)snr_db / 10.0);
    float sigma = (float)std::sqrt(sig_power / (2.0 * snr_lin));
    std::normal_distribution<float> noise(0.0f, sigma);
    for (unsigned int i = 0; i < n_samples; i++)
        samples[i] += complexf(noise(rng), noise(rng));
}

/* ---------- Main ---------- */

int main(int argc, char* argv[])
{
    std::string modcod_str;
    std::string tap_name;
    bool pipe_mode = false;
    constexpr bool pilots = true; // pilots are mandatory
    bool short_frames = false;
    int sps = 4;
    float rolloff = 0.2f;
    int gold_code = 0;
    std::string out_file;
    int n_frames = 0;
    bool awgn_enabled = false;
    float awgn_snr_db = 0.0f;
    unsigned int noise_seed = 1234;
    bool sym_rate = false;
    bool log_enabled = false;
    double pace_sample_rate = 0.0;
    std::string frame_trace_csv;
    int gse_flush_ms = 2;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--modcod" && i + 1 < argc) modcod_str = to_lower(argv[++i]);
        else if (arg == "--tap" && i + 1 < argc) tap_name = argv[++i];
        else if (arg == "--pipe") pipe_mode = true;
        else if (arg == "--short") short_frames = true;
        else if (arg == "--sps" && i + 1 < argc) sps = atoi(argv[++i]);
        else if (arg == "--rolloff" && i + 1 < argc) rolloff = atof(argv[++i]);
        else if (arg == "--gold-code" && i + 1 < argc) gold_code = atoi(argv[++i]);
        else if (arg == "--out-file" && i + 1 < argc) out_file = argv[++i];
        else if (arg == "--n-frames" && i + 1 < argc) n_frames = atoi(argv[++i]);
        else if (arg == "--awgn-snr-db" && i + 1 < argc) {
            awgn_enabled = true;
            awgn_snr_db = atof(argv[++i]);
        }
        else if (arg == "--noise-seed" && i + 1 < argc)
            noise_seed = (unsigned int)strtoul(argv[++i], nullptr, 10);
        else if (arg == "--sym-rate") sym_rate = true;
        else if (arg == "--pace-sample-rate" && i + 1 < argc)
            pace_sample_rate = atof(argv[++i]);
        else if (arg == "--frame-trace-csv" && i + 1 < argc)
            frame_trace_csv = argv[++i];
        else if (arg == "--gse-flush-ms" && i + 1 < argc)
            gse_flush_ms = atoi(argv[++i]);
        else if (arg == "--log") log_enabled = true;
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
    if (sps != 2 && sps != 4) {
        fprintf(stderr, "Error: --sps must be 2 or 4 (got %d). Note: the sps=2 "
                "timing path is currently unreliable; 4 is the validated default.\n", sps);
        return 1;
    }
    if (pace_sample_rate < 0.0 || gse_flush_ms < 0) {
        fprintf(stderr, "Error: pacing and GSE flush values must be non-negative\n");
        return 1;
    }

    // --- Open I/O ---
    std::unique_ptr<tap_interface> tap;
    int in_fd = STDIN_FILENO;
    if (!tap_name.empty()) {
        tap = std::make_unique<tap_interface>(tap_name);
        in_fd = tap->fd();
    }

    int out_fd = STDOUT_FILENO;
    if (!out_file.empty()) {
        out_fd = open(out_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out_fd < 0) { perror("Cannot open output file"); return 1; }
    }

    // --- Create Tx pipeline ---
    fecframe_size_t framesize = short_frames ? FECFRAME_SHORT : FECFRAME_NORMAL;
    if (short_frames && !modcod_supports_framesize(mc, framesize)) {
        fprintf(stderr, "Error: %s has no short-FECFRAME support\n", modcod_str.c_str());
        return 1;
    }
    tx_fec_chain fec(framesize, mc.code_rate, mc.constellation);
    fec.set_rolloff(rolloff);
    pl_framer framer((uint8_t)mc.modcod_idx, framesize == FECFRAME_SHORT, pilots, gold_code);

    std::unique_ptr<transmitter_frontend> frontend;
    const int rrc_delay = 25;
    if (!sym_rate)
        frontend = std::make_unique<transmitter_frontend>(sps, rolloff, rrc_delay);

    // --- Buffers ---
    const unsigned int max_dfl = fec.max_dfl_bytes();
    const unsigned int xfecframe_len = fec.xfecframe_len();
    const unsigned int plframe_len = framer.plframe_len();

    std::vector<uint8_t> gse_data(max_dfl);
    std::vector<uint8_t> pdu_buf(2048);
    std::vector<complexf> xfecframe(xfecframe_len);
    std::vector<complexf> plframe(plframe_len);

    unsigned int max_iq_out = 0;
    std::vector<complexf> iq_out;
    if (!sym_rate) {
        max_iq_out = plframe_len * sps + frontend->ntaps() * 2;
        iq_out.resize(max_iq_out);
    }

    // GSE encapsulator (broadcast, IPv4 protocol type)
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    std::mt19937 noise_rng(noise_seed);
    FILE* frame_trace = nullptr;
    if (!frame_trace_csv.empty()) {
        frame_trace = fopen(frame_trace_csv.c_str(), "w");
        if (!frame_trace) {
            perror("Cannot open TX frame trace CSV");
            return 1;
        }
        fprintf(frame_trace,
                "time_s,frame,modcod,pdus,application_bytes,gse_used_bytes,"
                "gse_padding_bytes,iq_samples,awgn_sample_snr_db\n");
    }

    fprintf(stderr,
            "dvbs2-gse-tx: modcod=%s sps=%d rolloff=%.2f "
            "max_dfl=%u awgn=%s",
            modcod_str.c_str(),
            sym_rate ? 1 : sps, rolloff, max_dfl,
            awgn_enabled ? "on" : "off");
    if (awgn_enabled)
        fprintf(stderr, "(%.1f dB, seed=%u)", awgn_snr_db, noise_seed);
    fprintf(stderr, " input=%s\n",
            tap_name.empty() ? "pipe" : tap_name.c_str());

    // --- Helper: encode and output one frame ---
    auto next_output_due = std::chrono::steady_clock::now();
    auto encode_and_output = [&](uint8_t* gse_buf, unsigned int gse_len) {
        fec.process_gse(gse_buf, gse_len, xfecframe.data());
        framer.frame(xfecframe.data(), plframe.data());
        unsigned int n_output_samples = plframe_len;
        if (sym_rate) {
            if (awgn_enabled)
                apply_awgn(plframe.data(), plframe_len, awgn_snr_db, noise_rng);
            if (pace_sample_rate > 0.0) {
                const auto now = std::chrono::steady_clock::now();
                if (now > next_output_due) next_output_due = now;
                std::this_thread::sleep_until(next_output_due);
                next_output_due += std::chrono::duration_cast<
                    std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(plframe_len /
                                                      pace_sample_rate));
            }
            ssize_t w = write(out_fd, plframe.data(), plframe_len * sizeof(complexf));
            (void)w;
        } else {
            unsigned int n_iq = frontend->pulse_shape(plframe.data(), plframe_len, iq_out.data());
            n_output_samples = n_iq;
            if (awgn_enabled)
                apply_awgn(iq_out.data(), n_iq, awgn_snr_db, noise_rng);
            if (pace_sample_rate > 0.0) {
                const auto now = std::chrono::steady_clock::now();
                // Blocking for the next MGEN packet creates a genuine idle
                // gap. Rebase instead of catching up by emitting a burst.
                if (now > next_output_due) next_output_due = now;
                std::this_thread::sleep_until(next_output_due);
                next_output_due += std::chrono::duration_cast<
                    std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(n_iq /
                                                      pace_sample_rate));
            }
            ssize_t w = write(out_fd, iq_out.data(), n_iq * sizeof(complexf));
            (void)w;
        }
        return n_output_samples;
    };

    // --- Main loop: fill GSE data fields and encode ---
    auto start_time = std::chrono::steady_clock::now();
    uint64_t frame_count = 0;
    uint64_t pdu_count = 0;
    uint64_t application_bytes = 0;
    bool eof = false;
    std::vector<uint8_t> pending_pdu;

    while (!eof || enc.has_pending() || !pending_pdu.empty()) {
        if (n_frames > 0 && (int)frame_count >= n_frames) break;

        int gse_pos = 0;
        const uint64_t frame_pdu_start = pdu_count;
        const uint64_t frame_bytes_start = application_bytes;

        // 1. Write pending GSE fragments from previous frame
        while (enc.has_pending() && gse_pos + 4 <= (int)max_dfl) {
            int w = enc.write_pending(gse_data.data() + gse_pos, max_dfl - gse_pos);
            if (w == 0) break;
            gse_pos += w;
        }

        // 2. Try to fit the saved PDU from previous iteration
        if (!pending_pdu.empty() && !enc.has_pending()) {
            int w = enc.encapsulate(pending_pdu.data(), pending_pdu.size(),
                                    gse_data.data() + gse_pos, max_dfl - gse_pos);
            if (w > 0) {
                const size_t accepted_bytes = pending_pdu.size();
                gse_pos += w;
                pending_pdu.clear();
                pdu_count++;
                application_bytes += accepted_bytes;
            } else if (w < 0) {
                pending_pdu.clear();
            }
        }

        // 3. Read and encapsulate new PDUs until frame is full
        while (pending_pdu.empty() && !enc.has_pending() && !eof) {
            int pdu_len;
            if (tap) {
                struct pollfd pfd = {in_fd, POLLIN, 0};
                int ready = poll(&pfd, 1, gse_flush_ms);
                pdu_len = ready > 0
                    ? tap->read_frame(pdu_buf.data(), pdu_buf.size())
                    : (ready == 0 ? 0 : -1);
            } else {
                pdu_len = read_pdu_pipe_timeout(in_fd, pdu_buf.data(),
                                                pdu_buf.size(), gse_flush_ms);
            }
            if (pdu_len == 0) break; // frame-fill timeout: emit partial frame
            if (pdu_len < 0) { eof = true; break; }

            // TAP mode: strip Ethernet header, encapsulate L3 (IP) only.
            // Pipe mode: data is already L3.
            uint8_t* payload = pdu_buf.data();
            int payload_len = pdu_len;
            if (tap) {
                if (pdu_len <= ETH_HLEN) continue; // runt frame
                uint16_t etype = ((uint16_t)pdu_buf[12] << 8) | pdu_buf[13];
                if (etype != ETH_P_IP) continue;   // drop non-IPv4
                payload += ETH_HLEN;
                payload_len -= ETH_HLEN;
            }

            int w = enc.encapsulate(payload, payload_len,
                                    gse_data.data() + gse_pos, max_dfl - gse_pos);
            if (w > 0) {
                gse_pos += w;
                pdu_count++;
                application_bytes += payload_len;
            } else if (w < 0) {
                continue;
            } else {
                // Doesn't fit, save for next frame (already stripped)
                pending_pdu.assign(payload, payload + payload_len);
                break;
            }
        }

        if (gse_pos == 0 && !enc.has_pending() && pending_pdu.empty()) {
            if (eof) break;
            continue;
        }

        // 4. Pad remaining data field with GSE padding
        const int gse_used = gse_pos;
        if (gse_pos < (int)max_dfl)
            gse_pos += gse_encapsulator::write_padding(gse_data.data() + gse_pos,
                                                       max_dfl - gse_pos);

        // 5. Encode and output
        const unsigned int n_iq = encode_and_output(gse_data.data(), max_dfl);
        frame_count++;
        if (frame_trace) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start_time).count();
            fprintf(frame_trace, "%.9f,%lu,%s,%lu,%lu,%d,%d,%u,",
                    elapsed, (unsigned long)frame_count, mc.name,
                    (unsigned long)(pdu_count - frame_pdu_start),
                    (unsigned long)(application_bytes - frame_bytes_start),
                    gse_used, (int)max_dfl - gse_used, n_iq);
            if (awgn_enabled) fprintf(frame_trace, "%.3f\n", awgn_snr_db);
            else fprintf(frame_trace, "\n");
            fflush(frame_trace);
        }

        if (log_enabled && (frame_count % 10 == 0)) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            fprintf(stderr, "dvbs2-gse-tx: %lu frames, %lu PDUs, %.1f sec\n",
                    (unsigned long)frame_count, (unsigned long)pdu_count, elapsed);
        }
    }

    // Flush RRC filter tail
    if (!sym_rate && frontend) {
        unsigned int n_flush = frontend->flush(iq_out.data());
        if (n_flush > 0) {
            ssize_t w = write(out_fd, iq_out.data(), n_flush * sizeof(complexf));
            (void)w;
        }
    }

    if (out_fd != STDOUT_FILENO) close(out_fd);
    if (frame_trace) fclose(frame_trace);

    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    fprintf(stderr, "dvbs2-gse-tx: finished. %.1f sec, %lu frames, %lu PDUs\n",
            total_sec, (unsigned long)frame_count, (unsigned long)pdu_count);

    return 0;
}
