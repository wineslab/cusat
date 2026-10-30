// SPDX-License-Identifier: Apache-2.0
#pragma once
// spectrum_tap.h — Periodically writes raw IQ samples + KPIs to a named pipe.
// Header-only. Lives in apps/, not in lib/ — fully decoupled from the DSP stack.
// The sidecar Flask app (sidecar.py) reads from the pipe and serves over HTTP.
//
// Usage:
//   spectrum_tap tap("/tmp/dvbs2_spectrum");
//   // In RX frontend thread:
//   tap.feed(in_buf_ptr, n_samples);
//   // In main logging loop:
//   tap.set_kpis(snr, snr_var, modcod, tx_modcod, fer, locked);

#include <atomic>
#include <chrono>
#include <complex>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

class spectrum_tap {
public:
    static constexpr uint32_t MAGIC = 0x53504543; // 'SPEC'
    static constexpr int FFT_BINS = 1024;
    static constexpr int HEADER_SIZE = 32;
    // Header: magic(4) + snr(4) + snr_var(4) + modcod(1) + tx_modcod(1)
    //         + fer(4) + locked(1) + pad(1) + throughput_mbps(4)
    //         + rx_freq_mhz(4) + sample_rate_mhz(4)
    // Payload: 1024 × (re f32 + im f32) = 8192 bytes
    static constexpr int IQ_SIZE = FFT_BINS * 2 * sizeof(float); // 8192
    static constexpr int PACKET_SIZE = HEADER_SIZE + IQ_SIZE;     // 8212

    explicit spectrum_tap(const std::string& pipe_path = "/tmp/dvbs2_spectrum",
                          double interval_ms = 50.0)
        : d_pipe_path(pipe_path)
        , d_interval(std::chrono::microseconds((int)(interval_ms * 1000.0)))
        , d_last_send(std::chrono::steady_clock::now())
    {
        // Create FIFO if it doesn't exist
        struct stat st;
        if (stat(pipe_path.c_str(), &st) != 0) {
            if (mkfifo(pipe_path.c_str(), 0666) != 0) {
                d_ok = false;
                return;
            }
        }

        // Open non-blocking so we don't stall if no reader yet
        d_fd = open(pipe_path.c_str(), O_WRONLY | O_NONBLOCK);
        // ENXIO is normal — means no reader yet. We'll retry in feed().
        if (d_fd < 0 && errno != ENXIO) {
            d_ok = false;
            return;
        }

        // Pre-fill header magic
        uint32_t magic = MAGIC;
        memcpy(d_packet, &magic, 4);
        d_ok = true;
    }

    ~spectrum_tap() {
        if (d_fd >= 0) close(d_fd);
    }

    // Called from the main logging loop (every 200ms–1s) to update KPIs.
    // Thread-safe: stores to atomics read by feed().
    void set_kpis(float snr_db, float snr_var_db, uint8_t rx_modcod,
                  uint8_t tx_modcod, float fer, bool locked,
                  float throughput_mbps = 0.0f)
    {
        d_snr.store(snr_db, std::memory_order_relaxed);
        d_snr_var.store(snr_var_db, std::memory_order_relaxed);
        d_rx_modcod.store(rx_modcod, std::memory_order_relaxed);
        d_tx_modcod.store(tx_modcod, std::memory_order_relaxed);
        d_fer.store(fer, std::memory_order_relaxed);
        d_locked.store(locked, std::memory_order_relaxed);
        d_throughput.store(throughput_mbps, std::memory_order_relaxed);
    }

    // Called once at startup to set static RF parameters.
    void set_rf_params(double rx_freq_hz, double sample_rate_hz)
    {
        d_rx_freq_mhz = (float)(rx_freq_hz / 1e6);
        d_sample_rate_mhz = (float)(sample_rate_hz / 1e6);
    }

    // Called from the RX frontend thread after reading fresh IQ from the ring.
    // Copies up to FFT_BINS samples and writes a frame to the pipe.
    void feed(const std::complex<float>* samples, int n_samples)
    {
        if (!d_ok || n_samples < FFT_BINS) return;

        auto now = std::chrono::steady_clock::now();
        if (now - d_last_send < d_interval) return;
        d_last_send = now;

        // Lazy-open: retry if no reader was available at construction time
        if (d_fd < 0) {
            d_fd = open(d_pipe_path.c_str(), O_WRONLY | O_NONBLOCK);
            if (d_fd < 0) return; // still no reader
        }

        // Pack header (after magic which is pre-filled)
        float snr = d_snr.load(std::memory_order_relaxed);
        float snr_var = d_snr_var.load(std::memory_order_relaxed);
        uint8_t mc = d_rx_modcod.load(std::memory_order_relaxed);
        uint8_t txmc = d_tx_modcod.load(std::memory_order_relaxed);
        float fer = d_fer.load(std::memory_order_relaxed);
        uint8_t locked = d_locked.load(std::memory_order_relaxed) ? 1 : 0;
        uint8_t pad = 0;

        memcpy(d_packet + 4, &snr, 4);
        memcpy(d_packet + 8, &snr_var, 4);
        d_packet[12] = mc;
        d_packet[13] = txmc;
        memcpy(d_packet + 14, &fer, 4);
        d_packet[18] = locked;
        d_packet[19] = pad;
        float tput = d_throughput.load(std::memory_order_relaxed);
        memcpy(d_packet + 20, &tput, 4);
        memcpy(d_packet + 24, &d_rx_freq_mhz, 4);
        memcpy(d_packet + 28, &d_sample_rate_mhz, 4);

        // Copy raw IQ samples
        memcpy(d_packet + HEADER_SIZE, samples, IQ_SIZE);

        ssize_t ret = write(d_fd, d_packet, PACKET_SIZE);
        if (ret < 0 && (errno == EPIPE || errno == EBADF)) {
            // Reader disconnected, close and retry next time
            close(d_fd);
            d_fd = -1;
        }
    }

    explicit operator bool() const { return d_ok; }

private:
    std::string d_pipe_path;
    int d_fd = -1;
    bool d_ok = false;
    std::chrono::steady_clock::duration d_interval;
    std::chrono::steady_clock::time_point d_last_send;
    uint8_t d_packet[PACKET_SIZE]{};

    // KPIs — written by main thread, read by RX frontend thread
    std::atomic<float> d_snr{0.0f};
    std::atomic<float> d_snr_var{0.0f};
    std::atomic<uint8_t> d_rx_modcod{0};
    std::atomic<uint8_t> d_tx_modcod{0};
    std::atomic<float> d_fer{0.0f};
    std::atomic<bool> d_locked{false};
    std::atomic<float> d_throughput{0.0f};

    // Static RF parameters (set once at startup)
    float d_rx_freq_mhz = 0.0f;
    float d_sample_rate_mhz = 0.0f;
};
