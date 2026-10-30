// SPDX-License-Identifier: Apache-2.0
#pragma once
// channel_impairment.h — Receives channel commands via UDP, applies impairments
// to TX IQ samples. Header-only, lives in apps/.
//
// Two modes:
//   NOISE  — legacy additive AWGN injection (--noise)
//   PATHLOSS — relative free-space attenuation (--pathloss)
//
// In PATHLOSS mode, DIST derives differential free-space loss relative to the
// configured reference distance. LINK independently represents whether the
// satellite is in view. The physical receiver supplies the absolute noise
// floor, while radio gain establishes the clear-sky operating point.
//
// Uses a pre-generated circular noise buffer so the hot path is just
// a scaled add + multiply — no per-sample RNG calls in the TX send thread.
//
// Legacy 8-byte runtime updates take effect on the next TX chunk. A 24-byte
// extension adds an absolute CLOCK_REALTIME deadline and sequence number; the
// command is queued without blocking the real-time thread. The modem
// acknowledges queue insertion and application as separate events.
// The external channel controller owns any desired trajectory smoothing.

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <queue>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include <vector>

class channel_impairment {
public:
    enum mode_t { NOISE, PATHLOSS };

    static constexpr uint32_t MAGIC = 0x4E4F4953; // 'NOIS' — SNR/pathloss command
    static constexpr uint32_t DIST_MAGIC = 0x44495354; // 'DIST' — path distance (km)
    static constexpr uint32_t LINK_MAGIC = 0x4C494E4B; // 'LINK' — 1=available, 0=out of view
    static constexpr uint32_t ACK_MAGIC = 0x41434B31; // 'ACK1' — scheduled command applied
    static constexpr uint32_t QUEUE_ACK_MAGIC = 0x41434B51; // 'ACKQ' — command queued
    static constexpr uint32_t CLEAR_MAGIC = 0x434C4552; // 'CLER' — discard future commands
    static constexpr uint32_t ACK_APPLIED = 0;
    static constexpr uint32_t ACK_LATE_ON_ARRIVAL = 1U << 0;
    static constexpr uint32_t ACK_QUEUE_FULL = 1U << 1;
    static constexpr size_t LEGACY_COMMAND_BYTES = 8;
    static constexpr size_t SCHEDULED_COMMAND_BYTES = 24;
    static constexpr size_t ACK_BYTES = 32;
    static constexpr size_t MAX_SCHEDULED_COMMANDS = 32768;
    static constexpr int NOISE_BUF_SIZE = 262144;  // 256K complex samples

    // --- propagation-delay line (orbital-regime emulator) ---
    static constexpr double   LIGHT_MPS = 299792458.0;
    // Fixed 512 MiB ring (complex<float> = 8 B/sample).
    static constexpr uint64_t DELAY_BUF_BYTES = 512ULL * 1024 * 1024;
    static constexpr double   GEO_ALT_KM = 35786.0;   // GEO altitude (one-way proxy)
    // Hard cap on requested distance = 2x GEO (clamped further to ring capacity).
    static constexpr double   MAX_DISTANCE_KM = 2.0 * GEO_ALT_KM;
    // thermal_noise_std is retained for source compatibility with the older
    // synthetic-noise path; hardware PATHLOSS operation uses receiver noise.
    explicit channel_impairment(int listen_port, mode_t mode = NOISE,
                                float thermal_noise_std = 0.001f,
                                double sample_rate_hz = 0.0,
                                bool delay_enabled = false,
                                double init_distance_km = 0.0,
                                uint64_t delay_buf_bytes = DELAY_BUF_BYTES)
        : d_mode(mode), d_thermal_noise_std(thermal_noise_std),
          d_sample_rate(sample_rate_hz),
          d_reference_distance_km(init_distance_km)
    {
        d_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (d_sock < 0) return;

        int flags = fcntl(d_sock, F_GETFL, 0);
        fcntl(d_sock, F_SETFL, flags | O_NONBLOCK);

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(listen_port);
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(d_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(d_sock);
            d_sock = -1;
            return;
        }

        // Pre-generate unit-variance Gaussian noise buffer
        d_noise_buf.resize(NOISE_BUF_SIZE);
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < NOISE_BUF_SIZE; i++)
            d_noise_buf[i] = std::complex<float>(dist(gen), dist(gen));

        // Propagation-delay ring (only if requested and we know the sample rate).
        if (delay_enabled && sample_rate_hz > 0.0) {
            size_t cap = static_cast<size_t>(
                delay_buf_bytes / sizeof(std::complex<float>));
            if (cap <= 1024)
                return;
            d_delay_buf.assign(cap, std::complex<float>(0.0f, 0.0f));
            // Cap = min(one GEO bent-pipe direction, ring capacity minus a guard).
            double cap2geo = km_to_samps(MAX_DISTANCE_KM);
            d_delay_max_samps = std::min((double)(cap - 1024), cap2geo);
            d_delay_enabled = true;
            // A two-tap linear interpolation is not an all-pass fractional
            // delay. At low SPS it attenuates the occupied band and introduces
            // ISI. Round once at startup so the delay line only moves samples.
            // The resulting time error is at most half a sample.
            d_delay_samps = std::round(std::max(0.0, std::min(
                km_to_samps(init_distance_km), d_delay_max_samps)));
            set_distance_km(init_distance_km);
        }

    }

    ~channel_impairment() {
        if (d_sock >= 0) ::close(d_sock);
    }

    // Called from TX send thread before UHD send.
    void apply(std::complex<float>* samples, int n_samples)
    {
        if (d_sock < 0) return;

        // Poll without blocking. Legacy commands retain their immediate
        // semantics; scheduled commands enter a deadline-ordered queue.
        uint8_t buf[SCHEDULED_COMMAND_BYTES];
        while (true) {
            struct sockaddr_in peer{};
            socklen_t peer_len = sizeof(peer);
            ssize_t n = recvfrom(d_sock, buf, sizeof(buf), MSG_DONTWAIT,
                                 reinterpret_cast<struct sockaddr*>(&peer),
                                 &peer_len);
            if (n < 0) break;
            if (n != static_cast<ssize_t>(LEGACY_COMMAND_BYTES) &&
                n != static_cast<ssize_t>(SCHEDULED_COMMAND_BYTES))
                continue;

            uint32_t magic;
            float value;
            memcpy(&magic, buf, 4);
            memcpy(&value, buf + 4, 4);
            if (n == static_cast<ssize_t>(LEGACY_COMMAND_BYTES)) {
                if (magic == CLEAR_MAGIC) {
                    clear_schedule();
                    continue;
                }
                if (!valid_magic(magic)) continue;
                apply_command(magic, value);
                continue;
            }
            if (!valid_magic(magic)) continue;

            uint64_t apply_unix_ns;
            uint64_t sequence;
            memcpy(&apply_unix_ns, buf + 8, 8);
            memcpy(&sequence, buf + 16, 8);
            const uint64_t received_ns = realtime_ns();
            uint32_t status = ACK_APPLIED;
            if (received_ns >= apply_unix_ns)
                status |= ACK_LATE_ON_ARRIVAL;

            if (d_scheduled.size() >= MAX_SCHEDULED_COMMANDS) {
                send_ack(QUEUE_ACK_MAGIC, peer, sequence, apply_unix_ns,
                         received_ns, status | ACK_QUEUE_FULL);
                continue;
            }

            scheduled_command command{
                magic, value, apply_unix_ns, sequence, status, peer};
            d_scheduled.push(command);
            send_ack(QUEUE_ACK_MAGIC, peer, sequence, apply_unix_ns,
                     received_ns, status);
        }

        // Do not sleep in this real-time path. Commands become due naturally
        // as TX/RX chunks arrive; ACK time therefore exposes both clock error
        // and chunk-boundary application latency to the controller.
        const uint64_t now_ns = realtime_ns();
        while (!d_scheduled.empty() &&
               d_scheduled.top().apply_unix_ns <= now_ns) {
            const scheduled_command command = d_scheduled.top();
            d_scheduled.pop();
            apply_command(command.magic, command.value);
            const uint64_t applied_ns = realtime_ns();
            send_ack(ACK_MAGIC, command.peer, command.sequence,
                     command.apply_unix_ns, applied_ns, command.status);
        }

        // Link availability is separate from path loss: orbital range is only
        // meaningful while the satellite is in view. Feed zeros into the delay
        // line while unavailable so stale signal cannot reappear at recovery.
        const bool link_available =
            d_link_available.load(std::memory_order_relaxed);
        if (!link_available)
            std::fill(samples, samples + n_samples,
                      std::complex<float>(0.0f, 0.0f));

        // Propagation delay first (signal travels, *then* noise is added at RX).
        // The IQ delay is fixed at the YAML startup distance. Runtime DIST
        // commands vary path loss only: moving an IQ delay tap necessarily
        // time-scales the waveform (baseband Doppler), which this deliberately
        // Doppler-free emulator does not claim to model.
        if (d_delay_enabled)
            apply_delay(samples, n_samples);

        if (!link_available) {
            std::fill(samples, samples + n_samples,
                      std::complex<float>(0.0f, 0.0f));
            return;
        }

        if (d_mode == PATHLOSS)
            apply_pathloss(samples, n_samples);
        else
            apply_noise(samples, n_samples);
    }

    float current_db() const {
        return d_target_db.load(std::memory_order_relaxed);
    }

    // Monotonic notification for the RX frontend.  A hard visibility gap
    // invalidates acquisition state even when the emulator itself is placed
    // on the TX path; the orbital controller sends the same visibility event
    // to both ends of the bidirectional link.
    uint64_t link_up_epoch() const {
        return d_link_up_epoch.load(std::memory_order_acquire);
    }

    // Set the path-loss distance at runtime. The propagation-delay line remains
    // fixed at the YAML startup/reference distance. Updating it on a live IQ
    // stream either cuts/repeats samples or introduces a sample-rate offset;
    // both are Doppler effects and used to cause false lock loss in accelerated
    // orbital demonstrations.
    void set_distance_km(double km) {
        // The real radio gains/noise establish the absolute operating point.
        // Model only the free-space variation relative to the YAML start range:
        // FSPL(d)-FSPL(d0) = 20 log10(d/d0).  If YAML used zero, the first
        // positive UDP distance becomes the reference point.
        if (d_mode == PATHLOSS && km > 0.0) {
            if (d_reference_distance_km <= 0.0)
                d_reference_distance_km = km;
            const float loss_db = 20.0f * std::log10(
                static_cast<float>(km / d_reference_distance_km));
            d_target_db.store(loss_db, std::memory_order_relaxed);
        }
    }
    double current_distance_km() const {
        if (!d_delay_enabled || d_sample_rate <= 0.0) return 0.0;
        return d_delay_samps / d_sample_rate * LIGHT_MPS / 1000.0;
    }
    bool delay_enabled() const { return d_delay_enabled; }
    size_t delay_samples() const {
        return static_cast<size_t>(d_delay_samps);
    }

    mode_t mode() const { return d_mode; }

    explicit operator bool() const { return d_sock >= 0; }

private:
    struct scheduled_command {
        uint32_t magic;
        float value;
        uint64_t apply_unix_ns;
        uint64_t sequence;
        uint32_t status;
        struct sockaddr_in peer;
    };

    struct scheduled_command_later {
        bool operator()(const scheduled_command& lhs,
                        const scheduled_command& rhs) const
        {
            if (lhs.apply_unix_ns != rhs.apply_unix_ns)
                return lhs.apply_unix_ns > rhs.apply_unix_ns;
            return lhs.sequence > rhs.sequence;
        }
    };

    static bool valid_magic(uint32_t magic)
    {
        return magic == MAGIC || magic == DIST_MAGIC || magic == LINK_MAGIC;
    }

    static uint64_t realtime_ns()
    {
        struct timespec now{};
        clock_gettime(CLOCK_REALTIME, &now);
        return static_cast<uint64_t>(now.tv_sec) * 1000000000ULL +
               static_cast<uint64_t>(now.tv_nsec);
    }

    void apply_command(uint32_t magic, float value)
    {
        if (magic == MAGIC) {
            d_target_db.store(value, std::memory_order_relaxed);
        } else if (magic == DIST_MAGIC) {
            set_distance_km(value); // runtime distance changes path loss only
        } else if (magic == LINK_MAGIC) {
            const bool available = value > 0.5f;
            const bool previous = d_link_available.exchange(
                available, std::memory_order_relaxed);
            if (available != previous) {
                if (available)
                    d_link_up_epoch.fetch_add(1, std::memory_order_release);
                fprintf(stderr, "channel_impairment: link %s\n",
                        available ? "available" : "out of view");
            }
        }
    }

    void clear_schedule()
    {
        while (!d_scheduled.empty())
            d_scheduled.pop();
    }

    void send_ack(uint32_t ack_magic, const struct sockaddr_in& peer,
                  uint64_t sequence,
                  uint64_t requested_ns, uint64_t applied_ns,
                  uint32_t status)
    {
        uint8_t ack[ACK_BYTES]{};
        memcpy(ack, &ack_magic, 4);
        memcpy(ack + 4, &status, 4);
        memcpy(ack + 8, &sequence, 8);
        memcpy(ack + 16, &requested_ns, 8);
        memcpy(ack + 24, &applied_ns, 8);
        (void)sendto(d_sock, ack, sizeof(ack), MSG_DONTWAIT,
                     reinterpret_cast<const struct sockaddr*>(&peer),
                     sizeof(peer));
    }

    // --- NOISE mode: additive AWGN (legacy behavior) ---
    void apply_noise(std::complex<float>* samples, int n_samples)
    {
        const float current_db = d_target_db.load(std::memory_order_relaxed);
        if (current_db <= 0.01f) return;

        float sig_pwr = 0.0f;
        for (int i = 0; i < n_samples; i++)
            sig_pwr += std::norm(samples[i]);
        sig_pwr /= n_samples;
        if (sig_pwr < 1e-12f) return;

        float sig_rms = sqrtf(sig_pwr);
        float injection_snr_db = 30.0f - 2.0f * current_db;
        float noise_std = sig_rms * powf(10.0f, -injection_snr_db / 20.0f);

        int pos = d_buf_pos;
        for (int i = 0; i < n_samples; i++) {
            samples[i] += d_noise_buf[pos] * noise_std;
            if (++pos >= NOISE_BUF_SIZE) pos = 0;
        }
        d_buf_pos = pos;
    }

    // --- PATHLOSS mode: attenuation only; receiver hardware supplies noise ---
    // d_target_db = differential path loss in dB (0 = configured start range)
    // Negative relative loss is valid when the new range is closer than the
    // configured reference and therefore increases received signal amplitude.
    void apply_pathloss(std::complex<float>* samples, int n_samples)
    {
        // At 0 dB relative loss the configured radio gain defines the link's
        // operating point. No synthetic noise is added in this model.
        const float current_db = d_target_db.load(std::memory_order_relaxed);
        float atten = powf(10.0f, -current_db / 20.0f);
        for (int i = 0; i < n_samples; i++) samples[i] *= atten;
    }

    // --- propagation delay: fixed integer-sample ring ---
    // Copy an earlier sample exactly. This is an all-pass delay and therefore
    // cannot change the waveform spectrum, EVM, or SNR. Sub-sample propagation
    // accuracy is immaterial at orbital delays; the rounding error is at most
    // half of one sample period.
    void apply_delay(std::complex<float>* samples, int n_samples)
    {
        if (d_delay_buf.empty()) return;
        const size_t CAP = d_delay_buf.size();
        size_t w = d_delay_w;
        size_t delay = static_cast<size_t>(d_delay_samps);
        if (delay >= CAP) delay = CAP - 1;
        for (int i = 0; i < n_samples; i++) {
            d_delay_buf[w] = samples[i];
            samples[i] = d_delay_buf[(w + CAP - delay) % CAP];
            if (++w >= CAP) w = 0;
        }
        d_delay_w = w;
    }

    double km_to_samps(double km) const {
        return (km * 1000.0 / LIGHT_MPS) * d_sample_rate;
    }

    mode_t d_mode;
    float d_thermal_noise_std;
    double d_sample_rate = 0.0;
    int d_sock = -1;
    std::atomic<float> d_target_db{0.0f};
    std::atomic<bool> d_link_available{true};
    std::atomic<uint64_t> d_link_up_epoch{0};
    std::priority_queue<scheduled_command, std::vector<scheduled_command>,
                        scheduled_command_later> d_scheduled;
    double d_reference_distance_km = 0.0;
    std::vector<std::complex<float>> d_noise_buf;
    int d_buf_pos = 0;

    // propagation-delay state
    bool d_delay_enabled = false;
    std::vector<std::complex<float>> d_delay_buf;
    size_t d_delay_w = 0;
    double d_delay_max_samps = 0.0;
    double d_delay_samps = 0.0;
};
