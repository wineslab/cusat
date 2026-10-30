/* SPDX-License-Identifier: Apache-2.0 */

#include "channel_impairment.h"

#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

uint64_t realtime_ns()
{
    struct timespec now{};
    clock_gettime(CLOCK_REALTIME, &now);
    return static_cast<uint64_t>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
}

uint16_t unused_udp_port()
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    require(sock >= 0, "failed to create temporary UDP socket");
    struct sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    require(bind(sock, reinterpret_cast<struct sockaddr*>(&address),
                 sizeof(address)) == 0,
            "failed to bind temporary UDP socket");
    socklen_t length = sizeof(address);
    require(getsockname(sock, reinterpret_cast<struct sockaddr*>(&address),
                        &length) == 0,
            "failed to read temporary UDP port");
    const uint16_t port = ntohs(address.sin_port);
    close(sock);
    return port;
}

void send_packet(int sock, uint16_t port, const void* data, size_t bytes)
{
    struct sockaddr_in destination{};
    destination.sin_family = AF_INET;
    destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    destination.sin_port = htons(port);
    require(sendto(sock, data, bytes, 0,
                   reinterpret_cast<struct sockaddr*>(&destination),
                   sizeof(destination)) == static_cast<ssize_t>(bytes),
            "failed to send test command");
}

} // namespace

int main()
{
    // A fixed propagation delay must move IQ samples without changing them.
    // Request 3.6 samples so this test also verifies nearest-sample rounding.
    const uint16_t delay_port = unused_udp_port();
    constexpr double delay_rate_hz = 1000.0;
    constexpr double requested_delay_samples = 3.6;
    const double delay_distance_km = requested_delay_samples /
        delay_rate_hz * channel_impairment::LIGHT_MPS / 1000.0;
    channel_impairment delay_impairment(
        delay_port, channel_impairment::PATHLOSS, 0.001f,
        delay_rate_hz, true, delay_distance_km, 64 * 1024);
    require(static_cast<bool>(delay_impairment),
            "delay impairment failed to bind");
    require(delay_impairment.delay_samples() == 4,
            "propagation delay was not rounded to the nearest sample");

    std::complex<float> delayed[16];
    for (int i = 0; i < 16; ++i)
        delayed[i] = {(i & 1) ? -1.0f : 1.0f,
                      static_cast<float>(i + 1)};
    const std::complex<float> original[16] = {
        delayed[0], delayed[1], delayed[2], delayed[3],
        delayed[4], delayed[5], delayed[6], delayed[7],
        delayed[8], delayed[9], delayed[10], delayed[11],
        delayed[12], delayed[13], delayed[14], delayed[15]};
    delay_impairment.apply(delayed, 16);
    for (int i = 0; i < 4; ++i)
        require(delayed[i] == std::complex<float>(0.0f, 0.0f),
                "delay line startup was not zero-filled");
    for (int i = 4; i < 16; ++i)
        require(delayed[i] == original[i - 4],
                "integer delay changed an IQ sample");

    const uint16_t port = unused_udp_port();
    channel_impairment impairment(port, channel_impairment::PATHLOSS);
    require(static_cast<bool>(impairment), "channel impairment failed to bind");

    int controller = socket(AF_INET, SOCK_DGRAM, 0);
    require(controller >= 0, "controller socket failed");
    struct timeval timeout{1, 0};
    require(setsockopt(controller, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       sizeof(timeout)) == 0,
            "controller timeout setup failed");

    // Scheduled format: command magic, float value, Unix/UTC ns, sequence.
    uint8_t scheduled[channel_impairment::SCHEDULED_COMMAND_BYTES]{};
    const uint32_t magic = channel_impairment::MAGIC;
    const float loss_db = 6.0f;
    const uint64_t requested_ns = realtime_ns() + 100000000ULL;
    const uint64_t sequence = 42;
    memcpy(scheduled, &magic, 4);
    memcpy(scheduled + 4, &loss_db, 4);
    memcpy(scheduled + 8, &requested_ns, 8);
    memcpy(scheduled + 16, &sequence, 8);
    send_packet(controller, port, scheduled, sizeof(scheduled));

    uint8_t ack[channel_impairment::ACK_BYTES]{};
    auto receive_ack = [&](uint32_t expected_magic,
                           uint64_t expected_sequence,
                           uint64_t expected_requested_ns) {
        require(recv(controller, ack, sizeof(ack), 0) ==
                static_cast<ssize_t>(sizeof(ack)),
                "scheduled command ACK was not returned");
        uint32_t ack_magic = 0;
        uint32_t status = 0;
        uint64_t ack_sequence = 0;
        uint64_t ack_requested_ns = 0;
        uint64_t event_ns = 0;
        memcpy(&ack_magic, ack, 4);
        memcpy(&status, ack + 4, 4);
        memcpy(&ack_sequence, ack + 8, 8);
        memcpy(&ack_requested_ns, ack + 16, 8);
        memcpy(&event_ns, ack + 24, 8);
        require(ack_magic == expected_magic, "wrong ACK magic");
        require(status == channel_impairment::ACK_APPLIED, "wrong ACK status");
        require(ack_sequence == expected_sequence, "wrong ACK sequence");
        require(ack_requested_ns == expected_requested_ns,
                "wrong ACK deadline");
        return event_ns;
    };

    std::complex<float> sample(1.0f, 0.0f);
    impairment.apply(&sample, 1);
    require(std::abs(sample.real() - 1.0f) < 1e-6f,
            "scheduled command applied before its deadline");
    const uint64_t queued_ns = receive_ack(
        channel_impairment::QUEUE_ACK_MAGIC, sequence, requested_ns);
    require(queued_ns < requested_ns, "queue ACK did not precede deadline");

    while (realtime_ns() < requested_ns + 2000000ULL) {
        sample = {1.0f, 0.0f};
        impairment.apply(&sample, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    sample = {1.0f, 0.0f};
    impairment.apply(&sample, 1);
    require(std::abs(sample.real() - std::pow(10.0f, -6.0f / 20.0f)) < 1e-5f,
            "scheduled command was not applied");

    const uint64_t applied_ns = receive_ack(
        channel_impairment::ACK_MAGIC, sequence, requested_ns);
    require(applied_ns >= requested_ns, "ACK predates requested application");

    // The original 8-byte command remains immediate and unacknowledged.
    uint8_t legacy[channel_impairment::LEGACY_COMMAND_BYTES]{};
    const float legacy_loss_db = 20.0f;
    memcpy(legacy, &magic, 4);
    memcpy(legacy + 4, &legacy_loss_db, 4);
    send_packet(controller, port, legacy, sizeof(legacy));
    sample = {1.0f, 0.0f};
    impairment.apply(&sample, 1);
    require(std::abs(sample.real() - 0.1f) < 1e-6f,
            "legacy command did not remain immediate");

    // CLER removes future commands but retains the current channel state.
    const float future_loss_db = 3.0f;
    const uint64_t future_requested_ns = realtime_ns() + 30000000ULL;
    const uint64_t future_sequence = 43;
    memcpy(scheduled + 4, &future_loss_db, 4);
    memcpy(scheduled + 8, &future_requested_ns, 8);
    memcpy(scheduled + 16, &future_sequence, 8);
    send_packet(controller, port, scheduled, sizeof(scheduled));
    sample = {1.0f, 0.0f};
    impairment.apply(&sample, 1);
    receive_ack(channel_impairment::QUEUE_ACK_MAGIC, future_sequence,
                future_requested_ns);

    const uint32_t clear_magic = channel_impairment::CLEAR_MAGIC;
    const float clear_value = 0.0f;
    memcpy(legacy, &clear_magic, 4);
    memcpy(legacy + 4, &clear_value, 4);
    send_packet(controller, port, legacy, sizeof(legacy));
    impairment.apply(&sample, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(35));
    sample = {1.0f, 0.0f};
    impairment.apply(&sample, 1);
    require(std::abs(sample.real() - 0.1f) < 1e-6f,
            "cleared command changed the retained channel state");

    close(controller);
    return 0;
}
