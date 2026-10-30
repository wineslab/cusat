/* SPDX-License-Identifier: Apache-2.0 */
#include "trx/radio_device.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

class radio_tcp final : public radio_device {
public:
    ~radio_tcp() override {
        if (d_tx >= 0) close(d_tx);
        if (d_rx >= 0) close(d_rx);
        if (d_listener >= 0) close(d_listener);
    }

    bool setup(const radio_config& cfg) override {
        d_rate = cfg.rate;
        std::string tx_host;
        int tx_port = 0, rx_port = 0;
        parse(cfg.usrp_args, tx_host, tx_port, rx_port);
        if (cfg.do_rx && rx_port <= 0) {
            fprintf(stderr, "radio_tcp: rx_listen=<port> is required\n");
            return false;
        }
        if (cfg.do_tx && (tx_host.empty() || tx_port <= 0)) {
            fprintf(stderr, "radio_tcp: tx=<host>:<port> is required\n");
            return false;
        }
        if (cfg.do_rx && !listen_on(rx_port)) return false;
        if (cfg.do_tx && !connect_to(tx_host, tx_port)) return false;
        if (cfg.do_rx) {
            d_rx = accept(d_listener, nullptr, nullptr);
            if (d_rx < 0) { perror("radio_tcp accept"); return false; }
        }
        d_epoch = std::chrono::steady_clock::now();
        fprintf(stderr, "radio_tcp: ready rate=%.3f Msps tx=%s:%d rx=:%d\n",
                d_rate / 1e6, tx_host.c_str(), tx_port, rx_port);
        return true;
    }

    double get_time_now() override {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - d_epoch).count();
    }
    size_t rx_max_samps() override { return 65536; }
    size_t tx_max_samps() override { return 65536; }
    void start_rx_continuous(double) override {}
    void stop_rx() override {}

    size_t recv(void* buf, size_t nsamps, radio_rx_meta& md, double timeout) override {
        md.error = radio_rx_error::none;
        size_t want = nsamps * sizeof(float) * 2, got = 0;
        timeval tv{(long)timeout, (long)((timeout - (long)timeout) * 1e6)};
        setsockopt(d_rx, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        while (got < want) {
            ssize_t n = ::recv(d_rx, (char*)buf + got, want - got, 0);
            if (n > 0) { got += (size_t)n; continue; }
            if (n == 0) { md.error = radio_rx_error::other; break; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                md.error = radio_rx_error::timeout; break;
            }
            if (errno == EINTR) continue;
            md.error = radio_rx_error::other; break;
        }
        return got / (sizeof(float) * 2);
    }

    size_t send(const void* buf, size_t nsamps, const radio_tx_meta&, double) override {
        if (nsamps == 0) return 0;
        if (!d_tx_started) {
            d_tx_epoch = std::chrono::steady_clock::now();
            d_tx_started = true;
        }
        size_t want = nsamps * sizeof(float) * 2, sent = 0;
        while (sent < want) {
            ssize_t n = ::send(d_tx, (const char*)buf + sent, want - sent, MSG_NOSIGNAL);
            if (n > 0) sent += (size_t)n;
            else if (n < 0 && errno == EINTR) continue;
            else break;
        }
        d_tx_samples += sent / (sizeof(float) * 2);
        if (d_rate > 0) {
            auto due = d_tx_epoch + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(d_tx_samples / d_rate));
            std::this_thread::sleep_until(due);
        }
        return sent / (sizeof(float) * 2);
    }
    radio_async_msg poll_tx_async() override { return {}; }

private:
    static void parse(const std::string& args, std::string& host, int& tx, int& rx) {
        size_t start = 0;
        while (start <= args.size()) {
            size_t end = args.find(',', start);
            std::string item = args.substr(start, end - start);
            if (item.rfind("tx=", 0) == 0) {
                std::string value = item.substr(3);
                size_t colon = value.rfind(':');
                if (colon != std::string::npos) {
                    host = value.substr(0, colon);
                    tx = atoi(value.c_str() + colon + 1);
                }
            } else if (item.rfind("rx_listen=", 0) == 0) {
                rx = atoi(item.c_str() + 10);
            }
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }
    bool listen_on(int port) {
        d_listener = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(d_listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY;
        a.sin_port = htons((uint16_t)port);
        if (bind(d_listener, (sockaddr*)&a, sizeof(a)) || listen(d_listener, 1)) {
            perror("radio_tcp listen"); return false;
        }
        return true;
    }
    bool connect_to(const std::string& host, int port) {
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
        if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) return false;
        for (int i = 0; i < 100; ++i) {
            d_tx = socket(AF_INET, SOCK_STREAM, 0);
            if (connect(d_tx, (sockaddr*)&a, sizeof(a)) == 0) return true;
            close(d_tx); d_tx = -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        perror("radio_tcp connect"); return false;
    }
    int d_tx = -1, d_rx = -1, d_listener = -1;
    double d_rate = 0;
    uint64_t d_tx_samples = 0;
    bool d_tx_started = false;
    std::chrono::steady_clock::time_point d_epoch = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point d_tx_epoch = d_epoch;
};

extern "C" radio_device* radio_device_create() { return new radio_tcp(); }
