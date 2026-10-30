// SPDX-License-Identifier: Apache-2.0
//
// tenant_server.h — public API for an embedded UHD-host "tenant server"
// that speaks the usrpproxy shm-ring protocol to an OAI softmodem running
// the usrpproxy OAI radio plugin.
//
// Intended usage inside a process that already owns a `multi_usrp::sptr`:
//
//     auto usrp = uhd::usrp::multi_usrp::make(args);
//     // ... the host's own init: subdev_spec, clock/time source, PPS,
//     // ... its own streamers on its own channels (e.g. 1, 3).
//
//     usrpproxy::TenantServer::Config cfg{};
//     cfg.shm_name      = "/usrpproxy";
//     cfg.tx_channels   = {0};       // physical UHD channel IDs reserved
//     cfg.rx_channels   = {0};       // for the OAI tenant
//     cfg.tx_core       = 28;
//     cfg.rx_core       = 29;
//     auto server = std::make_unique<usrpproxy::TenantServer>(usrp, cfg);
//     server->start();
//     // ... host's main loop ...
//     server->stop();  // or let the unique_ptr destructor handle it, but
//                      // ensure the server is destroyed BEFORE the usrp.

#ifndef USRPPROXY_TENANT_SERVER_H
#define USRPPROXY_TENANT_SERVER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <uhd/usrp/multi_usrp.hpp>

namespace usrpproxy {

class TenantServer {
public:
    struct Config {
        // POSIX shm segment name the OAI side will attach to. Leading "/" is
        // auto-added if missing. Must match OAI's `usrpproxy.shm_name` YAML.
        std::string shm_name = "/usrpproxy";

        // Physical UHD channel IDs reserved for the OAI tenant. OAI's logical
        // channel k binds to tx_channels[k] / rx_channels[k]. The server
        // rejects CMD_INIT if OAI asks for more channels than reserved.
        std::vector<int> tx_channels;
        std::vector<int> rx_channels;

        // Empty preserves UHD's default connector for the reserved channels.
        std::string tx_antenna;
        std::string rx_antenna;

        // CPU cores for the TX and RX worker threads. -1 = not pinned.
        int tx_core = -1;
        int rx_core = -1;

        // Ring capacities in bytes, each must be a power of two.
        uint64_t tx_ring_size = 16ull * 1024 * 1024;
        uint64_t rx_ring_size = 16ull * 1024 * 1024;

        // UHD recv() request size in samples. 0 = use rx_stream->get_max_num_samps().
        int recv_chunk = 0;
    };

    // The USRP must already be opened with subdev/master-clock/time/PPS set
    // and must remain valid for the whole lifetime of this server. The
    // server creates its own streamers against the reserved channels when
    // the OAI side issues CMD_INIT; it does NOT call multi_usrp::make().
    TenantServer(uhd::usrp::multi_usrp::sptr usrp, Config cfg);
    ~TenantServer();  // calls stop() idempotently

    TenantServer(const TenantServer&) = delete;
    TenantServer& operator=(const TenantServer&) = delete;

    // Creates the shm segment and spawns the control thread (which in turn
    // will spawn TX/RX/telemetry threads on the first CMD_INIT). Returns
    // immediately; does not wait for an OAI tenant to connect.
    void start();

    // Stops streaming, joins all threads, releases streamers (but NOT the
    // multi_usrp handle), unlinks the shm segment. Idempotent.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace usrpproxy

#endif  // USRPPROXY_TENANT_SERVER_H
