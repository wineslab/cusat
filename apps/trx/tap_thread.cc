/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TAP reader thread for dvbs2-uhd-trx.
 */

#include "trx/tap_thread.h"

#include "trx/acm_signaling.h"
#include "trx/thread_util.h"

#include <sys/select.h>

#include <cstdint>
#include <vector>

void tap_reader_thread(trx_context& ctx)
{
    auto& opt = *ctx.opt;
    auto& tap = *ctx.tap;
    auto& tap_pdus = *ctx.tap_pdus;

    if (!tap) return;
    if (opt.pin_threads) pin_thread("tap_rd", 4, opt.realtime_threads);
    // Sized to hold a full Ethernet frame at a jumbo TAP MTU (up to ~16 KB);
    // a large MTU cuts per-packet GSE overhead so UDP goodput tracks the PHY.
    std::vector<uint8_t> buf(16384);
    while (!g_stop.load(std::memory_order_relaxed)) {
        fd_set fds;
        struct timeval tv = {0, 10000}; // 10ms timeout
        FD_ZERO(&fds);
        FD_SET(tap->fd(), &fds);
        int sel = select(tap->fd() + 1, &fds, nullptr, nullptr, &tv);
        if (sel <= 0) continue;
        int len = tap->read_frame(buf.data(), buf.size());
        if (len <= ETH_HLEN) continue;
        uint16_t etype = ((uint16_t)buf[12] << 8) | buf[13];
        if (etype != ETH_P_IP) continue;
        if (!tap_pdus.push(buf.data() + ETH_HLEN, len - ETH_HLEN))
            ctx.stats.tx_pdu_queue_drops.fetch_add(
                1, std::memory_order_relaxed);
    }
}
