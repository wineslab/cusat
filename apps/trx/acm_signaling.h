/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Ethernet constants and in-band ACM signaling definitions for dvbs2-uhd-trx.
 */

#pragma once

#include <cstdint>

/* ---------- Ethernet L2/L3 helpers ---------- */
inline constexpr int ETH_HLEN = 14;
inline constexpr uint16_t ETH_P_IP = 0x0800;
// Synthetic peer MAC (locally-administered unicast) used as src in
// reconstructed Ethernet frames on the RX path.
inline constexpr uint8_t PEER_MAC[6] = {0x02, 0x00, 0x5a, 0x54, 0x00, 0x01};

/* ---------- ACM feedback packet ---------- */

struct acm_feedback_pkt_t {
    uint32_t magic;           // 0xACFEED01
    float    fer;             // EWMA FER from our receiver
    float    snr_db;          // Pilot Es/N0 measured by our receiver
    uint16_t rx_modcod;       // MODCOD being received
    uint16_t seq;
    uint64_t fec_frames;      // Cumulative outcomes for rx_modcod's data chain
    uint64_t fec_errors;      // Cumulative failures for rx_modcod's data chain
};
static_assert(sizeof(acm_feedback_pkt_t) == 32,
              "ACM feedback wire format must remain 32 bytes");
inline constexpr uint32_t ACM_FEEDBACK_MAGIC = 0xACFEED01;
inline constexpr uint16_t ACM_SIG_PROTO = 0xACFB; // GSE protocol type for in-band ACM signaling
inline constexpr uint8_t SIG_MODCOD = 1;           // QPSK 1/4 — most robust MODCOD
