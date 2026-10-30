/*
 * SPDX-License-Identifier: Apache-2.0 */

#include "usrpproxy_mvp/shared_ring.h"
#include <boost/test/unit_test.hpp>
#include <new>
#include <vector>

BOOST_AUTO_TEST_CASE(test_usrpproxy_tx_header_validation)
{
    shm_tx_hdr_t hdr{};
    hdr.nsamps = 1024;
    hdr.nb_ant = 2;
    hdr.payload_bytes = 1024 * 2 * 4;

    BOOST_CHECK(usrpproxy_tx_hdr_valid(&hdr, 2, 16384));

    hdr.nb_ant = 3;
    hdr.payload_bytes = 1024 * 3 * 4;
    BOOST_CHECK(!usrpproxy_tx_hdr_valid(&hdr, 2, 16384));
    BOOST_CHECK(usrpproxy_tx_packet_shape_valid(&hdr, 16384));

    hdr.nb_ant = 2;
    hdr.payload_bytes = 1024 * 4;
    BOOST_CHECK(!usrpproxy_tx_hdr_valid(&hdr, 2, 16384));
    BOOST_CHECK(!usrpproxy_tx_packet_shape_valid(&hdr, 16384));

    hdr.payload_bytes = 1024 * 2 * 4;
    BOOST_CHECK(!usrpproxy_tx_hdr_valid(&hdr, 0, 16384));
    BOOST_CHECK(!usrpproxy_tx_hdr_valid(&hdr, USRPPROXY_MAX_CH + 1, 16384));
    BOOST_CHECK(!usrpproxy_tx_hdr_valid(&hdr, 2, sizeof(hdr)));

    hdr.nsamps = UINT32_MAX;
    hdr.nb_ant = 2;
    hdr.payload_bytes = UINT32_MAX;
    BOOST_CHECK(!usrpproxy_tx_hdr_valid(&hdr, 2, UINT64_MAX));
    BOOST_CHECK(!usrpproxy_tx_packet_shape_valid(&hdr, UINT64_MAX));
}

BOOST_AUTO_TEST_CASE(test_usrpproxy_rx_overflow_policy_does_not_consume_tail)
{
    constexpr uint64_t cap = 64;
    std::vector<uint8_t> storage(sizeof(usrpproxy_ring_t) + cap);
    auto* ring = new (storage.data()) usrpproxy_ring_t{};
    ring->head.store(0, std::memory_order_relaxed);
    ring->tail.store(0, std::memory_order_relaxed);
    ring->capacity = cap;

    uint8_t hdr[8] = {};
    uint8_t payload[48] = {};
    BOOST_REQUIRE(ring_write2_drop_new(ring, hdr, sizeof(hdr),
                                       payload, sizeof(payload)));

    const uint64_t tail_before = ring->tail.load(std::memory_order_relaxed);
    const uint64_t readable_before = ring_readable(ring);

    uint8_t payload2[16] = {};
    BOOST_CHECK(!ring_write2_drop_new(ring, hdr, sizeof(hdr),
                                      payload2, sizeof(payload2)));
    BOOST_CHECK_EQUAL(ring->tail.load(std::memory_order_relaxed), tail_before);
    BOOST_CHECK_EQUAL(ring_readable(ring), readable_before);
}

BOOST_AUTO_TEST_CASE(test_usrpproxy_segment_sizing_rejects_invalid_caps)
{
    uint64_t total = 0;
    BOOST_CHECK(usrpproxy_segment_bytes_checked(16ull * 1024 * 1024,
                                                16ull * 1024 * 1024,
                                                &total));
    BOOST_CHECK_GT(total, 0u);

    BOOST_CHECK(!usrpproxy_segment_bytes_checked(0, 4096, &total));
    BOOST_CHECK(!usrpproxy_segment_bytes_checked(4097, 4096, &total));
    BOOST_CHECK(!usrpproxy_segment_bytes_checked(4096,
                                                 USRPPROXY_MAX_RING_BYTES * 2,
                                                 &total));
    BOOST_CHECK(!usrpproxy_segment_bytes_checked(UINT64_MAX,
                                                 4096, &total));
}

BOOST_AUTO_TEST_CASE(test_usrpproxy_channel_count_validation)
{
    BOOST_CHECK(usrpproxy_channel_count_valid(1));
    BOOST_CHECK(usrpproxy_channel_count_valid(USRPPROXY_MAX_CH));
    BOOST_CHECK(!usrpproxy_channel_count_valid(0));
    BOOST_CHECK(!usrpproxy_channel_count_valid(-1));
    BOOST_CHECK(!usrpproxy_channel_count_valid(USRPPROXY_MAX_CH + 1));
}

BOOST_AUTO_TEST_CASE(test_usrpproxy_ring_peek_at_rejects_offset_overflow)
{
    constexpr uint64_t cap = 64;
    std::vector<uint8_t> storage(sizeof(usrpproxy_ring_t) + cap);
    auto* ring = new (storage.data()) usrpproxy_ring_t{};
    ring->head.store(32, std::memory_order_relaxed);
    ring->tail.store(0, std::memory_order_relaxed);
    ring->capacity = cap;

    uint8_t out[2] = {}; // two bytes: the rejected 2-byte peek must not look out of bounds to GCC
    BOOST_CHECK(!ring_peek_at(ring, UINT64_MAX, out, 1));
    BOOST_CHECK(!ring_peek_at(ring, 31, out, 2));
    BOOST_CHECK(ring_peek_at(ring, 31, out, 1));
}
