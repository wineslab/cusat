/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cusat_dsp/gse.h"
#include <boost/test/unit_test.hpp>
#include <cstring>
#include <vector>

namespace cusat_dsp {

BOOST_AUTO_TEST_CASE(test_gse_crc32)
{
    // Known test vector: ASCII "123456789" -> 0xCBF43926
    const uint8_t data[] = { '1', '2', '3', '4', '5', '6', '7', '8', '9' };
    BOOST_CHECK_EQUAL(gse_calc_crc32(data, 9), 0xCBF43926u);

    // Empty data
    BOOST_CHECK_EQUAL(gse_calc_crc32(nullptr, 0), 0x00000000u);

    // Single byte
    const uint8_t one = 0x00;
    uint32_t crc_zero = gse_calc_crc32(&one, 1);
    BOOST_CHECK_NE(crc_zero, 0u); // CRC of 0x00 should not be 0
}

BOOST_AUTO_TEST_CASE(test_gse_complete_pdu_broadcast)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    uint8_t pdu[] = { 0x01, 0x02, 0x03, 0x04, 0x05 };
    uint8_t buf[256];

    int written = enc.encapsulate(pdu, sizeof(pdu), buf, sizeof(buf));
    // header(2) + proto(2) + pdu(5) = 9
    BOOST_CHECK_EQUAL(written, 9);
    BOOST_CHECK(!enc.has_pending());

    bool received = false;
    dec.decapsulate(buf, written,
                    [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                        const uint8_t* label, int label_len) {
                        BOOST_CHECK_EQUAL(rx_len, (int)sizeof(pdu));
                        BOOST_CHECK_EQUAL(proto, 0x0800);
                        BOOST_CHECK_EQUAL(label_len, 0);
                        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu, sizeof(pdu)), 0);
                        received = true;
                    });
    BOOST_CHECK(received);
    BOOST_CHECK_EQUAL(dec.pdu_count(), 1u);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gse_complete_pdu_6byte_label)
{
    uint8_t label[] = { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF };
    gse_encapsulator enc(0x86DD, GSE_LT_6BYTE, label);
    gse_decapsulator dec;

    uint8_t pdu[] = { 0xDE, 0xAD, 0xBE, 0xEF };
    uint8_t buf[256];

    int written = enc.encapsulate(pdu, sizeof(pdu), buf, sizeof(buf));
    // header(2) + proto(2) + label(6) + pdu(4) = 14
    BOOST_CHECK_EQUAL(written, 14);

    bool received = false;
    dec.decapsulate(buf, written,
                    [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                        const uint8_t* rx_label, int label_len) {
                        BOOST_CHECK_EQUAL(rx_len, (int)sizeof(pdu));
                        BOOST_CHECK_EQUAL(proto, 0x86DD);
                        BOOST_CHECK_EQUAL(label_len, 6);
                        BOOST_CHECK_EQUAL(memcmp(rx_label, label, 6), 0);
                        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu, sizeof(pdu)), 0);
                        received = true;
                    });
    BOOST_CHECK(received);
}

BOOST_AUTO_TEST_CASE(test_gse_complete_pdu_3byte_label)
{
    uint8_t label[] = { 0x01, 0x02, 0x03 };
    gse_encapsulator enc(0x0800, GSE_LT_3BYTE, label);
    gse_decapsulator dec;

    uint8_t pdu[] = { 0x42 };
    uint8_t buf[256];

    int written = enc.encapsulate(pdu, sizeof(pdu), buf, sizeof(buf));
    // header(2) + proto(2) + label(3) + pdu(1) = 8
    BOOST_CHECK_EQUAL(written, 8);

    bool received = false;
    dec.decapsulate(buf, written,
                    [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                        const uint8_t* rx_label, int label_len) {
                        BOOST_CHECK_EQUAL(rx_len, 1);
                        BOOST_CHECK_EQUAL(label_len, 3);
                        BOOST_CHECK_EQUAL(memcmp(rx_label, label, 3), 0);
                        BOOST_CHECK_EQUAL(rx_pdu[0], 0x42);
                        received = true;
                    });
    BOOST_CHECK(received);
}

BOOST_AUTO_TEST_CASE(test_gse_fragmented_2_fragments)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    // 100-byte PDU, force into 2 fragments
    std::vector<uint8_t> pdu(100);
    for (int i = 0; i < 100; i++) pdu[i] = i;

    // First fragment: 60 bytes available
    // overhead = header(2) + frag_id(1) + total_len(2) + proto(2) = 7
    // payload = 60 - 7 = 53 bytes
    uint8_t buf1[60];
    int w1 = enc.encapsulate(pdu.data(), pdu.size(), buf1, sizeof(buf1));
    BOOST_CHECK_GT(w1, 0);
    BOOST_CHECK_EQUAL(w1, 60);
    BOOST_CHECK(enc.has_pending());

    // Last fragment: plenty of space
    // remaining = 100 - 53 = 47, overhead = header(2) + frag_id(1) + crc(4) = 7
    // total = 47 + 7 = 54
    uint8_t buf2[256];
    int w2 = enc.write_pending(buf2, sizeof(buf2));
    BOOST_CHECK_GT(w2, 0);
    BOOST_CHECK_EQUAL(w2, 54);
    BOOST_CHECK(!enc.has_pending());

    // Decapsulate
    bool received = false;
    auto cb = [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                  const uint8_t*, int) {
        BOOST_CHECK_EQUAL(rx_len, 100);
        BOOST_CHECK_EQUAL(proto, 0x0800);
        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu.data(), 100), 0);
        received = true;
    };

    dec.decapsulate(buf1, w1, cb);
    BOOST_CHECK(!received);
    dec.decapsulate(buf2, w2, cb);
    BOOST_CHECK(received);
    BOOST_CHECK_EQUAL(dec.pdu_count(), 1u);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gse_fragmented_3_fragments)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    std::vector<uint8_t> pdu(100);
    for (int i = 0; i < 100; i++) pdu[i] = i;

    // Small buffers to force 3+ fragments
    uint8_t buf1[30], buf2[30], buf3[256];

    int w1 = enc.encapsulate(pdu.data(), pdu.size(), buf1, sizeof(buf1));
    BOOST_CHECK_GT(w1, 0);
    BOOST_CHECK(enc.has_pending());

    int w2 = enc.write_pending(buf2, sizeof(buf2));
    BOOST_CHECK_GT(w2, 0);
    BOOST_CHECK(enc.has_pending());

    int w3 = enc.write_pending(buf3, sizeof(buf3));
    BOOST_CHECK_GT(w3, 0);
    BOOST_CHECK(!enc.has_pending());

    bool received = false;
    auto cb = [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                  const uint8_t*, int) {
        BOOST_CHECK_EQUAL(rx_len, 100);
        BOOST_CHECK_EQUAL(proto, 0x0800);
        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu.data(), 100), 0);
        received = true;
    };

    dec.decapsulate(buf1, w1, cb);
    dec.decapsulate(buf2, w2, cb);
    dec.decapsulate(buf3, w3, cb);
    BOOST_CHECK(received);
    BOOST_CHECK_EQUAL(dec.pdu_count(), 1u);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gse_fragment_reassembly_rejects_payload_overrun)
{
    gse_decapsulator dec;
    int received = 0;
    auto cb = [&](const uint8_t*, int, uint16_t, const uint8_t*, int) {
        received++;
    };

    auto write_hdr = [](uint8_t* out, bool start, bool end, int lt, uint16_t len) {
        uint16_t hdr = 0;
        if (start) hdr |= 1u << 15;
        if (end) hdr |= 1u << 14;
        hdr |= (uint16_t)(lt & 0x3) << 12;
        hdr |= len & 0x0FFF;
        out[0] = (hdr >> 8) & 0xFF;
        out[1] = hdr & 0xFF;
    };

    // First fragment claims total payload is 5 bytes:
    // total_length = protocol_type(2) + payload(5).
    uint8_t first[12] = {};
    write_hdr(first, true, false, GSE_LT_BROADCAST, 10);
    first[2] = 7;        // Frag ID
    first[3] = 0x00;     // Total length = 7
    first[4] = 0x07;
    first[5] = 0x08;     // Protocol type 0x0800
    first[6] = 0x00;
    for (int i = 0; i < 5; i++)
        first[7 + i] = (uint8_t)i;

    dec.decapsulate(first, sizeof(first), cb);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
    BOOST_CHECK_EQUAL(received, 0);

    // Additional intermediate payload would exceed total_length and must drop
    // the fragment context instead of growing the reassembly vector.
    uint8_t overrun[4] = {};
    write_hdr(overrun, false, false, GSE_LT_BROADCAST, 2);
    overrun[2] = 7;
    overrun[3] = 0xFF;

    dec.decapsulate(overrun, sizeof(overrun), cb);
    BOOST_CHECK_EQUAL(dec.error_count(), 1u);
    BOOST_CHECK_EQUAL(received, 0);
}

BOOST_AUTO_TEST_CASE(test_gse_encapsulator_rejects_oversized_pdu)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    std::vector<uint8_t> pdu(65534);
    std::vector<uint8_t> out(4096);

    BOOST_CHECK(!enc.pdu_size_valid((int)pdu.size()));
    BOOST_CHECK_EQUAL(enc.encapsulate(pdu.data(), (int)pdu.size(),
                                      out.data(), (int)out.size()), -1);
    BOOST_CHECK(!enc.has_pending());

    pdu.resize(65533);
    BOOST_CHECK(enc.pdu_size_valid((int)pdu.size()));
}

BOOST_AUTO_TEST_CASE(test_gse_multiple_pdus_one_buffer)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    uint8_t pdu1[] = { 0xAA, 0xBB };
    uint8_t pdu2[] = { 0xCC, 0xDD, 0xEE };
    uint8_t buf[256];
    int pos = 0;

    pos += enc.encapsulate(pdu1, sizeof(pdu1), buf + pos, sizeof(buf) - pos);
    pos += enc.encapsulate(pdu2, sizeof(pdu2), buf + pos, sizeof(buf) - pos);

    int count = 0;
    dec.decapsulate(buf, pos,
                    [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                        const uint8_t*, int) {
                        if (count == 0) {
                            BOOST_CHECK_EQUAL(rx_len, (int)sizeof(pdu1));
                            BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu1, sizeof(pdu1)), 0);
                        } else {
                            BOOST_CHECK_EQUAL(rx_len, (int)sizeof(pdu2));
                            BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu2, sizeof(pdu2)), 0);
                        }
                        BOOST_CHECK_EQUAL(proto, 0x0800);
                        count++;
                    });
    BOOST_CHECK_EQUAL(count, 2);
}

BOOST_AUTO_TEST_CASE(test_gse_padding)
{
    uint8_t buf[32];
    int written = gse_encapsulator::write_padding(buf, sizeof(buf));
    BOOST_CHECK_EQUAL(written, (int)sizeof(buf));

    // All bytes should be zero (GSE_LENGTH=0 header + zero padding)
    for (int i = 0; i < 32; i++) {
        BOOST_CHECK_EQUAL(buf[i], 0);
    }

    // Decapsulator should handle padding gracefully (no PDUs)
    gse_decapsulator dec;
    int count = 0;
    dec.decapsulate(buf, written,
                    [&](const uint8_t*, int, uint16_t, const uint8_t*, int) { count++; });
    BOOST_CHECK_EQUAL(count, 0);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gse_padding_min_size)
{
    // Minimum padding is 2 bytes
    uint8_t buf[2];
    BOOST_CHECK_EQUAL(gse_encapsulator::write_padding(buf, 2), 2);
    BOOST_CHECK_EQUAL(buf[0], 0);
    BOOST_CHECK_EQUAL(buf[1], 0);

    // Less than 2 bytes: can't write padding
    BOOST_CHECK_EQUAL(gse_encapsulator::write_padding(buf, 1), 0);
    BOOST_CHECK_EQUAL(gse_encapsulator::write_padding(buf, 0), 0);
}

BOOST_AUTO_TEST_CASE(test_gse_pdu_then_padding)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    uint8_t pdu[] = { 0x01, 0x02 };
    uint8_t buf[64];
    int pos = 0;

    pos += enc.encapsulate(pdu, sizeof(pdu), buf + pos, sizeof(buf) - pos);
    pos += gse_encapsulator::write_padding(buf + pos, sizeof(buf) - pos);
    BOOST_CHECK_EQUAL(pos, (int)sizeof(buf));

    int count = 0;
    dec.decapsulate(buf, pos,
                    [&](const uint8_t* rx_pdu, int rx_len, uint16_t, const uint8_t*, int) {
                        BOOST_CHECK_EQUAL(rx_len, (int)sizeof(pdu));
                        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu, sizeof(pdu)), 0);
                        count++;
                    });
    BOOST_CHECK_EQUAL(count, 1);
}

BOOST_AUTO_TEST_CASE(test_gse_fragment_crc_validation)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    std::vector<uint8_t> pdu(200);
    for (int i = 0; i < 200; i++) pdu[i] = i & 0xFF;

    uint8_t buf1[50], buf2[256];
    int w1 = enc.encapsulate(pdu.data(), pdu.size(), buf1, sizeof(buf1));
    BOOST_CHECK(enc.has_pending());

    int w2 = enc.write_pending(buf2, sizeof(buf2));
    BOOST_CHECK(!enc.has_pending());

    // Corrupt a payload byte in the last fragment
    if (w2 > 4) buf2[3] ^= 0xFF;

    bool received = false;
    auto cb = [&](const uint8_t*, int, uint16_t, const uint8_t*, int) {
        received = true;
    };

    dec.decapsulate(buf1, w1, cb);
    dec.decapsulate(buf2, w2, cb);
    BOOST_CHECK(!received);
    BOOST_CHECK_EQUAL(dec.pdu_count(), 0u);
    BOOST_CHECK_EQUAL(dec.error_count(), 1u);
}

BOOST_AUTO_TEST_CASE(test_gse_large_pdu_roundtrip)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    // Realistic Ethernet MTU
    std::vector<uint8_t> pdu(1500);
    for (int i = 0; i < 1500; i++) pdu[i] = (i * 7) & 0xFF;

    // Simulate BBFRAME data fields of ~200 bytes
    std::vector<std::vector<uint8_t>> fragments;
    uint8_t buf[200];

    int w = enc.encapsulate(pdu.data(), pdu.size(), buf, sizeof(buf));
    fragments.push_back(std::vector<uint8_t>(buf, buf + w));

    while (enc.has_pending()) {
        w = enc.write_pending(buf, sizeof(buf));
        BOOST_CHECK_GT(w, 0);
        fragments.push_back(std::vector<uint8_t>(buf, buf + w));
    }

    BOOST_CHECK_GT(fragments.size(), 1u);
    BOOST_TEST_MESSAGE("1500B PDU fragmented into " << fragments.size() << " fragments");

    bool received = false;
    auto cb = [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                  const uint8_t*, int) {
        BOOST_CHECK_EQUAL(rx_len, 1500);
        BOOST_CHECK_EQUAL(proto, 0x0800);
        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu.data(), 1500), 0);
        received = true;
    };

    for (auto& frag : fragments) {
        dec.decapsulate(frag.data(), frag.size(), cb);
    }

    BOOST_CHECK(received);
    BOOST_CHECK_EQUAL(dec.pdu_count(), 1u);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gse_jumbo_pdu_large_frames)
{
    // Regression: gse_length is a 12-bit field (max 0xFFF). With baseband data
    // fields larger than 4095 B (normal FECFRAMEs at high modcods), a jumbo PDU
    // must still be split so each GSE packet's gse_length fits. Previously the
    // encapsulator sized fragments to the whole frame and overflowed the field,
    // so the receiver dropped everything (UDP goodput = 0 at MTU 9000).
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);
    gse_decapsulator dec;

    std::vector<uint8_t> pdu(9000);
    for (int i = 0; i < 9000; i++) pdu[i] = (i * 13 + 5) & 0xFF;

    // ~6000-byte data fields, like a 32apsk3/4 normal FECFRAME.
    std::vector<std::vector<uint8_t>> packets;
    uint8_t buf[6000];

    int w = enc.encapsulate(pdu.data(), pdu.size(), buf, sizeof(buf));
    BOOST_CHECK_GT(w, 0);
    packets.push_back(std::vector<uint8_t>(buf, buf + w));
    while (enc.has_pending()) {
        w = enc.write_pending(buf, sizeof(buf));
        BOOST_CHECK_GT(w, 0);
        packets.push_back(std::vector<uint8_t>(buf, buf + w));
    }

    // Every emitted GSE packet must honor the 12-bit gse_length field.
    for (auto& p : packets) {
        uint16_t gse_length = ((uint16_t)p[0] << 8 | p[1]) & 0xFFF;
        BOOST_CHECK_LE((int)gse_length, 0xFFF);
        BOOST_CHECK_EQUAL((int)p.size(), 2 + (int)gse_length);
    }
    BOOST_CHECK_GT(packets.size(), 2u); // 9000B over 6000B frames -> >=3 packets

    bool received = false;
    auto cb = [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                  const uint8_t*, int) {
        BOOST_CHECK_EQUAL(rx_len, 9000);
        BOOST_CHECK_EQUAL(proto, 0x0800);
        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu.data(), 9000), 0);
        received = true;
    };
    for (auto& p : packets)
        dec.decapsulate(p.data(), p.size(), cb);

    BOOST_CHECK(received);
    BOOST_CHECK_EQUAL(dec.pdu_count(), 1u);
    BOOST_CHECK_EQUAL(dec.error_count(), 0u);
}

BOOST_AUTO_TEST_CASE(test_gse_encapsulate_insufficient_space)
{
    gse_encapsulator enc(0x0800, GSE_LT_BROADCAST);

    uint8_t pdu[] = { 0x01, 0x02, 0x03 };
    uint8_t buf[4]; // Too small for any GSE packet (min complete = 7, min first frag = 8)

    int written = enc.encapsulate(pdu, sizeof(pdu), buf, sizeof(buf));
    BOOST_CHECK_EQUAL(written, 0);
    BOOST_CHECK(!enc.has_pending());
}

BOOST_AUTO_TEST_CASE(test_gse_fragmented_with_label)
{
    uint8_t label[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06 };
    gse_encapsulator enc(0x0800, GSE_LT_6BYTE, label);
    gse_decapsulator dec;

    std::vector<uint8_t> pdu(80);
    for (int i = 0; i < 80; i++) pdu[i] = i;

    // First fragment overhead with 6-byte label:
    // header(2) + frag_id(1) + total_len(2) + proto(2) + label(6) = 13
    uint8_t buf1[40], buf2[256];

    int w1 = enc.encapsulate(pdu.data(), pdu.size(), buf1, sizeof(buf1));
    BOOST_CHECK_GT(w1, 0);
    BOOST_CHECK(enc.has_pending());

    int w2 = enc.write_pending(buf2, sizeof(buf2));
    BOOST_CHECK(!enc.has_pending());

    bool received = false;
    auto cb = [&](const uint8_t* rx_pdu, int rx_len, uint16_t proto,
                  const uint8_t* rx_label, int label_len) {
        BOOST_CHECK_EQUAL(rx_len, 80);
        BOOST_CHECK_EQUAL(proto, 0x0800);
        BOOST_CHECK_EQUAL(label_len, 6);
        BOOST_CHECK_EQUAL(memcmp(rx_label, label, 6), 0);
        BOOST_CHECK_EQUAL(memcmp(rx_pdu, pdu.data(), 80), 0);
        received = true;
    };

    dec.decapsulate(buf1, w1, cb);
    dec.decapsulate(buf2, w2, cb);
    BOOST_CHECK(received);
}

} // namespace cusat_dsp
