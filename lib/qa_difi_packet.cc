/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for the DIFI (IEEE-ISTO 4900) packetizer.
 */

#define BOOST_TEST_MODULE qa_difi_packet
#include <boost/test/unit_test.hpp>

#include <cusat_dsp/difi_packet.h>

#include <cmath>
#include <complex>
#include <vector>

using namespace cusat_dsp::difi;

BOOST_AUTO_TEST_CASE(test_data_roundtrip)
{
    std::vector<std::complex<float>> iq(DATA_SAMPLES_PER_PACKET);
    for (int i = 0; i < DATA_SAMPLES_PER_PACKET; i++)
        iq[i] = { std::sin(0.01f * i) * 0.8f, std::cos(0.02f * i) * 0.8f };

    uint8_t pkt[DATA_PACKET_BYTES];
    size_t len = pack_data(pkt, iq.data(), 0xC0FFEE, 5, 1751400000,
                           123456789012ull);
    BOOST_CHECK_EQUAL(len, DATA_PACKET_BYTES);

    parsed_pkt p;
    BOOST_REQUIRE(parse(pkt, len, p));
    BOOST_CHECK(p.type == pkt_type::data);
    BOOST_CHECK_EQUAL(p.stream_id, 0xC0FFEEu);
    BOOST_CHECK_EQUAL(p.seq4, 5);
    BOOST_CHECK_EQUAL(p.ts_int, 1751400000u);
    BOOST_CHECK_EQUAL(p.ts_frac_ps, 123456789012ull);
    BOOST_REQUIRE_EQUAL(p.n_samples, DATA_SAMPLES_PER_PACKET);

    std::vector<std::complex<float>> out(p.n_samples);
    payload_to_fc32(p.payload, p.n_samples, out.data());
    for (int i = 0; i < p.n_samples; i++) {
        BOOST_CHECK_SMALL(out[i].real() - iq[i].real(), 1.0f / 32000.0f);
        BOOST_CHECK_SMALL(out[i].imag() - iq[i].imag(), 1.0f / 32000.0f);
    }
}

BOOST_AUTO_TEST_CASE(test_clipping)
{
    std::vector<std::complex<float>> iq(DATA_SAMPLES_PER_PACKET,
                                        { 2.0f, -2.0f });
    uint8_t pkt[DATA_PACKET_BYTES];
    pack_data(pkt, iq.data(), 1, 0, 0, 0);
    parsed_pkt p;
    BOOST_REQUIRE(parse(pkt, DATA_PACKET_BYTES, p));
    std::vector<std::complex<float>> out(p.n_samples);
    payload_to_fc32(p.payload, p.n_samples, out.data());
    BOOST_CHECK_CLOSE(out[0].real(), 1.0f, 0.1f);
    BOOST_CHECK_CLOSE(out[0].imag(), -32768.0f / 32767.0f, 0.1f);
}

BOOST_AUTO_TEST_CASE(test_context_parse_skip)
{
    uint8_t pkt[128];
    size_t len = pack_context(pkt, 7, 3, 100, 0, 122.88e6, 2.05e9, 36e6);
    BOOST_CHECK_EQUAL(len, CONTEXT_PACKET_BYTES); // 27 words per DIFI v1.1
    BOOST_REQUIRE(len <= sizeof(pkt));
    parsed_pkt p;
    BOOST_REQUIRE(parse(pkt, len, p));
    BOOST_CHECK(p.type == pkt_type::context);
    BOOST_CHECK_EQUAL(p.stream_id, 7u);
    BOOST_CHECK_EQUAL(p.n_samples, 0);
}

BOOST_AUTO_TEST_CASE(test_malformed_rejected)
{
    uint8_t junk[16] = { 0 };
    parsed_pkt p;
    BOOST_CHECK(!parse(junk, 8, p));           // too short
    BOOST_CHECK(!parse(junk, sizeof(junk), p)); // no class id bit
    // Truncated: header claims more words than the buffer holds.
    std::vector<std::complex<float>> iq(DATA_SAMPLES_PER_PACKET, { 0, 0 });
    uint8_t pkt[DATA_PACKET_BYTES];
    pack_data(pkt, iq.data(), 1, 0, 0, 0);
    BOOST_CHECK(!parse(pkt, DATA_PACKET_BYTES / 2, p));
}

BOOST_AUTO_TEST_CASE(test_seq_wraps_mod16)
{
    std::vector<std::complex<float>> iq(DATA_SAMPLES_PER_PACKET, { 0, 0 });
    uint8_t pkt[DATA_PACKET_BYTES];
    pack_data(pkt, iq.data(), 1, (uint8_t)(15 + 3), 0, 0); // 18 -> 2
    parsed_pkt p;
    BOOST_REQUIRE(parse(pkt, DATA_PACKET_BYTES, p));
    BOOST_CHECK_EQUAL(p.seq4, 2);
}
