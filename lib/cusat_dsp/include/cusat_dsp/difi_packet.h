/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * DIFI (IEEE-ISTO 4900-2021) packetizer / depacketizer.
 *
 * Implements the VITA 49.2 subset DIFI uses for IQ-over-IP: Standard Flow
 * Signal Data packets (type 1, class ID present, TSI=UTC, TSF=real-time
 * picoseconds, complex int16 cartesian payload) and a minimal Standard Flow
 * Context packet (type 4: CIF0 with bandwidth / RF frequency / sample rate /
 * data format). All fields are big-endian on the wire per VITA 49.
 *
 * Interop status: the TX output (data + context packets) passes the DIFI
 * Consortium's own v1.1 packet validators (DIFI-Consortium/
 * DIFI-Certification, packet_definitions/difi_{data,context}_v1_1.py) —
 * see scripts/difi_certify_stream.py for the live-stream check. The RX
 * parser is validated against our own TX only so far.
 */

#pragma once

#include <complex>
#include <cstdint>
#include <cstring>
#include <endian.h>

namespace cusat_dsp {
namespace difi {

/* DIFI Consortium class identifier (confirmed against the consortium's
 * certification checkers: OUI 0x6A621E, data class 0x0000/0x0000,
 * context class 0x0000/0x0001). */
constexpr uint32_t DIFI_OUI = 0x6A621E;
constexpr uint16_t INFO_CLASS_CODE = 0x0000;
constexpr uint16_t PKT_CLASS_DATA = 0x0000;    // signal data
constexpr uint16_t PKT_CLASS_CONTEXT = 0x0001; // standard context

/* Header words before payload: hdr, stream id, class id x2, ts int,
 * ts frac x2. */
constexpr int PRELUDE_WORDS = 7;
constexpr size_t PRELUDE_BYTES = PRELUDE_WORDS * 4;

/* 360 complex int16 samples -> 1468-byte UDP payload, under a 1500 MTU. */
constexpr int DATA_SAMPLES_PER_PACKET = 360;
constexpr size_t DATA_PACKET_BYTES =
    PRELUDE_BYTES + (size_t)DATA_SAMPLES_PER_PACKET * 4;

enum class pkt_type : uint8_t { data = 0x1, context = 0x4, unknown = 0xF };

static inline void put_u32(uint8_t* p, uint32_t v)
{
    uint32_t be = htobe32(v);
    memcpy(p, &be, 4);
}

static inline uint32_t get_u32(const uint8_t* p)
{
    uint32_t be;
    memcpy(&be, p, 4);
    return be32toh(be);
}

static inline void put_u64(uint8_t* p, uint64_t v)
{
    uint64_t be = htobe64(v);
    memcpy(p, &be, 8);
}

/* Write the 7-word prelude shared by data and context packets.
 * ts_int = UTC seconds, ts_frac_ps = picoseconds within the second.
 * tsm must be true for context packets, false for data (DIFI v1.1). */
static inline void write_prelude(uint8_t* out, pkt_type type,
                                 uint16_t pkt_class, uint32_t stream_id,
                                 uint8_t seq4, uint16_t size_words,
                                 uint32_t ts_int, uint64_t ts_frac_ps,
                                 bool tsm = false)
{
    /* [31:28] type, [27] C=1 class id, [26:25] reserved=0, [24] TSM,
     * [23:22] TSI=01 UTC, [21:20] TSF=10 real-time ps,
     * [19:16] packet count mod 16, [15:0] size in 32-bit words. */
    uint32_t hdr = ((uint32_t)type << 28) | (1u << 27) |
                   ((tsm ? 1u : 0u) << 24) | (0x1u << 22) |
                   (0x2u << 20) | ((uint32_t)(seq4 & 0xF) << 16) | size_words;
    put_u32(out + 0, hdr);
    put_u32(out + 4, stream_id);
    put_u32(out + 8, DIFI_OUI); /* [31:24] pad-bit count 0, [23:0] OUI */
    put_u32(out + 12, ((uint32_t)INFO_CLASS_CODE << 16) | pkt_class);
    put_u32(out + 16, ts_int);
    put_u64(out + 20, ts_frac_ps);
}

/**
 * Pack one signal-data packet: fc32 -> complex int16 big-endian.
 * iq must hold DATA_SAMPLES_PER_PACKET samples; out must hold
 * DATA_PACKET_BYTES. fc32 full scale (+-1.0) maps to +-32767.
 * Returns the packet size in bytes.
 */
static inline size_t pack_data(uint8_t* out, const std::complex<float>* iq,
                               uint32_t stream_id, uint8_t seq4,
                               uint32_t ts_int, uint64_t ts_frac_ps)
{
    const uint16_t words = PRELUDE_WORDS + DATA_SAMPLES_PER_PACKET;
    write_prelude(out, pkt_type::data, PKT_CLASS_DATA, stream_id, seq4,
                  words, ts_int, ts_frac_ps);
    uint8_t* p = out + PRELUDE_BYTES;
    for (int i = 0; i < DATA_SAMPLES_PER_PACKET; i++) {
        float fi = iq[i].real() * 32767.0f;
        float fq = iq[i].imag() * 32767.0f;
        if (fi > 32767.0f) fi = 32767.0f;
        if (fi < -32768.0f) fi = -32768.0f;
        if (fq > 32767.0f) fq = 32767.0f;
        if (fq < -32768.0f) fq = -32768.0f;
        uint16_t bi = htobe16((uint16_t)(int16_t)fi);
        uint16_t bq = htobe16((uint16_t)(int16_t)fq);
        memcpy(p, &bi, 2);
        memcpy(p + 2, &bq, 2);
        p += 4;
    }
    return DATA_PACKET_BYTES;
}

/* DIFI v1.1 Standard Flow Signal Context packet is a FIXED 27-word layout:
 * CIF0 = 0xFBB98000 (change indicator, reference point, bandwidth, IF ref
 * freq, RF ref freq, IF band offset, reference level, gain, sample rate,
 * timestamp adjustment, timestamp calibration, state/event, data format). */
constexpr uint32_t CONTEXT_CIF0 = 0xFBB98000u;
constexpr uint16_t CONTEXT_WORDS = 27;
constexpr size_t CONTEXT_PACKET_BYTES = (size_t)CONTEXT_WORDS * 4;

/**
 * Pack a DIFI v1.1 Standard Flow Signal Context packet (27 words, TSM=1,
 * validated against the DIFI-Consortium certification checker). Frequencies
 * and rates are VITA 49 64-bit fixed point, radix point at bit 20 (Hz).
 * Fields we do not measure (IF offset, reference level, gain, timestamp
 * adjustment) are reported as 0. Returns the packet size in bytes.
 */
static inline size_t pack_context(uint8_t* out, uint32_t stream_id,
                                  uint8_t seq4, uint32_t ts_int,
                                  uint64_t ts_frac_ps, double sample_rate_hz,
                                  double rf_freq_hz, double bandwidth_hz)
{
    write_prelude(out, pkt_type::context, PKT_CLASS_CONTEXT, stream_id,
                  seq4, CONTEXT_WORDS, ts_int, ts_frac_ps, /*tsm=*/true);
    uint8_t* p = out + PRELUDE_BYTES;
    auto put_word = [&](uint32_t v) { put_u32(p, v); p += 4; };
    auto put_hz = [&](double hz) {
        put_u64(p, (uint64_t)(int64_t)(hz * (double)(1 << 20)));
        p += 8;
    };
    put_word(CONTEXT_CIF0);
    put_word(0x64);          /* reference point: 0x64 per DIFI v1.1 */
    put_hz(bandwidth_hz);
    put_hz(0.0);             /* IF reference frequency */
    put_hz(rf_freq_hz);
    put_hz(0.0);             /* IF band offset */
    put_word(0);             /* reference level (two radix-7 int16 halves) */
    put_word(0);             /* stage 1 / stage 2 gain (radix-7 int16 each) */
    put_hz(sample_rate_hz);
    put_u64(p, 0); p += 8;   /* timestamp adjustment (femtoseconds) */
    put_word(ts_int);        /* timestamp calibration time */
    put_word(0);             /* state and event indicators */
    /* Data payload format (2 words): packing_method[31]=1 link-efficient,
     * real/complex-cartesian [30:29]=01, item format [28:24]=0 (signed
     * fixed point), item packing field size [11:6]=15, item size [5:0]=15
     * (both "one less than 16 bits"). Second word: repeat/vector size 0. */
    put_word((1u << 31) | (0x1u << 29) | (15u << 6) | 15u);
    put_word(0);
    return CONTEXT_PACKET_BYTES;
}

/** Parsed view of a received packet (payload points into the input). */
struct parsed_pkt {
    pkt_type type = pkt_type::unknown;
    uint8_t seq4 = 0;
    uint32_t stream_id = 0;
    uint32_t ts_int = 0;
    uint64_t ts_frac_ps = 0;
    const uint8_t* payload = nullptr; // data packets: complex int16 BE
    int n_samples = 0;                // data packets only
};

/** Parse a packet; returns false if malformed / not a DIFI shape we know. */
static inline bool parse(const uint8_t* pkt, size_t len, parsed_pkt& p)
{
    if (len < PRELUDE_BYTES)
        return false;
    uint32_t hdr = get_u32(pkt);
    uint8_t type = (hdr >> 28) & 0xF;
    bool has_class = (hdr >> 27) & 1;
    uint16_t words = hdr & 0xFFFF;
    if (!has_class || (size_t)words * 4 > len)
        return false;
    p.seq4 = (hdr >> 16) & 0xF;
    p.stream_id = get_u32(pkt + 4);
    p.ts_int = get_u32(pkt + 16);
    uint64_t hi = get_u32(pkt + 20), lo = get_u32(pkt + 24);
    p.ts_frac_ps = (hi << 32) | lo;
    if (type == (uint8_t)pkt_type::data) {
        p.type = pkt_type::data;
        p.payload = pkt + PRELUDE_BYTES;
        p.n_samples = words - PRELUDE_WORDS;
        if (p.n_samples < 0)
            return false;
    } else if (type == (uint8_t)pkt_type::context) {
        p.type = pkt_type::context;
        p.n_samples = 0;
    } else {
        p.type = pkt_type::unknown;
    }
    return true;
}

/** Convert a data packet payload to fc32 (inverse of pack_data scaling). */
static inline void payload_to_fc32(const uint8_t* payload, int n_samples,
                                   std::complex<float>* out)
{
    constexpr float S = 1.0f / 32767.0f;
    for (int i = 0; i < n_samples; i++) {
        uint16_t bi, bq;
        memcpy(&bi, payload + (size_t)i * 4, 2);
        memcpy(&bq, payload + (size_t)i * 4 + 2, 2);
        out[i] = { (float)(int16_t)be16toh(bi) * S,
                   (float)(int16_t)be16toh(bq) * S };
    }
}

} // namespace difi
} // namespace cusat_dsp
