/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GSE (Generic Stream Encapsulation) per ETSI TS 102 606
 */

#include "gse.h"
#include <cassert>
#include <cstdint>
#include <cstring>

namespace cusat_dsp {

/*------------------------------------------------------------------------
 * CRC-32 (IEEE 802.3, reflected algorithm)
 *----------------------------------------------------------------------*/

static std::array<uint32_t, 256> build_crc32_table()
{
    std::array<uint32_t, 256> table;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320u;
            else
                crc >>= 1;
        }
        table[i] = crc;
    }
    return table;
}

static const auto s_crc32_table = build_crc32_table();

uint32_t gse_calc_crc32(const uint8_t* data, int len)
{
    uint32_t crc = 0xFFFFFFFF;
    for (int i = 0; i < len; i++) {
        crc = (crc >> 8) ^ s_crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

/*------------------------------------------------------------------------
 * GSE Encapsulator
 *----------------------------------------------------------------------*/

gse_encapsulator::gse_encapsulator(uint16_t protocol_type,
                                   gse_label_type_t label_type,
                                   const uint8_t* label)
    : d_protocol_type(protocol_type),
      d_label_type(label_type),
      d_frag_id_counter(0),
      d_pending_offset(0),
      d_current_frag_id(0)
{
    int llen = gse_label_len(label_type);
    if (label && llen > 0) {
        d_label.assign(label, label + llen);
    }
}

void gse_encapsulator::write_gse_header(uint8_t* out, bool start, bool end,
                                        int lt, uint16_t gse_length)
{
    uint16_t hdr = 0;
    if (start) hdr |= (1 << 15);
    if (end) hdr |= (1 << 14);
    hdr |= (static_cast<uint16_t>(lt) & 0x3) << 12;
    hdr |= gse_length & 0xFFF;
    out[0] = (hdr >> 8) & 0xFF;
    out[1] = hdr & 0xFF;
}

int gse_encapsulator::complete_overhead() const
{
    // header(2) + protocol_type(2) + label(0-6)
    return 2 + 2 + gse_label_len(d_label_type);
}

int gse_encapsulator::first_frag_overhead() const
{
    // header(2) + frag_id(1) + total_length(2) + protocol_type(2) + label(0-6)
    return 2 + 1 + 2 + 2 + gse_label_len(d_label_type);
}

int gse_encapsulator::cont_frag_overhead(bool last)
{
    // header(2) + frag_id(1) + [CRC-32(4) if last]
    return 2 + 1 + (last ? 4 : 0);
}

bool gse_encapsulator::pdu_size_valid(int pdu_len) const
{
    if (pdu_len < 0)
        return false;
    int llen = gse_label_len(d_label_type);
    return (uint64_t)2 + (uint64_t)llen + (uint64_t)pdu_len <= UINT16_MAX;
}

int gse_encapsulator::encapsulate(const uint8_t* pdu, int pdu_len,
                                  uint8_t* out, int space_available)
{
    assert(!has_pending());
    if (!pdu_size_valid(pdu_len))
        return -1;

    int total_complete = complete_overhead() + pdu_len;
    // gse_length is a 12-bit field (TS 102 606), so the bytes after the 2-byte
    // GSE header must be <= 0xFFF. A complete packet carries proto+label+pdu;
    // anything larger must be fragmented even if it would fit the frame space.
    int complete_gse_length = 2 + gse_label_len(d_label_type) + pdu_len;

    if (total_complete <= space_available && complete_gse_length <= 0xFFF) {
        // Complete GSE packet (S=1, E=1)
        int llen = gse_label_len(d_label_type);
        uint16_t gse_length = 2 + llen + pdu_len;
        write_gse_header(out, true, true, d_label_type, gse_length);
        int pos = 2;

        // Protocol Type
        out[pos++] = (d_protocol_type >> 8) & 0xFF;
        out[pos++] = d_protocol_type & 0xFF;

        // Label
        if (llen > 0) {
            memcpy(out + pos, d_label.data(), llen);
            pos += llen;
        }

        // PDU data
        memcpy(out + pos, pdu, pdu_len);
        pos += pdu_len;

        return pos;
    }

    // Fragmentation required — write first fragment (S=1, E=0)
    int overhead = first_frag_overhead();
    if (space_available < overhead + 1) {
        return 0; // Not enough space for even 1 byte of payload
    }

    d_pending_pdu.assign(pdu, pdu + pdu_len);
    d_pending_offset = 0;
    d_current_frag_id = d_frag_id_counter++;

    int llen = gse_label_len(d_label_type);
    int payload_len = space_available - overhead;
    // Cap so gse_length (= 1+2+2+llen+payload_len) stays within the 12-bit field.
    int max_payload = 0xFFF - (1 + 2 + 2 + llen);
    if (payload_len > max_payload) payload_len = max_payload;
    uint16_t total_length = 2 + llen + pdu_len; // proto + label + all PDU data
    uint16_t gse_length = 1 + 2 + 2 + llen + payload_len;

    write_gse_header(out, true, false, d_label_type, gse_length);
    int pos = 2;

    out[pos++] = d_current_frag_id;

    // Total Length
    out[pos++] = (total_length >> 8) & 0xFF;
    out[pos++] = total_length & 0xFF;

    // Protocol Type
    out[pos++] = (d_protocol_type >> 8) & 0xFF;
    out[pos++] = d_protocol_type & 0xFF;

    // Label
    if (llen > 0) {
        memcpy(out + pos, d_label.data(), llen);
        pos += llen;
    }

    // PDU data (partial)
    memcpy(out + pos, pdu, payload_len);
    pos += payload_len;
    d_pending_offset = payload_len;

    return pos;
}

int gse_encapsulator::write_pending(uint8_t* out, int space_available)
{
    if (!has_pending()) return 0;

    int remaining = d_pending_pdu.size() - d_pending_offset;

    // Check if this can be the last fragment. Besides fitting the frame space,
    // the last-fragment gse_length (= 1+remaining+4) must fit the 12-bit field.
    int last_overhead = cont_frag_overhead(true);
    bool is_last = (remaining + last_overhead <= space_available) &&
                   (1 + remaining + 4 <= 0xFFF);

    if (is_last) {
        // Last fragment (S=0, E=1)
        uint16_t gse_length = 1 + remaining + 4;
        write_gse_header(out, false, true, d_label_type, gse_length);
        int pos = 2;

        out[pos++] = d_current_frag_id;

        // Payload
        memcpy(out + pos, d_pending_pdu.data() + d_pending_offset, remaining);
        pos += remaining;

        // CRC-32 over: Total_Length + Protocol_Type + Label + all PDU data
        int llen = gse_label_len(d_label_type);
        uint16_t total_length = 2 + llen + d_pending_pdu.size();

        std::vector<uint8_t> crc_data;
        crc_data.reserve(2 + 2 + llen + d_pending_pdu.size());
        crc_data.push_back((total_length >> 8) & 0xFF);
        crc_data.push_back(total_length & 0xFF);
        crc_data.push_back((d_protocol_type >> 8) & 0xFF);
        crc_data.push_back(d_protocol_type & 0xFF);
        if (llen > 0) {
            crc_data.insert(crc_data.end(), d_label.begin(), d_label.end());
        }
        crc_data.insert(crc_data.end(), d_pending_pdu.begin(), d_pending_pdu.end());

        uint32_t crc = gse_calc_crc32(crc_data.data(), crc_data.size());
        out[pos++] = (crc >> 24) & 0xFF;
        out[pos++] = (crc >> 16) & 0xFF;
        out[pos++] = (crc >> 8) & 0xFF;
        out[pos++] = crc & 0xFF;

        d_pending_pdu.clear();
        d_pending_offset = 0;

        return pos;
    }

    // Intermediate fragment (S=0, E=0)
    int overhead = cont_frag_overhead(false);
    if (space_available < overhead + 1) {
        return 0;
    }

    int payload_len = space_available - overhead;
    if (payload_len > remaining) payload_len = remaining;
    // gse_length = 1 + payload_len must fit the 12-bit field.
    if (payload_len > 0xFFF - 1) payload_len = 0xFFF - 1;

    uint16_t gse_length = 1 + payload_len;
    write_gse_header(out, false, false, d_label_type, gse_length);
    int pos = 2;

    out[pos++] = d_current_frag_id;

    memcpy(out + pos, d_pending_pdu.data() + d_pending_offset, payload_len);
    pos += payload_len;
    d_pending_offset += payload_len;

    return pos;
}

int gse_encapsulator::write_padding(uint8_t* out, int space_available)
{
    if (space_available < 2) return 0;
    memset(out, 0, space_available);
    return space_available;
}

/*------------------------------------------------------------------------
 * GSE Decapsulator
 *----------------------------------------------------------------------*/

gse_decapsulator::gse_decapsulator() : d_pdu_count(0), d_error_count(0)
{
    for (auto& ctx : d_reassembly) {
        ctx.active = false;
    }
}

void gse_decapsulator::reset()
{
    for (auto& ctx : d_reassembly)
        ctx.active = false;
    d_pdu_count = 0;
    d_error_count = 0;
}

void gse_decapsulator::decapsulate(const uint8_t* bbdata, int len, pdu_callback_t cb)
{
    int pos = 0;

    auto expected_payload_len = [](const reassembly_ctx& ctx) -> int {
        int non_payload = 2 + (int)ctx.label.size(); // Protocol_Type + Label
        if (ctx.total_length < non_payload)
            return -1;
        return ctx.total_length - non_payload;
    };

    auto append_fragment_payload =
        [&](reassembly_ctx& ctx, const uint8_t* payload, int payload_len) -> bool {
            int expected = expected_payload_len(ctx);
            if (expected < 0 ||
                payload_len < 0 ||
                ctx.payload.size() + (size_t)payload_len > (size_t)expected) {
                ctx.active = false;
                d_error_count++;
                return false;
            }
            ctx.payload.insert(ctx.payload.end(), payload, payload + payload_len);
            return true;
        };

    while (pos + 2 <= len) {
        // Parse 2-byte GSE header
        uint16_t hdr = (static_cast<uint16_t>(bbdata[pos]) << 8) | bbdata[pos + 1];
        bool start = (hdr >> 15) & 1;
        bool end = (hdr >> 14) & 1;
        int lt = (hdr >> 12) & 0x3;
        uint16_t gse_length = hdr & 0xFFF;

        if (gse_length == 0) {
            break; // Padding — rest of BBFRAME is zeros
        }

        pos += 2;
        if (pos + gse_length > len) {
            d_error_count++;
            break; // Truncated
        }
        int pkt_end = pos + gse_length;

        if (start && end) {
            // Complete PDU (S=1, E=1)
            if (gse_length < 2) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            uint16_t proto =
                (static_cast<uint16_t>(bbdata[pos]) << 8) | bbdata[pos + 1];
            pos += 2;

            int llen = gse_label_len(lt);
            if (pos + llen > pkt_end) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }
            const uint8_t* label = (llen > 0) ? &bbdata[pos] : nullptr;
            pos += llen;

            int pdu_len = pkt_end - pos;
            if (pdu_len > 0) {
                cb(&bbdata[pos], pdu_len, proto, label, llen);
                d_pdu_count++;
            }
            pos = pkt_end;

        } else if (start && !end) {
            // First fragment (S=1, E=0)
            if (gse_length < 5) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            uint8_t frag_id = bbdata[pos++];
            uint16_t total_length =
                (static_cast<uint16_t>(bbdata[pos]) << 8) | bbdata[pos + 1];
            pos += 2;
            uint16_t proto =
                (static_cast<uint16_t>(bbdata[pos]) << 8) | bbdata[pos + 1];
            pos += 2;

            int llen = gse_label_len(lt);
            if (pos + llen > pkt_end) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }
            const uint8_t* label = (llen > 0) ? &bbdata[pos] : nullptr;
            pos += llen;

            int payload_len = pkt_end - pos;
            if (total_length < (uint16_t)(2 + llen) ||
                payload_len > (int)total_length - 2 - llen) {
                auto& ctx = d_reassembly[frag_id];
                ctx.active = false;
                ctx.payload.clear();
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            auto& ctx = d_reassembly[frag_id];
            ctx.active = true;
            ctx.total_length = total_length;
            ctx.protocol_type = proto;
            ctx.label.clear();
            if (llen > 0 && label) {
                ctx.label.assign(label, label + llen);
            }
            ctx.payload.clear();
            if (!append_fragment_payload(ctx, &bbdata[pos], payload_len)) {
                pos = pkt_end;
                continue;
            }

            // Pre-build CRC header: Total_Length + Protocol_Type + Label
            ctx.crc_header.clear();
            ctx.crc_header.push_back((total_length >> 8) & 0xFF);
            ctx.crc_header.push_back(total_length & 0xFF);
            ctx.crc_header.push_back((proto >> 8) & 0xFF);
            ctx.crc_header.push_back(proto & 0xFF);
            if (llen > 0 && label) {
                ctx.crc_header.insert(ctx.crc_header.end(), label, label + llen);
            }

            pos = pkt_end;

        } else if (!start && !end) {
            // Intermediate fragment (S=0, E=0)
            if (gse_length < 1) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            uint8_t frag_id = bbdata[pos++];
            int payload_len = pkt_end - pos;

            auto& ctx = d_reassembly[frag_id];
            if (!ctx.active) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            if (!append_fragment_payload(ctx, &bbdata[pos], payload_len)) {
                pos = pkt_end;
                continue;
            }
            pos = pkt_end;

        } else {
            // Last fragment (S=0, E=1)
            if (gse_length < 5) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            uint8_t frag_id = bbdata[pos++];

            auto& ctx = d_reassembly[frag_id];
            if (!ctx.active) {
                d_error_count++;
                pos = pkt_end;
                continue;
            }

            int payload_plus_crc = pkt_end - pos;
            if (payload_plus_crc < 4) {
                d_error_count++;
                ctx.active = false;
                pos = pkt_end;
                continue;
            }
            int payload_len = payload_plus_crc - 4;

            if (!append_fragment_payload(ctx, &bbdata[pos], payload_len)) {
                pos = pkt_end;
                continue;
            }
            pos += payload_len;

            if ((int)ctx.payload.size() != expected_payload_len(ctx)) {
                d_error_count++;
                ctx.active = false;
                pos = pkt_end;
                continue;
            }

            // Read received CRC-32
            uint32_t received_crc =
                (static_cast<uint32_t>(bbdata[pos]) << 24) |
                (static_cast<uint32_t>(bbdata[pos + 1]) << 16) |
                (static_cast<uint32_t>(bbdata[pos + 2]) << 8) |
                static_cast<uint32_t>(bbdata[pos + 3]);
            pos += 4;

            // Compute CRC-32 over: Total_Length + Protocol_Type + Label + all payload
            std::vector<uint8_t> crc_input = ctx.crc_header;
            crc_input.insert(crc_input.end(), ctx.payload.begin(), ctx.payload.end());
            uint32_t computed_crc = gse_calc_crc32(crc_input.data(), crc_input.size());

            if (computed_crc == received_crc) {
                cb(ctx.payload.data(), ctx.payload.size(), ctx.protocol_type,
                   ctx.label.empty() ? nullptr : ctx.label.data(), ctx.label.size());
                d_pdu_count++;
            } else {
                d_error_count++;
            }

            ctx.active = false;
        }
    }
}

} // namespace cusat_dsp
