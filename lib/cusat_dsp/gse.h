/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GSE (Generic Stream Encapsulation) per ETSI TS 102 606
 */

#ifndef INCLUDED_CUSAT_DSP_GSE_H
#define INCLUDED_CUSAT_DSP_GSE_H

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace cusat_dsp {

/** GSE label types (LT field, 2 bits) */
enum gse_label_type_t {
    GSE_LT_6BYTE = 0,     // 6-byte MAC address
    GSE_LT_3BYTE = 1,     // 3-byte label
    GSE_LT_BROADCAST = 2, // No label (broadcast/multicast)
    GSE_LT_REUSE = 3      // Re-use previous label
};

/** Get label length in bytes for a given label type */
inline int gse_label_len(int lt)
{
    switch (lt) {
    case GSE_LT_6BYTE: return 6;
    case GSE_LT_3BYTE: return 3;
    default: return 0;
    }
}

/** Compute IEEE 802.3 CRC-32 (used for GSE fragment reassembly) */
uint32_t gse_calc_crc32(const uint8_t* data, int len);

/**
 * @brief GSE Encapsulator
 *
 * Encapsulates PDUs into GSE packets per ETSI TS 102 606.
 * Handles fragmentation when a PDU exceeds available space.
 *
 * Usage:
 *   1. Call encapsulate() with a PDU and available BBFRAME data field space.
 *   2. If has_pending(), call write_pending() with subsequent BBFRAME spaces.
 *   3. Call write_padding() to fill remaining BBFRAME data field space.
 */
class gse_encapsulator
{
public:
    /**
     * @param protocol_type EtherType for encapsulated PDUs (e.g. 0x0800 for IPv4)
     * @param label_type Label type for all packets
     * @param label Label bytes (required for 6BYTE and 3BYTE types)
     */
    gse_encapsulator(uint16_t protocol_type = 0x0800,
                     gse_label_type_t label_type = GSE_LT_BROADCAST,
                     const uint8_t* label = nullptr);

    /**
     * @brief Encapsulate a complete PDU into GSE packet(s).
     *
     * If the PDU fits in the available space, writes a single complete GSE packet.
     * If not, writes the first fragment and stores remaining data internally.
     * Must not be called while has_pending() is true.
     *
     * @param pdu PDU data (e.g. IP packet or Ethernet frame)
     * @param pdu_len PDU length in bytes
     * @param out Output buffer
     * @param space_available Bytes available in output buffer
     * @return Bytes written to out (0 if not enough space for even a first fragment)
     */
    int encapsulate(const uint8_t* pdu, int pdu_len,
                    uint8_t* out, int space_available);

    /**
     * @brief Write pending fragment(s) from a previously started PDU.
     *
     * Call this at the start of each new BBFRAME data field when has_pending().
     *
     * @param out Output buffer
     * @param space_available Bytes available
     * @return Bytes written to out
     */
    int write_pending(uint8_t* out, int space_available);

    /**
     * @brief Fill remaining BBFRAME data field space with padding.
     *
     * Writes a padding header (GSE_LENGTH=0) followed by zero bytes.
     *
     * @param out Output buffer
     * @param space_available Bytes to fill
     * @return Bytes written (== space_available if >= 2, else 0)
     */
    static int write_padding(uint8_t* out, int space_available);

    bool has_pending() const { return d_pending_offset < (int)d_pending_pdu.size(); }

    /** True if the PDU can be represented in the 16-bit GSE Total Length field. */
    bool pdu_size_valid(int pdu_len) const;

private:
    uint16_t d_protocol_type;
    gse_label_type_t d_label_type;
    std::vector<uint8_t> d_label;
    uint8_t d_frag_id_counter;

    // Fragmentation state
    std::vector<uint8_t> d_pending_pdu;
    int d_pending_offset;
    uint8_t d_current_frag_id;

    /** Write 2-byte GSE header */
    static void write_gse_header(uint8_t* out, bool start, bool end,
                                 int lt, uint16_t gse_length);

    /** Overhead (bytes) for a complete GSE packet, excluding PDU data */
    int complete_overhead() const;
    /** Overhead (bytes) for a first fragment, excluding PDU payload */
    int first_frag_overhead() const;
    /** Overhead (bytes) for intermediate/last fragment, excluding payload */
    static int cont_frag_overhead(bool last);
};

/**
 * @brief GSE Decapsulator
 *
 * Parses GSE packets from BBFRAME data fields and reassembles fragmented PDUs.
 * Calls a callback for each complete reassembled PDU.
 */
class gse_decapsulator
{
public:
    using pdu_callback_t = std::function<void(const uint8_t* pdu, int pdu_len,
                                              uint16_t protocol_type,
                                              const uint8_t* label, int label_len)>;

    gse_decapsulator();

    /**
     * @brief Process GSE packets from a BBFRAME data field.
     *
     * @param bbdata BBFRAME data field (after BBHEADER)
     * @param len Length of data field in bytes
     * @param cb Callback invoked for each reassembled PDU
     */
    void decapsulate(const uint8_t* bbdata, int len, pdu_callback_t cb);

    /** Reset all reassembly state (call on lock loss). */
    void reset();

    uint64_t pdu_count() const { return d_pdu_count; }
    uint64_t error_count() const { return d_error_count; }

private:
    struct reassembly_ctx {
        std::vector<uint8_t> payload;
        uint16_t total_length;
        uint16_t protocol_type;
        std::vector<uint8_t> label;
        std::vector<uint8_t> crc_header; // Total_Length + Protocol_Type + Label
        bool active;
    };

    std::array<reassembly_ctx, 256> d_reassembly;
    uint64_t d_pdu_count;
    uint64_t d_error_count;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_GSE_H */
