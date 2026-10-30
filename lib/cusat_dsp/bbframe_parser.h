/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BBHEADER parser and GSE data-field extraction (ETSI EN 302 307-1 clause
 * 5.1.6, table 3). The 80-bit header is MATYPE-1, MATYPE-2, UPL, DFL, SYNC,
 * SYNCD, CRC-8, transmitted MSB first.
 */

#ifndef INCLUDED_CUSAT_DSP_BBFRAME_PARSER_H
#define INCLUDED_CUSAT_DSP_BBFRAME_PARSER_H

#include <cstddef>
#include <cstdint>

namespace cusat_dsp {

struct bbheader_fields {
    // MATYPE-1
    uint8_t matype1 = 0; // raw byte
    uint8_t ts_gs = 0;   // 2 bits: 11 transport, 00 generic packetized, 01 continuous
    bool sis = false;    // SIS/MIS: true = single input stream
    bool ccm = false;    // CCM/ACM: true = CCM
    bool issyi = false;  // input stream synchronisation indicator
    bool npd = false;    // null-packet deletion
    uint8_t ro = 0;      // 2 bits roll-off code (00 0.35, 01 0.25, 10 0.20)
    // MATYPE-2 (ISI when MIS)
    uint8_t matype2 = 0;
    uint16_t upl = 0;    // user packet length, bits
    uint16_t dfl = 0;    // data field length, bits
    uint8_t sync = 0;
    uint16_t syncd = 0;
    uint8_t crc8 = 0;
};

/* Splits the 10 header bytes of a BBFRAME of kbch_bytes bytes into `out`
 * (filled even on failure). Returns false on a CRC-8 failure, on
 * DFL > 8 kbch_bytes - 80, on DFL not a multiple of 8, or when the frame is
 * shorter than the header. */
bool decode_bbheader(const uint8_t* bbframe, size_t kbch_bytes, bbheader_fields& out);

/* For a valid header whose TS/GS field is the generic-packetized value the
 * cuSAT transmitter writes for GSE (bb_header.h), copies the DFL / 8 data-
 * field bytes after the header into out and returns their count; returns 0
 * for an invalid or non-GSE frame or when the field exceeds max_out. */
int extract_gse_datafield(const uint8_t* bbframe, size_t kbch_bytes, uint8_t* out, int max_out);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_BBFRAME_PARSER_H */
