/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * BBHEADER construction for DVB-S2 Tx.
 */

#ifndef INCLUDED_CUSAT_DSP_BB_HEADER_H
#define INCLUDED_CUSAT_DSP_BB_HEADER_H

#include "fec/code_params.h"
#include <array>
#include <cstdint>

namespace cusat_dsp {

// BBHEADER (EN 302 307-1 clause 5.1.6)
constexpr int BBHEADER_BITS = 80;
constexpr int BBHEADER_BYTES = BBHEADER_BITS / 8;
constexpr int TS_GS_GENERIC_PACKETIZED = 0; // MATYPE-1 TS/GS field

class bb_header
{
public:
    bb_header(fecframe_size_t framesize,
              code_rate_t rate);

    /**
     * @brief Build one BBFRAME carrying GSE data.
     *
     * Writes BBHEADER with TS_GS=00 (generic packetized), then copies
     * gse_data into the data field, then zero-pads to kbch_bytes.
     *
     * @param gse_data GSE packet data (from gse_encapsulator).
     * @param gse_len Number of GSE data bytes (must be <= max_dfl_bytes()).
     * @param bbframe_out Output buffer (must be at least kbch_bytes()).
     * @return Number of GSE data bytes consumed (== min(gse_len, max_dfl_bytes)).
     */
    unsigned int build_gse_bbframe(const uint8_t* gse_data, unsigned int gse_len,
                                   uint8_t* bbframe_out);

    /** Enable ACM mode (CCM_ACM=0 in MATYPE). Default is CCM (CCM_ACM=1). */
    void set_acm_mode(bool acm) { d_acm_mode = acm; }

    /** Set the DVB-S2/S2X MATYPE roll-off signaling. */
    void set_rolloff(float rolloff);

    unsigned int kbch_bytes() const { return d_kbch_bytes; }
    unsigned int max_dfl_bytes() const { return d_max_dfl_bytes; }

private:
    unsigned int d_kbch_bytes;
    unsigned int d_max_dfl_bytes; // (kbch - 80) / 8
    bool d_acm_mode = false;
    bool d_low_rolloff = false;
    bool d_low_rolloff_marker_next = true;
    uint8_t d_rolloff_bits = 0b10; // default CLI rolloff is 0.20

    uint8_t next_rolloff_bits();
    void write_gse_header(uint8_t* out, unsigned int dfl_bits);
    uint8_t compute_crc8(const uint8_t* data, int len);
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_BB_HEADER_H */
