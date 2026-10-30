/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bb_header.h"
#include "fec/crc8.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cusat_dsp {

bb_header::bb_header(fecframe_size_t framesize,
                     code_rate_t rate)
{
    const fec_params_t fec = get_fec_params(framesize, rate);
    d_kbch_bytes = fec.k_bch / 8;
    d_max_dfl_bytes = (fec.k_bch - BBHEADER_BITS) / 8;
}

uint8_t bb_header::compute_crc8(const uint8_t* data, int len)
{
    return crc8_bbheader(data, len);
}

void bb_header::write_gse_header(uint8_t* out, unsigned int dfl_bits)
{
    // MATYPE-1: TS_GS=00 (generic packetized), SIS=1, CCM/ACM,
    // ISSYI=0, NPD=0, RO=<configured roll-off>.
    out[0] = (d_acm_mode ? 0b00100000 : 0b00110000) | next_rolloff_bits();
    // MATYPE-2: reserved (ISI=0)
    out[1] = 0x00;
    // UPL: 0 (not used for generic packetized)
    out[2] = 0x00;
    out[3] = 0x00;
    // DFL: Data Field Length
    out[4] = (dfl_bits >> 8) & 0xFF;
    out[5] = dfl_bits & 0xFF;
    // SYNC: 0x00 (not used for GSE)
    out[6] = 0x00;
    // SYNCD: 0xFFFF (not applicable for generic streams)
    out[7] = 0xFF;
    out[8] = 0xFF;
    // CRC-8 over bytes 0..8
    out[9] = compute_crc8(out, 9);
}

void bb_header::set_rolloff(float rolloff)
{
    d_low_rolloff = false;
    d_low_rolloff_marker_next = true;

    if (std::fabs(rolloff - 0.35f) < 0.005f) {
        d_rolloff_bits = 0b00;
    } else if (std::fabs(rolloff - 0.25f) < 0.005f) {
        d_rolloff_bits = 0b01;
    } else if (std::fabs(rolloff - 0.20f) < 0.005f) {
        d_rolloff_bits = 0b10;
    } else if (std::fabs(rolloff - 0.15f) < 0.005f) {
        d_low_rolloff = true;
        d_rolloff_bits = 0b00;
    } else if (std::fabs(rolloff - 0.10f) < 0.005f) {
        d_low_rolloff = true;
        d_rolloff_bits = 0b01;
    } else if (std::fabs(rolloff - 0.05f) < 0.005f) {
        d_low_rolloff = true;
        d_rolloff_bits = 0b10;
    } else {
        throw std::invalid_argument("unsupported DVB-S2/S2X rolloff");
    }
}

uint8_t bb_header::next_rolloff_bits()
{
    if (!d_low_rolloff)
        return d_rolloff_bits;

    if (d_low_rolloff_marker_next) {
        d_low_rolloff_marker_next = false;
        return 0b11;
    }

    d_low_rolloff_marker_next = true;
    return d_rolloff_bits;
}

unsigned int bb_header::build_gse_bbframe(const uint8_t* gse_data, unsigned int gse_len,
                                          uint8_t* bbframe_out)
{
    unsigned int dfl_bytes = (gse_len < d_max_dfl_bytes) ? gse_len : d_max_dfl_bytes;
    unsigned int dfl_bits = dfl_bytes * 8;

    write_gse_header(bbframe_out, dfl_bits);

    // Copy GSE data
    uint8_t* data_out = bbframe_out + BBHEADER_BYTES;
    memcpy(data_out, gse_data, dfl_bytes);

    // Zero-pad remaining
    unsigned int remaining = d_kbch_bytes - BBHEADER_BYTES - dfl_bytes;
    if (remaining > 0) {
        memset(data_out + dfl_bytes, 0, remaining);
    }

    return dfl_bytes;
}

} // namespace cusat_dsp
