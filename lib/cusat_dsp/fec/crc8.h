/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CRC-8 of the BBHEADER (EN 302 307-1 clause 5.1.6): generator
 * x^8 + x^7 + x^6 + x^4 + x^2 + 1, register cleared at start, MSB first.
 */

#ifndef INCLUDED_CUSAT_DSP_FEC_CRC8_H
#define INCLUDED_CUSAT_DSP_FEC_CRC8_H

#include <cstdint>

namespace cusat_dsp {

/* CRC of data[0..len). Over data followed by its own CRC the result is 0. */
inline uint8_t crc8_bbheader(const uint8_t* data, int len)
{
    uint8_t reg = 0;
    for (int i = 0; i < len; i++) {
        reg ^= data[i];
        for (int b = 0; b < 8; b++)
            reg = (reg & 0x80) ? (uint8_t)((reg << 1) ^ 0xD5) : (uint8_t)(reg << 1);
    }
    return reg;
}

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FEC_CRC8_H */
