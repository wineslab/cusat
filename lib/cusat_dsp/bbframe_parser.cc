/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BBHEADER parser (EN 302 307-1 clause 5.1.6, table 3).
 */

#include "bbframe_parser.h"
#include "bb_header.h"
#include "fec/crc8.h"
#include <cstring>

namespace cusat_dsp {

bool decode_bbheader(const uint8_t* bbframe, size_t kbch_bytes, bbheader_fields& out)
{
    out = bbheader_fields{};
    if (!bbframe || kbch_bytes < (size_t)BBHEADER_BYTES)
        return false;
    const uint8_t* h = bbframe;
    out.matype1 = h[0];
    out.ts_gs = (uint8_t)(h[0] >> 6);
    out.sis = (h[0] >> 5) & 1;
    out.ccm = (h[0] >> 4) & 1;
    out.issyi = (h[0] >> 3) & 1;
    out.npd = (h[0] >> 2) & 1;
    out.ro = (uint8_t)(h[0] & 3);
    out.matype2 = h[1];
    out.upl = (uint16_t)((h[2] << 8) | h[3]);
    out.dfl = (uint16_t)((h[4] << 8) | h[5]);
    out.sync = h[6];
    out.syncd = (uint16_t)((h[7] << 8) | h[8]);
    out.crc8 = h[9];
    // the CRC over the header including its CRC byte is 0 for a valid header
    if (crc8_bbheader(h, BBHEADER_BYTES) != 0)
        return false;
    if ((size_t)out.dfl > 8 * kbch_bytes - BBHEADER_BITS)
        return false;
    if (out.dfl % 8 != 0)
        return false;
    return true;
}

int extract_gse_datafield(const uint8_t* bbframe, size_t kbch_bytes, uint8_t* out, int max_out)
{
    bbheader_fields f;
    if (!decode_bbheader(bbframe, kbch_bytes, f))
        return 0;
    if (f.ts_gs != (uint8_t)TS_GS_GENERIC_PACKETIZED)
        return 0;
    const int n = f.dfl / 8;
    if (n > max_out || (n > 0 && !out))
        return 0;
    if (n > 0)
        std::memcpy(out, bbframe + BBHEADER_BYTES, (size_t)n);
    return n;
}

} // namespace cusat_dsp
