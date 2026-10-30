/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLFRAME geometry.
 *
 * Sources:
 *   EN 302 307-1 Table 11  slots S per constellation order, 64800/16200 bits
 *   EN 302 307-1 Table 12  MODCOD field (b0 = 0): 1-11 QPSK, 12-17 8PSK,
 *                          18-23 16APSK, 24-28 32APSK, 29-31 reserved
 *   EN 302 307-1 Fig. 13   pilot blocks: one per 16 slots, int{(S-1)/16}
 *   EN 302 307-2 Table 16  S for orders 6, 7 (103 slots, padded) and 8
 *   EN 302 307-2 Table 17a S2X MODCODs (b0 = 1), b7 = pilots
 *   EN 302 307-2 Table 17b reserved codes with a fixed frame length
 *
 * Every Table 17b length equals 90 (S + 1) + 36 int{(S-1)/16} for the
 * S of the stated constellation order and pilot setting (checked in
 * qa_pl_layout), so reserved codes are described by (slots, pilots) as
 * well and need no separate length table.
 */

#include "framing/plframe.h"
#include "acm_config.h"
#include "fec/code_params.h"

#include <algorithm>
#include <array>

namespace cusat_dsp {

namespace {

/* Slots of an XFECFRAME for constellation order eta (bits/symbol). */
constexpr int slots_for(int eta, fecframe_size_t size)
{
    const int bits = fecframe_bits(size);
    if (eta == 7) // 128APSK: 6 stuffing bits, 12 padding symbols -> 103 slots
        return size == FECFRAME_NORMAL ? 103 : 0;
    if (eta <= 0 || bits % (90 * eta) != 0)
        return 0;
    return bits / (90 * eta);
}

/* Table 17a, pilots-off codes 132..248 (even). eta = constellation order,
 * modcod = cuSAT index (acm_config.h), 0 if cuSAT has none. */
struct s2x_entry_t {
    uint8_t code;
    uint8_t eta;
    fecframe_size_t size;
    uint8_t modcod;
};

constexpr s2x_entry_t S2X_TABLE[] = {
    { 132, 2, FECFRAME_NORMAL, 33 }, // QPSK 13/45
    { 134, 2, FECFRAME_NORMAL, 43 }, // QPSK 9/20
    { 136, 2, FECFRAME_NORMAL, 44 }, // QPSK 11/20
    { 138, 3, FECFRAME_NORMAL, 56 }, // 8APSK 5/9-L
    { 140, 3, FECFRAME_NORMAL, 57 }, // 8APSK 26/45-L
    { 142, 3, FECFRAME_NORMAL, 45 }, // 8PSK 23/36
    { 144, 3, FECFRAME_NORMAL, 46 }, // 8PSK 25/36
    { 146, 3, FECFRAME_NORMAL, 47 }, // 8PSK 13/18
    { 148, 4, FECFRAME_NORMAL, 58 }, // 16APSK 1/2-L
    { 150, 4, FECFRAME_NORMAL, 59 }, // 16APSK 8/15-L
    { 152, 4, FECFRAME_NORMAL, 60 }, // 16APSK 5/9-L
    { 154, 4, FECFRAME_NORMAL, 48 }, // 16APSK 26/45
    { 156, 4, FECFRAME_NORMAL, 49 }, // 16APSK 3/5
    { 158, 4, FECFRAME_NORMAL, 61 }, // 16APSK 3/5-L
    { 160, 4, FECFRAME_NORMAL, 50 }, // 16APSK 28/45
    { 162, 4, FECFRAME_NORMAL, 51 }, // 16APSK 23/36
    { 164, 4, FECFRAME_NORMAL, 62 }, // 16APSK 2/3-L
    { 166, 4, FECFRAME_NORMAL, 52 }, // 16APSK 25/36
    { 168, 4, FECFRAME_NORMAL, 53 }, // 16APSK 13/18
    { 170, 4, FECFRAME_NORMAL, 54 }, // 16APSK 7/9
    { 172, 4, FECFRAME_NORMAL, 55 }, // 16APSK 77/90
    { 174, 5, FECFRAME_NORMAL, 63 }, // 32APSK 2/3-L
    { 178, 5, FECFRAME_NORMAL, 64 }, // 32APSK 32/45
    { 180, 5, FECFRAME_NORMAL, 65 }, // 32APSK 11/15
    { 182, 5, FECFRAME_NORMAL, 66 }, // 32APSK 7/9
    { 184, 6, FECFRAME_NORMAL, 36 }, // 64APSK 32/45-L
    { 186, 6, FECFRAME_NORMAL, 29 }, // 64APSK 11/15
    { 190, 6, FECFRAME_NORMAL, 30 }, // 64APSK 7/9
    { 194, 6, FECFRAME_NORMAL, 31 }, // 64APSK 4/5
    { 198, 6, FECFRAME_NORMAL, 35 }, // 64APSK 5/6
    { 200, 7, FECFRAME_NORMAL, 32 }, // 128APSK 3/4
    { 202, 7, FECFRAME_NORMAL, 34 }, // 128APSK 7/9
    { 204, 8, FECFRAME_NORMAL, 39 }, // 256APSK 29/45-L
    { 206, 8, FECFRAME_NORMAL, 41 }, // 256APSK 2/3-L
    { 208, 8, FECFRAME_NORMAL, 40 }, // 256APSK 31/45-L
    { 210, 8, FECFRAME_NORMAL, 38 }, // 256APSK 32/45
    { 212, 8, FECFRAME_NORMAL, 42 }, // 256APSK 11/15-L
    { 214, 8, FECFRAME_NORMAL, 37 }, // 256APSK 3/4
    { 216, 2, FECFRAME_SHORT, 0 },   // QPSK 11/45
    { 218, 2, FECFRAME_SHORT, 0 },   // QPSK 4/15
    { 220, 2, FECFRAME_SHORT, 0 },   // QPSK 14/45
    { 222, 2, FECFRAME_SHORT, 0 },   // QPSK 7/15
    { 224, 2, FECFRAME_SHORT, 0 },   // QPSK 8/15
    { 226, 2, FECFRAME_SHORT, 0 },   // QPSK 32/45
    { 228, 3, FECFRAME_SHORT, 0 },   // 8PSK 7/15
    { 230, 3, FECFRAME_SHORT, 0 },   // 8PSK 8/15
    { 232, 3, FECFRAME_SHORT, 0 },   // 8PSK 26/45
    { 234, 3, FECFRAME_SHORT, 0 },   // 8PSK 32/45
    { 236, 4, FECFRAME_SHORT, 0 },   // 16APSK 7/15
    { 238, 4, FECFRAME_SHORT, 0 },   // 16APSK 8/15
    { 240, 4, FECFRAME_SHORT, 0 },   // 16APSK 26/45
    { 242, 4, FECFRAME_SHORT, 0 },   // 16APSK 3/5
    { 244, 4, FECFRAME_SHORT, 0 },   // 16APSK 32/45
    { 246, 5, FECFRAME_SHORT, 0 },   // 32APSK 2/3
    { 248, 5, FECFRAME_SHORT, 0 },   // 32APSK 32/45
};

/* Table 17b: reserved codes, generic constellation order and pilots. The
 * pilot bit does not signal pilots for 250..255 (all "pilots on"). */
struct reserved_entry_t {
    uint8_t code;
    uint8_t eta;
    bool pilots;
};

constexpr reserved_entry_t RESERVED_TABLE[] = {
    { 128, 3, false }, { 130, 4, false }, { 176, 5, false }, { 177, 5, true },
    { 188, 6, false }, { 189, 6, true },  { 192, 6, false }, { 193, 6, true },
    { 196, 6, false }, { 197, 6, true },  { 250, 3, true },  { 251, 4, true },
    { 252, 5, true },  { 253, 6, true },  { 254, 8, true },  { 255, 10, true },
};

/* Constellation order of a b0 = 0 MODCOD field (Table 12), 0 if reserved. */
constexpr int s2_eta(int modcod)
{
    if (modcod >= 1 && modcod <= 11) return 2;
    if (modcod >= 12 && modcod <= 17) return 3;
    if (modcod >= 18 && modcod <= 23) return 4;
    if (modcod >= 24 && modcod <= 28) return 5;
    return 0;
}

pl_layout build_layout(uint8_t code)
{
    pl_layout l;
    l.pls_code = code;
    if (!(code & 0x80)) {
        const int modcod = code >> 2;
        l.fec_size = (code & 2) ? FECFRAME_SHORT : FECFRAME_NORMAL;
        if (modcod == 0) {
            // clause 5.5.1: header + 36 slots, never pilots
            l.dummy = true;
            l.slots = pl::kDummySlots;
            return l;
        }
        const int eta = s2_eta(modcod);
        if (eta == 0)
            return l; // reserved MODCOD 29-31: length unknown
        l.pilots = code & 1;
        l.bits_per_symbol = eta;
        l.slots = slots_for(eta, l.fec_size);
        l.modcod = modcod;
        return l;
    }
    for (const reserved_entry_t& r : RESERVED_TABLE) {
        if (r.code == code) {
            l.reserved = true;
            l.pilots = r.pilots;
            // a generic n-ary normal frame: S = 64800 / (90 log2 n)
            l.slots = slots_for(r.eta, FECFRAME_NORMAL);
            return l;
        }
    }
    for (const s2x_entry_t& e : S2X_TABLE) {
        if (e.code == (code & 0xFE)) {
            l.fec_size = e.size;
            l.pilots = code & 1;
            l.bits_per_symbol = e.eta;
            l.slots = slots_for(e.eta, e.size);
            l.modcod = e.modcod;
            return l;
        }
    }
    return l; // VL-SNR (129, 131): length not defined here
}

const std::array<pl_layout, pl::kNumPlsCodes>& layout_table()
{
    static const std::array<pl_layout, pl::kNumPlsCodes> table = [] {
        std::array<pl_layout, pl::kNumPlsCodes> t{};
        for (int c = 0; c < pl::kNumPlsCodes; c++)
            t[c] = build_layout((uint8_t)c);
        return t;
    }();
    return table;
}

} // namespace

namespace pl {

int max_payload_symbols()
{
    static const int v = [] {
        int m = 0;
        for (const pl_layout& l : layout_table())
            m = std::max(m, l.payload_symbols());
        return m;
    }();
    return v;
}

int max_plframe_symbols() { return kHeaderLen + max_payload_symbols(); }

} // namespace pl

int pl_layout::pilot_block_count() const
{
    if (!pilots || slots == 0)
        return 0;
    return (slots - 1) / pl::kSlotsPerPilotGroup;
}

int pl_layout::data_symbols() const { return pl::kSlotLen * slots; }

int pl_layout::payload_symbols() const
{
    return data_symbols() + pl::kPilotBlockLen * pilot_block_count();
}

int pl_layout::plframe_symbols() const
{
    return slots == 0 ? 0 : pl::kHeaderLen + payload_symbols();
}

bool pl_layout::deliverable() const
{
    if (dummy || reserved || modcod == 0 || bits_per_symbol == 0 || slots == 0)
        return false;
    const modcod_def_t* m = find_modcod(modcod);
    return m != nullptr && modcod_supports_framesize(*m, fec_size);
}

pl_layout pl_layout_from_code(uint8_t pls_code) { return layout_table()[pls_code]; }

uint8_t pls_code_for(uint8_t modcod, fecframe_size_t fec_size, bool pilots)
{
    if (modcod <= 28) {
        if (fec_size == FECFRAME_MEDIUM)
            return 0; // no b0 = 0 code for medium frames
        return (uint8_t)((modcod << 2) | ((fec_size == FECFRAME_SHORT) << 1) |
                         (pilots ? 1 : 0));
    }
    for (const s2x_entry_t& e : S2X_TABLE)
        if (e.modcod == modcod && e.size == fec_size)
            return (uint8_t)(e.code | (pilots ? 1 : 0));
    return 0; // no PLS code: callers check modcod_supports_framesize()
}

pl_layout pl_layout_for(uint8_t modcod, fecframe_size_t fec_size, bool pilots)
{
    return pl_layout_from_code(pls_code_for(modcod, fec_size, pilots));
}

} // namespace cusat_dsp
