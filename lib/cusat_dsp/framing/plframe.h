/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLFRAME geometry derived from an 8-bit PLS code: ETSI EN 302 307-1
 * clause 5.5 (Table 11, Table 12, Figure 13) and EN 302 307-2 clause 5.5
 * (Table 16, Table 17a, Table 17b).
 */

#ifndef INCLUDED_CUSAT_DSP_FRAMING_PLFRAME_H
#define INCLUDED_CUSAT_DSP_FRAMING_PLFRAME_H

#include <cusat_dsp/modcod_types.h>
#include <cstdint>

namespace cusat_dsp {
namespace pl {

/* Standard-mandated sizes (EN 302 307-1 clause 5.5, 5.5.2, 5.5.3). */
inline constexpr int kSofLen = 26;          // SOF field, symbols
inline constexpr int kPlsCodeLen = 64;      // PLS code field, symbols
inline constexpr int kHeaderLen = 90;       // PLHEADER = one slot
inline constexpr int kSlotLen = 90;         // M
inline constexpr int kPilotBlockLen = 36;   // P
inline constexpr int kSlotsPerPilotGroup = 16;
inline constexpr int kDummySlots = 36;      // clause 5.5.1
inline constexpr int kMaxSlots = 360;       // Table 11/16
inline constexpr int kNumPlsCodes = 256;    // 8-bit PLS code (EN 302 307-2)

/* Upper bounds over every layout this implementation handles, in symbols
 *: scrambled part (data + pilots) and whole PLFRAME. */
int max_payload_symbols();
int max_plframe_symbols();

} // namespace pl

/* Layout of one PLFRAME. Only the signalled quantities are stored; every
 * length is derived. */
struct pl_layout {
    uint8_t pls_code = 0;              // 8-bit PLS code of the PLHEADER
    uint8_t modcod = 0;                // cuSAT MODCOD index (acm_config.h)
    fecframe_size_t fec_size = FECFRAME_NORMAL;
    bool pilots = false;               // pilot blocks present
    bool dummy = false;                // dummy PLFRAME (no payload delivered)
    bool reserved = false;             // reserved code with a fixed length (Table 17b)
    uint8_t bits_per_symbol = 0;       // 0 when no supported constellation
    uint16_t slots = 0;                // S

    int pilot_block_count() const;     // P-blocks actually transmitted
    int data_symbols() const;          // XFECFRAME length = 90 S
    int payload_symbols() const;       // data + pilot symbols (scrambled part)
    int plframe_symbols() const;       // PLHEADER + payload
    bool deliverable() const;          // carries a decodable XFECFRAME
};

/* Layout implied by a received or configured PLS code. Never fails: codes
 * without a supported constellation yield a layout with deliverable() ==
 * false but with the correct total length whenever the standard defines one
 *. */
pl_layout pl_layout_from_code(uint8_t pls_code);

/* PLS code that cuSAT transmits for (modcod, frame size, pilots). */
uint8_t pls_code_for(uint8_t modcod, fecframe_size_t fec_size, bool pilots);

/* Convenience: pl_layout_from_code(pls_code_for(...)). */
pl_layout pl_layout_for(uint8_t modcod, fecframe_size_t fec_size, bool pilots);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FRAMING_PLFRAME_H */
