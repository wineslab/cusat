/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Baseband (de)scrambler, ETSI EN 302 307-1 clause 5.2.2 / figure 5: the
 * complete BBFRAME is XORed, MSB first, with the PRBS 1 + X^14 + X^15 whose
 * register is loaded with 100101010000000 at the start of every BBFRAME.
 * The operation is its own inverse (scrambling = descrambling).
 */

#ifndef INCLUDED_CUSAT_DSP_BB_SCRAMBLER_H
#define INCLUDED_CUSAT_DSP_BB_SCRAMBLER_H

#include <cstddef>
#include <cstdint>

namespace cusat_dsp {

/* Largest BBFRAME (Kbch / 8 of the normal 9/10 code). */
constexpr size_t BB_SCRAMBLER_MAX_BYTES = 7274;

/* out[i] = in[i] ^ prbs[i] for one BBFRAME of n_bytes <= BB_SCRAMBLER_MAX_BYTES
 * (in and out may alias). */
void bb_scramble(const uint8_t* in, uint8_t* out, size_t n_bytes);
void bb_scramble_inplace(uint8_t* data, size_t n_bytes);

/* The PRBS bytes of one BBFRAME (BB_SCRAMBLER_MAX_BYTES of them). */
const uint8_t* bb_scrambler_sequence();

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_BB_SCRAMBLER_H */
