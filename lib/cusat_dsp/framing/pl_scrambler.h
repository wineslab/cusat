/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLFRAME payload scrambling (EN 302 307-1 clause 5.5.4). The PLHEADER is
 * never scrambled; the payload, pilots included, is multiplied by
 * e^{j R_n(i) pi/2} for scrambling code n.
 */

#ifndef INCLUDED_CUSAT_DSP_FRAMING_PL_SCRAMBLER_H
#define INCLUDED_CUSAT_DSP_FRAMING_PL_SCRAMBLER_H

#include <cusat_dsp/compat.h>

namespace cusat_dsp {

constexpr int N_GOLD_CODES = (1 << 18) - 2;

/* Descrambling factors e^{-j R_n(i) pi/2}, i = 0..len-1. */
void pl_descrambling_sequence(int gold_code, complexf* seq, int len);

/* sym[i] *= seq[i] */
inline void pl_descramble(complexf* sym, const complexf* seq, int len)
{
    for (int i = 0; i < len; i++)
        sym[i] *= seq[i];
}

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FRAMING_PL_SCRAMBLER_H */
