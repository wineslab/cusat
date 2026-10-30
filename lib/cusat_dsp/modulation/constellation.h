/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Signal constellations of ETSI EN 302 307-1 clause 5.4 (QPSK, 8PSK,
 * 16APSK, 32APSK) and EN 302 307-2 clause 5.4 (8APSK to 256APSK), with the
 * ring ratios and label assignments of the selected code rate.
 */

#ifndef INCLUDED_CUSAT_DSP_MODULATION_CONSTELLATION_H
#define INCLUDED_CUSAT_DSP_MODULATION_CONSTELLATION_H

#include <cusat_dsp/compat.h>
#include <cusat_dsp/modcod_types.h>
#include <array>

namespace cusat_dsp {

/* Points indexed by the bit label of the symbol, first bit = MSB. APSK
 * constellations are normalized to unit average energy. */
struct constellation_t {
    int bits_per_sym = 0;
    int n_points = 0;
    std::array<complexf, 256> points{};
};

/* The constellation used by a MODCOD. The frame size matters only for the
 * 4+12+16rb 32APSK ring ratios. Throws for an unsupported modulation. */
constellation_t make_constellation(modulation_t modulation,
                                   code_rate_t rate,
                                   fecframe_size_t framesize = FECFRAME_NORMAL);

/* Label of the point nearest to sym. */
int nearest_label(const constellation_t& c, complexf sym);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_MODULATION_CONSTELLATION_H */
