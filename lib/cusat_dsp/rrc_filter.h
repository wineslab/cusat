/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Root-raised-cosine pulse design (TX pulse shaping).
 */

#ifndef INCLUDED_CUSAT_DSP_RRC_FILTER_H
#define INCLUDED_CUSAT_DSP_RRC_FILTER_H

#include <vector>

namespace cusat_dsp {

/* Normalization applied to a designed pulse. */
enum class pulse_norm {
    unit_energy,      // sum of squared taps = 1
    dc_gain_sps,      // sum of taps = sps (each polyphase branch ~unit DC gain);
                      // the level the TX chain is calibrated for
};

/* Root-raised-cosine pulse sampled at an integer `sps` samples/symbol,
 * spanning +-span_symbols symbols around its centre: 2 * span_symbols * sps
 * + 1 taps, symmetric. */
std::vector<float> rrc_pulse(int sps, double rolloff, int span_symbols,
                             pulse_norm norm);

/* Closed-form RRC impulse response at t symbol intervals (unnormalized),
 * shared by the TX design above and the RX bank (interpolator.h). */
double rrc_impulse(double t, double rolloff);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_RRC_FILTER_H */
