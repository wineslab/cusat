/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Decision-directed SNR estimates: every symbol is replaced by its nearest
 * constellation point, and the SNR is the ratio of the energy of those
 * decisions to the energy of the decision errors. The estimate is biased
 * upwards once decision errors become frequent.
 */

#ifndef INCLUDED_CUSAT_DSP_MODULATION_DD_SNR_H
#define INCLUDED_CUSAT_DSP_MODULATION_DD_SNR_H

#include "modulation/constellation.h"
#include <cusat_dsp/compat.h>

namespace cusat_dsp {

/* QPSK: the decisions are the signs of I and Q. Linear SNR. */
float qpsk_snr_dd(const complexf* sym, int n_sym);

/* Any constellation, evaluated on every stride-th symbol. Linear SNR. */
float constellation_snr_dd(const constellation_t& c, const complexf* sym, int n_sym, int stride = 1);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_MODULATION_DD_SNR_H */
