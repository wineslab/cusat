/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Polyphase matched-filter / interpolation bank for the GPU timing recovery.
 *
 * Designs n_phases fractional-delay subfilters consumed by
 * cuda/timing_recovery_kernel.cu. Interpolation itself runs on the GPU;
 * this class only computes coefficients.
 */

#ifndef INCLUDED_CUSAT_DSP_INTERPOLATOR_H
#define INCLUDED_CUSAT_DSP_INTERPOLATOR_H

#include <cstddef>
#include <vector>

namespace cusat_dsp {

/* Prototype response of the bank. */
enum class bank_prototype {
    rrc_matched,    // root-raised-cosine matched filter (EN 302 307-1 clause 5.6)
    band_limited,   // neutral band-limited interpolator for pre-filtered input
};

struct filter_bank_design {
    double samples_per_symbol = 2.0; // input oversampling, may be fractional
    double rolloff = 0.2;            // EN 302 307-1 clause 5.6 / EN 302 307-2 clause 5.6
    int span_symbols = 5;            // one-sided prototype span, symbols
    int n_phases = 128;              // number of fractional-delay subfilters
    bank_prototype prototype = bank_prototype::rrc_matched;
};

class polyphase_filter_bank
{
public:
    explicit polyphase_filter_bank(const filter_bank_design& design);

    int n_phases() const;
    int taps_per_phase() const;          // odd, identical for every phase
    unsigned history() const;            // taps_per_phase() - 1 past samples

    /* Coefficients of phase p, stored for a dot product against samples
     * held newest-first. */
    const float* taps(int phase) const;

    /* All phases, phase-major: [n_phases * taps_per_phase]. */
    std::vector<float> flat_taps() const;

    /* Group delay of phase 0 in input samples (may be fractional); phase p
     * adds p / n_phases samples. */
    double group_delay_samples() const;

private:
    int d_n_phases = 0;
    int d_taps_per_phase = 0;
    double d_group_delay = 0.0;
    std::vector<float> d_taps; // phase-major
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_INTERPOLATOR_H */
