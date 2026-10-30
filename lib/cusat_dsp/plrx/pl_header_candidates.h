/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Result of the bulk PLHEADER search, produced on the GPU and consumed by
 * the PL receiver on the host.
 */

#ifndef INCLUDED_CUSAT_DSP_PLRX_PL_HEADER_CANDIDATES_H
#define INCLUDED_CUSAT_DSP_PLRX_PL_HEADER_CANDIDATES_H

#include <cstdint>

namespace cusat_dsp {

struct pl_header_candidates {
    static constexpr int kMaxCandidates = 64;

    /* Number of symbols the search covered (the caller's buffer length). */
    int searched = 0;
    /* Candidates, strongest first. index: position of the FIRST SOF symbol
     * relative to the first symbol of the batch handed to
     * pl_receiver::process(); may be negative down to -pl::kHeaderLen + 1
     * when the header started in the previous batch (overlap region). */
    int count = 0;
    int index[kMaxCandidates] = {};
    /* Detection statistic, normalized so that it does not depend on the
     * input amplitude. */
    float score[kMaxCandidates] = {};
    /* Mean input power over the searched span (for diagnostics/thresholds). */
    float mean_power = 0.0f;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_PLRX_PL_HEADER_CANDIDATES_H */
