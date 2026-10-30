/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLHEADER: start-of-frame field and PLS code (EN 302 307-1 clause 5.5.2,
 * extended to 8-bit PLS codes by EN 302 307-2 clause 5.5.2).
 */

#ifndef INCLUDED_CUSAT_DSP_FRAMING_PLHEADER_H
#define INCLUDED_CUSAT_DSP_FRAMING_PLHEADER_H

#include "framing/plframe.h"
#include <cusat_dsp/compat.h>
#include <array>
#include <cstdint>
#include <vector>

namespace cusat_dsp {

// Bit strings are right-aligned; the most significant bit is sent first.
constexpr uint32_t SOF_BITS = 0x18D2E82;
constexpr uint64_t PLSC_SCRAMBLER = 0x719D83C953422DFAull;

/* 64-bit PLS codeword, before scrambling. */
uint64_t pls_codeword(uint8_t pls);

/* pi/2-BPSK: bit value v at position k is sent as (1 - 2v) e^{j pi/4 (1 + 2 (k & 1))}. */
void pi2bpsk_map(uint64_t bits, int n_bits, complexf* out);

/* The 90 PLHEADER symbols of a PLS code: SOF, then the scrambled codeword,
 * rotated by pi/2 when b0 = 1. */
void plheader_symbols(uint8_t pls, complexf* out);

struct pls_decision_t {
    uint8_t pls;
    float score;  // correlation with the chosen codeword, 64 when noiseless
    float margin; // gap to the best other candidate (infinite if none)
};

/* Maximum-likelihood soft decoder of the PLS code, restricted to a set of
 * candidate codes. */
class pls_decoder
{
public:
    explicit pls_decoder(const std::vector<uint8_t>& candidates);

    /* plsc: the 64 PLSC symbols with carrier phase already removed. */
    pls_decision_t decode(const complexf* plsc) const;

private:
    std::array<bool, pl::kNumPlsCodes> d_candidate{};
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FRAMING_PLHEADER_H */
