/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BCH outer code (EN 302 307-1 clause 5.3.1, EN 302 307-2 clause 5.3.1).
 * The generator of a t-error-correcting code is g1(x) g2(x) ... gt(x),
 * with the polynomials listed for the normal (GF(2^16), Table 6a), short
 * (GF(2^14), Table 6b) and medium (GF(2^15)) FECFRAME sizes.
 */

#ifndef INCLUDED_CUSAT_DSP_FEC_BCH_CODE_H
#define INCLUDED_CUSAT_DSP_FEC_BCH_CODE_H

#include <cusat_dsp/modcod_types.h>
#include <array>
#include <cstdint>

namespace cusat_dsp {

/* GF(2) polynomial of degree < 256: bit i % 64 of word i / 64 is the
 * coefficient of x^i. */
typedef std::array<uint64_t, 4> poly256_t;

struct bch_code_t {
    int n_bits = 0;      // N_bch
    int k_bits = 0;      // K_bch
    int parity_bits = 0; // deg g(x)
    poly256_t generator{};
    /* T[b] = b(x) x^parity_bits mod g(x), shared by CPU and GPU BCH. */
    std::array<poly256_t, 256> encoder_lut{};
};

bch_code_t make_bch_code(fecframe_size_t framesize, int n_bits, int t);

/* Systematic encoding of a byte-aligned code: the k_bits / 8 message bytes
 * followed by parity_bits / 8 parity bytes, most significant bit first. */
void bch_encode(const bch_code_t& code, const uint8_t* msg, uint8_t* codeword);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FEC_BCH_CODE_H */
