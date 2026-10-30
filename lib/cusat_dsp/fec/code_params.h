/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Outer BCH and inner LDPC code dimensions per FECFRAME size and code rate:
 * EN 302 307-1 Tables 5a/5b, EN 302 307-2 Tables 4a/4b/4c.
 */

#ifndef INCLUDED_CUSAT_DSP_FEC_CODE_PARAMS_H
#define INCLUDED_CUSAT_DSP_FEC_CODE_PARAMS_H

#include <cusat_dsp/modcod_types.h>
#include <cstdint>

namespace cusat_dsp {

struct fec_params_t {
    uint32_t k_bch;  // BBFRAME bits
    uint32_t n_bch;  // BCH codeword bits = LDPC message bits
    uint32_t t_bch;  // correctable errors
    uint32_t k_ldpc; // = n_bch
    uint32_t n_ldpc; // FECFRAME bits
};

/* Throws std::invalid_argument for a rate that the frame size does not define. */
fec_params_t get_fec_params(fecframe_size_t framesize, code_rate_t rate);

bool has_fec_params(fecframe_size_t framesize, code_rate_t rate);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_FEC_CODE_PARAMS_H */
