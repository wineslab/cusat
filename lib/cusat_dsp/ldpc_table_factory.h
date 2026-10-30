/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared LDPC table selection used by both encoder and decoder.
 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_TABLE_FACTORY_H
#define INCLUDED_CUSAT_DSP_LDPC_TABLE_FACTORY_H

#include "ldpc_tables/ldpc_code.h"
#include <cusat_dsp/modcod_types.h>

namespace cusat_dsp {

/**
 * @brief Create the ldpc_code descriptor for the given standard/framesize/rate.
 *
 * The returned struct points into static constexpr table data; the struct
 * itself is heap-allocated — caller owns it (delete when done).
 * Returns nullptr if the combination is unsupported.
 */
ldpc_code* create_ldpc_table(fecframe_size_t framesize,
                             code_rate_t rate);

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_LDPC_TABLE_FACTORY_H */
