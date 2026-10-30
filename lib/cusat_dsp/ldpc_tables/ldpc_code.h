/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Flat LDPC code description for the DVB-S2/S2X IRA codes.
 *
 * Each ETSI parity-check table is stored as three flat arrays, which is the
 * exact layout the CUDA encoder/decoder kernels consume:
 *
 *   row_deg[r]  — number of accumulator addresses for address row r
 *   row_off[r]  — prefix offset of row r into addr[] (row_off[n_rows] = total)
 *   addr[]      — accumulator base addresses, row-major
 *
 * One "address row" covers m (=360) consecutive information bits: bit
 * i = r*m + j checks accumulator (addr[row_off[r] + c] + j*q) % (n - k)
 * for c in [0, row_deg[r]).
 */

#ifndef INCLUDED_CUSAT_DSP_LDPC_TABLES_LDPC_CODE_H
#define INCLUDED_CUSAT_DSP_LDPC_TABLES_LDPC_CODE_H

#include <cstdint>

namespace cusat_dsp {

struct ldpc_code {
    int n;            ///< codeword length in bits (N)
    int k;            ///< information length in bits (K)
    int m;            ///< circulant block size (360)
    int links_total;  ///< Tanner-graph edges incl. parity chain (BNL sizing)
    int links_max_cn; ///< max check-node degree incl. the 2 parity edges
    int n_rows;       ///< number of address rows (k/m)

    const uint8_t* row_deg;  ///< [n_rows]
    const uint32_t* row_off; ///< [n_rows + 1]
    const uint16_t* addr;    ///< [row_off[n_rows]]

    constexpr int r() const { return n - k; } ///< parity bits (R = N - K)
    constexpr int q() const { return (n - k) / m; } ///< circulant shift step
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_LDPC_TABLES_LDPC_CODE_H */
