/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tx FEC chain: GSE data → BBFRAME → scramble → BCH → LDPC → interleave+map → symbols.
 */

#ifndef INCLUDED_CUSAT_DSP_TX_FEC_CHAIN_H
#define INCLUDED_CUSAT_DSP_TX_FEC_CHAIN_H

#include "bb_scrambler.h"
#include "bb_header.h"
#include "fec/bch_code.h"
#include "ldpc_encode.h"
#include "mapper.h"
#include <cusat_dsp/modcod_types.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace cusat_dsp {

class tx_fec_chain
{
public:
    tx_fec_chain(fecframe_size_t framesize,
                 code_rate_t rate,
                 modulation_t constellation);

    /**
     * @brief Encode one GSE frame: GSE data → XFECFRAME symbols.
     *
     * Builds a BBFRAME with TS_GS=00 (generic packetized), then encodes.
     *
     * @param gse_data GSE packet data (from gse_encapsulator).
     * @param gse_len Number of GSE bytes.
     * @param symbols_out Output buffer for XFECFRAME symbols.
     * @return Number of GSE data bytes consumed.
     */
    unsigned int process_gse(const uint8_t* gse_data, unsigned int gse_len,
                             complexf* symbols_out);

    /**
     * @brief Process GSE data through CPU stages up to BB header only.
     *
     * Builds a BBFRAME but does NOT scramble or BCH encode.
     * Use with GPU BCH path where scramble + BCH are done on device.
     * Result accessible via bbframe().
     */
    void process_gse_to_bbframe(const uint8_t* gse_data, unsigned int gse_len);

    /**
     * @brief Process GSE data through CPU stages up to BCH only (BB→BCH).
     *
     * Stops before LDPC encode — use with encode_ldpc_to_device() for
     * the single-stream GPU path where LDPC writes directly to device memory.
     * BCH output accessible via bch_output().
     */
    void process_gse_to_bch(const uint8_t* gse_data, unsigned int gse_len);

    /**
     * @brief Encode the BCH output through LDPC to a device buffer.
     *
     * Must call process_gse_to_bch() first.
     *
     * @param fecframe_device  Device pointer for output FECFRAME (N/8 bytes).
     * @param cuda_stream      CUDA stream (cudaStream_t cast to void*).
     */
    void encode_ldpc_to_device(uint8_t* fecframe_device, void* cuda_stream);

    /**
     * @brief Batch-encode N BCH outputs through LDPC to device.
     *
     * @param bch_staging       Host pointer to N × bch_len_bytes contiguous.
     * @param fecframes_device  Device pointer to N × n_bytes contiguous.
     * @param batch_size        Number of codewords.
     * @param cuda_stream       CUDA stream (cudaStream_t cast to void*).
     */
    void encode_ldpc_batch_to_device(const uint8_t* bch_staging,
                                     uint8_t* fecframes_device,
                                     int batch_size,
                                     void* cuda_stream);

    /**
     * @brief Batch-encode N BCH outputs from device memory through LDPC (D2D).
     *
     * Same as encode_ldpc_batch_to_device but reads from device memory instead
     * of host memory. Use after GPU BCH encode where BCH output is already on device.
     */
    void encode_ldpc_batch_d2d(const uint8_t* bch_device,
                               uint8_t* fecframes_device,
                               int batch_size,
                               void* cuda_stream);

    /** Access raw BBFRAME after process_gse_to_bbframe(). */
    const uint8_t* bbframe() const { return d_bbframe.data(); }

    /** Access BCH output after process_gse_to_bch(). */
    const uint8_t* bch_output() const { return d_bch_out.data(); }
    unsigned int bch_len_bytes() const { return d_bch_out.size(); }

    /** Access FECFRAME packed bytes after process_*_to_fecframe(). */
    const uint8_t* fecframe() const { return d_fecframe.data(); }

    /** Access mapper configuration for pipeline registration. */
    const mapper& get_mapper() const { return d_mapper; }
    unsigned int fecframe_bits() const { return d_ldpc.n_bits(); }

    /** Enable ACM mode (CCM_ACM=0 in BBHEADER MATYPE). */
    void set_acm_mode(bool acm) { d_bb_header.set_acm_mode(acm); }

    /** Configure BBHEADER roll-off signaling. */
    void set_rolloff(float rolloff) { d_bb_header.set_rolloff(rolloff); }

    /** Access BCH codec (for extracting LUT/params for GPU BCH). */
    const bch_code_t& get_bch() const { return d_bch; }


    unsigned int xfecframe_len() const { return d_mapper.xfecframe_len(); }
    unsigned int kbch_bytes() const { return d_bb_header.kbch_bytes(); }
    unsigned int max_dfl_bytes() const { return d_bb_header.max_dfl_bytes(); }

    /** Sub-stage timing from last process_*_to_fecframe() call (µs). */
    uint64_t lat_bb_bch_us() const { return d_lat_bb_bch_us; }
    uint64_t lat_ldpc_us() const { return d_lat_ldpc_us; }

private:
    bb_header d_bb_header;
    bch_code_t d_bch;
    ldpc_encode d_ldpc;
    mapper d_mapper;

    // Intermediate buffers
    std::vector<uint8_t> d_bbframe;     // kbch bytes
    std::vector<uint8_t> d_bch_out;     // nbch bytes
    std::vector<uint8_t> d_fecframe;    // n_bits bytes

    // Sub-stage timing
    uint64_t d_lat_bb_bch_us = 0;
    uint64_t d_lat_ldpc_us = 0;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_TX_FEC_CHAIN_H */
