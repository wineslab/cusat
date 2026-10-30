/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tx_fec_chain.h"
#include <chrono>

namespace cusat_dsp {

tx_fec_chain::tx_fec_chain(fecframe_size_t framesize,
                           code_rate_t rate,
                           modulation_t constellation)
    : d_bb_header(framesize, rate),
      d_ldpc(framesize, rate),
      d_mapper(framesize, rate, constellation)
{
    const fec_params_t fec = get_fec_params(framesize, rate);

    d_bch = make_bch_code(framesize, fec.n_bch, fec.t_bch);

    // Allocate intermediate buffers
    d_bbframe.resize(fec.k_bch / 8);
    d_bch_out.resize(fec.n_bch / 8);
    d_fecframe.resize(fec.n_ldpc / 8);
}

unsigned int tx_fec_chain::process_gse(const uint8_t* gse_data, unsigned int gse_len,
                                       complexf* symbols_out)
{
    // 1. Build GSE BBFRAME
    unsigned int gse_consumed =
        d_bb_header.build_gse_bbframe(gse_data, gse_len, d_bbframe.data());

    // 2. BB scramble
    bb_scramble_inplace(d_bbframe.data(), d_bbframe.size());

    // 3. BCH encode
    bch_encode(d_bch, d_bbframe.data(), d_bch_out.data());

    // 4. LDPC encode
    d_ldpc.encode(d_bch_out.data(), d_fecframe.data());

    // 5. Interleave + constellation map
    d_mapper.interleave_and_map(d_fecframe.data(), symbols_out);

    return gse_consumed;
}

void tx_fec_chain::process_gse_to_bbframe(const uint8_t* gse_data, unsigned int gse_len)
{
    d_bb_header.build_gse_bbframe(gse_data, gse_len, d_bbframe.data());
}

void tx_fec_chain::process_gse_to_bch(const uint8_t* gse_data, unsigned int gse_len)
{
    auto t0 = std::chrono::steady_clock::now();
    d_bb_header.build_gse_bbframe(gse_data, gse_len, d_bbframe.data());
    bb_scramble_inplace(d_bbframe.data(), d_bbframe.size());
    bch_encode(d_bch, d_bbframe.data(), d_bch_out.data());
    auto t1 = std::chrono::steady_clock::now();
    d_lat_bb_bch_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

void tx_fec_chain::encode_ldpc_to_device(uint8_t* fecframe_device, void* cuda_stream)
{
    auto t0 = std::chrono::steady_clock::now();
    d_ldpc.encode_to_device(d_bch_out.data(), fecframe_device, cuda_stream);
    auto t1 = std::chrono::steady_clock::now();
    d_lat_ldpc_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

void tx_fec_chain::encode_ldpc_batch_to_device(const uint8_t* bch_staging,
                                                uint8_t* fecframes_device,
                                                int batch_size,
                                                void* cuda_stream)
{
    auto t0 = std::chrono::steady_clock::now();
    d_ldpc.encode_batch_to_device(bch_staging, fecframes_device,
                                  batch_size, cuda_stream);
    auto t1 = std::chrono::steady_clock::now();
    d_lat_ldpc_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

void tx_fec_chain::encode_ldpc_batch_d2d(const uint8_t* bch_device,
                                          uint8_t* fecframes_device,
                                          int batch_size,
                                          void* cuda_stream)
{
    auto t0 = std::chrono::steady_clock::now();
    d_ldpc.encode_batch_d2d(bch_device, fecframes_device,
                            batch_size, cuda_stream);
    auto t1 = std::chrono::steady_clock::now();
    d_lat_ldpc_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

} // namespace cusat_dsp
