/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tx frontend: RRC pulse shaping via polyphase filterbank.
 *
 * The RRC filter h[n] (length L) is decomposed into SPS sub-filters:
 *   hp[k] = h[k*SPS + p]   for phase p = 0..SPS-1, tap k = 0..sub_taps-1
 *
 * For each input symbol x[n], the SPS output samples are:
 *   out[n*SPS + p] = sum_k hp[k] * x[n-k]
 *
 * Taps are interleaved [h0[0],h1[0],h2[0],h3[0], h0[1],h1[1],...] so all
 * phases are computed together per symbol with NEON SIMD — one symbol load
 * drives all 4 output samples.
 */

#include "transmitter_frontend.h"
#include "rrc_filter.h"
#include <cstring>

#include "cuda/rrc_kernel.cuh"

namespace cusat_dsp {

transmitter_frontend::transmitter_frontend(int sps, float rolloff, int rrc_delay)
    : d_sps(sps), d_rrc_cuda(nullptr)
{
    d_ntaps = 2 * rrc_delay * sps + 1;
    d_taps = rrc_pulse(sps, rolloff, rrc_delay, pulse_norm::dc_gain_sps);

    // Decompose into polyphase sub-filters, stored interleaved and reversed.
    d_sub_taps = (d_ntaps + d_sps - 1) / d_sps;
    d_history_len = d_sub_taps - 1;

    d_interleaved_taps.resize(d_sub_taps * d_sps, 0.0f);
    for (int p = 0; p < d_sps; p++) {
        for (int k = 0; k < d_sub_taps; k++) {
            int tap_idx = k * d_sps + p;
            if (tap_idx < d_ntaps) {
                d_interleaved_taps[(d_sub_taps - 1 - k) * d_sps + p] = d_taps[tap_idx];
            }
        }
    }

    d_symbol_history.resize(d_history_len, complexf(0, 0));

    d_rrc_cuda = new cuda::rrc_cuda(
        d_sps, d_sub_taps, d_history_len, d_interleaved_taps.data());
    fprintf(stderr, "transmitter_frontend: using CUDA RRC (sps=%d, sub_taps=%d)\n",
            d_sps, d_sub_taps);
}

transmitter_frontend::~transmitter_frontend()
{
    delete d_rrc_cuda;
}

unsigned int transmitter_frontend::pulse_shape(const complexf* symbols,
                                               unsigned int n_symbols,
                                               complexf* samples_out)
{
    const int sps = d_sps;
    unsigned int n_out = n_symbols * sps;
    unsigned int buf_len = d_history_len + n_symbols;

    // Build contiguous buffer: [history | current symbols]
    if (d_symbol_buf.size() < buf_len)
        d_symbol_buf.resize(buf_len);
    memcpy(d_symbol_buf.data(),
           d_symbol_history.data(),
           d_history_len * sizeof(complexf));
    memcpy(d_symbol_buf.data() + d_history_len,
           symbols,
           n_symbols * sizeof(complexf));

    const float* buf = (const float*)d_symbol_buf.data();

    n_out = d_rrc_cuda->pulse_shape(buf, n_symbols, (float*)samples_out);

    // Save history for next frame
    memcpy(d_symbol_history.data(),
           d_symbol_buf.data() + n_symbols,
           d_history_len * sizeof(complexf));

    return n_out;
}

unsigned int transmitter_frontend::flush(complexf* samples_out)
{
    unsigned int flush_symbols = d_sub_taps;
    std::vector<complexf> zeros(flush_symbols, complexf(0, 0));
    return pulse_shape(zeros.data(), flush_symbols, samples_out);
}

} // namespace cusat_dsp
