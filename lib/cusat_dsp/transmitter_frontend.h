/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tx frontend: RRC pulse shaping via polyphase filterbank.
 *
 * Decomposes the RRC FIR into SPS sub-filters. Each input symbol produces
 * SPS output samples — no zero-insertion waste. All phases computed together
 * per symbol for cache efficiency. Each output sample is independent → GPU-
 * friendly (one thread per sample).
 */

#ifndef INCLUDED_CUSAT_DSP_TRANSMITTER_FRONTEND_H
#define INCLUDED_CUSAT_DSP_TRANSMITTER_FRONTEND_H

#include <cusat_dsp/compat.h>
#include <vector>

namespace cusat_dsp {
namespace cuda { class rrc_cuda; }

class transmitter_frontend
{
public:
    transmitter_frontend(int sps, float rolloff, int rrc_delay = 25);
    ~transmitter_frontend();

    /**
     * @brief Pulse-shape symbols into IQ samples.
     *
     * @param symbols Input symbols.
     * @param n_symbols Number of input symbols.
     * @param samples_out Output IQ samples (must hold at least n_symbols * sps values).
     * @return Number of output samples written.
     */
    unsigned int pulse_shape(const complexf* symbols, unsigned int n_symbols,
                             complexf* samples_out);

    int sps() const { return d_sps; }
    int ntaps() const { return d_ntaps; }
    int sub_taps() const { return d_sub_taps; }
    int history_len() const { return d_history_len; }
    const float* interleaved_taps() const { return d_interleaved_taps.data(); }
    /**
     * @brief Flush the filter tail.
     *
     * @param samples_out Output buffer.
     * @return Number of samples written.
     */
    unsigned int flush(complexf* samples_out);

private:
    int d_sps;
    int d_ntaps;
    std::vector<float> d_taps;

    // Polyphase filterbank
    int d_sub_taps;    // taps per sub-filter (ceil(ntaps/sps))
    int d_history_len; // symbols of history needed (sub_taps - 1)
    std::vector<float> d_interleaved_taps;    // [sub_taps * sps] interleaved + reversed
    std::vector<complexf> d_symbol_buf;     // [history | current frame]
    std::vector<complexf> d_symbol_history; // last history_len symbols

    cuda::rrc_cuda* d_rrc_cuda;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_TRANSMITTER_FRONTEND_H */
