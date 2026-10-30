/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test helper (header-only, CPU): synthetic PLFRAME streams and channel
 * impairments for the PL receiver tests.
 *
 * The framing follows EN 302 307-1 clause 5.5 with the retained
 * plheader_symbols() / pl_descrambling_sequence(); data symbols are drawn
 * from the MODCOD's constellation. Impairments: static CFO, linear
 * frequency ramp, Wiener phase noise, AWGN, gain; a carrier rotator with a
 * configurable latency models the frontend feedback loop.
 */

#ifndef INCLUDED_CUSAT_LIB_PL_TEST_STREAM_H
#define INCLUDED_CUSAT_LIB_PL_TEST_STREAM_H

#include "framing/pl_scrambler.h"
#include "framing/plframe.h"
#include "framing/plheader.h"
#include "modulation/constellation.h"
#include "acm_config.h"

#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace cusat_dsp {
namespace pltest {

struct tx_frame {
    uint8_t pls = 0;
    uint64_t sof = 0;                // absolute index of the first symbol
    std::vector<complexf> data;      // XFECFRAME data symbols (empty: dummy)
};

/* PLFRAME generator. */
class framer
{
public:
    explicit framer(int gold = 0, uint32_t seed = 1)
        : d_rng(seed), d_seq(pl::max_payload_symbols())
    {
        pl_descrambling_sequence(gold, d_seq.data(), pl::max_payload_symbols());
    }

    /* Appends one PLFRAME for `pls` to `out`; returns the data symbols. The
     * dummy payload is constant (1+j)/sqrt2 or random QPSK. */
    tx_frame append(uint8_t pls, std::vector<complexf>& out, bool random_dummy = false)
    {
        tx_frame f;
        f.pls = pls;
        f.sof = d_count;
        const pl_layout l = pl_layout_from_code(pls);
        complexf h[pl::kHeaderLen];
        plheader_symbols(pls, h);
        out.insert(out.end(), h, h + pl::kHeaderLen);
        const float a = (float)M_SQRT1_2;
        std::vector<complexf> payload;
        payload.reserve(l.payload_symbols());
        if (l.dummy) {
            std::uniform_int_distribution<int> q(0, 3);
            for (int i = 0; i < l.data_symbols(); i++) {
                if (random_dummy) {
                    const int v = q(d_rng);
                    payload.push_back(complexf((v & 1) ? -a : a, (v & 2) ? -a : a));
                } else {
                    payload.push_back(complexf(a, a));
                }
            }
        } else {
            const modcod_def_t* mc = find_modcod(l.modcod);
            constellation_t c = make_constellation(mc ? mc->constellation : MOD_QPSK,
                                                   mc ? mc->code_rate : RATE_1_2, l.fec_size);
            std::uniform_int_distribution<int> pick(0, c.n_points - 1);
            f.data.resize(l.data_symbols());
            for (auto& s : f.data)
                s = c.points[pick(d_rng)];
            const int P = l.pilot_block_count();
            for (int s = 0; s < l.slots; s++) {
                payload.insert(payload.end(), f.data.begin() + s * 90, f.data.begin() + (s + 1) * 90);
                if (l.pilots && (s + 1) % 16 == 0 && (s + 1) / 16 <= P)
                    for (int i = 0; i < 36; i++)
                        payload.push_back(complexf(a, a));
            }
        }
        for (size_t i = 0; i < payload.size(); i++)
            out.push_back(payload[i] * std::conj(d_seq[i]));
        d_count += pl::kHeaderLen + payload.size();
        return f;
    }

    uint64_t count() const { return d_count; }

private:
    std::mt19937 d_rng;
    std::vector<complexf> d_seq;
    uint64_t d_count = 0;
};

struct impairments {
    double cfo = 0.0;        // cycles/symbol
    double ramp = 0.0;       // cycles/symbol^2
    double pn_dnu_ts = 0.0;  // Wiener phase noise, increment variance 2 pi dnu Ts
    double esn0_db = 100.0;
    double gain = 1.0;
    double phase0 = 0.0;
    uint32_t seed = 7;
};

/* Channel + emulated frontend rotator (removes rot_freq with latency). */
class channel
{
public:
    explicit channel(const impairments& p) : d_p(p), d_rng(p.seed), d_phase(p.phase0) {}

    void set_rotator(double f, uint32_t latency) { d_pending.push_back({ d_n + latency, f }); }

    void apply(const complexf* in, complexf* out, int n)
    {
        const double sigma = std::sqrt(0.5 * std::pow(10.0, -d_p.esn0_db / 10.0));
        std::normal_distribution<double> g(0.0, 1.0);
        const double pn_sd = std::sqrt(2.0 * M_PI * d_p.pn_dnu_ts);
        for (int i = 0; i < n; i++) {
            while (!d_pending.empty() && d_pending.front().first <= d_n) {
                d_rot = d_pending.front().second;
                d_pending.pop_front();
            }
            const double f = d_p.cfo + d_p.ramp * (double)d_n;
            d_phase += 2.0 * M_PI * f;
            if (pn_sd > 0)
                d_phase += pn_sd * g(d_rng);
            d_rot_phase -= 2.0 * M_PI * d_rot;
            const double ph = d_phase + d_rot_phase;
            const std::complex<double> rot(std::cos(ph), std::sin(ph));
            std::complex<double> v = std::complex<double>(in[i]) * rot * d_p.gain;
            if (d_p.esn0_db < 99.0)
                v += std::complex<double>(sigma * g(d_rng), sigma * g(d_rng)) * d_p.gain;
            out[i] = complexf((float)v.real(), (float)v.imag());
            d_n++;
            d_phase = std::remainder(d_phase, 2.0 * M_PI);
            d_rot_phase = std::remainder(d_rot_phase, 2.0 * M_PI);
        }
    }

    /* channel phase (excluding the rotator) at the last symbol */
    double phase() const { return d_phase; }
    double true_freq() const { return d_p.cfo + d_p.ramp * (double)d_n; }

private:
    impairments d_p;
    std::mt19937 d_rng;
    double d_phase = 0.0;
    double d_rot = 0.0;
    double d_rot_phase = 0.0;
    uint64_t d_n = 0;
    std::deque<std::pair<uint64_t, double>> d_pending;
};

} // namespace pltest
} // namespace cusat_dsp

#endif
