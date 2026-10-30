/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * R_n(i) = 2 z_n(i + 2^17) + z_n(i), with z_n(i) = x(i + n) ^ y(i) and the
 * two 18-stage m-sequences
 *
 *     x(i + 18) = x(i + 7) ^ x(i),                     x(0) = 1, x(1..17) = 0
 *     y(i + 18) = y(i + 10) ^ y(i + 7) ^ y(i + 5) ^ y(i),   y(0..17) = 1
 *
 * Each sequence is produced by a shift register holding 18 consecutive
 * values (bit k = value at i + k). Four registers run side by side: x at
 * offsets n and n + 2^17, y at offsets 0 and 2^17 (offsets are taken
 * modulo the period 2^18 - 1 by the recursion itself).
 */

#include "framing/pl_scrambler.h"
#include <cstdint>
#include <stdexcept>

namespace cusat_dsp {

namespace {

struct lfsr18_t {
    uint32_t state;
    uint32_t taps; // feedback taps other than stage 0

    bool out() const { return state & 1; }

    void step()
    {
        const uint32_t next = __builtin_parity(state & (taps | 1u));
        state = (state >> 1) | (next << 17);
    }

    void advance(int n)
    {
        for (int i = 0; i < n; i++)
            step();
    }
};

constexpr uint32_t X_TAPS = 1u << 7;
constexpr uint32_t Y_TAPS = (1u << 10) | (1u << 7) | (1u << 5);
constexpr uint32_t X_INIT = 1u;
constexpr uint32_t Y_INIT = (1u << 18) - 1;
constexpr int HALF_PERIOD_OFFSET = 1 << 17;

} // namespace

void pl_descrambling_sequence(int gold_code, complexf* seq, int len)
{
    if (gold_code < 0 || gold_code >= N_GOLD_CODES)
        throw std::out_of_range("pl_descrambling_sequence: invalid scrambling code");

    lfsr18_t xa{ X_INIT, X_TAPS }, xb{ X_INIT, X_TAPS };
    lfsr18_t ya{ Y_INIT, Y_TAPS }, yb{ Y_INIT, Y_TAPS };
    xa.advance(gold_code);
    xb.advance(gold_code + HALF_PERIOD_OFFSET);
    yb.advance(HALF_PERIOD_OFFSET);

    // e^{-j R pi/2} for R = 0..3
    static const complexf ROT[4] = { { 1, 0 }, { 0, -1 }, { -1, 0 }, { 0, 1 } };
    for (int i = 0; i < len; i++) {
        const int r = 2 * (xb.out() ^ yb.out()) + (xa.out() ^ ya.out());
        seq[i] = ROT[r];
        xa.step();
        xb.step();
        ya.step();
        yb.step();
    }
}

} // namespace cusat_dsp
