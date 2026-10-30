/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Table-driven baseband scrambler (EN 302 307-1 clause 5.2.2). The PRBS of
 * one BBFRAME does not depend on the data, so it is generated once (bit-
 * serially, from the register of figure 5) and applied as an XOR, eight
 * bytes at a time.
 */

#include "bb_scrambler.h"
#include <array>
#include <cstring>
#include <stdexcept>

namespace cusat_dsp {

namespace {

std::array<uint8_t, BB_SCRAMBLER_MAX_BYTES> make_sequence()
{
    // stages 1..15 held in bits 0..14; initial content 100101010000000
    // (stage 1 first, figure 5)
    const char* init = "100101010000000";
    unsigned reg = 0;
    for (int s = 0; s < 15; s++)
        if (init[s] == '1')
            reg |= 1u << s;
    std::array<uint8_t, BB_SCRAMBLER_MAX_BYTES> seq{};
    for (size_t i = 0; i < seq.size(); i++) {
        uint8_t byte = 0;
        for (int b = 0; b < 8; b++) {
            // output = stage 14 XOR stage 15, fed back into stage 1
            const unsigned out = ((reg >> 13) ^ (reg >> 14)) & 1u;
            reg = ((reg << 1) | out) & 0x7fffu;
            byte = (uint8_t)((byte << 1) | out); // MSB first
        }
        seq[i] = byte;
    }
    return seq;
}

const std::array<uint8_t, BB_SCRAMBLER_MAX_BYTES>& sequence()
{
    static const std::array<uint8_t, BB_SCRAMBLER_MAX_BYTES> seq = make_sequence();
    return seq;
}

} // namespace

const uint8_t* bb_scrambler_sequence() { return sequence().data(); }

void bb_scramble(const uint8_t* in, uint8_t* out, size_t n_bytes)
{
    if (n_bytes > BB_SCRAMBLER_MAX_BYTES)
        throw std::invalid_argument("bb_scramble: BBFRAME longer than Kbch max");
    const uint8_t* p = sequence().data();
    size_t i = 0;
    for (; i + 8 <= n_bytes; i += 8) {
        uint64_t a, s;
        std::memcpy(&a, in + i, 8);
        std::memcpy(&s, p + i, 8);
        a ^= s;
        std::memcpy(out + i, &a, 8);
    }
    for (; i < n_bytes; i++)
        out[i] = (uint8_t)(in[i] ^ p[i]);
}

void bb_scramble_inplace(uint8_t* data, size_t n_bytes) { bb_scramble(data, data, n_bytes); }

} // namespace cusat_dsp
