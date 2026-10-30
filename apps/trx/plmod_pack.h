/* -*- c++ -*- */
/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host-side frame packer for the dvb_plmod RFNoC block (modulation-compressed
 * fronthaul): per PLFRAME, a 3-word prologue with explicit geometry, the 90
 * literal PLHEADER symbols (sc16), and the payload as packed constellation-
 * index bits. The FPGA maps the indices through its constellation ROM,
 * inserts pilot blocks and applies PL scrambling (Gold code 0), so those
 * stay off the fronthaul entirely.
 *
 * Wire format and ROM layout must match rfnoc_block_dvb_plmod
 * (gen_rom_and_vectors.cc in the dvb-compressed-fh repo): the ROM was dumped
 * from this project's own constellation classes, and the bit groups below
 * use the exact index the GPU mapper computes, so FPGA output is
 * point-identical to the GPU transmitter.
 */

#pragma once

#include <cusat_dsp/compat.h>
#include "modcod_table.h"   // plmod_lookup: modcod -> (n_mod, rom_base)

#include <cmath>
#include <cstdint>

/* dvb_plmod constellation ROM geometry -> (n_mod, ROM base for group 0).
 *
 * Resolved from modcod_table.h (plmod_lookup), a firmware-generated table
 * baked from the same acm_config the modem uses. It covers every supported
 * DVB-S2/S2X modcod (all constellation variants and rate aliases resolved),
 * and the rom_base values are a byproduct of the FPGA's ROM construction —
 * so this cannot drift from the flashed image as long as modcod_table.h is
 * regenerated with it. Returns false for an unmapped modcod. */
inline bool plmod_geom(int modcod, int& n_mod, int& rom_base)
{
    const plmod_modcod_t* e = plmod_lookup((uint8_t)modcod);
    if (!e) return false;
    n_mod    = e->n_mod;
    rom_base = e->rom_base;
    return true;
}

/* Constellation/header amplitude of the block (0.4 FS; the rrc_interp block
 * needs the ~5.8 dB headroom for its worst per-phase gain). */
constexpr int   PLMOD_SCALE  = 13107;
constexpr int   PLMOD_NHDR   = 90;
constexpr int   PLMOD_PERIOD = 16 * 90;   // pilot cadence in data symbols
constexpr int   PLMOD_PLEN   = 36;

inline uint32_t plmod_hdr_word(complexf s)
{
    const int16_t i = (int16_t)lrintf(s.real() * PLMOD_SCALE);
    const int16_t q = (int16_t)lrintf(s.imag() * PLMOD_SCALE);
    return ((uint32_t)(uint16_t)i << 16) | (uint16_t)q;
}

/* Build one modulation-compressed data frame. Returns words written; out
 * must hold 3 + 90 + ceil(n_syms*n_mod/32) words (n_syms up to 9270 for
 * 128APSK -> 2028 bit words). */
inline size_t plmod_build_frame(uint32_t* out, int modcod, bool pilots,
                                const complexf* plheader,
                                const uint8_t* fecframe_packed,
                                unsigned int fecframe_bits,
                                const unsigned int* col_offsets, int stride,
                                int backoff_override = -1)
{
    const plmod_modcod_t* e = plmod_lookup((uint8_t)modcod);
    if (!e) return 0;
    // Per-frame PAPR backoff -> prologue W1[31:24] (0 = unity, N = scale by
    // N/256). Default is the table's geometry-derived value (nonzero only for
    // high-PAPR 128APSK); backoff_override >= 0 forces a value (sweeping).
    const uint32_t backoff = (backoff_override >= 0)
        ? (uint32_t)(backoff_override & 0xFF) : (uint32_t)e->backoff;
    const int n_mod    = e->n_mod;
    const int rom_base = e->rom_base;
    // n_syms = total payload symbols (data + any trailing pad). For every
    // modcod except 128APSK, data_syms*n_mod == fecframe_bits exactly and
    // pad_syms == 0. 128APSK (n_mod 7) is non-uniform: 64800 FECFRAME bits +
    // 6 zero-stuff bits -> 9258 data syms, then 12 pad syms of a fixed
    // constellation index; the FPGA maps/scrambles all n_syms identically.
    const int n_syms    = (int)e->n_slots * 90;
    const int data_syms = n_syms - (int)e->pad_syms;
    const int n_slots   = e->n_slots;
    const int pblocks   = pilots ? ((n_slots - 1) >> 4) : 0;

    size_t w = 0;
    /* n_mod field is 3 bits: 2..7 direct, 0 encodes 8 (256APSK). const_base is
     * 13 bits split low[8:0] in W0, high[12:9] in W1[23:20] (S2X extension). */
    out[w++] = 0xD5000000u | (pilots ? (1u << 22) : 0) | (1u << 21) /*scramble*/ |
               ((uint32_t)(PLMOD_NHDR & 0x1FF) << 12) |
               ((uint32_t)(n_mod & 7) << 9) | (uint32_t)(rom_base & 0x1FF);
    out[w++] = (backoff << 24) |
               ((uint32_t)((rom_base >> 9) & 0xF) << 20) |
               ((uint32_t)n_syms & 0xFFFFF);
    out[w++] = ((uint32_t)(PLMOD_PERIOD & 0xFFFF) << 16) |
               ((uint32_t)(PLMOD_PLEN & 0xFF) << 8) | (uint32_t)(pblocks & 0xFF);

    for (int j = 0; j < PLMOD_NHDR; j++)
        out[w++] = plmod_hdr_word(plheader[j]);

    /* Packed constellation indices: per symbol j, the same LUT index the GPU
     * mapper builds (bit b at col_offsets[b] + j*stride, MSB first), appended
     * MSB-first into MSB-filled 32-bit words. */
    uint32_t word = 0;
    int nb = 0;
    for (int j = 0; j < data_syms; j++) {
        for (int b = 0; b < n_mod; b++) {
            const int bit_pos = (int)col_offsets[b] + j * stride;
            /* bit_pos >= fecframe_bits is a zero-stuff bit (128APSK: 6 bits
             * padded past the 64800-bit FECFRAME before interleaving). */
            const int bit = (bit_pos < (int)fecframe_bits)
                ? ((fecframe_packed[bit_pos >> 3] >> (7 - (bit_pos & 7))) & 1)
                : 0;
            word = (word << 1) | (uint32_t)bit;
            if (++nb == 32) { out[w++] = word; word = 0; nb = 0; }
        }
    }
    /* Trailing pad symbols (128APSK: 12 x index 127): fixed constellation
     * index, MSB-first, mapped/scrambled by the FPGA like any payload symbol. */
    const uint32_t pad_group = (uint32_t)e->pad_index & ((1u << n_mod) - 1);
    for (int i = 0; i < (int)e->pad_syms; i++) {
        for (int b = 0; b < n_mod; b++) {
            const int bit = (int)((pad_group >> (n_mod - 1 - b)) & 1);
            word = (word << 1) | (uint32_t)bit;
            if (++nb == 32) { out[w++] = word; word = 0; nb = 0; }
        }
    }
    if (nb != 0)
        out[w++] = word << (32 - nb);
    return w;
}

/* Dummy PLFRAME (MODCOD 0): prologue + dummy PLHEADER; the block generates
 * the 3240 scrambled (SCALE, 0) payload symbols itself. */
inline size_t plmod_build_dummy(uint32_t* out, const complexf* dummy_plheader)
{
    size_t w = 0;
    out[w++] = 0xD5000000u | (1u << 23) /*dummy*/ | (1u << 21) /*scramble*/ |
               ((uint32_t)(PLMOD_NHDR & 0x1FF) << 12);
    out[w++] = 36 * 90;
    out[w++] = 0;
    for (int j = 0; j < PLMOD_NHDR; j++)
        out[w++] = plmod_hdr_word(dummy_plheader[j]);
    return w;
}
