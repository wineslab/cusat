/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL Framer: constructs complete PLFRAMEs from XFECFRAME symbols.
 * SOF + PLSC + data slots + pilot blocks + PL scrambling.
 */

#ifndef INCLUDED_CUSAT_DSP_PL_FRAMER_H
#define INCLUDED_CUSAT_DSP_PL_FRAMER_H

#include "framing/plframe.h"
#include <cusat_dsp/compat.h>
#include <vector>

namespace cusat_dsp {
namespace cuda { class pl_scramble_cuda; }

class pl_framer
{
public:
    /**
     * @brief Construct a PL framer.
     *
     * @param modcod 5-bit MODCOD value.
     * @param short_fecframe True for short FECFRAME.
     * @param has_pilots True to insert pilot blocks.
     * @param gold_code Gold code for PL scrambling (default 0).
     */
    pl_framer(uint8_t modcod, bool short_fecframe, bool has_pilots, int gold_code = 0);
    ~pl_framer();

    /**
     * @brief Frame one XFECFRAME into a complete PLFRAME.
     *
     * @param xfecframe_in XFECFRAME symbols (xfecframe_len complex values).
     * @param plframe_out Output PLFRAME symbols (plframe_len complex values).
     */
    void frame(const complexf* xfecframe_in, complexf* plframe_out);

    /**
     * @brief Switch MODCOD for ACM/VCM (updates PLSC and PLS info for next frame).
     *
     * @param modcod 5-bit MODCOD value.
     * @param short_fecframe True for short FECFRAME.
     * @param has_pilots True to insert pilot blocks.
     */
    void set_modcod(uint8_t modcod, bool short_fecframe, bool has_pilots);

    /**
     * @brief Generate a dummy PLFRAME (MODCOD=0, 36 slots, no pilots).
     *
     * Payload symbols are (1/sqrt2, 1/sqrt2) before scrambling
     * (EN 302 307-1 clause 5.5.1).
     * Used to keep the RF signal active when no user data is available.
     * The RX recognizes dummy frames via the PLSC and discards them.
     *
     * @param plframe_out Output buffer (must hold at least dummy_plframe_len() symbols).
     * @return Number of symbols written.
     */
    int generate_dummy_frame(complexf* plframe_out);

    /** Length of a dummy PLFRAME in symbols. */
    static constexpr int dummy_plframe_len() { return pl::kHeaderLen + pl::kDummySlots * pl::kSlotLen; }

    uint16_t plframe_len() const { return (uint16_t)d_format.plframe_symbols(); }
    uint16_t xfecframe_len() const { return (uint16_t)d_format.data_symbols(); }
    uint16_t payload_len() const { return (uint16_t)d_format.payload_symbols(); }
    const pl_layout& format() const { return d_format; }

    /** Access PLHEADER symbols (90 complex values, SOF+PLSC). */
    const complexf* plheader() const { return d_plheader.data(); }

    /** Access PL descrambling sequence for pipeline init. */
    const complexf* descrambling_seq() const { return d_descrambling_seq.data(); }

private:
    pl_layout d_format;
    std::vector<complexf> d_descrambling_seq; // pl::max_payload_symbols()
    std::vector<complexf> d_plheader; // 90 symbols (SOF + PLSC)
    cuda::pl_scramble_cuda* d_scramble_cuda;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_PL_FRAMER_H */
