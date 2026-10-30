/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL Framer implementation.
 *
 * PLFRAME structure:
 *   [SOF: 26 π/2 BPSK] [PLSC: 64 π/2 BPSK]
 *   [slot 0..15: 90 sym] [pilot: 36 sym]
 *   [slot 16..31: 90 sym] [pilot: 36 sym]
 *   ... [remaining slots]
 *
 * - PLHEADER (SOF + PLSC) is NOT scrambled.
 * - Payload (data slots + pilot blocks) IS scrambled with PL scrambling sequence.
 * - Tx scrambling = multiply by conj(Rx descrambling sequence).
 */

#include "pl_framer.h"
#include "framing/plheader.h"
#include "framing/pl_scrambler.h"

#include "cuda/pl_scramble_kernel.cuh"
#include <cstring>

namespace cusat_dsp {

pl_framer::pl_framer(uint8_t modcod, bool short_fecframe, bool has_pilots, int gold_code)
    : d_descrambling_seq(pl::max_payload_symbols()),
      d_plheader(pl::kHeaderLen)
{
    pl_descrambling_sequence(gold_code, d_descrambling_seq.data(), pl::max_payload_symbols());
    set_modcod(modcod, short_fecframe, has_pilots);

    d_scramble_cuda = new cuda::pl_scramble_cuda(
        d_descrambling_seq.data(), pl::max_payload_symbols());
    fprintf(stderr, "pl_framer: using CUDA PL scramble\n");
}

pl_framer::~pl_framer()
{
    delete d_scramble_cuda;
}

void pl_framer::set_modcod(uint8_t modcod, bool short_fecframe, bool has_pilots)
{
    d_format = pl_layout_for(modcod, short_fecframe ? FECFRAME_SHORT : FECFRAME_NORMAL,
                             has_pilots);
    plheader_symbols(d_format.pls_code, d_plheader.data());
}

void pl_framer::frame(const complexf* xfecframe_in, complexf* plframe_out)
{
    const uint16_t n_slots = d_format.slots;
    const bool has_pilots = d_format.pilots;
    const int n_pilots = d_format.pilot_block_count();

    // 1. Write PLHEADER (not scrambled)
    memcpy(plframe_out, d_plheader.data(), pl::kHeaderLen * sizeof(complexf));
    complexf* out = plframe_out + pl::kHeaderLen;

    // 2. Write payload: data slots interleaved with pilot blocks
    // Pilot reference symbol: (1+j)/sqrt(2)
    const complexf pilot_ref((float)M_SQRT1_2, (float)M_SQRT1_2);
    unsigned int xfec_idx = 0;  // index into xfecframe_in
    unsigned int payload_idx = 0; // index into payload (for scrambling)
    unsigned int slot_count = 0;

    for (unsigned int s = 0; s < n_slots; s++) {
        // Copy one slot (90 symbols) from XFECFRAME
        memcpy(out, &xfecframe_in[xfec_idx], pl::kSlotLen * sizeof(complexf));
        out += pl::kSlotLen;
        xfec_idx += pl::kSlotLen;
        payload_idx += pl::kSlotLen;
        slot_count++;

        // Insert pilot block after every 16 slots (if pilots enabled)
        if (has_pilots && (slot_count % pl::kSlotsPerPilotGroup == 0) &&
            ((int)(slot_count / pl::kSlotsPerPilotGroup) <= n_pilots)) {
            for (int p = 0; p < pl::kPilotBlockLen; p++) {
                out[p] = pilot_ref;
            }
            out += pl::kPilotBlockLen;
            payload_idx += pl::kPilotBlockLen;
        }
    }

    // 3. PL scramble the payload (everything after PLHEADER)
    complexf* payload = plframe_out + pl::kHeaderLen;
    d_scramble_cuda->scramble_inplace(payload, payload_idx);
}

int pl_framer::generate_dummy_frame(complexf* plframe_out)
{
    // Dummy PLFRAME: MODCOD=0, normal FECFRAME, no pilots
    plheader_symbols(pls_code_for(0, FECFRAME_NORMAL, false), plframe_out);

    // Payload: 36 slots of (I, Q) = (1/sqrt2, 1/sqrt2) (EN 302 307-1
    // clause 5.5.1), then PL scrambling
    complexf* payload = plframe_out + pl::kHeaderLen;
    const int payload_len = pl::kDummySlots * pl::kSlotLen; // 36 * 90 = 3240
    const complexf dummy_sym((float)M_SQRT1_2, (float)M_SQRT1_2);
    for (int i = 0; i < payload_len; i++)
        payload[i] = dummy_sym;

    // PL scramble
    d_scramble_cuda->scramble_inplace(payload, payload_len);

    return pl::kHeaderLen + payload_len;
}

} // namespace cusat_dsp
