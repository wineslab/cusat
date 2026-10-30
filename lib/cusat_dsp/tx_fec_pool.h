/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TX FEC pool: one tx_fec_chain per enabled MODCOD for ACM transmission.
 * Chains are built lazily on first use to minimize memory/cache footprint.
 */

#ifndef INCLUDED_CUSAT_DSP_TX_FEC_POOL_H
#define INCLUDED_CUSAT_DSP_TX_FEC_POOL_H

#include "acm_config.h"
#include "tx_fec_chain.h"
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

namespace cusat_dsp {

class tx_fec_pool
{
public:
    /**
     * @brief Construct a TX FEC pool for the given MODCOD set.
     *
     * Chains are NOT built eagerly — they are created on first get() call.
     * This avoids allocating 15+ LDPC tables, BCH codecs, etc. at startup.
     *
     * @param standard DVB-S2 or DVB-S2X.
     * @param framesize Normal or short FECFRAME.
     * @param enabled_modcods List of modcod_idx values to support.
     */
    tx_fec_pool(fecframe_size_t framesize,
                const std::vector<uint8_t>& enabled_modcods,
                float rolloff = 0.2f);

    /** Get the FEC chain for a specific MODCOD index (builds lazily). */
    tx_fec_chain& get(uint8_t modcod_idx);

    /** Get the list of enabled MODCOD definitions. */
    const std::vector<modcod_def_t>& enabled_modcods() const { return d_enabled; }

private:
    fecframe_size_t d_framesize;
    float d_rolloff;
    std::set<uint8_t> d_enabled_set;
    std::unordered_map<uint8_t, std::unique_ptr<tx_fec_chain>> d_chains;
    std::vector<modcod_def_t> d_enabled;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_TX_FEC_POOL_H */
