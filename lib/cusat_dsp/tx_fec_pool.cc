/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tx_fec_pool.h"
#include <cstdio>
#include <stdexcept>

namespace cusat_dsp {

tx_fec_pool::tx_fec_pool(fecframe_size_t framesize,
                         const std::vector<uint8_t>& enabled_modcods,
                         float rolloff)
    : d_framesize(framesize), d_rolloff(rolloff)
{
    for (uint8_t mc_idx : enabled_modcods) {
        const modcod_def_t* def = find_modcod(mc_idx);
        if (!def)
            throw std::invalid_argument("Unknown modcod_idx: " + std::to_string(mc_idx));
        d_enabled_set.insert(mc_idx);
        d_enabled.push_back(*def);
    }
}

tx_fec_chain& tx_fec_pool::get(uint8_t modcod_idx)
{
    auto it = d_chains.find(modcod_idx);
    if (it != d_chains.end())
        return *it->second;

    // Lazy build
    if (d_enabled_set.count(modcod_idx) == 0)
        throw std::out_of_range("MODCOD " + std::to_string(modcod_idx) + " not in pool");

    const modcod_def_t* def = find_modcod(modcod_idx);
    fprintf(stderr, "tx_fec_pool: building chain for %s (modcod_idx=%d)\n",
            def->name, modcod_idx);

    auto chain = std::make_unique<tx_fec_chain>(
        d_framesize, def->code_rate, def->constellation);
    chain->set_acm_mode(true);
    chain->set_rolloff(d_rolloff);

    auto& ref = *chain;
    d_chains[modcod_idx] = std::move(chain);
    return ref;
}

} // namespace cusat_dsp
