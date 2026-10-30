/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Abstract IQ source interface for the standalone receiver pipeline.
 * Implementations provide IQ samples from files, pipes, or radio hardware.
 */

#ifndef INCLUDED_CUSAT_DSP_IQ_SOURCE_H
#define INCLUDED_CUSAT_DSP_IQ_SOURCE_H

#include <cusat_dsp/compat.h>

namespace cusat_dsp {

class iq_source
{
public:
    virtual ~iq_source() = default;

    /**
     * @brief Read up to max_samples complex IQ samples.
     *
     * @param buf Output buffer for complex samples.
     * @param max_samples Maximum number of samples to read.
     * @return Number of samples read. 0 on EOF or error.
     */
    virtual int read(complexf* buf, int max_samples) = 0;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_IQ_SOURCE_H */
