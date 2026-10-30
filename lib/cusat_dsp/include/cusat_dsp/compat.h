/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Core complex sample type and symbol-visibility macro used across the
 * library.
 */

#ifndef INCLUDED_CUSAT_DSP_COMPAT_H
#define INCLUDED_CUSAT_DSP_COMPAT_H

#include <complex>
#include <cmath>

// Complex baseband sample
typedef std::complex<float> complexf;

// Symbol-visibility annotation for public classes and functions. The library
// is built as a static archive, so it expands to nothing.
#ifndef CUSAT_DSP_API
#define CUSAT_DSP_API
#endif

#endif /* INCLUDED_CUSAT_DSP_COMPAT_H */
