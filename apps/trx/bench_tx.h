/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TX benchmark mode for dvbs2-uhd-trx.
 */

#pragma once

#include "trx/options.h"

/**
 * Run the TX benchmark (no UHD hardware). Returns the process exit code.
 */
int run_benchmark_tx(const trx_options& opt);
