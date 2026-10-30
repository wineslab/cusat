/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TAP reader thread for dvbs2-uhd-trx.
 */

#pragma once

#include "trx/context.h"

/**
 * TAP reader thread: continuously drains TAP into a queue so PDUs
 * are immediately available to the TX thread (no wait during UHD send).
 */
void tap_reader_thread(trx_context& ctx);
