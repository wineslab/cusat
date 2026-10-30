/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * TX worker threads for dvbs2-uhd-trx: frame generation and UHD send.
 */

#pragma once

#include "trx/context.h"

/**
 * TX gen thread: PDU queue → GSE → FEC → PL frame → RRC → TX IQ ring.
 * Pipelined: generation runs ahead while the send thread drains the ring.
 */
void tx_gen_thread(trx_context& ctx);

/**
 * TX send thread: TX IQ ring → tx_stream->send().
 * send() blocks until UHD accepts, providing natural sample-rate pacing.
 */
void tx_send_thread(trx_context& ctx);
