/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * RX worker threads for dvbs2-uhd-trx: UHD receive, frontend DSP and FEC.
 */

#pragma once

#include "trx/context.h"

/** RX UHD thread: rx_stream->recv() → RX ring. */
void rx_uhd_thread(trx_context& ctx);

/** RX frontend thread: RX ring → receiver_frontend → frame_queue. */
void rx_frontend_thread(trx_context& ctx);

/** RX FEC thread: frame_queue → phase → demap+LDPC → BCH → GSE → TAP. */
void rx_fec_thread(trx_context& ctx);
