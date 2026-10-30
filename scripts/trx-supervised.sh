#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# ====================================================================
# trx-supervised.sh — keep the DVB-S2X RX alive across GPU faults.
#
# A CUDA illegal-access (700) poisons the CUDA context and is unrecoverable
# in-process, so dvbs2-uhd-trx now exits cleanly with code 42 (CUDA_FAULT_EXIT_CODE)
# instead of abort(). This wrapper detects that and restarts the transceiver,
# which re-acquires the link ("reattach"). Any other exit (0 clean, 130 SIGINT,
# etc.) stops the loop.
#
# Usage:
#   [MAX_RESTARTS=20] [BACKOFF_S=2] [RADIO_RESET_CMD="..."] \
#   bash scripts/trx-supervised.sh <full dvbs2-uhd-trx command + args...>
#
# Example (inside a pod):
#   RADIO_RESET_CMD="ssh -o StrictHostKeyChecking=no root@192.168.40.18 systemctl restart usrp-hwd" \
#   bash scripts/trx-supervised.sh /workspace/dvb-s2x/build/apps/dvbs2-uhd-trx --usrp addr=... --acm ...
# ====================================================================
set -uo pipefail
FAULT_CODE="${FAULT_CODE:-42}"            # matches cuda::CUDA_FAULT_EXIT_CODE
MAX_RESTARTS="${MAX_RESTARTS:-20}"
BACKOFF_S="${BACKOFF_S:-2}"
RADIO_RESET_CMD="${RADIO_RESET_CMD:-}"    # optional: run before each restart

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <dvbs2-uhd-trx ...args>" >&2; exit 64
fi

n=0
while :; do
    echo "=== [trx-supervised] start attempt $((n+1))/$((MAX_RESTARTS+1)) $(date +%H:%M:%S)"
    "$@"
    rc=$?
    if [ "$rc" -eq "$FAULT_CODE" ]; then
        n=$((n+1))
        if [ "$n" -gt "$MAX_RESTARTS" ]; then
            echo "=== [trx-supervised] GPU fault, but exhausted $MAX_RESTARTS restarts — giving up" >&2
            exit "$rc"
        fi
        echo "=== [trx-supervised] GPU fault (code $rc) — reattaching in ${BACKOFF_S}s (restart $n/$MAX_RESTARTS)" >&2
        if [ -n "$RADIO_RESET_CMD" ]; then
            echo "=== [trx-supervised] resetting radio: $RADIO_RESET_CMD"
            eval "$RADIO_RESET_CMD" || echo "=== [trx-supervised] WARN radio reset failed"
        fi
        sleep "$BACKOFF_S"
        continue
    fi
    echo "=== [trx-supervised] exited with code $rc (not a GPU fault) — stopping"
    exit "$rc"
done
