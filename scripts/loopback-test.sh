#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# ====================================================================
# loopback-test.sh — GPU loopback: gse-tx → IQ file → gse-rx
#
# Runs wherever it's invoked (pod or local machine). Generates
# length-framed random PDUs (2B big-endian length + payload), encodes
# N frames to a temp IQ file, then decodes it and reports lock /
# FECFRAME counts.
#
# Usage: ./scripts/loopback-test.sh [modcod] [n_frames] [build_dir]
#   defaults: qpsk3/5, 2000, build
#
# Exit codes: 0 = lock + frames decoded, 1 = TX failed,
#             2 = RX no lock, 3 = locked but 0 FECFRAMEs (known
#                 master bug in the gse-rx decode path)
# ====================================================================
set -uo pipefail
SRCDIR="$(cd "$(dirname "$0")/.." && pwd)"
MODCOD="${1:-qpsk3/5}"
N_FRAMES="${2:-2000}"
BUILD_DIR="${3:-build}"
TX="$SRCDIR/$BUILD_DIR/apps/dvbs2-gse-tx"
RX="$SRCDIR/$BUILD_DIR/apps/dvbs2-gse-rx"
IQ=$(mktemp /tmp/loopback_iq.XXXXXX.raw)
RXLOG=$(mktemp /tmp/loopback_rx.XXXXXX.log)
trap 'rm -f "$IQ" "$RXLOG"' EXIT

[ -x "$TX" ] && [ -x "$RX" ] || { echo "FAIL: apps not built in $BUILD_DIR"; exit 1; }

echo "=== Loopback: $MODCOD, $N_FRAMES frames ==="
python3 -c "
import os, sys, struct
for _ in range(${N_FRAMES} * 20):
    p = os.urandom(1000)
    sys.stdout.buffer.write(struct.pack('>H', len(p)) + p)
" 2>/dev/null | timeout 180 "$TX" --modcod "$MODCOD" --pipe \
    --n-frames "$N_FRAMES" --out-file "$IQ" 2>&1 | tail -1

[ -s "$IQ" ] || { echo "FAIL: TX produced no IQ"; exit 1; }
echo "IQ file: $(du -h "$IQ" | cut -f1)"

timeout 300 "$RX" --modcod "$MODCOD" --pipe \
    --in-file "$IQ" --out-file /dev/null --log > "$RXLOG" 2>&1

FINISH=$(grep "finished\." "$RXLOG" | tail -1)
echo "$FINISH"
FECFRAMES=$(echo "$FINISH" | grep -oP '\d+(?= FECFRAMEs)' || echo 0)

if ! grep -q "Lock=True" "$RXLOG"; then
    echo "RESULT: NO LOCK"
    tail -3 "$RXLOG"
    exit 2
elif [ "${FECFRAMES:-0}" -eq 0 ]; then
    echo "RESULT: LOCKED but 0 FECFRAMEs decoded (known gse-rx decode-path bug)"
    exit 3
else
    echo "RESULT: PASS (lock + $FECFRAMES FECFRAMEs)"
    exit 0
fi
