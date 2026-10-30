#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Sum of the frontend trace column fe_timing_us (ms) per test file.
# Usage: scripts/fe-timing-bench.sh <build_dir> <file_dir> [modcod:sps ...]
#   Files are generated once in <file_dir> (gse-tx, 2000 frames, random
#   1000-byte GSE PDUs, clean channel).
set -u
B="$1"; D="$2"; shift 2
CASES="${*:-qpsk3/5:4 qpsk3/5:2 8psk3/5:4 8psk3/5:2 16apsk2/3:4 16apsk2/3:2 32apsk3/4:4 32apsk3/4:2 64apsk4/5:4 64apsk4/5:2 256apsk3/4:4}"
mkdir -p "$D"
for c in $CASES; do
    m=${c%:*}; s=${c#*:}; f="$D/$(echo "$m" | tr / _)_sps$s.iq"
    if [ ! -s "$f" ]; then
        python3 -c "
import os, sys, struct
for _ in range(2000 * 20):
    p = os.urandom(1000)
    sys.stdout.buffer.write(struct.pack('>H', len(p)) + p)
" 2>/dev/null | "$B/apps/dvbs2-gse-tx" --modcod "$m" --sps "$s" --pipe --n-frames 2000 --out-file "$f" > /dev/null 2>&1
    fi
    csv=$(mktemp /tmp/fet.XXXXXX.csv)
    "$B/apps/dvbs2-gse-rx" --modcod "$m" --sps "$s" --pipe --in-file "$f" --out-file /dev/null --log \
        --frontend-trace-csv "$csv" > "$csv.log" 2>&1
    fin=$(grep -o "[0-9]* FECFRAMEs [0-9]* errors" "$csv.log" | tail -1)
    python3 - "$csv" "$m" "$s" "$fin" <<'P'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
t = sum(float(r["fe_timing_us"]) for r in rows) / 1e3
g = sum(float(r.get("gpu_timing_us", 0) or 0) for r in rows) / 1e3
gi = sum(float(r.get("gpu_interp_us", 0) or 0) for r in rows) / 1e3
print(f"{sys.argv[2]:<11} sps {sys.argv[3]}: fe_timing {t:8.1f} ms  (gpu est {g:7.1f}, gpu interp {gi:7.1f}, calls {len(rows)})  {sys.argv[4]}")
P
    rm -f "$csv" "$csv.log"
done
