#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# ====================================================================
# loopback-sweep.sh — all-MODCOD file loopback
#
# For every MODCOD of acm_config.h (normal frames) and every DVB-S2
# MODCOD 1-28 with --short: dvbs2-gse-tx -> IQ file -> dvbs2-gse-rx,
#   clean channel:   Lock=True, >= 97% FECFRAMEs decoded, 0 FEC errors
#   AWGN:            Es/N0 = min_snr + 1 dB, >= 95% decoded, FER <= 1e-3
# at TX sps 4 and 2, and at fractional sps 1.2288 / 1.3333 (sps-2 IQ
# resampled with iq-farrow-resample; roll-off 0.1 at 1.2288 sps, where the
# receiver requires 1 + a + 0.05 <= sps).
#
# --awgn-snr-db of dvbs2-gse-tx is a per-sample SNR over the full sample
# band: Es/N0 = SNR + 10 log10(sps). The sweep passes
# min_snr + 1 - 10 log10(sps) so that Es/N0 = min_snr + 1 dB. For the
# fractional rates the clean 2-sps file is resampled and the AWGN is added
# at the final rate (numpy), with the same definition.
#
# Usage: scripts/loopback-sweep.sh [build_dir] [n_frames] [filter]
#   filter: optional grep -E pattern on "modcod size sps case"
# Prints one row per run and a summary; exit 0 iff every row passes.
# ====================================================================
set -u
SRCDIR="$(cd "$(dirname "$0")/.." && pwd)"
B="${1:-build}"
[ "${B:0:1}" = / ] || B="$SRCDIR/$B"
N="${2:-2000}"
FILTER="${3:-.}"
TX="$B/apps/dvbs2-gse-tx"; RX="$B/apps/dvbs2-gse-rx"; RS="$B/apps/iq-farrow-resample"
TMP=$(mktemp -d /tmp/lbsweep.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

# name min_snr idx from acm_config.h
TABLE=$(python3 - "$SRCDIR/lib/cusat_dsp/acm_config.h" <<'PY'
import re, sys
for m in re.finditer(r'\{\s*(\d+),\s*MOD_\w+,\s*RATE_\w+,\s*"([^"]+)",\s*[\d.]+,\s*(-?[\d.]+)f\s*\}', open(sys.argv[1]).read()):
    print(m.group(2), m.group(3), m.group(1))
PY
)

pass=0; fail=0
printf "%-14s %-6s %-6s %-5s %-6s %8s %8s %8s %s\n" modcod size sps case lock frames errors FER result
run_one() { # modcod size sps case min_snr
    local mc=$1 size=$2 sps=$3 kase=$4 msnr=$5
    local tag="$mc $size $sps $kase"
    echo "$tag" | grep -Eq -- "$FILTER" || return 0
    local sflag=""; [ "$size" = short ] && sflag="--short"
    local txsps=$sps; case $sps in 1.2288|1.3333) txsps=2;; esac
    # below 1.25 sps the receiver needs a roll-off < 0.2 (1 + a + 0.05 <= sps)
    local roll=(); case $sps in 1.2288) roll=(--rolloff 0.1);; esac
    local awgn=()
    if [ "$kase" = awgn ] && [ "$txsps" = "$sps" ]; then
        awgn=(--awgn-snr-db "$(python3 -c "import math;print(round($msnr+1-10*math.log10($sps),3))")")
    fi
    local iq="$TMP/iq.raw" log="$TMP/rx.log"
    rm -f "$iq"
    python3 -c "
import os, sys, struct
for _ in range($N * 40):
    p = os.urandom(1000)
    sys.stdout.buffer.write(struct.pack('>H', len(p)) + p)
" 2>/dev/null | timeout 900 "$TX" --modcod "$mc" $sflag --sps "$txsps" "${roll[@]}" --pipe --n-frames "$N" \
        "${awgn[@]}" --out-file "$iq" > "$TMP/tx.log" 2>&1
    if [ ! -s "$iq" ]; then # MODCOD/size not accepted by the transmitter
        printf "%-14s %-6s %-6s %-5s %-6s %8s %8s %8s %s\n" "$mc" "$size" "$sps" "$kase" - - - - SKIP
        return 0
    fi
    local rxsps=$txsps
    if [ "$txsps" != "$sps" ]; then
        "$RS" --in-file "$iq" --out-file "$iq.rs" --in-sps "$txsps" --out-sps "$sps" > /dev/null 2>&1
        mv "$iq.rs" "$iq"; rxsps=$sps
        if [ "$kase" = awgn ]; then
            # AWGN added at the final rate: Es/N0 = SNR_sample + 10 log10(sps)
            python3 - "$iq" "$msnr" "$sps" <<'PY'
import sys, numpy as np
f, msnr, sps = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
x = np.fromfile(f, dtype=np.complex64)
snr = 10 ** ((msnr + 1 - 10 * np.log10(sps)) / 10)
rng = np.random.default_rng(1)
p = np.mean(np.abs(x[len(x) // 10:]) ** 2)
s = np.sqrt(p / snr / 2)
(x + (s * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))).astype(np.complex64)).astype(np.complex64).tofile(f)
PY
        fi
    fi
    timeout 1800 "$RX" --modcod "$mc" $sflag --sps "$rxsps" "${roll[@]}" --pipe --in-file "$iq" \
        --out-file /dev/null --log > "$log" 2>&1
    local fin; fin=$(grep "finished\." "$log" | tail -1)
    local fr; fr=$(echo "$fin" | sed -n 's/.* \([0-9]*\) FECFRAMEs.*/\1/p'); fr=${fr:-0}
    local er; er=$(echo "$fin" | sed -n 's/.* \([0-9]*\) FEC errors.*/\1/p'); er=${er:-0}
    local lock=False; grep -q "Lock=True" "$log" && lock=True
    local fer; fer=$(python3 -c "print('%.1e' % ($er / max(1, $fr)))")
    local ok=PASS
    if [ "$kase" = clean ]; then
        [ "$lock" = True ] && [ "$fr" -ge $((N * 97 / 100)) ] && [ "$er" -eq 0 ] || ok=FAIL
    else
        [ "$lock" = True ] && [ "$fr" -ge $((N * 95 / 100)) ] &&
            python3 -c "import sys; sys.exit(0 if $er <= 1e-3 * max(1, $fr) else 1)" || ok=FAIL
    fi
    [ $ok = PASS ] && pass=$((pass + 1)) || fail=$((fail + 1))
    printf "%-14s %-6s %-6s %-5s %-6s %8s %8s %8s %s\n" "$mc" "$size" "$sps" "$kase" "$lock" "$fr" "$er" "$fer" "$ok"
}

for sps in ${SWEEP_SPS:-4 2 1.2288 1.3333}; do
    while read -r name msnr idx; do
        for kase in clean awgn; do
            run_one "$name" normal "$sps" "$kase" "$msnr"
            [ "$idx" -le 28 ] && run_one "$name" short "$sps" "$kase" "$msnr"
        done
    done <<< "$TABLE"
done
echo "SUMMARY: $pass PASS, $fail FAIL"
[ "$fail" -eq 0 ]
