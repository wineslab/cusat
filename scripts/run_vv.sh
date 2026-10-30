#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run dvbs2x-vv-check over the regular-frame DVB-S2X cases of the DVB.org
# V&V package (TM5859). The package is distributed by DVB and is not part
# of this repository.
#
# Usage: scripts/run_vv.sh <build_dir> <vv_dir> [--gpu-bch]
#   <vv_dir> holds the extracted S2X*/ test case directories.
set -u
B=$1; VV=$2
CHECK=$B/apps/dvbs2x-vv-check
rc=0
for d in "$VV"/S2X*/; do
    d=${d%/}; name=$(basename "$d")
    extra=("${@:3}")
    # Shipped without its configuration file; scrambling code recovered
    # from PLFRAME / CLEARPLFR.
    [ "$name" = S2X_16APSK_with_pilots ] && extra+=(--gold 32)
    if "$CHECK" "$d" "${extra[@]}" > "$VV/$name.log" 2>/dev/null; then
        status=ok
    else
        status=FAIL; rc=1
    fi
    printf "%-30s %-4s %s\n" "$name" "$status" "$(grep -E '^  frames' "$VV/$name.log" | sed 's/^ *frames *//')"
done
exit $rc
