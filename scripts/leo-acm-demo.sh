#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Accelerated, tutorial-sized LEO trace: three acquisition-safe visible arcs
# separated by deterministic five-second out-of-view intervals.  The 0.4 arc
# fraction starts each pass near 23 degrees elevation (~6.8 dB below zenith),
# where the validated radios can reacquire promptly after a hard outage.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec python3 "$script_dir/acm-pathloss-driver.py" \
  --regime leo \
  --passes 3 \
  --accel 10 \
  --arc-fraction 0.4 \
  --gap-sec 5 \
  --rate-hz 5 \
  "$@"
