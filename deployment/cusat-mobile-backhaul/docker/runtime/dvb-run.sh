#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cfg=${1:-/etc/cusat/node.yaml}
trx=${CUSAT_TRX:-/build/apps/dvbs2-uhd-trx}

test -r "$cfg" || { echo "[dvb-run] cannot read $cfg" >&2; exit 1; }
test -x "$trx" || { echo "[dvb-run] cannot execute $trx" >&2; exit 1; }

rm -f /dev/shm/usrpproxy /dev/shm/sem.usrpproxy* 2>/dev/null || true
echo "[dvb-run] cpuset=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
echo "[dvb-run] exec $trx --config $cfg"
exec "$trx" --config "$cfg"
