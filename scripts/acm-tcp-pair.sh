#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# ====================================================================
# acm-tcp-pair.sh — two-modem ACM run over the TCP radio
#
# Two dvbs2-gse-acm instances cross-connected through radio_tcp on
# localhost, each with its TAP moved into its own network namespace
# (10.99.0.1 <-> 10.99.0.2). The channel emulator of each modem (NOISE
# mode, TX side) is driven with channel-ctl.py along a triangular Es/N0
# trajectory high -> low -> high spanning the MODCOD table, while iperf3
# carries IP traffic over the DVB link.
#
# NOISE mode: noise std = signal rms x 10^(-(30 - 2x)/20) applied to a
# complex noise buffer of variance 2 per sample (x = dB command), so the
# per-sample SNR is 26.99 - 2x dB; at 4.096 samples/symbol
# Es/N0 = 33.11 - 2x dB, i.e. x = (33.11 - Es/N0) / 2 (checked against the
# receivers' pilot SNR). The lowest ACM MODCOD (qpsk13/45) has min_snr
# 0.57 dB, so keep esn0_lo >= ~1.5 dB.
#
# Usage: scripts/acm-tcp-pair.sh <build_dir> <minutes> <esn0_hi> <esn0_lo> <out_dir>
# ====================================================================
set -u
SRCDIR="$(cd "$(dirname "$0")/.." && pwd)"
B="${1:-build}"; [ "${B:0:1}" = / ] || B="$SRCDIR/$B"
MIN="${2:-30}"; HI="${3:-26}"; LO="${4:-1.5}"; OUT="${5:-/tmp/acmpair}"
mkdir -p "$OUT"
APP="$B/apps/dvbs2-gse-acm"
COMMON=(--rate 20.48e6 --sym-rate 5e6 --modcod qpsk1/4 --noise --log --gold-code 0)

cleanup() {
    pkill -P $$ 2>/dev/null
    [ -n "${PA:-}" ] && kill "$PA" 2>/dev/null
    [ -n "${PB:-}" ] && kill "$PB" 2>/dev/null
    for ns in cusatA cusatB; do ip netns pids "$ns" 2>/dev/null | xargs -r kill 2>/dev/null; done
    ip netns del cusatA 2>/dev/null; ip netns del cusatB 2>/dev/null
}
trap cleanup EXIT
ip netns del cusatA 2>/dev/null; ip netns del cusatB 2>/dev/null
ip netns add cusatA; ip netns add cusatB

# every log line is prefixed with the wall time it was read (ms), so the
# counters of the two nodes can be aligned
stamp() { python3 -u -c '
import sys, time
for l in sys.stdin:
    sys.stdout.write("%.3f %s" % (time.time(), l))' > "$1"; }
"$APP" "${COMMON[@]}" --usrp "tx=127.0.0.1:7101,rx_listen=7100" --tap dvbsA \
    --noise-port 5571 > >(stamp "$OUT/nodeA.log") 2>&1 &
PA=$!
sleep 1
"$APP" "${COMMON[@]}" --usrp "tx=127.0.0.1:7100,rx_listen=7101" --tap dvbsB \
    --noise-port 5572 > >(stamp "$OUT/nodeB.log") 2>&1 &
PB=$!

# move the TAPs into the namespaces once they exist
for i in $(seq 1 60); do
    ip link show dvbsA >/dev/null 2>&1 && ip link show dvbsB >/dev/null 2>&1 && break
    sleep 1
done
ip link set dvbsA netns cusatA && ip netns exec cusatA ip addr add 10.99.0.1/24 dev dvbsA
ip link set dvbsB netns cusatB && ip netns exec cusatB ip addr add 10.99.0.2/24 dev dvbsB
ip netns exec cusatA ip link set dvbsA up; ip netns exec cusatB ip link set dvbsB up
ip netns exec cusatA ip link set lo up; ip netns exec cusatB ip link set lo up
# The modem carries IPv4 only (ARP is not forwarded; the receiver rewrites the
# destination MAC to its own TAP), so install static neighbour entries.
ip netns exec cusatA ip neigh replace 10.99.0.2 lladdr 02:00:00:00:00:02 dev dvbsA nud permanent
ip netns exec cusatB ip neigh replace 10.99.0.1 lladdr 02:00:00:00:00:01 dev dvbsB nud permanent

set_esn0() { # both directions
    local x; x=$(python3 -c "print(round((33.11 - $1) / 2.0, 3))")
    python3 "$SRCDIR/scripts/channel-ctl.py" --pathloss-db "$x" 127.0.0.1:5571 127.0.0.1:5572 > /dev/null 2>&1
}
set_esn0 "$HI"
sleep 20

# IP traffic: iperf3 UDP 2 Mb/s each way for the whole run + periodic ping
ip netns exec cusatB iperf3 -s -D -p 5201 --logfile "$OUT/iperf_srvB.log"
ip netns exec cusatA iperf3 -s -D -p 5202 --logfile "$OUT/iperf_srvA.log"
sleep 1
DUR=$((MIN * 60))
ip netns exec cusatA iperf3 -c 10.99.0.2 -p 5201 -u -b 2M -t "$DUR" -i 10 > "$OUT/iperf_AtoB.log" 2>&1 &
ip netns exec cusatB iperf3 -c 10.99.0.1 -p 5202 -u -b 2M -t "$DUR" -i 10 > "$OUT/iperf_BtoA.log" 2>&1 &
ip netns exec cusatA ping -i 1 -w "$DUR" 10.99.0.2 > "$OUT/ping.log" 2>&1 &

# triangular trajectory, one step per 2 s, logged with wall time
STEPS=$((MIN * 30))
: > "$OUT/trajectory.log"
for k in $(seq 0 "$STEPS"); do
    e=$(python3 -c "
import sys
k, n, hi, lo = $k, $STEPS, $HI, $LO
half = n / 2
f = k / half if k <= half else (n - k) / half
print(round(hi - (hi - lo) * f, 2))")
    set_esn0 "$e"
    echo "$(date +%s.%N) $e" >> "$OUT/trajectory.log"
    sleep 2
done
sleep 30
kill "$PA" "$PB" 2>/dev/null; wait "$PA" "$PB" 2>/dev/null; sleep 1
python3 "$SRCDIR/scripts/acm-pair-report.py" "$OUT" > "$OUT/report.txt" 2>&1
cat "$OUT/report.txt"
echo "done"
