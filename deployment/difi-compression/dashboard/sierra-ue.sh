#!/bin/sh
# Sierra UE session on Spark .11 (cellular demo).
#   ./sierra-ue.sh start   attach the PDU session (DNN oai) inside netns "ue"
#   ./sierra-ue.sh test    ping the UPF gateway and fetch the data-network page
#   ./sierra-ue.sh status  modem registration, link and session state
#   ./sierra-ue.sh stop    delete the session and put the UE in airplane mode
# Spark .11 also runs a local Open5GS UPF on ogstun 10.45.0.1/16, the same pool our
# core on .12 hands out, so the UE interface lives in its own namespace: its traffic
# must cross the radio and the DVB backhaul instead of the local ogstun.
# Root steps use the host-namespace container pattern (no sudo password needed).
set -eu
cd "$(dirname "$0")"
SOCK=.ssh11.sock
[ -S "$SOCK" ] || python3 -c 'import launcher; launcher.connect("11")'
root() { ssh -S "$SOCK" wineslab@192.168.50.11 "docker run --rm --pid host --network host --privileged --entrypoint nsenter cusat:local -t 1 -m -u -i -n -p bash -c $(printf %s "$1" | python3 -c 'import shlex,sys;print(shlex.quote(sys.stdin.read()))')"; }
AT='python3 -c "
import serial,sys,time
s=serial.Serial(\"/dev/ttyUSB0\",115200,timeout=1)
for c in sys.argv[1:]:
    s.reset_input_buffer();s.write((c+\"\r\n\").encode());time.sleep(0.8);o=b\"\"
    while s.in_waiting:o+=s.read(s.in_waiting);time.sleep(0.1)
    print(\" | \".join(l for l in o.decode(errors=\"replace\").split(\"\r\n\") if l.strip() and l.strip()!=c))
"'
# Between sessions: delete_pdu (stop file + flush wwan0), airplane (CFUN=0), then radio
# on again. Reusing a session without this leaves stale addresses and routes behind.
CLEAN="touch /tmp/stop.txt
for i in \$(seq 1 10); do systemctl is-active --quiet ue-mbim || break; sleep 1; done
systemctl stop ue-mbim 2>/dev/null || true; systemctl reset-failed ue-mbim 2>/dev/null || true
ip netns exec ue ip -4 addr flush dev wwan0 2>/dev/null || ip -4 addr flush dev wwan0 2>/dev/null || true
$AT 'AT+CFUN=0' >/dev/null
rm -f /tmp/stop.txt"
case "${1:-status}" in
start)
  root "set -e
$CLEAN
ip netns list | grep -qw ue || ip netns add ue
ip link show wwan0 >/dev/null 2>&1 && ip link set wwan0 netns ue || true
ip netns exec ue ip link set lo up
sleep 2
$AT 'AT+CFUN=1' >/dev/null
for i in \$(seq 1 60); do $AT 'AT!GSTATUS?' | grep -q 'MM5G state: *Registered' && break; sleep 1; done
systemd-run --unit=ue-mbim --description='Sierra MBIM session (netns ue)' -p NetworkNamespacePath=/run/netns/ue /usr/local/bin/lite-mbim-connection-manager -s /usr/local/settings.conf
for i in \$(seq 1 90); do ip netns exec ue ip -4 -br addr show wwan0 | grep -q 10.45 && break; sleep 1; done
ip netns exec ue ip route replace default via 10.45.0.1 dev wwan0
ip netns exec ue ip -br addr show wwan0" ;;
test)
  root "ip netns exec ue ping -c 6 -i 0.5 -W 3 10.45.0.1 | tail -2
ip netns exec ue curl --silent --max-time 20 -o /dev/null -w 'HTTP %{http_code} in %{time_total}s from the data network\n' http://10.200.0.2:8080/" ;;
status)
  root "$AT 'AT!UIMS?' 'AT+CIMI' 'AT!GSTATUS?' | tr '|' '\n' | grep -E 'UIMS|^ *0010|MM5G|RRC state|Tx Power|RSRP|SINR' ; systemctl is-active ue-mbim || true; ip netns exec ue ip -br addr show wwan0 2>/dev/null || echo 'no wwan0 in netns ue'" ;;
stop)
  root "$CLEAN
echo 'session deleted, UE in airplane mode'" ;;
*) echo "usage: $0 start|test|status|stop" >&2; exit 2 ;;
esac
