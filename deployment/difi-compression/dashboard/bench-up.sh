#!/bin/bash
# Bring the bench back after a reboot or power loss, ending with an idle dashboard:
# reopen the SSH control sockets, wait for links and radios, check that nothing is
# left running, then start the dashboard server. Click Start on the page afterwards.
#   ./bench-up.sh
# Both Sparks accept the laptop SSH key; SSHPASS remains an optional recovery fallback.
# No passwords are stored here.
set -uo pipefail
cd -- "$(dirname -- "$0")"
DASHBOARD='cd '"$PWD"' && python3 -u dashboard-server.py 2>&1 | tee -a logs/dashboard-server.log'

# The old server's loops keep opening SSH connections, which stall sshd while it boots.
tmux kill-session -t cusat-dashboard 2>/dev/null
pkill -f dashboard-server.py 2>/dev/null
sleep 1

connect() {  # $1 = 11 | 12
  local n=$1 sock=.ssh$1.sock host=wineslab@192.168.50.$1 i
  ssh -S $sock -O check $host >/dev/null 2>&1 && { echo ".$n: SSH session already open"; return 0; }
  ssh -S $sock -O exit $host >/dev/null 2>&1; rm -f $sock
  local opts=(-M -N -f -S $sock -o ControlPersist=48h -o ServerAliveInterval=10 -o ConnectTimeout=15)
  for i in $(seq 1 30); do
    if ssh "${opts[@]}" -o BatchMode=yes $host 2>/dev/null; then
      break
    elif [ -n "${SSHPASS:-}" ] && command -v sshpass >/dev/null; then
      sshpass -e ssh "${opts[@]}" -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 $host 2>/dev/null && break
    else
      ssh "${opts[@]}" $host && break
    fi
    [ $i = 1 ] && echo ".$n: waiting for SSH (the Spark may still be booting)"
    sleep 10
  done
  ssh -S $sock -O check $host >/dev/null 2>&1 && echo ".$n: SSH session open" || { echo ".$n: SSH unreachable after 5 min"; return 1; }
}

check() {  # $1 = 11 | 12; waits up to 5 min for the port-0 link and the X410's MPM
  local n=$1 radio nic i link info
  [ $n = 11 ] && { radio=192.168.40.18; nic=enp1s0f0np0; } || { radio=192.168.40.20; nic=aerial00; }
  for i in $(seq 1 30); do
    link=$(ssh -S .ssh$n.sock wineslab@192.168.50.$n "cat /sys/class/net/$nic/carrier 2>/dev/null")
    info=$(ssh -S .ssh$n.sock wineslab@192.168.50.$n "python3 - $radio" < mpm-info.py 2>/dev/null | tail -1)
    [ "$link" = 1 ] && [[ $info == *fpga_version_hash* ]] && break
    [ $i = 1 ] && echo ".$n: waiting for the port-0 link and X410 $radio"
    sleep 10
  done
  [ "$link" = 1 ] || echo ".$n: WARNING port-0 link $nic is down; check the QSFP cable"
  [[ $info == *fpga_version_hash* ]] && echo ".$n: X410 $radio up, image $(sed -E 's/.*"fpga_version_hash": "([^"]*)".*/\1/' <<<"$info")" \
                                     || echo ".$n: WARNING X410 $radio not answering; check it is powered"
  local left=$(ssh -S .ssh$n.sock wineslab@192.168.50.$n "docker ps --format '{{.Names}}' | grep cusat | tr '\n' ' '")
  local gpu=$(ssh -S .ssh$n.sock wineslab@192.168.50.$n "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l | tr -d ' '")
  echo ".$n: leftover containers: ${left:-none}; GPU processes: ${gpu:-?}"
}

connect 11 || exit 1
connect 12 || exit 1
check 11
check 12
ssh -S .ssh11.sock wineslab@192.168.50.11 'lsusb | grep -q 1199' && echo ".11: Sierra on USB" || echo ".11: WARNING Sierra not on USB (only needed for the cellular demo)"

tmux new-session -d -s cusat-dashboard -n dashboard "$DASHBOARD"
for i in $(seq 1 30); do
  mode=$(curl -s -m3 'localhost:8765/api/state?lite=1' | python3 -c "import json,sys; print(json.load(sys.stdin)['demo_mode']['active'])" 2>/dev/null) && break
  sleep 3
done
echo "Dashboard: http://localhost:8765 (mode: ${mode:-not answering; see tmux attach -t cusat-dashboard})"
