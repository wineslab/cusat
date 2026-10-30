#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( EUID != 0 )); then
  echo "run as root: sudo $0 INTERFACE [NETWORK-CPUS]" >&2
  exit 1
fi

if (( $# < 1 || $# > 2 )); then
  echo "usage: sudo $0 INTERFACE [NETWORK-CPUS]" >&2
  exit 2
fi

nic=$1
network_cpus=${2:-0-3}

test -d "/sys/class/net/$nic"
sysctl -w net.core.wmem_max=250000000
sysctl -w net.core.wmem_default=250000000
sysctl -w net.core.rmem_max=250000000
sysctl -w net.core.rmem_default=250000000
systemctl stop irqbalance 2>/dev/null || true

affinity_tool=$(command -v set_irq_affinity_cpulist.sh || true)
if [[ -z "$affinity_tool" ]]; then
  echo "set_irq_affinity_cpulist.sh is required for the ConnectX NIC" >&2
  exit 1
fi
"$affinity_tool" "$network_cpus" "$nic"
