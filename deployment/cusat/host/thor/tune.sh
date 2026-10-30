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

declare -a cpus=()
IFS=',' read -r -a cpu_parts <<< "$network_cpus"
for part in "${cpu_parts[@]}"; do
  if [[ $part =~ ^([0-9]+)-([0-9]+)$ ]]; then
    if (( BASH_REMATCH[1] > BASH_REMATCH[2] )); then
      echo "invalid descending CPU range: $part" >&2
      exit 2
    fi
    mapfile -t expanded < <(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
    cpus+=("${expanded[@]}")
  elif [[ $part =~ ^[0-9]+$ ]]; then
    cpus+=("$part")
  else
    echo "invalid CPU list: $network_cpus" >&2
    exit 2
  fi
done

(( ${#cpus[@]} > 0 )) || { echo "the network CPU list is empty" >&2; exit 2; }
for cpu in "${cpus[@]}"; do
  cpu_dir="/sys/devices/system/cpu/cpu$cpu"
  test -d "$cpu_dir"
  if [[ -f "$cpu_dir/online" ]] && [[ $(<"$cpu_dir/online") != 1 ]]; then
    echo "CPU $cpu is offline" >&2
    exit 1
  fi
done

sysctl -w net.core.wmem_max=250000000
sysctl -w net.core.wmem_default=250000000
sysctl -w net.core.rmem_max=250000000
sysctl -w net.core.rmem_default=250000000
systemctl stop irqbalance 2>/dev/null || true

# The Thor nvethernet TX offload path produces UHD send stalls under sustained
# full-duplex X410 traffic. Keep VLAN insertion enabled, but let UHD provide
# complete UDP frames to the driver.
if ! command -v ethtool >/dev/null 2>&1; then
  echo "ethtool is required for the Thor Ethernet profile" >&2
  exit 1
fi
ethtool -K "$nic" tx off sg off gso off tso off

mapfile -t irqs < <(awk -v nic="$nic" '$0 ~ nic {gsub(":", "", $1); print $1}' /proc/interrupts)
if (( ${#irqs[@]} == 0 )); then
  echo "no IRQs found for $nic" >&2
  exit 1
fi

queue_index=0
for irq in "${irqs[@]}"; do
  irq_description=$(awk -v irq="$irq:" '$1 == irq {$1=""; print}' /proc/interrupts)
  if [[ $irq_description == *common_irq* ]]; then
    cpu=${cpus[0]}
  else
    cpu=${cpus[queue_index % ${#cpus[@]}]}
    ((queue_index += 1))
  fi
  printf '%s' "$cpu" > "/proc/irq/$irq/smp_affinity_list"
done

echo "$nic IRQ queues distributed across CPUs $network_cpus: ${irqs[*]}"
for irq in "${irqs[@]}"; do
  printf 'IRQ %s -> ' "$irq"
  cat "/proc/irq/$irq/smp_affinity_list"
done
ethtool -k "$nic" | grep -E \
  '^(tx-checksumming|scatter-gather|tcp-segmentation-offload|generic-segmentation-offload|tx-vlan-offload):'
