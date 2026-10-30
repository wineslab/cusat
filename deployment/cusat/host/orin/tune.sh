#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Validate the Orin radio-interface IRQ partition and apply socket-buffer tuning.
# Some Jetson Ethernet drivers manage MSI affinity themselves and reject procfs
# writes, so irqaffinity=0-1 in the boot profile supplies the enforced placement.

set -euo pipefail

if (( EUID != 0 )); then
  echo "run as root: sudo $0 INTERFACE" >&2
  exit 1
fi

if (( $# < 1 || $# > 2 )); then
  echo "usage: sudo $0 PHYSICAL_INTERFACE [RADIO_INTERFACE]" >&2
  exit 2
fi

nic=$1
radio_nic=${2:-$nic}
test -d "/sys/class/net/$nic"
test -d "/sys/class/net/$radio_nic"

# The tested 1 Gb/s r8168 transport cannot sustain the packet rate of the
# full-duplex sc16 stream with standard Ethernet frames.  The Orin scenario
# files use 8000-byte UHD frames, so configure the complete host-side path for
# jumbo packets.  RADIO_INTERFACE is the VLAN interface when one is present.
ip link set "$nic" mtu 9000
if [[ $radio_nic != "$nic" ]]; then
  ip link set "$radio_nic" mtu 9000
fi
echo "$nic MTU $(<"/sys/class/net/$nic/mtu"); $radio_nic MTU $(<"/sys/class/net/$radio_nic/mtu")"

sysctl -w net.core.wmem_max=250000000
sysctl -w net.core.wmem_default=250000000
sysctl -w net.core.rmem_max=250000000
sysctl -w net.core.rmem_default=250000000
systemctl stop irqbalance 2>/dev/null || true

mapfile -t irqs < <(awk -v nic="$nic" '$0 ~ nic {gsub(":", "", $1); print $1}' /proc/interrupts)
if (( ${#irqs[@]} == 0 )); then
  echo "no IRQs found for $nic" >&2
  exit 1
fi

for irq in "${irqs[@]}"; do
  allowed=$(<"/proc/irq/$irq/smp_affinity_list")
  case $allowed in
    0|1|0-1|0,1) ;;
    *)
      echo "IRQ $irq for $nic can run outside CPUs 0-1 ($allowed)" >&2
      echo "apply the Orin boot profile and reboot before running cuSAT" >&2
      exit 1
      ;;
  esac
  effective=$(<"/proc/irq/$irq/effective_affinity_list")
  echo "$nic IRQ $irq allowed on CPUs $allowed; effective=${effective:-driver-managed}"
done
