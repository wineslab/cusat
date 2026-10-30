#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Configure the host-side cuSAT TAP address and permanent peer neighbor.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sudo ./scripts/setup-network.sh node0 [tap-device]
       sudo ./scripts/setup-network.sh node1 [tap-device]

The default TAP is dvbs2a0 for node0 and dvbs2b0 for node1. The optional
tap-device argument is only needed when radio.tap was changed in node.yaml.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

if (( EUID != 0 )); then
    echo "error: run this script with sudo" >&2
    exit 1
fi

case "$1" in
    node0)
        default_tap=dvbs2a0
        local_cidr=10.30.0.1/30
        peer_ip=10.30.0.2
        peer_mac=02:00:00:00:00:02
        ;;
    node1)
        default_tap=dvbs2b0
        local_cidr=10.30.0.2/30
        peer_ip=10.30.0.1
        peer_mac=02:00:00:00:00:01
        ;;
    *)
        echo "error: role must be node0 or node1" >&2
        usage >&2
        exit 2
        ;;
esac

tap=${2:-$default_tap}

if ! ip link show dev "$tap" >/dev/null 2>&1; then
    echo "error: TAP '$tap' does not exist" >&2
    echo "Start the cuSAT Compose service for $1, then run this command again." >&2
    exit 1
fi

ip addr replace "$local_cidr" dev "$tap"
ip link set dev "$tap" up
ip neigh replace "$peer_ip" \
    lladdr "$peer_mac" nud permanent dev "$tap"

echo "cuSAT network configured for $1 on $tap"
ip -br addr show dev "$tap"
ip neigh show "$peer_ip" dev "$tap"
