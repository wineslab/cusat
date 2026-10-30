#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export PATH="/usr/sbin:/sbin:$PATH"

tap=${CUSAT_TAP:-dvbs2b0}
boat_ip=${CUSAT_BOAT_IP:-10.200.0.1}
shore_ip=${CUSAT_SHORE_IP:-10.200.0.2}
amf=${CORE_AMF_IP:?set CORE_AMF_IP in docker/site.env}
amf_port=${CORE_AMF_PORT:-38412}
smf=${CORE_SMF_PFCP_HOST:?set CORE_SMF_PFCP_HOST in docker/site.env}
smf_port=${CORE_SMF_PFCP_PORT:-8805}
dnn=${CORE_DNN:-oai}
ue_subnet=${CORE_UE_SUBNET:-10.45.0.0/16}
ue_gateway=${CORE_UE_GATEWAY:-10.45.0.1}
upfd=/opt/open5gs/bin/open5gs-upfd
cfg=/tmp/cusat-upf.yaml

smf_ip=$(getent ahostsv4 "$smf" | awk 'NR == 1 {print $1}')
test -n "$smf_ip" || { echo "[upf] cannot resolve SMF PFCP host $smf" >&2; exit 1; }
pfcp_bind=$(ip route get "$smf_ip" | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')
test -n "$pfcp_bind" || { echo "[upf] cannot select a source address for $smf_ip" >&2; exit 1; }

for _ in $(seq 1 150); do
  ip link show "$tap" &>/dev/null && break
  sleep 0.2
done
ip link show "$tap" &>/dev/null || {
  echo "[upf] $tap is absent; start shore DVB first" >&2; exit 1;
}

ip addr replace "${shore_ip}/30" dev "$tap"
ip link set "$tap" up
ip neigh replace "$boat_ip" lladdr 02:00:00:00:00:01 nud permanent dev "$tap"

if ! ip link show ogstun &>/dev/null; then
  ip tuntap add name ogstun mode tun
  ip link set ogstun up
  ip link set dev ogstun txqueuelen 4096
fi
ip addr replace "${ue_gateway}/${ue_subnet#*/}" dev ogstun
sysctl -w net.ipv4.ip_forward=1 >/dev/null

iptables -t nat -C POSTROUTING -s "$ue_subnet" ! -o ogstun -j MASQUERADE 2>/dev/null || \
  iptables -t nat -A POSTROUTING -s "$ue_subnet" ! -o ogstun -j MASQUERADE
# Keep the changing AMF destination in a private chain. Flushing this chain
# makes a restart pick up a new site.env without leaving an older DNAT rule in
# front of it.
iptables -t nat -N CUSAT_N2 2>/dev/null || true
iptables -t nat -F CUSAT_N2
iptables -t nat -C PREROUTING -d "$shore_ip" -p sctp --dport 38412 \
  -j CUSAT_N2 -m comment --comment cusat-backhaul 2>/dev/null || \
  iptables -t nat -I PREROUTING 1 -d "$shore_ip" -p sctp --dport 38412 \
  -j CUSAT_N2 -m comment --comment cusat-backhaul
iptables -t nat -A CUSAT_N2 -p sctp \
  -j DNAT --to-destination "${amf}:${amf_port}"
# Docker sets the host FORWARD policy to DROP. NGAP enters on the DVB TAP and
# leaves through the host uplink, so allow this one SCTP flow and its replies.
iptables -C FORWARD -i "$tap" -p sctp -d "$amf" --dport "$amf_port" \
  -j ACCEPT 2>/dev/null || \
  iptables -I FORWARD 1 -i "$tap" -p sctp -d "$amf" --dport "$amf_port" \
  -j ACCEPT
iptables -C FORWARD -o "$tap" -p sctp -d "$boat_ip" \
  -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || \
  iptables -I FORWARD 1 -o "$tap" -p sctp -d "$boat_ip" \
  -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -t nat -C POSTROUTING -d "$amf" -p sctp -j MASQUERADE \
  -m comment --comment cusat-backhaul 2>/dev/null || \
  iptables -t nat -A POSTROUTING -d "$amf" -p sctp -j MASQUERADE \
  -m comment --comment cusat-backhaul

sed \
  -e "s|@SMF@|$smf|g" \
  -e "s|@SMF_PORT@|$smf_port|g" \
  -e "s|@PFCP_BIND@|$pfcp_bind|g" \
  -e "s|@SHORE_IP@|$shore_ip|g" \
  -e "s|@DNN@|$dnn|g" \
  -e "s|@UE_GATEWAY@|$ue_gateway|g" \
  -e "s|@UE_SUBNET@|$ue_subnet|g" \
  /opt/cusat-runtime/upf.yaml.in > "$cfg"

echo "[upf] N2 ${shore_ip}:38412 -> ${amf}:${amf_port}"
echo "[upf] N3 GTP-U is bound to ${shore_ip}:2152; PFCP ${pfcp_bind} -> ${smf}:${smf_port}"
exec "$upfd" -c "$cfg"
