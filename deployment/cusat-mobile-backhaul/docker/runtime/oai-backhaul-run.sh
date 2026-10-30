#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

src=/opt/cusat-oai-config/usrp_config_proxy.conf
cfg=/tmp/usrp_config_backhaul.live.conf
nr=/opt/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem
shlib=/opt/usrpproxy-build
tap=${CUSAT_TAP:-dvbs2a0}
boat_ip=${CUSAT_BOAT_IP:-10.200.0.1}
shore_ip=${CUSAT_SHORE_IP:-10.200.0.2}
usrp_addr=${CUSAT_USRP_ADDR:-192.168.40.20}

test -x "$nr" || { echo "[oai-backhaul] missing $nr" >&2; exit 1; }
test -f "$shlib/libusrpproxy.so" || {
  echo "[oai-backhaul] missing $shlib/libusrpproxy.so" >&2; exit 1;
}

for _ in $(seq 1 150); do
  ip link show "$tap" &>/dev/null && test -e /dev/shm/usrpproxy && break
  sleep 0.2
done
ip link show "$tap" &>/dev/null || {
  echo "[oai-backhaul] $tap is absent; start boat DVB first" >&2; exit 1;
}
test -e /dev/shm/usrpproxy || {
  echo "[oai-backhaul] usrpproxy shm is absent; start boat DVB first" >&2; exit 1;
}

ip addr replace "${boat_ip}/30" dev "$tap"
ip link set "$tap" up
ip neigh replace "$shore_ip" lladdr 02:00:00:00:00:02 nud permanent dev "$tap"
route=$(ip route get "$shore_ip")
grep -q "dev $tap" <<<"$route" || {
  echo "[oai-backhaul] route to shore does not use $tap: $route" >&2; exit 1;
}

cp "$src" "$cfg"
sed -i -E \
  -e "s|(shlibpath[[:space:]]*=[[:space:]]*)\"[^\"]*\"|\1\"${shlib}\"|" \
  -e "s|(amf_ip_address[^\"]*\")[0-9.]+(\")|\1${shore_ip}\2|" \
  -e "s|(GNB_IPV4_ADDRESS_FOR_NG_AMF[[:space:]]*=[[:space:]]*\")[0-9.]+(\")|\1${boat_ip}\2|" \
  -e "s|(GNB_IPV4_ADDRESS_FOR_NGU[[:space:]]*=[[:space:]]*\")[0-9.]+(\")|\1${boat_ip}\2|" \
  -e "s|(sdr_addrs[[:space:]]*=[[:space:]]*\")[^\"]*(\")|\1addr=${usrp_addr}\2|" \
  -e "s|(tracking_area_code[[:space:]]*=[[:space:]]*)[0-9]+|\1${TRACKING_AREA_CODE:-1}|" \
  -e "s|(mcc[[:space:]]*=[[:space:]]*)[0-9]+|\1${PLMN_MCC:-001}|" \
  -e "s|(mnc[[:space:]]*=[[:space:]]*)[0-9]+|\1${PLMN_MNC:-05}|" \
  -e "s|(mnc_length[[:space:]]*=[[:space:]]*)[0-9]+|\1${PLMN_MNC_LENGTH:-2}|" \
  "$cfg"

echo "[oai-backhaul] cpuset=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
echo "[oai-backhaul] N2 ${boat_ip} -> ${shore_ip}:38412 over $tap"
echo "[oai-backhaul] N3 gNB=${boat_ip}; edge UPF=${shore_ip} over $tap"
exec "$nr" -O "$cfg" "$@"
