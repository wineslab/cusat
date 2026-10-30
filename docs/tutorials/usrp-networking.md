# Make a network USRP accessible

cuSAT does not require a specified host interface, IP subnet, MTU, or VLAN.
UHD must be able to find and open the USRP. This access must work from the
host and from the cuSAT container.

Use this guide only for a network USRP. A USB radio does not need a host
network or VLAN configuration. Its device files must be available to Docker.

## Use an untagged USRP network

If the switch sends untagged USRP traffic to the host, use this procedure.

Assign an unused address from the radio subnet to the physical interface. Use
the normal network configuration method for your operating system. Do not
make a VLAN interface.

For example, a radio has the address `192.168.40.20/24`. The host can use the
unused address `192.168.40.64/24`. Each host and radio on this subnet must
have a unique address.

## Optional tagged VLAN

Use this procedure only if the switch sends tagged USRP traffic to the host.
If the switch removes the tag, use the untagged procedure.

The following configuration is temporary. It supports a first test on all
three documented Linux platforms. Replace the four placeholder values before
you run it:

```bash
PARENT=<physical-usrp-interface>
VLAN_ID=<vlan-id>
HOST_CIDR=<unused-host-address/prefix>
USRP_IP=<usrp-address>
VLAN_IF="${PARENT}.${VLAN_ID}"

if ! ip link show "$VLAN_IF" >/dev/null 2>&1; then
  sudo ip link add link "$PARENT" name "$VLAN_IF" type vlan id "$VLAN_ID"
fi
sudo ip addr replace "$HOST_CIDR" dev "$VLAN_IF"
sudo ip link set "$PARENT" up
sudo ip link set "$VLAN_IF" up
ip -br addr show "$VLAN_IF"
ip route get "$USRP_IP"
ping -c 3 "$USRP_IP"
```

The tested laboratory VLAN uses these example values:

```bash
VLAN_ID=21
HOST_CIDR=192.168.40.64/24
USRP_IP=192.168.40.20
```

The value of `PARENT` is different on each host. Use a different unused
`HOST_CIDR` on each host.

A host restart removes this configuration. If an equivalent connection
exists, do not make a second persistent connection.

To remove the temporary VLAN interface, use this command:

```bash
sudo ip link delete "$VLAN_IF"
```

### Make the configuration persistent with NetworkManager

Use the values from the temporary test. Make the NetworkManager connection
one time:

```bash
CONNECTION="cusat-usrp-vlan${VLAN_ID}"
sudo ip link delete "$VLAN_IF" 2>/dev/null || true
sudo nmcli connection add type vlan \
  con-name "$CONNECTION" ifname "$VLAN_IF" \
  dev "$PARENT" id "$VLAN_ID" \
  ipv4.method manual ipv4.addresses "$HOST_CIDR" \
  ipv4.never-default yes ipv6.method disabled
sudo nmcli connection up "$CONNECTION"
ip -br addr show "$VLAN_IF"
```

If the connection exists, `nmcli` reports an error. Examine and use the
existing connection. Do not add a duplicate connection.

### Make the configuration persistent with Netplan

Add the following content to the existing Netplan configuration. Replace the
uppercase placeholder values. Do not declare the same physical interface in
two Netplan files.

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    PARENT_INTERFACE: {}
  vlans:
    cusat-usrp:
      id: VLAN_ID
      link: PARENT_INTERFACE
      dhcp4: false
      addresses:
        - HOST_ADDRESS/PREFIX
```

Validate the configuration before you apply it. This check is important on a
remote host.

```bash
sudo netplan generate
sudo netplan try
ip -br addr show cusat-usrp
```

## Verify radio access with UHD

IP connectivity is not sufficient. UHD must find and open the correct radio.
After you build the `cusat:local` image, run these commands:

```bash
sudo docker run --rm --runtime=nvidia --network host --privileged \
  -v /dev:/dev cusat:local \
  uhd_find_devices --args "addr=$USRP_IP"
sudo docker run --rm --runtime=nvidia --network host --privileged \
  -v /dev:/dev cusat:local \
  uhd_usrp_probe --args "addr=$USRP_IP"
```

A platform guide can ask for the radio interface during IRQ configuration.
Supply the physical parent interface. Do not supply the VLAN interface.
