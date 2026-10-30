# Connect two endpoints with CCM

Use this guide to connect two prepared cuSAT Docker hosts. Assign the node0
role to one host. Assign the node1 role to the other host. This procedure
tests a two-way DVB-S2X link with a fixed MODCOD.

The platform guides contain the host installation, kernel configuration,
radio network, and image-build procedures.

## 1. Prepare two supported hosts

Complete the applicable setup guide on each host:

- [Jetson Orin NX setup](setup-orin.md)
- [Jetson Thor setup](setup-thor.md)
- [DGX Spark setup](setup-spark.md)

Each host must have these items:

- a USRP that UHD can find.
- no other process that uses the USRP.
- the tested boot and IRQ configuration.
- `deployment/cusat/docker/platform.env` for its platform.
- a tested Docker image with the name `cusat:local`.

Any supported platform can have either role. This Docker procedure does not
include GH200. For GH200, use the [OpenShift guide](04-openshift.md). That
guide supplies the tested SM90, exclusive-CPU, SR-IOV, and Multus settings.

cuSAT uses UHD. It can operate with most UHD radios that supply the necessary
TX and RX channels and sample rate. Release tests and measurements cover only
USRP X410 radios.

For a different radio, examine all device-specific settings. These settings
include the UHD arguments, subdevice selection, channels, rates, and
transport. The maximum real-time performance of other radios is not
validated.

## 2. Select the node roles

The supplied files define this paired X410 FDD configuration:

| Setting | node0 | node1 |
|---|---:|---:|
| Tested USRP | X410 at `192.168.40.20` | X410 at `192.168.40.18` |
| TX frequency | 1.95 GHz | 2.05 GHz |
| RX frequency | 2.05 GHz | 1.95 GHz |
| TX logical channel | 1 (`A:1`) | 1 (`A:1`) |
| RX logical channel | 3 (`B:1`) | 3 (`B:1`) |
| TAP | `dvbs2a0` | `dvbs2b0` |
| TAP address | `10.30.0.1/30` | `10.30.0.2/30` |

The roles do not depend on the platform. For example, a Thor can be node0
and an Orin can be node1. An Orin can also be node0 and a Spark can be node1.

Connect the node0 TX port to the node1 RX port. Connect the node1 TX port to
the node0 RX port.

WARNING: For a conducted test, use sufficient attenuation. For an over-the-air
test, use only permitted frequencies and antennas.

## 3. Install one role on each host

On node0, copy the node0 file for its platform. This example uses Thor:

```bash
cd ~/cusat-release-test/dvb-s2x
cp deployment/cusat/configs/thor/ccm/node0.yaml \
  deployment/cusat/docker/node.yaml
${EDITOR:-nano} deployment/cusat/docker/node.yaml
```

On node1, copy the node1 file for its platform. This example uses Orin:

```bash
cd ~/cusat-release-test/dvb-s2x
cp deployment/cusat/configs/orin/ccm/node1.yaml \
  deployment/cusat/docker/node.yaml
${EDITOR:-nano} deployment/cusat/docker/node.yaml
```

Use `spark` in the path for a Spark host. Do not copy the node0 file to both
hosts. If both hosts use node0, the frequencies and TAP interfaces do not make
a valid pair.

## 4. Examine the important CCM settings

The supplied files give a conservative X410 QPSK 1/4 test. An X410 user
usually changes only the device address and the RF channels. A user of a
different USRP must examine all radio settings.

### UHD device and FDD frequencies

```yaml
radio:
  usrp: "addr=192.168.40.20"
  tx_freq: 1.95e9
  rx_freq: 2.05e9
```

The `usrp` field accepts UHD device arguments. For a network radio, an example
is `addr=192.168.40.20`. For one USB device, an example is
`type=b200,serial=...`.

The frequency settings must make a pair:

- node0 TX frequency equals node1 RX frequency.
- node1 TX frequency equals node0 RX frequency.

### RF-channel assignment

```yaml
radio:
  tx_channel: 1
  rx_channel: 3
  subdev_spec: "A:0 A:1 B:0 B:1"
```

For an X410, the supplied `subdev_spec` gives this map:

| Logical channel | Physical RF chain |
|---:|---|
| 0 | `A:0` |
| 1 | `A:1` |
| 2 | `B:0` |
| 3 | `B:1` |

Keep this full X410 order. Use `tx_channel` and `rx_channel` to select the
connected ports. At startup, cuSAT prints the selected map. Connect each TX
port to the selected RX port on the other radio.

For a different USRP, first run `uhd_usrp_probe`. Select only channels that
the radio supplies. Many radios supply only channel 0. Use the correct UHD
subdevice specification for that radio. Alternatively, remove `subdev_spec`
to use the UHD default.

cuSAT stops if a selected channel is outside the number of channels that UHD
reports.

### Sample rate and symbol rate

```yaml
radio:
  rate: 20.48e6
  sym_rate: 5e6
```

The `rate` field is the USRP sample rate. The `sym_rate` field is the DVB
symbol rate. The ratio is the number of hardware samples per symbol. The
conservative configuration uses 4.096 samples per symbol.

The mobile-backhaul guide uses a separate, higher-rate profile on its tested
10 Gb/s hosts.

Both hosts must use the same sample rate and symbol rate. Both radios must
support the sample rate. A higher rate requires sufficient radio transport
and processing capacity on both hosts.

The Spark and Thor X410 files use these standard-MTU settings:

```yaml
radio:
  send_frame_size: 1472
  recv_frame_size: 1472
  tx_send_chunk: 8
```

These settings change only the host packet format. They do not remove IQ
samples. They do not decrease spectral efficiency.

The Orin files use these settings because its tested 1 Gb/s r8168 path needs a
lower packet rate:

```yaml
radio:
  send_frame_size: 8000
  recv_frame_size: 8000
  tx_send_chunk: 1
  tx_send_realtime: false
```

Complete the jumbo-MTU check in the Orin setup guide before you use these
files. These settings change only host-to-radio packetization. They do not
remove IQ samples and do not decrease spectral efficiency.

The settings support the tested Ethernet paths. A different UHD transport can
reject `send_frame_size` or `recv_frame_size`. If the transport rejects these
fields, remove or change them. Initially, keep the value supplied in the file
for the selected platform.

### MODCOD, gain, and initial mode

```yaml
radio:
  tx_gain: 20
  rx_gain: 20

modulation:
  modcod: qpsk1/4

acm:
  enabled: false

channel:
  propagation_delay: false
  pathloss_model: none
  distance_km: 0
```

Start with QPSK 1/4 in both directions. Change `modcod` only after the first
test is successful. For the first CCM test, use the same MODCOD on both hosts.

The correct gain depends on the cables, attenuation, and local RF rules.

WARNING: Do not copy these gain values to a conducted test without a link
budget check.

Disable ACM and channel emulation for this procedure. The ping time includes
modem framing, batching, radio transfer, and host processing. It does not
include a simulated satellite delay. To use these functions, refer to the
[ACM and orbital-channel guide](02-acm-orbital-channels.md).

Keep `dvbs2a0` as the node0 TAP name. Keep `dvbs2b0` as the node1 TAP name.

## 5. Examine radio access and start cuSAT

On each host, run the short run-time tuning command from its platform guide.

Use the release image to verify that UHD can find the selected radio:

```bash
sudo docker run --rm --runtime=nvidia --network host --privileged \
  -v /dev:/dev cusat:local \
  uhd_find_devices --args "<uhd-device-args>"
```

Verify that no other cuSAT process, Kubernetes pod, SDR application, or OAI
process uses either radio.

On each host, start Docker Compose in a separate terminal:

```bash
cd ~/cusat-release-test/dvb-s2x
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml up
```

The platform file supplies the CUDA architecture and CPU set. A serviceable
endpoint has these indications:

- `Lock=True`.
- `FECFRAMEs` increases when the hosts exchange normal frames.
- `UHD_OF`, `RingDrop`, `q_drop`, and `gpu_drop` stay at zero.

During an idle period, the modem sends only dummy frames. Lock can stay true
while the data SNR has no value. Send ping or iperf traffic before you examine
the data path.

## 6. Configure the TAP interfaces

On node0, run this command:

```bash
sudo ./scripts/setup-network.sh node0
```

On node1, run this command:

```bash
sudo ./scripts/setup-network.sh node1
```

The script can run more than one time. It prints the TAP address and the
permanent peer entry. By default, it uses `dvbs2a0` for node0 and `dvbs2b0`
for node1.

If you changed `radio.tap`, supply the new TAP name as the second argument:

```bash
sudo ./scripts/setup-network.sh node1 mytap0
```

The static peer entries prevent an ARP wait on the DVB link. GSE transports
the IP packet. It does not transmit the placeholder MAC addresses.

## 7. Test the link with ping

From node1, ping node0:

```bash
ping -c 20 10.30.0.1
```

A minimized Ubuntu installation can prevent an unprivileged ping. It then
reports `SOCK_RAW: Operation not permitted`. Use this command for the test:

```bash
sudo ping -c 20 10.30.0.1
```

To restore the normal executable capability, use these commands:

```bash
ping_bin=$(command -v ping)
sudo setcap cap_net_raw+ep "$ping_bin"
getcap "$ping_bin"
```

This procedure disables channel emulation. Thus, the result is modem latency,
not GEO or LEO propagation delay. If packets are lost, examine both modem
logs. One successful ICMP reply does not show that the radio transport has no
drops.

## 8. Test sustained load in both directions

Start the iperf3 server on node0:

```bash
iperf3 -s
```

Run these commands on node1:

```bash
iperf3 -c 10.30.0.1 -b 2M -t 30
iperf3 -c 10.30.0.1 -R -b 2M -t 30
```

The explicit rate limit keeps this first test conservative. An unlimited TCP
test can fill the modem ingress queue and delay the final iperf3 control
message.

During both tests, lock must be stable. These counters must not increase:

```text
UHD_OF  RingDrop  q_drop  gpu_drop  TX_SEQ  TX_LATE
```

Do not increase the MODCOD or symbol rate until both hosts remain serviceable.

## 9. Stop cuSAT

Press `Ctrl-C` in each Docker Compose terminal. Then run this command on both
hosts:

```bash
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml down
```

## Fault-isolation sequence

1. Verify that UHD finds each selected radio.
2. Verify that no other process uses a selected radio.
3. Verify that one host is node0 and the other host is node1.
4. Verify that the TX and RX frequencies make a pair.
5. Verify that each cable uses the selected logical RF channel.
6. Verify that both hosts use the same rates and MODCOD.
7. Verify that both run-time tuning checks pass.
8. Examine the lock, transport, queue, GPU, and TX-time counters.
9. Change the RF gain or receiver limits only after these checks.
