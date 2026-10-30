# Run OAI backhaul on Spark and Thor

Use this guide to carry OAI 5G traffic through a cuSAT GEO link.

Use this tested topology:

```text
  PHYSICAL RADIO PATHS

  +-------+    5G RF    +------------------+    DVB-S2X GEO RF    +----------------+
  | 5G UE |<===========>| Spark X410 .20   |<====================>| Thor X410 .18 |
  +-------+             | ch 0: OAI       |                      | ch 1: DVB TX   |
                        | ch 1: DVB TX    |                      | ch 3: DVB RX   |
                        | ch 3: DVB RX    |                      +----------------+
                        +------------------+


  FUNCTIONAL PATHS

  +------------------------------- SPARK (boat) --------------------------------+
  |                                                                             |
  |  +-------------+  radio IQ  +-------------------+  UHD ch 0  +-------------+ |
  |  | OAI gNB     |<---------->| usrpproxy        |<----------->| Spark X410  | |
  |  | nr-softmodem|   shared   | in cuSAT process |             | .20          | |
  |  +------+------+   memory   +-------------------+             +-------------+ |
  |         | N2 and N3 IP packets                                                |
  |         v                                                                     |
  |  +----------------------+ GSE/DVB +-------------+ UHD +-------------------+  |
  |  | dvbs2a0             |<------->| cuSAT modem |<--->| Spark X410 .20   |  |
  |  | 10.200.0.1          |          +-------------+     | ch 1 TX / ch 3 RX |  |
  |  +----------------------+                              +-------------------+  |
  +-----------------------------------------------------------------------------+
                                      ^
                                      | DVB-S2X GEO RF
                                      v
  +------------------------------- THOR (shore) --------------------------------+
  |                                                                             |
  |  +------------------+  UHD  +-------------+  GSE/DVB  +------------------+   |
  |  | Thor X410 .18    |<---->| cuSAT modem |<--------->| dvbs2b0         |   |
  |  | ch 1 TX / ch 3 RX|      +-------------+           | 10.200.0.2      |   |
  |  +------------------+                                +-----+--------+---+   |
  |                                                             |        |       |
  |                                                  N2 forward |        | N3    |
  |                                                  and DNAT   |        v       |
  |                                                             |  +----------+  |
  |                                                             |  | Edge UPF |  |
  |                                                             |  | Open5GS  |  |
  |                                                             |  +----+-----+  |
  +-------------------------------------------------------------+-------|--------+
                                                                |       |
                                                                | N2    | N4/N6
                                                                v       v
                                  +---------------------------------------------+
                                  | EXTERNAL 5G CORE AND DATA NETWORK           |
                                  | AMF | SMF | AUSF/UDM | external data network|
                                  +---------------------------------------------+

  N2: OAI gNB <-> dvbs2a0 <-> DVB <-> dvbs2b0 <-> Thor DNAT <-> AMF
  N3: OAI gNB <-> dvbs2a0 <-> DVB <-> dvbs2b0 <-> edge UPF
  N4: edge UPF <-> Thor normal network <-> SMF
  N6: edge UPF <-> Thor normal network <-> external data network
```

Spark and Thor run the radio and backhaul components in Docker. The 5G core
is an external service. The core can run on another host or in a cluster.

Do not use Orin for this test. Its tested USRP connection uses 1 GbE. One
30.72-Msample/s SC16 stream uses approximately 983 Mbit/s before Ethernet
overhead. This rate is too close to the port limit. Spark and Thor use 10 GbE
radio connections.

## System configuration

Use Spark for these functions:

- OAI gNB.
- Boat cuSAT endpoint.
- Shared access to the boat X410.

Use Thor for these functions:

- Shore cuSAT endpoint.
- Edge Open5GS UPF.
- Forwarding between the DVB link and the external 5G core.

The Spark X410 uses these channels:

- Channel 0 carries OAI 5G samples.
- Channel 1 carries cuSAT transmit samples.
- Channel 3 carries cuSAT receive samples.

cuSAT owns the X410 device. The `usrpproxy` shared-memory interface gives
channel 0 to OAI. OAI does not open a second UHD connection to the X410.

The OAI N2 and N3 interfaces use the cuSAT TAP link:

- The gNB N2 and N3 address is `10.200.0.1` on Spark.
- The edge UPF N3 address is `10.200.0.2` on Thor.
- Thor forwards N2 signaling to the AMF.
- The edge UPF uses the normal Thor network to reach the SMF through PFCP.

The tested DVB profile uses these values:

- 30.72 Msamples/s.
- 24 Msymbol/s.
- 1.28 samples per symbol.
- 0.10 roll-off.
- ACM.
- A fixed GEO propagation delay.

OAI uses 30.72 Msamples/s on channel 0 of the Spark X410.

## Before you start

Complete these host guides:

- [Prepare Spark](setup-spark.md).
- [Prepare Thor](setup-thor.md).

Also complete the [ACM guide](02-acm-orbital-channels.md). Make sure that the
standalone DVB link is stable before you add OAI.

You must also have these items:

- Two tested X410 radio paths.
- A UE and SIM configuration that work with the external core.
- An Open5GS AMF that Thor can reach.
- An Open5GS SMF PFCP endpoint that Thor can reach.
- Permission to stop the normal core UPF during the test.
- Permission to change and restart the SMF.

Run all commands from the repository root unless a step gives a different
location.

## 1. Prepare the external 5G core

This deployment does not install a 5G core. It does not add a subscriber.
First, make sure that the core and the subscriber configuration work.

The test starts a temporary edge UPF on Thor. The SMF must use this edge UPF.
Stop the normal core UPF before the test. If two UPFs serve the same DNN, user
traffic can bypass the DVB link or the PDU session can fail.

Make these core changes with the procedures for your core installation:

1. Stop the normal UPF.
2. Remove the fixed normal-UPF peer from the SMF configuration.
3. Let the Thor UPF start the PFCP association with the SMF.
4. Make the AMF SCTP port reachable from Thor.
5. Make the SMF PFCP UDP port reachable from Thor.
6. Restart the SMF.

Record the original core configuration. You must restore it after the test.

## 2. Prepare Thor

Make sure that no other UPF runs on Thor:

```bash
pgrep -a open5gs-upfd || true
```

If this command shows a local UPF service, stop it for the test:

```bash
sudo systemctl stop open5gs-upfd
```

Set the Thor clocks and tune the USRP-facing network interface. Replace the
interface name with the physical 10 GbE interface. Do not use a VLAN
interface name.

```bash
sudo jetson_clocks
sudo deployment/cusat/host/thor/tune.sh <usrp-facing-interface> 0-3
```

## 3. Prepare Spark

Tune the physical ConnectX interface that carries the X410 traffic. Replace
the interface name with the physical interface. Do not use a VLAN interface
name.

```bash
sudo deployment/cusat/host/spark/tune.sh <usrp-facing-interface> 0-3
```

Do not run another real-time workload on the CPU cores that this test uses.

## 4. Set the site values

Run these commands on Spark and Thor:

```bash
cp deployment/cusat-mobile-backhaul/docker/site.env.example \
  deployment/cusat-mobile-backhaul/docker/site.env
${EDITOR:-nano} deployment/cusat-mobile-backhaul/docker/site.env
```

Set these values for your core:

- `CORE_AMF_IP` is an AMF address that Thor can reach.
- `CORE_AMF_PORT` is the reachable AMF SCTP port.
- `CORE_SMF_PFCP_HOST` is an SMF address that Thor can reach.
- `CORE_SMF_PFCP_PORT` is the reachable SMF PFCP UDP port.
- `PLMN_MCC`, `PLMN_MNC`, and `PLMN_MNC_LENGTH` identify the PLMN.
- `TRACKING_AREA_CODE` identifies the tracking area.
- `CORE_DNN` identifies the data network.
- `CORE_UE_SUBNET` and `CORE_UE_GATEWAY` must agree with the core.

The tested laboratory core uses these values:

- PLMN: `001/05`.
- DNN: `oai`.
- UE subnet: `10.45.0.0/16`.
- UE gateway: `10.45.0.1`.

Use AMF port `38412` and SMF PFCP port `8805` when the core uses the standard
ports. If your core exposes different ports, put those ports in `site.env`.
Do not use the example addresses until you verify that Thor can reach them.

## 5. Check the radio configuration

On Spark, examine this file:

```text
deployment/cusat-mobile-backhaul/configs/docker/spark-boat.yaml
```

On Thor, examine this file:

```text
deployment/cusat-mobile-backhaul/configs/docker/thor-shore.yaml
```

Set the correct USRP address in each file. The tested values are:

- Spark X410: `192.168.40.20`.
- Thor X410: `192.168.40.18`.

Keep the supplied channel assignments if your antenna connections are the
same as the tested connections.

## 6. Build the images one time

If the mobile-backhaul images are already present, you can skip this section.

On Spark, check for the cuSAT and OAI images:

```bash
sudo docker image inspect \
  cusat-mobile-backhaul-dvb:sm121 \
  cusat-mobile-backhaul-oai:develop >/dev/null
```

If an image is absent, build both images:

```bash
sudo env CUSAT_CUDA_ARCH=121 \
  ./deployment/cusat-mobile-backhaul/docker/release.sh build boat
```

On Thor, check for the cuSAT image:

```bash
sudo docker image inspect cusat-mobile-backhaul-dvb:sm110 >/dev/null
```

If the image is absent, build it:

```bash
sudo env CUSAT_CUDA_ARCH=110 \
  ./deployment/cusat-mobile-backhaul/docker/release.sh build shore
```

The OAI build gets source from the public DURANTA repository. The default
reference is `develop`. Set `OAI_REF` to a tag or a commit for a repeatable
build.

## 7. Start the containers

On Spark, start the boat containers:

```bash
sudo env CUSAT_CUDA_ARCH=121 \
  ./deployment/cusat-mobile-backhaul/docker/release.sh up boat
```

On Thor, start the shore containers:

```bash
sudo env CUSAT_CUDA_ARCH=110 \
  ./deployment/cusat-mobile-backhaul/docker/release.sh up shore
```

These commands start long-running containers. They do not start the radio
processes.

On each host, examine the containers and their CPU sets:

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh status
```

The default CPU assignments are:

- Spark cuSAT: CPUs 4 through 11.
- Spark OAI: CPUs 12 through 19.
- Thor cuSAT: CPUs 4 through 11.
- Thor edge UPF: CPUs 0 through 3.

## 8. Start the processes

Use four terminals. Keep each process in the foreground. Start the processes
in the specified sequence.

### Terminal 1: Thor shore cuSAT

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh run shore-dvb
```

### Terminal 2: Spark boat cuSAT

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh run boat-dvb
```

### Terminal 3: Thor edge UPF

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh run upf
```

Wait until the UPF log shows `PFCP associated`.

### Terminal 4: Spark OAI gNB

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh run oai
```

Start OAI soon after you start boat cuSAT. OAI must consume the shared radio
stream. Start the edge UPF before OAI. The UPF installs the N2 forwarding
rules before OAI makes its first AMF connection.

## 9. Check the DVB link

The cuSAT logs on both hosts must show these results:

- `Lock=True`.
- `FECFRAMEs` increases.
- `PDUs` increases when traffic is present.
- The ACM feedback counters increase.
- `UHD_OF`, `RingDrop`, `q_drop`, and `gpu_drop` do not increase.

The initial MODCOD is QPSK 1/4. ACM uses the measured SNR to select a higher
MODCOD after it receives feedback. The selected MODCOD depends on the RF
path.

On Spark, check the route to Thor:

```bash
ping -I dvbs2a0 -c 10 10.200.0.2
ip route get 10.200.0.2
```

The route must use `dvbs2a0`.

On Thor, check the route to Spark:

```bash
ping -I dvbs2b0 -c 10 10.200.0.1
ip route get 10.200.0.1
```

The route must use `dvbs2b0`.

The fixed GEO model gives an RTT of approximately 500 ms. The value includes
the delay in both directions and the processing time.

## 10. Check OAI and the PDU session

The OAI log must show `RU 0 RF started`. It must not continuously show these
messages:

- `RX stalled`.
- `PRACH job is in the past`.
- Late-sample messages.

Start the UE. The OAI log must show that the UE connects. The SMF and the
Thor UPF logs must then show the PDU session.

The tested core gives the UE an address in `10.45.0.0/16`. From the UE, ping
the configured gateway:

```bash
ping 10.45.0.1
```

The default GEO delay causes an RTT of approximately 500 ms. User-plane
traffic must appear in both cuSAT logs.

Available capacity depends on the selected MODCOD, radio conditions and OAI
configuration. Measure it on the deployed system before selecting an offered
traffic rate.

## 11. Stop the test

Press `Ctrl-C` in the four process terminals.

On Spark, remove the boat containers:

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh down
```

On Thor, remove the shore containers:

```bash
sudo ./deployment/cusat-mobile-backhaul/docker/release.sh down
```

Use the procedures for your core installation to restore the original SMF
configuration. Restart the normal core UPF.

If you stopped a local Thor UPF service, restart it:

```bash
sudo systemctl start open5gs-upfd
```

Make sure that the normal core UPF is active before you finish the test.
