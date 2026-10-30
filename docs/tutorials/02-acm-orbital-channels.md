# Run ACM with orbital channel models

Use this guide after you complete the [CCM guide](01-ccm-bringup.md). This
guide first starts ACM with a fixed GEO channel. It then shows how to control
path loss and link availability. Finally, it runs faster-than-real-time GEO,
LEO, and MEO channel models.

Keep the node0 and node1 radio settings from the CCM test. Also keep the
platform environment files, TAP addresses, and RF connections.

The capacity tests use iperf2 with UDP. They do not use iperf3 with TCP. If
the `iperf` command is not available, install it on both hosts:

```bash
sudo apt-get update
sudo apt-get install -y iperf
```

## Synchronize the system clocks

One controller uploads the same time-tagged schedule to both hosts. Thus, the
`CLOCK_REALTIME` clocks must agree to within 10 ms.

Use the same local time source on both hosts. The source can be an NTP server,
a chrony server, or a PTP grandmaster. A synchronized PTP hardware clock is
not sufficient. If you use PTP, use `phc2sys` or an equivalent service. This
service must make the Linux system clock follow the PTP hardware clock.

To use `systemd-timesyncd`, add these lines to
`/etc/systemd/timesyncd.conf` on both hosts:

```ini
[Time]
NTP=<LAN-NTP-SERVER>
```

Enable the service. Then examine its status:

```bash
sudo timedatectl set-ntp true
sudo systemctl restart systemd-timesyncd
timedatectl status
timedatectl timesync-status
```

Both hosts must show `System clock synchronized: yes`. They must show the same
server. The absolute `Offset` must be less than 10 ms.

If these services use different sources, do not run `phc2sys`, chrony, and
`systemd-timesyncd` at the same time.

The tested `10.112.100.0/24` laboratory uses `10.112.100.1`. The measured
host offsets were less than 6 ms. The LAN delay was less than 1 ms. This
address is for the tested site only. Use the local time source for your
network.

## 1. Start an ACM link with a fixed GEO channel

Stop the CCM modem on both hosts:

```bash
cd ~/cusat-release-test/dvb-s2x
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml down
```

On node0, copy the GEO ACM file for its platform. This example uses Thor:

```bash
cp deployment/cusat/configs/thor/geo-acm/node0.yaml \
  deployment/cusat/docker/node.yaml
```

On node1, copy the GEO ACM file for its platform. This example uses Orin:

```bash
cp deployment/cusat/configs/orin/geo-acm/node1.yaml \
  deployment/cusat/docker/node.yaml
```

Use `spark` in the path for a Spark host.

Before you transmit, examine `radio.usrp`, both frequencies, and both RF
channels. Use the values that passed the CCM test.

The supplied files enable ACM. They exclude MODCODs that have unsuitable
measured results. The optional `max_modcod` field is a comment. Thus, ACM can
use all supported MODCODs except the listed exclusions. If your test needs a
maximum MODCOD index, remove the comment character.

The value `71572 km` is two times the GEO altitude of 35786 km. It models one
ground-to-satellite-to-ground DVB direction. Apply this path separately to
both directions. The approximate propagation-only ping RTT is:

$$
2 \times \frac{71572\ \mathrm{km}}{299792.458\ \mathrm{km/s}}
\approx 478\ \mathrm{ms}.
$$

Start Docker Compose on both hosts:

```bash
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml up
```

In a second terminal on node0, configure its TAP:

```bash
sudo ./scripts/setup-network.sh node0
```

In a second terminal on node1, configure its TAP:

```bash
sudo ./scripts/setup-network.sh node1
```

The script can run more than one time. It examines the address and permanent
peer entry. Run it after each new Docker Compose container. A new container
makes a new TAP interface.

Both modem logs must change to `Lock=True`. After feedback arrives, `TxMC`
must move from QPSK 1/4. The two directions can select different MODCODs. Each
receiver measures and reports its local channel. ACM does not use FDD
reciprocity.

After a new start, ACM tests the MODCOD sequence with padding-only probation
frames. It does not release queued application packets on a new MODCOD until
the probation test is successful.

With no `max_modcod` limit, `TxMC` can change for some minutes. During this
time, OLLA learns the RF implementation loss for each MODCOD. This loss is not
in the AWGN waterfall data. A failed high-order test does not add a penalty to
all lower MODCODs.

For a stable measurement, wait until `TxMC` does not change for several log
periods. During the idle calibration, these counters must stay at zero:

```text
UF  RingDrop  tap_drop  q_drop  gpu_drop
```

From node1, test the GEO path:

```bash
ping -c 10 10.30.0.1
```

On node0, start an iperf2 UDP server:

```bash
iperf -s -u -i 1
```

On node1, offer 200 Mbit/s for two minutes:

```bash
iperf -c 10.30.0.1 -u -b 200M -t 120 -i 1
```

This offered rate is higher than the capacity of this configuration. On the
node0 server, examine the one-second `Bandwidth` and `Lost/Total Datagrams`
columns. These columns show the received goodput and loss. The client output
shows only the offered rate. It is not a capacity result.

High UDP loss is normal during this full-load test. It does not directly show
a DVB frame error. The source sends much more traffic than the selected
MODCOD can transport. Thus, `tap_drop` can increase on the sending host. This
counter shows that cuSAT removed excess offered traffic.

The following counters must stay at zero on both hosts:

```text
UF  TX_SEQ  TX_LATE  UHD_OF  RingDrop  q_drop  gpu_drop
```

At a fixed distance, `pilotSNR`, `TxMC`, and received UDP goodput must become
approximately stable. UDP prevents TCP congestion control and the GEO RTT
from controlling this measurement.

## 2. Use the channel-control interface

Each host applies the channel model to its local TX path. Thus, the node0
model controls node0 to node1. The node1 model controls node1 to node0.

Run one controller on either host. For a symmetric channel, give it both
management endpoints. For an asymmetric channel, give it one endpoint.

Use `127.0.0.1:5556` for the local modem. Use the peer's SSH management IP
with port 5556 for the remote modem. Do not use the DVB TAP address.

The `channel` fields have these functions:

| Field | Function |
|---|---|
| `side` | Applies the channel model to the local `tx` or `rx` IQ stream. Use `tx` for two independent hosts. |
| `propagation_delay` | Applies a fixed integer-sample IQ delay from the initial `distance_km`. |
| `pathloss_model` | `freespace` applies the relative loss $20\log_{10}(d/d_0)$. `none` disables distance-dependent loss. |
| `distance_km` | Sets the initial propagation distance and the free-space-loss reference $d_0$. |
| `noise_port` | Sets the host UDP port for run-time channel commands. |

All supplied GEO, MEO, and LEO profiles use integer-sample propagation delay.
The modem rounds the requested delay to the nearest sample. The error is at
most half of one sample. At 30.72 Msamples/s, this is less than 17 ns. The
orbit schedules change visibility and path loss. They do not move the delay
tap.

The free-space model is relative. At the reference distance, it adds 0 dB of
loss. The radio gains and the physical RF path set the absolute SNR. A
distance change causes the applicable free-space-loss change.

The model does not calculate these items:

- antenna gain.
- transmit power.
- receiver noise figure.
- atmospheric loss.
- rain loss.

### UDP command format

The initial UDP command is an eight-byte little-endian packet. It contains a
32-bit command identifier and one 32-bit floating-point value. This command
format is backward compatible. A command has an effect on the next IQ block.

| Command | Identifier | Value |
|---|---:|---|
| `DIST` | `0x44495354` | Distance in km for relative free-space path loss. It does not change the active propagation delay. |
| `LINK` | `0x4c494e4b` | `1.0` for an available link. `0.0` for an out-of-view link. |
| `NOIS` | `0x4e4f4953` | Direct noise or path-loss control for special tests. |
| `CLER` | `0x434c4552` | Discards future scheduled commands. It retains the current channel state. |

The orbital script uses a scheduled extension. This format adds a 64-bit Unix
UTC application time in nanoseconds. It also adds a 64-bit sequence number.
The complete command is 24 bytes.

The controller first builds the complete schedule. The schedule has one UTC
start time and one relative time offset for each orbital position.

The controller selects a safe future start time by default. It uploads every
point to both modems before the schedule starts. Thus, the user does not
calculate or copy an epoch.

cuSAT puts each command in a deadline-ordered queue. It sends `ACKQ` after it
queues a command. It sends `ACK1` after it applies the command. `ACK1` contains
the requested time and the actual application time.

The script stops in these conditions:

- an ACK is missing.
- a modem cannot store the complete schedule.
- a command arrives after its application time.
- the application is more than 25 ms late.

These checks can find an old modem, a stopped modem, or local scheduling
problems. The common system clock aligns the two radio directions.

### Apply a fixed path loss

Use `scripts/channel-ctl.py` to send commands. Do not make the packets
manually. Run this example on node0. Replace the second address with the
management IP that you use to connect to node1.

Add 6 dB of loss to both GEO directions with one command:

```bash
python3 scripts/channel-ctl.py --pathloss-db 6 \
  127.0.0.1:5556 10.112.100.85:5556
```

After some log periods, `pilotSNR` must decrease. ACM can select a lower
MODCOD. This command does not change the fixed propagation delay.

Restore the reference condition on both hosts:

```bash
python3 scripts/channel-ctl.py --pathloss-db 0 \
  127.0.0.1:5556 10.112.100.85:5556
```

`pilotSNR` must increase. The selected MODCOD must follow after the applicable
feedback and hysteresis delay.

The 6 dB command changes the TX amplitude. A short cabled path can have
front-end distortion. In this condition, the pilot estimator can show a
change of less than 6 dB.

### Control orbital path loss

The orbital script uses `DIST`. With `channel-ctl.py`, use this option:

```text
--pathloss-distance-km DISTANCE
```

The script calculates $20\log_{10}(d/d_0)$ from the reference distance in the
YAML file. The `DIST` command does not move the active propagation-delay tap.

Do not use a distance less than $d_0$ only to make a test fade. A shorter
distance gives negative loss, which is digital gain. This gain can clip a
strong cabled signal.

The LEO example starts at the minimum distance of 1100 km. It sends only
equal or longer visible-path distances. Thus, its distance commands add only
loss.

The active distance command changes path loss but does not change IQ delay. A
moving delay tap causes a sample-rate offset, which is baseband Doppler. This
model does not cut or repeat live samples. Thus, it keeps Doppler outside the
scope of this test.

### Control link availability

To make a symmetric outage, run this sequence on one management host:

```bash
python3 scripts/channel-ctl.py --link down \
  127.0.0.1:5556 10.112.100.85:5556
sleep 5
python3 scripts/channel-ctl.py --link up \
  127.0.0.1:5556 10.112.100.85:5556
```

Traffic must stop during the outage. Both receivers must lose lock. They must
get lock again after the link becomes available.

SECURITY: UDP port 5556 does not authenticate commands. Use it only on a
trusted management network. Alternatively, use the host firewall to limit
access.

## 3. Run orbital demonstrations

### Run the tested LEO demonstration

A LEO pass needs a different initial reference. Stop Docker Compose on both
hosts. Use the command from section 1.

On node0, copy the LEO ACM file for its platform. This example uses Thor:

```bash
cp deployment/cusat/configs/thor/leo-acm/node0.yaml \
  deployment/cusat/docker/node.yaml
```

On node1, copy the LEO ACM file for its platform. This example uses Orin:

```bash
cp deployment/cusat/configs/orin/leo-acm/node1.yaml \
  deployment/cusat/docker/node.yaml
```

Examine the radio fields again. Then start Docker Compose on both hosts. The
LEO files use an initial reference distance of 1100 km.

This reference models a bent-pipe path at the zenith of a 550 km orbit. The
path is 550 km up and 550 km down.

After the new containers start, configure the TAP on each host. Use the
commands from section 1.

On node0, start the iperf2 UDP server:

```bash
iperf -s -u -i 1
```

On node1, offer a full-load flow for the complete demonstration:

```bash
iperf -c 10.30.0.1 -u -b 200M -t 180 -i 1
```

On node0, run one channel controller. Replace the second address with the
management IP that you use to connect to node1:

```bash
scripts/leo-acm-demo.sh \
  127.0.0.1:5556 10.112.100.85:5556 \
  | tee /tmp/cusat-leo-channel.csv
```

The controller calculates a safe start time. It then uploads the complete
schedule to both modems. Both channel paths use this common start time.

The script models a circular orbit at an altitude of 550 km. It sends two
times the current satellite slant range to the bent-pipe channel model.

The test contains three visible arcs. Each arc is approximately 29 seconds at
10 times the real orbital rate. Two five-second out-of-view periods separate
the arcs. Each arc starts and ends near 23 degrees of elevation. The receiver
can get lock again at this elevation after a complete outage.

The script schedules five positions each second. After the final position,
cuSAT retains the final path loss and link state.

The controller prints its queue ACK count and application ACK count. It also
prints the maximum application lateness. A valid test does not show an error.
The lateness must be less than 25 ms.

The actual SNR, MODCOD, and goodput depend on the radio gain and RF path.
Examine these changes during the trajectory:

- `pilotSNR` increases from the pass edge to the zenith.
- `pilotSNR` decreases from the zenith to the pass edge.
- `TxMC` and received iperf goodput follow the usable channel.
- received goodput becomes zero during each five-second outage.
- `Lock=False` occurs during each outage.
- the receiver gets lock again after each outage.
- ACM initially selects a conservative MODCOD after each outage.

If `TxMC` stays at its maximum value, decrease the TX gain by the same amount
in both YAML files. Then repeat the test.

The tested arc gives approximately 7 dB of free-space-loss change. It does not
include the last degrees above the geometric horizon. A locked receiver can
track at these low elevations. It cannot always get lock there after a
complete outage.

For a full horizon-to-horizon test, run `acm-pathloss-driver.py` with a larger
`--arc-fraction`. An out-of-view period always causes a complete outage. The
gain does not change this behavior.

The controller writes one CSV file. The file contains these values:

- time offset from the common start.
- relative free-space loss.
- one-way slant range.
- modeled path-loss distance.
- elevation.
- link availability.

A relative loss of `-1` identifies an out-of-view period.

To repeat the complete schedule, add `--cycle`:

```bash
scripts/leo-acm-demo.sh --cycle \
  127.0.0.1:5556 10.112.100.85:5556 \
  | tee /tmp/cusat-leo-channel.csv
```

The controller uploads the next cycle while the current cycle runs. The last
time offset connects to the first time offset of the next cycle. Press
Ctrl-C to stop repetition. The controller then removes future events and
retains the current channel state.

This test runs faster than real time. Distance commands change free-space
path loss. The integer-sample propagation delay stays at its initial
1100 km value. This prevents an artificial baseband Doppler effect.

The model does not include carrier Doppler, atmospheric fading, or antenna
patterns.

### Run GEO, LEO, and MEO models

The channel model uses the initial YAML distance as its reference. Before you
change the orbital regime, stop Docker Compose. Set the applicable
`channel.distance_km` on both hosts. Then restart both modems.

The initial value sets the fixed propagation delay. It also sets the 0 dB
relative-loss reference.

| Regime | Nominal altitude | `channel.distance_km` for the zenith bent-pipe path |
|---|---:|---:|
| LEO | 550 km | 1100 km |
| MEO | 8000 km | 16000 km |
| GEO | 35786 km | 71572 km |

The `leo-acm` files use 1100 km. The `geo-acm` files use 71572 km.

For MEO, first copy the applicable `leo-acm` file for each platform and role.
Then change this field in both `deployment/cusat/docker/node.yaml` files:

```yaml
channel:
  distance_km: 16000
```

After each new container starts, configure the TAP on each host. Use the
commands from section 1.

Run one controller for both channel directions. Replace the second address
with the management IP of the other modem.

To apply and retain a fixed GEO state, run this command on one host:

```bash
python3 scripts/acm-pathloss-driver.py \
  --regime geo \
  127.0.0.1:5556 10.112.100.85:5556
```

For a one-minute LEO model, run this command on one host:

```bash
python3 scripts/acm-pathloss-driver.py \
  --regime leo --passes 1 --pass-duration 60 \
  --arc-fraction 0.4 --rate-hz 5 \
  127.0.0.1:5556 10.112.100.85:5556
```

For a one-minute MEO model, run this command on one host:

```bash
python3 scripts/acm-pathloss-driver.py \
  --regime meo --passes 1 --pass-duration 60 \
  --arc-fraction 0.4 --rate-hz 5 \
  127.0.0.1:5556 10.112.100.85:5556
```

These models are test inputs. Do not use them as calibrated link-budget
predictions. Examine the logs and the received iperf output. Do not expect a
specified SNR, MODCOD, or goodput value for a different RF installation.

The script also supplies these optional controls:

| Option | Function |
|---|---|
| `--alt-km N` | Sets a different altitude. Before the test, set the YAML reference distance to approximately two times this altitude. |
| `--passes N` | Sets the number of visible passes. |
| `--accel N` | Sets the orbital time scale if `--pass-duration` is not set. |
| `--pass-duration N` | Sets the test duration of one visible pass. |
| `--arc-fraction F` | Sets the fraction of the geometric horizon arc. A larger value gives a lower elevation and a larger loss change. |
| `--gap-sec N` | Sets the out-of-view time between passes. |
| `--pre-gap-sec N` | Sets an out-of-view time before the first pass. |
| `--rate-hz N` | Sets the number of path-loss and link updates each second. |
| `--static-elevation DEG` | Sets a fixed elevation instead of a moving pass. |
| `--start-in N` | Starts the uploaded schedule after N seconds. Usually, use the automatic default. |
| `--start-at TIME` | Sets an ISO-8601 UTC time or Unix epoch. Usually, use the automatic default. |
| `--cycle` | Repeats the complete schedule until the user presses Ctrl-C. |

You can change these channel-model inputs. They are not performance settings.
During a test, movement changes relative path loss and link availability. The
propagation delay stays at the initial YAML value.

## 4. Understand the ACM response

Each receiver reports pilot Es/N0 and cumulative post-BCH frame results. It
sends these reports to the other host in QPSK 1/4 signaling frames.

Standard DVB dummy PLFRAMEs are the normal idle signal. In ACM mode, cuSAT
adds signaling frames at a low rate. Thus, feedback and reacquisition can
continue when there is no application traffic.

The status line uses `pilotSNR` for the current estimate. It uses `acmSNR` for
the filtered value that ACM receives.

The default `acm.snr_ewma_alpha: 0.1` operates at 5 Hz. A lower value removes
more isolated estimator changes. A higher value follows a changing channel
more quickly. The standard deviation uses the unfiltered 5 Hz samples. It
also includes large samples that are structurally valid.

The sender uses measured implementation waterfalls at a post-BCH FER of
$10^{-5}$. It calculates the limit for each MODCOD as follows:

$$
\gamma_{\mathrm{required}} = \gamma_{\mathrm{waterfall}}
  + \texttt{guard\_db} + \Delta_{\mathrm{OLLA}}.
$$

OLLA stores a separate residual SNR correction for each MODCOD. It updates
the correction from the successful and failed frames for that MODCOD.

At startup, ACM waits for three valid SNR reports. It then selects a MODCOD
directly from the measured waterfall. The supplied configurations add
`acm.bootstrap_margin_db: 2.0` dB to the normal guard for this first
selection. The selected MODCOD carries only padding until 64 frames decode
without an error. Application data continues to use the safe MODCOD during
this check. Thus, startup does not test each intermediate MODCOD, but an
incorrect initial estimate cannot corrupt application data.

After this initial check, ACM does not use the bootstrap margin. It uses the
normal guard and the learned OLLA correction. A feedback outage resets the
initial SNR collection, but it does not delete learned OLLA corrections.

ACM immediately selects a lower MODCOD when necessary. It also uses a
short-term SNR trend. To select a higher MODCOD, ACM requires repeated valid
reports.

The supplied exclusions remove MODCODs with dominated or nonmonotonic results
on the measured INT16 waterfall and efficiency envelope.

During a test, examine these values on both hosts:

- `pilotSNR` and `acmSNR`.
- `TxMC` and the OLLA correction.
- ACM change messages.
- post-BCH FER.
- absolute LDPC iteration count.
- received iperf2 UDP goodput and loss.
- ping RTT.
- `UHD_OF`, `RingDrop`, `q_drop`, and `gpu_drop`.
- source `tap_drop`.
- lock loss and reacquisition during each outage.

Before the full-load UDP test, all drop counters must stay at zero. During the
200 Mbit/s test, source `tap_drop` can increase. The other drop counters must
stay at zero.

A successful ACM test keeps application traffic active while it changes the
MODCOD. A high MODCOD must not conceal a radio overrun, queue drop, or decoder
drop.
