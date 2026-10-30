# Prepare a Jetson Orin NX endpoint

Use this guide to prepare a Jetson Orin NX 16 GB host for cuSAT. The tested
host has eight CPU cores. At the end of this procedure, the host has these
items:

- access to the radio.
- the correct real-time host settings.
- a Docker image with the name `cusat:local`.
- a successful test result.

After this procedure, assign the node0 or node1 role. Refer to the
[CCM guide](01-ccm-bringup.md).

## 1. Examine the host

The tested host uses Ubuntu 24.04 and Jetson Linux. It also has these
components:

- eight CPU cores.
- an NVIDIA driver that supports CUDA.
- Docker Engine.
- NVIDIA Container Toolkit.
- the Docker Compose plug-in.

Use these commands to examine the host:

```bash
getconf _NPROCESSORS_ONLN
nvpmodel -q
docker --version
docker compose version
```

The tested power mode has eight online CPU cores and a 40 W limit. Power-mode
identifiers can change with the carrier board and the JetPack release. Select
the equivalent mode for your board. Do not copy a numeric identifier without
a check.

CAUTION: Do not use this CPU configuration on an Orin model that has a
different number of CPU cores.

## 2. Clone the release

```bash
mkdir -p ~/cusat-release-test
cd ~/cusat-release-test
git clone https://github.com/wineslab/dvb-s2x.git
cd dvb-s2x
git submodule update --init --recursive
git rev-parse --short HEAD
sudo timedatectl set-ntp true
```

## 3. Install the CPU-isolation configuration

The tested configuration keeps CPU cores 0 and 1 for the kernel and the radio
NIC. It assigns CPU cores 2 through 7 to cuSAT.

The supplied script changes only the active Jetson `extlinux` entry. The
script also makes a backup with a time stamp.

First, examine the configuration. Then apply it:

```bash
deployment/cusat/host/orin/boot-profile.sh check
sudo deployment/cusat/host/orin/boot-profile.sh apply
```

If the `apply` command reports a change, reboot the host:

```bash
sudo reboot
```

After the host starts, examine the active configuration:

```bash
cd ~/cusat-release-test/dvb-s2x
deployment/cusat/host/orin/boot-profile.sh check
```

Both profile lines must show `configured`. The isolated CPU list must be
`2-7`.

To restore the previous configuration, use the backup path that the `apply`
command printed:

```bash
sudo deployment/cusat/host/orin/boot-profile.sh restore \
  /boot/extlinux/extlinux.conf.cusat-backup.YYYYMMDD-HHMMSS
sudo reboot
```

## 4. Make the radio accessible

Configure the host for your radio and its transport. A network radio must be
reachable from the host. A USB radio must be visible on the host and in the
container.

cuSAT uses UHD. It can operate with most UHD radios that supply the necessary
TX and RX channels and sample rate. Release tests cover only USRP X410 radios.
Other radio models can
have different maximum rates and IRQ requirements.

The repository does not specify a VLAN, interface name, MTU, or host IP
address. Your radio network specifies these values. If the switch sends a
tagged VLAN to the host, use the
[optional VLAN procedure](usrp-networking.md#optional-tagged-vlan). If the
switch sends untagged traffic, do not make a VLAN interface.

Use the UHD commands in section 6 to examine radio access.

## 5. Install the run-time settings

Install the UHD socket-buffer settings:

```bash
sudo install -m 0644 \
  deployment/cusat/host/common/99-cusat-socket-buffers.conf \
  /etc/sysctl.d/99-cusat-socket-buffers.conf
sudo sysctl --system | \
  grep -E '^net\.core\.(rmem|wmem)_(default|max) = '
sysctl net.core.rmem_default net.core.rmem_max \
  net.core.wmem_default net.core.wmem_max
```

The last command must show these four values:

```text
net.core.rmem_default = 250000000
net.core.rmem_max = 250000000
net.core.wmem_default = 250000000
net.core.wmem_max = 250000000
```

Before each CCM modem test, set all clocks to their maximum values:

```bash
sudo jetson_clocks
```

For a high-rate Ethernet radio, put the physical NIC IRQs on CPU cores 0 and
1. The tested Orin profile also uses 8000-byte UHD transport frames. Thus,
the host-to-radio path must support an MTU of 9000.

Supply the physical interface name first. If the radio address is on a VLAN,
also supply the VLAN interface name. The script sets the required MTU on both
interfaces.

```bash
sudo deployment/cusat/host/orin/tune.sh \
  <physical-usrp-interface> [<radio-or-vlan-interface>]
```

For example, the tested tagged laboratory link uses this command:

```bash
sudo deployment/cusat/host/orin/tune.sh enP8p1s0 enP8p1s0.21
```

Verify jumbo packets before you start cuSAT. Replace the address with the
address of the selected radio:

```bash
sudo ping -M do -s 8000 -c 3 192.168.40.18
```

All three packets must arrive. If this test fails, configure the host, switch,
and radio path for jumbo packets. Do not start the supplied Orin scenario with
an MTU of 1500. The smaller frames can cause UHD sequence errors even when the
nominal bit rate is less than 1 Gb/s.

Do not use this NIC command for a USB radio. A different network driver can
require a different IRQ method. The script stops if it cannot verify the
tested Orin IRQ configuration.

## 6. Build and test the Docker image

Install the local Docker Compose settings for Orin:

```bash
cp deployment/cusat/configs/orin/runtime.env \
  deployment/cusat/docker/platform.env
```

Build the image. Then run the tests:

```bash
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml build
sudo docker run --rm --runtime=nvidia cusat:local \
  ctest --test-dir /build --output-on-failure
```

Use the container to find and examine the radio:

```bash
sudo docker run --rm --runtime=nvidia --network host --privileged \
  -v /dev:/dev cusat:local \
  uhd_find_devices --args "<uhd-device-args>"
sudo docker run --rm --runtime=nvidia --network host --privileged \
  -v /dev:/dev cusat:local \
  uhd_usrp_probe --args "<uhd-device-args>"
```

Use the UHD device arguments for your radio. For a network X410, an example
is `addr=192.168.40.20`. For one USB device, an example is
`type=b200,serial=...`.

All tests must pass. UHD must find the correct radio. The Orin host is now
ready. Continue with the [two-radio CCM guide](01-ccm-bringup.md).
