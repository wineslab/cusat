# Prepare a DGX Spark endpoint

Use this guide to prepare a 20-core DGX Spark host for cuSAT. At the end of
this procedure, the host has these items:

- access to the radio.
- the correct real-time host settings.
- a Docker image with the name `cusat:local`.
- a successful test result.

After this procedure, assign the node0 or node1 role. Refer to the
[CCM guide](01-ccm-bringup.md).

## 1. Examine the host

The tested host uses Ubuntu. It also has these components:

- 20 CPU cores.
- an NVIDIA driver that supports CUDA.
- Docker Engine.
- NVIDIA Container Toolkit.
- the Docker Compose plug-in.
- a ConnectX interface for the radio.

Use these commands to examine the host:

```bash
getconf _NPROCESSORS_ONLN
nvidia-smi -L
docker --version
docker compose version
```

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

The tested configuration keeps CPU cores 0 through 3 for housekeeping and
ConnectX IRQs. It assigns CPU cores 4 through 19 to cuSAT.

Make a backup of the main GRUB configuration file. Then open the file:

```bash
if [ ! -e /etc/default/grub.pre-cusat ]; then
  sudo cp -a /etc/default/grub /etc/default/grub.pre-cusat
fi
sudo nano /etc/default/grub
```

Replace the complete `GRUB_CMDLINE_LINUX=` assignment with this line:

```bash
GRUB_CMDLINE_LINUX="pci=realloc=off pci=pcie_bus_safe module_blacklist=nouveau acpi_power_meter.force_cap_on=y numa_balancing=disable isolcpus=managed_irq,domain,4-19 nohz_full=4-19 rcu_nocbs=4-19 irqaffinity=0-3"
```

Do not add a second assignment. Do not duplicate an existing `isolcpus`,
`nohz_full`, `rcu_nocbs`, or `irqaffinity` argument. If the Spark also runs an
OAI or DPDK load, put all requirements in one reviewed line.

Update GRUB. Then reboot the host:

```bash
sudo update-grub
sudo reboot
```

After the host starts, examine the active configuration:

```bash
cat /proc/cmdline
cat /sys/devices/system/cpu/isolated
cat /sys/devices/system/cpu/nohz_full
```

Both CPU lists must include `4-19`.

To restore the previous configuration, use these commands:

```bash
sudo cp -a /etc/default/grub.pre-cusat /etc/default/grub
sudo update-grub
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

For a high-rate Ethernet radio, distribute the physical ConnectX IRQs across
CPU cores 0 through 3. Do this before each modem test:

```bash
command -v set_irq_affinity_cpulist.sh
sudo deployment/cusat/host/spark/tune.sh <usrp-facing-interface> 0-3
```

Supply the physical interface name. Do not supply the name of a VLAN
interface. Do not use this command for a USB radio.

The supplied script supports the tested ConnectX path. A different NIC can
require a different IRQ method. The script stops if it cannot find the
NVIDIA/Mellanox helper or the interface.

## 6. Build and test the Docker image

Install the local Docker Compose settings for Spark:

```bash
cp deployment/cusat/configs/spark/runtime.env \
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

All tests must pass. UHD must find the correct radio. The Spark host is now
ready. Continue with the [two-radio CCM guide](01-ccm-bringup.md).
