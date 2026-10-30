# Docker Compose deployment

Each radio endpoint runs the same single-service Compose deployment. First
complete that host's setup guide; it installs a platform-specific
`deployment/cusat/docker/platform.env` and builds `cusat:local`:

- [Orin NX](../../../docs/tutorials/setup-orin.md)
- [Thor](../../../docs/tutorials/setup-thor.md)
- [Spark](../../../docs/tutorials/setup-spark.md)

Then copy the selected endpoint role to the untracked local configuration:

```bash
cp deployment/cusat/configs/thor/ccm/node0.yaml deployment/cusat/docker/node.yaml
# On the peer Spark:
# cp deployment/cusat/configs/spark/ccm/node1.yaml deployment/cusat/docker/node.yaml
# Or use deployment/cusat/configs/orin/ccm/node0.yaml on an Orin endpoint 0.
$EDITOR deployment/cusat/docker/node.yaml
```

The supplied templates are ready-to-run at QPSK 1/4 and 4.096 SPS, within a
1-GbE USRP link budget. Set `radio.usrp`; review the connected RF ports through
`radio.tx_channel` and `radio.rx_channel`. For an X410, keep
`subdev_spec: "A:0 A:1 B:0 B:1"`; the application reports the logical-to-RF
port mapping at startup.

The released OTA profiles have been validated with X410s. Since the radio
plugin uses UHD, other UHD-supported USRPs are expected to work when their
device arguments, channel/subdevice mapping, supported sample rate, and
transport settings are reflected in `node.yaml`.

UHD transmission is paced automatically against the radio clock, so no
additional real-time pacing option is required in `node.yaml`.

Each platform directory supplies complete `ccm/`, `geo-acm/`, and `leo-acm/`
node roles. Select the platform first and then the scenario and role, for
example `deployment/cusat/configs/thor/geo-acm/node0.yaml` on a Thor node0.
The adjacent `runtime.env` supplies that platform's CUDA architecture and CPU
set.

Start the already-built modem image in the foreground:

```bash
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml up
```

The platform file selects the reviewed CUDA architecture and container CPU
set. Matching boot and IRQ profiles are installed by the platform guide and
supplied under `deployment/cusat/host`.
Stop the endpoint with Ctrl-C, then remove it with:

```bash
sudo docker compose \
  --env-file deployment/cusat/docker/platform.env \
  -f deployment/cusat/docker/compose.yaml down
```

See the [CCM tutorial](../../../docs/tutorials/01-ccm-bringup.md) for TAP
addressing through `scripts/setup-network.sh`, key configuration fields, and
ping/iperf. Channel emulation is in
the [ACM/orbit tutorial](../../../docs/tutorials/02-acm-orbital-channels.md).
