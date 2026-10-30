# cuSAT documentation

cuSAT is a GPU-accelerated, bidirectional DVB-S2/S2X transceiver with GSE
networking, CCM and closed-loop ACM. Start with the conservative two-radio CCM
link, then add adaptation, orbital channel emulation, or 5G backhaul.

## Prepare Docker endpoints

Prepare each physical host once, including its kernel/IRQ profile, X410
network, local image, and container tests:

- [Jetson Orin NX](tutorials/setup-orin.md)
- [Jetson Thor](tutorials/setup-thor.md)
- [DGX Spark](tutorials/setup-spark.md)

## Run the system tutorials

1. [Connect two prepared endpoints with CCM](tutorials/01-ccm-bringup.md) —
   assign node0/node1 roles, review the key radio and waveform options, then
   validate with ping and iperf3.
2. [Run ACM over orbital channel models](tutorials/02-acm-orbital-channels.md)
   — OLLA, configurable guard, GEO/LEO/MEO models, and the
   UDP distance controller.
3. [Carry OAI 5G backhaul over cuSAT](tutorials/03-oai-backhaul.md) — share the
   boat radio with OAI and force N2/N3 through the DVB link to a shore UPF.
4. [Build and run on OpenShift](tutorials/04-openshift.md) — binary builds,
   long-running GPU/radio pods, configuration injection, and validation.

The first two system tutorials use Docker Compose on two prepared physical
NVIDIA hosts. The OpenShift workflows document the additional assumptions
made by the cuSAT laboratory deployment.

## Technical notes

- [DVB-S2X specification tables](dvbs2x-spec-tables.md)
