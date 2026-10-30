# Standalone cuSAT deployment

This directory contains the supported deployment workflows for the standalone
cuSAT DVB-S2X stack:

- [`docs/tutorials/setup-orin.md`](../../docs/tutorials/setup-orin.md),
  [`setup-thor.md`](../../docs/tutorials/setup-thor.md), and
  [`setup-spark.md`](../../docs/tutorials/setup-spark.md) prepare Docker hosts.
- [`docs/tutorials/01-ccm-bringup.md`](../../docs/tutorials/01-ccm-bringup.md)
  is the end-to-end two-radio CCM validation guide.
- [`docs/tutorials/02-acm-orbital-channels.md`](../../docs/tutorials/02-acm-orbital-channels.md)
  adds ACM and GEO/LEO/MEO channel emulation.
- [`docs/tutorials/04-openshift.md`](../../docs/tutorials/04-openshift.md)
  documents the cluster workflow.

- `docker/` builds and runs cuSAT with Docker Compose on NVIDIA GPU hosts;
  reviewed Thor, Orin NX, and Spark endpoint profiles are under `configs/`.
- `kubernetes/` builds and deploys cuSAT on the tested GH200 OpenShift setup.
