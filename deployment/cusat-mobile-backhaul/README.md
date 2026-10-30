# cuSAT mobile backhaul

This deployment combines the cuSAT DVB-S2X link with a DURANTA OAI gNB. OAI
uses an out-of-tree `libusrpproxy.so` radio plugin to share an X410 with cuSAT
on separate RF chains. No OAI fork or patch is required.

- `docker/`: portable image builds and the tested Spark/Thor run-time files.
- `configs/`: cuSAT radio configurations that enable the OAI proxy tenant.
- `oai-plugin/`: the OAI-side shared-memory radio plugin.

The build defaults to the public DURANTA repository and its `develop` branch:
`https://github.com/duranta-project/openairinterface5g`.

See the [OAI backhaul guide](../../docs/tutorials/03-oai-backhaul.md) for the
source build, Docker options, four-terminal bring-up and path verification.
