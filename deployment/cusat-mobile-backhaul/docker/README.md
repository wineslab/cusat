# Docker build and two-host run

Build both the standalone cuSAT modem image and an OAI gNB image containing
the matching out-of-tree `libusrpproxy.so` plugin:

```bash
./deployment/cusat-mobile-backhaul/docker/release.sh test
```

The default OAI source is the DURANTA `develop` branch. Pin a tag or commit
for a reproducible release with `OAI_REF=<revision>`.

For the tested two-host deployment, copy `site.env.example` to `site.env`.
Then use `release.sh up boat` on Spark and `release.sh up shore` on Thor. The
[OAI backhaul guide](../../../docs/tutorials/03-oai-backhaul.md) gives the
four process commands and the core safety procedure.
