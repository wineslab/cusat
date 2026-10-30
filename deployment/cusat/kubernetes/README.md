# Kubernetes and OpenShift deployment

This directory contains the OpenShift release manifest and its wrapper. For
the tested GH200 OpenShift workflow, authenticate `oc` and run from the
repository root:

```bash
BUILD_NS=... RUN_NS=... RADIO_NETWORK=... BASE_IMAGE=... \
  ./deployment/cusat/kubernetes/release.sh
```

The wrapper uploads the exact local checkout as a binary build and deploys the
rendered `release-openshift.yaml` resources. See the
[OpenShift tutorial](../../../docs/tutorials/04-openshift.md) for site
assumptions, configurable names, endpoint injection, and OTA validation.
