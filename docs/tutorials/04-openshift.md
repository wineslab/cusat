# Build and run cuSAT on OpenShift

Use this guide to build the current local source on OpenShift. The procedure
uses an OpenShift binary build. It then makes two long-running pods that use
GPUs and radios.

This procedure is the cluster equivalent of the Docker Compose procedure.
For the OAI backhaul procedure, refer to the
[OAI backhaul guide](03-oai-backhaul.md).

## Cluster requirements

The release has been tested on a GH200 OpenShift cluster with USRP X410
radios. It requires these resources:

- an ARM64 CUDA and UHD 4.8 base image that the build service account can
  pull (`BASE_IMAGE`);
- the `nvidia.com/gpu` GPU resource.
- the `openshift.io/vfgh` radio SR-IOV resource.
- a Multus network for the radios (`RADIO_NETWORK`).
- privileged pods with `NET_ADMIN` for TAP interfaces.
- two schedulable nodes.
- one GPU and one radio VF on each node.
- eight CPU cores and 32 GiB of memory on each node.

You can set the build namespace, run-time namespace, workload name, and
Multus network in the release script. For a different architecture or
resource name, change the manifest and the base image.

## 1. Authenticate and examine the requirements

```bash
oc whoami
oc get nodes
oc get network-attachment-definitions -A
```

Verify that the build service account can pull the base image. Verify
that the run-time service account can pull the new image.

## 2. Examine the resources and start the release

The release script uses `.dockerignore` to stage the local source. It does not
upload experiment data or build directories.

Before you adapt the release for a different site, examine the parameterized
resource file:

```bash
sed -n '1,260p' deployment/cusat/kubernetes/release-openshift.yaml
```

Set the namespaces, the release name and the radio network of your site once;
the rest of this guide uses these variables:

```bash
export BUILD_NS=my-builds          # namespace that builds the image
export RUN_NS=my-runtime           # namespace that runs the pods
export NAME=cusat-release          # StatefulSet name; pods are $NAME-0 and $NAME-1
export RADIO_NETWORK=my-radio-net  # NetworkAttachmentDefinition of the USRP network
export BASE_IMAGE=image-registry.openshift-image-registry.svc:5000/my-builds/cuda-uhd:latest
./deployment/cusat/kubernetes/release.sh
```

The script does these operations:

1. It makes or updates an ImageStream and a BuildConfig.
2. It uploads and builds the current source tree.
3. It starts a StatefulSet with two replicas.
4. It runs CTest in each pod.
5. It keeps both containers active.

## 3. Examine the build and the pods

```bash
oc get builds -n "$BUILD_NS" -l buildconfig="$NAME"
oc get pods -n "$RUN_NS" -l app="$NAME" -o wide
oc logs -n "$RUN_NS" "$NAME-0"
oc logs -n "$RUN_NS" "$NAME-1"
```

Both pods must show `Ready`. Each pod must show a visible NVIDIA GPU. All
tests must pass.

## 4. Install the endpoint configuration files

For the tested GH/X410 frequency plan, start with the general configuration
files. Make local copies and edit them before you copy them to the pods:

```bash
cp configs/node0.yaml /tmp/cusat-node0.yaml
cp configs/node1.yaml /tmp/cusat-node1.yaml
${EDITOR:-nano} /tmp/cusat-node0.yaml
${EDITOR:-nano} /tmp/cusat-node1.yaml

oc cp /tmp/cusat-node0.yaml "$RUN_NS/$NAME-0":/tmp/node.yaml
oc cp /tmp/cusat-node1.yaml "$RUN_NS/$NAME-1":/tmp/node.yaml
```

Examine these settings in both files:

- USRP IP address.
- paired TX and RX frequencies.
- TX and RX gains.
- logical RF channels.
- sample rate and symbol rate.
- TAP interface names.
- `acm.enabled`.

For the first cluster test, use QPSK 1/4 and disable ACM.

## 5. Start one endpoint in each terminal

In terminal 1, start endpoint 0:

```bash
oc exec -it -n "$RUN_NS" "$NAME-0" -- \
  /workspace/cusat-release/build/release/apps/dvbs2-uhd-trx \
  --config /tmp/node.yaml
```

In terminal 2, start endpoint 1:

```bash
oc exec -it -n "$RUN_NS" "$NAME-1" -- \
  /workspace/cusat-release/build/release/apps/dvbs2-uhd-trx \
  --config /tmp/node.yaml
```

## 6. Configure and test the TAP interfaces

Configure endpoint 0:

```bash
oc exec -n "$RUN_NS" "$NAME-0" -- bash -lc '
  ip addr replace 10.30.0.1/30 dev dvbs2a0
  ip link set dvbs2a0 up
  ip neigh replace 10.30.0.2 lladdr 02:00:00:00:00:02 nud permanent dev dvbs2a0
  iperf3 -s -D'
```

Configure and test endpoint 1:

```bash
oc exec -n "$RUN_NS" "$NAME-1" -- bash -lc '
  ip addr replace 10.30.0.2/30 dev dvbs2b0
  ip link set dvbs2b0 up
  ip neigh replace 10.30.0.1 lladdr 02:00:00:00:00:01 nud permanent dev dvbs2b0
  ping -c 10 10.30.0.1
  iperf3 -c 10.30.0.1 -t 30
  iperf3 -c 10.30.0.1 -R -t 30'
```

## 7. Stop or remove the release

To stop an interactive transceiver, press `Ctrl-C` in its terminal.

To keep the OpenShift resources but release the GPUs and radios, set the
replica count to zero:

```bash
oc scale statefulset/"$NAME" -n "$RUN_NS" --replicas=0
```

To remove only this cuSAT release, run these commands:

```bash
oc delete statefulset/"$NAME" -n "$RUN_NS"
oc delete buildconfig/"$NAME" imagestream/"$NAME" \
  -n "$BUILD_NS"
```

If you used a different `NAME`, `BUILD_NS`, or `RUN_NS`, first use `oc get` to
find the exact resource names. Delete only the resources that you find.
