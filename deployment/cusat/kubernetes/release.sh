#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the exact local checkout and deploy two long-running GH200 test pods.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BUILD_NS=${BUILD_NS:?set BUILD_NS to the namespace that builds the image}
RUN_NS=${RUN_NS:?set RUN_NS to the namespace that runs the pods}
NAME=${NAME:-cusat-release}
RADIO_NETWORK=${RADIO_NETWORK:?set RADIO_NETWORK to the NetworkAttachmentDefinition of the radio network}
BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to an ARM64 CUDA + UHD 4.8 base image pull spec}
NO_CACHE=${NO_CACHE:-0}
STAGING=$(mktemp -d "${TMPDIR:-/tmp}/cusat-openshift-source.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT

for value in "$BUILD_NS" "$RUN_NS" "$NAME" "$RADIO_NETWORK"; do
    case "$value" in
        ''|*[!a-z0-9.-]*)
            echo "ERROR: OpenShift names must contain only lowercase letters, digits, dots, and hyphens: $value" >&2
            exit 2
            ;;
    esac
done

case "$NO_CACHE" in
    0) build_cache_args=() ;;
    1) build_cache_args=(--no-cache) ;;
    *) echo "ERROR: NO_CACHE must be 0 or 1" >&2; exit 2 ;;
esac

# `oc start-build --from-dir` does not honor .dockerignore.  Stage the binary
# build context explicitly so local measurements and multi-GB experiment data
# cannot be uploaded by accident.
rsync -a --exclude-from="$ROOT/.dockerignore" "$ROOT/" "$STAGING/"

manifest="$STAGING/release-openshift.rendered.yaml"
sed -e "s/__BUILD_NS__/$BUILD_NS/g" \
    -e "s/__RUN_NS__/$RUN_NS/g" \
    -e "s/__NAME__/$NAME/g" \
    -e "s/__RADIO_NETWORK__/$RADIO_NETWORK/g" \
    -e "s|__BASE_IMAGE__|$BASE_IMAGE|g" \
    "$ROOT/deployment/cusat/kubernetes/release-openshift.yaml" > "$manifest"

oc apply -f "$manifest"
build=$(oc start-build "$NAME" -n "$BUILD_NS" --from-dir="$STAGING" \
    "${build_cache_args[@]}" -o name)
echo "Started $build"
while :; do
    phase=$(oc get -n "$BUILD_NS" "$build" -o jsonpath='{.status.phase}')
    case "$phase" in
        Complete) break ;;
        Failed|Error|Cancelled)
            oc logs -n "$BUILD_NS" "$build" --tail=200 || true
            echo "ERROR: $build ended in phase $phase" >&2
            exit 1
            ;;
        *) echo "Waiting for $build ($phase)"; sleep 10 ;;
    esac
done
echo "$build completed"
oc rollout restart "statefulset/$NAME" -n "$RUN_NS"
oc rollout status "statefulset/$NAME" -n "$RUN_NS" --timeout=10m

for pod in "$NAME-0" "$NAME-1"; do
    echo "=== $pod ==="
    oc exec -n "$RUN_NS" "$pod" -- nvidia-smi -L
    oc exec -n "$RUN_NS" "$pod" -- \
        /workspace/cusat-release/build/release/apps/bench-ldpc-decode \
        --modcod qpsk1/4 --batch-size 4 --batches 2
done
