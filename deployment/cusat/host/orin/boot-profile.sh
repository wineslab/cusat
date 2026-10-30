#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Apply the tested 8-core Orin NX CPU-isolation profile through the shared
# Jetson extlinux editor. The wrapper keeps the public command platform-local.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CUSAT_JETSON_PLATFORM=Orin
export CUSAT_JETSON_CPU_RANGE=2-7
export CUSAT_JETSON_IRQAFFINITY=0-1
export CUSAT_JETSON_PROFILE_COMMAND=deployment/cusat/host/orin/boot-profile.sh
exec "$script_dir/../thor/boot-profile.sh" "$@"
