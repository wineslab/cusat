#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build, validate, and run the cuSAT + OAI mobile-backhaul containers.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ACTION=${1:-test}
ROLE=${2:-}
SITE_ENV="$ROOT/deployment/cusat-mobile-backhaul/docker/site.env"

detect_arch() {
    local caps arch
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | sort -u)
    [ -n "$caps" ] || {
        echo "ERROR: cannot detect a GPU; set CUSAT_CUDA_ARCH explicitly" >&2
        return 1
    }
    [ "$(printf '%s\n' "$caps" | wc -l)" -eq 1 ] || {
        echo "ERROR: GPUs have different compute capabilities: $caps" >&2
        return 1
    }
    arch=${caps/./}
    printf '%s' "$arch"
}

export CUSAT_CUDA_ARCH=${CUSAT_CUDA_ARCH:-$(detect_arch)}
export OAI_REPOSITORY=${OAI_REPOSITORY:-https://github.com/duranta-project/openairinterface5g.git}
export OAI_REF=${OAI_REF:-develop}
export OAI_IMAGE_TAG=${OAI_IMAGE_TAG:-develop}

compose=(docker compose --project-directory "$ROOT")
[ ! -f "$SITE_ENV" ] || compose+=(--env-file "$SITE_ENV")
compose+=(-f "$ROOT/deployment/cusat-mobile-backhaul/docker/compose.yaml")

require_site_env() {
    [ -f "$SITE_ENV" ] || {
        echo "ERROR: copy docker/site.env.example to docker/site.env and edit it" >&2
        exit 2
    }
}

usage() {
    cat >&2 <<'EOF'
Usage:
  release.sh build [boat|shore|all]
  release.sh test
  release.sh up boat|shore
  release.sh run boat-dvb|oai|shore-dvb|upf
  release.sh status
  release.sh down
EOF
    exit 2
}

case "$ACTION" in
    build)
        case "${ROLE:-all}" in
            boat) "${compose[@]}" build boat-dvb oai ;;
            shore) "${compose[@]}" build shore-dvb ;;
            all) "${compose[@]}" build cusat oai ;;
            *) usage ;;
        esac
        ;;
    test)
        "${compose[@]}" build cusat oai
        "${compose[@]}" run --rm cusat bash -lc \
            'ctest --test-dir "$CUSAT_BUILD_DIR" --output-on-failure'
        "${compose[@]}" run --rm oai bash -lc \
            'test -x /opt/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem && test -f /opt/usrpproxy-build/libusrpproxy.so && cat /opt/openairinterface5g/.cusat-oai-revision'
        ;;
    up)
        require_site_env
        case "$ROLE" in
            boat|shore) "${compose[@]}" --profile "$ROLE" up -d ;;
            *) usage ;;
        esac
        ;;
    run)
        require_site_env
        case "$ROLE" in
            boat-dvb)
                "${compose[@]}" --profile boat exec boat-dvb \
                    bash /opt/cusat-runtime/dvb-run.sh
                ;;
            oai)
                "${compose[@]}" --profile boat exec oai \
                    bash /opt/cusat-runtime/oai-backhaul-run.sh
                ;;
            shore-dvb)
                "${compose[@]}" --profile shore exec shore-dvb \
                    bash /opt/cusat-runtime/dvb-run.sh
                ;;
            upf)
                "${compose[@]}" --profile shore exec upf \
                    bash /opt/cusat-runtime/upf-run.sh
                ;;
            *) usage ;;
        esac
        ;;
    status)
        "${compose[@]}" --profile boat --profile shore ps
        ;;
    down)
        "${compose[@]}" --profile boat --profile shore down
        ;;
    *)
        usage
        ;;
esac
