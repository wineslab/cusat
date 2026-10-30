# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# Portable ARM64 CUDA release image for Spark and Jetson platforms.
# The platform.env selected by the host guide supplies the GPU architecture
# because GPUs are not visible during a normal Docker image build.
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES

RUN test -n "${CUDA_ARCHITECTURES}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake libboost-test-dev libuhd-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -S . -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DENABLE_UHD=ON \
      -DNATIVE_OPTIMIZATIONS=OFF \
    && cmake --build /build --parallel "$(nproc)"

FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       cmake iproute2 iputils-ping iperf3 libboost-test1.83.0 libuhd4.6.0 \
       uhd-host \
       openssh-client procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src /src
COPY --from=build /build /build
WORKDIR /src
ENV CUSAT_BUILD_DIR=/build
CMD ["sleep", "infinity"]
