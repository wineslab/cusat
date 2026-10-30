# cuSat

GPU-accelerated DVB-S2/S2X software transceiver in C++17/CUDA.

The self-contained DSP library (`lib/cusat_dsp/`) provides a CUDA pipeline and
standalone applications for bidirectional links on NVIDIA GPUs and UHD radios.

Start with the [documentation](docs/index.md): prepare each Orin, Thor, or
Spark Docker host once, then use the
[two-radio CCM guide](docs/tutorials/01-ccm-bringup.md). ACM/orbital channels,
OAI backhaul, and OpenShift each have a separate operator guide.

## Features

- All 28 DVB-S2 MODCODs (QPSK 1/4 through 32APSK 9/10) in normal and short FECFRAMEs, plus 38 DVB-S2X normal-FECFRAME MODCODs (QPSK 13/45 through 256APSK 3/4, including the -L variants)
- Full-GPU TX pipeline: BCH + LDPC (IRA) encode, bit interleave, constellation mapping, PL framing/scrambling, RRC pulse shaping — batched with double buffering
- Full-GPU RX pipeline: AGC + rotation, polyphase symbol timing recovery, PLHEADER search, pilot-aided phase tracking, batched soft demapping, LDPC decode, BCH syndrome check
- ACM (adaptive coding and modulation) with in-band QPSK 1/4 signaling, per-MODCOD SNR thresholds, FER feedback
- GSE encapsulation with TAP networking (IP over DVB-S2)
- Configurable symbol rates: TX Farrow resampling to the hardware rate; RX processes at the hardware rate directly via a fractional-stride timing NCO
- Pilot-based SNR estimation (modcod-independent, accurate down to LDPC threshold)
- YAML configuration with CLI override
- TX chain bit-exact against the DVB.org DVB-S2X V&V test vectors at every tap through the transmitted PLFRAME (`dvbs2x-vv-check`, `scripts/run_vv.sh`; the vectors are distributed by DVB and not included)

## Docker deployment

The supported physical-host workflow requires Docker Engine, the Docker
Compose plugin, the NVIDIA Container Toolkit, a supported NVIDIA GPU, and a
UHD-supported USRP accessible from the host. OTA validation currently covers
USRP X410s; other UHD radios require device-specific channel, rate, and
transport configuration. Compiler, CUDA toolkit, UHD, Boost, and CMake
dependencies are installed and pinned inside the image build.

Prepare each host through its platform guide; each guide installs the reviewed
kernel/IRQ profile, builds `cusat:local`, and runs the container tests:

- [Jetson Orin NX](docs/tutorials/setup-orin.md)
- [Jetson Thor](docs/tutorials/setup-thor.md)
- [DGX Spark](docs/tutorials/setup-spark.md)

After preparing any two hosts, the [CCM guide](docs/tutorials/01-ccm-bringup.md)
assigns node0/node1 roles and uses the same Compose command on both.

### OpenShift on GH200

From a workstation authenticated with `oc`, build the exact local checkout and
deploy two long-running GH200/USRP test pods. The binary build avoids
repository credentials and private branch names. The cluster must provide an
ARM64 CUDA/UHD 4.8 base image and a network attachment for the radios; both
are passed to the release script:

```bash
BUILD_NS=my-builds RUN_NS=my-runtime NAME=cusat \
RADIO_NETWORK=my-radio-network BASE_IMAGE=<cuda-uhd-image> \
  ./deployment/cusat/kubernetes/release.sh
```

## Applications

| Binary | Purpose |
|--------|---------|
| `dvbs2-uhd-trx` | Full FDD transceiver on one USRP: simultaneous TX+RX, ACM, TAP networking |
| `dvbs2-gse-tx` | PDU/GSE → IQ samples (stdin/stdout pipe transmitter) |
| `dvbs2-gse-rx` | IQ samples → PDU/GSE (stdin/stdout pipe receiver) |
| `dvbs2-gse-acm` | Launches `dvbs2-uhd-trx` in ACM mode over the TCP radio driver (no hardware) |
| `bench-ldpc-decode` | Standalone LDPC decoder throughput benchmark |
| `bench-coded-modulation` | Coded-modulation AWGN FER curves through the real mapper, demapper and decoder |
| `bench-timing-recovery` | Host launch-cost micro-benchmark of the GPU timing recovery |
| `iq-farrow-resample` | Resample an IQ file between samples-per-symbol ratios |
| `dvbs2x-vv-check` | Conformance check of the TX chain against a DVB.org V&V test case |

### Transceiver (USRP)

Docker Compose starts `/build/apps/dvbs2-uhd-trx` with the reviewed local YAML
mounted at `/etc/cusat/node.yaml`. With a TAP configured, IP packets are
GSE-encapsulated and transmitted; received frames are decapsulated onto the
peer TAP. The CCM guide explains the primary radio, RF-chain, rate, MODCOD,
gain, and channel-model fields before first launch.

## Testing

- `sudo docker run --rm --runtime=nvidia cusat:local ctest --test-dir /build --output-on-failure` — unit tests in the release image
- `scripts/loopback-test.sh` — GPU file loopback (`dvbs2-gse-tx` → `dvbs2-gse-rx`)
- `scripts/run_vv.sh <build> <vv-dir>` — conformance against the DVB.org DVB-S2X V&V package (obtained from DVB)
- `scripts/check_ldpc_tables.py` — LDPC table checksums; `--etsi` verifies them against the ETSI PDFs

## Signal Flow

```
TX:  PDUs (TAP/stdin)
       → GSE encapsulation → BBHEADER + BB scramble       [CPU]
       → BCH encode (GPU ≥16-frame batches, else CPU)
       → LDPC encode → bit interleave + constellation map
         → PL frame + pilots + PL scramble → RRC ×SPS     [GPU, fused batch]
       → Farrow resample to hardware rate                 [GPU]
       → UHD send

RX:  UHD recv → SPSC ring (2 s)
       → AGC + frequency rotation                         [GPU]
       → symbol timing recovery at hardware rate          [GPU]
         (fractional-stride NCO; polyphase energy search)
       → PLHEADER search (acquisition)                    [GPU]
       → PLS decode, pilot SNR/phase tracking             [CPU]
       → soft demap + deinterleave (int16 LLRs)           [GPU]
       → LDPC decode                                      [GPU]
       → BCH syndrome check                               [GPU]
       → BB descramble + deheader → GSE decap → TAP       [CPU]
```

## Project Structure

```
apps/
  dvbs2-uhd-trx.cc        # UHD FDD transceiver (TX+RX, ACM, TAP)
  dvbs2-gse-tx.cc         # Pipe transmitter
  dvbs2-gse-rx.cc         # Pipe receiver
  bench_ldpc_decode.cc    # LDPC benchmark
  dvbs2x-vv-check.cc      # DVB.org V&V conformance check
  config_yaml.h           # YAML config parser
  sidecar.py              # Serves the --spectrum FIFO and status over HTTP
lib/cusat_dsp/
  cuda/                   # All CUDA kernels (TX pipeline, timing, header search, demap, LDPC, BCH, ...)
  cuda/ldpc_schedules/    # Conflict-free LDPC layer schedules (generated, scripts/gen_blossom_schedules.py)
  ldpc_tables/            # LDPC parity-check tables (ETSI EN 302 307-1/-2 annexes)
  framing/                # PLFRAME layout, PLHEADER/PLS code, PL scrambling
  modulation/             # Constellations, bit interleaver, data-aided SNR
  fec/                    # BCH code, CRC-8, code dimensions
  plrx/                   # PL receiver: frame sync, PLS, carrier/phase, metrics
  receiver_frontend.*     # RX orchestration: AGC, timing, header search, PL receiver
  transmitter_frontend.*  # TX orchestration
  tx_fec_chain.* / rx_fec_chain.*  # FEC chains (GPU batching, ACM MODCOD pools)
  acm_config.h / acm_controller.h  # MODCOD table + ACM decision logic
  gse.* / bb_*.{h,cc}     # GSE encapsulation, BBFRAME handling
lib/qa_*.cc               # Unit tests (Boost.UTF)
configs/                  # Example node0/node1 YAML configurations
scripts/                  # Setup, loopback and conformance helpers
deployment/               # Docker and Kubernetes/OpenShift workflows
docs/                     # Operator guides and implementation notes
```


## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
