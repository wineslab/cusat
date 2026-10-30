# Experimental DIFI compression profiles

The `compfh` radio plugin ports the compressed-fronthaul implementation from
`603795c` (the `fractional-sps-frontend` branch). FPGA sources and the profile
proposal are in [melkael/dvb-compressed-fh, rx-compress](https://github.com/melkael/dvb-compressed-fh/tree/rx-compress), commit `7227c9e`.

The FLEX1 X410 image routes one compressed TX and one compressed RX chain to
any of the four physical RF channels through switchboards. It needs UHD 4.8,
the full DIFI framer, deframer, PL modulator, TX RRC, IQ compressor, and RX RRC.
The original `uhd`, `difi`, and `tcp` plugins remain selectable. Compressed TX
sizes its ring in packed-frame entries and paces generation at the PLFRAME
cadence with sixteen frames of credit, avoiding seconds of queued idle padding.

RX modes are `rx=plain` (sc16), `rx=trim` (sc8), and `rx=dpcm` with `dbits=1..8`.
`shift=0..8` controls quantization before compression. TX modulation compression
uses `--tx-modcomp`, normal frames, Gold code 0, rolloff 0.2, and CCM. It packs
FEC constellation indices; the FPGA maps, scrambles, inserts pilots and shapes
the waveform. `--tx-symbols` supports the separate symbol-TX DIFI image.

For an image containing RX RRC, use `--rx-prefiltered` so the host interpolates
without applying a second matched filter. The comparison uses unity resampling
(`p=1,q=1`) at four samples per symbol, with RX stream commands propagated
through the UHD streamer (`issue=streamer`). The optional rung-3 symbol timing
path (`--rx-symbol-sync`) is experimental and requires a corresponding image.

Example (first Spark, .18 TX2/RX1, gain 0):

```sh
dvbs2-uhd-trx --radio-driver compfh \
  --usrp 'compfh:addr=192.168.40.18,dest_addr=192.168.40.111,dest_port=7777,adapter=sfp1,tx_ant=TX/RX0,rx_ant=RX1,tx_radio=1,tx_chan=0,rx_radio=0,rx_chan=1,rx=trim,shift=6,rolloff=0.2,p=1,q=1,issue=streamer' \
  --tx-modcomp --rx-prefiltered --rate 61.44e6 --sym-rate 15.36e6 \
  --tx-freq 2.05e9 --rx-freq 1.95e9 --tx-gain 0 --rx-gain 20 \
  --modcod qpsk1/2 --rolloff 0.2 --tap dvbs2b0
```

Compression profiles use candidate PCCs `0xDC02` (DPCM) and `0xDC81` (modulation
TX). They preserve DIFI framing but are not standardized compressed DIFI 1.x
payloads. Trim uses eight-bit VRT items; static reconstruction gain is advertised
in dB (`6.0206 * shift`) and configured through UHD. The plugin intentionally
rejects the older basic DIFI framer, which cannot advertise this context.

The UHD-free `qa_difi_compression` test covers quantization and saturation,
DPCM closed-loop reconstruction at all residual widths, the allocation-free RX
decoder, malformed packets, modulation-frame geometry, and TX byte order across
the DIFI deframer. `ctest` includes it. Live hardware comparison uses the laptop
launcher in `~/Documents/demo-cusat/run-compression-demo.sh`.

## Live CCM dashboard

The source bundle is in `dashboard/`; the configured lab copy is
`~/Documents/demo-cusat/`. Run `./run-compression-dashboard.sh` there, then open
`http://localhost:8765`. The launcher starts laptop tmux `cusat-dashboard`, a
modem/iperf session on each Spark, and 10 Mbit/s UDP traffic in both directions.
Use `./run-compression-dashboard.sh --stop` to stop the compression demo.
Open5GS services are left running; starting this demo stops other radio consumers.

The browser has six compression buttons, an added-noise SNR slider (2–26 dB,
plus off), and an eight-step manual MODCOD slider (QPSK 1/2 through 32APSK 9/10).
Controls apply to both directions. Actual TX/RX MODCOD, compression registers,
lock, pilot SNR, FEC errors, delivered IP bytes, clipping, and transport events
are read from the running modem. Disconnected telemetry disables controls.
Fronthaul counters measure received/transmitted DIFI UDP payload bytes, including
DIFI headers and contexts, but excluding UDP/IP/Ethernet/UHD transport overhead.
IP goodput includes IP headers; the offered iperf payload rate is 10 Mbit/s.
FEC error rate is the interval ratio of post-BCH error frames to decoded frames;
no decoded frames is displayed as unavailable, not as zero errors.

Enable manual control using `--manual-modcod --demo-control-port 5560`, or
`modulation.manual: true` and `demo.control_port: 5560` in YAML. This uses the
multi-MODCOD receiver and warmed TX FEC chains while leaving the ACM controller
and in-band rate-selection reports disabled. TX changes at a PLFRAME boundary,
including its mapper interleaver, PL header, packed FPGA prologue, and pacing.
The current implementation supports normal DVB-S2 MODCOD indices 1–28.

Compression commands go to loopback UDP 5558, owned by the RX worker. The worker
stops RX, drains old-format packets, writes SHIFT/DBITS/MODE and DIFI context,
updates its decoder, then resumes RX. A short reception gap is expected; no
FPGA reload, Docker restart, or modem process restart is needed. A failed switch
attempt restores the old format; failed rollback leaves RX stopped. The endpoint
is enabled only when demo control is enabled. UDP 5560 accepts `stats` and
`modcod N`; 5558 accepts `compression MODE SHIFT DBITS` (MODE 0/1/2, SHIFT 0–8,
DBITS 2/4/6/8). These local endpoints are reached through SSH from the laptop.

Noise is injected **after RX decompression**, using `channel.noise: true` and
`channel.side: rx`. It does not model RF noise before quantization. The dashboard
corrects for the existing injector's complex Gaussian variance and shows its
added-noise SNR separately from measured pilot SNR. Applying channel emulation
to packed TX indices is rejected. Processing placement remains the validated
FLEX1 configuration (TX FPGA mapping/pilots/RRC, RX FPGA matched filtering).
Other processing placements require separate validated profiles and are not
advertised as working live switches.

For a new laptop checkout, copy the verified `fpga/manifest.json`, FLEX1 image
pair/timing report, and per-radio stock backups from the provisioned lab folder
beside the scripts. The bundle intentionally excludes FPGA binaries and SSH
credentials. It assumes both Spark installations at
`/home/wineslab/cusat-compression-release`, isolated UHD 4.8 at
`/home/wineslab/cusat-compression-deps/uhd-4.8`, Docker `cusat:local`, and the
existing physical RF wiring. Rebuild the app **and every radio plugin** after
updating the radio interface. No software or FPGA build is performed by the
presentation launcher.

Run `python3 validate-compression-dashboard.py` with the dashboard running to
repeat the hardware sweep. It checks actual RX/TX MODCOD, compression, monotonic
uptime, low-SNR FEC failure, and baseline recovery, and saves JSON measurements.
It restores the baseline even if an assertion fails.

Hardware validation on 2026-10-06: all six compression modes and all eight UI
MODCODs changed live on both X410s, with unchanged host PIDs/container start
times. At 15.36 Msym/s, RX plain IQ was about 2.02 Gb/s and DPCM 2 about
315 Mb/s; DPCM 2 introduced observable FEC errors. In plain QPSK 1/2, an added
noise SNR of 2 dB produced roughly 99% FER and nearly zero goodput; switching
noise off recovered 10 Mbit/s bidirectional traffic. Both Sparks passed all
22 CTest tests. These measurements are demo operating points, not maximum PHY
capacity or a reproduction of paper throughput results.
