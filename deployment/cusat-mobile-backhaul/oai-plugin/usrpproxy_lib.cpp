// SPDX-License-Identifier: Apache-2.0
//
// usrpproxy_lib.cpp — OAI radio driver that forwards TX/RX IQ to an external
// process (shm_usrp_proxy) via a POSIX shared-memory segment.
//
// The external process owns the UHD handle and is pinned to two isolated
// cores (one TX, one RX). This driver converts OAI IQ into "UHD-ready" SC16
// (same <<4 TX shift as usrp_lib.cpp) and enqueues packets in the TX ring.
// On RX it dequeues packets from the RX ring and shifts right by the
// device-specific amount (4 for B200, 2 for X/N/X4xx).
//
// Built out-of-tree against an OAI source checkout — the user points cmake at
// their OAI via -DOAI_SOURCE_DIR=/path/to/openairinterface5g. No changes to
// OAI itself are required; ship this .so and set OAI's `[loader]` section:
//     loader: { device: { shlibpath = "..."; }; };
//     device: { name = "usrpproxy"; };

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" {
#include "common_lib.h"
#include "common/config/config_userapi.h"
#include "common/config/config_paramdesc.h"
}

#include "shared_ring.h"

#ifndef sizeofArray
#define sizeofArray(x) (sizeof(x)/sizeof(*(x)))
#endif

// This plugin deliberately does NOT use OAI's LOG_I / LOG_W / LOG_E macros,
// to avoid a runtime coupling on the softmodem's log subsystem. We log to
// stderr with a stable prefix.
#define PXY_I(...)  do { std::fprintf(stderr, "[usrpproxy] "       __VA_ARGS__); } while (0)
#define PXY_W(...)  do { std::fprintf(stderr, "[usrpproxy][warn] " __VA_ARGS__); } while (0)
#define PXY_E(...)  do { std::fprintf(stderr, "[usrpproxy][err] "  __VA_ARGS__); } while (0)

namespace {

struct pending_rx_t {
  // Leftovers from the last RX packet we started consuming but didn't finish.
  std::vector<uint8_t> buf;   // holds cc * nsamps_left * 4 bytes, packed per-ant
  int      nb_ant    = 0;
  int      nsamps    = 0;
  int      consumed  = 0;     // samples already returned (per antenna)
  uint64_t timestamp = 0;     // timestamp of the NEXT unreturned sample
  bool     active    = false;
};

struct state_t {
  // Shared memory.
  int       fd          = -1;
  void      *base       = nullptr;
  uint64_t  segment_size = 0;
  usrpproxy_ctl_t  *ctl = nullptr;
  usrpproxy_ring_t *tx  = nullptr;
  usrpproxy_ring_t *rx  = nullptr;

  // Config (from [usrpproxy] section).
  char     *shm_name          = nullptr;
  uint64_t  tx_ring_bytes     = USRPPROXY_DEFAULT_RING_BYTES;
  uint64_t  rx_ring_bytes     = USRPPROXY_DEFAULT_RING_BYTES;
  int       connect_timeout_s = 30;
  int       wait_timeout_us   = 200000;   // max spin on ring-full before drop
  int       tx_sample_advance_override = -1;  // <0 = use MVP-resolved value

  // Cached after CMD_INIT.
  int       rxshift          = 2;
  double    sample_rate      = 0.0;

  // Accounting.
  int64_t   tx_count         = 0;
  int64_t   rx_count         = 0;

  pending_rx_t pending;
};

// ---------- shm attach ------------------------------------------------------

static bool shm_attach(state_t *s)
{
  auto t0 = std::chrono::steady_clock::now();
  while (true) {
    s->fd = shm_open(s->shm_name, O_RDWR, 0);
    if (s->fd >= 0) break;
    if (std::chrono::steady_clock::now() - t0 >
        std::chrono::seconds(s->connect_timeout_s)) {
      PXY_E("shm_open(%s) timed out — is shm_usrp_proxy running?\n",
            s->shm_name);
      return false;
    }
    usleep(100000);
  }

  struct stat st{};
  if (fstat(s->fd, &st) != 0) {
    PXY_E("fstat failed: %s\n", strerror(errno));
    return false;
  }
  s->segment_size = st.st_size;

  s->base = mmap(nullptr, s->segment_size, PROT_READ | PROT_WRITE,
                 MAP_SHARED, s->fd, 0);
  if (s->base == MAP_FAILED) {
    PXY_E("mmap failed: %s\n", strerror(errno));
    s->base = nullptr;
    return false;
  }

  s->ctl = reinterpret_cast<usrpproxy_ctl_t *>(s->base);

  // Poll for the MVP to have finished initializing the control block.
  t0 = std::chrono::steady_clock::now();
  while (s->ctl->magic != USRPPROXY_MAGIC) {
    if (std::chrono::steady_clock::now() - t0 >
        std::chrono::seconds(s->connect_timeout_s)) {
      PXY_E("control block magic not set by MVP\n");
      return false;
    }
    usleep(50000);
  }

  if (s->ctl->version != USRPPROXY_VERSION) {
    PXY_E("version mismatch: driver %u, shm %u\n",
          USRPPROXY_VERSION, s->ctl->version);
    return false;
  }

  s->tx = reinterpret_cast<usrpproxy_ring_t *>(
      reinterpret_cast<uint8_t *>(s->base) + s->ctl->tx_ring_off);
  s->rx = reinterpret_cast<usrpproxy_ring_t *>(
      reinterpret_cast<uint8_t *>(s->base) + s->ctl->rx_ring_off);

  return true;
}

static void shm_detach(state_t *s)
{
  if (s->ctl) s->ctl->driver_alive.store(0, std::memory_order_release);
  if (s->base && s->base != MAP_FAILED)
    munmap(s->base, s->segment_size);
  if (s->fd >= 0) close(s->fd);
  s->base = nullptr;
  s->fd   = -1;
  s->ctl  = nullptr;
  s->tx   = s->rx = nullptr;
}

// ---------- command channel -------------------------------------------------

static int wait_for_mvp(state_t *s)
{
  auto t0 = std::chrono::steady_clock::now();
  while (s->ctl->mvp_alive.load(std::memory_order_acquire) == 0) {
    if (std::chrono::steady_clock::now() - t0 >
        std::chrono::seconds(s->connect_timeout_s)) {
      PXY_E("MVP not alive after %ds\n", s->connect_timeout_s);
      return -1;
    }
    usleep(50000);
  }
  return 0;
}

static int issue_cmd(state_t *s, int cmd, int timeout_s)
{
  s->ctl->cmd.store(cmd, std::memory_order_relaxed);
  uint64_t new_seq = s->ctl->cmd_seq.fetch_add(1, std::memory_order_release) + 1;

  auto t0 = std::chrono::steady_clock::now();
  while (s->ctl->cmd_ack_seq.load(std::memory_order_acquire) < new_seq) {
    if (s->ctl->abort_flag.load(std::memory_order_acquire)) {
      PXY_E("MVP signaled abort while awaiting cmd %d\n", cmd);
      return -1;
    }
    if (std::chrono::steady_clock::now() - t0 >
        std::chrono::seconds(timeout_s)) {
      PXY_E("cmd %d timed out after %ds\n", cmd, timeout_s);
      return -1;
    }
    usleep(10000);
  }
  int rc = s->ctl->cmd_result.load(std::memory_order_acquire);
  if (rc != USRPPROXY_ACK_OK) {
    PXY_E("cmd %d failed (rc=%d)\n", cmd, rc);
    return -1;
  }
  return 0;
}

static void publish_openair0_to_ctl(state_t *s, openair0_config_t *cfg)
{
  s->ctl->sample_rate     = cfg->sample_rate;
  s->ctl->tune_offset     = cfg->tune_offset;
  s->ctl->rx_bw           = cfg->rx_bw;
  s->ctl->tx_bw           = cfg->tx_bw;
  s->ctl->tx_num_channels = cfg->tx_num_channels;
  s->ctl->rx_num_channels = cfg->rx_num_channels;
  s->ctl->clock_source    = cfg->clock_source;
  s->ctl->time_source     = cfg->time_source;
  for (int i = 0; i < USRPPROXY_MAX_CH; i++) {
    s->ctl->tx_freq[i]        = cfg->tx_freq[i];
    s->ctl->rx_freq[i]        = cfg->rx_freq[i];
    s->ctl->tx_gain[i]        = cfg->tx_gain[i];
    s->ctl->rx_gain[i]        = cfg->rx_gain[i];
    s->ctl->rx_gain_offset[i] = cfg->rx_gain_offset[i];
  }
  std::memset(s->ctl->sdr_addrs, 0, sizeof(s->ctl->sdr_addrs));
  if (cfg->sdr_addrs)
    std::strncpy(s->ctl->sdr_addrs, cfg->sdr_addrs, sizeof(s->ctl->sdr_addrs) - 1);
  std::memset(s->ctl->tx_subdev, 0, sizeof(s->ctl->tx_subdev));
  if (cfg->tx_subdev)
    std::strncpy(s->ctl->tx_subdev, cfg->tx_subdev, sizeof(s->ctl->tx_subdev) - 1);
  std::memset(s->ctl->rx_subdev, 0, sizeof(s->ctl->rx_subdev));
  if (cfg->rx_subdev)
    std::strncpy(s->ctl->rx_subdev, cfg->rx_subdev, sizeof(s->ctl->rx_subdev) - 1);
}

// ---------- device callbacks -----------------------------------------------

static int trx_usrpproxy_start(openair0_device_t *device)
{
  state_t *s = (state_t *)device->priv;
  // The server may stop, flush and reschedule a late X410 command up to three
  // times. Do not let the client time out before that bounded recovery ends.
  return issue_cmd(s, USRPPROXY_CMD_START, 15);
}

static int trx_usrpproxy_stop(openair0_device_t *device)
{
  state_t *s = (state_t *)device->priv;
  return issue_cmd(s, USRPPROXY_CMD_STOP, 5);
}

static int trx_usrpproxy_get_stats(openair0_device_t *device)
{
  state_t *s = (state_t *)device->priv;
  PXY_I("stats: tx=%lu rx=%lu under=%lu over=%lu seq=%lu\n",
        (unsigned long)s->ctl->tx_count.load(std::memory_order_relaxed),
        (unsigned long)s->ctl->rx_count.load(std::memory_order_relaxed),
        (unsigned long)s->ctl->underflows.load(std::memory_order_relaxed),
        (unsigned long)s->ctl->overflows.load(std::memory_order_relaxed),
        (unsigned long)s->ctl->seq_errors.load(std::memory_order_relaxed));
  return 0;
}

static int trx_usrpproxy_reset_stats(openair0_device_t *device) { (void)device; return 0; }

static int trx_usrpproxy_set_freq(openair0_device_t *device,
                                  openair0_config_t *cfg)
{
  state_t *s = (state_t *)device->priv;
  for (int i = 0; i < USRPPROXY_MAX_CH; i++) {
    s->ctl->cmd_tx_freq[i] = cfg[0].tx_freq[i];
    s->ctl->cmd_rx_freq[i] = cfg[0].rx_freq[i];
  }
  s->ctl->cmd_tune_offset = cfg[0].tune_offset;
  return issue_cmd(s, USRPPROXY_CMD_SET_FREQ, 5);
}

static int trx_usrpproxy_set_gains(openair0_device_t *device,
                                   openair0_config_t *cfg)
{
  state_t *s = (state_t *)device->priv;
  for (int i = 0; i < USRPPROXY_MAX_CH; i++) {
    s->ctl->cmd_tx_gain[i]        = cfg[0].tx_gain[i];
    s->ctl->cmd_rx_gain[i]        = cfg[0].rx_gain[i];
    s->ctl->cmd_rx_gain_offset[i] = cfg[0].rx_gain_offset[i];
  }
  return issue_cmd(s, USRPPROXY_CMD_SET_GAINS, 5);
}

static void trx_usrpproxy_end(openair0_device_t *device)
{
  if (!device) return;
  state_t *s = (state_t *)device->priv;
  if (!s) return;
  PXY_I("releasing\n");
  issue_cmd(s, USRPPROXY_CMD_END, 5);
  shm_detach(s);
  free(s->shm_name);
  delete s;
  device->priv = nullptr;
  device->trx_start_func        = nullptr;
  device->trx_get_stats_func    = nullptr;
  device->trx_reset_stats_func  = nullptr;
  device->trx_end_func          = nullptr;
  device->trx_stop_func         = nullptr;
  device->trx_set_freq_func     = nullptr;
  device->trx_set_gains_func    = nullptr;
  device->trx_write_init        = nullptr;
}

// ---------- TX / RX --------------------------------------------------------

// Shift int16 I/Q samples left by 4 (OAI 12-bit -> UHD 16-bit MSBs).
static inline void shift_left_4(int16_t *dst, const int16_t *src, int n_int16)
{
  for (int k = 0; k < n_int16; k++) dst[k] = (int16_t)(src[k] << 4);
}

// Shift int16 I/Q samples right by `rxshift` (UHD 16-bit -> OAI low bits).
static inline void shift_right(int16_t *dst, const int16_t *src, int n_int16,
                               int rxshift)
{
  for (int k = 0; k < n_int16; k++) dst[k] = (int16_t)(src[k] >> rxshift);
}

static int trx_usrpproxy_write(openair0_device_t *device,
                               openair0_timestamp_t timestamp,
                               void **buff,
                               int nsamps,
                               int cc,
                               int flags)
{
  state_t *s = (state_t *)device->priv;
  if (cc > USRPPROXY_MAX_CH) {
    PXY_E("cc=%d exceeds %d\n", cc, USRPPROXY_MAX_CH);
    return -1;
  }

  timestamp -= device->openair0_cfg->command_line_sample_advance
             + device->openair0_cfg->tx_sample_advance;

  const size_t per_ant_bytes = (size_t)nsamps * 4;  // SC16 = 4 bytes/sample
  const size_t payload_bytes = per_ant_bytes * cc;

  // Heap-allocated staging — nsamps can be in the hundreds of thousands at
  // very high sample rates, so stack VLAs aren't safe.
  std::vector<uint8_t> pack(payload_bytes);
  for (int i = 0; i < cc; i++) {
    const int16_t *src = (const int16_t *)buff[i];
    int16_t *dst = (int16_t *)(pack.data() + i * per_ant_bytes);
    shift_left_4(dst, src, nsamps * 2);  // nsamps pairs of (I, Q)
  }

  shm_tx_hdr_t hdr;
  hdr.timestamp     = (uint64_t)timestamp;
  hdr.nsamps        = (uint32_t)nsamps;
  hdr.nb_ant        = (uint32_t)cc;
  hdr.flags         = (uint32_t)flags;
  hdr.payload_bytes = (uint32_t)payload_bytes;

  // Spin-wait for space. Typical cadence is slot-rate (1 kHz @ 1ms slots), so
  // this should never actually wait in steady-state if the MVP is draining.
  const int spin_us   = 50;
  const int max_spins = (s->wait_timeout_us > 0)
                            ? (s->wait_timeout_us + spin_us - 1) / spin_us
                            : 10000;
  int spins = 0;
  while (!ring_write2(s->tx, &hdr, sizeof(hdr), pack.data(), payload_bytes)) {
    if (s->ctl->abort_flag.load(std::memory_order_acquire)) return -1;
    if (++spins > max_spins) {
      PXY_E("TX ring full, dropping %d samples at ts=%ld\n",
            nsamps, (long)timestamp);
      return nsamps;  // pretend we sent — OAI won't retry anyway
    }
    usleep(spin_us);
  }

  s->tx_count += nsamps;
  return nsamps;
}

// Returns 0 on success, -1 on abort, -2 on timeout (no header within budget).
static int drain_one_rx_packet(state_t *s, shm_rx_hdr_t *hdr,
                               std::vector<uint8_t> *payload, int timeout_us)
{
  const int spin_us = 50;
  int waited = 0;
  while (ring_readable(s->rx) < sizeof(*hdr)) {
    if (s->ctl->abort_flag.load(std::memory_order_acquire)) return -1;
    if (timeout_us > 0 && waited >= timeout_us) return -2;
    usleep(spin_us);
    waited += spin_us;
  }
  ring_peek(s->rx, hdr, sizeof(*hdr));
  uint64_t total = sizeof(*hdr) + hdr->payload_bytes;

  // Once the header is visible the producer has already published the payload
  // in the same head bump — it must be there.
  while (ring_readable(s->rx) < total) {
    if (s->ctl->abort_flag.load(std::memory_order_acquire)) return -1;
    if (timeout_us > 0 && waited >= timeout_us) return -2;
    usleep(spin_us);
    waited += spin_us;
  }
  payload->resize(hdr->payload_bytes);
  ring_consume(s->rx, sizeof(*hdr));
  ring_peek(s->rx, payload->data(), hdr->payload_bytes);
  ring_consume(s->rx, hdr->payload_bytes);
  return 0;
}

static int trx_usrpproxy_read(openair0_device_t *device,
                              openair0_timestamp_t *ptimestamp,
                              void **buff,
                              int nsamps,
                              int cc)
{
  state_t *s = (state_t *)device->priv;
  int got = 0;
  bool timestamp_set = false;

  // Step 1: drain leftovers from a previously partially-consumed packet.
  while (got < nsamps && s->pending.active) {
    int left = s->pending.nsamps - s->pending.consumed;
    int take = nsamps - got;
    if (take > left) take = left;

    const int cc_eff = std::min(cc, s->pending.nb_ant);
    const int per_ant_stride = s->pending.nsamps * 4;
    for (int a = 0; a < cc_eff; a++) {
      const int16_t *src = (const int16_t *)(
          s->pending.buf.data() + (size_t)a * per_ant_stride +
          (size_t)s->pending.consumed * 4);
      int16_t *dst = (int16_t *)buff[a] + (size_t)got * 2;
      shift_right(dst, src, take * 2, s->rxshift);
    }
    if (!timestamp_set) {
      *ptimestamp = (openair0_timestamp_t)s->pending.timestamp;
      timestamp_set = true;
    }
    s->pending.consumed  += take;
    s->pending.timestamp += (uint64_t)take;
    got                  += take;
    if (s->pending.consumed >= s->pending.nsamps) {
      s->pending.active = false;
      s->pending.buf.clear();
    }
  }

  // Step 2: pull fresh packets from the ring.
  // First call after CMD_START may take seconds (MVP waits a PPS edge + 1s
  // schedules STREAM_START, then UHD needs to flush). Be patient for the
  // first read; tighten the budget afterwards so OAI exits quickly if the
  // MVP stalls.
  const int RX_TIMEOUT_FIRST = 5'000'000;   // 5 s
  const int RX_TIMEOUT_STEADY = 1'000'000;  // 1 s
  const int per_packet_timeout =
      (s->rx_count == 0) ? RX_TIMEOUT_FIRST : RX_TIMEOUT_STEADY;
  std::vector<uint8_t> pkt;
  while (got < nsamps) {
    shm_rx_hdr_t hdr;
    int rc = drain_one_rx_packet(s, &hdr, &pkt, per_packet_timeout);
    if (rc == -1) return -1;  // abort
    if (rc == -2) {
      PXY_W("RX stalled after %d/%d samples (MVP not producing "
                "— check shm_usrp_proxy telemetry). Returning partial read so "
                "OAI can react.\n",
            got, nsamps);
      if (got == 0) *ptimestamp = 0;
      return got;
    }
    if (hdr.err_code != 0) {
      PXY_W("RX packet err_code=%u, ts=%lu, nsamps=%u\n",
            hdr.err_code, (unsigned long)hdr.timestamp, hdr.nsamps);
    }

    const int cc_eff = std::min((int)hdr.nb_ant, cc);
    const int per_ant_stride = (int)hdr.nsamps * 4;  // bytes

    int take = (int)hdr.nsamps;
    if (take > nsamps - got) take = nsamps - got;

    for (int a = 0; a < cc_eff; a++) {
      const int16_t *src = (const int16_t *)(pkt.data() +
                                             (size_t)a * per_ant_stride);
      int16_t *dst = (int16_t *)buff[a] + (size_t)got * 2;
      shift_right(dst, src, take * 2, s->rxshift);
    }
    if (!timestamp_set) {
      *ptimestamp = (openair0_timestamp_t)hdr.timestamp;
      timestamp_set = true;
    }
    got += take;

    // Remainder goes to pending.
    if (take < (int)hdr.nsamps) {
      s->pending.active    = true;
      s->pending.nsamps    = (int)hdr.nsamps;
      s->pending.nb_ant    = (int)hdr.nb_ant;
      s->pending.consumed  = take;
      s->pending.timestamp = hdr.timestamp + (uint64_t)take;
      s->pending.buf       = std::move(pkt);
      pkt.clear();
    }
  }

  s->rx_count += got;
  return got;
}

// ---------- config -----------------------------------------------------------

#define USRPPROXY_SECTION          "usrpproxy"
#define USRPPROXY_OPT_SHM_NAME     "shm_name"
#define USRPPROXY_OPT_TX_RING_SZ   "tx_ring_size"
#define USRPPROXY_OPT_RX_RING_SZ   "rx_ring_size"
#define USRPPROXY_OPT_CONNECT_TMO  "connect_timeout_s"
#define USRPPROXY_OPT_WAIT_US      "wait_timeout_us"
#define USRPPROXY_OPT_TX_SAMP_ADV  "tx_sample_advance"

static void read_usrpproxy_config(state_t *s)
{
  char *shm_name = nullptr;
  uint64_t tx_sz = USRPPROXY_DEFAULT_RING_BYTES;
  uint64_t rx_sz = USRPPROXY_DEFAULT_RING_BYTES;
  int connect_tmo = 30;
  int wait_us = 200000;
  int tx_samp_adv_override = -1;

  paramdef_t params[] = {
    STRINGPARAM(USRPPROXY_OPT_SHM_NAME,
                "POSIX shm name (leading slash added if missing)",
                0, &shm_name, "/usrpproxy"),
    UINT64PARAM(USRPPROXY_OPT_TX_RING_SZ,
                "TX ring capacity bytes (power of 2)", 0, &tx_sz,
                USRPPROXY_DEFAULT_RING_BYTES),
    UINT64PARAM(USRPPROXY_OPT_RX_RING_SZ,
                "RX ring capacity bytes (power of 2)", 0, &rx_sz,
                USRPPROXY_DEFAULT_RING_BYTES),
    INTPARAM(USRPPROXY_OPT_CONNECT_TMO,
             "seconds to wait for shm_usrp_proxy", 0, &connect_tmo, 30),
    INTPARAM(USRPPROXY_OPT_WAIT_US,
             "microseconds to spin on ring-full before dropping TX samples",
             0, &wait_us, 200000),
    INTPARAM(USRPPROXY_OPT_TX_SAMP_ADV,
             "override tx_sample_advance (<0: keep value resolved by MVP)",
             0, &tx_samp_adv_override, -1),
  };

  int ret = config_get(config_get_if(), params, sizeofArray(params),
                       USRPPROXY_SECTION);
  if (ret < 0) {
    PXY_W("[usrpproxy] section missing, using defaults\n");
  }

  // Normalize shm_name: POSIX shm_open requires a leading slash.
  const char *raw = shm_name ? shm_name : "/usrpproxy";
  if (raw[0] == '/') {
    s->shm_name = strdup(raw);
  } else {
    size_t n = strlen(raw);
    s->shm_name = (char *)malloc(n + 2);
    s->shm_name[0] = '/';
    memcpy(s->shm_name + 1, raw, n + 1);
  }
  s->tx_ring_bytes = tx_sz;
  s->rx_ring_bytes = rx_sz;
  s->connect_timeout_s = connect_tmo;
  s->wait_timeout_us = wait_us;
  s->tx_sample_advance_override = tx_samp_adv_override;

  PXY_I("config: shm=%s tx_ring=%lu rx_ring=%lu connect=%ds "
            "wait=%dus tx_adv_override=%d\n",
        s->shm_name, (unsigned long)tx_sz, (unsigned long)rx_sz,
        connect_tmo, wait_us, tx_samp_adv_override);
}

} // anonymous namespace

extern "C" {

int device_init(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  if (device->priv != nullptr) {
    PXY_W("device already initialized\n");
    return 0;
  }

  state_t *s = new state_t();
  device->priv = s;

  read_usrpproxy_config(s);

  if (!shm_attach(s)) {
    PXY_E("failed to attach shared memory '%s'\n", s->shm_name);
    delete s;
    device->priv = nullptr;
    return -1;
  }

  s->ctl->driver_alive.store(1, std::memory_order_release);

  if (wait_for_mvp(s) < 0) {
    shm_detach(s);
    delete s;
    device->priv = nullptr;
    return -1;
  }

  // Sanity check: capacities the MVP allocated. We warn if the user's config
  // asked for more than what the MVP gave us, but we don't fail — the MVP is
  // the authority on segment sizing.
  if (s->ctl->tx_ring_capacity < s->tx_ring_bytes ||
      s->ctl->rx_ring_capacity < s->rx_ring_bytes) {
    PXY_W("shm ring sizes (tx=%lu rx=%lu) smaller than config "
              "(tx=%lu rx=%lu) — MVP wins\n",
          (unsigned long)s->ctl->tx_ring_capacity,
          (unsigned long)s->ctl->rx_ring_capacity,
          (unsigned long)s->tx_ring_bytes, (unsigned long)s->rx_ring_bytes);
  }

  device->openair0_cfg = openair0_cfg;
  publish_openair0_to_ctl(s, openair0_cfg);

  // Ask MVP to open USRP with this config.
  if (issue_cmd(s, USRPPROXY_CMD_INIT, 60) < 0) {
    PXY_E("CMD_INIT failed (MVP could not open USRP)\n");
    shm_detach(s);
    delete s;
    device->priv = nullptr;
    return -1;
  }

  // Pull back resolved device-specific fields.
  const int32_t dev_type    = s->ctl->resolved_dev_type.load(std::memory_order_acquire);
  const int32_t tx_adv      = s->ctl->resolved_tx_sample_advance.load(std::memory_order_acquire);
  const int32_t rxshift     = s->ctl->resolved_rx_shift.load(std::memory_order_acquire);

  device->type = (dev_type_t)dev_type;
  const int eff_tx_adv = (s->tx_sample_advance_override >= 0)
                             ? s->tx_sample_advance_override
                             : tx_adv;
  openair0_cfg[0].tx_sample_advance = eff_tx_adv;
  s->rxshift = rxshift;
  s->sample_rate = openair0_cfg[0].sample_rate;

  // Pull the per-channel calibration offsets the MVP computed from the
  // device's calib table, so OAI downstream code (e.g. RSRP reporting) sees
  // the same numbers as the direct USRP driver.
  for (int i = 0; i < USRPPROXY_MAX_CH; i++) {
    openair0_cfg[0].rx_gain_offset[i] = s->ctl->resolved_rx_gain_offset[i];
  }

  PXY_I("MVP resolved dev_type=%d tx_sample_advance=%d "
            "(override=%d eff=%d) rxshift=%d\n",
        dev_type, tx_adv, s->tx_sample_advance_override, eff_tx_adv, rxshift);

  // Install function pointers.
  device->trx_start_func       = trx_usrpproxy_start;
  device->trx_stop_func        = trx_usrpproxy_stop;
  device->trx_end_func         = trx_usrpproxy_end;
  device->trx_get_stats_func   = trx_usrpproxy_get_stats;
  device->trx_reset_stats_func = trx_usrpproxy_reset_stats;
  device->trx_set_freq_func    = trx_usrpproxy_set_freq;
  device->trx_set_gains_func   = trx_usrpproxy_set_gains;
  device->trx_write_func       = trx_usrpproxy_write;
  device->trx_read_func        = trx_usrpproxy_read;
  device->trx_write_init       = nullptr;  // no dedicated write thread

  return 0;
}

} // extern "C"
