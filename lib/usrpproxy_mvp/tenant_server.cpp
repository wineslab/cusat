// SPDX-License-Identifier: Apache-2.0
//
// tenant_server.cpp — embedded UHD-host-side implementation of the
// usrpproxy shm-ring protocol. See tenant_server.h for the public API and
// docs/usage-from-host expectations.
//
// This file is a refactor of the standalone shm-usrp-proxy binary from the
// usrpproxy reference repo, dropped down to just the parts that make sense
// to run inside a host process (like dvbs2-uhd-trx) that already owns the
// multi_usrp handle. Device open, CLI parsing, signal handling, and the
// forever-loop in main() are gone.

#define _GNU_SOURCE 1

#include "tenant_server.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <uhd/version.hpp>
#if UHD_VERSION < 3110000
#include <uhd/utils/thread_priority.hpp>
#else
#include <uhd/utils/thread.hpp>
#endif

#include "shared_ring.h"

#define PXY_I(...) do { std::fprintf(stderr, "[usrpproxy/tenant] "       __VA_ARGS__); } while (0)
#define PXY_W(...) do { std::fprintf(stderr, "[usrpproxy/tenant][warn] " __VA_ARGS__); } while (0)
#define PXY_E(...) do { std::fprintf(stderr, "[usrpproxy/tenant][err] "  __VA_ARGS__); } while (0)

namespace usrpproxy {
namespace {

// Mirror of OAI's dev_type_t enum values — must match radio/COMMON/common_lib.h
enum : int32_t {
  MIN_RF_DEV_TYPE = 0,
  USRP_B200_DEV,
  USRP_X300_DEV,
  USRP_N300_DEV,
  USRP_X400_DEV,
};

// Mirror of OAI's radio_tx_burst_flag_t.
enum : uint32_t {
  TX_BURST_INVALID = 0,
  TX_BURST_MIDDLE = 1,
  TX_BURST_START = 2,
  TX_BURST_END = 3,
  TX_BURST_START_AND_END = 4,
  TX_BURST_END_NO_TIME_SPEC = 10,
};

// Per-device RX gain calibration tables — lifted verbatim from
// openairinterface5g/radio/USRP/usrp_lib.cpp so the tenant server picks
// the same offset the direct driver would as a function of rx_freq.
struct calib_row_t { double freq; double offset; };

static const calib_row_t calib_table_b210[]    = {
  {3500000000.0,44.0},{2660000000.0,49.0},{2300000000.0,50.0},
  {1880000000.0,53.0},{816000000.0,58.0},{-1,0}};
static const calib_row_t calib_table_b210_38[] = {
  {3500000000.0,44.0},{2660000000.0,49.8},{2300000000.0,51.0},
  {1880000000.0,53.0},{816000000.0,57.0},{-1,0}};
static const calib_row_t calib_table_x310[]    = {
  {3500000000.0,77.0},{2660000000.0,81.0},{2300000000.0,81.0},
  {1880000000.0,82.0},{816000000.0,85.0},{-1,0}};
static const calib_row_t calib_table_n310[]    = {
  {3500000000.0,0.0},{2660000000.0,0.0},{2300000000.0,0.0},
  {1880000000.0,0.0},{816000000.0, 0.0},{-1,0}};
static const calib_row_t calib_table_none[]    = {
  {3500000000.0,0.0},{2660000000.0,0.0},{2300000000.0,0.0},
  {1880000000.0,0.0},{816000000.0, 0.0},{-1,0}};

static double bw_gain_adjust_for(double sr_hz, int bw_gain_adjust)
{
  if (!bw_gain_adjust) return 0.0;
  switch ((int)sr_hz) {
    case 46080000: return 0.0;
    case 30720000: return 0.0;
    case 23040000: return 1.25;
    case 15360000: return 3.0;
    case 7680000:  return 6.0;
    case 3840000:  return 9.0;
    case 1920000:  return 12.0;
    default:       return 0.0;
  }
}

static double pick_rx_gain_offset(const calib_row_t *table,
                                  double rx_freq, double sample_rate,
                                  int bw_gain_adjust)
{
  double best_off  = 0.0;
  double best_diff = 6e9;
  for (int i = 0; table[i].freq > 0; i++) {
    double diff = std::fabs(rx_freq - table[i].freq);
    if (diff < best_diff) { best_diff = diff; best_off = table[i].offset; }
  }
  return best_off + bw_gain_adjust_for(sample_rate, bw_gain_adjust);
}

static void pin_this_thread(int core, const char *name)
{
  if (core < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0)
    PXY_W("[%s] pin to core %d failed: %s\n", name, core, std::strerror(rc));
  else
    PXY_I("[%s] pinned to core %d\n", name, core);
}

static void bump_rt_priority(const char *name)
{
  struct sched_param sp{};
  sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
    PXY_W("[%s] SCHED_FIFO not granted (need CAP_SYS_NICE) — continuing\n", name);
}

// Infer the OAI dev_type code the direct USRP driver would stamp for this
// USRP. Uses UHD's mboard name. Needed for calib table + tx_sample_advance
// selection at CMD_INIT time. OAI's set_device() refuses MIN_RF_DEV_TYPE
// with "invalid HW device", so any new hardware needs a pattern here.
static int32_t classify_device(uhd::usrp::multi_usrp::sptr usrp)
{
  try {
    std::string mboard = usrp->get_mboard_name(0);
    // Lowercase copy for case-insensitive substring matching.
    std::string m = mboard;
    for (auto &c : m) c = (char)std::tolower((unsigned char)c);
    PXY_I("classify_device: mboard_name=\"%s\"\n", mboard.c_str());

    if (m.find("b200") != std::string::npos ||
        m.find("b210") != std::string::npos) return USRP_B200_DEV;
    if (m.find("n300") != std::string::npos ||
        m.find("n310") != std::string::npos ||
        m.find("n320") != std::string::npos ||
        m.find("n3xx") != std::string::npos) return USRP_N300_DEV;
    if (m.find("x300") != std::string::npos ||
        m.find("x310") != std::string::npos) return USRP_X300_DEV;
    if (m.find("x410") != std::string::npos ||
        m.find("x440") != std::string::npos ||
        m.find("x4xx") != std::string::npos) return USRP_X400_DEV;

    PXY_W("classify_device: unrecognized mboard \"%s\" — OAI will reject "
          "dev_type=0. Add a pattern for this device.\n", mboard.c_str());
  } catch (const std::exception &e) {
    PXY_W("classify_device: get_mboard_name threw: %s\n", e.what());
  } catch (...) {
    PXY_W("classify_device: get_mboard_name threw (unknown exception)\n");
  }
  return MIN_RF_DEV_TYPE;
}

} // anonymous namespace

// =====================================================================
// Private impl
// =====================================================================
struct TenantServer::Impl {
  uhd::usrp::multi_usrp::sptr usrp;
  Config                      cfg;

  // shm segment
  int       fd    = -1;
  void     *base  = nullptr;
  uint64_t  size  = 0;
  std::string name;
  usrpproxy_ctl_t  *ctl = nullptr;
  usrpproxy_ring_t *tx  = nullptr;
  usrpproxy_ring_t *rx  = nullptr;

  // Streamers built on first CMD_INIT.
  uhd::tx_streamer::sptr tx_stream;
  uhd::rx_streamer::sptr rx_stream;
  uhd::tx_metadata_t     tx_md;
  std::atomic<int64_t>   tx_count{0};

  // Resolved device + stream parameters published back to the driver.
  int32_t dev_type          = MIN_RF_DEV_TYPE;
  int32_t rx_shift          = 2;
  int32_t tx_sample_advance = 0;
  double  sample_rate       = 0.0;

  // Physical channel mapping actually bound into the streamers.
  std::vector<int> phys_tx;
  std::vector<int> phys_rx;

  // Control flags used across threads.
  // UHD explicitly supports independent TX and RX streamer calls from
  // different threads.  Do not serialize them behind one mutex: RX recv() is
  // blocking and used to hold TX long enough for every timestamped packet to
  // become late.  The per-direction locks only protect streamer replacement
  // against the corresponding worker.
  std::mutex tx_streamer_mtx;
  std::mutex rx_streamer_mtx;
  std::atomic<bool> streaming{false};
  std::atomic<bool> first_rx_seen{false};
  std::atomic<bool> start_rx_error{false};
  std::atomic<bool> exit_workers{false};
  std::atomic<bool> started{false};
  std::atomic<bool> stopped{false};

  std::thread ctrl_thr;
  std::thread tx_thr;
  std::thread rx_thr;
  std::thread tel_thr;

  // Normalizes shm_name (ensures leading '/'), returns the normalized copy.
  static std::string normalize_shm_name(std::string s) {
    if (!s.empty() && s[0] != '/') s = "/" + s;
    return s;
  }

  bool shm_create()
  {
    name = normalize_shm_name(cfg.shm_name);
    uint64_t total = 0;
    if (!usrpproxy_segment_bytes_checked(cfg.tx_ring_size,
                                         cfg.rx_ring_size, &total)) {
      PXY_E("ring sizes must be powers of two in [4096,%lu] bytes "
            "(got tx=%lu rx=%lu)\n",
            (unsigned long)USRPPROXY_MAX_RING_BYTES,
            (unsigned long)cfg.tx_ring_size, (unsigned long)cfg.rx_ring_size);
      return false;
    }

    if (shm_unlink(name.c_str()) == 0) {
      PXY_I("removed stale shm %s\n", name.c_str());
    } else if (errno != ENOENT) {
      PXY_W("shm_unlink(%s): %s\n", name.c_str(), std::strerror(errno));
    }

    fd = shm_open(name.c_str(), O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (fd < 0) {
      PXY_E("shm_open(%s) failed: %s\n", name.c_str(), std::strerror(errno));
      return false;
    }
    if (ftruncate(fd, (off_t)total) != 0) {
      PXY_E("ftruncate failed: %s\n", std::strerror(errno));
      return false;
    }
    base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
      PXY_E("mmap failed: %s\n", std::strerror(errno));
      base = nullptr;
      return false;
    }
    size = total;
    std::memset(base, 0, total);

    ctl = reinterpret_cast<usrpproxy_ctl_t *>(base);

    const uint64_t page = 4096;
    auto align_up = [&](uint64_t v) { return (v + page - 1) & ~(page - 1); };

    uint64_t off = align_up(sizeof(usrpproxy_ctl_t));
    ctl->tx_ring_off = off;
    off += align_up(sizeof(usrpproxy_ring_t) + cfg.tx_ring_size);
    ctl->rx_ring_off = off;

    ctl->tx_ring_capacity = cfg.tx_ring_size;
    ctl->rx_ring_capacity = cfg.rx_ring_size;
    ctl->segment_size     = total;

    tx = reinterpret_cast<usrpproxy_ring_t *>(
        reinterpret_cast<uint8_t *>(base) + ctl->tx_ring_off);
    rx = reinterpret_cast<usrpproxy_ring_t *>(
        reinterpret_cast<uint8_t *>(base) + ctl->rx_ring_off);

    tx->capacity = cfg.tx_ring_size;
    rx->capacity = cfg.rx_ring_size;

    ctl->version = USRPPROXY_VERSION;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    ctl->magic = USRPPROXY_MAGIC;

    PXY_I("created shm %s (%lu bytes)\n",
          name.c_str(), (unsigned long)size);
    return true;
  }

  void shm_destroy()
  {
    if (ctl) ctl->mvp_alive.store(0, std::memory_order_release);
    if (base && base != MAP_FAILED)
      munmap(base, size);
    if (fd >= 0) close(fd);
    if (!name.empty()) shm_unlink(name.c_str());
    base = nullptr;
    fd   = -1;
    ctl  = nullptr;
    tx   = nullptr;
    rx   = nullptr;
  }

  // Per-channel configuration + streamer (re)build at CMD_INIT.
  bool configure_oai_tenant()
  {
    if (!usrpproxy_channel_count_valid(ctl->tx_num_channels) ||
        !usrpproxy_channel_count_valid(ctl->rx_num_channels)) {
      PXY_E("invalid OAI channel counts tx=%d rx=%d (valid range 1..%d)\n",
            ctl->tx_num_channels, ctl->rx_num_channels, USRPPROXY_MAX_CH);
      return false;
    }

    auto mapping_tx = cfg.tx_channels;
    auto mapping_rx = cfg.rx_channels;
    if (mapping_tx.empty())
      for (int i = 0; i < ctl->tx_num_channels; i++) mapping_tx.push_back(i);
    if (mapping_rx.empty())
      for (int i = 0; i < ctl->rx_num_channels; i++) mapping_rx.push_back(i);

    if (ctl->tx_num_channels > (int)mapping_tx.size()) {
      PXY_E("OAI asked for %d TX channels but only %zu reserved\n",
            ctl->tx_num_channels, mapping_tx.size());
      return false;
    }
    if (ctl->rx_num_channels > (int)mapping_rx.size()) {
      PXY_E("OAI asked for %d RX channels but only %zu reserved\n",
            ctl->rx_num_channels, mapping_rx.size());
      return false;
    }
    mapping_tx.resize(ctl->tx_num_channels);
    mapping_rx.resize(ctl->rx_num_channels);
    phys_tx = mapping_tx;
    phys_rx = mapping_rx;

    // Classify device (for tx_sample_advance + calib + rx_shift).
    dev_type = classify_device(usrp);
    rx_shift = (dev_type == USRP_B200_DEV) ? 4 : 2;

    // Per-device tx_sample_advance table (mirrors usrp_lib.cpp).
    tx_sample_advance = 15;
    if (dev_type == USRP_B200_DEV) {
      switch ((int)ctl->sample_rate) {
        case 46080000: tx_sample_advance = 164; break;
        case 30720000: tx_sample_advance = 115; break;
        case 23040000: tx_sample_advance = 113; break;
        case 15360000: tx_sample_advance = 103; break;
        case 7680000:  tx_sample_advance = 80;  break;
        case 1920000:  tx_sample_advance = 40;  break;
        default:
          PXY_E("unsupported B200 sample rate %f\n", ctl->sample_rate);
          return false;
      }
    } else {
      switch ((int)ctl->sample_rate) {
        case 245760000: case 184320000: case 122880000: case 92160000:
        case 61440000: case 46080000: case 30720000: case 23040000:
          tx_sample_advance = 15; break;
        case 15360000: tx_sample_advance = 45; break;
        case 7680000:  tx_sample_advance = 50; break;
        case 1920000:  tx_sample_advance = 50; break;
        default:
          PXY_E("unsupported sample rate %f\n", ctl->sample_rate);
          return false;
      }
    }
    sample_rate = ctl->sample_rate;

    const calib_row_t *calib = calib_table_none;
    int bw_gain_adjust = 0;
    switch (dev_type) {
      case USRP_B200_DEV: {
        int vers=0, subvers=0, subsubvers=0;
        std::sscanf(uhd::get_version_string().c_str(), "%d.%d.%d",
                    &vers, &subvers, &subsubvers);
        if (vers == 3 && subvers == 9 && subsubvers >= 2) {
          calib = calib_table_b210;    bw_gain_adjust = 0;
        } else {
          calib = calib_table_b210_38; bw_gain_adjust = 1;
        }
        break;
      }
      case USRP_X300_DEV: calib = calib_table_x310; break;
      case USRP_N300_DEV: calib = calib_table_n310; break;
      case USRP_X400_DEV: calib = calib_table_none; break;
      default: break;
    }

    for (int k = 0; k < ctl->rx_num_channels; k++) {
      const int phys = mapping_rx[k];
      if (!cfg.rx_antenna.empty()) {
        usrp->set_rx_antenna(cfg.rx_antenna, phys);
        PXY_I("OAI rx[%d] -> phys%d antenna=%s\n", k, phys,
              usrp->get_rx_antenna(phys).c_str());
      }
      usrp->set_rx_rate(ctl->sample_rate, phys);
      usrp->set_rx_freq(uhd::tune_request_t(ctl->rx_freq[k], ctl->tune_offset), phys);

      const double off = pick_rx_gain_offset(calib, ctl->rx_freq[k],
                                             ctl->sample_rate, bw_gain_adjust);
      ctl->resolved_rx_gain_offset[k] = off;

      auto gr = usrp->get_rx_gain_range(phys);
      double g = ctl->rx_gain[k] - off;
      if (g > gr.stop()) {
        PXY_W("RX gain %.2f (rx_gain=%.2f - off=%.2f) exceeds max %.2f — clamping\n",
              g, ctl->rx_gain[k], off, gr.stop());
        g = gr.stop();
      }
      usrp->set_rx_gain(g, phys);
      PXY_I("OAI rx[%d] -> phys%d  rate=%.2fMHz freq=%.3fGHz gain=%.2f "
            "(rx_gain=%.2f - calib_off=%.2f)\n",
            k, phys, ctl->sample_rate/1e6, ctl->rx_freq[k]/1e9, g,
            ctl->rx_gain[k], off);
    }
    for (int k = 0; k < ctl->tx_num_channels; k++) {
      const int phys = mapping_tx[k];
      if (!cfg.tx_antenna.empty()) {
        usrp->set_tx_antenna(cfg.tx_antenna, phys);
        PXY_I("OAI tx[%d] -> phys%d antenna=%s\n", k, phys,
              usrp->get_tx_antenna(phys).c_str());
      }
      usrp->set_tx_rate(ctl->sample_rate, phys);
      usrp->set_tx_freq(uhd::tune_request_t(ctl->tx_freq[k], ctl->tune_offset), phys);
      auto gr = usrp->get_tx_gain_range(phys);
      usrp->set_tx_gain(gr.stop() - ctl->tx_gain[k], phys);
      PXY_I("OAI tx[%d] -> phys%d  rate=%.2fMHz freq=%.3fGHz tx_gain=%.2f\n",
            k, phys, ctl->sample_rate/1e6, ctl->tx_freq[k]/1e9,
            gr.stop() - ctl->tx_gain[k]);
    }

    {
      std::scoped_lock lock(tx_streamer_mtx, rx_streamer_mtx);
      tx_stream.reset();
      rx_stream.reset();

      uhd::stream_args_t sa_rx("sc16", "sc16");
      for (int phys : mapping_rx) sa_rx.channels.push_back(phys);
      rx_stream = usrp->get_rx_stream(sa_rx);

      uhd::stream_args_t sa_tx("sc16", "sc16");
      for (int phys : mapping_tx) sa_tx.channels.push_back(phys);
      tx_stream = usrp->get_tx_stream(sa_tx);
    }

    // Bandwidths (skip N3xx — set_*_bandwidth there misbehaves per
    // usrp_lib.cpp notes).
    if (dev_type != USRP_N300_DEV) {
      for (int phys : mapping_tx) usrp->set_tx_bandwidth(ctl->tx_bw, phys);
      for (int phys : mapping_rx) usrp->set_rx_bandwidth(ctl->rx_bw, phys);
    }

    PXY_I("OAI configured — tx_adv=%d rxshift=%d\n", tx_sample_advance, rx_shift);
    return true;
  }

  // ---------- worker threads ----------

  void tx_worker_loop(int core)
  {
    pin_this_thread(core, "tx");
    bump_rt_priority("tx");

    std::vector<uint8_t> payload;
    std::vector<void *>  buff_ptrs;

    while (!exit_workers.load(std::memory_order_acquire)) {
      if (!streaming.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }

      shm_tx_hdr_t hdr;
      if (!ring_peek(tx, &hdr, sizeof(hdr))) {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
        continue;
      }
      uint32_t stream_channels = 0;
      {
        std::lock_guard<std::mutex> lock(tx_streamer_mtx);
        stream_channels = tx_stream ? (uint32_t)tx_stream->get_num_channels() : 0;
      }
      if (!usrpproxy_tx_hdr_valid(&hdr, stream_channels, tx->capacity)) {
        PXY_W("tx_worker: dropping malformed TX header "
              "(nsamps=%u nb_ant=%u payload=%u stream_ch=%u cap=%lu)\n",
              hdr.nsamps, hdr.nb_ant, hdr.payload_bytes, stream_channels,
              (unsigned long)tx->capacity);
        if (usrpproxy_tx_packet_shape_valid(&hdr, tx->capacity)) {
          const uint64_t total = sizeof(hdr) + (uint64_t)hdr.payload_bytes;
          if (ring_readable(tx) < total) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
            continue;
          }
          ring_consume(tx, total);
        } else {
          ring_consume(tx, ring_readable(tx));
        }
        ctl->seq_errors.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      const uint64_t total = sizeof(hdr) + hdr.payload_bytes;
      if (ring_readable(tx) < total) {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
        continue;
      }

      payload.resize(hdr.payload_bytes);
      ring_consume(tx, sizeof(hdr));
      ring_peek(tx, payload.data(), hdr.payload_bytes);
      ring_consume(tx, hdr.payload_bytes);

      tx_md.has_time_spec  = true;
      tx_md.time_spec      =
          uhd::time_spec_t::from_ticks((int64_t)hdr.timestamp, sample_rate);
      const uint32_t burst_flag = hdr.flags & 0xf;
      bool first = (burst_flag == TX_BURST_START || burst_flag == TX_BURST_START_AND_END);
      bool last  = (burst_flag == TX_BURST_END   || burst_flag == TX_BURST_START_AND_END
                    || burst_flag == TX_BURST_END_NO_TIME_SPEC);
      int64_t tx_seq = tx_count.fetch_add(1, std::memory_order_relaxed);
      tx_md.start_of_burst = (tx_seq == 0) ? true : first;
      tx_md.end_of_burst   = last;

      const size_t per_ant_bytes = (size_t)hdr.nsamps * 4;
      buff_ptrs.clear();
      for (uint32_t a = 0; a < hdr.nb_ant; a++)
        buff_ptrs.push_back(payload.data() + a * per_ant_bytes);

      {
        std::lock_guard<std::mutex> lock(tx_streamer_mtx);
        if (!streaming.load(std::memory_order_acquire) || !tx_stream)
          continue;

        size_t sent = 0;
        if (hdr.nb_ant == 1)
          sent = tx_stream->send(buff_ptrs[0], hdr.nsamps, tx_md);
        else
          sent = tx_stream->send(buff_ptrs, hdr.nsamps, tx_md);

        if (sent != hdr.nsamps)
          PXY_W("tx_worker: short send %zu / %u\n", sent, hdr.nsamps);

        ctl->tx_count.fetch_add(sent, std::memory_order_relaxed);

        uhd::async_metadata_t am;
        while (tx_stream->recv_async_msg(am, 0.0)) {
          switch (am.event_code) {
            case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW:
            case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW_IN_PACKET:
              ctl->underflows.fetch_add(1, std::memory_order_relaxed);
              std::fputc('U', stderr); std::fflush(stderr);
              break;
            case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR:
            case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST:
              ctl->seq_errors.fetch_add(1, std::memory_order_relaxed);
              std::fputc('S', stderr); std::fflush(stderr);
              break;
            case uhd::async_metadata_t::EVENT_CODE_TIME_ERROR:
              std::fputc('L', stderr); std::fflush(stderr);
              break;
            default: break;
          }
        }
      }
    }

    std::lock_guard<std::mutex> lock(tx_streamer_mtx);
    if (tx_stream) {
      uhd::tx_metadata_t eob;
      eob.end_of_burst = true;
      eob.has_time_spec = false;
      try { tx_stream->send("", 0, eob); } catch (...) {}
    }
    PXY_I("tx_worker: exit\n");
  }

  void rx_worker_loop(int core, int recv_chunk)
  {
    pin_this_thread(core, "rx");
    bump_rt_priority("rx");

    size_t chunk = 0;
    int nb_ant = 0;
    size_t per_ant_bytes = 0;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> packed;
    std::vector<void *>  ptrs;

    while (!exit_workers.load(std::memory_order_acquire)) {
      if (!streaming.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }

      size_t max = 2048;
      int cur_nb_ant = 0;
      {
        std::lock_guard<std::mutex> lock(rx_streamer_mtx);
        if (!rx_stream)
          continue;
        max = rx_stream->get_max_num_samps();
        cur_nb_ant = (int)rx_stream->get_num_channels();
      }
      if (cur_nb_ant <= 0)
        continue;
      const size_t cur_chunk = (recv_chunk > 0 && (size_t)recv_chunk <= max)
                                   ? (size_t)recv_chunk : max;
      if (cur_chunk != chunk || cur_nb_ant != nb_ant) {
        chunk = cur_chunk;
        nb_ant = cur_nb_ant;
        per_ant_bytes = chunk * 4;
        payload.resize(per_ant_bytes * nb_ant);
        packed.resize(per_ant_bytes * nb_ant);
        ptrs.resize(nb_ant);
      }

      for (int a = 0; a < nb_ant; a++)
        ptrs[a] = payload.data() + (size_t)a * per_ant_bytes;

      uhd::rx_metadata_t md;
      size_t got = 0;
      {
        std::lock_guard<std::mutex> lock(rx_streamer_mtx);
        if (!streaming.load(std::memory_order_acquire) || !rx_stream)
          continue;
        if (nb_ant == 1)
          got = rx_stream->recv(ptrs[0], chunk, md, 1.0);
        else
          got = rx_stream->recv(ptrs, chunk, md, 1.0);
      }

      if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT)
        continue;

      if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND && got == 0)
        start_rx_error.store(true, std::memory_order_release);

      // A callback is not proof that RX is live: UHD also returns zero-sample
      // error callbacks (notably LATE_COMMAND).  Acknowledge START only after
      // an actual sample packet has crossed the radio boundary.
      if (got > 0 && !first_rx_seen.load(std::memory_order_relaxed)) {
        first_rx_seen.store(true, std::memory_order_release);
        PXY_I("rx_worker: stream is live (first recv: got=%zu err=%d) — "
              "CMD_START handshake complete\n", got, (int)md.error_code);
      }

      if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
        ctl->overflows.fetch_add(1, std::memory_order_relaxed);
        std::fputc('O', stderr); std::fflush(stderr);
      } else if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND) {
        std::fputc('L', stderr); std::fflush(stderr);
      } else if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN) {
        std::fputc('B', stderr); std::fflush(stderr);
      } else if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
        PXY_W("rx_worker: recv err %d\n", (int)md.error_code);
      }
      if (got == 0) continue;

      shm_rx_hdr_t hdr;
      hdr.timestamp     = (uint64_t)md.time_spec.to_ticks(sample_rate);
      hdr.nsamps        = (uint32_t)got;
      hdr.nb_ant        = (uint32_t)nb_ant;
      hdr.err_code      = (uint32_t)md.error_code;
      hdr.payload_bytes = (uint32_t)(got * 4 * nb_ant);

      const uint8_t *src_base = payload.data();
      if ((size_t)got != chunk) {
        for (int a = 0; a < nb_ant; a++)
          std::memcpy(packed.data() + a * got * 4,
                      payload.data() + a * per_ant_bytes, got * 4);
        src_base = packed.data();
      }

      if (!ring_write2_drop_new(rx, &hdr, sizeof(hdr), src_base, hdr.payload_bytes)) {
        ctl->overflows.fetch_add(1, std::memory_order_relaxed);
        std::fputc('O', stderr); std::fflush(stderr);
        continue;
      }
      ctl->rx_count.fetch_add(got, std::memory_order_relaxed);
      ctl->rx_last_timestamp.store(hdr.timestamp + got,
                                   std::memory_order_relaxed);
    }
    PXY_I("rx_worker: exit\n");
  }

  void telemetry_loop()
  {
    uint64_t prev_tx = 0, prev_rx = 0, prev_uf = 0, prev_of = 0, prev_se = 0;
    int tick = 0;
    while (!exit_workers.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::seconds(1));

      const uint64_t tx_n = ctl->tx_count.load(std::memory_order_relaxed);
      const uint64_t rx_n = ctl->rx_count.load(std::memory_order_relaxed);
      const uint64_t uf   = ctl->underflows.load(std::memory_order_relaxed);
      const uint64_t of   = ctl->overflows.load(std::memory_order_relaxed);
      const uint64_t se   = ctl->seq_errors.load(std::memory_order_relaxed);
      const uint64_t rxt  = ctl->rx_last_timestamp.load(std::memory_order_relaxed);

      const uint64_t tx_fill = ring_readable(tx);
      const uint64_t rx_fill = ring_readable(rx);

      const bool stream_on = streaming.load(std::memory_order_acquire);
      const double sr_mhz  = sample_rate / 1e6;

      std::fprintf(stderr,
          "[usrpproxy/tenant][t+%3ds] %s sr=%.2fMHz  TX +%lu (fill=%lu B)  "
          "RX +%lu (fill=%lu B, ts=%lu)  uf=+%lu(%lu) of=+%lu(%lu) seq=+%lu(%lu)\n",
          tick, stream_on ? "STREAM" : "idle  ", sr_mhz,
          (unsigned long)(tx_n - prev_tx), (unsigned long)tx_fill,
          (unsigned long)(rx_n - prev_rx), (unsigned long)rx_fill, (unsigned long)rxt,
          (unsigned long)(uf   - prev_uf), (unsigned long)uf,
          (unsigned long)(of   - prev_of), (unsigned long)of,
          (unsigned long)(se   - prev_se), (unsigned long)se);

      prev_tx = tx_n; prev_rx = rx_n; prev_uf = uf; prev_of = of; prev_se = se;
      tick++;
    }
  }

  // ---------- command handlers ----------

  bool handle_cmd_init()
  {
    streaming.store(false, std::memory_order_release);
    first_rx_seen.store(false, std::memory_order_release);
    start_rx_error.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(rx_streamer_mtx);
      if (rx_stream) {
        uhd::stream_cmd_t c(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
        try { rx_stream->issue_stream_cmd(c); } catch (...) {}
      }
    }

    if (!configure_oai_tenant()) return false;

    ctl->resolved_dev_type.store(dev_type, std::memory_order_relaxed);
    ctl->resolved_tx_sample_advance.store(tx_sample_advance, std::memory_order_relaxed);
    ctl->resolved_rx_shift.store(rx_shift, std::memory_order_relaxed);
    ctl->resolved_ready.store(1, std::memory_order_release);
    tx_count.store(0, std::memory_order_relaxed);

    // Start workers on first INIT only. On re-INIT (OAI reconnect), streamers
    // were rebuilt in configure_oai_tenant() and the running workers already
    // see the new handles via the member pointers.
    if (!tx_thr.joinable()) {
      tx_thr = std::thread(&Impl::tx_worker_loop, this, cfg.tx_core);
      rx_thr = std::thread(&Impl::rx_worker_loop, this, cfg.rx_core, cfg.recv_chunk);
      tel_thr = std::thread(&Impl::telemetry_loop, this);
    }
    return true;
  }

  bool handle_cmd_start()
  {
    {
      std::lock_guard<std::mutex> lock(rx_streamer_mtx);
      if (!rx_stream) { PXY_E("cmd_start: rx_stream is null\n"); return false; }
    }

    first_rx_seen.store(false, std::memory_order_release);
    tx_count.store(0, std::memory_order_relaxed);

    // Defensive flush: clear any stale "streaming-but-overflowed" state on
    // the USRP's RFNoC graph before issuing the real START.
    {
      std::lock_guard<std::mutex> lock(rx_streamer_mtx);
      uhd::stream_cmd_t stop(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
      rx_stream->issue_stream_cmd(stop);
      const size_t drain_chunk = rx_stream->get_max_num_samps();
      const int nb_ant = (int)rx_stream->get_num_channels();
      std::vector<uint8_t> drain_buf(drain_chunk * 4 * nb_ant);
      std::vector<void *>  drain_ptrs(nb_ant);
      for (int a = 0; a < nb_ant; a++)
        drain_ptrs[a] = drain_buf.data() + (size_t)a * drain_chunk * 4;
      uhd::rx_metadata_t md;
      int drained = 0;
      for (int i = 0; i < 64; i++) {
        size_t n = (nb_ant == 1)
            ? rx_stream->recv(drain_ptrs[0], drain_chunk, md, 0.05)
            : rx_stream->recv(drain_ptrs,    drain_chunk, md, 0.05);
        drained++;
        if (n == 0 && md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT)
          break;
      }
      PXY_I("cmd_start: drained pre-start state in %d recv() calls\n", drained);
    }

    // A shared X410 can retain a late command after a previous tenant exits.
    // Retry against fresh device time, but never ACK START until real samples
    // arrive. This makes first launch and restart follow the same contract.
    for (int attempt = 1; attempt <= 3; ++attempt) {
      first_rx_seen.store(false, std::memory_order_release);
      start_rx_error.store(false, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(rx_streamer_mtx);
        if (!rx_stream) { PXY_E("cmd_start: rx_stream vanished\n"); return false; }
        uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
        // Prefer a scheduled first start, which aligns cleanly on a fresh
        // graph.  If the shared DVB streamer is already running, some X410
        // graph states accept the command but never deliver tenant samples;
        // retry immediately instead of repeatedly scheduling into that stale
        // state.
        if (attempt == 1) {
          cmd.time_spec  = usrp->get_time_now() + 0.5;
          cmd.stream_now = false;
        } else {
          cmd.stream_now = true;
        }
        rx_stream->issue_stream_cmd(cmd);
      }
      streaming.store(true, std::memory_order_release);

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!first_rx_seen.load(std::memory_order_acquire) &&
             !start_rx_error.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() <= deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

      if (first_rx_seen.load(std::memory_order_acquire)) return true;

      streaming.store(false, std::memory_order_release);
      PXY_W("cmd_start: attempt %d produced no samples%s; rescheduling\n",
            attempt, start_rx_error.load(std::memory_order_acquire)
                         ? " (late command)" : "");
      std::lock_guard<std::mutex> lock(rx_streamer_mtx);
      if (rx_stream) {
        uhd::stream_cmd_t stop(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
        try { rx_stream->issue_stream_cmd(stop); } catch (...) {}
      }
    }
    PXY_E("cmd_start: RX stream failed to produce samples after 3 attempts\n");
    return false;
  }

  bool handle_cmd_stop()
  {
    streaming.store(false, std::memory_order_release);
    first_rx_seen.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(rx_streamer_mtx);
    if (!rx_stream) return true;
    uhd::stream_cmd_t c(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    try { rx_stream->issue_stream_cmd(c); } catch (...) {}
    return true;
  }

  bool handle_cmd_set_freq()
  {
    for (size_t k = 0; k < phys_tx.size(); k++)
      usrp->set_tx_freq(uhd::tune_request_t(ctl->cmd_tx_freq[k],
                                            ctl->cmd_tune_offset),
                        phys_tx[k]);
    for (size_t k = 0; k < phys_rx.size(); k++)
      usrp->set_rx_freq(uhd::tune_request_t(ctl->cmd_rx_freq[k],
                                            ctl->cmd_tune_offset),
                        phys_rx[k]);
    return true;
  }

  bool handle_cmd_set_gains()
  {
    for (size_t k = 0; k < phys_tx.size(); k++) {
      const int phys = phys_tx[k];
      auto gr = usrp->get_tx_gain_range(phys);
      usrp->set_tx_gain(gr.stop() - ctl->cmd_tx_gain[k], phys);
    }
    for (size_t k = 0; k < phys_rx.size(); k++) {
      const int phys = phys_rx[k];
      auto gr = usrp->get_rx_gain_range(phys);
      double g = ctl->cmd_rx_gain[k] - ctl->cmd_rx_gain_offset[k];
      if (g > gr.stop()) g = gr.stop();
      usrp->set_rx_gain(g, phys);
    }
    return true;
  }

  // ---------- control thread ----------

  void ctrl_loop()
  {
    PXY_I("waiting for driver on %s...\n", name.c_str());
    ctl->mvp_alive.store(1, std::memory_order_release);

    uint64_t last_seq = 0;
    while (!exit_workers.load(std::memory_order_acquire)) {
      uint64_t seq = ctl->cmd_seq.load(std::memory_order_acquire);
      if (seq == last_seq) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      int cmd = ctl->cmd.load(std::memory_order_acquire);
      bool ok = false;
      switch (cmd) {
        case USRPPROXY_CMD_INIT:      ok = handle_cmd_init();      break;
        case USRPPROXY_CMD_START:     ok = handle_cmd_start();     break;
        case USRPPROXY_CMD_STOP:      ok = handle_cmd_stop();      break;
        case USRPPROXY_CMD_SET_FREQ:  ok = handle_cmd_set_freq();  break;
        case USRPPROXY_CMD_SET_GAINS: ok = handle_cmd_set_gains(); break;
        case USRPPROXY_CMD_END:
          ok = true;
          // Host process owns the lifetime of the server; don't self-terminate
          // on CMD_END. Just release streamers so another OAI instance can
          // re-attach and re-INIT.
          handle_cmd_stop();
          {
            std::scoped_lock lock(tx_streamer_mtx, rx_streamer_mtx);
            tx_stream.reset();
            rx_stream.reset();
          }
          break;
        default:
          PXY_W("unknown cmd %d\n", cmd);
          ok = false;
          break;
      }
      ctl->cmd_result.store(ok ? USRPPROXY_ACK_OK : USRPPROXY_ACK_FAIL,
                            std::memory_order_release);
      ctl->cmd_ack_seq.store(seq, std::memory_order_release);
      last_seq = seq;
    }
  }

  // ---------- lifecycle ----------

  void start()
  {
    if (started.exchange(true, std::memory_order_acq_rel)) return;
    if (!shm_create()) {
      started.store(false, std::memory_order_release);
      throw std::runtime_error("usrpproxy::TenantServer: shm_create failed");
    }
    ctrl_thr = std::thread(&Impl::ctrl_loop, this);
  }

  void stop()
  {
    if (stopped.exchange(true, std::memory_order_acq_rel)) return;
    if (!started.load(std::memory_order_acquire)) {
      shm_destroy();
      return;
    }

    exit_workers.store(true, std::memory_order_release);
    streaming.store(false, std::memory_order_release);

    if (ctrl_thr.joinable()) ctrl_thr.join();
    if (tx_thr.joinable())   tx_thr.join();
    if (rx_thr.joinable())   rx_thr.join();
    if (tel_thr.joinable())  tel_thr.join();

    {
      std::scoped_lock lock(tx_streamer_mtx, rx_streamer_mtx);
      if (rx_stream) {
        uhd::stream_cmd_t stop(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
        try { rx_stream->issue_stream_cmd(stop); } catch (...) {}
      }

      tx_stream.reset();
      rx_stream.reset();
    }
    shm_destroy();
    PXY_I("server stopped\n");
  }
};

// =====================================================================
// Public API
// =====================================================================

TenantServer::TenantServer(uhd::usrp::multi_usrp::sptr usrp, Config cfg)
    : impl_(std::make_unique<Impl>())
{
  if (!usrp)
    throw std::invalid_argument("usrpproxy::TenantServer: usrp sptr is null");
  impl_->usrp = usrp;
  impl_->cfg  = std::move(cfg);
}

TenantServer::~TenantServer()
{
  if (impl_) impl_->stop();
}

void TenantServer::start() { impl_->start(); }
void TenantServer::stop()  { impl_->stop(); }

}  // namespace usrpproxy
