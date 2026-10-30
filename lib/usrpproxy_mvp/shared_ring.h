// SPDX-License-Identifier: Apache-2.0
//
// NOTE: this file is the wire protocol between the OAI-side plugin and the
// UHD-host-side tenant server. The authoritative copy lives in the standalone
// `usrpproxy` repo (usrpproxy/shared_ring.h). This is a vendored mirror.
// If you edit this file, edit both and bump USRPPROXY_VERSION — the driver's
// attach-time handshake rejects mismatched versions loudly.
//
// shared_ring.h — shared-memory layout and SPSC ring-buffer helpers used by
// both sides of the USRP proxy: the OAI radio driver (producer on TX,
// consumer on RX) and the UHD-host tenant server (the converse).
//
// Single POSIX shm segment layout:
//   offset 0:                       usrpproxy_ctl_t       (control block)
//   offset ctl->tx_ring_off:        usrpproxy_ring_t + tx_ring_capacity bytes
//   offset ctl->rx_ring_off:        usrpproxy_ring_t + rx_ring_capacity bytes
//
// All ring capacities are strict powers of two; indexing uses (idx & mask).
// Head/tail are monotonically increasing uint64_t (never wrapped), capacity
// sets the modulus for the addressing only.
//
// Data flow:
//   TX ring:  driver (OAI)     -> mvp (USRP host)     [SC16 post-shift]
//   RX ring:  mvp (USRP host)  -> driver (OAI)        [SC16 pre-shift]
//
// Packets are written atomically (hdr + payload in one head bump).

#ifndef USRPPROXY_SHARED_RING_H
#define USRPPROXY_SHARED_RING_H

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>

#define USRPPROXY_MAGIC   0x55445052u   // 'UDPR'
#define USRPPROXY_VERSION 1

#define USRPPROXY_MAX_CH  8

// Default ring capacity if config omits it (16 MiB per direction).
#define USRPPROXY_DEFAULT_RING_BYTES (16u * 1024u * 1024u)
// Hard cap for one ring. Larger segments should be an explicit protocol change.
#define USRPPROXY_MAX_RING_BYTES (1024ull * 1024ull * 1024ull)

// Command codes issued by driver to MVP via the control block.
enum usrpproxy_cmd_e {
  USRPPROXY_CMD_NONE      = 0,
  USRPPROXY_CMD_INIT      = 1,  // open USRP with control-block config
  USRPPROXY_CMD_START     = 2,  // start RX streaming
  USRPPROXY_CMD_STOP      = 3,  // stop RX streaming
  USRPPROXY_CMD_SET_FREQ  = 4,  // retune (fields in cmd_tx_freq[]/cmd_rx_freq[])
  USRPPROXY_CMD_SET_GAINS = 5,  // change gains
  USRPPROXY_CMD_END       = 6,  // release USRP, MVP may exit
};

// Ack/result codes written by MVP.
enum usrpproxy_ack_e {
  USRPPROXY_ACK_OK   =  1,
  USRPPROXY_ACK_FAIL = -1,
};

// Lock-free SPSC ring descriptor. Storage bytes follow the struct in memory.
struct usrpproxy_ring_t {
  // Cache-line isolate head/tail/capacity to avoid false sharing across
  // producer and consumer cores.
  std::atomic<uint64_t> head;      // bytes ever written by producer
  uint8_t               _pad0[64 - sizeof(std::atomic<uint64_t>)];
  std::atomic<uint64_t> tail;      // bytes ever consumed by consumer
  uint8_t               _pad1[64 - sizeof(std::atomic<uint64_t>)];
  uint64_t              capacity;  // power of two
  uint8_t               _pad2[64 - sizeof(uint64_t)];
};
static_assert(sizeof(usrpproxy_ring_t) == 192,
              "usrpproxy_ring_t layout changed — both sides must rebuild");

// Per-packet headers. Packed so both sides agree regardless of compiler.
struct __attribute__((packed)) shm_tx_hdr_t {
  uint64_t timestamp;       // OAI timestamp (already -= tx_sample_advance)
  uint32_t nsamps;          // samples per antenna
  uint32_t nb_ant;          // number of antennas (channels)
  uint32_t flags;           // radio_tx_burst_flag_t + radio_tx_gpio_flag_t
  uint32_t payload_bytes;   // = nsamps * nb_ant * 4 (SC16, post-shift)
};

struct __attribute__((packed)) shm_rx_hdr_t {
  uint64_t timestamp;       // timestamp of first sample in this chunk
  uint32_t nsamps;          // samples per antenna actually received
  uint32_t nb_ant;          // number of antennas
  uint32_t err_code;        // uhd::rx_metadata_t::error_code_t numeric (0=OK)
  uint32_t payload_bytes;   // = nsamps * nb_ant * 4 (SC16, pre-shift)
};

// Control block (start of shm segment).
struct usrpproxy_ctl_t {
  uint32_t magic;
  uint32_t version;

  // Segment layout (written once by segment creator).
  uint64_t segment_size;       // total shm size
  uint64_t tx_ring_off;        // byte offset of TX ring struct
  uint64_t rx_ring_off;        // byte offset of RX ring struct
  uint64_t tx_ring_capacity;   // storage bytes (power of two)
  uint64_t rx_ring_capacity;

  // Liveness.
  std::atomic<int32_t> mvp_alive;
  std::atomic<int32_t> driver_alive;
  std::atomic<int32_t> abort_flag;

  // Config written by driver before CMD_INIT, read by MVP.
  double   sample_rate;
  double   tune_offset;
  double   rx_bw;
  double   tx_bw;
  double   tx_freq[USRPPROXY_MAX_CH];
  double   rx_freq[USRPPROXY_MAX_CH];
  double   tx_gain[USRPPROXY_MAX_CH];
  double   rx_gain[USRPPROXY_MAX_CH];
  double   rx_gain_offset[USRPPROXY_MAX_CH];
  int32_t  tx_num_channels;
  int32_t  rx_num_channels;
  int32_t  clock_source;    // clock_source_t
  int32_t  time_source;
  char     sdr_addrs[256];
  char     tx_subdev[64];
  char     rx_subdev[64];

  // Resolved by MVP after CMD_INIT, read back by driver.
  std::atomic<int32_t> resolved_dev_type;           // dev_type_t numeric
  std::atomic<int32_t> resolved_tx_sample_advance;
  std::atomic<int32_t> resolved_rx_shift;           // 4 for B200, 2 otherwise
  std::atomic<int32_t> resolved_ready;              // set 1 after INIT completes
  // rx_gain_offset computed by MVP from the per-device calib table. Driver
  // copies into openair0_cfg so OAI-side code (e.g. RSRP) sees the same
  // numbers the direct USRP driver would have produced.
  double               resolved_rx_gain_offset[USRPPROXY_MAX_CH];

  // Command channel (driver -> MVP).
  std::atomic<uint64_t> cmd_seq;     // driver bumps after filling fields+cmd
  std::atomic<uint64_t> cmd_ack_seq; // MVP sets to cmd_seq after processing
  std::atomic<int32_t>  cmd;         // usrpproxy_cmd_e
  std::atomic<int32_t>  cmd_result;  // usrpproxy_ack_e

  // Per-command parameters (driver fills before bumping cmd_seq).
  double cmd_tx_freq[USRPPROXY_MAX_CH];
  double cmd_rx_freq[USRPPROXY_MAX_CH];
  double cmd_tx_gain[USRPPROXY_MAX_CH];
  double cmd_rx_gain[USRPPROXY_MAX_CH];
  double cmd_rx_gain_offset[USRPPROXY_MAX_CH];
  double cmd_tune_offset;

  // Stats (MVP publishes).
  std::atomic<uint64_t> tx_count;
  std::atomic<uint64_t> rx_count;
  std::atomic<uint64_t> underflows;
  std::atomic<uint64_t> overflows;
  std::atomic<uint64_t> seq_errors;
  std::atomic<uint64_t> rx_last_timestamp;
};

// Inline helpers — same contract on both ends.
//
// Convention:
//   ring_writable / ring_write2 are called by the producer only.
//   ring_readable / ring_peek / ring_consume are called by the consumer only.

static inline uint8_t *usrpproxy_ring_storage(usrpproxy_ring_t *r)
{
  return reinterpret_cast<uint8_t *>(r) + sizeof(usrpproxy_ring_t);
}

static inline uint64_t ring_readable(usrpproxy_ring_t *r)
{
  uint64_t h = r->head.load(std::memory_order_acquire);
  uint64_t t = r->tail.load(std::memory_order_relaxed);
  return h - t;
}

static inline uint64_t ring_writable(usrpproxy_ring_t *r)
{
  uint64_t t = r->tail.load(std::memory_order_acquire);
  uint64_t h = r->head.load(std::memory_order_relaxed);
  return r->capacity - (h - t);
}

// Wrap-aware memcpy helpers.
static inline void ring_memcpy_out(uint8_t *dst, uint8_t *ring_buf,
                                   uint64_t cap, uint64_t logical_off,
                                   uint64_t n)
{
  uint64_t off = logical_off & (cap - 1);
  if (off + n <= cap) {
    std::memcpy(dst, ring_buf + off, n);
  } else {
    uint64_t first = cap - off;
    std::memcpy(dst, ring_buf + off, first);
    std::memcpy(dst + first, ring_buf, n - first);
  }
}

static inline void ring_memcpy_in(uint8_t *ring_buf, uint64_t cap,
                                  uint64_t logical_off,
                                  const uint8_t *src, uint64_t n)
{
  uint64_t off = logical_off & (cap - 1);
  if (off + n <= cap) {
    std::memcpy(ring_buf + off, src, n);
  } else {
    uint64_t first = cap - off;
    std::memcpy(ring_buf + off, src, first);
    std::memcpy(ring_buf, src + first, n - first);
  }
}

// Write [hdr|payload] atomically in a single head bump. Returns false if the
// ring doesn't have room (caller decides whether to spin or drop).
static inline bool ring_write2(usrpproxy_ring_t *r,
                               const void *hdr, uint64_t hdr_sz,
                               const void *payload, uint64_t pay_sz)
{
  uint64_t total = hdr_sz + pay_sz;
  if (ring_writable(r) < total) return false;
  uint64_t h = r->head.load(std::memory_order_relaxed);
  uint8_t *buf = usrpproxy_ring_storage(r);
  ring_memcpy_in(buf, r->capacity, h,         (const uint8_t *)hdr,     hdr_sz);
  ring_memcpy_in(buf, r->capacity, h + hdr_sz,(const uint8_t *)payload, pay_sz);
  r->head.store(h + total, std::memory_order_release);
  return true;
}

// Producer-side overflow policy for SPSC rings: drop the new packet if there
// is no space. The producer must never advance tail, which is owned by the
// consumer in the other process.
static inline bool ring_write2_drop_new(usrpproxy_ring_t *r,
                                        const void *hdr, uint64_t hdr_sz,
                                        const void *payload, uint64_t pay_sz)
{
  return ring_write2(r, hdr, hdr_sz, payload, pay_sz);
}

static inline bool usrpproxy_tx_payload_shape_valid(uint32_t nsamps,
                                                    uint32_t nb_ant,
                                                    uint32_t payload_bytes)
{
  if (nb_ant == 0 || nb_ant > USRPPROXY_MAX_CH) return false;
  const uint64_t expected = (uint64_t)nsamps * (uint64_t)nb_ant * 4u;
  return expected <= UINT32_MAX && payload_bytes == expected;
}

static inline bool usrpproxy_tx_hdr_valid(const shm_tx_hdr_t *h,
                                          uint32_t stream_channels,
                                          uint64_t ring_capacity)
{
  if (!h) return false;
  if (stream_channels == 0 || stream_channels > USRPPROXY_MAX_CH) return false;
  if (h->nb_ant != stream_channels) return false;
  if (!usrpproxy_tx_payload_shape_valid(h->nsamps, h->nb_ant,
                                        h->payload_bytes)) return false;
  return sizeof(*h) + (uint64_t)h->payload_bytes <= ring_capacity;
}

static inline bool usrpproxy_tx_packet_shape_valid(const shm_tx_hdr_t *h,
                                                   uint64_t ring_capacity)
{
  if (!h) return false;
  if (!usrpproxy_tx_payload_shape_valid(h->nsamps, h->nb_ant,
                                        h->payload_bytes)) return false;
  return sizeof(*h) + (uint64_t)h->payload_bytes <= ring_capacity;
}

static inline bool usrpproxy_channel_count_valid(int32_t n)
{
  return n >= 1 && n <= USRPPROXY_MAX_CH;
}

static inline bool usrpproxy_ring_capacity_valid(uint64_t cap)
{
  return cap >= 4096 && cap <= USRPPROXY_MAX_RING_BYTES &&
         (cap & (cap - 1)) == 0;
}

// Peek n bytes starting at a given byte offset into the unread region (0 =
// at tail). Returns false if not enough data available.
static inline bool ring_peek_at(usrpproxy_ring_t *r, uint64_t rel_off,
                                void *out, uint64_t n)
{
  if (rel_off > UINT64_MAX - n) return false;
  uint64_t readable = ring_readable(r);
  if (readable < rel_off || readable - rel_off < n) return false;
  uint64_t t = r->tail.load(std::memory_order_relaxed);
  ring_memcpy_out((uint8_t *)out, usrpproxy_ring_storage(r),
                  r->capacity, t + rel_off, n);
  return true;
}

static inline bool ring_peek(usrpproxy_ring_t *r, void *out, uint64_t n)
{
  return ring_peek_at(r, 0, out, n);
}

// Consume (drop) n bytes from the tail.
static inline void ring_consume(usrpproxy_ring_t *r, uint64_t n)
{
  uint64_t t = r->tail.load(std::memory_order_relaxed);
  r->tail.store(t + n, std::memory_order_release);
}

static inline bool usrpproxy_segment_bytes_checked(uint64_t tx_cap,
                                                   uint64_t rx_cap,
                                                   uint64_t *out)
{
  if (!out || !usrpproxy_ring_capacity_valid(tx_cap) ||
      !usrpproxy_ring_capacity_valid(rx_cap))
    return false;

  const uint64_t page = 4096;
  auto align_up_checked = [&](uint64_t v, uint64_t *aligned) -> bool {
    if (v > UINT64_MAX - (page - 1)) return false;
    *aligned = (v + page - 1) & ~(page - 1);
    return true;
  };

  uint64_t ctl_sz, tx_sz, rx_sz;
  if (!align_up_checked(sizeof(usrpproxy_ctl_t), &ctl_sz)) return false;
  if (tx_cap > UINT64_MAX - sizeof(usrpproxy_ring_t)) return false;
  if (rx_cap > UINT64_MAX - sizeof(usrpproxy_ring_t)) return false;
  if (!align_up_checked(sizeof(usrpproxy_ring_t) + tx_cap, &tx_sz)) return false;
  if (!align_up_checked(sizeof(usrpproxy_ring_t) + rx_cap, &rx_sz)) return false;
  if (ctl_sz > UINT64_MAX - tx_sz || ctl_sz + tx_sz > UINT64_MAX - rx_sz)
    return false;

  *out = ctl_sz + tx_sz + rx_sz;
  return true;
}

#endif // USRPPROXY_SHARED_RING_H
