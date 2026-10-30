/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Queue and ring-buffer adapters for dvbs2-uhd-trx.
 */

#pragma once

#include "iq_source.h"
#include "framing/plframe.h"
#include "framing/plheader.h"
#include "receiver_frontend.h"
#include "spsc_ring.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

using namespace cusat_dsp;

/* ---------- Thread-safe PDU queue for TAP reader ---------- */
class pdu_queue
{
    std::queue<std::vector<uint8_t>> d_queue;
    std::mutex d_mtx;
    // ACM deliberately holds application PDUs while a candidate MODCOD is
    // validated. At the 30 Msymbol/s tutorial rate that bounded probation can
    // accumulate several thousand full-size packets at PHY saturation. Keep
    // enough room for it without making this queue unbounded during an outage.
    static constexpr int MAX_QUEUED = 8192;
public:
    bool push(const uint8_t* data, int len) {
        std::lock_guard<std::mutex> lk(d_mtx);
        if ((int)d_queue.size() >= MAX_QUEUED) return false;
        d_queue.emplace(data, data + len);
        return true;
    }
    bool pop(std::vector<uint8_t>& out) {
        std::lock_guard<std::mutex> lk(d_mtx);
        if (d_queue.empty()) return false;
        out = std::move(d_queue.front());
        d_queue.pop();
        return true;
    }
    bool empty() {
        std::lock_guard<std::mutex> lk(d_mtx);
        return d_queue.empty();
    }
};

/* ---------- Lock-free SPSC frame queue for RX pipeline split ---------- */

struct fec_frame_t {
    std::vector<complexf> symbols;
    pl_layout pls;
    int payload_len = 0;
    float pilot_snr_db = 0.0f;
};

/**
 * Lock-free bounded SPSC queue with pre-allocated slots.
 * Producer (frontend thread) writes frames; consumer (FEC thread) reads.
 * No heap allocation on the hot path.
 */
class spsc_frame_queue
{
public:
    explicit spsc_frame_queue(int capacity, int max_xfecframe_len)
        : d_capacity(capacity), d_head(0), d_tail(0)
    {
        d_slots.resize(capacity);
        for (auto& s : d_slots)
            s.symbols.resize(max_xfecframe_len);
    }

    /** Producer: deep-copy a frontend_frame_t into the next slot. */
    bool try_push(const frontend_frame_t& frame) {
        uint64_t head = d_head.load(std::memory_order_relaxed);
        uint64_t tail = d_tail.load(std::memory_order_acquire);
        if (head - tail >= (uint64_t)d_capacity)
            return false; // full
        auto& slot = d_slots[head % d_capacity];
        memcpy(slot.symbols.data(), frame.payload,
               frame.payload_len * sizeof(complexf));
        slot.pls = frame.pls;
        slot.payload_len = frame.payload_len;
        slot.pilot_snr_db = frame.pilot_snr_db;
        d_head.store(head + 1, std::memory_order_release);
        return true;
    }

    /** Consumer: get pointer to next available frame (or nullptr). */
    fec_frame_t* try_pop() {
        uint64_t tail = d_tail.load(std::memory_order_relaxed);
        uint64_t head = d_head.load(std::memory_order_acquire);
        if (tail >= head)
            return nullptr; // empty
        return &d_slots[tail % d_capacity];
    }

    /** Consumer: advance tail after processing the frame from try_pop(). */
    void pop_commit() {
        uint64_t tail = d_tail.load(std::memory_order_relaxed);
        d_tail.store(tail + 1, std::memory_order_release);
    }

    int available() const {
        uint64_t head = d_head.load(std::memory_order_acquire);
        uint64_t tail = d_tail.load(std::memory_order_relaxed);
        uint64_t n = head - tail;
        return n > (uint64_t)d_capacity ? d_capacity : (int)n;
    }

private:
    std::vector<fec_frame_t> d_slots;
    int d_capacity;
    alignas(64) std::atomic<uint64_t> d_head;
    alignas(64) std::atomic<uint64_t> d_tail;
};

/* ---------- IQ source backed by SPSC ring buffer ---------- */

class iq_source_ring : public iq_source
{
public:
    iq_source_ring(spsc_ring<complexf>& ring, std::atomic<bool>& stop)
        : d_ring(ring), d_stop(stop) {}

    int read(complexf* buf, int max_samples) override
    {
        // Spin-wait with back-off until data available or stopped
        while (!d_stop.load(std::memory_order_relaxed)) {
            size_t n = d_ring.read(buf, max_samples);
            if (n > 0)
                return (int)n;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return 0;
    }
private:
    spsc_ring<complexf>& d_ring;
    std::atomic<bool>& d_stop;
};
