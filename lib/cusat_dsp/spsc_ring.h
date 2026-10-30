/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Lock-free single-producer single-consumer ring buffer for IQ streaming.
 */

#ifndef INCLUDED_CUSAT_DSP_SPSC_RING_H
#define INCLUDED_CUSAT_DSP_SPSC_RING_H

#include <atomic>
#include <cstddef>
#include <cstring>
#include <vector>

namespace cusat_dsp {

template <typename T>
class spsc_ring
{
public:
    explicit spsc_ring(size_t capacity)
        : d_buf(capacity), d_capacity(capacity), d_head(0), d_tail(0)
    {
    }

    /** Write up to count items. Returns number actually written. */
    size_t write(const T* data, size_t count)
    {
        size_t head = d_head.load(std::memory_order_relaxed);
        size_t tail = d_tail.load(std::memory_order_acquire);
        size_t avail = d_capacity - (head - tail);
        if (count > avail)
            count = avail;
        if (count == 0)
            return 0;

        size_t idx = head % d_capacity;
        size_t first = d_capacity - idx;
        if (first >= count) {
            memcpy(&d_buf[idx], data, count * sizeof(T));
        } else {
            memcpy(&d_buf[idx], data, first * sizeof(T));
            memcpy(&d_buf[0], data + first, (count - first) * sizeof(T));
        }
        d_head.store(head + count, std::memory_order_release);
        return count;
    }

    /** Read up to count items. Returns number actually read. */
    size_t read(T* data, size_t count)
    {
        size_t tail = d_tail.load(std::memory_order_relaxed);
        size_t head = d_head.load(std::memory_order_acquire);
        size_t avail = head - tail;
        if (count > avail)
            count = avail;
        if (count == 0)
            return 0;

        size_t idx = tail % d_capacity;
        size_t first = d_capacity - idx;
        if (first >= count) {
            memcpy(data, &d_buf[idx], count * sizeof(T));
        } else {
            memcpy(data, &d_buf[idx], first * sizeof(T));
            memcpy(data + first, &d_buf[0], (count - first) * sizeof(T));
        }
        d_tail.store(tail + count, std::memory_order_release);
        return count;
    }

    /** Discard up to count items without copying. Returns number skipped. */
    size_t skip(size_t count)
    {
        size_t tail = d_tail.load(std::memory_order_relaxed);
        size_t head = d_head.load(std::memory_order_acquire);
        size_t avail = head - tail;
        if (count > avail)
            count = avail;
        if (count == 0)
            return 0;
        d_tail.store(tail + count, std::memory_order_release);
        return count;
    }

    size_t available_read() const
    {
        return d_head.load(std::memory_order_acquire) -
               d_tail.load(std::memory_order_relaxed);
    }

    size_t capacity() const { return d_capacity; }

private:
    std::vector<T> d_buf;
    size_t d_capacity;
    alignas(64) std::atomic<size_t> d_head;
    alignas(64) std::atomic<size_t> d_tail;
};

} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_SPSC_RING_H */
