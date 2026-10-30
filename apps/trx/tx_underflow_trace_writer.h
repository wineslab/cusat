/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Optional TX send/underflow trace CSV writer for dvbs2-uhd-trx.
 */

#pragma once

#include "trx/context.h"
#include "trx/radio_device.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

class tx_underflow_trace_writer
{
public:
    tx_underflow_trace_writer() = default;
    ~tx_underflow_trace_writer() { close(); }

    tx_underflow_trace_writer(const tx_underflow_trace_writer&) = delete;
    tx_underflow_trace_writer& operator=(const tx_underflow_trace_writer&) = delete;

    bool open(const std::string& path, const char* owner)
    {
        if (path.empty())
            return true;
        d_fp = fopen(path.c_str(), "w");
        if (!d_fp) {
            fprintf(stderr, "%s: cannot open TX underflow trace CSV: %s\n",
                    owner, path.c_str());
            return false;
        }
        setvbuf(d_fp, nullptr, _IOLBF, 0);
        fprintf(d_fp,
                "t_us,event,uf_count,async_present,async_underflow,"
                "async_event_code,async_dev_time_s,send_n,send_dur_us,"
                "send_gap_us,recent_max_dur_us,recent_max_gap_us,"
                "recent_over1ms,tx_ring_samples,tx_ring_cap,tx_frames,"
                "tx_dummy_frames,rx_fec_frames,rx_fec_errors,rx_pdus,"
                "rx_uhd_overflows,rx_ring_drops,rx_fec_queue_depth,"
                "lat_tx_loop_us,lat_tx_gen_us,lat_tx_ring_wait_us,"
                "lat_frontend_us,lat_rx_loop_us,lat_fec_loop_us,"
                "lat_fec_decode_us,lat_gpu_drain_us,lat_gpu_phase_us,"
                "lat_gpu_submit_us,gpu_inflight,fe_stage,cuda_owner,"
                "cuda_hold_us,ewma_fer,txs_dur_over200,txs_gap_over200,"
                "txs_dur_over1ms,txs_gap_over1ms\n");
        return true;
    }

    bool active() const { return d_fp != nullptr; }

    void close()
    {
        if (d_fp) {
            fclose(d_fp);
            d_fp = nullptr;
        }
    }

    bool should_write(std::chrono::steady_clock::time_point now)
    {
        if (!d_fp)
            return false;
        if ((now - d_last_periodic) < std::chrono::milliseconds(100))
            return false;
        d_last_periodic = now;
        return true;
    }

    void write(const trx_context& ctx,
               const char* event,
               std::chrono::steady_clock::time_point now,
               const radio_async_msg& async_md,
               uint64_t underflow_count,
               size_t send_n,
               uint32_t send_dur_us,
               uint32_t send_gap_us,
               uint32_t recent_max_dur_us,
               uint32_t recent_max_gap_us,
               uint32_t recent_over1ms,
               size_t tx_ring_samples,
               size_t tx_ring_cap)
    {
        if (!d_fp)
            return;

        const auto& s = ctx.stats;
        uint64_t t_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                now - ctx.start_time).count();

        fprintf(d_fp,
                "%llu,%s,%llu,%d,%d,0x%x,%.9f,%zu,%u,%u,%u,%u,%u,%zu,%zu,"
                "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%zu,%llu,%llu,%llu,"
                "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%d,%llu,%.6g,"
                "%llu,%llu,%llu,%llu\n",
                (unsigned long long)t_us,
                event,
                (unsigned long long)underflow_count,
                async_md.present ? 1 : 0,
                async_md.underflow ? 1 : 0,
                async_md.event_code,
                async_md.time_spec,
                send_n,
                send_dur_us,
                send_gap_us,
                recent_max_dur_us,
                recent_max_gap_us,
                recent_over1ms,
                tx_ring_samples,
                tx_ring_cap,
                (unsigned long long)s.tx_frames.load(std::memory_order_relaxed),
                (unsigned long long)s.tx_dummy_frames.load(std::memory_order_relaxed),
                (unsigned long long)s.rx_fec_frames.load(std::memory_order_relaxed),
                (unsigned long long)s.rx_fec_errors.load(std::memory_order_relaxed),
                (unsigned long long)s.rx_pdus.load(std::memory_order_relaxed),
                (unsigned long long)s.rx_uhd_overflows.load(std::memory_order_relaxed),
                (unsigned long long)s.rx_ring_drops.load(std::memory_order_relaxed),
                (size_t)(ctx.frame_queue ? ctx.frame_queue->available() : 0),
                (unsigned long long)s.lat_tx_loop_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_tx_gen_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_tx_ring_wait_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_frontend_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_rx_loop_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_fec_loop_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_fec_decode_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_gpu_drain_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_gpu_phase_us.load(std::memory_order_relaxed),
                (unsigned long long)s.lat_gpu_submit_us.load(std::memory_order_relaxed),
                s.gpu_inflight_count.load(std::memory_order_relaxed),
                s.fe_stage_atomic.load(std::memory_order_relaxed),
                s.cuda_mtx_owner.load(std::memory_order_relaxed),
                (unsigned long long)s.cuda_mtx_hold_us.load(std::memory_order_relaxed),
                s.ewma_fer.load(std::memory_order_relaxed),
                (unsigned long long)s.txs_dur_over200.load(std::memory_order_relaxed),
                (unsigned long long)s.txs_gap_over200.load(std::memory_order_relaxed),
                (unsigned long long)s.txs_dur_over1ms.load(std::memory_order_relaxed),
                (unsigned long long)s.txs_gap_over1ms.load(std::memory_order_relaxed));
    }

private:
    FILE* d_fp = nullptr;
    std::chrono::steady_clock::time_point d_last_periodic =
        std::chrono::steady_clock::now();
};
