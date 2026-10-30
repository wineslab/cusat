/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared RX FEC scheduling trace CSV writer for dvbs2-uhd-trx.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

class fec_trace_writer
{
public:
    fec_trace_writer() = default;
    ~fec_trace_writer() { close(); }

    fec_trace_writer(const fec_trace_writer&) = delete;
    fec_trace_writer& operator=(const fec_trace_writer&) = delete;

    bool open(const std::string& path, const char* owner)
    {
        if (path.empty())
            return true;
        d_fp = fopen(path.c_str(), "w");
        if (!d_fp) {
            fprintf(stderr, "%s: cannot open FEC trace CSV: %s\n",
                    owner, path.c_str());
            return false;
        }
        setvbuf(d_fp, nullptr, _IOLBF, 0);
        fprintf(d_fp,
                "t_us,loop,event,queue_depth,pending,inflight,submitted,"
                "drained,batch_ready,timeout_flush,lat_fec_loop_us,"
                "lat_gpu_drain_us,lat_gpu_submit_us,lat_fec_decode_us,"
                "raw_frames,raw_errors,rx_fec_frames,rx_fec_errors,"
                "ewma_fer,gpu_force_drain_count\n");
        return true;
    }

    bool active() const { return d_fp != nullptr; }

    bool should_write(std::chrono::steady_clock::time_point now,
                      bool drained,
                      int submitted,
                      bool timeout_flush)
    {
        if (!d_fp)
            return false;
        bool periodic =
            (now - d_last_periodic) >= std::chrono::milliseconds(100);
        if (periodic)
            d_last_periodic = now;
        return drained || submitted > 0 || timeout_flush || periodic;
    }

    void write(std::chrono::steady_clock::time_point start_time,
               std::chrono::steady_clock::time_point now,
               size_t queue_depth,
               int pending,
               int inflight,
               int submitted,
               bool drained,
               bool batch_ready,
               bool timeout_flush,
               uint64_t lat_fec_loop_us,
               uint64_t lat_gpu_drain_us,
               uint64_t lat_gpu_submit_us,
               uint64_t lat_fec_decode_us,
               uint64_t raw_frames,
               uint64_t raw_errors,
               uint64_t rx_fec_frames,
               uint64_t rx_fec_errors,
               float ewma_fer,
               uint64_t gpu_force_drain_count)
    {
        if (!d_fp)
            return;

        const char* event = timeout_flush ? "timeout_flush"
            : batch_ready ? "batch_ready"
            : submitted > 0 ? "submit"
            : drained ? "drain" : "periodic";
        uint64_t t_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                now - start_time).count();

        fprintf(d_fp,
                "%llu,%llu,%s,%zu,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,"
                "%llu,%llu,%llu,%llu,%llu,%.6g,%llu\n",
                (unsigned long long)t_us,
                (unsigned long long)d_loop++,
                event,
                queue_depth,
                pending,
                inflight,
                submitted,
                drained ? 1 : 0,
                batch_ready ? 1 : 0,
                timeout_flush ? 1 : 0,
                (unsigned long long)lat_fec_loop_us,
                (unsigned long long)lat_gpu_drain_us,
                (unsigned long long)lat_gpu_submit_us,
                (unsigned long long)lat_fec_decode_us,
                (unsigned long long)raw_frames,
                (unsigned long long)raw_errors,
                (unsigned long long)rx_fec_frames,
                (unsigned long long)rx_fec_errors,
                ewma_fer,
                (unsigned long long)gpu_force_drain_count);
    }

private:
    void close()
    {
        if (d_fp) {
            fclose(d_fp);
            d_fp = nullptr;
        }
    }

    FILE* d_fp = nullptr;
    uint64_t d_loop = 0;
    std::chrono::steady_clock::time_point d_last_periodic =
        std::chrono::steady_clock::now();
};
