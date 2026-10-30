/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared frontend trace CSV writer for live and offline RX runs.
 */

#pragma once

#include "receiver_frontend.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

class frontend_trace_writer
{
public:
    frontend_trace_writer() = default;
    ~frontend_trace_writer() { close(); }

    frontend_trace_writer(const frontend_trace_writer&) = delete;
    frontend_trace_writer& operator=(const frontend_trace_writer&) = delete;

    bool open(const std::string& path, const char* owner)
    {
        if (path.empty())
            return true;
        d_fp = fopen(path.c_str(), "w");
        if (!d_fp) {
            fprintf(stderr, "%s: cannot open frontend trace CSV: %s\n",
                    owner, path.c_str());
            return false;
        }
        setvbuf(d_fp, nullptr, _IOLBF, 0);
        fprintf(d_fp,
                "t_us,call,locked,n_in,n_consumed,n_syms,n_frames,"
                "period_us,fe_total_us,fe_agc_us,fe_timing_us,fe_sync_us,"
                "tr_memcpy_us,tr_launch_us,tr_lnch_est_us,"
                "tr_lnch_interp_us,tr_sync_us,tr_readback_us,"
                "gpu_timing_us,gpu_interp_us,fused,pilot_snr_db,"
                "pilot_coherence,ewma_fer,rx_fec_frames,rx_fec_errors,"
                "fec_queue_depth,gpu_inflight,rx_ring_samples,"
                "deadline_miss,slack_us\n");
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

    void flush()
    {
        if (d_fp)
            fflush(d_fp);
    }

    void write(const cusat_dsp::receiver_frontend& frontend,
               std::chrono::steady_clock::time_point start_time,
               std::chrono::steady_clock::time_point end_time,
               uint64_t fe_total_us,
               int n_consumed,
               size_t n_frames,
               double sample_rate,
               double ewma_fer,
               uint64_t rx_fec_frames,
               uint64_t rx_fec_errors,
               size_t fec_queue_depth,
               int gpu_inflight,
               size_t rx_ring_samples)
    {
        if (!d_fp)
            return;

        const auto& p = frontend.perf();
        double period_us = (sample_rate > 0.0)
            ? ((double)p.n_in / sample_rate * 1e6) : 0.0;
        double slack_us = period_us - (double)fe_total_us;
        int deadline_miss = (period_us > 0.0 &&
                             (double)fe_total_us > period_us) ? 1 : 0;
        uint64_t t_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();

        fprintf(d_fp,
                "%llu,%llu,%d,%d,%d,%d,%zu,%.3f,%llu,%u,%u,%u,"
                "%u,%u,%u,%u,%u,%u,%u,%u,%d,%.3f,%.3f,%.6g,"
                "%llu,%llu,%zu,%d,%zu,%d,%.3f\n",
                (unsigned long long)t_us,
                (unsigned long long)d_call++,
                frontend.locked() ? 1 : 0,
                p.n_in, n_consumed, p.n_syms, n_frames,
                period_us,
                (unsigned long long)fe_total_us,
                p.agc_us, p.timing_us, p.sync_us,
                p.tr_memcpy_us, p.tr_launch_us,
                p.tr_lnch_est_us, p.tr_lnch_interp_us,
                p.tr_sync_us, p.tr_readback_us,
                p.gpu_timing_us, p.gpu_interp_us,
                p.fused ? 1 : 0,
                frontend.pilot_snr_db(),
                frontend.pilot_coherence(),
                ewma_fer,
                (unsigned long long)rx_fec_frames,
                (unsigned long long)rx_fec_errors,
                fec_queue_depth,
                gpu_inflight,
                rx_ring_samples,
                deadline_miss,
                slack_us);

        if ((d_call & 0x3ff) == 0)
            fflush(d_fp);
    }

private:
    FILE* d_fp = nullptr;
    uint64_t d_call = 0;
};
