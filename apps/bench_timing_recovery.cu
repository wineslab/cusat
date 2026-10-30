/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host-launch-cost micro-benchmark for the GPU timing recovery path.
 *
 * Reproduces the trx locked-path call pattern (stage input once, then
 * process_launch / process_complete per batch) at any stride, with an
 * optional concurrent LDPC decode load to expose CUDA driver contention.
 * The launch/sync host cost is data-independent, so no radio is needed:
 * this isolates the fractional-stride "launch storm" (lnch=0.5 ms/batch
 * observed OTA at stride 1.536 vs ~0 at stride 2.0) without an OTA run.
 *
 * Examples:
 *   bench-timing-recovery --stride 1.536
 *   bench-timing-recovery --stride 2.0 --search-cap 16384
 *   bench-timing-recovery --stride 1.536 --ldpc-load 64
 *   bench-timing-recovery --stride 1.536 --est-period 8
 *   bench-timing-recovery --profile-stage --n-input 131072
 */

#include "interpolator.h"
#include "ldpc_decode.h"
#include "ldpc_encode.h"
#include "cuda/timing_recovery_kernel.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace cusat_dsp;
using clk = std::chrono::steady_clock;

/* Same mapping as bench-ldpc-decode (kept in sync by hand; benches only). */
static code_rate_t rate_from_modcod(const char* modcod)
{
    if (strcmp(modcod, "qpsk1/4") == 0) return RATE_1_4;
    if (strcmp(modcod, "qpsk1/2") == 0) return RATE_1_2;
    if (strcmp(modcod, "qpsk3/5") == 0 || strcmp(modcod, "8psk3/5") == 0)
        return RATE_3_5;
    if (strcmp(modcod, "qpsk2/3") == 0 || strcmp(modcod, "16apsk2/3") == 0)
        return RATE_2_3;
    if (strcmp(modcod, "qpsk3/4") == 0 || strcmp(modcod, "32apsk3/4") == 0 ||
        strcmp(modcod, "16apsk3/4") == 0)
        return RATE_3_4;
    if (strcmp(modcod, "128apsk3/4") == 0) return RATE_135_180;   /* S2X */
    if (strcmp(modcod, "64apsk11/15") == 0) return RATE_132_180;  /* S2X */
    if (strcmp(modcod, "64apsk7/9") == 0) return RATE_140_180;    /* S2X */
    if (strcmp(modcod, "qpsk13/45") == 0) return RATE_13_45;      /* S2X */
    if (strcmp(modcod, "64apsk4/5") == 0 || strcmp(modcod, "qpsk4/5") == 0)
        return RATE_4_5;
    if (strcmp(modcod, "qpsk5/6") == 0 || strcmp(modcod, "16apsk5/6") == 0)
        return RATE_5_6;
    if (strcmp(modcod, "32apsk8/9") == 0 || strcmp(modcod, "qpsk8/9") == 0)
        return RATE_8_9;
    if (strcmp(modcod, "qpsk9/10") == 0) return RATE_9_10;
    return RATE_3_5;
}

static uint32_t us_between(clk::time_point a, clk::time_point b)
{
    return (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(
        b - a).count();
}

static double pctile(std::vector<uint32_t> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p * (v.size() - 1) + 0.5);
    return (double)v[idx];
}

static double mean_of(const std::vector<uint32_t>& v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (auto x : v) s += x;
    return s / v.size();
}

int main(int argc, char* argv[])
{
    double stride = 1.536;
    int iters = 500, warmup = 50;
    int n_input = 131072;
    int est_period = 4;
    int energy_k = 32;
    int search_cap = 2048;
    int ldpc_batch = 0;
    int ldpc_trials = 25;
    double ldpc_ebn0 = NAN;     // NAN = legacy random load (worst case)
    double ldpc_fps = 0.0;      // 0 = free-run; else pace to this frame rate
    double ldpc_llr_scale = 8.0; // LLR scale for --ldpc-ebn0 (live demap ~2)
    std::string ldpc_modcod = "qpsk3/5";
    float rolloff = 0.2f;
    bool profile_stage = false;
    bool profile_stage_direct = false;
    std::string dump_samples;   // path: write raw per-iter sync latencies (us) for CDF

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stride" && i + 1 < argc) stride = atof(argv[++i]);
        else if (arg == "--profile-stage") profile_stage = true;
        else if (arg == "--profile-stage-direct") profile_stage_direct = profile_stage = true;
        else if (arg == "--dump-samples" && i + 1 < argc) dump_samples = argv[++i];
        else if (arg == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (arg == "--n-input" && i + 1 < argc) n_input = atoi(argv[++i]);
        else if (arg == "--est-period" && i + 1 < argc) est_period = atoi(argv[++i]);
        else if (arg == "--energy-k" && i + 1 < argc) energy_k = atoi(argv[++i]);
        else if (arg == "--search-cap" && i + 1 < argc) search_cap = atoi(argv[++i]);
        else if (arg == "--ldpc-load" && i + 1 < argc) ldpc_batch = atoi(argv[++i]);
        else if (arg == "--ldpc-trials" && i + 1 < argc) ldpc_trials = atoi(argv[++i]);
        else if (arg == "--ldpc-ebn0" && i + 1 < argc) ldpc_ebn0 = atof(argv[++i]);
        else if (arg == "--ldpc-fps" && i + 1 < argc) ldpc_fps = atof(argv[++i]);
        else if (arg == "--ldpc-llr-scale" && i + 1 < argc) ldpc_llr_scale = atof(argv[++i]);
        else if (arg == "--ldpc-modcod" && i + 1 < argc) ldpc_modcod = argv[++i];
        else if (arg == "--rolloff" && i + 1 < argc) rolloff = atof(argv[++i]);
        else {
            fprintf(stderr,
                "Usage: %s [--stride S] [--iters N] [--n-input N]\n"
                "          [--est-period N] [--energy-k K] [--search-cap N]\n"
                "          [--ldpc-load BATCH] [--ldpc-trials N] [--ldpc-modcod mc]\n"
                "          [--ldpc-ebn0 dB] [--ldpc-fps F]\n"
                "          [--rolloff R] [--profile-stage] [--profile-stage-direct]\n"
                "  --profile-stage: measure timing_recovery_cuda::stage_input()\n"
                "      only: host copy+enqueue time, stream-serialized time,\n"
                "      and host wait-to-complete time.\n"
                "  --profile-stage-direct: additionally compare pinned-source\n"
                "      direct H2D, simulating a cudaMallocHost RX input buffer.\n"
                "  --ldpc-ebn0: realistic load (encoded codewords + AWGN at the\n"
                "      operating point -> live convergence/early-exit pattern)\n"
                "      instead of the default never-converging worst case.\n"
                "  --ldpc-fps: pace decode batches to this frame rate instead\n"
                "      of free-running (mimics live arrival rate).\n",
                argv[0]);
            return 1;
        }
    }

    // Same prototype selection as the trx app
    const int rrc_delay = (stride < 2.5) ? 16 : 5;
    filter_bank_design design;
    design.samples_per_symbol = stride;
    design.rolloff = rolloff;
    design.span_symbols = rrc_delay;
    design.n_phases = 128;
    polyphase_filter_bank interp(design);
    const int nf = (int)interp.n_phases();
    const int fl = (int)interp.taps_per_phase();
    std::vector<float> flat = interp.flat_taps();

    cuda::timing_recovery_cuda tr(nf, fl, flat.data(), stride,
                                  interp.group_delay_samples(), n_input);
    tr.set_estimation_period(est_period);
    tr.set_energy_k(energy_k);
    tr.set_energy_search_cap(search_cap);
    tr.set_profiling(false);   // match trx at debug_level < 2

    // Same history formula as receiver_frontend
    const unsigned ss_history =
        (unsigned)(fl - 1) + (unsigned)std::ceil(stride / 2.0) + 2;

    // Random noise input (launch cost is data-independent)
    std::vector<complexf> in(n_input);
    std::mt19937 rng(42);
    std::normal_distribution<float> g(0.0f, 0.5f);
    for (auto& c : in) c = complexf(g(rng), g(rng));

    const int max_syms = (int)((double)n_input / stride) + 2;
    std::vector<complexf> sym_out(max_syms);

    fprintf(stderr, "bench-timing-recovery: stride=%.6f n_input=%d "
            "subfilt=%dx%d ss_history=%u est_period=%d K=%d cap=%d "
            "ldpc_load=%d\n",
            stride, n_input, nf, fl, ss_history, est_period, energy_k,
            search_cap, ldpc_batch);

    if (profile_stage) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        std::vector<uint32_t> v_host, v_stream, v_wait, v_total;
        std::vector<uint32_t> v_d_host, v_d_stream, v_d_wait, v_d_total;
        v_host.reserve(iters); v_stream.reserve(iters);
        v_wait.reserve(iters); v_total.reserve(iters);
        v_d_host.reserve(iters); v_d_stream.reserve(iters);
        v_d_wait.reserve(iters); v_d_total.reserve(iters);

        const float* src = reinterpret_cast<const float*>(in.data());
        float* pinned_src = nullptr;
        if (profile_stage_direct) {
            cudaMallocHost((void**)&pinned_src, (size_t)n_input * 2 * sizeof(float));
            memcpy(pinned_src, src, (size_t)n_input * 2 * sizeof(float));
        }
        for (int i = 0; i < warmup + iters; i++) {
            cudaEventRecord(ev0, tr.stream());
            auto t0 = clk::now();
            tr.stage_input(src, n_input);
            auto t1 = clk::now();
            cudaEventRecord(ev1, tr.stream());
            cudaEventSynchronize(ev1);
            auto t2 = clk::now();

            if (i >= warmup) {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                v_host.push_back(us_between(t0, t1));
                v_stream.push_back((uint32_t)llround(ms * 1000.0f));
                v_wait.push_back(us_between(t1, t2));
                v_total.push_back(us_between(t0, t2));
            }

            if (profile_stage_direct) {
                size_t bytes = (size_t)n_input * 2 * sizeof(float);
                cudaEventRecord(ev0, tr.stream());
                auto td0 = clk::now();
                cudaMemcpyAsync(tr.device_input(), pinned_src, bytes,
                                cudaMemcpyHostToDevice, tr.stream());
                auto td1 = clk::now();
                cudaEventRecord(ev1, tr.stream());
                cudaEventSynchronize(ev1);
                auto td2 = clk::now();
                if (i >= warmup) {
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, ev0, ev1);
                    v_d_host.push_back(us_between(td0, td1));
                    v_d_stream.push_back((uint32_t)llround(ms * 1000.0f));
                    v_d_wait.push_back(us_between(td1, td2));
                    v_d_total.push_back(us_between(td0, td2));
                }
            }
        }

        if (pinned_src) cudaFreeHost(pinned_src);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        double mib = (double)n_input * 2.0 * sizeof(float) / (1024.0 * 1024.0);
        double gbps = mean_of(v_total) > 0.0
            ? ((double)n_input * 2.0 * sizeof(float)) / mean_of(v_total) / 1000.0
            : 0.0;
        printf("stage_input: n_input=%d bytes=%.2f MiB iters=%d\n",
               n_input, mib, iters);
        printf("  host copy+enqueue us: mean=%7.1f p50=%7.1f p99=%7.1f\n",
               mean_of(v_host), pctile(v_host, 0.5), pctile(v_host, 0.99));
        printf("  stream serialized us: mean=%7.1f p50=%7.1f p99=%7.1f\n",
               mean_of(v_stream), pctile(v_stream, 0.5), pctile(v_stream, 0.99));
        printf("  host wait us:         mean=%7.1f p50=%7.1f p99=%7.1f\n",
               mean_of(v_wait), pctile(v_wait, 0.5), pctile(v_wait, 0.99));
        printf("  host total us:        mean=%7.1f p50=%7.1f p99=%7.1f "
               "(%.1f GB/s effective)\n",
               mean_of(v_total), pctile(v_total, 0.5), pctile(v_total, 0.99),
               gbps);
        if (profile_stage_direct) {
            double d_gbps = mean_of(v_d_total) > 0.0
                ? ((double)n_input * 2.0 * sizeof(float)) / mean_of(v_d_total) / 1000.0
                : 0.0;
            printf("direct pinned-source H2D:\n");
            printf("  host enqueue us:      mean=%7.1f p50=%7.1f p99=%7.1f\n",
                   mean_of(v_d_host), pctile(v_d_host, 0.5), pctile(v_d_host, 0.99));
            printf("  stream serialized us: mean=%7.1f p50=%7.1f p99=%7.1f\n",
                   mean_of(v_d_stream), pctile(v_d_stream, 0.5), pctile(v_d_stream, 0.99));
            printf("  host wait us:         mean=%7.1f p50=%7.1f p99=%7.1f\n",
                   mean_of(v_d_wait), pctile(v_d_wait, 0.5), pctile(v_d_wait, 0.99));
            printf("  host total us:        mean=%7.1f p50=%7.1f p99=%7.1f "
                   "(%.1f GB/s effective)\n",
                   mean_of(v_d_total), pctile(v_d_total, 0.5),
                   pctile(v_d_total, 0.99), d_gbps);
        }
        return 0;
    }

    // Optional concurrent LDPC decode load. Default input is random soft
    // values that never converge (all trials burned = worst-case block
    // lifetimes). --ldpc-ebn0 switches to encoded codewords + AWGN so the
    // early-exit/convergence pattern matches live operation; --ldpc-fps
    // paces batches to the live frame arrival rate instead of free-running.
    std::atomic<bool> stop{false};
    std::atomic<bool> load_ready{false};
    std::atomic<long> ldpc_frames{0};
    std::atomic<long> ldpc_iters{0};
    std::thread loader;
    if (ldpc_batch > 0) {
        loader = std::thread([&] {
            code_rate_t rate = rate_from_modcod(ldpc_modcod.c_str());
            ldpc_decode ldpc(FECFRAME_NORMAL, rate,
                             ldpc_trials, ldpc_batch);
            const int N = ldpc.n_bits();
            const int K_bytes = ldpc.k_bytes();
            std::vector<int16_t> soft((size_t)N * ldpc_batch);
            std::vector<uint8_t> hard((size_t)K_bytes * ldpc_batch);
            std::vector<int> trials(ldpc_batch);
            std::mt19937 r2(7);

            // Pool of pre-noised batches (regenerating noise per batch would
            // make the load thread CPU-bound and open decode gaps).
            std::vector<std::vector<int16_t>> pool;
            if (!std::isnan(ldpc_ebn0)) {
                ldpc_encode enc(FECFRAME_NORMAL, rate);
                const int N_bytes = (N + 7) / 8;
                std::vector<uint8_t> info(K_bytes), cw(N_bytes);
                for (auto& b : info) b = (uint8_t)(r2() & 0xff);
                enc.encode(info.data(), cw.data());
                auto cw_bit = [&](int i) {
                    return (cw[i >> 3] >> (7 - (i & 7))) & 1; };
                // LLR sign convention probe (same as bench-ldpc-decode)
                double sign = 1.0;
                for (int c = 0; c < ldpc_batch; c++)
                    for (int i = 0; i < N; i++)
                        soft[(size_t)c * N + i] = cw_bit(i) ? -3000 : 3000;
                ldpc.decode_batch_cuda(soft.data(), hard.data(),
                                       trials.data(), ldpc_batch);
                if (memcmp(hard.data(), info.data(), K_bytes) != 0)
                    sign = -1.0;

                const double R = (double)(K_bytes * 8) / N;
                double sigma = sqrt(1.0 /
                    (2.0 * R * pow(10.0, ldpc_ebn0 / 10.0)));
                double llr_gain = sign * ldpc_llr_scale * 2.0 / (sigma * sigma);
                std::normal_distribution<float> gauss(0.0f, 1.0f);
                pool.resize(4);
                for (auto& p : pool) {
                    p.resize((size_t)N * ldpc_batch);
                    for (int c = 0; c < ldpc_batch; c++)
                        for (int i = 0; i < N; i++) {
                            float s = cw_bit(i) ? -1.0f : 1.0f;
                            double v = llr_gain * (s + sigma * gauss(r2));
                            if (v > 32767.0) v = 32767.0;
                            if (v < -32768.0) v = -32768.0;
                            p[(size_t)c * N + i] = (int16_t)lrint(v);
                        }
                }
            } else {
                for (auto& s : soft) s = (int16_t)((int)(r2() % 21) - 10);
            }

            const bool paced = ldpc_fps > 0.0;
            const auto period = std::chrono::duration_cast<clk::duration>(
                std::chrono::duration<double>(ldpc_batch / std::max(ldpc_fps, 1.0)));
            auto next = clk::now();
            size_t pi = 0;
            load_ready = true;
            while (!stop.load(std::memory_order_relaxed)) {
                if (paced) {
                    next += period;
                    if (next > clk::now())
                        std::this_thread::sleep_until(next);
                    else
                        next = clk::now();  // overrun: don't accumulate backlog
                }
                const int16_t* src = pool.empty()
                    ? soft.data() : pool[pi++ % pool.size()].data();
                ldpc.decode_batch_cuda(src, hard.data(),
                                       trials.data(), ldpc_batch);
                for (int c = 0; c < ldpc_batch; c++)
                    ldpc_iters += (trials[c] >= 0) ? ldpc_trials - trials[c]
                                                   : ldpc_trials;
                ldpc_frames += ldpc_batch;
            }
        });
        // Let the load finish setup (encode + noise pool) and reach steady
        // state before measuring
        while (!load_ready.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    cuda::timing_recovery_cuda::state_t st;
    int n_consumed = 0;

    // Stage once; every iteration reuses the device buffer (skip_input_copy),
    // matching the trx path where AGC staging happens upstream.
    tr.stage_input(reinterpret_cast<const float*>(in.data()), n_input);

    std::vector<uint32_t> v_lnch, v_sync, v_est, v_intp;
    v_lnch.reserve(iters); v_sync.reserve(iters);
    v_est.reserve(iters); v_intp.reserve(iters);
    long total_syms = 0;
    int est_launches = 0;

    auto t_bench0 = clk::now();
    for (int i = 0; i < warmup + iters; i++) {
        auto t0 = clk::now();
        tr.process_launch(in.data(), n_input, max_syms, st, ss_history, true);
        auto t1 = clk::now();
        int n = tr.process_complete(sym_out.data(), max_syms, st, n_consumed);
        auto t2 = clk::now();
        if (i >= warmup) {
            v_lnch.push_back(us_between(t0, t1));
            v_sync.push_back(us_between(t1, t2));
            v_est.push_back(tr.last_launch_est_us());
            v_intp.push_back(tr.last_launch_interp_us());
            total_syms += n;
            if (tr.last_launched_energy()) est_launches++;
        }
    }
    auto t_bench1 = clk::now();

    stop = true;
    if (loader.joinable()) loader.join();

    double wall_ms = std::chrono::duration<double, std::milli>(
        t_bench1 - t_bench0).count();
    // average (lnch+sync) per call divided by avg syms per call
    double per_sym_ns = (total_syms > 0)
        ? (mean_of(v_lnch) + mean_of(v_sync)) * 1000.0 /
          ((double)total_syms / iters)
        : 0.0;

    printf("stride=%.6f iters=%d syms/call=%.0f energy_launches=%d/%d\n",
           stride, iters, (double)total_syms / iters, est_launches, iters);
    printf("  lnch  us: mean=%7.1f p50=%7.1f p99=%7.1f\n",
           mean_of(v_lnch), pctile(v_lnch, 0.5), pctile(v_lnch, 0.99));
    printf("    est us: mean=%7.1f p50=%7.1f p99=%7.1f\n",
           mean_of(v_est), pctile(v_est, 0.5), pctile(v_est, 0.99));
    printf("    intp us: mean=%6.1f p50=%7.1f p99=%7.1f\n",
           mean_of(v_intp), pctile(v_intp, 0.5), pctile(v_intp, 0.99));
    printf("  sync  us: mean=%7.1f p50=%7.1f p99=%7.1f max=%7.1f\n",
           mean_of(v_sync), pctile(v_sync, 0.5), pctile(v_sync, 0.99),
           pctile(v_sync, 1.0));
    printf("  timing path: %.2f ns/sym; bench wall %.1f ms",
           per_sym_ns, wall_ms);
    if (ldpc_batch > 0) {
        printf("; ldpc load: %.0f fr/s",
               (double)ldpc_frames.load() / (wall_ms / 1000.0));
        if (ldpc_frames.load() > 0)
            printf(" it_mean=%.1f",
                   (double)ldpc_iters.load() / ldpc_frames.load());
    }
    printf("\n");

    /* Raw per-iteration sync latencies (us) -> one value per line, for a CDF.
     * In insertion (iteration) order; the plotter sorts. A CDF only needs the
     * multiset of samples, so order is irrelevant either way. */
    if (!dump_samples.empty()) {
        FILE* f = fopen(dump_samples.c_str(), "w");
        if (!f) {
            fprintf(stderr, "ERROR: cannot open --dump-samples '%s'\n",
                    dump_samples.c_str());
            return 1;
        }
        for (uint32_t s : v_sync)
            fprintf(f, "%u\n", s);
        fclose(f);
        fprintf(stderr, "wrote %zu sync-latency samples -> %s\n",
                v_sync.size(), dump_samples.c_str());
    }
    return 0;
}
