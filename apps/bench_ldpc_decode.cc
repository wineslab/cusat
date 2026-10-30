// SPDX-License-Identifier: Apache-2.0
/*
 * Micro-benchmark for GPU LDPC decode.
 *
 * Two modes:
 *  - Worst-case (default): random non-codeword LLRs — every frame runs the
 *    full trials budget. Measures the deterministic throughput floor.
 *  - AWGN (--ebn0 <list>): encodes REAL codewords, BPSK-maps them onto a
 *    binary-input AWGN channel at the given Eb/N0 points, and decodes.
 *    Reports FER / info-BER, the iteration distribution, and the
 *    early-termination throughput (decode-time only, LLR generation
 *    excluded). This is the quality gate for decoder optimization work:
 *    run identical seeds before/after a change and diff the curves.
 *
 * LLR convention is auto-detected with a noiseless probe batch at startup
 * (mapping flips if hard decisions don't match the encoder output).
 *
 * Note: FER here is LDPC-level (full k_bits block compared, BCH inner bits
 * included as data). ETSI QEF tables include the outer BCH cleanup, so
 * absolute dB anchors differ slightly; A/B comparisons are exact.
 */

#include "acm_config.h"
#include "fec/code_params.h"
#include "ldpc_decode.h"
#include "ldpc_encode.h"
#include <algorithm>
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

static code_rate_t rate_from_modcod(const char* modcod)
{
    // Any MODCOD of acm_config.h; the LDPC code is determined by the rate.
    const modcod_def_t* mc = find_modcod_by_name(modcod);
    if (!mc) {
        fprintf(stderr, "unknown MODCOD '%s'\n", modcod);
        exit(1);
    }
    return mc->code_rate;
}

static double code_rate_value(code_rate_t r)
{
    const fec_params_t p = get_fec_params(FECFRAME_NORMAL, r);
    return (double)p.k_ldpc / (double)p.n_ldpc;
}

int main(int argc, char* argv[])
{
    int n_batches = 100;
    int gpu_batch = 16;
    int max_trials = 25;
    const char* modcod = "qpsk3/5";
    std::vector<double> ebn0_list;
    long frames_per_point = 2000;
    long max_frame_errors = 200;
    double llr_scale = 8.0;
    double pace_fps = 0.0;
    unsigned seed = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--batches" && i+1 < argc) n_batches = atoi(argv[++i]);
        else if (arg == "--batch-size" && i+1 < argc) gpu_batch = atoi(argv[++i]);
        else if (arg == "--modcod" && i+1 < argc) modcod = argv[++i];
        else if (arg == "--trials" && i+1 < argc) max_trials = atoi(argv[++i]);
        else if (arg == "--frames" && i+1 < argc) frames_per_point = atol(argv[++i]);
        else if (arg == "--max-errors" && i+1 < argc) max_frame_errors = atol(argv[++i]);
        else if (arg == "--llr-scale" && i+1 < argc) llr_scale = atof(argv[++i]);
        else if (arg == "--pace-fps" && i+1 < argc) pace_fps = atof(argv[++i]);
        else if (arg == "--seed" && i+1 < argc) seed = (unsigned)atol(argv[++i]);
        else if (arg == "--ebn0" && i+1 < argc) {
            char* s = argv[++i];
            for (char* tok = strtok(s, ","); tok; tok = strtok(nullptr, ","))
                ebn0_list.push_back(atof(tok));
        } else {
            fprintf(stderr,
                "Usage: %s [--modcod mc] [--batch-size N] [--trials N]\n"
                "  worst-case mode: [--batches N] [--pace-fps FPS]\n"
                "  AWGN mode:       --ebn0 dB[,dB...] [--frames N]\n"
                "                   [--max-errors N] [--llr-scale S] [--seed S]\n",
                argv[0]);
            return 1;
        }
    }

    code_rate_t rate = rate_from_modcod(modcod);
    double R = code_rate_value(rate);

    ldpc_decode ldpc(FECFRAME_NORMAL, rate, max_trials, gpu_batch);
    const int N = (int)ldpc.n_bits();
    const int K_bytes = (int)ldpc.k_bytes();
    const int K_bits = K_bytes * 8;

    std::vector<int16_t> soft((size_t)N * gpu_batch);
    std::vector<uint8_t> hard((size_t)K_bytes * gpu_batch);
    std::vector<int> trials(gpu_batch);

    /* ---------------- Worst-case mode (original behavior) -------------- */
    if (ebn0_list.empty()) {
        fprintf(stderr, "LDPC decode benchmark (worst-case): N=%d batch=%d "
                "trials=%d, %d batches\n", N, gpu_batch, max_trials, n_batches);
        srand(seed);
        for (auto& s : soft)
            s = (int16_t)((rand() % 21) - 10); // small magnitude: never converges

        ldpc.decode_batch_cuda(soft.data(), hard.data(), trials.data(), gpu_batch);

        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        const auto batch_period = std::chrono::duration<double>(
            pace_fps > 0.0 ? gpu_batch / pace_fps : 0.0);
        double busy_ms = 0.0;
        int total_frames = 0;
        for (int b = 0; b < n_batches; b++) {
            auto d0 = clock::now();
            ldpc.decode_batch_cuda(soft.data(), hard.data(), trials.data(), gpu_batch);
            auto d1 = clock::now();
            busy_ms += std::chrono::duration<double, std::milli>(d1 - d0).count();
            total_frames += gpu_batch;
            if (pace_fps > 0.0) {
                auto release = t0 + std::chrono::duration_cast<clock::duration>(
                    batch_period * (b + 1));
                std::this_thread::sleep_until(release);
            }
        }
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        fprintf(stderr, "Total: %.1f ms for %d frames (%.2f ms/frame, %.2f ms/batch)\n",
                ms, total_frames, ms / total_frames, ms / n_batches);
        fprintf(stderr, "Throughput: %.0f frames/sec\n", total_frames / (ms / 1000.0));
        if (pace_fps > 0.0)
            fprintf(stderr, "Paced: offered=%.0f frames/sec, GPU-busy=%.1f%%, "
                    "decode-only=%.0f frames/sec\n",
                    pace_fps, 100.0 * busy_ms / ms,
                    total_frames / (busy_ms / 1000.0));
        return 0;
    }

    /* ---------------------------- AWGN mode ---------------------------- */
    ldpc_encode enc(FECFRAME_NORMAL, rate);
    const int N_bytes = (N + 7) / 8;

    // Codeword pool: one distinct codeword per batch slot (fresh noise per
    // use dominates the statistics; re-encoding per frame would make the
    // CPU the bottleneck).
    std::mt19937 rng(seed);
    std::vector<std::vector<uint8_t>> pool_info(gpu_batch),
        pool_cw(gpu_batch);
    for (int c = 0; c < gpu_batch; c++) {
        pool_info[c].resize(K_bytes);
        pool_cw[c].resize(N_bytes);
        for (auto& b : pool_info[c]) b = (uint8_t)(rng() & 0xff);
        enc.encode(pool_info[c].data(), pool_cw[c].data());
    }

    auto cw_bit = [&](int c, int i) -> int {
        return (pool_cw[c][i >> 3] >> (7 - (i & 7))) & 1;
    };

    // --- LLR sign convention probe: noiseless, huge confidence ---
    double sign = 1.0;   // llr = sign * (bit ? -big : +big)
    {
        for (int c = 0; c < gpu_batch; c++)
            for (int i = 0; i < N; i++)
                soft[(size_t)c * N + i] = cw_bit(c, i) ? -3000 : 3000;
        ldpc.decode_batch_cuda(soft.data(), hard.data(), trials.data(), gpu_batch);
        if (memcmp(hard.data(), pool_info[0].data(), K_bytes) != 0) {
            sign = -1.0;
            for (int c = 0; c < gpu_batch; c++)
                for (int i = 0; i < N; i++)
                    soft[(size_t)c * N + i] = cw_bit(c, i) ? 3000 : -3000;
            ldpc.decode_batch_cuda(soft.data(), hard.data(), trials.data(),
                                   gpu_batch);
            if (memcmp(hard.data(), pool_info[0].data(), K_bytes) != 0) {
                fprintf(stderr, "FATAL: noiseless decode failed under both "
                        "LLR conventions — wiring bug\n");
                return 2;
            }
        }
        fprintf(stderr, "LLR convention: positive = bit %s\n",
                sign > 0 ? "0" : "1");
    }

    fprintf(stderr, "LDPC AWGN benchmark: %s (R=%.3f) N=%d batch=%d trials=%d "
            "llr_scale=%.1f seed=%u frames/point=%ld\n",
            modcod, R, N, gpu_batch, max_trials, llr_scale, seed,
            frames_per_point);
    printf("# ebn0_dB esn0q_dB frames ferrs FER info_BER it_mean it_p95 "
           "it_max decode_frps\n");

    std::normal_distribution<float> gauss(0.0f, 1.0f);

    for (double ebn0 : ebn0_list) {
        double ebn0_lin = pow(10.0, ebn0 / 10.0);
        double sigma = sqrt(1.0 / (2.0 * R * ebn0_lin));
        double llr_gain = sign * llr_scale * 2.0 / (sigma * sigma);

        long frames = 0, ferrs = 0, bit_errs = 0;
        double decode_s = 0.0;
        std::vector<int> iters;
        iters.reserve(frames_per_point);

        while (frames < frames_per_point && ferrs < max_frame_errors) {
            for (int c = 0; c < gpu_batch; c++) {
                int16_t* dst = &soft[(size_t)c * N];
                for (int i = 0; i < N; i++) {
                    float s = cw_bit(c, i) ? -1.0f : 1.0f;
                    float x = s + (float)sigma * gauss(rng);
                    double v = llr_gain * x;
                    if (v > 32767.0) v = 32767.0;
                    if (v < -32768.0) v = -32768.0;
                    dst[i] = (int16_t)lrint(v);
                }
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            ldpc.decode_batch_cuda(soft.data(), hard.data(), trials.data(),
                                   gpu_batch);
            auto t1 = std::chrono::high_resolution_clock::now();
            decode_s += std::chrono::duration<double>(t1 - t0).count();

            for (int c = 0; c < gpu_batch; c++) {
                const uint8_t* h = &hard[(size_t)c * K_bytes];
                long be = 0;
                for (int j = 0; j < K_bytes; j++)
                    be += __builtin_popcount(h[j] ^ pool_info[c][j]);
                bit_errs += be;
                if (be > 0)
                    ferrs++;
                iters.push_back(trials[c] >= 0 ? max_trials - trials[c]
                                               : max_trials);
            }
            frames += gpu_batch;
        }

        std::sort(iters.begin(), iters.end());
        double it_mean = 0;
        for (int v : iters) it_mean += v;
        it_mean /= iters.size();
        int it_p95 = iters[(size_t)(0.95 * (iters.size() - 1))];
        int it_max = iters.back();
        double esn0q = ebn0 + 10.0 * log10(2.0 * R);

        printf("%.2f %.2f %ld %ld %.3e %.3e %.1f %d %d %.0f\n",
               ebn0, esn0q, frames, ferrs, (double)ferrs / frames,
               (double)bit_errs / ((double)frames * K_bits),
               it_mean, it_p95, it_max, frames / decode_s);
        fflush(stdout);
    }
    return 0;
}
