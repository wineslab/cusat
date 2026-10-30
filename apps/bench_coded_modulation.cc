/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Coded-modulation AWGN FER waterfall: encode -> real bit-interleave+map ->
 * complex AWGN at a target Eb/N0 -> real GPU demapper -> LDPC decode. Unlike
 * bench-ldpc-decode's BPSK-proxy AWGN mode, this exercises the actual
 * constellation + demapper, so the curve captures the MODULATION's effect
 * (APSK ring geometry, demapper LLR quality), not just the LDPC code. Per
 * modcod using the production INT16 APP decoder.
 *
 * Calibration matches the deployed RX exactly (rx_fec_chain): unit-energy
 * symbols, N0 = 1/SNR_lin, precision = min(4/N0, 1000), where
 * SNR_lin = Es/N0 and Es/N0[dB] = Eb/N0[dB] + 10log10(n_mod * code_rate).
 *
 * Example:
 *   bench-coded-modulation --modcod 16apsk3/4 --ebn0 4,4.5,5,5.5,6 \
 *       --max-errors 200 --frames 2000000 --trials 10 --batch 64
 */

#include "ldpc_encode.h"
#include "ldpc_decode.h"
#include "fec/code_params.h"
#include <memory>
#include <cuda_runtime.h>
#include "mapper.h"
#include "cuda/demapper_kernel.cuh"
#include "rx_fec_chain.h"   // --async: exercise the live RX async micro-batch path
#include "rx_fec_pool.h"    // --async-pool: drive the multi-chain ACM pool offline
#include "acm_config.h"     // modcod table (name -> idx/constellation/rate)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace cusat_dsp;

static code_rate_t rate_from_modcod(const char* m)
{
    if (!strcmp(m,"qpsk1/4")) return RATE_1_4;
    if (!strcmp(m,"qpsk1/2")) return RATE_1_2;
    if (!strcmp(m,"qpsk3/5")||!strcmp(m,"8psk3/5")) return RATE_3_5;
    if (!strcmp(m,"qpsk2/3")||!strcmp(m,"16apsk2/3")||!strcmp(m,"32apsk2/3l")) return RATE_2_3;
    if (!strcmp(m,"qpsk3/4")||!strcmp(m,"16apsk3/4")||!strcmp(m,"32apsk3/4")) return RATE_3_4;
    if (!strcmp(m,"128apsk3/4")) return RATE_135_180;
    if (!strcmp(m,"256apsk3/4")) return RATE_135_180;
    if (!strcmp(m,"256apsk32/45")) return RATE_128_180;
    if (!strcmp(m,"256apsk29/45")) return RATE_116_180;
    if (!strcmp(m,"256apsk31/45")) return RATE_124_180;
    if (!strcmp(m,"256apsk2/3")) return RATE_2_3;
    if (!strcmp(m,"256apsk11/15")) return RATE_132_180;
    if (!strcmp(m,"128apsk7/9")) return RATE_140_180;
    if (!strcmp(m,"64apsk11/15")) return RATE_132_180;
    if (!strcmp(m,"64apsk7/9")) return RATE_140_180;
    if (!strcmp(m,"qpsk13/45")) return RATE_13_45;
    if (!strcmp(m,"32apsk32/45")) return RATE_128_180;
    if (!strcmp(m,"32apsk11/15")) return RATE_132_180;
    if (!strcmp(m,"32apsk7/9")) return RATE_140_180;
    if (!strcmp(m,"qpsk9/20")) return RATE_9_20;
    if (!strcmp(m,"qpsk11/20")) return RATE_11_20;
    if (!strcmp(m,"8psk23/36")||!strcmp(m,"16apsk23/36")) return RATE_23_36;
    if (!strcmp(m,"8psk25/36")||!strcmp(m,"16apsk25/36")) return RATE_25_36;
    if (!strcmp(m,"8psk13/18")||!strcmp(m,"16apsk13/18")) return RATE_13_18;
    if (!strcmp(m,"8apsk5/9")) return RATE_100_180;
    if (!strcmp(m,"8apsk26/45")) return RATE_104_180;
    if (!strcmp(m,"16apsk1/2l")) return RATE_90_180;
    if (!strcmp(m,"16apsk8/15l")) return RATE_96_180;
    if (!strcmp(m,"16apsk5/9l")) return RATE_100_180;
    if (!strcmp(m,"16apsk3/5l")) return RATE_18_30;
    if (!strcmp(m,"16apsk2/3l")) return RATE_20_30;
    if (!strcmp(m,"16apsk26/45")) return RATE_26_45;
    if (!strcmp(m,"16apsk3/5")) return RATE_3_5;
    if (!strcmp(m,"16apsk28/45")) return RATE_28_45;
    if (!strcmp(m,"16apsk7/9")) return RATE_140_180;
    if (!strcmp(m,"16apsk77/90")) return RATE_154_180;
    if (!strcmp(m,"64apsk4/5")||!strcmp(m,"qpsk4/5")) return RATE_4_5;
    if (!strcmp(m,"64apsk5/6")) return RATE_5_6;
    if (!strcmp(m,"64apsk32/45")) return RATE_128_180;
    if (!strcmp(m,"qpsk5/6")||!strcmp(m,"16apsk5/6")) return RATE_5_6;
    if (!strcmp(m,"32apsk8/9")||!strcmp(m,"qpsk8/9")) return RATE_8_9;
    if (!strcmp(m,"qpsk9/10")) return RATE_9_10;
    return RATE_3_5;
}

static modulation_t cons_from_modcod(const char* m)
{
    if (!strncmp(m,"qpsk",4))    return MOD_QPSK;
    if (!strncmp(m,"8apsk",5))   return MOD_8APSK;
    if (!strncmp(m,"8psk",4))    return MOD_8PSK;
    if (!strncmp(m,"16apsk",6))  return MOD_16APSK;
    if (!strncmp(m,"32apsk",6))  return MOD_32APSK;
    if (!strncmp(m,"64apsk",6))  return MOD_64APSK;
    if (!strncmp(m,"128apsk",7)) return MOD_128APSK;
    if (!strncmp(m,"256apsk",7)) return MOD_256APSK;
    return MOD_QPSK;
}

int main(int argc, char** argv)
{
    const char* modcod = "16apsk3/4";
    std::vector<double> ebn0_list;
    long frames_cap = 2000000, max_errors = 200;
    int trials = 10, batch = 64;
    unsigned seed = 1;
    double llr_factor = 4.0;   // precision = min(llr_factor/N0, 1000); 4 = deployed
    bool short_frame = false;
    bool async_mode = false;   // route symbols through rx_fec_chain (live RX path)
    int async_streams = 8, async_micro = 4, async_maxb = 4; // live-crash config
    int async_partial = 0;     // call launch_partial() every N frames (mimics ACM modcod switches)
    const char* async_pool = nullptr; // comma list of modcod NAMES -> drive rx_fec_pool with alternating modcods

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--modcod" && i+1 < argc) modcod = argv[++i];
        else if (a == "--short") short_frame = true;
        else if (a == "--ebn0" && i+1 < argc) {
            char* s = argv[++i];
            for (char* t = strtok(s, ","); t; t = strtok(nullptr, ","))
                ebn0_list.push_back(atof(t));
        }
        else if (a == "--frames" && i+1 < argc) frames_cap = atol(argv[++i]);
        else if (a == "--max-errors" && i+1 < argc) max_errors = atol(argv[++i]);
        else if (a == "--trials" && i+1 < argc) trials = atoi(argv[++i]);
        else if (a == "--batch" && i+1 < argc) batch = atoi(argv[++i]);
        else if (a == "--seed" && i+1 < argc) seed = atoi(argv[++i]);
        else if (a == "--async") async_mode = true;
        else if (a == "--async-streams" && i+1 < argc) async_streams = atoi(argv[++i]);
        else if (a == "--async-micro" && i+1 < argc) async_micro = atoi(argv[++i]);
        else if (a == "--async-maxb" && i+1 < argc) async_maxb = atoi(argv[++i]);
        else if (a == "--async-partial" && i+1 < argc) async_partial = atoi(argv[++i]);
        else if (a == "--async-pool" && i+1 < argc) async_pool = argv[++i];
        else if (a == "--llr-factor" && i+1 < argc) llr_factor = atof(argv[++i]);
        else { fprintf(stderr, "Usage: %s --modcod mc --ebn0 dB[,dB...] "
                "[--max-errors N] [--frames N] [--trials N] [--batch N] "
                "[--seed S]\n", argv[0]);
               return 1; }
    }
    if (ebn0_list.empty()) { fprintf(stderr, "need --ebn0\n"); return 1; }

    const fecframe_size_t fs = short_frame ? FECFRAME_SHORT : FECFRAME_NORMAL;

    // --- ASYNC-POOL mode: drive the LIVE multi-chain rx_fec_pool with frames
    // of ALTERNATING modcods. Each submit() with a changed modcod triggers
    // launch_partial() on the chain being left (cross-chain, mixed-geometry
    // path), which is useful for radio-free ACM stress tests.
    //   --async-pool qpsk1/4,128apsk7/9   (names from the modcod table)
    if (async_pool) {
        std::vector<std::string> names;
        { std::string s(async_pool), cur;
          for (char ch : s) { if (ch==',') { if(!cur.empty()) names.push_back(cur); cur.clear(); } else cur+=ch; }
          if (!cur.empty()) names.push_back(cur); }
        if (names.size() < 2)
            fprintf(stderr, "WARN --async-pool: <2 modcods, no switching will occur\n");

        // Per-modcod: resolve table entry, build clean normalized codeword pool.
        struct pool_mc {
            uint8_t idx; const char* name; int n_mod; unsigned xfec; double R;
            std::vector<complexf> clean;   // batch * xfec, unit-Es normalized
            float esn0_db, sigma;
            pl_layout pls;
        };
        std::vector<pool_mc> mcs;
        std::vector<uint8_t> enabled;
        std::mt19937 prng(seed);
        for (const auto& nm : names) {
            const modcod_def_t* def = find_modcod_by_name(nm.c_str());
            if (!def) { fprintf(stderr, "unknown modcod '%s'\n", nm.c_str()); return 1; }
            pool_mc m;
            m.idx = def->modcod_idx; m.name = def->name;
            ldpc_encode penc(fs, def->code_rate);
            mapper      pmap(fs, def->code_rate, def->constellation);
            fec_params_t  pfi; pfi = get_fec_params(fs, def->code_rate);
            m.R = (double)pfi.k_ldpc / (double)pfi.n_ldpc;
            m.n_mod = pmap.n_mod();
            m.xfec  = pmap.xfecframe_len();
            unsigned Nbits = pmap.fecframe_len();
            int Kb = (int)(pfi.k_ldpc / 8);
            int Npad = (int)((Nbits + 90 + 7) / 8);
            m.clean.assign((size_t)batch * m.xfec, complexf(0,0));
            double es_sum = 0.0;
            std::vector<uint8_t> info(Kb), cwb(Npad);
            for (int c = 0; c < batch; c++) {
                for (auto& b : info) b = (uint8_t)(prng() & 0xff);
                std::fill(cwb.begin(), cwb.end(), 0);
                penc.encode(info.data(), cwb.data());
                pmap.interleave_and_map(cwb.data(), &m.clean[(size_t)c * m.xfec]);
                for (unsigned k = 0; k < m.xfec; k++) {
                    const complexf& s = m.clean[(size_t)c * m.xfec + k];
                    es_sum += (double)s.real()*s.real() + (double)s.imag()*s.imag();
                }
            }
            double es = es_sum / ((double)batch * m.xfec);
            float norm = (float)(1.0 / std::sqrt(es));
            for (auto& s : m.clean) s = complexf(s.real()*norm, s.imag()*norm);
            // PLS info for routing (modcod is all the pool uses for routing).
            m.pls = pl_layout_for(m.idx, fs, /*pilots*/ true);
            mcs.push_back(std::move(m));
            enabled.push_back(def->modcod_idx);
        }

        rx_fec_pool pool(fs, enabled, trials,
                         async_maxb, async_streams, async_micro);
        if (getenv("CUSAT_BENCH_CONV")) pool.set_diag_ldpc_convergence(true);
        std::vector<uint8_t> gse_out(1 << 20);
        std::normal_distribution<float> gn(0.0f, 1.0f);

        for (double ebn0 : ebn0_list) {
            for (auto& m : mcs) {
                m.esn0_db = (float)(ebn0 + 10.0 * std::log10((double)m.n_mod * m.R));
                float N0 = (float)(1.0 / std::pow(10.0, m.esn0_db / 10.0));
                m.sigma = std::sqrt(N0 / 2.0f);
                pool.update_snr(m.idx, m.esn0_db);
            }
            pool.reset_all();
            std::vector<complexf> nsym;
            long fr = 0;
            while (fr < frames_cap) {
                // Alternate modcods every frame -> max switch frequency.
                pool_mc& m = mcs[fr % mcs.size()];
                nsym.resize(m.xfec);
                int c = (int)(fr % batch);
                for (unsigned k = 0; k < m.xfec; k++) {
                    const complexf& s = m.clean[(size_t)c * m.xfec + k];
                    nsym[k] = complexf(s.real() + m.sigma * gn(prng),
                                       s.imag() + m.sigma * gn(prng));
                }
                if (pool.submit(nsym.data(), (int)m.xfec, m.pls))
                    pool.get_gse_output(gse_out.data(), (int)gse_out.size());
                pool.drain_all_completed();
                if (++fr % 2000 == 0)
                    fprintf(stderr, "  ebn0=%.1f fr=%ld decoded=%lu ferr=%lu\n",
                            ebn0, fr, (unsigned long)pool.total_frame_count(),
                            (unsigned long)pool.total_error_count());
            }
            pool.finish_all_pending();
            pool.drain_all_completed();
            uint64_t fc = pool.total_frame_count(), fe = pool.total_error_count();
            printf("%.2f %lu %lu %.3e (async-pool n_mc=%zu)\n", ebn0,
                   (unsigned long)fc, (unsigned long)fe,
                   (double)fe / (double)std::max<uint64_t>(1, fc), mcs.size());
            fflush(stdout);
        }
        return 0;
    }

    // Resolve from the authoritative acm_config table; the legacy string maps
    // above silently default unmapped names (e.g. 8psk9/10) to RATE_3_5/QPSK.
    code_rate_t rate;
    modulation_t cons;
    if (const modcod_def_t* mcdef = find_modcod_by_name(modcod)) {
        rate = mcdef->code_rate;
        cons = mcdef->constellation;
    } else {
        fprintf(stderr, "WARN: '%s' not in acm_config table, legacy name map\n",
                modcod);
        rate = rate_from_modcod(modcod);
        cons = cons_from_modcod(modcod);
    }
    // Use the ACTUAL LDPC rate (k_ldpc/n_ldpc) for the Es/N0 axis, not the
    // nominal label — DVB-S2 short codes differ (e.g. short 1/2 = 7200/16200
    // = 0.444, not 0.5). For normal frames this equals the nominal rate.
    fec_params_t fi_r;
    fi_r = get_fec_params(fs, rate);
    const double R = (double)fi_r.k_ldpc / (double)fi_r.n_ldpc;

    ldpc_encode enc(fs, rate);
    mapper map(fs, rate, cons);
    cuda::demapper_cuda demap(cons, rate, fs, batch);
    ldpc_decode dec(fs, rate, trials, batch);

    const int n_mod = map.n_mod();
    const unsigned N = map.fecframe_len();          // LDPC codeword bits
    const unsigned xfec = map.xfecframe_len();       // symbols/frame
    const int K_bytes = dec.k_bytes();           // info bytes
    const int N_bytes_pad = (int)((N + 90 + 7) / 8); // padded (128APSK reads past)

    fprintf(stderr, "coded-modulation AWGN: %s  n_mod=%d R=%.3f N=%u xfec=%u "
            "trials=%d batch=%d int16 APP\n", modcod, n_mod, R, N,
            xfec, trials, batch);
    printf("# modcod=%s n_mod=%d R=%.4f trials=%d\n", modcod, n_mod, R, trials);
    printf("# ebn0_dB esn0_dB frames ferrs FER info_BER it_mean\n");

    // --- Build a pool of clean (noise-free) modulated codewords ---
    std::mt19937 rng(seed);
    std::vector<std::vector<uint8_t>> info(batch), cw(batch);
    std::vector<complexf> clean((size_t)batch * xfec);
    double es_sum = 0.0;
    for (int c = 0; c < batch; c++) {
        info[c].resize(K_bytes);
        cw[c].assign(N_bytes_pad, 0);
        for (auto& b : info[c]) b = (uint8_t)(rng() & 0xff);
        enc.encode(info[c].data(), cw[c].data());
        map.interleave_and_map(cw[c].data(), &clean[(size_t)c * xfec]);
        for (unsigned k = 0; k < xfec; k++) {
            const complexf& s = clean[(size_t)c * xfec + k];
            es_sum += (double)s.real()*s.real() + (double)s.imag()*s.imag();
        }
    }
    // Normalize to unit average symbol energy so N0 = 1/SNR_lin is exact.
    double es = es_sum / ((double)batch * xfec);
    float norm = (float)(1.0 / std::sqrt(es));
    for (auto& s : clean) s = complexf(s.real()*norm, s.imag()*norm);
    fprintf(stderr, "  pool: %d codewords, measured Es=%.4f -> normalized\n",
            batch, es);

    std::normal_distribution<float> g(0.0f, 1.0f);

    // --- ASYNC mode: route noisy symbols through the LIVE RX async path
    // (rx_fec_chain: demap -> async micro-batch LDPC -> BCH -> GSE), which the
    // sync decode below never hits. Useful under compute-sanitizer at low Eb/N0
    // to stress failing-decode paths. ---
    if (async_mode) {
        fprintf(stderr, "ASYNC mode: rx_fec_chain streams=%d micro_batch=%d "
                "max_batch=%d (live RX path)\n",
                async_streams, async_micro, async_maxb);
        rx_fec_chain fec(fs, rate, cons, trials,
                         async_maxb, async_streams, async_micro);
        if (getenv("CUSAT_BENCH_CONV")) fec.set_diag_ldpc_convergence(true);
        std::vector<complexf> nsym(xfec);
        std::vector<uint8_t> gse_out(1 << 20);
        for (double ebn0 : ebn0_list) {
            double esn0_db = ebn0 + 10.0 * std::log10((double)n_mod * R);
            float N0 = (float)(1.0 / std::pow(10.0, esn0_db / 10.0));
            float sigma = std::sqrt(N0 / 2.0f);
            fec.reset();
            // Set the chain's noise variance / LLR gain from the known Es/N0 (the OTA frontend
            // does this from pilot SNR). Without it the chain's data-based SNR
            // self-estimate is unreliable for dense constellations (128APSK+),
            // mis-scaling LLRs -> FER=1.0 at all SNR (masks real decode behavior).
            fec.update_snr((float)esn0_db);
            long fr = 0;
            while (fr < frames_cap) {
                int c = (int)(fr % batch);
                for (unsigned k = 0; k < xfec; k++) {
                    const complexf& s = clean[(size_t)c * xfec + k];
                    nsym[k] = complexf(s.real() + sigma * g(rng),
                                       s.imag() + sigma * g(rng));
                }
                if (fec.submit_xfecframe(nsym.data(), (int)xfec))
                    fec.get_gse_output(gse_out.data(), (int)gse_out.size());
                // Mimic ACM modcod-switch: flush the partial micro-batch mid-stream
                // (this is what rx_fec_pool::submit does on the chain being left).
                if (async_partial && (fr % async_partial) == 0)
                    fec.launch_partial();
                fec.drain_completed();
                if (++fr % 2000 == 0)
                    fprintf(stderr, "  ebn0=%.1f fr=%ld decoded=%lu ferr=%lu\n",
                            ebn0, fr, (unsigned long)fec.frame_count(),
                            (unsigned long)fec.frame_error_count());
            }
            fec.flush();
            fec.drain_completed();
            uint64_t fc = fec.frame_count(), fe = fec.frame_error_count();
            printf("%.2f %.2f %lu %lu %.3e (async)\n", ebn0, esn0_db,
                   (unsigned long)fc, (unsigned long)fe,
                   (double)fe / (double)std::max<uint64_t>(1, fc));
            fflush(stdout);
        }
        return 0;
    }

    std::vector<float> noisy((size_t)batch * xfec * 2);
    std::vector<uint8_t> hard((size_t)batch * K_bytes);
    std::vector<int> tr(batch);
#define CUDA_OK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { \
    fprintf(stderr, "!! CUDA %s\n", cudaGetErrorString(e_)); exit(1); } } while(0)

    for (double ebn0 : ebn0_list) {
        double esn0_db = ebn0 + 10.0 * std::log10((double)n_mod * R);
        double snr_lin = std::pow(10.0, esn0_db / 10.0);
        float  N0 = (float)(1.0 / snr_lin);
        float  precision = std::min((float)(llr_factor / N0), 1000.0f);
        float  sigma = std::sqrt(N0 / 2.0f);   // per-component std

        long fr = 0, ferr = 0, bit_err = 0, it_acc = 0;
        long ferr_le_t = 0;   /* failed frames the outer BCH would correct */
        /* Anytime-FER histogram: with syndrome period 1, a frame converging
         * clean at k iterations behaves identically at any cap T'>=k, so one
         * run at a large cap yields FER(T') for ALL T' from the histogram. */
        std::vector<long> conv_hist(trials + 1, 0);
        while (fr < frames_cap && ferr < max_errors) {
            for (int c = 0; c < batch; c++)
                for (unsigned k = 0; k < xfec; k++) {
                    const complexf& s = clean[(size_t)c * xfec + k];
                    noisy[((size_t)c*xfec + k)*2]   = s.real() + sigma*g(rng);
                    noisy[((size_t)c*xfec + k)*2+1] = s.imag() + sigma*g(rng);
                }
            demap.demap_batch(noisy.data(), batch, N0, precision);
            dec.decode_batch_cuda_d2h(demap.soft_out_dev(), hard.data(),
                                      tr.data(), batch);
            for (int c = 0; c < batch; c++) {
                long be = 0;
                for (int j = 0; j < K_bytes; j++)
                    be += __builtin_popcount(hard[(size_t)c*K_bytes + j]
                                             ^ info[c][j]);
                bit_err += be;
                if (be > 0) {
                    ferr++;
                    if (be <= (long)fi_r.t_bch) ferr_le_t++;
                } else if (tr[c] >= 0) {
                    int used = trials - tr[c];
                    if (used >= 0 && used <= trials) conv_hist[used]++;
                }
                it_acc += (tr[c] >= 0) ? trials - tr[c] : trials;
            }
            fr += batch;
        }
        /* cols 8-9: failed frames within BCH correction radius (<=t bit
         * errors, exact by the BCH guarantee) and the resulting SYSTEM
         * (post-BCH) FER. Col 5 stays the LDPC-alone (pre-BCH) FER. */
        {
            fprintf(stderr, "# ithist:");
            long cum = 0;
            for (int k = 0; k <= trials; k++) {
                cum += conv_hist[k];
                fprintf(stderr, " %ld", fr - cum);
            }
            fprintf(stderr, "  (failed frames at cap k, frames=%ld)\n", fr);
        }
        printf("%.2f %.2f %ld %ld %.3e %.3e %.1f %ld %.3e\n", ebn0, esn0_db,
               fr, ferr, (double)ferr / fr,
               (double)bit_err / ((double)fr * K_bytes * 8),
               (double)it_acc / fr, ferr_le_t,
               (double)(ferr - ferr_le_t) / fr);
        fflush(stdout);
    }
    return 0;
}
