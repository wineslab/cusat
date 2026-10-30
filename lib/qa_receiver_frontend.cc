/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Full receiver frontend (GPU AGC/rotator/timing + PL receiver):
 * R1.7d rotator sign/scale, acquisition time, CFO range, CPU time of the
 * PL receiver per symbol, and noise-only false lock.
 * RRC-shaped streams are synthesized at any (fractional) sps from the
 * continuous pulse.
 */

#include "cusat_dsp/acm_config.h"
#include "cusat_dsp/receiver_frontend.h"
#include "cusat_dsp/rrc_filter.h"
#include "pl_test_stream.h"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

namespace cusat_dsp {

namespace {

double scale()
{
    const char* e = getenv("CUSAT_QA_SCALE");
    return e ? atof(e) : 1.0;
}

/* RRC pulse shaping at fractional sps (span 10 symbols), CFO in
 * cycles/symbol, AWGN at Es/N0, random delay. */
std::vector<complexf> shape(const std::vector<complexf>& sym, double sps, double cfo,
                            double esn0_db, double delay, uint32_t seed, int lead_symbols,
                            double rolloff = 0.2)
{
    const int span = 10;
    const int n = (int)((sym.size() + lead_symbols) * sps);
    std::vector<complexf> y(n);
    std::mt19937 rng(seed);
    const double sigma = std::sqrt(0.5 * std::pow(10.0, -esn0_db / 10.0) * sps);
    std::normal_distribution<double> g(0.0, sigma);
    for (int i = 0; i < n; i++) {
        const double t = i / sps - delay - lead_symbols; // symbols
        const int kc = (int)std::floor(t);
        std::complex<double> acc = 0;
        for (int k = std::max(0, kc - span); k <= std::min((int)sym.size() - 1, kc + span); k++)
            acc += std::complex<double>(sym[k]) * rrc_impulse(t - k, rolloff);
        acc *= std::polar(1.0, 2 * M_PI * cfo * (i / sps));
        if (esn0_db < 99)
            acc += std::complex<double>(g(rng), g(rng));
        y[i] = complexf((float)acc.real(), (float)acc.imag());
    }
    return y;
}

struct fe_result {
    int frames = 0;
    uint64_t first_sym = 0;   // symbol index (input time) of the first delivered frame
    double cfo = 0.0;
    double sync_us = 0.0;
    long syms = 0;
    double max_call_ms = 0.0;
    int locked_calls = 0, calls = 0;
};

fe_result run_frontend(receiver_frontend& fe, const std::vector<complexf>& x, double sps)
{
    fe_result r;
    const int chunk = 100000; // within the frontend's maximum GPU batch
    const int history = fe.history();
    std::vector<complexf> in(chunk + 4 * history + 1000), out(400000);
    int n_buf = 0;
    size_t off = 0;
    uint64_t consumed_total = 0;
    while (off < x.size() || n_buf > (int)fe.history() + 200) {
        const int take = (int)std::min<size_t>(in.size() - n_buf, x.size() - off);
        if (take > 0)
            memcpy(in.data() + n_buf, &x[off], take * sizeof(complexf));
        off += take;
        const int n_in = n_buf + take;
        int n_consumed = 0;
        std::vector<frontend_frame_t> frames;
        fe.process(in.data(), n_in, out.data(), (int)out.size(), n_consumed, frames);
        const auto& p = fe.perf();
        r.sync_us += p.sync_us;
        r.syms += p.n_syms;
        r.max_call_ms = std::max(r.max_call_ms, p.sync_us / 1000.0);
        r.calls++;
        r.locked_calls += fe.locked();
        if (!frames.empty() && r.frames == 0)
            r.first_sym = (uint64_t)((consumed_total + n_consumed) / sps);
        r.frames += (int)frames.size();
        consumed_total += n_consumed;
        n_buf = n_in - n_consumed;
        if (n_consumed > 0 && n_buf > 0)
            memmove(in.data(), in.data() + n_consumed, n_buf * sizeof(complexf));
        if (n_consumed == 0 && take == 0)
            break;
    }
    std::vector<frontend_frame_t> frames;
    r.frames += fe.flush(out.data(), (int)out.size(), frames) >= 0 ? (int)frames.size() : 0;
    r.cfo = fe.freq_offset();
    return r;
}

/* below 1.25 sps the frontend requires a roll-off < 0.2 (1 + a + 0.05) */
double rolloff_for(double sps) { return sps < 1.25 ? 0.1 : 0.2; }

receiver_frontend_config ccm_cfg(double sps, uint8_t code, bool symbols_in = false)
{
    receiver_frontend_config c;
    c.sps = (float)sps;
    c.rolloff = (float)rolloff_for(sps);
    c.rrc_delay = sps < 2.5 ? 16 : 5;
    c.symbols_in = symbols_in;
    c.pl.deliver_codes = { code };
    return c;
}

std::vector<complexf> make_symbols(uint8_t code, int n_frames, uint32_t seed)
{
    pltest::framer fr(0, seed);
    std::vector<complexf> s;
    for (int i = 0; i < n_frames; i++)
        fr.append(code, s);
    return s;
}

} // namespace

/* R1.7d: a CFO injected at the input is cancelled by carrier_frequency()
 * (freq_offset()) to 1e-6 cycles/symbol for sps 1.2288 / 2 / 4.096 and in
 * symbols-in mode. Also the acquisition range and time. */
BOOST_AUTO_TEST_CASE(test_rotator_sign_and_scale)
{
    const uint8_t code = pls_code_for(5, FECFRAME_NORMAL, true); // qpsk3/5
    const int n_frames = 45;
    const auto sym = make_symbols(code, n_frames, 2);
    for (double sps : { 1.2288, 2.0, 4.096, 1.0 }) {
        for (double cfo : { -0.05, 0.0123, 0.05 }) {
            const bool symbols_in = sps == 1.0;
            std::vector<complexf> xin;
            if (!symbols_in) {
                xin = shape(sym, sps, cfo, 10.0, 0.37, 3, 500, rolloff_for(sps));
            } else { // on-time symbols: no pulse shaping
                std::mt19937 rng(5);
                std::normal_distribution<float> g(0.0f, 0.22f);
                xin.resize(sym.size() + 500);
                for (size_t i = 0; i < xin.size(); i++) {
                    const complexf s = i >= 500 ? sym[i - 500] : complexf(0, 0);
                    xin[i] = s * std::polar(1.0f, (float)(2 * M_PI * cfo * i)) + complexf(g(rng), g(rng));
                }
            }
            receiver_frontend fe(ccm_cfg(symbols_in ? 1.0 : sps, code, symbols_in));
            const fe_result r = run_frontend(fe, xin, symbols_in ? 1.0 : sps);
            BOOST_TEST_CONTEXT("sps " << sps << " cfo " << cfo)
            {
                // frames after lock validation (~25 PLFRAMEs, frontend gate)
                BOOST_CHECK_GE(r.frames, n_frames - 30);
                BOOST_CHECK_SMALL(r.cfo - cfo, 1e-6);
                BOOST_TEST_MESSAGE("sps " << sps << " cfo " << cfo << ": frames " << r.frames
                                          << " est " << r.cfo << " first frame after "
                                          << r.first_sym / 33282.0 << " PLFRAMEs, PL CPU "
                                          << 1000.0 * r.sync_us / std::max(1L, r.syms)
                                          << " ns/sym, max call " << r.max_call_ms << " ms");
            }
        }
    }
}

/* Delivered-symbol quality through the full frontend (timing recovery +
 * PL receiver): SNR from the error vector against the transmitted data
 * versus the channel Es/N0 (the receiver implementation loss at symbol
 * level) and the rms per-symbol phase error (R1.7b). */
BOOST_AUTO_TEST_CASE(test_data_quality)
{
    for (const char* name : { "qpsk1/2", "8psk3/5", "16apsk2/3", "32apsk3/4", "64apsk4/5", "256apsk3/4" }) {
        const modcod_def_t* m = find_modcod_by_name(name);
        const uint8_t code = pls_code_for(m->modcod_idx, FECFRAME_NORMAL, true);
        const double esn0 = m->min_snr_db + 1.0;
        for (double sps : { 4.096, 2.0 }) {
            pltest::framer fr(0, 31);
            std::vector<complexf> sym;
            std::vector<std::vector<complexf>> data;
            const int n_tx = getenv("QA_FRAMES") ? atoi(getenv("QA_FRAMES")) : 40;
            for (int i = 0; i < n_tx; i++)
                data.push_back(fr.append(code, sym).data);
            const auto x = shape(sym, sps, 0.021, esn0, 0.3, 41, 0);
            receiver_frontend fe(ccm_cfg(sps, code));
            // collect delivered frames
            const int chunk = 100000;
            std::vector<complexf> in(chunk + 4 * fe.history() + 1000), out(400000);
            std::vector<std::vector<complexf>> got;
            int n_buf = 0;
            size_t off = 0;
            while (off < x.size()) {
                const int take = (int)std::min<size_t>(in.size() - n_buf, x.size() - off);
                memcpy(in.data() + n_buf, &x[off], take * sizeof(complexf));
                off += take;
                const int n_in = n_buf + take;
                int n_consumed = 0;
                std::vector<frontend_frame_t> frames;
                fe.process(in.data(), n_in, out.data(), (int)out.size(), n_consumed, frames);
                for (auto& f : frames)
                    got.emplace_back(f.payload, f.payload + f.payload_len);
                n_buf = n_in - n_consumed;
                if (n_consumed > 0 && n_buf > 0)
                    memmove(in.data(), in.data() + n_consumed, n_buf * sizeof(complexf));
            }
            // align with the transmitted frames by correlation
            double err = 0, sig = 0, ph2 = 0;
            long n = 0;
            for (const auto& g : got) {
                int best = -1;
                double bc = 0;
                for (size_t k = 0; k < data.size(); k++) {
                    std::complex<double> c = 0;
                    for (int i = 0; i < 2000; i++)
                        c += std::complex<double>(g[i]) * std::conj(std::complex<double>(data[k][i]));
                    if (std::abs(c) > bc) {
                        bc = std::abs(c);
                        best = (int)k;
                    }
                }
                const auto& t = data[best];
                double fe = 0, fs = 0;
                for (size_t i = 0; i < g.size(); i++) {
                    fe += std::norm(std::complex<double>(g[i]) - std::complex<double>(t[i]));
                    fs += std::norm(std::complex<double>(t[i]));
                }
                if (getenv("QA_DEBUG") && 10 * std::log10(fs / fe) < esn0 - 0.5) {
                    printf("frame %d: EVM-SNR %.2f dB; per-slot worst:", best, 10 * std::log10(fs / fe));
                    for (size_t s0 = 0; s0 + 90 <= g.size(); s0 += 90) {
                        double se = 0;
                        for (int i = 0; i < 90; i++)
                            se += std::norm(std::complex<double>(g[s0 + i]) - std::complex<double>(t[s0 + i]));
                        if (10 * std::log10(90.0 / se) < esn0 - 4)
                            printf(" %zu", s0 / 90);
                    }
                    printf("\n");
                }
                for (size_t i = 0; i < g.size(); i++) {
                    const std::complex<double> d(g[i]), r(t[i]);
                    err += std::norm(d - r);
                    sig += std::norm(r);
                    ph2 += std::pow(std::arg(d * std::conj(r)), 2) * std::norm(r);
                    n++;
                }
            }
            BOOST_REQUIRE_GT(n, 0);
            const double snr_evm = 10 * std::log10(sig / err);
            // phase error of a symbol ~ noise/|x| + tracking error; report the
            // noise-free part sqrt(<ph^2> - N0 / (2 Es)) as the tracking error
            const double n0 = std::pow(10.0, -esn0 / 10.0);
            const double ph_rms = std::sqrt(std::max(0.0, ph2 / sig - n0 / 2.0)) * 180 / M_PI;
            BOOST_TEST_MESSAGE(name << " sps " << sps << ": Es/N0 " << esn0 << " dB, EVM-SNR " << snr_evm
                                    << " dB (loss " << esn0 - snr_evm << " dB), tracking phase rms ~"
                                    << ph_rms << " deg, frames " << got.size());
            BOOST_CHECK_LE(esn0 - snr_evm, 0.3);
        }
    }
}

/* Study helper (skipped unless QA_IQ_CLEAN/QA_IQ_NOISY are set): EVM of a
 * noisy IQ file against the same file without noise, both through the
 * frontend (e.g. dvbs2-gse-tx with and without --awgn-snr-db). */
BOOST_AUTO_TEST_CASE(test_iq_file_evm)
{
    const char* fc = getenv("QA_IQ_CLEAN");
    const char* fn = getenv("QA_IQ_NOISY");
    const char* mc = getenv("QA_MODCOD");
    if (!fc || !fn || !mc)
        return;
    const modcod_def_t* m = find_modcod_by_name(mc);
    const uint8_t code = pls_code_for(m->modcod_idx, FECFRAME_NORMAL, true);
    auto load = [](const char* f) {
        FILE* fp = fopen(f, "rb");
        std::vector<complexf> v;
        complexf buf[65536];
        size_t n;
        while (fp && (n = fread(buf, sizeof(complexf), 65536, fp)) > 0)
            v.insert(v.end(), buf, buf + n);
        if (fp)
            fclose(fp);
        return v;
    };
    std::map<uint64_t, std::vector<complexf>> frames[2];
    for (int w = 0; w < 2; w++) {
        const auto x = load(w ? fn : fc);
        receiver_frontend fe(ccm_cfg(4.0, code));
        std::vector<complexf> in(100000 + 4 * fe.history() + 1000), out(400000);
        int n_buf = 0;
        size_t off = 0;
        uint64_t idx = 0;
        while (off < x.size()) {
            const int take = (int)std::min<size_t>(in.size() - n_buf, x.size() - off);
            memcpy(in.data() + n_buf, &x[off], take * sizeof(complexf));
            off += take;
            const int n_in = n_buf + take;
            int n_consumed = 0;
            std::vector<frontend_frame_t> fr;
            fe.process(in.data(), n_in, out.data(), (int)out.size(), n_consumed, fr);
            for (auto& f : fr) // key: first 4 symbols hashed loosely -> use order
                frames[w][idx++] = std::vector<complexf>(f.payload, f.payload + f.payload_len);
            n_buf = n_in - n_consumed;
            if (n_consumed > 0 && n_buf > 0)
                memmove(in.data(), in.data() + n_consumed, n_buf * sizeof(complexf));
        }
    }
    // align by correlation of frame 0 of the noisy run within the clean run
    double e = 0, sg = 0;
    long nf = 0;
    for (auto& [k, g] : frames[1]) {
        int best = -1;
        double bc = 0;
        for (auto& [k2, c] : frames[0]) {
            if (std::abs((long)k2 - (long)k) > 5)
                continue;
            std::complex<double> acc = 0;
            for (int i = 0; i < 3000; i++)
                acc += std::complex<double>(g[i]) * std::conj(std::complex<double>(c[i]));
            if (std::abs(acc) > bc) {
                bc = std::abs(acc);
                best = (int)k2;
            }
        }
        if (best < 0)
            continue;
        const auto& c = frames[0][best];
        double fe = 0, fs = 0;
        for (size_t i = 0; i < g.size(); i++) {
            fe += std::norm(std::complex<double>(g[i]) - std::complex<double>(c[i]));
            fs += std::norm(std::complex<double>(c[i]));
        }
        e += fe;
        sg += fs;
        nf++;
        if (getenv("QA_DEBUG"))
            printf("frame %ld: EVM-SNR %.2f dB\n", (long)k, 10 * std::log10(fs / fe));
    }
    BOOST_TEST_MESSAGE("iq-file EVM-SNR " << 10 * std::log10(sg / e) << " dB over " << nf << " frames");
}

/* R1.2 (reduced): acquisition time at min_snr + 1 dB, random CFO in
 * +-0.05, random delay, sps 4.096. */
BOOST_AUTO_TEST_CASE(test_acquisition_time)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (const char* name : { "qpsk1/4", "8psk3/5", "16apsk2/3", "32apsk3/4" }) {
        const modcod_def_t* m = find_modcod_by_name(name);
        const uint8_t code = pls_code_for(m->modcod_idx, FECFRAME_NORMAL, true);
        const int trials = std::max(2, (int)(6 * scale()));
        std::vector<double> t;
        for (int k = 0; k < trials; k++) {
            const auto sym = make_symbols(code, 60, 10 + k);
            const double cfo = -0.05 + 0.1 * u(rng);
            const auto x = shape(sym, 4.096, cfo, m->min_snr_db + 1.0, u(rng), 20 + k, 0);
            receiver_frontend fe(ccm_cfg(4.096, code));
            const fe_result r = run_frontend(fe, x, 4.096);
            const double plf = pl_layout_from_code(code).plframe_symbols();
            t.push_back(r.frames ? r.first_sym / plf : 1e9);
        }
        std::sort(t.begin(), t.end());
        BOOST_TEST_MESSAGE(name << ": acquisition (PLFRAMEs) median " << t[t.size() / 2] << " max "
                                << t.back());
        BOOST_TEST_CONTEXT(name)
        {
            BOOST_CHECK_LE(t.back(), 60.0);
            BOOST_CHECK_LE(t[t.size() / 2], 30.0);
        }
    }
}

/* R1.3 (reduced): pure noise through the full frontend. */
BOOST_AUTO_TEST_CASE(test_noise_false_lock)
{
    const uint8_t code = pls_code_for(5, FECFRAME_NORMAL, true);
    receiver_frontend fe(ccm_cfg(4.0, code));
    std::mt19937 rng(9);
    std::normal_distribution<float> g(0.0f, 0.7f);
    const long n = (long)(2e7 * scale()); // samples
    std::vector<complexf> x(n);
    for (auto& v : x)
        v = complexf(g(rng), g(rng));
    const fe_result r = run_frontend(fe, x, 4.0);
    BOOST_TEST_MESSAGE("noise: " << n / 4.0 / 5e6 << " s at 5 Msym/s, frames " << r.frames
                                 << ", locked calls " << r.locked_calls << "/" << r.calls
                                 << ", PL CPU " << 1000.0 * r.sync_us / std::max(1L, r.syms)
                                 << " ns/sym (acquisition)");
    BOOST_CHECK_EQUAL(r.frames, 0);
    BOOST_CHECK_LE(r.locked_calls, r.calls / 100 + 1);
}

} // namespace cusat_dsp
