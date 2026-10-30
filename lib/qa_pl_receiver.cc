/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL receiver alone: synthetic symbol-spaced
 * streams with ideal timing, channel impairments and an emulated frontend
 * rotator (carrier feedback with latency).
 *
 * Frame counts: CUSAT_QA_SCALE (default 1) multiplies the per-case frame
 * counts; the defaults give >= 1e4 frames for the dummy, short-frame and
 * ACM cases and 2000 normal frames for the impairment cases (CI time).
 */

#include "cusat_dsp/acm_config.h"
#include "cusat_dsp/plrx/pl_receiver.h"
#include "pl_test_stream.h"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

namespace cusat_dsp {

namespace {

constexpr uint32_t NO_FEEDBACK = 0xFFFFFFFFu; // carrier commands not applied

int scaled(int n)
{
    const char* e = getenv("CUSAT_QA_SCALE");
    const double s = e ? atof(e) : 1.0;
    return std::max(1, (int)(n * s));
}

uint8_t code_of(const char* name, fecframe_size_t fs = FECFRAME_NORMAL, bool pilots = true)
{
    const modcod_def_t* m = find_modcod_by_name(name);
    BOOST_REQUIRE(m);
    return pls_code_for(m->modcod_idx, fs, pilots);
}

struct run_stats {
    long delivered = 0;
    long wrong_layout = 0;
    long unknown_sof = 0;
    long bad = 0;            // EVM far above the channel noise
    long lost = 0, acquired = 0;
    uint64_t first_sof = 0;
    long tx_data_after_lock = 0;
    double evm_sum = 0, phase_sq = 0, gain_sum = 0;
    long n_evm = 0;
    double snr_err_sum = 0;
    long n_snr = 0;
    std::vector<pl_frame> frames_copy;
    std::vector<std::vector<complexf>> data_copy;
};

struct stream {
    std::vector<complexf> tx;
    std::map<uint64_t, pltest::tx_frame> sent;
};

/* Feeds `s` through the channel and the receiver in batches. */
run_stats run(const stream& s, pl_receiver& rx, const pltest::impairments& imp, uint32_t latency,
              const std::vector<int>& batches, double esn0_db, bool keep = false)
{
    run_stats st;
    pltest::channel ch(imp);
    std::vector<complexf> buf, out(200000);
    std::vector<pl_frame> frames;
    std::vector<pl_sync_event> ev;
    size_t off = 0;
    size_t bi = 0;
    while (off < s.tx.size()) {
        const int n = (int)std::min<size_t>(batches[bi++ % batches.size()], s.tx.size() - off);
        buf.resize(n);
        ch.apply(&s.tx[off], buf.data(), n);
        off += n;
        frames.clear();
        ev.clear();
        rx.process(buf.data(), n, nullptr, out.data(), (int)out.size(), frames, ev);
        for (const auto& e : ev) {
            st.lost += e.type == pl_sync_event_type::lost;
            st.acquired += e.type == pl_sync_event_type::acquired;
        }
        double f;
        if (rx.take_carrier_update(f) && latency != NO_FEEDBACK)
            ch.set_rotator(f, latency);
        for (const pl_frame& fm : frames) {
            st.delivered++;
            if (!st.first_sof)
                st.first_sof = fm.sof_symbol;
            auto it = s.sent.find(fm.sof_symbol);
            if (it == s.sent.end()) {
                st.unknown_sof++;
                continue;
            }
            if (it->second.pls != fm.layout.pls_code) {
                st.wrong_layout++;
                continue;
            }
            const auto& t = it->second.data;
            BOOST_REQUIRE_EQUAL((int)t.size(), fm.n_data);
            double e = 0, p = 0;
            std::complex<double> cr = 0;
            for (int i = 0; i < fm.n_data; i++) {
                const std::complex<double> d(fm.data[i]), x(t[i]);
                e += std::norm(d - x);
                p += std::norm(x);
                cr += d * std::conj(x);
            }
            const double evm = e / p;
            st.evm_sum += evm;
            const double ph = std::arg(cr);
            st.phase_sq += ph * ph;
            st.gain_sum += std::abs(cr) / p;
            st.n_evm++;
            if (evm > 3.0 * std::pow(10.0, -esn0_db / 10.0) + 0.02) {
                st.bad++;
                if (getenv("QA_DEBUG"))
                    printf("bad frame sof %llu evm %.3f phase %.3f resid %.3e coh %.3f\n",
                           (unsigned long long)fm.sof_symbol, evm, ph, fm.residual_cfo, fm.coherence);
            }
            if (fm.snr_valid) {
                st.snr_err_sum += fm.snr_db - esn0_db;
                st.n_snr++;
            }
            if (keep) {
                st.frames_copy.push_back(fm);
                st.data_copy.emplace_back(fm.data, fm.data + fm.n_data);
            }
        }
    }
    // a frame is decided once the next header has arrived: the last frame
    // of the stream is never complete
    for (const auto& kv : s.sent) {
        const uint64_t end = kv.first + pl_layout_from_code(kv.second.pls).plframe_symbols() + 91;
        if (st.first_sof && kv.first >= st.first_sof && !kv.second.data.empty() && end <= s.tx.size())
            st.tx_data_after_lock++;
    }
    return st;
}

stream make_ccm(uint8_t code, int n_frames, uint32_t seed, int dummy_every = 0, bool random_dummy = false)
{
    stream s;
    pltest::framer fr(0, seed);
    for (int i = 0; i < n_frames; i++) {
        if (dummy_every && i % dummy_every == dummy_every - 1)
            fr.append(0, s.tx, random_dummy);
        const uint64_t sof = fr.count();
        s.sent[sof] = fr.append(code, s.tx);
    }
    return s;
}

pl_rx_config ccm_config(uint8_t code, uint32_t latency)
{
    pl_rx_config c;
    c.deliver_codes = { code };
    c.correction_latency_symbols = latency;
    c.valid_plframe_lens = { (uint32_t)pl_layout_from_code(code).plframe_symbols() };
    return c;
}

} // namespace

/* §1.4.1 configuration validation */
BOOST_AUTO_TEST_CASE(test_config_validation)
{
    pl_rx_config c;
    BOOST_CHECK_THROW(pl_receiver{ c }, std::invalid_argument);           // empty
    c.deliver_codes = { 0 };
    BOOST_CHECK_THROW(pl_receiver{ c }, std::invalid_argument);           // dummy
    c.deliver_codes = { code_of("qpsk1/2", FECFRAME_NORMAL, false) };
    BOOST_CHECK_THROW(pl_receiver{ c }, std::invalid_argument);           // no pilots
    c.deliver_codes = { 120 };                                            // reserved MODCOD 30
    BOOST_CHECK_THROW(pl_receiver{ c }, std::invalid_argument);
    c.deliver_codes = { code_of("qpsk1/2"), code_of("qpsk3/5") };
    BOOST_CHECK_THROW(pl_receiver{ c }, std::invalid_argument);           // ccm, two MODCODs
    c.deliver_codes = { code_of("qpsk1/2"), code_of("qpsk1/2", FECFRAME_SHORT) };
    BOOST_CHECK_THROW(pl_receiver{ c }, std::invalid_argument);           // ccm, two sizes
    c.mode = pl_service_mode::vcm_acm;
    BOOST_CHECK_NO_THROW(pl_receiver{ c });
}

/* R1.1 / R1.2 / R1.5 / R1.7 (PL receiver alone): CCM QPSK 1/2 at
 * min_snr + 1 dB over the acquisition range, rotator latency of 4 PLFRAMEs. */
BOOST_AUTO_TEST_CASE(test_ccm_acquisition_and_delivery)
{
    const uint8_t code = code_of("qpsk1/2");
    const double esn0 = find_modcod_by_name("qpsk1/2")->min_snr_db + 1.0;
    const uint32_t latency = 4 * 33282;
    for (double cfo : { -0.05, -0.021, 0.0, 0.0337, 0.05, 0.10 }) {
        stream s = make_ccm(code, 60, 5);
        pl_receiver rx(ccm_config(code, latency));
        pltest::impairments imp;
        imp.cfo = cfo;
        imp.esn0_db = esn0;
        imp.gain = 0.6;
        imp.phase0 = 2.0;
        imp.seed = 11;
        run_stats st = run(s, rx, imp, latency, { 30000 }, esn0);
        BOOST_TEST_CONTEXT("cfo " << cfo)
        {
            BOOST_CHECK_EQUAL(st.acquired, 1);
            BOOST_CHECK_EQUAL(st.lost, 0);
            BOOST_CHECK_EQUAL(st.unknown_sof + st.wrong_layout, 0);
            BOOST_CHECK_EQUAL(st.bad, 0);
            // every data frame after the first delivered one
            BOOST_CHECK_EQUAL(st.delivered, st.tx_data_after_lock);
            // acquisition (signal start -> first delivered frame) <= 15 PLFRAMEs here
            BOOST_CHECK_LE(st.first_sof, 15u * 33282u);
            BOOST_CHECK_SMALL(std::fabs(rx.carrier_frequency() - cfo), 2e-6);
            const double rms_phase_deg = std::sqrt(st.phase_sq / st.n_evm) * 180 / M_PI;
            BOOST_CHECK_LE(rms_phase_deg, 2.0);
            BOOST_CHECK_SMALL(20 * std::log10(st.gain_sum / st.n_evm), 0.25);
            BOOST_CHECK_SMALL(st.snr_err_sum / std::max(1L, st.n_snr), 0.5);
        }
    }
}

/* Every constellation family, Es/N0 = min_snr + 1 dB, PN-A, LEO ramp. */
BOOST_AUTO_TEST_CASE(test_constellations_with_impairments)
{
    const char* names[] = { "qpsk1/4", "8psk3/5", "8apsk5/9", "16apsk2/3", "16apsk1/2l",
                            "32apsk3/4", "64apsk4/5", "128apsk3/4", "256apsk3/4" };
    for (const char* name : names) {
        const uint8_t code = code_of(name);
        const double esn0 = find_modcod_by_name(name)->min_snr_db + 1.0;
        const int plf = pl_layout_from_code(code).plframe_symbols();
        const uint32_t latency = 4 * 33282;
        stream s = make_ccm(code, scaled(1000000 / plf * 2 + 30), 7);
        pl_receiver rx(ccm_config(code, latency));
        pltest::impairments imp;
        imp.cfo = 0.031;
        imp.ramp = 1e-10;
        imp.pn_dnu_ts = 1e-6;
        imp.esn0_db = esn0;
        imp.seed = 3;
        run_stats st = run(s, rx, imp, latency, { 30000 }, esn0);
        BOOST_TEST_CONTEXT(name)
        {
            BOOST_CHECK_EQUAL(st.acquired, 1);
            BOOST_CHECK_EQUAL(st.lost, 0);
            BOOST_CHECK_EQUAL(st.unknown_sof + st.wrong_layout, 0);
            BOOST_CHECK_LE(st.bad, 0);
            BOOST_CHECK_EQUAL(st.delivered, st.tx_data_after_lock);
            const double rms_phase_deg = std::sqrt(st.phase_sq / std::max(1L, st.n_evm)) * 180 / M_PI;
            BOOST_TEST_MESSAGE(name << ": delivered " << st.delivered << " rms frame phase "
                                    << rms_phase_deg << " deg, gain "
                                    << 20 * std::log10(st.gain_sum / std::max(1L, st.n_evm)) << " dB, snr err "
                                    << st.snr_err_sum / std::max(1L, st.n_snr) << " dB");
            BOOST_CHECK_SMALL(20 * std::log10(st.gain_sum / std::max(1L, st.n_evm)), 0.25);
        }
    }
}

/* R1.13: short FECFRAMEs (32APSK short is the smallest PLFRAME, 3402). */
BOOST_AUTO_TEST_CASE(test_short_frames)
{
    for (const char* name : { "qpsk1/2", "32apsk3/4" }) {
        const uint8_t code = code_of(name, FECFRAME_SHORT);
        const double esn0 = find_modcod_by_name(name)->min_snr_db + 1.5;
        const uint32_t latency = 4 * 33282;
        stream s = make_ccm(code, scaled(10000), 9);
        pl_rx_config cfg = ccm_config(code, latency);
        cfg.acq_plframe_hint = pl_layout_from_code(code).plframe_symbols();
        pl_receiver rx(cfg);
        pltest::impairments imp;
        imp.cfo = -0.04;
        imp.pn_dnu_ts = 1e-6;
        imp.esn0_db = esn0;
        run_stats st = run(s, rx, imp, latency, { 8000 }, esn0);
        BOOST_TEST_CONTEXT(name)
        {
            BOOST_CHECK_EQUAL(st.acquired, 1);
            BOOST_CHECK_EQUAL(st.lost, 0);
            BOOST_CHECK_EQUAL(st.unknown_sof + st.wrong_layout, 0);
            BOOST_CHECK_EQUAL(st.delivered, st.tx_data_after_lock);
            BOOST_CHECK_LE(st.bad, st.delivered / 1000);
        }
    }
}

/* R1.10: identical frames for any partition of the same input (>= 4096).
 * The input must be the same for every partition, so the carrier commands
 * are not fed back (the residual offset stays within the tracking range). */
BOOST_AUTO_TEST_CASE(test_partition_invariance)
{
    const uint8_t code = code_of("16apsk2/3");
    const double esn0 = 12.0;
    stream s = make_ccm(code, 40, 21);
    const uint32_t latency = NO_FEEDBACK;
    pltest::impairments imp;
    imp.cfo = 2e-5;
    imp.esn0_db = esn0;
    std::vector<run_stats> res;
    for (const std::vector<int>& b : { std::vector<int>{ 4096 }, std::vector<int>{ 50000 },
                                       std::vector<int>{ 7001, 4096, 33000, 5003 } }) {
        pl_receiver rx(ccm_config(code, 0));
        res.push_back(run(s, rx, imp, latency, b, esn0, true));
    }
    // compare frames delivered by all runs
    const run_stats& a = res[0];
    for (size_t r = 1; r < res.size(); r++) {
        const run_stats& b = res[r];
        int compared = 0;
        for (size_t i = 0; i < a.frames_copy.size(); i++)
            for (size_t j = 0; j < b.frames_copy.size(); j++)
                if (a.frames_copy[i].sof_symbol == b.frames_copy[j].sof_symbol) {
                    BOOST_CHECK_EQUAL(a.frames_copy[i].layout.pls_code, b.frames_copy[j].layout.pls_code);
                    double e = 0;
                    for (size_t k = 0; k < a.data_copy[i].size(); k++)
                        e += std::norm(a.data_copy[i][k] - b.data_copy[j][k]);
                    BOOST_CHECK_LE(std::sqrt(e / a.data_copy[i].size()), 1e-4);
                    compared++;
                }
        BOOST_CHECK_GE(compared, 20);
        // once both are locked the sequences coincide
        const uint64_t from = std::max(a.first_sof, b.first_sof);
        long na = 0, nb = 0;
        for (const auto& f : a.frames_copy)
            na += f.sof_symbol >= from;
        for (const auto& f : b.frames_copy)
            nb += f.sof_symbol >= from;
        BOOST_CHECK_EQUAL(na, nb);
    }
}

/* R1.11: >= 1e4 consecutive dummy PLFRAMEs, constant and random payload. */
BOOST_AUTO_TEST_CASE(test_dummy_runs)
{
    for (bool random_payload : { false, true }) {
        const uint8_t code = code_of("qpsk3/5");
        stream s;
        pltest::framer fr(0, 4);
        for (int i = 0; i < 10; i++) {
            const uint64_t sof = fr.count();
            s.sent[sof] = fr.append(code, s.tx);
        }
        const int n_dummy = scaled(10000);
        for (int i = 0; i < n_dummy; i++)
            fr.append(0, s.tx, random_payload);
        for (int i = 0; i < 10; i++) {
            const uint64_t sof = fr.count();
            s.sent[sof] = fr.append(code, s.tx);
        }
        const uint32_t latency = 33282;
        pl_receiver rx(ccm_config(code, latency));
        pltest::impairments imp;
        imp.cfo = 0.02;
        imp.esn0_db = 4.0;
        run_stats st = run(s, rx, imp, latency, { 30000 }, 4.0);
        BOOST_TEST_CONTEXT("random payload " << random_payload)
        {
            BOOST_CHECK_EQUAL(st.acquired, 1);
            BOOST_CHECK_EQUAL(st.lost, 0);
            BOOST_CHECK_EQUAL((long)rx.dummy_count(), n_dummy);
            BOOST_CHECK_EQUAL(st.unknown_sof + st.wrong_layout, 0);
            BOOST_CHECK_EQUAL(st.delivered, st.tx_data_after_lock);
        }
    }
}

/* R1.12 (reduced): ACM changes every 1-20 frames among a MODCOD set with
 * signalling (MODCOD 1) and dummy frames interleaved. */
BOOST_AUTO_TEST_CASE(test_acm_switching)
{
    std::vector<uint8_t> codes;
    for (const char* n : { "qpsk1/4", "qpsk1/2", "8psk3/5", "16apsk2/3", "32apsk3/4", "qpsk13/45",
                           "16apsk28/45", "64apsk4/5", "8apsk5/9" })
        codes.push_back(code_of(n));
    pl_rx_config cfg;
    cfg.mode = pl_service_mode::vcm_acm;
    cfg.deliver_codes = codes;
    cfg.correction_latency_symbols = 2 * 33282;
    stream s;
    pltest::framer fr(0, 8);
    std::mt19937 rng(2);
    int n_total = scaled(10000) / 4;
    uint8_t cur = codes[0];
    int left = 0;
    for (int i = 0; i < n_total; i++) {
        if (left-- <= 0) {
            cur = codes[rng() % codes.size()];
            left = 1 + rng() % 20;
        }
        const int r = rng() % 10;
        if (r == 0) {
            fr.append(0, s.tx, true);
            continue;
        }
        const uint8_t c = r == 1 ? codes[0] : cur; // in-band signalling MODCOD 1
        const uint64_t sof = fr.count();
        s.sent[sof] = fr.append(c, s.tx);
    }
    pl_receiver rx(cfg);
    pltest::impairments imp;
    imp.cfo = -0.017;
    imp.esn0_db = 16.0; // above every member's min_snr
    run_stats st = run(s, rx, imp, cfg.correction_latency_symbols, { 30000 }, 16.0);
    BOOST_CHECK_EQUAL(st.acquired, 1);
    BOOST_CHECK_EQUAL(st.lost, 0);
    BOOST_CHECK_EQUAL(st.wrong_layout + st.unknown_sof, 0);
    BOOST_CHECK_EQUAL(st.delivered, st.tx_data_after_lock);
    BOOST_CHECK_EQUAL(st.bad, 0);
    BOOST_TEST_MESSAGE("acm: delivered " << st.delivered << " dummy " << rx.dummy_count());
}

/* R1.3 (reduced) / R1.14: noise, zeros, clipping, NaN, random PLS codes. */
BOOST_AUTO_TEST_CASE(test_robustness)
{
    const uint8_t code = code_of("qpsk1/2");
    pl_receiver rx(ccm_config(code, 0));
    std::vector<complexf> out(100000);
    std::vector<pl_frame> frames;
    std::vector<pl_sync_event> ev;
    std::mt19937 rng(1);
    std::normal_distribution<float> g(0.0f, 0.7f);
    long delivered = 0, locked_calls = 0, calls = 0;
    std::vector<complexf> buf(20000);
    for (int b = 0; b < scaled(300); b++) { // 6e6 noise symbols
        for (auto& v : buf)
            v = complexf(g(rng), g(rng));
        frames.clear();
        rx.process(buf.data(), (int)buf.size(), nullptr, out.data(), (int)out.size(), frames, ev);
        delivered += (long)frames.size();
        locked_calls += rx.locked();
        calls++;
    }
    BOOST_CHECK_EQUAL(delivered, 0);
    BOOST_CHECK_LE(locked_calls, calls / 100);
    // zeros, clipping, NaN/Inf, tiny batches
    std::fill(buf.begin(), buf.end(), complexf(0, 0));
    rx.process(buf.data(), (int)buf.size(), nullptr, out.data(), (int)out.size(), frames, ev);
    for (auto& v : buf)
        v = complexf((rng() & 1) ? 1e30f : -1e30f, (rng() & 1) ? 1e30f : -1e30f);
    rx.process(buf.data(), (int)buf.size(), nullptr, out.data(), (int)out.size(), frames, ev);
    for (size_t i = 0; i < buf.size(); i++)
        buf[i] = (i % 3) ? complexf(NAN, 1) : complexf(INFINITY, -INFINITY);
    rx.process(buf.data(), (int)buf.size(), nullptr, out.data(), (int)out.size(), frames, ev);
    rx.process(buf.data(), 0, nullptr, out.data(), (int)out.size(), frames, ev);
    rx.process(buf.data(), 1, nullptr, out.data(), (int)out.size(), frames, ev);
    // a stream of random PLS codes (incl. VL-SNR and reserved) in vcm mode
    pl_rx_config cfg;
    cfg.mode = pl_service_mode::vcm_acm;
    cfg.deliver_codes = { code };
    pl_receiver rx2(cfg);
    std::vector<complexf> tx;
    pltest::framer fr(0, 3);
    for (int i = 0; i < 200; i++) {
        const uint8_t c = rng() & 0xFF;
        const pl_layout l = pl_layout_from_code(c);
        if (l.plframe_symbols() > 0 && l.plframe_symbols() < 34000 && (l.deliverable() || l.dummy))
            fr.append(c, tx);
        else { // header + noise of a plausible length
            complexf h[90];
            plheader_symbols(c, h);
            tx.insert(tx.end(), h, h + 90);
            for (int k = 0; k < 5000; k++)
                tx.push_back(complexf(g(rng), g(rng)));
        }
    }
    for (size_t off = 0; off < tx.size(); off += 25000) {
        const int n = (int)std::min<size_t>(25000, tx.size() - off);
        frames.clear();
        rx2.process(&tx[off], n, nullptr, out.data(), (int)out.size(), frames, ev);
        for (const auto& f : frames)
            for (int k = 0; k < f.n_data; k++)
                BOOST_REQUIRE(std::isfinite(f.data[k].real()) && std::isfinite(f.data[k].imag()));
    }
    BOOST_CHECK(std::isfinite(rx2.snr_db()));
    BOOST_CHECK(std::isfinite(rx2.coherence()));
}

/* §1.4.5 / §1.8.4: events, reset scopes, metric conventions, output space. */
BOOST_AUTO_TEST_CASE(test_events_resets_metrics)
{
    const uint8_t code = code_of("8psk3/5");
    const double esn0 = 3.0;
    stream s = make_ccm(code, 30, 13);
    pl_receiver rx(ccm_config(code, 0));
    pltest::impairments imp;
    imp.esn0_db = esn0;
    imp.cfo = 0.01;
    pltest::channel ch(imp);
    std::vector<complexf> buf, out(30000);
    std::vector<pl_frame> frames;
    std::vector<pl_sync_event> ev;
    int acquired = 0, headers = 0;
    size_t off = 0;
    std::vector<float> coh;
    while (off < s.tx.size() / 2) {
        const int n = 30000;
        buf.resize(n);
        ch.apply(&s.tx[off], buf.data(), n);
        off += n;
        frames.clear();
        ev.clear();
        // max_out smaller than two frames: frames stay pending, never lost
        rx.process(buf.data(), n, nullptr, out.data(), (int)out.size(), frames, ev);
        for (auto& e : ev) {
            acquired += e.type == pl_sync_event_type::acquired;
            headers += e.type == pl_sync_event_type::header;
        }
        for (auto& f : frames)
            coh.push_back(f.coherence);
        double f;
        if (rx.take_carrier_update(f)) {
            ch.set_rotator(f, 0);
            BOOST_CHECK(!rx.take_carrier_update(f)); // once per change
        }
    }
    BOOST_CHECK_EQUAL(acquired, 1);
    BOOST_CHECK_GE(headers, 3);
    BOOST_CHECK(rx.locked());
    BOOST_CHECK(!rx.wants_header_search());
    BOOST_CHECK_EQUAL(rx.expected_plframe_len(), (uint32_t)pl_layout_from_code(code).plframe_symbols());
    // coherence convention: C ~ sqrt((36 s + 1) / (36 (s + 1)))
    const double sl = std::pow(10.0, esn0 / 10.0);
    const double c_expect = std::sqrt((36 * sl + 1) / (36 * (sl + 1)));
    double c_mean = 0;
    for (float c : coh)
        c_mean += c;
    c_mean /= std::max<size_t>(1, coh.size());
    BOOST_CHECK_SMALL(c_mean - c_expect, 0.03);
    BOOST_CHECK(rx.snr_valid());
    BOOST_CHECK_GT(rx.frame_count(), 0u);
    // positional reset while locked -> lost on the next call
    rx.reset(pl_reset_scope::positional);
    BOOST_CHECK(!rx.locked());
    BOOST_CHECK(rx.wants_header_search());
    ev.clear();
    frames.clear();
    rx.process(buf.data(), 100, nullptr, out.data(), (int)out.size(), frames, ev);
    bool lost = false;
    for (auto& e : ev)
        lost |= e.type == pl_sync_event_type::lost;
    BOOST_CHECK(lost);
    BOOST_CHECK(rx.carrier_frequency() != 0.0); // kept
    BOOST_CHECK(rx.snr_valid());
    rx.reset(pl_reset_scope::sync);
    BOOST_CHECK(rx.snr_valid());
    rx.reset(pl_reset_scope::full);
    BOOST_CHECK(!rx.snr_valid());
    BOOST_CHECK_EQUAL(rx.carrier_frequency(), 0.0);
    double f = 1.0;
    BOOST_CHECK(rx.take_carrier_update(f));
    BOOST_CHECK_EQUAL(f, 0.0);
}

} // namespace cusat_dsp
