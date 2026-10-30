/* dvbs2x-vv-check.cc
 * SPDX-License-Identifier: Apache-2.0
 *
 * Conformance check of the DVB-S2X TX chain against the DVB.org
 * Verification & Validation test packages (TM5859, regular-frame cases).
 *
 * For every frame, the V&V BBFRAME is run through our complete chain and
 * compared at each tap the package provides:
 *
 *   LDPC       BCH + LDPC encoder output              *_LDPC.txt (optional)
 *   FECFRAME   after the bit interleaver               *_FECFRAME.txt
 *   XFECFRAME  after constellation mapping             *_XFECFRAME.txt
 *   CLEARPLFR  PLFRAME before PL scrambling            *_CLEARPLFR.txt
 *   PLFRAME    transmitted symbols (the V&V reference) *_PLFRAME.txt
 *
 * Every stage consumes our own output of the previous stage, never the
 * reference, so a PASS on PLFRAME covers the whole chain.
 *
 * Frames outside the cuSAT feature set are reported as SKIP: VL-SNR and
 * DSF frames (no standard FECFRAME size) entirely, S2X short-FECFRAME
 * MODCODs after the LDPC stage. For frames with the Annex M time-slicing
 * PLHEADER (EN 302 307-1 Annex M, 180 symbols) only the PLFRAME payload
 * is compared.
 *
 * FECFRAME.txt is the interleaver output in every package (V&V tool
 * description, "V&V structure"); for QPSK the two coincide.
 *
 * Usage:
 *   dvbs2x-vv-check <test-case-dir> [--gold N] [--fec-only] [--gpu-bch]
 *
 * The PL scrambling code is read from Scrambling_gold_index in the test
 * case configuration file; --gold overrides it (some packages ship
 * without that file).
 */

#include <cusat_dsp/modcod_types.h>
#include <cusat_dsp/compat.h>
#include "framing/plframe.h"
#include "framing/pl_scrambler.h"
#include "acm_config.h"
#include "fec/code_params.h"
#include "fec/bch_code.h"
#include "cuda/bch_encode_kernel.cuh"
#include "cuda/bch_check_kernel.cuh"
#include "cuda/cuda_util.cuh"
#include "ldpc_encode.h"
#include "mapper.h"
#include "pl_framer.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cusat_dsp;

/* V&V files hold 4 decimal digits. */
static constexpr float SYMBOL_TOL = 1e-4f;

/* 128APSK: 6 stuffing bits before interleaving, 12 padding symbols after
 * (EN 302 307-2 clause 5.3.3). */
static constexpr uint32_t APSK128_STUFFED_BITS = 64806;
static constexpr uint32_t APSK128_PADDING_SYMS = 12;

/* ------------------------------------------------------------------ */
/*  Test case files                                                   */
/* ------------------------------------------------------------------ */

struct vv_frame_t {
    unsigned pls;
    uint32_t kbch;
    uint32_t fecframe_bits; /* 64806 for 128APSK (stuffing included) */
    uint32_t mapperout;     /* 128APSK: padding symbols excluded */
    uint32_t plframe_len;
};

struct vv_case_t {
    std::vector<vv_frame_t> frames;
    int gold_code = -1;
    std::map<std::string, fs::path> files; /* "BBFRAME" -> path, ... */
};

static const char* const TAPS[] = {
    "readme", "BBFRAME", "LDPC", "FECFRAME", "XFECFRAME", "CLEARPLFR", "PLFRAME",
};

static std::vector<vv_frame_t> parse_readme(const fs::path& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path.string());

    std::vector<vv_frame_t> frames;
    std::string line;
    bool in_table = false;
    while (std::getline(in, line)) {
        if (line.find("PLS") != std::string::npos && line.find("Kbch") != std::string::npos) {
            in_table = true;
            continue;
        }
        if (!in_table)
            continue;
        std::istringstream iss(line);
        vv_frame_t f{};
        if (iss >> f.pls >> f.kbch >> f.fecframe_bits >> f.mapperout >> f.plframe_len)
            frames.push_back(f);
    }
    return frames;
}

/* Integer value of `key = N` in the configuration file, or -1. */
static long parse_config_value(const fs::path& path, const std::string& key)
{
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find(key);
        if (pos == std::string::npos)
            continue;
        const auto eq = line.find('=', pos);
        if (eq == std::string::npos)
            continue;
        std::istringstream iss(line.substr(eq + 1));
        long val;
        if (iss >> val)
            return val;
    }
    return -1;
}

/* Values of `key = [a, b, ...]` in the configuration file. */
static std::vector<long> parse_config_list(const fs::path& path, const std::string& key)
{
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find(key);
        const auto open = line.find('[', pos);
        const auto close = line.find(']', open);
        if (pos == std::string::npos || open == std::string::npos || close == std::string::npos)
            continue;
        std::string body = line.substr(open + 1, close - open - 1);
        for (char& c : body)
            if (c == ',')
                c = ' ';
        std::istringstream iss(body);
        std::vector<long> vals;
        long v;
        while (iss >> v)
            vals.push_back(v);
        return vals;
    }
    return {};
}

static bool ends_with(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static vv_case_t load_case(const fs::path& dir)
{
    vv_case_t tc;
    fs::path config;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (entry.path().extension() != ".txt")
            continue;
        bool is_tap = false;
        for (const char* tap : TAPS) {
            if (ends_with(name, std::string("_") + tap + ".txt")) {
                tc.files[tap] = entry.path();
                is_tap = true;
            }
        }
        if (!is_tap && !ends_with(name, "_DATAFIELD.txt") && !ends_with(name, "_ITLVOUT.txt") &&
            name.find("input") == std::string::npos)
            config = entry.path();
    }
    if (!tc.files.count("readme") || !tc.files.count("BBFRAME"))
        throw std::runtime_error("need *_readme.txt and *_BBFRAME.txt in " + dir.string());

    tc.frames = parse_readme(tc.files["readme"]);
    if (config.empty())
        return tc;
    tc.gold_code = static_cast<int>(parse_config_value(config, "Scrambling_gold_index"));

    /* Some packages ship the readme of a sibling case; the configuration
     * file is authoritative for the PLS and PLFRAME length of each frame. */
    const std::vector<long> pls = parse_config_list(config, "PLS_list");
    const std::vector<long> len = parse_config_list(config, "PLF_length");
    if (pls.size() == tc.frames.size() && len.size() == tc.frames.size()) {
        for (size_t i = 0; i < pls.size(); i++) {
            vv_frame_t& f = tc.frames[i];
            if (f.pls != unsigned(pls[i]) || f.plframe_len != uint32_t(len[i])) {
                std::cout << "Note: frame " << i << " readme says PLS " << f.pls << "/"
                          << f.plframe_len << " symbols, configuration says " << pls[i] << "/"
                          << len[i] << "; using the configuration\n";
                f.pls = pls[i];
                f.plframe_len = len[i];
            }
        }
    }
    return tc;
}

static std::vector<uint8_t> read_bits(const fs::path& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path.string());
    std::vector<uint8_t> bits;
    int val;
    while (in >> val)
        bits.push_back(static_cast<uint8_t>(val));
    return bits;
}

static std::vector<complexf> read_symbols(const fs::path& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path.string());
    std::vector<complexf> syms;
    float re, im;
    while (in >> re >> im)
        syms.emplace_back(re, im);
    return syms;
}

/* ------------------------------------------------------------------ */
/*  Frame parameters                                                  */
/* ------------------------------------------------------------------ */

struct frame_params_t {
    bool valid = false;
    std::string label;          /* MODCOD name, or the reason for a skip */
    fecframe_size_t framesize;
    code_rate_t rate;
    modulation_t modulation;
    pl_layout format;
    bool framer_supported = false; /* PLS is a cuSAT pl_framer MODCOD */
};

static modulation_t modulation_for(int bits_per_sym, code_rate_t rate)
{
    switch (bits_per_sym) {
    case 2: return MOD_QPSK;
    case 3: return (rate == RATE_100_180 || rate == RATE_104_180) ? MOD_8APSK : MOD_8PSK;
    case 4: return MOD_16APSK;
    case 5: return MOD_32APSK;
    case 6: return MOD_64APSK;
    case 7: return MOD_128APSK;
    case 8: return MOD_256APSK;
    default: throw std::runtime_error("unsupported bits/symbol " + std::to_string(bits_per_sym));
    }
}

/* Several S2X rates share a Kbch with S2 rates, so the PLS is used first
 * and Kbch only for PLS values without a cuSAT MODCOD (S2X short frames). */
static code_rate_t rate_for_kbch(fecframe_size_t framesize, uint32_t kbch)
{
    for (int r = 0; r < static_cast<int>(RATE_UNKNOWN); r++) {
        const auto rate = static_cast<code_rate_t>(r);
        if (has_fec_params(framesize, rate) && get_fec_params(framesize, rate).k_bch == kbch)
            return rate;
    }
    throw std::runtime_error("no code rate with Kbch " + std::to_string(kbch));
}

static frame_params_t resolve_params(const vv_frame_t& f)
{
    frame_params_t p;
    const uint32_t n_ldpc = (f.fecframe_bits == APSK128_STUFFED_BITS) ? 64800 : f.fecframe_bits;
    if (n_ldpc == 64800)
        p.framesize = FECFRAME_NORMAL;
    else if (n_ldpc == 16200)
        p.framesize = FECFRAME_SHORT;
    else {
        p.label = "VL-SNR/DSF, FECFRAME " + std::to_string(f.fecframe_bits);
        return p;
    }

    p.format = pl_layout_from_code(static_cast<uint8_t>(f.pls));
    const modcod_def_t* mc = nullptr;
    if (!p.format.dummy && p.format.modcod != 0)
        mc = find_modcod(p.format.modcod);

    try {
        p.rate = mc ? mc->code_rate : rate_for_kbch(p.framesize, f.kbch);
        const int bits_per_sym = (f.fecframe_bits == APSK128_STUFFED_BITS)
                                     ? 7 : static_cast<int>(f.fecframe_bits / f.mapperout);
        p.modulation = modulation_for(bits_per_sym, p.rate);
    } catch (const std::exception& e) {
        p.label = e.what();
        return p;
    }

    p.valid = true;
    p.framer_supported = mc && (p.format.fec_size == FECFRAME_SHORT) == (p.framesize == FECFRAME_SHORT);
    p.label = mc ? mc->name : "pls" + std::to_string(f.pls);
    if (p.framesize == FECFRAME_SHORT)
        p.label += "-short";
    return p;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

static std::vector<uint8_t> pack_bits(const uint8_t* bits, size_t n_bits)
{
    std::vector<uint8_t> packed((n_bits + 7) / 8, 0);
    for (size_t i = 0; i < n_bits; i++)
        packed[i / 8] |= (bits[i] & 1) << (7 - (i & 7));
    return packed;
}

static std::vector<uint8_t> unpack_bits(const uint8_t* packed, size_t n_bits)
{
    std::vector<uint8_t> bits(n_bits);
    for (size_t i = 0; i < n_bits; i++)
        bits[i] = (packed[i / 8] >> (7 - (i & 7))) & 1;
    return bits;
}

/* Interleaver output as a bit stream, from the permutation the mapper
 * programs into the GPU: bit c of symbol s is input bit
 * col_offset[c] + s * stride. */
static std::vector<uint8_t> interleaved_bits(const mapper& m, const std::vector<uint8_t>& coded,
                                             size_t n_syms)
{
    unsigned col_offset[8];
    int stride;
    m.get_interleave_params(col_offset, stride);
    const int n_mod = m.n_mod();
    std::vector<uint8_t> out(n_syms * n_mod);
    for (size_t s = 0; s < n_syms; s++)
        for (int c = 0; c < n_mod; c++)
            out[s * n_mod + c] = coded[col_offset[c] + s * stride];
    return out;
}

static std::string bit_mismatch(const uint8_t* ref, const uint8_t* got, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (ref[i] != got[i])
            return "bit " + std::to_string(i) + " ref=" + std::to_string(ref[i]) +
                   " got=" + std::to_string(got[i]);
    return {};
}

static std::string symbol_mismatch(const complexf* ref, const complexf* got, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (std::abs(ref[i].real() - got[i].real()) > SYMBOL_TOL ||
            std::abs(ref[i].imag() - got[i].imag()) > SYMBOL_TOL) {
            std::ostringstream os;
            os << "symbol " << i << " ref=" << ref[i] << " got=" << got[i];
            return os.str();
        }
    }
    return {};
}

/* ------------------------------------------------------------------ */
/*  Per-stage bookkeeping                                             */
/* ------------------------------------------------------------------ */

enum stage_t { ST_LDPC, ST_FECFRAME, ST_XFECFRAME, ST_CLEARPLFR, ST_PLFRAME, N_STAGES };
static const char* const STAGE_NAMES[N_STAGES] = {
    "LDPC", "FECFRAME", "XFECFRAME", "CLEARPLFR", "PLFRAME",
};

struct stage_tally_t {
    int pass = 0, fail = 0, skip = 0;
};

static void report(stage_tally_t* tally, stage_t st, const std::string& mismatch,
                   const std::string& note = "")
{
    std::cout << "  " << STAGE_NAMES[st] << std::string(11 - strlen(STAGE_NAMES[st]), ' ');
    if (mismatch.empty()) {
        std::cout << "PASS" << note << "\n";
        tally[st].pass++;
    } else {
        std::cout << "FAIL  first mismatch at " << mismatch << note << "\n";
        tally[st].fail++;
    }
}

static int total_failures(const stage_tally_t* tally)
{
    int n = 0;
    for (int st = 0; st < N_STAGES; st++)
        n += tally[st].fail;
    return n;
}

static void report_skip(stage_tally_t* tally, stage_t st, const std::string& why)
{
    std::cout << "  " << STAGE_NAMES[st] << std::string(11 - strlen(STAGE_NAMES[st]), ' ')
              << "SKIP  " << why << "\n";
    tally[st].skip++;
}

/* ------------------------------------------------------------------ */
/*  One frame through the chain                                       */
/* ------------------------------------------------------------------ */

struct reference_t {
    std::vector<uint8_t> bb, ldpc, fec;
    std::vector<complexf> xfec, clear, pl;
};

/* Position of the current frame in each tap file. */
struct offsets_t {
    size_t bb = 0, ldpc = 0, fec = 0, xfec = 0, pl = 0;
};

/* The V&V BBFRAME tap is already scrambled, so use an all-zero XOR sequence. */
static std::vector<uint8_t> encode_bch_gpu(const bch_code_t& code,
                                         const std::vector<uint8_t>& bb)
{
    struct device_buffers {
        uint8_t* codeword = nullptr;
        int* valid = nullptr;
        ~device_buffers() { cudaFree(codeword); cudaFree(valid); }
    } buffers;
    const unsigned k = code.k_bits / 8, n = code.n_bits / 8;
    const auto* table = code.encoder_lut.data()->data();
    std::vector<uint8_t> zeros(k), output(n);
    cuda::bch_bb_scramble_cuda encoder(table, code.generator.data(), code.parity_bits,
                                      zeros.data(), k, k, n, n - k);
    cuda::bch_check_cuda checker(table, code.generator.data(), code.parity_bits, n);
    CUDA_CHECK(cudaMalloc(&buffers.codeword, n));
    CUDA_CHECK(cudaMalloc(&buffers.valid, sizeof(int)));
    encoder.encode_batch(bb.data(), buffers.codeword, 1, nullptr);
    checker.check_async(buffers.codeword, buffers.valid, 1, nullptr);
    int valid = 0;
    CUDA_CHECK(cudaMemcpy(output.data(), buffers.codeword, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&valid, buffers.valid, sizeof(int), cudaMemcpyDeviceToHost));
    if (valid != 1)
        throw std::runtime_error("GPU BCH checker rejected the encoded V&V BBFRAME");
    return output;
}

static void check_frame(const vv_frame_t& f, const frame_params_t& p, const reference_t& ref,
                        const offsets_t& off, int gold_code, bool fec_only,
                        stage_tally_t* tally, bool gpu_bch)
{
    const bool apsk128 = (f.fecframe_bits == APSK128_STUFFED_BITS);
    const uint32_t n_ldpc = apsk128 ? 64800 : f.fecframe_bits;

    /* BBFRAME -> BCH -> LDPC */
    const fec_params_t fec = get_fec_params(p.framesize, p.rate);
    const std::vector<uint8_t> bb = pack_bits(&ref.bb[off.bb], f.kbch);
    const bch_code_t bch = make_bch_code(p.framesize, fec.n_bch, fec.t_bch);
    std::vector<uint8_t> bch_out(fec.n_bch / 8);
    if (gpu_bch)
        bch_out = encode_bch_gpu(bch, bb);
    else
        bch_encode(bch, bb.data(), bch_out.data());
    ldpc_encode ldpc(p.framesize, p.rate);
    std::vector<uint8_t> coded_packed(n_ldpc / 8);
    ldpc.encode(bch_out.data(), coded_packed.data());
    std::vector<uint8_t> coded = unpack_bits(coded_packed.data(), n_ldpc);

    if (!ref.ldpc.empty())
        report(tally, ST_LDPC, bit_mismatch(&ref.ldpc[off.ldpc], coded.data(), n_ldpc));
    else
        report_skip(tally, ST_LDPC, "no LDPC tap in this package");

    /* The interleaver and constellation tables cover the cuSAT MODCODs only. */
    if (!p.framer_supported) {
        for (stage_t st : { ST_FECFRAME, ST_XFECFRAME, ST_CLEARPLFR, ST_PLFRAME })
            report_skip(tally, st, "PLS not a cuSAT MODCOD");
        return;
    }

    /* Bit interleaver (128APSK: stuffing bits are zero) */
    mapper m(p.framesize, p.rate, p.modulation);
    coded.resize(f.fecframe_bits, 0);
    const std::vector<uint8_t> itlv = interleaved_bits(m, coded, f.fecframe_bits / m.n_mod());
    report(tally, ST_FECFRAME, bit_mismatch(&ref.fec[off.fec], itlv.data(), f.fecframe_bits));
    if (fec_only)
        return;

    /* Constellation mapping */
    std::vector<complexf> xfec(m.xfecframe_len());
    m.interleave_and_map(coded_packed.data(), xfec.data());
    const uint32_t xfec_in_file = f.mapperout + (apsk128 ? APSK128_PADDING_SYMS : 0);
    report(tally, ST_XFECFRAME, symbol_mismatch(&ref.xfec[off.xfec], xfec.data(), xfec_in_file));

    /* PL framing. Annex M (time slicing) frames carry a 180-symbol
     * PLHEADER that cuSAT does not generate; their payload, pilots and
     * scrambling are the standard ones and are still compared. */
    const uint32_t our_len = (uint32_t)p.format.plframe_symbols();
    const bool annex_m = (f.plframe_len == our_len + uint32_t(pl::kSlotLen));
    if (!annex_m && f.plframe_len != our_len) {
        const std::string why = "PLFRAME length " + std::to_string(f.plframe_len);
        report_skip(tally, ST_CLEARPLFR, why);
        report_skip(tally, ST_PLFRAME, why);
        return;
    }
    const size_t first = annex_m ? pl::kHeaderLen : 0; /* first compared symbol, ours */
    const size_t ref_first = off.pl + (annex_m ? pl::kHeaderLen + pl::kSlotLen : 0);
    const std::string note = annex_m ? " (payload only, Annex M PLHEADER)" : "";

    const int gold = gold_code >= 0 ? gold_code : 0;
    pl_framer framer(p.format.modcod, p.format.fec_size == FECFRAME_SHORT, p.format.pilots, gold);
    std::vector<complexf> plframe(framer.plframe_len());
    framer.frame(xfec.data(), plframe.data());

    std::vector<complexf> clear = plframe;
    std::vector<complexf> seq(framer.payload_len());
    pl_descrambling_sequence(gold, seq.data(), framer.payload_len());
    pl_descramble(clear.data() + pl::kHeaderLen, seq.data(), framer.payload_len());
    const size_t n = plframe.size() - first;
    report(tally, ST_CLEARPLFR, symbol_mismatch(&ref.clear[ref_first], clear.data() + first, n),
           note);

    if (gold_code >= 0)
        report(tally, ST_PLFRAME, symbol_mismatch(&ref.pl[ref_first], plframe.data() + first, n),
               note);
    else
        report_skip(tally, ST_PLFRAME, "gold code unknown, use --gold");
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: dvbs2x-vv-check <test-case-dir> [--gold N] [--fec-only] [--gpu-bch]\n";
        return 1;
    }
    const fs::path dir(argv[1]);
    bool fec_only = false;
    bool gpu_bch = false;
    int gold_override = -1;
    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--fec-only")
            fec_only = true;
        else if (arg == "--gpu-bch")
            gpu_bch = true;
        else if (arg == "--gold" && i + 1 < argc)
            gold_override = std::atoi(argv[++i]);
    }

    vv_case_t tc;
    try {
        tc = load_case(dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    if (gold_override >= 0)
        tc.gold_code = gold_override;

    auto tap_bits = [&](const char* tap) {
        return tc.files.count(tap) ? read_bits(tc.files[tap]) : std::vector<uint8_t>{};
    };
    auto tap_symbols = [&](const char* tap) {
        return (tc.files.count(tap) && !fec_only) ? read_symbols(tc.files[tap])
                                                  : std::vector<complexf>{};
    };
    reference_t ref;
    ref.bb = tap_bits("BBFRAME");
    ref.ldpc = tap_bits("LDPC");
    ref.fec = tap_bits("FECFRAME");
    ref.xfec = tap_symbols("XFECFRAME");
    ref.clear = tap_symbols("CLEARPLFR");
    ref.pl = tap_symbols("PLFRAME");

    std::cout << "Test case : " << dir.filename().string() << "\n"
              << "Frames    : " << tc.frames.size() << "\n"
              << "Gold code : " << (tc.gold_code >= 0 ? std::to_string(tc.gold_code) : "unknown")
              << "\n\n";

    stage_tally_t tally[N_STAGES];
    offsets_t off;
    int n_frames_failed = 0, n_frames_skipped = 0;

    for (size_t fi = 0; fi < tc.frames.size(); fi++) {
        const vv_frame_t& f = tc.frames[fi];
        const frame_params_t p = resolve_params(f);
        const bool apsk128 = (f.fecframe_bits == APSK128_STUFFED_BITS);
        const int failed_before = total_failures(tally);

        std::cout << "Frame " << fi << "  [PLS=" << f.pls << "  " << p.label << "]\n";

        if (!p.valid) {
            std::cout << "  SKIP  " << p.label << "\n";
            n_frames_skipped++;
        } else {
            try {
                check_frame(f, p, ref, off, tc.gold_code, fec_only, tally, gpu_bch);
            } catch (const std::exception& e) {
                std::cout << "  ERROR " << e.what() << "\n";
                tally[ST_LDPC].fail++;
            }
        }

        const int failed_after = total_failures(tally);
        if (failed_after > failed_before)
            n_frames_failed++;

        /* The LDPC tap has no 128APSK stuffing bits. */
        off.bb += f.kbch;
        off.ldpc += apsk128 ? 64800 : f.fecframe_bits;
        off.fec += f.fecframe_bits;
        off.xfec += f.mapperout + (apsk128 ? APSK128_PADDING_SYMS : 0);
        off.pl += f.plframe_len;
    }

    std::cout << "\n=== Summary: " << dir.filename().string() << " ===\n";
    for (int st = 0; st < N_STAGES; st++)
        std::cout << "  " << STAGE_NAMES[st] << std::string(11 - strlen(STAGE_NAMES[st]), ' ')
                  << "PASS " << tally[st].pass << "  FAIL " << tally[st].fail
                  << "  SKIP " << tally[st].skip << "\n";
    std::cout << "  frames     " << tc.frames.size() - n_frames_failed - n_frames_skipped
              << " ok, " << n_frames_failed << " failed, " << n_frames_skipped
              << " out of scope\n";

    return n_frames_failed > 0 ? 1 : 0;
}
