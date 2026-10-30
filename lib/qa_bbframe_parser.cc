/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BBHEADER parser / GSE data-field extraction (EN 302 307-1
 * clause 5.1.6 and table 3).
 */

#include "cusat_dsp/bb_header.h"
#include "cusat_dsp/bbframe_parser.h"
#include "cusat_dsp/fec/code_params.h"
#include "cusat_dsp/fec/crc8.h"
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include <random>
#include <vector>

using namespace cusat_dsp;

namespace {

struct hdr_in {
    unsigned ts_gs = 0, sis = 1, ccm = 1, issyi = 0, npd = 0, ro = 2;
    unsigned matype2 = 0, upl = 0, dfl = 0, sync = 0, syncd = 0;
};

/* BBHEADER written field by field per table 3, CRC-8 over the first 9 bytes */
std::vector<uint8_t> make_frame(const hdr_in& h, size_t kbch_bytes, std::mt19937& rng)
{
    std::vector<uint8_t> f(kbch_bytes);
    for (auto& x : f)
        x = (uint8_t)rng();
    f[0] = (uint8_t)((h.ts_gs << 6) | (h.sis << 5) | (h.ccm << 4) | (h.issyi << 3) | (h.npd << 2) | h.ro);
    f[1] = (uint8_t)h.matype2;
    f[2] = (uint8_t)(h.upl >> 8);
    f[3] = (uint8_t)h.upl;
    f[4] = (uint8_t)(h.dfl >> 8);
    f[5] = (uint8_t)h.dfl;
    f[6] = (uint8_t)h.sync;
    f[7] = (uint8_t)(h.syncd >> 8);
    f[8] = (uint8_t)h.syncd;
    f[9] = crc8_bbheader(f.data(), 9);
    return f;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_fields_and_extraction)
{
    std::mt19937 rng(1);
    const size_t kbch = get_fec_params(FECFRAME_NORMAL, RATE_1_2).k_bch / 8;
    hdr_in h;
    h.ts_gs = TS_GS_GENERIC_PACKETIZED;
    h.ccm = 0;
    h.npd = 1;
    h.ro = 1;
    h.matype2 = 0x5a;
    h.upl = 0x1234;
    h.dfl = 8 * 1000;
    h.sync = 0x47;
    h.syncd = 0xabcd;
    auto f = make_frame(h, kbch, rng);
    bbheader_fields p;
    BOOST_REQUIRE(decode_bbheader(f.data(), kbch, p));
    BOOST_CHECK_EQUAL(p.ts_gs, 0);
    BOOST_CHECK(p.sis);
    BOOST_CHECK(!p.ccm);
    BOOST_CHECK(!p.issyi);
    BOOST_CHECK(p.npd);
    BOOST_CHECK_EQUAL(p.ro, 1);
    BOOST_CHECK_EQUAL(p.matype2, 0x5a);
    BOOST_CHECK_EQUAL(p.upl, 0x1234);
    BOOST_CHECK_EQUAL(p.dfl, 8000);
    BOOST_CHECK_EQUAL(p.sync, 0x47);
    BOOST_CHECK_EQUAL(p.syncd, 0xabcd);
    BOOST_CHECK_EQUAL(p.crc8, f[9]);
    std::vector<uint8_t> out(2000);
    BOOST_REQUIRE_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), 1000);
    BOOST_CHECK(std::equal(out.begin(), out.begin() + 1000, f.begin() + 10));
    // max_out too small
    BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), 999), 0);
    BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), 1000), 1000);
    // largest legal DFL and an empty data field
    h.dfl = (unsigned)(8 * kbch - 80);
    f = make_frame(h, kbch, rng);
    out.resize(kbch);
    BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), (int)kbch - 10);
    h.dfl = 0;
    f = make_frame(h, kbch, rng);
    BOOST_CHECK(decode_bbheader(f.data(), kbch, p));
    BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), 0);
}

BOOST_AUTO_TEST_CASE(test_rejections)
{
    std::mt19937 rng(2);
    const size_t kbch = get_fec_params(FECFRAME_SHORT, RATE_1_2).k_bch / 8;
    std::vector<uint8_t> out(kbch);
    bbheader_fields p;
    hdr_in h;
    h.dfl = 8 * 100;
    // CRC error: any single flipped bit of the header
    for (int bit = 0; bit < 80; bit++) {
        auto f = make_frame(h, kbch, rng);
        f[bit / 8] ^= (uint8_t)(0x80 >> (bit % 8));
        BOOST_CHECK(!decode_bbheader(f.data(), kbch, p));
        BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), 0);
    }
    // oversize DFL
    h.dfl = (unsigned)(8 * kbch - 80 + 8);
    auto f = make_frame(h, kbch, rng);
    BOOST_CHECK(!decode_bbheader(f.data(), kbch, p));
    BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), 0);
    // DFL not a byte multiple
    h.dfl = 8 * 100 + 3;
    f = make_frame(h, kbch, rng);
    BOOST_CHECK(!decode_bbheader(f.data(), kbch, p));
    BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), 0);
    // valid header, wrong TS/GS (transport, continuous, reserved)
    h.dfl = 8 * 100;
    for (unsigned ts : { 1u, 2u, 3u }) {
        h.ts_gs = ts;
        f = make_frame(h, kbch, rng);
        BOOST_CHECK(decode_bbheader(f.data(), kbch, p));
        BOOST_CHECK_EQUAL(extract_gse_datafield(f.data(), kbch, out.data(), (int)out.size()), 0);
    }
    // frame shorter than a header
    BOOST_CHECK(!decode_bbheader(f.data(), 9, p));
}

BOOST_AUTO_TEST_CASE(test_roundtrip_with_transmitter_header)
{
    std::mt19937 rng(3);
    for (auto fs : { FECFRAME_NORMAL, FECFRAME_SHORT })
        for (auto rate : { RATE_1_4, RATE_3_5, RATE_9_10 }) {
            if (fs == FECFRAME_SHORT && rate == RATE_9_10)
                continue; // no short 9/10 code
            bb_header tx(fs, rate);
            const size_t kbch = tx.kbch_bytes();
            for (unsigned len : { 0u, 1u, 37u, tx.max_dfl_bytes() / 2, tx.max_dfl_bytes() }) {
                std::vector<uint8_t> gse(len), frame(kbch), out(kbch);
                for (auto& x : gse)
                    x = (uint8_t)rng();
                const unsigned used = tx.build_gse_bbframe(gse.data(), len, frame.data());
                BOOST_REQUIRE_EQUAL(used, len);
                bbheader_fields p;
                BOOST_REQUIRE(decode_bbheader(frame.data(), kbch, p));
                BOOST_CHECK_EQUAL(p.ts_gs, (unsigned)TS_GS_GENERIC_PACKETIZED);
                BOOST_CHECK_EQUAL(p.dfl, 8 * len);
                const int n = extract_gse_datafield(frame.data(), kbch, out.data(), (int)out.size());
                BOOST_REQUIRE_EQUAL(n, (int)len);
                BOOST_CHECK(std::equal(gse.begin(), gse.end(), out.begin()));
            }
        }
}
