/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * PL layout: lengths of all 256 PLS codes against
 * EN 302 307-1 Tables 11/12 and Figure 13, EN 302 307-2 Tables 16/17a/17b.
 */

#include "cusat_dsp/acm_config.h"
#include "cusat_dsp/framing/plframe.h"
#include <boost/test/unit_test.hpp>
#include <map>

namespace cusat_dsp {

/* 90 (S + 1) + 36 int{(S - 1) / 16} with pilots (Figure 13) */
static int plframe_len(int slots, bool pilots)
{
    return 90 * (slots + 1) + (pilots ? 36 * ((slots - 1) / 16) : 0);
}

BOOST_AUTO_TEST_CASE(test_constants)
{
    BOOST_CHECK_EQUAL(pl::kHeaderLen, pl::kSofLen + pl::kPlsCodeLen);
    BOOST_CHECK_EQUAL(pl::kSlotLen, 90);
    BOOST_CHECK_EQUAL(pl::kPilotBlockLen, 36);
    BOOST_CHECK_EQUAL(pl::kDummySlots, 36);
}

BOOST_AUTO_TEST_CASE(test_s2_codes)
{
    // EN 302 307-1 Table 11: S per constellation and frame size
    const std::map<int, std::pair<int, int>> slots = {
        { 2, { 360, 90 } }, { 3, { 240, 60 } }, { 4, { 180, 45 } }, { 5, { 144, 36 } }
    };
    for (int code = 0; code < 128; code++) {
        const pl_layout l = pl_layout_from_code(code);
        const int modcod = code >> 2;
        const bool shortf = code & 2, pilots = code & 1;
        BOOST_TEST_CONTEXT("code " << code)
        {
            BOOST_CHECK_EQUAL((int)l.pls_code, code);
            BOOST_CHECK(!l.reserved);
            if (modcod == 0) {
                BOOST_CHECK(l.dummy);
                BOOST_CHECK(!l.pilots); // never pilots
                BOOST_CHECK_EQUAL(l.slots, 36);
                BOOST_CHECK_EQUAL(l.plframe_symbols(), 3330);
                BOOST_CHECK(!l.deliverable());
                continue;
            }
            BOOST_CHECK(!l.dummy);
            if (modcod >= 29) { // reserved MODCODs: length unknown
                BOOST_CHECK_EQUAL(l.slots, 0);
                BOOST_CHECK_EQUAL(l.plframe_symbols(), 0);
                BOOST_CHECK(!l.deliverable());
                continue;
            }
            const int eta = modcod <= 11 ? 2 : modcod <= 17 ? 3 : modcod <= 23 ? 4 : 5;
            const int S = shortf ? slots.at(eta).second : slots.at(eta).first;
            BOOST_CHECK_EQUAL((int)l.bits_per_symbol, eta);
            BOOST_CHECK_EQUAL((int)l.slots, S);
            BOOST_CHECK_EQUAL(l.pilots, pilots);
            BOOST_CHECK_EQUAL(l.fec_size == FECFRAME_SHORT, shortf);
            BOOST_CHECK_EQUAL(l.data_symbols(), 90 * S);
            BOOST_CHECK_EQUAL(l.pilot_block_count(), pilots ? (S - 1) / 16 : 0);
            BOOST_CHECK_EQUAL(l.plframe_symbols(), plframe_len(S, pilots));
            BOOST_CHECK_EQUAL((int)l.modcod, modcod);
            // deliverable iff a cuSAT MODCOD with a FEC table at this size
            const modcod_def_t* m = find_modcod(modcod);
            const bool expect = m && modcod_supports_framesize(*m, l.fec_size);
            BOOST_CHECK_EQUAL(l.deliverable(), expect);
        }
    }
    // spot values of Table 11 / Figure 13
    BOOST_CHECK_EQUAL(pl_layout_from_code(4 * 4 + 1).plframe_symbols(), 33282);  // QPSK 1/2 pilots
    BOOST_CHECK_EQUAL(pl_layout_from_code(4 * 4).plframe_symbols(), 32490);      // no pilots
    BOOST_CHECK_EQUAL(pl_layout_from_code(28 * 4 + 3).plframe_symbols(), 3402);  // 32APSK short pilots
    BOOST_CHECK(!pl_layout_from_code(11 * 4 + 3).deliverable());                 // QPSK 9/10 short
}

BOOST_AUTO_TEST_CASE(test_s2x_codes)
{
    // Table 17b: reserved codes and their lengths
    const std::map<int, int> reserved = {
        { 128, 21690 }, { 130, 16290 }, { 176, 13050 }, { 177, 13338 }, { 188, 10890 },
        { 189, 11142 }, { 192, 10890 }, { 193, 11142 }, { 196, 10890 }, { 197, 11142 },
        { 250, 22194 }, { 251, 16686 }, { 252, 13338 }, { 253, 11142 }, { 254, 8370 },
        { 255, 6714 },
    };
    int n_known = 0;
    for (int code = 128; code < 256; code++) {
        const pl_layout l = pl_layout_from_code(code);
        BOOST_TEST_CONTEXT("code " << code)
        {
            BOOST_CHECK(!l.dummy);
            if (code == 129 || code == 131) { // VL-SNR
                BOOST_CHECK_EQUAL(l.plframe_symbols(), 0);
                BOOST_CHECK(!l.deliverable());
                continue;
            }
            n_known++;
            auto it = reserved.find(code);
            if (it != reserved.end()) {
                BOOST_CHECK(l.reserved);
                BOOST_CHECK_EQUAL(l.plframe_symbols(), it->second);
                BOOST_CHECK(!l.deliverable());
                continue;
            }
            BOOST_CHECK(!l.reserved);
            BOOST_CHECK_EQUAL(l.pilots, (bool)(code & 1));
            // Table 16 / Table 17a: S from the order and frame size
            const bool shortf = code >= 216;
            const int eta = l.bits_per_symbol;
            int S = shortf ? 16200 / (90 * eta) : (eta == 7 ? 103 : 64800 / (90 * eta));
            BOOST_CHECK_EQUAL((int)l.slots, S);
            BOOST_CHECK_EQUAL(l.plframe_symbols(), plframe_len(S, code & 1));
            BOOST_CHECK_EQUAL(l.deliverable(), l.modcod != 0);
        }
    }
    BOOST_CHECK_EQUAL(n_known, 126);
    // 128APSK: 103 slots (EN 302 307-2 clauses 5.3.3 / 5.5.0)
    BOOST_CHECK_EQUAL(pl_layout_from_code(200).slots, 103);
    BOOST_CHECK_EQUAL(pl_layout_from_code(200).data_symbols(), 9270);
}

BOOST_AUTO_TEST_CASE(test_s2x_mapping_table)
{
    // cuSAT index <-> Table 17a code (pilots off)
    const std::map<int, int> map = {
        { 33, 132 }, { 43, 134 }, { 44, 136 }, { 56, 138 }, { 57, 140 }, { 45, 142 }, { 46, 144 },
        { 47, 146 }, { 58, 148 }, { 59, 150 }, { 60, 152 }, { 48, 154 }, { 49, 156 }, { 61, 158 },
        { 50, 160 }, { 51, 162 }, { 62, 164 }, { 52, 166 }, { 53, 168 }, { 54, 170 }, { 55, 172 },
        { 63, 174 }, { 64, 178 }, { 65, 180 }, { 66, 182 }, { 36, 184 }, { 29, 186 }, { 30, 190 },
        { 31, 194 }, { 35, 198 }, { 32, 200 }, { 34, 202 }, { 39, 204 }, { 41, 206 }, { 40, 208 },
        { 38, 210 }, { 42, 212 }, { 37, 214 },
    };
    for (auto [idx, code] : map) {
        BOOST_CHECK_EQUAL((int)pls_code_for(idx, FECFRAME_NORMAL, false), code);
        BOOST_CHECK_EQUAL((int)pls_code_for(idx, FECFRAME_NORMAL, true), code + 1);
        const pl_layout l = pl_layout_from_code(code);
        BOOST_CHECK_EQUAL((int)l.modcod, idx);
        const modcod_def_t* m = find_modcod(idx);
        BOOST_REQUIRE(m);
        const int eta = m->constellation == MOD_QPSK ? 2 :
                        (m->constellation == MOD_8PSK || m->constellation == MOD_8APSK) ? 3 :
                        m->constellation == MOD_16APSK ? 4 : m->constellation == MOD_32APSK ? 5 :
                        m->constellation == MOD_64APSK ? 6 : m->constellation == MOD_128APSK ? 7 : 8;
        BOOST_CHECK_EQUAL((int)l.bits_per_symbol, eta);
    }
}

BOOST_AUTO_TEST_CASE(test_round_trip)
{
    for (const modcod_def_t& m : get_supported_modcods())
        for (fecframe_size_t fs : { FECFRAME_NORMAL, FECFRAME_SHORT })
            for (bool pilots : { false, true }) {
                if (!modcod_supports_framesize(m, fs))
                    continue;
                const uint8_t code = pls_code_for(m.modcod_idx, fs, pilots);
                const pl_layout l = pl_layout_from_code(code);
                BOOST_TEST_CONTEXT(m.name << " fs " << fs << " pilots " << pilots)
                {
                    BOOST_CHECK_EQUAL((int)l.modcod, (int)m.modcod_idx);
                    BOOST_CHECK_EQUAL(l.fec_size, fs);
                    BOOST_CHECK_EQUAL(l.pilots, pilots);
                    BOOST_CHECK(l.deliverable());
                    BOOST_CHECK_EQUAL((int)pl_layout_for(m.modcod_idx, fs, pilots).pls_code, (int)code);
                    // XFECFRAME carries the whole FECFRAME (128APSK: + padding)
                    const int bits = fecframe_bits(fs);
                    if (l.bits_per_symbol == 7)
                        BOOST_CHECK_EQUAL(l.data_symbols(), (bits + 6) / 7 + 12);
                    else
                        BOOST_CHECK_EQUAL(l.data_symbols() * l.bits_per_symbol, bits);
                }
            }
    // dummy
    BOOST_CHECK(pl_layout_for(0, FECFRAME_NORMAL, true).dummy);
    BOOST_CHECK(!pl_layout_for(0, FECFRAME_NORMAL, true).pilots);
}

BOOST_AUTO_TEST_CASE(test_bounds)
{
    int max_payload = 0, max_plf = 0;
    for (int code = 0; code < 256; code++) {
        const pl_layout l = pl_layout_from_code(code);
        max_payload = std::max(max_payload, l.payload_symbols());
        max_plf = std::max(max_plf, l.plframe_symbols());
    }
    BOOST_CHECK_EQUAL(pl::max_payload_symbols(), max_payload);
    BOOST_CHECK_EQUAL(pl::max_plframe_symbols(), max_plf);
    BOOST_CHECK_EQUAL(pl::max_plframe_symbols(), 33282);
    BOOST_CHECK_EQUAL(pl::max_payload_symbols(), 33192);
}

} // namespace cusat_dsp
