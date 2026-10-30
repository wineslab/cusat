/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tx FEC chain smoke test: verify encoding produces valid output.
 */

#include "cusat_dsp/tx_fec_chain.h"
#include "cusat_dsp/tx_fec_pool.h"
#include "cusat_dsp/bbframe_parser.h"
#include "cusat_dsp/fec/code_params.h"
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace cusat_dsp {

struct fec_test_case {
    code_rate_t rate;
    modulation_t constellation;
    const char* name;
    int n_mod; // bits per symbol
};

static const fec_test_case test_cases[] = {
    { RATE_1_4,  MOD_QPSK,   "QPSK 1/4",   2 },
    { RATE_1_2,  MOD_QPSK,   "QPSK 1/2",   2 },
    { RATE_3_5,  MOD_QPSK,   "QPSK 3/5",   2 },
    { RATE_3_4,  MOD_QPSK,   "QPSK 3/4",   2 },
    { RATE_9_10, MOD_QPSK,   "QPSK 9/10",  2 },
    { RATE_3_5,  MOD_8PSK,   "8PSK 3/5",   3 },
    { RATE_2_3,  MOD_8PSK,   "8PSK 2/3",   3 },
    { RATE_3_4,  MOD_8PSK,   "8PSK 3/4",   3 },
    { RATE_3_4,  MOD_16APSK, "16APSK 3/4", 4 },
    { RATE_2_3,  MOD_16APSK, "16APSK 2/3", 4 },
    { RATE_3_4,  MOD_32APSK, "32APSK 3/4", 5 },
    { RATE_132_180, MOD_64APSK, "64APSK 11/15", 6 },
};

BOOST_AUTO_TEST_CASE(test_tx_fec_chain_output)
{
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 255);

    for (const auto& tc : test_cases) {
        BOOST_TEST_MESSAGE("Testing Tx FEC chain for " << tc.name);

        tx_fec_chain tx(FECFRAME_NORMAL, tc.rate, tc.constellation);

        const unsigned int xfec_len = tx.xfecframe_len();
        const unsigned int max_dfl = tx.max_dfl_bytes();

        // Expected XFECFRAME length: 64800 / n_mod
        BOOST_CHECK_EQUAL(xfec_len, 64800u / tc.n_mod);
        BOOST_CHECK_GT(max_dfl, 0u);

        // Generate random GSE data
        std::vector<uint8_t> gse_data(max_dfl);
        for (unsigned int i = 0; i < max_dfl; i++)
            gse_data[i] = dis(gen);

        // Encode
        std::vector<complexf> symbols(xfec_len);
        unsigned int gse_consumed = tx.process_gse(gse_data.data(), max_dfl, symbols.data());
        BOOST_CHECK_GT(gse_consumed, 0u);

        // Verify output is non-trivial (not all zeros)
        float total_power = 0;
        for (unsigned int i = 0; i < xfec_len; i++) {
            total_power += std::norm(symbols[i]);
        }
        BOOST_CHECK_GT(total_power, 0.0f);

        // Verify average symbol power is reasonable (should be ~1.0 for unit-power constellations)
        float avg_power = total_power / xfec_len;
        BOOST_CHECK_GT(avg_power, 0.5f);
        BOOST_CHECK_LT(avg_power, 2.0f);

        // Encode a second frame with different data to verify determinism / no crash
        for (unsigned int i = 0; i < max_dfl; i++)
            gse_data[i] = dis(gen);
        std::vector<complexf> symbols2(xfec_len);
        gse_consumed = tx.process_gse(gse_data.data(), max_dfl, symbols2.data());
        BOOST_CHECK_GT(gse_consumed, 0u);

        // Different input → different output (with very high probability)
        bool differ = false;
        for (unsigned int i = 0; i < xfec_len; i++) {
            if (symbols[i] != symbols2[i]) {
                differ = true;
                break;
            }
        }
        BOOST_CHECK(differ);

        BOOST_TEST_MESSAGE("  " << tc.name << ": xfec_len=" << xfec_len
                           << " avg_power=" << avg_power);
    }
}

BOOST_AUTO_TEST_CASE(test_tx_fec_chain_bbheader_rolloff_bits)
{
    tx_fec_chain tx(FECFRAME_NORMAL, RATE_1_2, MOD_QPSK);
    std::vector<uint8_t> gse_data(16, 0xAB);
    auto next_ro = [&]() {
        tx.process_gse_to_bbframe(gse_data.data(), gse_data.size());
        return static_cast<unsigned>(tx.bbframe()[0] & 0x03);
    };

    tx.set_rolloff(0.35f);
    BOOST_CHECK_EQUAL(next_ro(), 0u);
    BOOST_CHECK_EQUAL(next_ro(), 0u);

    tx.set_rolloff(0.25f);
    BOOST_CHECK_EQUAL(next_ro(), 1u);
    BOOST_CHECK_EQUAL(next_ro(), 1u);

    tx.set_rolloff(0.20f);
    BOOST_CHECK_EQUAL(next_ro(), 2u);
    BOOST_CHECK_EQUAL(next_ro(), 2u);

    tx.set_rolloff(0.15f);
    BOOST_CHECK_EQUAL(next_ro(), 3u);
    BOOST_CHECK_EQUAL(next_ro(), 0u);
    BOOST_CHECK_EQUAL(next_ro(), 3u);
    BOOST_CHECK_EQUAL(next_ro(), 0u);

    tx.set_rolloff(0.10f);
    BOOST_CHECK_EQUAL(next_ro(), 3u);
    BOOST_CHECK_EQUAL(next_ro(), 1u);
    BOOST_CHECK_EQUAL(next_ro(), 3u);
    BOOST_CHECK_EQUAL(next_ro(), 1u);

    tx.set_rolloff(0.05f);
    BOOST_CHECK_EQUAL(next_ro(), 3u);
    BOOST_CHECK_EQUAL(next_ro(), 2u);
    BOOST_CHECK_EQUAL(next_ro(), 3u);
    BOOST_CHECK_EQUAL(next_ro(), 2u);

    BOOST_CHECK_THROW(tx.set_rolloff(0.12f), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_tx_fec_chain_bbheader_rolloff_bits_acm)
{
    tx_fec_chain tx(FECFRAME_NORMAL, RATE_1_2, MOD_QPSK);
    std::vector<uint8_t> gse_data(16, 0xAB);
    auto next_matype = [&]() {
        tx.process_gse_to_bbframe(gse_data.data(), gse_data.size());
        return static_cast<unsigned>(tx.bbframe()[0]);
    };

    tx.set_acm_mode(true);
    tx.set_rolloff(0.10f);
    BOOST_CHECK_EQUAL(next_matype(), 0b00100011);
    BOOST_CHECK_EQUAL(next_matype(), 0b00100001);
    BOOST_CHECK_EQUAL(next_matype(), 0b00100011);
    BOOST_CHECK_EQUAL(next_matype(), 0b00100001);
}

BOOST_AUTO_TEST_CASE(test_tx_fec_chain_acm_bbheader_roundtrip)
{
    tx_fec_chain tx(FECFRAME_NORMAL, RATE_1_4, MOD_QPSK);
    std::vector<uint8_t> gse_data(32, 0xA5);
    std::vector<uint8_t> decoded(gse_data.size());

    tx.set_acm_mode(true);
    tx.set_rolloff(0.20f);
    tx.process_gse_to_bbframe(gse_data.data(), gse_data.size());

    // MATYPE signals generic packetized SIS + ACM, and its changed value is
    // covered by a valid BBHEADER CRC accepted by the normal RX deheader.
    BOOST_CHECK_EQUAL(tx.bbframe()[0], 0b00100010);
    const size_t kbch_bytes = get_fec_params(FECFRAME_NORMAL, RATE_1_4).k_bch / 8;
    int decoded_len = extract_gse_datafield(
        tx.bbframe(), kbch_bytes, decoded.data(), static_cast<int>(decoded.size()));
    BOOST_REQUIRE_EQUAL(decoded_len, static_cast<int>(gse_data.size()));
    BOOST_CHECK_EQUAL_COLLECTIONS(
        decoded.begin(), decoded.end(), gse_data.begin(), gse_data.end());
}

BOOST_AUTO_TEST_CASE(test_tx_fec_pool_bbheader_low_rolloff_state_per_modcod)
{
    tx_fec_pool pool(FECFRAME_NORMAL, { 4, 5 }, 0.05f);
    std::vector<uint8_t> gse_data(16, 0xAB);

    auto next_matype = [&](uint8_t modcod_idx) {
        tx_fec_chain& tx = pool.get(modcod_idx);
        tx.process_gse_to_bbframe(gse_data.data(), gse_data.size());
        return static_cast<unsigned>(tx.bbframe()[0]);
    };

    BOOST_CHECK_EQUAL(next_matype(4), 0b00100011);
    BOOST_CHECK_EQUAL(next_matype(5), 0b00100011);
    BOOST_CHECK_EQUAL(next_matype(4), 0b00100010);
    BOOST_CHECK_EQUAL(next_matype(5), 0b00100010);
}

/* TX scramble + BCH, then the RX post-processing steps (descramble the
 * systematic BBFRAME part, extract the GSE data field) recover the input */
BOOST_AUTO_TEST_CASE(test_tx_bch_descramble_extract_roundtrip)
{
    std::mt19937 rng(7);
    for (auto fs : { FECFRAME_NORMAL, FECFRAME_SHORT }) {
        for (auto rate : { RATE_1_4, RATE_3_5 }) {
            tx_fec_chain tx(fs, rate, MOD_QPSK);
            const size_t kbch_bytes = get_fec_params(fs, rate).k_bch / 8;
            std::vector<uint8_t> gse(kbch_bytes - 10 - (rng() % 50));
            for (auto& b : gse)
                b = (uint8_t)rng();
            tx.process_gse_to_bch(gse.data(), (unsigned)gse.size());
            BOOST_REQUIRE_GE(tx.bch_len_bytes(), kbch_bytes);
            std::vector<uint8_t> bb(kbch_bytes), out(kbch_bytes);
            bb_scramble(tx.bch_output(), bb.data(), kbch_bytes);
            const int n = extract_gse_datafield(bb.data(), kbch_bytes, out.data(), (int)out.size());
            BOOST_REQUIRE_EQUAL(n, (int)gse.size());
            BOOST_CHECK(std::equal(gse.begin(), gse.end(), out.begin()));
            // the scrambled frame itself must not parse
            bbheader_fields h;
            BOOST_CHECK(!decode_bbheader(tx.bch_output(), kbch_bytes, h));
        }
    }
}

} // namespace cusat_dsp
