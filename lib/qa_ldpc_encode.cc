/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * LDPC encoder unit test: encode → GPU decode round-trip.
 */

#include "cusat_dsp/ldpc_encode.h"
#include "cusat_dsp/ldpc_decode.h"
#include <boost/test/unit_test.hpp>
#include <cstring>
#include <random>
#include <vector>

namespace cusat_dsp {

struct ldpc_test_case {
    code_rate_t rate;
    const char* name;
};

static const ldpc_test_case dvbs2_normal_rates[] = {
    { RATE_1_4,  "RATE_1_4"  },
    { RATE_1_3,  "RATE_1_3"  },
    { RATE_2_5,  "RATE_2_5"  },
    { RATE_1_2,  "RATE_1_2"  },
    { RATE_3_5,  "RATE_3_5"  },
    { RATE_2_3,  "RATE_2_3"  },
    { RATE_3_4,  "RATE_3_4"  },
    { RATE_4_5,  "RATE_4_5"  },
    { RATE_5_6,  "RATE_5_6"  },
    { RATE_8_9,  "RATE_8_9"  },
    { RATE_9_10, "RATE_9_10" },
};

BOOST_AUTO_TEST_CASE(test_ldpc_encode_decode_roundtrip)
{
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 255);

    for (const auto& tc : dvbs2_normal_rates) {
        BOOST_TEST_MESSAGE("Testing LDPC encode/decode for " << tc.name);

        ldpc_encode encoder(FECFRAME_NORMAL, tc.rate);
        ldpc_decode decoder(FECFRAME_NORMAL, tc.rate, 50);

        const unsigned int K = encoder.k_bits();
        const unsigned int N = encoder.n_bits();
        const unsigned int K_bytes = encoder.k_bytes();
        const unsigned int N_bytes = encoder.n_bytes();

        BOOST_CHECK_EQUAL(K, decoder.k_bits());
        BOOST_CHECK_EQUAL(N, decoder.n_bits());

        // Random info bits
        std::vector<uint8_t> info(K_bytes);
        for (auto& b : info)
            b = dis(gen);

        // Encode
        std::vector<uint8_t> codeword(N_bytes);
        encoder.encode(info.data(), codeword.data());

        // Verify systematic part is preserved
        BOOST_CHECK_EQUAL_COLLECTIONS(
            info.begin(), info.end(),
            codeword.begin(), codeword.begin() + K_bytes);

        // Prepare soft bits for GPU decoder (single codeword batch)
        std::vector<int16_t> soft(N);
        for (unsigned int j = 0; j < N_bytes; j++) {
            for (int b = 0; b < 8; b++) {
                int bit = (codeword[j] >> (7 - b)) & 1;
                soft[j * 8 + b] = bit ? -127 : 127;
            }
        }

        // GPU decode (batch_size=1)
        std::vector<uint8_t> decoded(K_bytes);
        int trials_left = 0;
        decoder.decode_batch_cuda(soft.data(), decoded.data(), &trials_left, 1);
        BOOST_CHECK_GT(trials_left, 0); // should converge quickly

        // Verify decoded matches original info
        BOOST_CHECK_EQUAL_COLLECTIONS(
            info.begin(), info.end(),
            decoded.begin(), decoded.end());
    }
}

} // namespace cusat_dsp
