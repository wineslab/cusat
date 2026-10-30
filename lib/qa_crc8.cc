/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BBHEADER CRC-8 (EN 302 307-1 clause 5.1.6).
 */

#include "cusat_dsp/fec/crc8.h"
#include <boost/test/unit_test.hpp>
#include <vector>

namespace cusat_dsp {

BOOST_AUTO_TEST_CASE(test_check_value)
{
    // Catalogued check value of CRC-8/DVB-S2 (poly 0xD5, init 0, no reflection)
    const uint8_t msg[] = { '1', '2', '3', '4', '5', '6', '7', '8', '9' };
    BOOST_CHECK_EQUAL(crc8_bbheader(msg, 9), 0xBC);
    BOOST_CHECK_EQUAL(crc8_bbheader(msg, 0), 0);
}

BOOST_AUTO_TEST_CASE(test_appended_crc_checks_to_zero)
{
    std::vector<uint8_t> hdr = { 0xF2, 0x00, 0x00, 0x00, 0x7E, 0x10, 0x00, 0x00, 0x00 };
    hdr.push_back(crc8_bbheader(hdr.data(), 9));
    BOOST_CHECK_EQUAL(crc8_bbheader(hdr.data(), 10), 0);
    hdr[3] ^= 0x10; // any single-bit error is detected
    BOOST_CHECK_NE(crc8_bbheader(hdr.data(), 10), 0);
}

} // namespace cusat_dsp
