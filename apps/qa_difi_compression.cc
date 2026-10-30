// SPDX-License-Identifier: Apache-2.0
#include <boost/test/unit_test.hpp>
#include "trx/difi_compression.h"
#include "trx/plmod_pack.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
using sample = std::complex<int16_t>;
namespace c = difipack::compression;

BOOST_AUTO_TEST_CASE(trim_rounding_clipping_and_odd_count)
{
    std::vector<sample> input{{1000,-2000},{32000,-32000},{5,-5},{123,456},{-1,1}};
    for (int shift = 0; shift <= 8; ++shift) {
        auto payload = c::encode_trim_native(input, shift);
        BOOST_CHECK_EQUAL(payload.size(), 12);
        auto output = c::decode_native(payload, {8,shift}, input.size());
        std::vector<std::complex<float>> runtime(input.size());
        BOOST_CHECK_EQUAL(c::decode_fc32(payload.data(),payload.size(),runtime.data(),runtime.size(),1,shift),input.size());
        for(size_t i=0;i<input.size();++i) {
            BOOST_CHECK_CLOSE(runtime[i].real(),output[i].real()/32768.0f,0.001f);
            BOOST_CHECK_CLOSE(runtime[i].imag(),output[i].imag()/32768.0f,0.001f);
        }
        for (size_t i=0; i<input.size(); ++i) {
            auto quantize = [shift](int x) {
                int q = shift ? (x+(1<<(shift-1))) >> shift : x;
                return std::clamp(q,-127,127) * (1<<shift);
            };
            BOOST_CHECK_EQUAL(output[i].real(), quantize(input[i].real()));
            BOOST_CHECK_EQUAL(output[i].imag(), quantize(input[i].imag()));
        }
    }
}

BOOST_AUTO_TEST_CASE(dpcm_closed_loop_all_residual_widths)
{
    std::vector<sample> input;
    for (int i=0; i<512; ++i) input.emplace_back(12000*std::sin(i*.025),12000*std::cos(i*.025));
    for (int bits=1; bits<=8; ++bits) {
        auto payload = c::encode_dpcm(input, 8, bits);
        auto output = c::decode_dpcm(payload);
        BOOST_REQUIRE_EQUAL(output.size(),input.size());
        std::vector<std::complex<float>> runtime(input.size());
        BOOST_CHECK_EQUAL(c::decode_fc32(payload.data(),payload.size(),runtime.data(),runtime.size(),2,8),input.size());
        for(size_t i=0;i<input.size();++i) {
            BOOST_CHECK_CLOSE(runtime[i].real(),output[i].real()/32768.0f,0.001f);
            BOOST_CHECK_CLOSE(runtime[i].imag(),output[i].imag()/32768.0f,0.001f);
        }
        int ri=0,rq=0;
        auto trim=[](int x){return std::clamp((x+128)>>8,-127,127);};
        for (size_t i=0; i<input.size(); ++i) {
            auto next=[&](int previous,int value){
                int target=trim(value);
                if (!i) return target;
                return std::clamp(previous+std::clamp(target-previous,-(1<<(bits-1)),(1<<(bits-1))-1),-127,127);
            };
            ri=next(ri,input[i].real()); rq=next(rq,input[i].imag());
            BOOST_CHECK_EQUAL(output[i].real(),ri*256);
            BOOST_CHECK_EQUAL(output[i].imag(),rq*256);
        }
        if(bits==8) {
            auto trimmed=c::decode_native(c::encode_trim_native(input,8),{8,8},input.size());
            BOOST_CHECK_EQUAL_COLLECTIONS(output.begin(),output.end(),trimmed.begin(),trimmed.end());
        }
    }
}

BOOST_AUTO_TEST_CASE(generative_tx_deframer_seam)
{
    std::vector<uint32_t> words{0xD525A400,0x00007E90,0x05A02416,0x1234ABCD,0xFEDC9876};
    auto payload=c::encode_tx_modulation(words);
    BOOST_CHECK_EQUAL(payload[0],0x25); BOOST_CHECK_EQUAL(payload[1],0xD5);
    auto recovered=c::deframe_tx_modulation(payload);
    BOOST_CHECK_EQUAL_COLLECTIONS(words.begin(),words.end(),recovered.begin(),recovered.end());
    BOOST_CHECK_EQUAL(c::registry().at(c::PCC_TX_MODULATION).host_decodable,false);
}

BOOST_AUTO_TEST_CASE(reject_invalid_and_truncated_payloads)
{
    BOOST_CHECK_THROW(c::encode_dpcm({},8,6),std::invalid_argument);
    BOOST_CHECK_THROW(c::encode_dpcm({{1,2}},8,0),std::invalid_argument);
    BOOST_CHECK_THROW(c::encode_trim_native({{1,2}},9),std::invalid_argument);
    BOOST_CHECK_THROW(c::decode_native({0,1},{16,0},1),std::invalid_argument);
    auto payload=c::encode_dpcm({{1,2},{3,4},{5,6}},8,6);
    payload.resize(5);
    BOOST_CHECK_THROW(c::decode_dpcm(payload),std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(modulation_frame_geometry_and_header)
{
    complexf header[90]; std::fill_n(header,90,complexf(1,0));
    uint8_t fec[8100]{}; unsigned int columns[8]{};
    uint32_t words[2200]{};
    auto n=plmod_build_frame(words,4,true,header,fec,64800,columns,2);
    BOOST_REQUIRE(n>93); BOOST_CHECK_EQUAL(words[0]>>24,0xD5);
    auto entry=plmod_lookup(4); BOOST_REQUIRE(entry);
    BOOST_CHECK_EQUAL(words[1]&0xFFFFF,entry->n_slots*90);
    auto wire=c::encode_tx_modulation(std::vector<uint32_t>(words,words+n));
    auto restored=c::deframe_tx_modulation(wire);
    BOOST_CHECK_EQUAL_COLLECTIONS(words,words+n,restored.begin(),restored.end());
}

BOOST_AUTO_TEST_CASE(uhd_sc16_converter_and_fpga_deframer_seam)
{
    for(uint32_t native : {0xD525A400u,0x1234ABCDu,0xFEDC9876u,0u,0xFFFFFFFFu}) {
        auto cpu=c::uhd_input_word(native);
        uint32_t item32=(uint32_t)(uint16_t)cpu.real()<<16 | (uint16_t)cpu.imag();
        BOOST_CHECK_EQUAL(c::wire_swab16(item32),native);
    }
}
