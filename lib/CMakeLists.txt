# lib/ — cusat_dsp static library + unit tests
#
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cusat_dsp)

# usrpproxy_mvp — embedded tenant server that shares the USRP with an OAI
# softmodem over a POSIX shm ring. It links UHD, so it is built ONLY with the
# GPLv3 radio plugin (CUSAT_BUILD_UHD) and linked into libradio_uhd.so; the
# Apache-2.0 transceiver app never links it (keeps UHD out of the app).
if(CUSAT_BUILD_UHD)
    add_subdirectory(usrpproxy_mvp)
endif()

########################################################################
# Unit tests
########################################################################
list(APPEND test_sources
    qa_acm_controller.cc
    qa_bch_code.cc
    qa_bit_interleaver.cc
    qa_constellation.cc
    qa_crc8.cc
    qa_difi_packet.cc
    qa_farrow_interpolator.cc
    qa_farrow_cuda.cc
    qa_bb_scrambler.cc
    qa_bbframe_parser.cc
    qa_gse.cc
    qa_ldpc_encode.cc
    qa_ldpc_message_update.cc
    qa_tx_fec_chain.cc
    qa_pl_scrambler.cc
    qa_plheader.cc
    qa_rx_fec_chain.cc
    qa_snr_ewma.cc
    qa_usrpproxy_shared_ring.cc
    qa_pl_layout.cc
    qa_pl_receiver.cc
    qa_plheader_search.cc
    qa_filter_bank.cc
    qa_rrc_filter.cc
    qa_receiver_frontend.cc
)

foreach(qa_file ${test_sources})
    get_filename_component(test_name ${qa_file} NAME_WE)
    add_executable(${test_name} ${qa_file})
    target_link_libraries(${test_name}
        PRIVATE cusat_dsp
        PRIVATE Boost::unit_test_framework
    )
    target_compile_definitions(${test_name}
        PRIVATE BOOST_TEST_DYN_LINK
        PRIVATE BOOST_TEST_MAIN
    )
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cusat_dsp/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cusat_dsp
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

########################################################################
# Standalone fractional-NCO timing check (needs a GPU; run manually,
# intentionally NOT registered with ctest)
########################################################################
add_executable(test_timing_nco test_timing_nco.cc)
target_link_libraries(test_timing_nco PRIVATE cusat_dsp)
target_include_directories(test_timing_nco
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cusat_dsp/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cusat_dsp
)
