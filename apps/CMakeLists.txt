# apps/ — standalone receiver and transmitter binaries
#
# SPDX-License-Identifier: Apache-2.0

add_executable(dvbs2-gse-tx dvbs2-gse-tx.cc)
target_link_libraries(dvbs2-gse-tx PRIVATE cusat_dsp)
install(TARGETS dvbs2-gse-tx DESTINATION bin)

add_executable(dvbs2-gse-rx dvbs2-gse-rx.cc)
target_link_libraries(dvbs2-gse-rx PRIVATE cusat_dsp)
install(TARGETS dvbs2-gse-rx DESTINATION bin)

add_executable(dvbs2-gse-acm dvbs2-gse-acm.cc)
install(TARGETS dvbs2-gse-acm DESTINATION bin)

add_executable(iq-farrow-resample iq_farrow_resample.cc)
target_link_libraries(iq-farrow-resample PRIVATE cusat_dsp)

add_executable(bench-ldpc-decode bench_ldpc_decode.cc)
target_link_libraries(bench-ldpc-decode PRIVATE cusat_dsp)

add_executable(bench-bch-gpu bench_bch_gpu.cc)
target_link_libraries(bench-bch-gpu PRIVATE cusat_dsp)
add_test(NAME qa_bch_gpu COMMAND bench-bch-gpu --verify-only)

add_executable(bench-timing-recovery bench_timing_recovery.cu)
target_link_libraries(bench-timing-recovery PRIVATE cusat_dsp)

add_executable(bench-coded-modulation bench_coded_modulation.cc)
target_link_libraries(bench-coded-modulation PRIVATE cusat_dsp)

add_executable(dvbs2x-vv-check dvbs2x-vv-check.cc)
target_link_libraries(dvbs2x-vv-check PRIVATE cusat_dsp)

# CPU-only wire/timing regression for immediate and timestamped channel
# commands. It intentionally exercises the header used by dvbs2-uhd-trx.
add_executable(qa-channel-impairment qa_channel_impairment.cc)
add_test(NAME qa_channel_impairment COMMAND qa-channel-impairment)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME qa_acm_pathloss_driver
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/qa_acm_pathloss_driver.py)
endif()

# Paced, bidirectional TCP sample transport for radio-free end-to-end ACM.
add_library(radio_tcp SHARED trx/radio_tcp.cc)
target_include_directories(radio_tcp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(radio_tcp PROPERTIES OUTPUT_NAME radio_tcp)
install(TARGETS radio_tcp DESTINATION lib)

# Transceiver app — Apache-2.0, hardware-agnostic. It loads a radio driver
# plugin (e.g. libradio_uhd.so) at runtime via dlopen and does NOT link the
# GPLv3 USRP driver, so it builds without UHD. Gated by ENABLE_UHD only to
# preserve the historical build switch (it still needs a plugin at runtime).
if(CUSAT_BUILD_UHD OR ENABLE_TRX_APP)
    add_executable(dvbs2-uhd-trx
        dvbs2-uhd-trx.cc
        trx/bench_tx.cc
        trx/options.cc
        trx/radio_loader.cc
        trx/rx_threads.cc
        trx/tap_thread.cc
        trx/tx_threads.cc)
    target_include_directories(dvbs2-uhd-trx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(dvbs2-uhd-trx PRIVATE cusat_dsp ${CMAKE_DL_LIBS})
    install(TARGETS dvbs2-uhd-trx DESTINATION bin)
endif()

# UHD radio driver plugins (radio_uhd, radio_difi) — the only components
# that link UHD. Sources are Apache-2.0; the built .so files link GPLv3 UHD,
# so the combined binaries are distributed under GPLv3 terms.
# Built as a standalone shared object and loaded at runtime via dlopen.
if(CUSAT_BUILD_UHD)
    add_library(radio_uhd SHARED trx/radio_uhd.cc)
    target_include_directories(radio_uhd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET UHD::UHD)
        target_link_libraries(radio_uhd PRIVATE UHD::UHD)
    else()
        target_include_directories(radio_uhd PRIVATE ${UHD_INCLUDE_DIRS})
        target_link_libraries(radio_uhd PRIVATE ${UHD_LIBRARIES})
    endif()
    # Embedded usrpproxy tenant server (shares the multi_usrp with an OAI
    # softmodem over shm). Lives on the UHD-linking side, inside this plugin.
    target_link_libraries(radio_uhd PRIVATE usrpproxy_mvp)
    install(TARGETS radio_uhd DESTINATION lib)

    # DIFI radio plugin (rfnoc-difi X410 images, DIFI v1.1 big-endian): RX =
    # DIFI/UDP straight from the FPGA (UHD 4.8 remote raw_payload divert),
    # TX + control plane = UHD RFNoC graph. Selected by "--usrp difi:...".
    # Source is Apache-2.0; the built .so links UHD (GPLv3), so the combined
    # binary follows GPLv3 like libradio_uhd. No usrpproxy in DIFI mode.
    add_library(radio_difi SHARED trx/radio_difi.cc)
    target_include_directories(radio_difi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET UHD::UHD)
        target_link_libraries(radio_difi PRIVATE UHD::UHD)
    else()
        target_include_directories(radio_difi PRIVATE ${UHD_INCLUDE_DIRS})
        target_link_libraries(radio_difi PRIVATE ${UHD_LIBRARIES})
    endif()
    install(TARGETS radio_difi DESTINATION lib)

    add_library(radio_compfh SHARED trx/radio_compfh.cc)
    target_link_libraries(radio_compfh PRIVATE Boost::boost)
    target_include_directories(radio_compfh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/lib/cusat_dsp/include)
    if(TARGET UHD::UHD)
        target_link_libraries(radio_compfh PRIVATE UHD::UHD)
    else()
        target_include_directories(radio_compfh PRIVATE ${UHD_INCLUDE_DIRS})
        target_link_libraries(radio_compfh PRIVATE ${UHD_LIBRARIES})
    endif()
    install(TARGETS radio_compfh DESTINATION lib)


endif()

# UHD-free candidate DIFI profile codecs, shared by the plugin and QA.
add_library(difi_compression STATIC trx/difi_compression.cc)
set_target_properties(difi_compression PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(difi_compression PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET radio_compfh)
    target_link_libraries(radio_compfh PRIVATE difi_compression)
endif()
add_executable(qa_difi_compression qa_difi_compression.cc)
target_link_libraries(qa_difi_compression PRIVATE difi_compression Boost::unit_test_framework)
target_include_directories(qa_difi_compression PRIVATE ${PROJECT_SOURCE_DIR}/lib/cusat_dsp/include)
target_compile_definitions(qa_difi_compression PRIVATE BOOST_TEST_DYN_LINK BOOST_TEST_MAIN)
add_test(NAME qa_difi_compression COMMAND qa_difi_compression)
