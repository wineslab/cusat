# libcusat_dsp — standalone DSP library
#
# SPDX-License-Identifier: Apache-2.0

list(APPEND cusat_dsp_sources
    framing/plframe.cc
    plrx/pl_receiver.cc
    framing/plheader.cc
    framing/pl_scrambler.cc
    fec/bch_code.cc
    fec/code_params.cc
    bb_header.cc
    bb_scrambler.cc
    bbframe_parser.cc
    rx_fec_chain.cc
    rx_fec_pool.cc
    gse.cc
    tap_interface.cc
    ldpc_decode.cc
    ldpc_encode.cc
    ldpc_table_factory.cc
    mapper.cc
    pl_framer.cc
    transmitter_frontend.cc
    tx_fec_chain.cc
    tx_fec_pool.cc
    farrow_interpolator.cc
    interpolator.cc
    rrc_filter.cc
    receiver_frontend.cc
)

add_library(cusat_dsp STATIC ${cusat_dsp_sources})

target_include_directories(cusat_dsp
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

list(APPEND cusat_dsp_cuda_sources
    # host-side tables shared by the GPU mapper/demapper and the C++ code
    modulation/bit_interleaver.cc
    modulation/constellation.cc
    modulation/dd_snr.cc
    cuda/cuda_util.cu
    cuda/farrow_kernel.cu
    cuda/rrc_kernel.cu
    cuda/pl_scramble_kernel.cu
    cuda/bch_encode_kernel.cu
    cuda/bch_check_kernel.cu
    cuda/ldpc_encode_kernel.cu
    cuda/ldpc_decode_kernel.cu
    cuda/demapper_kernel.cu
    cuda/mapper_kernel.cu
    cuda/tx_pipeline.cu
    cuda/plheader_search_kernel.cu
    cuda/timing_recovery_kernel.cu
    cuda/agc_rot_kernel.cu
    cuda/polyphase_bank_kernel.cu
)
add_library(cusat_dsp_cuda STATIC ${cusat_dsp_cuda_sources})
target_include_directories(cusat_dsp_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(cusat_dsp_cuda PUBLIC CUDA::cudart)
set_target_properties(cusat_dsp_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cusat_dsp PUBLIC cusat_dsp_cuda)

set_target_properties(cusat_dsp PROPERTIES POSITION_INDEPENDENT_CODE ON)
