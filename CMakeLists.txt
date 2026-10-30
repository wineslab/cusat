# cuSAT — GPU-accelerated DVB-S2/S2X transceiver
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(cusat CXX C)
enable_testing()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to Release.")
endif()
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "")

# CMake module path

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# compile_commands.json for editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

########################################################################
# Dependencies
########################################################################
find_package(Boost REQUIRED unit_test_framework)

########################################################################
# Options
########################################################################
set(ENABLE_UHD "AUTO" CACHE STRING
    "Build UHD radio plugins: AUTO, ON, or OFF")
set_property(CACHE ENABLE_UHD PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${ENABLE_UHD}" ENABLE_UHD)
# Preserve compatibility with the boolean values accepted by the former
# option(ENABLE_UHD ...) declaration and by common CMake automation.
if(ENABLE_UHD MATCHES "^(1|TRUE|YES|Y)$")
    set(ENABLE_UHD "ON")
elseif(ENABLE_UHD MATCHES "^(0|FALSE|NO|N)$")
    set(ENABLE_UHD "OFF")
endif()
if(NOT ENABLE_UHD MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "ENABLE_UHD must be AUTO, ON, or OFF (got '${ENABLE_UHD}')")
endif()

# Resolve UHD once, at the top level. AUTO makes an ordinary release build
# include radio support when the development package is installed, while ON
# remains the reproducible/strict choice for radio deployments.
set(CUSAT_BUILD_UHD OFF)
if(ENABLE_UHD STREQUAL "AUTO")
    find_package(UHD 4.6 QUIET)
    if(UHD_FOUND)
        set(CUSAT_BUILD_UHD ON)
    endif()
elseif(ENABLE_UHD STREQUAL "ON")
    find_package(UHD 4.6 REQUIRED)
    set(CUSAT_BUILD_UHD ON)
endif()

if(CUSAT_BUILD_UHD)
    if(DEFINED UHD_VERSION)
        set(_cusat_uhd_version "${UHD_VERSION}")
    elseif(DEFINED UHD_VERSION_STRING)
        set(_cusat_uhd_version "${UHD_VERSION_STRING}")
    else()
        set(_cusat_uhd_version "version unavailable")
    endif()
    message(STATUS "UHD: enabled (${_cusat_uhd_version})")
else()
    message(STATUS "UHD: disabled (${ENABLE_UHD})")
endif()
option(ENABLE_TRX_APP "Build the transceiver app (dvbs2-uhd-trx) without the UHD plugin (UHD-free; loads a radio plugin at runtime)" OFF)
option(NATIVE_OPTIMIZATIONS "Optimize build for the local machine (-march=native)" ON)
if(NATIVE_OPTIMIZATIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# CUDA is mandatory
# JetPack exposes the active toolkit through /usr/local/cuda but does not
# necessarily add nvcc to PATH for non-interactive shells (SSH, CI, systemd).
# Prefer an explicit user/toolchain choice; otherwise use NVIDIA's standard
# symlink before enabling the CUDA language.
if(NOT DEFINED CMAKE_CUDA_COMPILER
   AND NOT DEFINED ENV{CUDACXX}
   AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
        "CUDA compiler" FORCE)
endif()
# Auto-detect CUDA SM from the build machine's GPU.  CMake's `native` value
# asks nvcc to probe the device, but older CMake releases can silently fall
# back to nvcc's default architecture for newer GPUs (for example SM 110/121).
# Query the driver directly when possible so a successful native build is
# actually runnable on the host.
# Override at configure time: cmake -DCMAKE_CUDA_ARCHITECTURES=90
# CI/Docker builds without GPU must pass the flag explicitly.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND "${NVIDIA_SMI_EXECUTABLE}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _cuda_compute_caps
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _nvidia_smi_result
        )
    endif()
    if(_nvidia_smi_result EQUAL 0 AND NOT _cuda_compute_caps STREQUAL "")
        string(REPLACE "\r" "" _cuda_compute_caps "${_cuda_compute_caps}")
        string(REPLACE "\n" ";" _cuda_compute_caps "${_cuda_compute_caps}")
        list(REMOVE_DUPLICATES _cuda_compute_caps)
        list(LENGTH _cuda_compute_caps _cuda_compute_cap_count)
        if(NOT _cuda_compute_cap_count EQUAL 1)
            message(FATAL_ERROR
                "Detected GPUs with different CUDA capabilities (${_cuda_compute_caps}); "
                "set CMAKE_CUDA_ARCHITECTURES explicitly")
        endif()
        list(GET _cuda_compute_caps 0 _cuda_compute_cap)
        string(REPLACE "." "" _cuda_compute_cap "${_cuda_compute_cap}")
        set(CMAKE_CUDA_ARCHITECTURES "${_cuda_compute_cap}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "CUDA: arch=${CMAKE_CUDA_ARCHITECTURES}")

########################################################################
# Subdirectories
########################################################################
add_subdirectory(lib)
add_subdirectory(apps)

########################################################################
# Uninstall target
########################################################################
# Removes the files listed in install_manifest.txt.
if(NOT TARGET uninstall)
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND}
                -DMANIFEST=${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake
        COMMENT "Uninstalling cuSAT")
endif()
