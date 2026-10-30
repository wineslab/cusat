# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(cusat_oai_usrpproxy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OAI_SOURCE_DIR "" CACHE PATH "OAI source tree used to build nr-softmodem")
set(CUSAT_SOURCE_DIR "" CACHE PATH "cuSAT source tree containing the shared protocol header")

if(NOT EXISTS "${OAI_SOURCE_DIR}/radio/COMMON/common_lib.h")
  message(FATAL_ERROR "OAI_SOURCE_DIR does not point to a compatible OAI source tree")
endif()
if(NOT EXISTS "${CUSAT_SOURCE_DIR}/lib/usrpproxy_mvp/shared_ring.h")
  message(FATAL_ERROR "CUSAT_SOURCE_DIR does not contain lib/usrpproxy_mvp/shared_ring.h")
endif()

add_library(usrpproxy MODULE usrpproxy_lib.cpp)
target_include_directories(usrpproxy PRIVATE
  "${OAI_SOURCE_DIR}"
  "${OAI_SOURCE_DIR}/radio/COMMON"
  "${CUSAT_SOURCE_DIR}/lib/usrpproxy_mvp")
target_link_libraries(usrpproxy PRIVATE rt pthread)
target_compile_options(usrpproxy PRIVATE
  -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
set_target_properties(usrpproxy PROPERTIES PREFIX "lib")

install(TARGETS usrpproxy LIBRARY DESTINATION lib)
