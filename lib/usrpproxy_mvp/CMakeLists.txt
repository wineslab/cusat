# SPDX-License-Identifier: Apache-2.0
#
# libusrpproxy_mvp — embedded UHD-host-side tenant server that speaks the
# usrpproxy shm-ring protocol to an OAI softmodem.
#
# This library is an in-tree reimplementation (server side only) of the
# reference MVP from the standalone usrpproxy repo. The wire protocol
# (shared_ring.h) is kept in sync manually; the handshake rejects mismatched
# versions loudly at attach time.

add_library(usrpproxy_mvp STATIC
    tenant_server.cpp)

# Linked into libradio_uhd.so (a shared object), so the static lib's objects
# must be position-independent.
set_target_properties(usrpproxy_mvp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(usrpproxy_mvp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}   # for tenant_server.h + shared_ring.h
)

if(TARGET UHD::UHD)
    target_link_libraries(usrpproxy_mvp PUBLIC UHD::UHD)
else()
    target_include_directories(usrpproxy_mvp PUBLIC ${UHD_INCLUDE_DIRS})
    target_link_libraries(usrpproxy_mvp PUBLIC ${UHD_LIBRARIES})
endif()

target_link_libraries(usrpproxy_mvp PRIVATE pthread rt)

target_compile_options(usrpproxy_mvp PRIVATE
    -Wall -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers)
