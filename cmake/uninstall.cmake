# SPDX-License-Identifier: Apache-2.0
#
# Removes every file recorded by `cmake --install` / `make install` in
# install_manifest.txt. Invoked by the `uninstall` target:
#   cmake -DMANIFEST=<build>/install_manifest.txt -P uninstall.cmake

if(NOT DEFINED MANIFEST OR NOT EXISTS "${MANIFEST}")
    message(FATAL_ERROR "uninstall: no install manifest at '${MANIFEST}' (run the install step first)")
endif()

file(STRINGS "${MANIFEST}" installed_files)
set(n_removed 0)
foreach(f IN LISTS installed_files)
    if(f STREQUAL "")
        continue()
    endif()
    set(path "$ENV{DESTDIR}${f}")
    if(IS_SYMLINK "${path}" OR EXISTS "${path}")
        message(STATUS "Removing ${path}")
        file(REMOVE "${path}")
        if(IS_SYMLINK "${path}" OR EXISTS "${path}")
            message(FATAL_ERROR "uninstall: could not remove ${path}")
        endif()
        math(EXPR n_removed "${n_removed} + 1")
    else()
        message(STATUS "Not present: ${path}")
    endif()
endforeach()
message(STATUS "uninstall: removed ${n_removed} file(s)")
