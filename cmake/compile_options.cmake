#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

# put flags allowing dynamic symbols into target
macro(replace_compile_visibility_options)
    # Replace compiler flags
    foreach(flag IN ITEMS "-fvisibility=default" "-fvisibility=hidden" "-rdynamic" "-export-dynamic")
        string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
    endforeach()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -rdynamic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -export-dynamic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -export-dynamic")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -export-dynamic")
endmacro()

macro(replace_noerror TARGET_NAME)
    # TODO(E#78994): better way to wrap up code which uses deprecated declarations
    if(NOT MSVC)
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wno-error=deprecated-declarations
        )
    endif()
    # TODO(E#83264): consider making it enabled
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wno-error=covered-switch-default
        )
    endif()
endmacro()

if(MSVC)
    # Wile cmake default is /Zi OV, overrides /Zi with /Z7
    # We need /Z7 to avoid pdb creation issues with Ninja build and/or ccache
    # Add /debug:fastlink to link step to avoid pdb files exceeding 4GB limit
    # Note with fastlink object files are required for full debug information!
    foreach(link_flag_var
        CMAKE_EXE_LINKER_FLAGS_DEBUG
        CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
        CMAKE_MODULE_LINKER_FLAGS_DEBUG
        CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO
        CMAKE_SHARED_LINKER_FLAGS_DEBUG
        CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO
        )
        string(REGEX REPLACE "/debug" "/debug:fastlink" ${link_flag_var} "${${link_flag_var}}")
    endforeach()

    # Optimize global data
    add_compile_options(/Zc:inline /Gw)
    # Use compiler intrinsincs
    add_compile_options(/Oi)

endif()

function(enable_warnings_as_errors TARGET_NAME)

    cmake_parse_arguments(WARNIGS "WIN_STRICT" "" "" ${ARGN})

    if(MSVC)
        # Enforce standards conformance on MSVC
        target_compile_options(${TARGET_NAME}
            PRIVATE
                /permissive-
        )

        if(WARNIGS_WIN_STRICT)
            # Use W3 instead of Wall, since W4 introduces some hard-to-fix warnings
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /WX /W3 /wd4244 /wd4267 /wd4293
                    # TODO(E#86977): check and fix warnings to avoid error c2220
            )

            # Disable 3rd-party components warnings
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /experimental:external /external:anglebrackets /external:W0
            )
        endif()
    else()
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wall -Wextra -Werror -Werror=suggest-override
        )
    endif()
endfunction()

macro(set_llvm_flags)
    set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
    set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR ENABLE_DEVELOPER_BUILD)
        set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
    else()
        set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "" FORCE)
    endif()
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
    set(CROSS_TOOLCHAIN_FLAGS_ "" CACHE STRING "" FORCE)
    set(CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING "" FORCE)
    set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)
    # we do not use examples and having it enabled
    # makes cmake complains about long path on Windows
    set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
endmacro()

macro(enable_split_dwarf)
    if ((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            add_compile_options(-gsplit-dwarf)
            if (COMMAND check_linker_flag)
                check_linker_flag(CXX "-Wl,--gdb-index" LINKER_SUPPORTS_GDB_INDEX)
                append_if(LINKER_SUPPORTS_GDB_INDEX "-Wl,--gdb-index"
                CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
            endif()
            set(LLVM_USE_SPLIT_DWARF ON)
        endif()
    endif()
endmacro()
