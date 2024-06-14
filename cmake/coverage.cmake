#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

if(NOT VPUX_CODE_COVERAGE)
    return()
endif()

if("${VPUX_CODE_COVERAGE}" STREQUAL "GCOV")
    find_program(GCOV_PATH gcov)

    if(NOT GCOV_PATH)
        message(FATAL_ERROR "GCOV not found in path")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "[Cc]lang")
        if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 3)
            message(FATAL_ERROR "Clang version must be 3.0.0 or greater! Aborting...")
        endif()
    elseif(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -ftest-coverage" CACHE INTERNAL "")

    if(NOT CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
        message(WARNING "Code coverage results with an optimised (non-Debug) build may be misleading")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        link_libraries(gcov)
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()
endif()

message(STATUS "Code coverage enabled")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILER_FLAGS}")
