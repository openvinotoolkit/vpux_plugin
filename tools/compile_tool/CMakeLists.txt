# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME compile_tool)

find_package(gflags QUIET)
# for cross-compilation with gflags
find_package(Threads QUIET)

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_tool_target(
    ADD_CPPLINT
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::ie_samples_utils
        openvino::runtime::dev
        openvino::runtime
        gflags
        Threads::Threads
)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()
