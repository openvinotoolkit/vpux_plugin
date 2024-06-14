#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

find_package(ZLIB)

set(TARGET_NAME npu_algo_utils)

list(APPEND SOURCES
                simulated_annealing.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/algorithm
)

add_dependencies(${TARGET_NAME}
    npu_llvm_utils
)

target_link_libraries(${TARGET_NAME} PRIVATE npu_llvm_utils)
