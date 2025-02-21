#
# Copyright (C) 2023-2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

#
# profiling schema target
#

vpux_add_flatc_target(npu_profiling_schema
    SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schema"
    DST_DIR "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/profiling/generated"
    ARGS "--gen-object-api" "--reflect-names")
set_target_properties(npu_profiling_schema PROPERTIES FOLDER "src")

#
# profiling utils
#

set(TARGET_NAME npu_profiling_utils)

list(APPEND SOURCES
                common.cpp
                metadata.cpp
                tasknames.cpp
                parser/debug.cpp
                parser/freq.cpp
                parser/parser.cpp
                parser/sync.cpp
                reports/hooks.cpp
                reports/json.cpp
                reports/layers.cpp
                reports/stats.cpp
                reports/tasklist.cpp
                reports/text.cpp
)

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)

ov_link_system_libraries(${TARGET_NAME}
    PRIVATE
        flatbuffers
        npu_elf
        npu_mlir_compiler_schema
        npu_profiling_schema
)

target_link_libraries(${TARGET_NAME} PRIVATE npu_core_utils npu_llvm_utils)
