#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

find_package(ZLIB)

set(TARGET_NAME npu_core_utils)

list (APPEND SOURCES
            common_string_utils.cpp
            common_logger.cpp
            exceptions.cpp
            type/float8_e4m3.cpp
            type/float8_e5m2.cpp
            type/float16.cpp
            type/bfloat16.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} OBJECT ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)
target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        ${LLVM_SOURCE_DIR}/include
        $<BUILD_INTERFACE:${LLVM_BINARY_DIR}/include> # llvm/Config/llvm-config.h
)

if(NOT ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    add_dependencies(${TARGET_NAME}
        LLVMSupport
    )
endif()

