#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_conversion")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS
    PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_ov_utils
        npu_llvm_utils
        npu_profiling_utils
        npu_elf
        flatbuffers
        # E#98178: remove depedency on firmware headers
        npu_37xx_firmware_headers
        npu_40xx_firmware_headers
        npu_compiler_dialect_npureg40xx
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_vpurt
    PUBLIC
        mlir-dependencies)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
        ${VPU_COMPILER_BIN_INCLUDE_DIR}
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>)
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
add_src_target(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
