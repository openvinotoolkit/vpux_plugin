#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_dialect_vpu")
add_mlir_dialect_library(${TARGET_NAME}
    STATIC
        ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf
        #TODO:#-95278 remove dependency on VPUIP
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpuip_utils
        #TODO:#-102154 remove this dependency
        npu_compiler_dialect_vpu_utils
        inferenceStatic)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        ${VPUNN_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})

add_src_target(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
