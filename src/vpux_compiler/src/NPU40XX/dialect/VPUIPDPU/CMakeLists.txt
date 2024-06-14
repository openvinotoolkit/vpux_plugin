#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_vpuipdpu_40XX")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList npu_compiler_vpuipdpu_40XX)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
add_src_target(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
