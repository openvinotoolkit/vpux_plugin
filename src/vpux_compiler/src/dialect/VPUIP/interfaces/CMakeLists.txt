#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_dialect_vpuip_interfaces")
add_mlir_dialect_library(${TARGET_NAME}
    STATIC
        ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_compiler_act_kernels
        npu_compiler_core
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        flatbuffers
        npu_elf
        openvino::npu_al
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_graph_schema)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        ${VPUNN_INCLUDE_DIRS}
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
add_src_target(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
ov_add_version_defines(${VPUX_COMPILER_VERSION_FILE} ${TARGET_NAME})
