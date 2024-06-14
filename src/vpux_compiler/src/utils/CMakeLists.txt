#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_utils")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_algo_utils
        npu_ov_utils
        npu_llvm_utils
        npu_profiling_utils
        MLIRIR
        npu_compiler_dialect_vpu
        flatbuffers
        npu_elf
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)
add_dependencies(${TARGET_NAME} intrinsics_gen)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty>
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME}
    PRIVATE
        bitcompactor)
add_src_target(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
