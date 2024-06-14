#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_act_kernels")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf
        MLIRIR)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
        ${VPU_COMPILER_BIN_INCLUDE_DIR}
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>)
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
target_sources(${TARGET_NAME} PRIVATE $<TARGET_OBJECTS:act_shave_kernels_lib>)
add_src_target(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
