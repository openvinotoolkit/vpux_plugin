#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME "npu_compiler_utils")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
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
        npu_compiler_dialect_vpurt
    DEPENDS
        intrinsics_gen
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty>)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        bitcompactor)
