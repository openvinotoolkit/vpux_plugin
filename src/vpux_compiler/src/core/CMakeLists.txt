#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_core
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_compiler_utils
        npu_ov_utils
        npu_llvm_utils
        npu_profiling_utils
        MLIRAsyncDialect
        MLIRControlFlowInterfaces
        MLIRViewLikeInterface
        flatbuffers
        npu_elf
        npu_compiler_dialect_vpurt
        npu_profiling_schema
    SYSTEM_INCLUDES
        ${VPUNN_INCLUDE_DIRS})
