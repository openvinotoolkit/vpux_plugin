#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_npu37xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf
        MLIRIR
        MLIRFuncDialect)
