#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elfnpu37xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf
        openvino::npu_al
        inferenceStatic
        # E#98178: remove depedency on firmware headers
        npu_37xx_firmware_headers
        npu_compiler_dialect_vpurt
        npu_profiling_schema)
