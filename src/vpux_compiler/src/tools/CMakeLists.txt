#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_tools
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils)
