#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpu_interfaces
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_utils
        npu_llvm_utils
        flatbuffers
        npu_elf
        npu_compiler_dialect_vpu
        #TODO:#-95278 remove dependency on VPUIP
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_interfaces
        npu_llvm_utils
        npu_ov_utils
        inferenceStatic
    SYSTEM_INCLUDES
        ${VPUNN_INCLUDE_DIRS})
