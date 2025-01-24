#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_act_kernels
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_elf
        MLIRIR
        MLIRFuncDialect
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
    TARGET_SOURCES PRIVATE
        $<TARGET_OBJECTS:act_shave_kernels_lib>)
