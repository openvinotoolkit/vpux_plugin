#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME "npu-lsp-server")
find_package(MLIR REQUIRED CONFIG)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ENABLE_WARNINGS_AS_ERRORS
    LINK_LIBRARIES
         MLIRLspServerLib
         npu_mlir_compiler_static
)
