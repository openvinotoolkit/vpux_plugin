#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME "npu_compiler_frontend")
if(BUILD_COMPILER_FOR_DRIVER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        string(REPLACE "/sdl" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/sdl" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    endif()
endif()


file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf
        npu_compiler_dialect_vpuip
    INCLUDES
        $<TARGET_PROPERTY:openvino::npu_al,INTERFACE_INCLUDE_DIRECTORIES>)

ov_link_system_libraries(${TARGET_NAME}
    PUBLIC
        npu_ngraph_transformations)
