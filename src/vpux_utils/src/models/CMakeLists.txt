#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#


if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npu_model_utils")

file(GLOB_RECURSE SOURCES *.cpp)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/models

)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        openvino::runtime::dev
        npu_ov_utils
        npu_llvm_utils
)

target_link_libraries(${TARGET_NAME} PRIVATE npu_llvm_utils npu_ov_utils)
