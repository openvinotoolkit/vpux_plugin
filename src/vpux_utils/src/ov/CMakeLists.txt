#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

find_package(ZLIB)

set(TARGET_NAME npu_ov_utils)

list(APPEND SOURCES
        config.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/IE/
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/LLVM/
)

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(${TARGET_NAME} PUBLIC
        $<TARGET_PROPERTY:openvino::itt,INTERFACE_COMPILE_DEFINITIONS>)

ov_link_system_libraries(${TARGET_NAME}
    PUBLIC
        openvino::runtime
        openvino::runtime::dev
)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
)

target_link_libraries(${TARGET_NAME} PRIVATE npu_core_utils npu_llvm_utils)
