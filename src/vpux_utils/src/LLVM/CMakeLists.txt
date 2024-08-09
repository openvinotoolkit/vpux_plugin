#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

find_package(ZLIB)

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()
set(TARGET_NAME npu_llvm_utils)

list(APPEND SOURCES
                ../core/error.cpp
                ../core/logger.cpp
                ../core/mask.cpp
                ../core/mem_size.cpp
                ../core/simple_math.cpp
)

if(WIN32)
  list (APPEND SOURCES
                    ../core/win32/env.cpp
                    ../core/win32/memory_usage.cpp)
else()
  list (APPEND SOURCES
                    ../core/unix/env.cpp 
                    ../core/unix/memory_usage.cpp)
endif()

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${LLVM_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${LLVM_BINARY_DIR}/include>
)

add_dependencies(${TARGET_NAME}
    LLVMSupport
)

if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET_NAME} PRIVATE npu_core_utils PUBLIC LLVMSupport)
else()
    bundle_static_library(${TARGET_NAME} npu_core_utils LLVMSupport)
endif()
