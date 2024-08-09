#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npu_ngraph_transformations")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

ov_build_target_faster(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

enable_warnings_as_errors(${TARGET_NAME})

if(MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /wd4996     # code uses a function, class member, variable, or typedef that's marked deprecated
    )
endif()

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ov_link_system_libraries(${TARGET_NAME}
    PUBLIC
        openvino::reference
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_ov_utils
        npu_llvm_utils
)
