#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npuUnitTests")
set(EXCLUDED_UNIT_TESTS_DIR "")
set(OPTIONAL_UNIT_TESTS_INCLUDES "")
set(OPTIONAL_UNIT_TESTS_LIBS "")
set(MANDATORY_UNIT_TESTS_LIBS
    "npu_37xx_firmware_headers"
    "npu_40xx_firmware_headers"
    "npu_elf"
    "openvino::commonTestUtils"
    "openvino::gmock"
    "openvino::runtime"
    "npu_model_utils"
    "openvino::npu_al"
    "npu_algo_utils"
    "npu_llvm_utils"
    "npu_ov_utils"
    "npu_profiling_utils"
    "npu_plugin_utils"
    "openvino::npu_logger_utils"
)

# Skip tests for plugin targets
set(ENABLE_DRIVER_COMPILER_ADAPTER OFF)
set(ENABLE_ZEROAPI_BACKEND OFF)
list(APPEND EXCLUDED_UNIT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vpux_plugin")

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    list(APPEND OPTIONAL_UNIT_TESTS_INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_driver_compiler_adapter"
    )

    list(APPEND OPTIONAL_UNIT_TESTS_LIBS
            npu_driver_compiler_adapter
        )
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_driver_compiler_adapter"
    )
endif()

if(ENABLE_ZEROAPI_BACKEND)
   list(APPEND OPTIONAL_UNIT_TESTS_LIBS
       ze_loader
   )
endif()

if(NOT ENABLE_MLIR_COMPILER)
    list(APPEND EXCLUDED_UNIT_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_compiler"
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_translate_utils/hwtest/vpux37xx/pss"
    )
else()
    list(APPEND MANDATORY_UNIT_TESTS_LIBS
        "npu_mlir_compiler_static"
        "npu_translate_utils_static"
    )
endif()


ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_UNIT_TESTS_DIR}
    INCLUDES
        ${OPTIONAL_UNIT_TESTS_INCLUDES}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_translate_utils/include"
        ${CMAKE_CURRENT_SOURCE_DIR}/vpux_compiler
    LINK_LIBRARIES
        ${OPTIONAL_UNIT_TESTS_LIBS}
        ${MANDATORY_UNIT_TESTS_LIBS}
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        VPUX_PLUGIN
)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tests")

enable_warnings_as_errors(${TARGET_NAME})

# FIXME: E78489 gtest does not respect this option yet in some parts
# so we cannot enable it for npuUnitTests

if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-suggest-override)
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)
