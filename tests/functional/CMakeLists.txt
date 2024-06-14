#
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME npuFuncTests)
set(EXCLUDED_FUNC_TESTS_DIR "")
set(OPTIONAL_FUNC_TESTS_INCLUDES "")
set(OPTIONAL_FUNC_TESTS_LIBS "")

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_FUNC_TESTS_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPTIONAL_FUNC_TESTS_INCLUDES}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/shared_tests_instances"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/behavior"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/subgraph_tests"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/custom/single_layer_tests/classes"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_translate_utils/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_test_classes/include"
        ${LLVM_INCLUDE_DIRS}
    LINK_LIBRARIES
        ${OPTIONAL_FUNC_TESTS_LIBS}
        openvino::func_test_utils
        openvino::funcSharedTests
        openvino::format_reader
        openvino::reference
        openvino::runtime
        npu_model_utils
        npu_llvm_utils
        npu_ov_utils
        npu_plugin_utils
        openvino::npu_al
        npu_mlir_compiler_schema # used by subgraph_tests/compress_weights_btc.cpp
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                      CXX_STANDARD 17)

if(ENABLE_MLIR_COMPILER)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MLIR_COMPILER)
endif()

if(MSVC)
    # Enforce standards conformance on MSVC
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /Zc:preprocessor
    )
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

# TODO: Remove duplication E#31024
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)

if(ENABLE_DEVELOPER_BUILD AND UNIX)
    # add npuFuncTest improved with gtest-parallel to ctest
    # we are using run_gtest_parallel_manual_threads.sh script to provide
    # PARALLEL_WORKERS env variable as -w workers command line argument
    add_test(NAME ${TARGET_NAME}-parallel
            COMMAND
                ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/scripts/run_gtest_parallel_manual_threads.sh
                ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/gtest-parallel/gtest-parallel
                $<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
            set_tests_properties(${TARGET_NAME}-parallel PROPERTIES
            LABELS "NPU;PlatformSpecific;3720;4000"
    )
endif()
