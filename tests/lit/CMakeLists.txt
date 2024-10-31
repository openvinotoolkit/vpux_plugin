#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

vpux_setup_lit_tool(flatc)

if(ENABLE_NPU_MONO)
    if(NOT DEFINED GRAPHFILE_SCHEMA_SUBMODULE_PATH)
        message(FATAL_ERROR "Graphfile schema submodule path is not set while `npu_mono` was activated")
    endif()
    set(SCHEMA_SOURCE_DIR ${GRAPHFILE_SCHEMA_SUBMODULE_PATH}/src/schema)
else()
    set(SCHEMA_SOURCE_DIR ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/elf/src/schema) # Legacy path
endif(ENABLE_NPU_MONO)

add_custom_target(copy_vpuip_schema_tests ALL
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory
            "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/schema"
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
            "${SCHEMA_SOURCE_DIR}"
            "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/schema"
    COMMENT "[LIT] Copy VPUIP schema files for tests"
)
set_target_properties(copy_vpuip_schema_tests PROPERTIES FOLDER "tests")

set(vpuip_schema_file "path('../schema/graphfile.fbs')")

add_custom_target(copy_lit_tests_script ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/scripts/run_all_lit_tests.sh"
            "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/run_all_lit_tests.sh"
        COMMENT "[LIT] Copy runner script"
)

set(data_path_npu "path('NPU/data')")

vpux_setup_lit_tests(NPU
    PATTERNS "*.mlir" "*.json"
    EXTRA_SOURCES "lit.local.cfg" "*.bin" "*.txt" "*.xml"
    VARS "vpuip_schema_file" "data_path_npu"
    PARAMS "arch"
    PARAMS_DEFAULT_VALUES "NPU37XX"
    SUBSTITUTIONS "vpuip_schema_file" "data_path_npu"
)

set(_lit_tests_dir_full_path "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests")

install(PROGRAMS "${_lit_tests_dir_full_path}"
        DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
