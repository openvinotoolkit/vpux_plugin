#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

if(${CMAKE_VERSION} VERSION_LESS "3.19.0")
    message(WARNING "Source code package will not be generated. "
                    "Miminum CMake version required is 3.19.0")
    return()
endif()


function(read_config_file CONFIG_FILE EXCLUDE_PATTERNS)
    # read json file
    file(READ ${CONFIG_FILE} JSON_CONFIG_STR)

    # utility function to transform a json list to a cmake list
    # and replace ^ in the resulting list's patterns
    function(config_patterns_to_cmake_list CFG_PATTERNS CFG_PATTERNS_LENGTH OUTPUT_LIST)
        set(RESULT_LIST "")
        foreach(IDX RANGE ${CFG_PATTERNS_LENGTH})
            if(NOT ${IDX} STREQUAL ${CFG_PATTERNS_LENGTH})
                string(JSON PATTERN GET ${CFG_PATTERNS} ${IDX})
                string(REPLACE "^" "${CMAKE_CURRENT_SOURCE_DIR}" PATTERN ${PATTERN})
                list(APPEND RESULT_LIST ${PATTERN})
            endif()
        endforeach()

        set(${OUTPUT_LIST} ${RESULT_LIST} PARENT_SCOPE)
    endfunction()

    # read the `exclude_patterns` json variable
    string(JSON CFG_EXCLUDE_PATTERNS GET ${JSON_CONFIG_STR} "ignore_files")
    string(JSON CFG_EXCLUDE_PATTERNS_LEN LENGTH ${JSON_CONFIG_STR} "ignore_files")
    config_patterns_to_cmake_list(
        "${CFG_EXCLUDE_PATTERNS}"
        "${CFG_EXCLUDE_PATTERNS_LEN}"
        CFG_EXCLUDE_PATTERNS_LIST
    )

    list(APPEND CFG_EXCLUDE_PATTERNS_LIST "${CMAKE_CURRENT_BINARY_DIR}")

    # return variables
    set(${EXCLUDE_PATTERNS} ${CFG_EXCLUDE_PATTERNS_LIST} PARENT_SCOPE)
endfunction()

#
# setup source package configuration
#

read_config_file("${CMAKE_CURRENT_SOURCE_DIR}/SourcePackageConfig.json" EXCLUDE_PATTERNS)

set(CPACK_BUILD_SOURCE_DIRS "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}")
set(CPACK_SOURCE_INSTALLED_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR};/")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "CPackSourceConfigVPUX.cmake")
set(CPACK_SOURCE_IGNORE_FILES ${EXCLUDE_PATTERNS})
