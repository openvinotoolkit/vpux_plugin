# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(LINUX)
    # Add single-image-test to ctest
    find_package(Python3 QUIET)
    if(NOT Python3_FOUND)
            message(WARNING "Python3 is not found, SingleImageTests can't be included to ctest")
    else()
        add_test(NAME SingleImageTests
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/single_image_test/single_image_test.py
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
            set_tests_properties(SingleImageTests PROPERTIES
                LABELS "NPU;PlatformSpecific;3720;4000"
            )
    endif()
endif()
