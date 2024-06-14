#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

cmake_policy(SET CMP0054 NEW)

include(ExternalProject)

if(NOT BUILD_SHARED_LIBS)
    set(TEMP "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/temp")
else()
    ov_set_temp_directory(TEMP "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}")
endif()

# FIXME: Create empty file to avoid errors on CI
file(TOUCH "${CMAKE_BINARY_DIR}/ld_library_rpath_64.txt")

#
# Models and Images for tests
#

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=${MODELS_PATH}")

set(DATA_PATH "${TEMP}/validation_set/src/validation_set")
debug_message(STATUS "DATA_PATH=${DATA_PATH}")

if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ")
endif()
