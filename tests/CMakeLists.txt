#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(NOT ENABLE_NPU_MONO)
    add_definitions(-DOV_BUILD_POSTFIX=${OV_BUILD_POSTFIX_VAR})
endif()

add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS AND NOT BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(functional)
endif()

add_subdirectory(lit)

if (ENABLE_DIALECT_SHARED_LIBRARIES)
    add_subdirectory(smoke)
endif()

if (ENABLE_NPU_FUZZ_TESTS)
    add_subdirectory(fuzz)
endif()
