#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if(NOT ENABLE_TESTS)
    set(ENABLE_TESTS OFF)
endif()
ie_dependent_option(ENABLE_TESTS "Unit, behavior and functional tests" ${ENABLE_TESTS} "ENABLE_TESTS" OFF)

ie_dependent_option(ENABLE_VPUX_FUZZ_TESTS "NPU Fuzz tests" OFF "ENABLE_TESTS" OFF)

if(NOT ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()
ie_dependent_option(ENABLE_LTO "Enable Link Time Optimization" ${ENABLE_LTO} "LINUX OR WIN32;NOT CMAKE_CROSSCOMPILING" OFF)

if(NOT ENABLE_FASTER_BUILD)
    set(ENABLE_FASTER_BUILD OFF)
endif()
ie_dependent_option(ENABLE_FASTER_BUILD "Enable build features (PCH, UNITY) to speed up build time" ${ENABLE_FASTER_BUILD} "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)

if(NOT ENABLE_CPPLINT)
    set(ENABLE_CPPLINT OFF)
endif()
ie_dependent_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ${ENABLE_CPPLINT} "UNIX;NOT ANDROID" OFF)

if(NOT ENABLE_CLANG_FORMAT)
    set(ENABLE_CLANG_FORMAT OFF)
endif()
ie_option(ENABLE_CLANG_FORMAT "Enable clang-format checks during the build" ${ENABLE_CLANG_FORMAT})

# HDDL2 is deprecated
if(ENABLE_HDDL2 OR ENABLE_HDDL2_TESTS)
    message (WARNING "ENABLE_HDDL2 and ENABLE_HDDL2_TESTS are deprecated option due to hddl2 removing")
endif()

ie_dependent_option(ENABLE_MODELS "download all models required for functional testing. It requires MODELS_REPO variable set" OFF "ENABLE_FUNCTIONAL_TESTS" OFF)
if(ENABLE_MODELS)
    if(DEFINED ENV{MODELS_REPO})
        set(MODELS_REPO $ENV{MODELS_REPO})
    else()
        message(FATAL_ERROR "ENABLE_MODELS was enabled, but MODELS_REPO was not set. Abort")
    endif()
endif()

# TODO move it out into submodules
ie_dependent_option(ENABLE_VALIDATION_SET "download validation_set required for functional testing" OFF "ENABLE_FUNCTIONAL_TESTS" OFF)

ie_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)

ie_option(ENABLE_MLIR_COMPILER "Enable compilation of npu_mlir_compiler libraries" ON)
if(ENABLE_MLIR_COMPILER)
    add_definitions(-DENABLE_MLIR_COMPILER)
endif()

ie_option(BUILD_COMPILER_FOR_DRIVER "Enable build of VPUXCompilerL0" OFF)

ie_dependent_option(ENABLE_DRIVER_COMPILER_ADAPTER "Enable VPUX Compiler inside driver" ON "NOT BUILD_COMPILER_FOR_DRIVER" OFF)
if(ENABLE_DRIVER_COMPILER_ADAPTER)
    add_definitions(-DENABLE_DRIVER_COMPILER_ADAPTER)
endif()

if(NOT BUILD_SHARED_LIBS AND NOT ENABLE_MLIR_COMPILER AND NOT ENABLE_DRIVER_COMPILER_ADAPTER)
    message(FATAL_ERROR "No compiler found for static build!")
endif()

# if ENABLE_ZEROAPI_BACKEND=ON, it adds the ze_loader dependency for VPUXCompilerL0
ie_dependent_option(ENABLE_ZEROAPI_BACKEND "Enable zero-api as a plugin backend" ON "NOT BUILD_COMPILER_FOR_DRIVER" OFF)
if(ENABLE_ZEROAPI_BACKEND)
    add_definitions(-DENABLE_ZEROAPI_BACKEND)
endif()

ie_option(ENABLE_DEVELOPER_BUILD "Enable developer build with extra validation/logging functionality" OFF)

if(NOT DEFINED MV_TOOLS_PATH AND DEFINED ENV{MV_TOOLS_DIR} AND DEFINED ENV{MV_TOOLS_VERSION})
    set(MV_TOOLS_PATH $ENV{MV_TOOLS_DIR}/$ENV{MV_TOOLS_VERSION})
endif()

ie_dependent_option(ENABLE_IMD_BACKEND "Enable InferenceManagerDemo based VPUX AL backend" OFF "NOT WIN32;NOT CMAKE_CROSSCOMPILING" OFF)
if(ENABLE_IMD_BACKEND)
    add_definitions(-DENABLE_IMD_BACKEND)
endif()

ie_option(ENABLE_BITCOMPACTOR "Enable bitcompactor compression codec" ON)
if(ENABLE_BITCOMPACTOR)
    add_definitions(-DENABLE_BITCOMPACTOR)
endif()

ie_option(ENABLE_HUFFMAN_CODEC "Enable huffman compression codec" ON)
if(ENABLE_HUFFMAN_CODEC)
    add_definitions(-DENABLE_HUFFMAN_CODEC)
endif()

if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    set(TBB_AVAILABLE ON)
endif()
ov_dependent_option(ENABLE_BACKGROUND_FOLDING "Enable constant folding in background" ON "TBB_AVAILABLE" OFF)
if(ENABLE_BACKGROUND_FOLDING)
    add_definitions(-DBACKGROUND_FOLDING_ENABLED)
endif()

ie_option(ENABLE_SOURCE_PACKAGE "Enable generation of source code package" OFF)

ie_option(ENABLE_VPUX_DOCS "Documentation for VPUX plugin" OFF)

ie_option(ENABLE_DIALECT_SHARED_LIBRARIES "Enable exporting vpux dialects as shared libraries" OFF)

if(ENABLE_VPUX_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/docs/VPUX_DG/Doxyfile.in)
        set(DOXYGEN_OUT ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/docs/VPUX_DG/generated/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        add_custom_target(vpux_plugin_docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    else()
        message(WARNING "Doxygen need to be installed to generate the doxygen documentation")
    endif()
endif()

function (print_enabled_kmb_features)
    message(STATUS "KMB Plugin enabled features: ")
    message(STATUS "")
    foreach(var IN LISTS IE_OPTIONS)
        message(STATUS "    ${var} = ${${var}}")
    endforeach()
    message(STATUS "")
endfunction()
