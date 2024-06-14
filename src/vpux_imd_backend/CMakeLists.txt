#
# Copyright (C) 2022-2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

ov_dependent_option(IMD_BACKEND_USE_LINKS "Link IMD binaries" OFF "ENABLE_IMD_BACKEND" OFF)

set(TARGET_NAME "npu_imd_backend")
find_package(MLIR REQUIRED CONFIG)

set(IMD_ARTIFACTS_SRC_DIR "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2")
set(IMD_ARTIFACTS_BIN_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/vpux")

if(IMD_BACKEND_USE_LINKS)
    set(ACTION "create_symlink")
    # create_symlink does no create directories automatically
    file(MAKE_DIRECTORY "${IMD_ARTIFACTS_BIN_DIR}/simulator/")
    file(MAKE_DIRECTORY "${IMD_ARTIFACTS_BIN_DIR}/fpga")
    file(MAKE_DIRECTORY "${IMD_ARTIFACTS_BIN_DIR}")
else()
    set(ACTION "copy")
endif()

add_custom_target("${TARGET_NAME}_copy_app" ALL
    COMMAND
        ${CMAKE_COMMAND} -E ${ACTION}
        "${IMD_ARTIFACTS_SRC_DIR}/simulator/InferenceManagerDemo_vpu_2_0.elf"
        "${IMD_ARTIFACTS_BIN_DIR}/simulator/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E ${ACTION}
        "${IMD_ARTIFACTS_SRC_DIR}/fpga/InferenceManagerDemo_vpu_2_0.elf"
        "${IMD_ARTIFACTS_BIN_DIR}/fpga/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E ${ACTION}
        "${IMD_ARTIFACTS_SRC_DIR}/InferenceManagerDemo_vpu_2_0.elf"
        "${IMD_ARTIFACTS_BIN_DIR}/InferenceManagerDemo_vpu_2_0.elf"
    COMMAND
        ${CMAKE_COMMAND} -E ${ACTION}
        "${IMD_ARTIFACTS_SRC_DIR}/InferenceManagerDemo_vpu_2_7.elf"
        "${IMD_ARTIFACTS_BIN_DIR}/InferenceManagerDemo_vpu_2_7.elf"
    COMMAND
        ${CMAKE_COMMAND} -E ${ACTION}
        "${IMD_ARTIFACTS_SRC_DIR}/InferenceManagerDemo_vpu_4.elf"
        "${IMD_ARTIFACTS_BIN_DIR}/InferenceManagerDemo_vpu_4.elf"
    COMMENT "[IMD] Copy InferenceManagerDemo application"
    VERBATIM
)
install(
    FILES
        "${IMD_ARTIFACTS_SRC_DIR}/simulator/InferenceManagerDemo_vpu_2_0.elf"
        "${IMD_ARTIFACTS_SRC_DIR}/fpga/InferenceManagerDemo_vpu_2_0.elf"
        "${IMD_ARTIFACTS_SRC_DIR}/InferenceManagerDemo_vpu_2_0.elf"
        "${IMD_ARTIFACTS_SRC_DIR}/InferenceManagerDemo_vpu_2_7.elf"
        "${IMD_ARTIFACTS_SRC_DIR}/InferenceManagerDemo_vpu_4.elf"
    DESTINATION "${OV_CPACK_RUNTIMEDIR}/vpux"
    COMPONENT    ${VPUX_PLUGIN_COMPONENT}
)

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} MODULE ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TARGET_NAME} "${TARGET_NAME}_copy_app")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})


target_compile_definitions(${TARGET_NAME}
    PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_PLUGIN
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        openvino::runtime::dev
        openvino::runtime
        LLVMSupport
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
)

install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR}
    COMPONENT ${VPUX_INTERNAL_COMPONENT}
)
