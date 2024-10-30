#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_subdirectory(functional)

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

# compilerTest as somke test
add_executable(compilerTest compilerTest.c)
target_link_libraries(compilerTest PUBLIC npu_driver_compiler)
ov_add_api_validator_post_build_step(TARGET compilerTest)

# profilingTest to check profiling API
add_executable(profilingTest profilingTest.c)
target_link_libraries(profilingTest PUBLIC npu_driver_compiler)
ov_add_api_validator_post_build_step(TARGET profilingTest)

# loaderTest to check npu_elf
add_executable(loaderTest loaderTest.cpp)
target_link_libraries(loaderTest PUBLIC npu_elf npu_llvm_utils npu_ov_utils LLVMSupport)
ov_add_api_validator_post_build_step(TARGET loaderTest)

# install tests to CiD component
install(TARGETS compilerTest
        RUNTIME DESTINATION cid
        COMPONENT ${CID_COMPONENT})

install(TARGETS profilingTest
        RUNTIME DESTINATION cid
        COMPONENT ${CID_COMPONENT})

install(TARGETS loaderTest
        RUNTIME DESTINATION cid
        COMPONENT ${CID_COMPONENT})

if(WIN32)
    install(
            FILES $<TARGET_PDB_FILE:compilerTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(
            FILES $<TARGET_PDB_FILE:profilingTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(
            FILES $<TARGET_PDB_FILE:loaderTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
endif()
