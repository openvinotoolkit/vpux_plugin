# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

cmake_minimum_required(VERSION 3.13.4)

project(kernels)

if(DEFINED TARGET_BINARY_DIR)
  set(target_binary_dir "${TARGET_BINARY_DIR}")
else()
  set(target_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin")
endif()

set(kernel_src_dir_default "src")

set(include_dirs_list_default
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc/3720"
)

set(define_symbols_list_default
  "__act_shave__"
)

set(always_inline_default "no")
set(rt_kernel_default "no")

set(target_prefix "sk.")
set(obj_suffix ".o")
set(elf_suffix ".elf")
set(blob_text_suffix ".text")
set(blob_data_suffix ".data")

set(kernel_descrip_dir "${CUSTOM_KERNELS_DIR}")

if(kernel_descrip_dir STREQUAL "")
  set(kernel_descrip_dir "${CMAKE_CURRENT_SOURCE_DIR}/descrip/")
endif()

option(ENABLE_SHAVE_BINARIES_BUILD "Enable shave binaries build, if disabled, prebuilt binaries will be used" OFF)
option(ENABLE_MANAGEMENT_KERNEL_BUILD "Enable management kernel build" OFF)
option(ENABLE_JTAG_BINARIES_BUILD "Enable jtag binaries build" OFF)

set(use_prebuilt_binaries OFF)

if(ENABLE_JTAG_BINARIES_BUILD)
  set(IE_MAIN_VPUX_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  include(${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/cmake/mv_tools.cmake)
endif()

if(ENABLE_SHAVE_BINARIES_BUILD)
  exists_mv_tools_version(available)
  if(NOT available)
    get_mv_tools()
  endif()

  get_mv_tools_path(MV_TOOLS_PATH)

  if(UNIX)
    set(mv_tools_compile "${MV_TOOLS_PATH}/linux64/bin/moviCompile")
    set(mv_tools_link "${MV_TOOLS_PATH}/linux64/bin/moviLLD")
    set(mv_tools_objcopy "${MV_TOOLS_PATH}/linux64/bin/llvm-objcopy")
  elseif(WIN32)
    set(mv_tools_compile "${MV_TOOLS_PATH}/win32/bin/moviCompile.exe")
    set(mv_tools_link "${MV_TOOLS_PATH}/win64/bin/moviLLD.exe")
    set(mv_tools_objcopy "${MV_TOOLS_PATH}/win64/bin/llvm-objcopy.exe")
  else()
    message(FATAL_ERROR "Unsupported operating system")
  endif()

  set(link_libraries_list_VPU3720
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibm.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibc_lite.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibc_lite_lgpl.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/37xxxx/mlibcrt.a"
  )

  set(link_libraries_list_VPU4000
    "${MV_TOOLS_PATH}/common/moviCompile/lib/4010A0/mlibm.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/4010A0/mlibc_lite.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/4010A0/mlibc_lite_lgpl.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/4010A0/mlibcrt.a"
  )

  set(link_libraries_list_VPU4000_B0
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibm.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibc_lite.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibc_lite_lgpl.a"
    "${MV_TOOLS_PATH}/common/moviCompile/lib/40xxxx/mlibcrt.a"
  )
else()
  set(use_prebuilt_binaries ON)

  if(ENABLE_MANAGEMENT_KERNEL_BUILD)
    message(FATAL_ERROR "When ENABLE_MANAGEMENT_KERNEL_BUILD=ON please use ENABLE_SHAVE_BINARIES_BUILD=ON")
  endif()
endif()

if(ENABLE_MANAGEMENT_KERNEL_BUILD)
  if(ENABLE_NPU_MONO)
    if(DEFINED ENV{FIRMWARE_VPU_DIR})
        message(AUTHOR_WARNING "FIRMWARE_VPU_DIR environment variable is deprecated when ENABLE_NPU_MONO=ON")
    endif()
    if(NOT DEFINED NPU_MONO_FIRMWARE_PROJECT_DIR)
        message(FATAL_ERROR "Firmware project dir is not set while `npu_mono` was activated")
    endif()
    set(FIRMWARE_VPU_DIR "${NPU_MONO_FIRMWARE_PROJECT_DIR}")
  else()
    if(NOT DEFINED ENV{FIRMWARE_VPU_DIR})
        message(FATAL_ERROR "FIRMWARE_VPU_DIR environment variable must be defined when ENABLE_MANAGEMENT_KERNEL_BUILD=ON")
    endif()

    set(FIRMWARE_VPU_DIR "$ENV{FIRMWARE_VPU_DIR}")
  endif()
  list(APPEND include_dirs_list_default "${FIRMWARE_VPU_DIR}/drivers/hardware/utils/inc")
endif()

add_custom_target(act_shave_kernels_ready)
add_custom_target(act_shave_kernels_build)

add_custom_target(act_shave_kernels_copy
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin" "${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin"
)

add_custom_target(update_shave_prebuilt_binaries
  DEPENDS ${act_shave_kernels}
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${target_binary_dir}/" "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/"
)
add_dependencies(update_shave_prebuilt_binaries act_shave_kernels_build)

if(use_prebuilt_binaries)
  add_dependencies(act_shave_kernels_ready act_shave_kernels_copy)
else()
  add_dependencies(act_shave_kernels_ready act_shave_kernels_build update_shave_prebuilt_binaries)
endif()

set(kernel_descrip_list "${CUSTOM_KERNELS_LIST}")

if(NOT kernel_descrip_list STREQUAL "")
  set(kernel_descrip_list ${CUSTOM_KERNELS_LIST})
  list(TRANSFORM kernel_descrip_list PREPEND "${kernel_descrip_dir}")
else()
  set(kernel_descrip_list "")

  file(GLOB kernel_list "${kernel_descrip_dir}*.txt")
  list(APPEND kernel_descrip_list ${kernel_list})
endif()

set(act_shave_kernels "")
file(MAKE_DIRECTORY ${target_binary_dir})

foreach(kernel_descrip ${kernel_descrip_list})
  set(kernel_entry "")
  set(kernel_src "<unknown>")
  set(kernel_src_dir "${kernel_src_dir_default}")
  set(kernel_cpunum "3720")
  set(optimization_opts "-O3")
  set(include_dirs_list "")
  set(define_symbols_list "")
  set(cppflags_list "")
  set(always_inline "${always_inline_default}")
  set(rt_kernel "${rt_kernel_default}")
  set(extra_src_list "")
  set(lsu0_wo "")
  set(B_stepping_only "")
  set(link_script_file "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_kernel.ld")

  get_filename_component(kernel_descrip_path ${kernel_descrip} DIRECTORY)

  include("${kernel_descrip}")
  get_filename_component(kernel_name ${kernel_src} NAME_WE)

  if(kernel_entry STREQUAL "")
    set(kernel_entry "${kernel_name}")
  endif()

  if(kernel_cpunum STREQUAL "3720")
    list(APPEND define_symbols_list "USE_3720_INTSTRUCTIONS")
    set(link_libraries_list ${link_libraries_list_VPU3720})
  elseif(kernel_cpunum STREQUAL "4000")
    list(APPEND define_symbols_list "USE_4000_INTSTRUCTIONS")
    set(link_libraries_list ${link_libraries_list_VPU4000})
    if(NOT B_stepping_only STREQUAL "yes")
      list(APPEND cppflags_list
        "-mllvm"
        "-shave-vpu4-load-store-distance=9"
        "-mllvm"
        "--shave-has-LSU1-fu=false"
      )
    endif()
  endif()

  list(APPEND define_symbols_list "CONFIG_TARGET_SOC_${kernel_cpunum}")

  if(always_inline STREQUAL "yes")
    list(APPEND define_symbols_list "CONFIG_ALWAYS_INLINE")
  endif()

  list(INSERT include_dirs_list 0 ${include_dirs_list_default})
  list(INSERT define_symbols_list 0 ${define_symbols_list_default})

  set(kernel_cpu "${kernel_cpunum}xx")
  set(kernel_cpu_suffix ".${kernel_cpunum}xx")

  set(kernel_src_file "${kernel_src_dir}/${kernel_src}")

  if(lsu0_wo STREQUAL "ON")
    set(blob_text_data_target "${target_prefix}${kernel_name}${kernel_cpu_suffix}_lsu0_wo")
    set(blob_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}_lsu0_wo${blob_text_suffix}")
    set(blob_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}_lsu0_wo${blob_data_suffix}")

    set(obj_file "${kernel_src}${kernel_cpu_suffix}_lsu0_wo${obj_suffix}")
    set(elf_file "${kernel_name}${kernel_cpu_suffix}_lsu0_wo${elf_suffix}")
    list(APPEND cppflags_list
      "-mllvm"
      "-shave-lsu-load-policy=use-only-lsu1"
      "-mllvm"
      "-shave-lsu-store-policy=prefer-lsu0"
    )
  else()
    set(blob_text_data_target "${target_prefix}${kernel_name}${kernel_cpu_suffix}")
    set(blob_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${blob_text_suffix}")
    set(blob_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}${blob_data_suffix}")

    set(obj_file "${kernel_src}${kernel_cpu_suffix}${obj_suffix}")
    set(elf_file "${kernel_name}${kernel_cpu_suffix}${elf_suffix}")
  endif()

  set(elf_path "${target_binary_dir}/${elf_file}")

  if(rt_kernel STREQUAL "yes")
    if(NOT ENABLE_MANAGEMENT_KERNEL_BUILD)
      add_custom_target("${kernel_name}${kernel_cpu_suffix}_kernel_copy"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${target_binary_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${blob_text_file}" "${target_binary_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${blob_data_file}" "${target_binary_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${elf_file}" "${target_binary_dir}"
      )

      add_dependencies(act_shave_kernels_build "${kernel_name}${kernel_cpu_suffix}_kernel_copy")

      # E#118643 : temp workaround to allow of IMD-backend with NPU_STEPPING option
      if(kernel_cpunum STREQUAL "4000")
        set(blob_text_file_B0 "${target_prefix}${kernel_name}${kernel_cpu_suffix}_B0${blob_text_suffix}")
        set(blob_data_file_B0 "${target_prefix}${kernel_name}${kernel_cpu_suffix}_B0${blob_data_suffix}")
        set(elf_file_B0 "${kernel_name}${kernel_cpu_suffix}_B0${elf_suffix}")
        add_custom_target(
          "${kernel_name}${kernel_cpu_suffix}_kernel_copy_b0"
          COMMAND ${CMAKE_COMMAND} -E make_directory "${target_binary_dir}"
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${blob_text_file_B0}" "${target_binary_dir}"
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${blob_data_file_B0}" "${target_binary_dir}"
          COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/act_shave_bin/${elf_file_B0}" "${target_binary_dir}")
        add_dependencies(act_shave_kernels_build "${kernel_name}${kernel_cpu_suffix}_kernel_copy_b0")
      endif()
      # E#118643 workaround end

      continue()
    endif()

    set(kernel_src_path "${kernel_src_file}")
    # ACT management kernel must be built with Shave preemption checks disabled
    set(shave_preemption_opt "-mshave-preemption-checks=off")
    set(link_script_file "${CMAKE_CURRENT_SOURCE_DIR}/prebuild/shave_rt_kernel.ld")
  else()
    set(kernel_src_path "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_src_file}")
    set(shave_preemption_opt "-mshave-preemption-checks=restore")
    list(APPEND shave_preemption_opt "-mshave-low-impact-preemption" )
    list(APPEND shave_preemption_opt "-mshave-preemption-max-loop-depth=1")
  endif()

  list(APPEND act_shave_kernels ${kernel_src_file})

  list(TRANSFORM include_dirs_list PREPEND "-I")
  list(TRANSFORM define_symbols_list PREPEND "-D")

  add_custom_command(
    OUTPUT "${obj_file}"
    DEPENDS "${kernel_src_file}"
    COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${cppflags_list} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${kernel_src_path}" -o "${obj_file}"
  )

  set(obj_file_list "${obj_file}")

  if(NOT extra_src_list STREQUAL "")
    foreach(extra_src_file ${extra_src_list})
      get_filename_component(src_name ${extra_src_file} NAME_WE)
      get_filename_component(dir_name ${extra_src_file} DIRECTORY)

      # Discard full path, just keep name of the dir the file is in
      get_filename_component(dir_name ${dir_name} NAME_WE)

      # Some extra_src files have the same filename so the .o files
      # must be in separate directories to avoid overwriting.
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name})
      set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")

      add_custom_command(
        OUTPUT "${obj_file}"
        DEPENDS "${kernel_src_file}"
        COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${cppflags_list} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${extra_src_file}" -o "${obj_file}"
      )

      list(APPEND obj_file_list "${obj_file}")
    endforeach()
  endif()

  if(kernel_cpunum STREQUAL "4000" AND B_stepping_only STREQUAL "yes")
    add_custom_command(
      OUTPUT "${elf_path}"
      DEPENDS ${obj_file_list}
      COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list} -EL ${link_libraries_list_VPU4000_B0} --output "${elf_path}"
    )
  else()
    add_custom_command(
      OUTPUT "${elf_path}"
      DEPENDS ${obj_file_list}
      COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list} -EL ${link_libraries_list} --output "${elf_path}"
    )
  endif()

  # Delete extra sources after compile such that we can apply different optimizations if needed
  if(NOT extra_src_list STREQUAL "")
    foreach(extra_src_file ${extra_src_list})
      get_filename_component(src_name ${extra_src_file} NAME_WE)
      file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")
    endforeach()
  endif()

  set(blob_text_path "${target_binary_dir}/${blob_text_file}")
  set(blob_data_path "${target_binary_dir}/${blob_data_file}")

  add_custom_target("${blob_text_data_target}"
    COMMAND "${mv_tools_objcopy}" -O binary --only-section=.text "${elf_path}" "${blob_text_path}"
    COMMAND "${mv_tools_objcopy}" -O binary --only-section=.data "${elf_path}" "${blob_data_path}"
    DEPENDS "${elf_path}"
  )

  add_dependencies(act_shave_kernels_build "${blob_text_data_target}")

  # NPU40XX B0 workaround removal binaries
  # E-116865: We also need to support duplicating the same binary for B0 and A0 when workaround disabeled due to compiler limitation
  if(kernel_cpunum STREQUAL "4000")
    set(link_libraries_list ${link_libraries_list_VPU4000_B0})

    set(blob_text_data_target "${target_prefix}${kernel_name}${kernel_cpu_suffix}_B0")
    set(blob_text_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}_B0${blob_text_suffix}")
    set(blob_data_file "${target_prefix}${kernel_name}${kernel_cpu_suffix}_B0${blob_data_suffix}")

    set(obj_file "${kernel_src}${kernel_cpu_suffix}_B0${obj_suffix}")
    set(elf_file "${kernel_name}${kernel_cpu_suffix}_B0${elf_suffix}")
    set(elf_path "${target_binary_dir}/${elf_file}")

    add_custom_command(
      OUTPUT "${obj_file}"
      DEPENDS "${kernel_src_file}"
      COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${kernel_src_path}" -o "${obj_file}"
    )

    set(obj_file_list "${obj_file}")

    if(NOT extra_src_list STREQUAL "")
      foreach(extra_src_file ${extra_src_list})
        get_filename_component(src_name ${extra_src_file} NAME_WE)
        get_filename_component(dir_name ${extra_src_file} DIRECTORY)

        # Discard full path, just keep name of the dir the file is in
        get_filename_component(dir_name ${dir_name} NAME_WE)

        # Some extra_src files have the same filename so the .o files
        # must be in separate directories to avoid overwriting.
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name})
        set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}_B0${obj_suffix}")

        add_custom_command(
          OUTPUT "${obj_file}"
          DEPENDS "${kernel_src_file}"
          COMMAND "${mv_tools_compile}" -mcpu=${kernel_cpu} ${optimization_opts} ${include_dirs_list} ${define_symbols_list} ${shave_preemption_opt} -c "${extra_src_file}" -o "${obj_file}"
        )

        list(APPEND obj_file_list "${obj_file}")
      endforeach()
    endif()

    add_custom_command(
      OUTPUT "${elf_path}"
      DEPENDS ${obj_file_list}
      COMMAND "${mv_tools_link}" --script "${link_script_file}" -entry "${kernel_entry}" --gc-sections --strip-debug --discard-all -zmax-page-size=16 ${obj_file_list} -EL ${link_libraries_list_VPU4000_B0} --output "${elf_path}"
    )

    # Delete extra sources after compile such that we can apply different optimizations if needed
    if(NOT extra_src_list STREQUAL "")
      foreach(extra_src_file ${extra_src_list})
        get_filename_component(src_name ${extra_src_file} NAME_WE)
        file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/extra_src/${kernel_cpu}/${dir_name}/${src_name}.${kernel_cpu}${obj_suffix}")
      endforeach()
    endif()

      set(blob_text_path "${target_binary_dir}/${blob_text_file}")
      set(blob_data_path "${target_binary_dir}/${blob_data_file}")

    add_custom_target("${blob_text_data_target}"
      COMMAND "${mv_tools_objcopy}" -O binary --only-section=.text "${elf_path}" "${blob_text_path}"
      COMMAND "${mv_tools_objcopy}" -O binary --only-section=.data "${elf_path}" "${blob_data_path}"
      DEPENDS "${elf_path}"
    )

    add_dependencies(act_shave_kernels_build "${blob_text_data_target}")
  endif()
endforeach()

# Add file-level and target-level dependencies
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp"
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/embed_shave_binaries.cmake"
  DEPENDS ${act_shave_kernels} act_shave_kernels_ready
)

add_library(act_shave_kernels_lib OBJECT "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp")

# The library contains many large arrays, because of which it compiles slowly
# if the compiler optimisations are enabled.
target_compile_options(act_shave_kernels_lib PRIVATE -O0)

if(ENABLE_JTAG_BINARIES_BUILD)
  include(../jtag_tests/app/layer_tests/test_icv/build/jtag_build.cmake)
  jtag_build()
endif()
