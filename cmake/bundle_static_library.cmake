# MIT License
#
# Copyright (c) 2019 Cristian Adam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# For a static libary ${TARGET_NAME}:
#  * recursively collects static libraries in ${ARGN} targets and their dependencies
#  * bundles all the collected libraries to ${TARGET_NAME}
function(bundle_static_library TARGET_NAME)
    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)
    if (NOT ${TARGET_TYPE} STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "bundle_static_library function should
                             be used only with static libraries.")
        return()
    endif()

    # collect static libs
    set(STATIC_LIBS "")
    set(STATIC_LIBS_TO_BUNDLE ${ARGN})

    function(_recursively_collect_dependencies INPUT_TARGET DEPENDENCIES_TO_INCLUDE)
        set(PUBLIC_DEPENDENCIES "")
        if (DEPENDENCIES_TO_INCLUDE STREQUAL "")
            get_target_property(_INPUT_TARGET_TYPE ${INPUT_TARGET} TYPE)
            if (${_INPUT_TARGET_TYPE} STREQUAL "INTERFACE_LIBRARY")
                get_target_property(PUBLIC_DEPENDENCIES ${INPUT_TARGET} INTERFACE_LINK_LIBRARIES)
            else()
                get_target_property(PUBLIC_DEPENDENCIES ${INPUT_TARGET} LINK_LIBRARIES)
            endif()
        else()
            set(PUBLIC_DEPENDENCIES ${DEPENDENCIES_TO_INCLUDE})
        endif()

        foreach(DEPENDENCY IN LISTS PUBLIC_DEPENDENCIES)
            if(TARGET ${DEPENDENCY})
                # replace with aliased target if needed
                get_target_property(_ALIAS ${DEPENDENCY} ALIASED_TARGET)
                if (TARGET ${_ALIAS})
                    set(DEPENDENCY ${_ALIAS})
                endif()

                # append if a static lib
                get_target_property(_type ${DEPENDENCY} TYPE)
                if (${_type} STREQUAL "STATIC_LIBRARY")
                    list(APPEND STATIC_LIBS ${DEPENDENCY})
                endif()

                # recursive call
                get_property(LIBRARY_ALREADY_ADDED
                    GLOBAL PROPERTY _${TARGET_NAME}_STATIC_BUNDLE_${DEPENDENCY})
                if(NOT LIBRARY_ALREADY_ADDED)
                    set_property(GLOBAL PROPERTY _${TARGET_NAME}_STATIC_BUNDLE_${DEPENDENCY} ON)
                    _recursively_collect_dependencies(${DEPENDENCY} "")
                endif()
            endif()
        endforeach()
        set(STATIC_LIBS ${STATIC_LIBS} PARENT_SCOPE)
    endfunction()

    _recursively_collect_dependencies(${TARGET_NAME} "${STATIC_LIBS_TO_BUNDLE}")
    list(REMOVE_DUPLICATES STATIC_LIBS)

    # bundle static libs into a single archive
    set(TARGET_FULL_NAME $<TARGET_FILE:${TARGET_NAME}>)

    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$")
        file(WRITE ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar.in
            "CREATE ${TARGET_FULL_NAME}\n"
        )

        file(APPEND ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar.in
                "ADDLIB $<TARGET_FILE:${TARGET_NAME}>\n"
        )

        foreach(TARGET IN LISTS STATIC_LIBS)
            file(APPEND ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar.in
                "ADDLIB $<TARGET_FILE:${TARGET}>\n"
            )
        endforeach()

        file(APPEND ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar.in "SAVE\n")
        file(APPEND ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar.in "END\n")

        file(GENERATE
            OUTPUT ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar
            INPUT ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar.in)

        set(AR_TOOL ${CMAKE_AR})
        if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
            set(AR_TOOL ${CMAKE_CXX_COMPILER_AR})
        endif()

        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${AR_TOOL} -M < ${CMAKE_BINARY_DIR}/${TARGET_NAME}.ar
            COMMENT "Bundling ${TARGET_NAME} with ${STATIC_LIBS}"
            VERBATIM)
    elseif(MSVC)
        find_program(lib_tool lib)
        set(STATIC_LIBS_FULL_NAMES "${TARGET_FULL_NAME}")

        foreach(TARGET IN LISTS STATIC_LIBS)
            list(APPEND STATIC_LIBS_FULL_NAMES $<TARGET_FILE:${TARGET}>)
        endforeach()

        # Remove vpux_mlir_dependencies from the list to prevent it from exceeding size limit
        if(${TARGET_NAME} STREQUAL "vpux_mlir_dependencies")
            list(REMOVE_ITEM STATIC_LIBS_FULL_NAMES $<TARGET_FILE:${TARGET_NAME}>)
        endif()

        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${lib_tool} /NOLOGO /OUT:${TARGET_FULL_NAME} ${STATIC_LIBS_FULL_NAMES}
            COMMENT "Bundling ${TARGET_NAME} with ${STATIC_LIBS}"
            VERBATIM
        )
    else()
        message(FATAL_ERROR "Unknown bundle scenario!")
    endif()
endfunction()
