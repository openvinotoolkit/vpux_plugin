#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#[[
Wrapper function over ov_add_target, that also the checks the tool's dependencies and
installs the tool's executable.
You could use in the following way:
ov_add_test_target(NAME targetName
                   ROOT ${CMAKE_CURRENT_SOURCE_DIR}
                   COMPONENT cpack_component # default to `tests`
                   INSTALL_DESTINATION install/destination # defaults to `tools`
                   ENABLE_WARNINGS_AS_ERRORS
                   INCLUDES includeOne IncludeTwo
                   LINK_LIBRARIES libOne libTwo
                   DEPENDENCIES depOne depTwo
)
Important: pay attention to the multivalued arguments like LINK_LIBRARIES,
otherwise any parameters that come after might be consumed.
#]]
function(add_tool_target)
    #
    # set up and parse options
    #

    set(options
        ENABLE_WARNINGS_AS_ERRORS
        ADD_CPPLINT                   # Enables code style checks for the target
    )

    set(oneValueRequiredArgs
        NAME
        ROOT
    )

    set(oneValueOptionalArgs
        COMPONENT
        INSTALL_DESTINATION
    )

    set(multiValueArgs
        LINK_LIBRARIES
        DEPENDENCIES
    )

    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN})

    # set up some default values
    if(NOT DEFINED ARG_COMPONENT)
        set(ARG_COMPONENT tests)
    endif()
    if(NOT DEFINED ARG_INSTALL_DESTINATION)
        set(ARG_INSTALL_DESTINATION "tools/${ARG_NAME}")
    endif()

    #
    # check for missing dependencies
    #

    set(MISSING_DEPENDENCIES "")
    foreach(LIB ${ARG_LINK_LIBRARIES} ${ARG_DEPENDENCIES})
        if(NOT TARGET ${LIB})
            list(APPEND MISSING_DEPENDENCIES ${LIB})
        endif()
    endforeach()

    if(NOT MISSING_DEPENDENCIES STREQUAL "")
        message(WARNING "${TARGET_NAME} tool is disabled due to missing dependencies: ${MISSING_DEPENDENCIES}")
        return()
    endif()

    #
    # define the target
    #

    if (ARG_ADD_CPPLINT)
        ov_add_target(ADD_CPPLINT
                      TYPE EXECUTABLE
                      NAME ${ARG_NAME}
                      ROOT ${ARG_ROOT}
                      DEPENDENCIES ${ARG_DEPENDENCIES}
                      LINK_LIBRARIES ${ARG_LINK_LIBRARIES}
                      ${ARG_UNPARSED_ARGUMENTS})
    else()
        ov_add_target(TYPE EXECUTABLE
                      NAME ${ARG_NAME}
                      ROOT ${ARG_ROOT}
                      DEPENDENCIES ${ARG_DEPENDENCIES}
                      LINK_LIBRARIES ${ARG_LINK_LIBRARIES}
                      ${ARG_UNPARSED_ARGUMENTS})
    endif()

    set_target_properties(${ARG_NAME} PROPERTIES
                          FOLDER ${ARG_ROOT}
                          CXX_STANDARD 17)

    if(ARG_ENABLE_WARNINGS_AS_ERRORS)
        enable_warnings_as_errors(${ARG_NAME} WIN_STRICT)
    endif()

    #
    # install the target
    #

    install(TARGETS ${ARG_NAME}
            RUNTIME DESTINATION ${ARG_INSTALL_DESTINATION}
            COMPONENT ${ARG_COMPONENT}
            EXCLUDE_FROM_ALL)

    if(EXISTS "${ARG_ROOT}/README.md")
        install(FILES "${ARG_ROOT}/README.md"
                DESTINATION ${ARG_INSTALL_DESTINATION}
                COMPONENT ${ARG_COMPONENT}
                EXCLUDE_FROM_ALL)

        # TODO: Remove duplication E#31024
        install(FILES "${ARG_ROOT}/README.md"
                DESTINATION ${ARG_INSTALL_DESTINATION}
                COMPONENT ${VPUX_TESTS_COMPONENT}
                EXCLUDE_FROM_ALL)
    endif()

    # TODO: Remove duplication E#31024
    install(TARGETS ${ARG_NAME}
            RUNTIME DESTINATION ${ARG_INSTALL_DESTINATION}
            COMPONENT ${VPUX_TESTS_COMPONENT}
            EXCLUDE_FROM_ALL
    )

endfunction()
