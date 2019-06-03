# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(newContent "        <plugin location=\"${IE_PLUGIN_LIBRARY_NAME}\" optional=\"yes\">")

if(IE_PLUGIN_PROPERTIES)
    set(newContent "${newContent}
            <properties>")

    foreach(props IN LISTS IE_PLUGIN_PROPERTIES)
        string(REPLACE "," ";" props "${props}")

        list(GET props 0 key)
        list(GET props 1 value)
        set(newContent "${newContent}
                <property key=\"${key}\" value=\"${value}\"/>")
    endforeach()

    set(newContent "${newContent}
            </properties>")
endif()

set(newContent "${newContent}
        </plugin>")

file(WRITE "${IE_CONFIG_OUTPUT_FILE}" "${newContent}")
