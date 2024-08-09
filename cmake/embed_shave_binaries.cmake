# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

# Creates C resources file from files in given directory
function(create_resources dir output)
    # Create empty output file
    file(WRITE ${output} "")

    file(WRITE ${output}
        "#include <utility>\n"
        "#include <unordered_map>\n"
        "#include <cstdint>\n"
        "#include <string>\n"
        "\n"
        "namespace {\n"
    )

    set(map_sym
        "std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveBinaryResourcesMap {\n"
    )

    # Iterate through input files
    file(GLOB bins ${dir}/*)
    foreach(bin ${bins})
        # Get short filename, replace spaces and extension separator
        string(REGEX MATCH "([^/]+)$" filename ${bin})
        string(REGEX REPLACE "\\.| |-" "_" filename ${filename})
        # Read and convert hex data for C compatibility
        file(READ ${bin} filedata HEX)
        string(LENGTH "${filedata}" hex_string_length)
        if(${hex_string_length} GREATER 0)
            string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
            # Append data to output file
            file(APPEND ${output} "static const uint8_t ${filename}[] = {${filedata}};\n")
            string(APPEND map_sym "{\"${filename}\", {${filename}, sizeof(${filename})}},\n")
        endif()
    endforeach()

    string(APPEND map_sym "}\;\n")
    file(APPEND ${output} "} // namespace\n")
    file(APPEND ${output} ${map_sym})
endfunction()

# Embed all the binaries from act_shave_bin folder into generated_shave_binary_resources.cpp
create_resources("${CMAKE_CURRENT_BINARY_DIR}/act_shave_bin" "${CMAKE_CURRENT_BINARY_DIR}/generated_shave_binary_resources.cpp")
