#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(BITCOMPACTOR_TARGET_NAME "bitcompactor")

add_library(${BITCOMPACTOR_TARGET_NAME}
    OBJECT
        "src/BitStream.cpp"
        "src/Encoder.cpp"
)

target_include_directories(${BITCOMPACTOR_TARGET_NAME}
    PRIVATE
        "include"
)
