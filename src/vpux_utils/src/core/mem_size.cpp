//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/mem_size.hpp"

using namespace vpux;

StringLiteral vpux::stringifyEnum(MemType val) {
    switch (val) {
    case MemType::Bit:
        return "Bit";
    case MemType::Byte:
        return "Byte";
    case MemType::KB:
        return "KB";
    case MemType::MB:
        return "MB";
    case MemType::GB:
        return "GB";
    default:
        return "<UNKNOWN>";
    }
}
