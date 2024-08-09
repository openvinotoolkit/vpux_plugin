//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/common_string_utils.hpp"

#include <algorithm>
#include <cstdarg>
#include <stdexcept>

namespace vpux {

std::string printFormattedCStr(const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    std::va_list apCopy;
    va_copy(apCopy, ap);
    const auto requiredBytes = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (requiredBytes < 0) {
        va_end(apCopy);
        throw std::runtime_error(std::string("vsnprintf got error: ") + strerror(errno) + ", fmt: " + fmt);
    }
    std::string out(requiredBytes, 0);  // +1 implicitly
    vsnprintf(out.data(), requiredBytes + 1, fmt, apCopy);
    va_end(apCopy);
    return out;
}
}  // namespace vpux
