//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Various string manipulation utility functions.
//

#pragma once

#include "vpux/utils/core/containers.hpp"

#include <string.h>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace vpux {

std::string printFormattedCStr(const char* fmt, ...)
#if defined(__clang__)
        ;
#elif defined(__GNUC__) || defined(__GNUG__)
        __attribute__((format(printf, 1, 2)));
#else
        ;
#endif

}  // namespace vpux
