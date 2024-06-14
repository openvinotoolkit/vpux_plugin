//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/env.hpp"
#include "vpux/utils/core/error.hpp"

#include <stdlib.h>

namespace vpux::env {

std::optional<std::string> getEnvVar(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == 0) {
        return std::nullopt;
    }
    return value;
}

void setEnvVar(const char* name, const char* value) {
    auto ret = setenv(name, value, true);
    VPUX_THROW_WHEN(ret != 0, "setenv failed");
}

void unsetEnvVar(const char* name) {
    auto ret = unsetenv(name);
    VPUX_THROW_WHEN(ret != 0, "setenv failed");
}

}  // namespace vpux::env
