//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/env.hpp"
#include "vpux/utils/core/error.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <processenv.h>

#include <memory>

namespace vpux::env {

std::optional<std::string> getEnvVar(const char* name) {
    DWORD size = GetEnvironmentVariableA(name, NULL, 0);
    if (size <= 1) {  // 1 is returned for empty variable
        return std::nullopt;
    }
    auto buf = std::unique_ptr<char[]>(new char[size]);
    size = GetEnvironmentVariableA(name, buf.get(), size);
    VPUX_THROW_WHEN(size == 0, "GetEnvironmentVariable failed");
    return buf.get();
}

void setEnvVar(const char* name, const char* value) {
    auto ret = SetEnvironmentVariableA(name, value);
    VPUX_THROW_WHEN(ret == 0, "SetEnvironmentVariable failed");
}

void unsetEnvVar(const char* name) {
    auto ret = SetEnvironmentVariableA(name, nullptr);
    VPUX_THROW_WHEN(ret == 0, "SetEnvironmentVariable failed");
}

}  // namespace vpux::env
