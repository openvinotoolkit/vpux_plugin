//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <optional>
#include <string>

namespace vpux::env {

// On Windows CRT getevn/putenv are not reliable across DLL boundary when used in the same process,
// hence these helpers use Win32 API directly bypassing the CRT library.
// As a side-effect std::getenv will not see variables set with setEnvVar within the running process
// and getEnvVar must be used.

/// getEnvVar will return nullopt if variable is empty or does not exist
std::optional<std::string> getEnvVar(const char* name);

inline std::string getEnvVar(const char* name, const char* default_value) {
    return getEnvVar(name).value_or(default_value);
}

/// setEnvVar will override existing variable of the same name
void setEnvVar(const char* name, const char* value);
void unsetEnvVar(const char* name);

}  // namespace vpux::env
