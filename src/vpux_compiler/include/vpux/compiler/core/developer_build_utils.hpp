//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

void parseEnv(StringRef envVarName, std::string& var);
void parseEnv(StringRef envVarName, bool& var);

constexpr bool isDeveloperBuild() {
#ifdef VPUX_DEVELOPER_BUILD
    return true;
#else
    return false;
#endif
}

}  // namespace vpux
