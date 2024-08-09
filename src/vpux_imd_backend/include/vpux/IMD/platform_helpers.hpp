//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "npu_private_properties.hpp"
#include "vpux/utils/IE/private_properties.hpp"

#include "vpux/utils/core/string_ref.hpp"

namespace intel_npu {

bool platformSupported(const std::string_view platform);
llvm::StringRef getAppName(const std::string_view platform);

}  // namespace intel_npu
