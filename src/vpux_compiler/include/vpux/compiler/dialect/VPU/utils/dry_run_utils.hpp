//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/options.hpp"

#include <string>

namespace vpux::VPU {

enum class DPUDryRunMode { NONE, STUB, STRIP };

DPUDryRunMode getDPUDryRunMode(std::string mode);
DPUDryRunMode getDPUDryRunMode(const StrOption& option);
bool isDPUDryRunEnabled(const StrOption& option);

}  // namespace vpux::VPU
