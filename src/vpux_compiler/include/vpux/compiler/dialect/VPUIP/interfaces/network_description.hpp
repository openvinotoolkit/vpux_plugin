//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/al/icompiler.hpp"

namespace vpux::VPUIP {

intel_npu::NetworkMetadata getNetworkMetadata(const std::vector<uint8_t>& blob);

}  // namespace vpux::VPUIP
