//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/al/icompiler.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

namespace vpux::VPUMI37XX {

intel_npu::NetworkMetadata getNetworkMetadata(const std::vector<uint8_t>& blob);

}  // namespace vpux::VPUMI37XX
