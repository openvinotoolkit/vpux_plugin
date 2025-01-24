//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/network_metadata.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"

namespace vpux::VPUMI37XX {

// E#-140887: replace mlir::ArrayRef<uint8_t> with BlobView
intel_npu::NetworkMetadata getNetworkMetadata(mlir::ArrayRef<uint8_t> blob);

}  // namespace vpux::VPUMI37XX
