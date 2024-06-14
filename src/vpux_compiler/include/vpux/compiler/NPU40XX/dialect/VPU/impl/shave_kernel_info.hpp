//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/shave_kernel_info.hpp"

namespace vpux::VPU::arch40xx {

class ShaveKernelInfo : public VPU::arch37xx::ShaveKernelInfo {
public:
    ShaveKernelInfo(mlir::Operation* op): VPU::arch37xx::ShaveKernelInfo(op) {
    }

    Bit getShaveVectorSize() const override;
};

}  // namespace vpux::VPU::arch40xx
