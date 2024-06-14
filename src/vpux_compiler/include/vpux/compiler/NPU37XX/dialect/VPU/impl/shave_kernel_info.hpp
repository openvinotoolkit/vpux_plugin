//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/shave_kernel_info.hpp"

namespace vpux::VPU::arch37xx {

class ShaveKernelInfo : public VPU::ShaveKernelInfo {
public:
    ShaveKernelInfo(mlir::Operation* op): VPU::ShaveKernelInfo(op) {
    }

    Bit getShaveVectorSize() const override;
};

}  // namespace vpux::VPU::arch37xx
