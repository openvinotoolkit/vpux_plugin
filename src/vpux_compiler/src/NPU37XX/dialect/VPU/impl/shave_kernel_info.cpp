//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/shave_kernel_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

Bit VPU::arch37xx::ShaveKernelInfo::getShaveVectorSize() const {
    if (mlir::isa<IE::MVNOp, VPU::MVNOp>(_swOp)) {
        return Bit(128);
    }
    VPUX_THROW("Unsupported operation: {0}", _swOp);
}
