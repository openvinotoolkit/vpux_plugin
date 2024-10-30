//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEAttr opaque_ppe) {
    build(builder, state, opaque_ppe, /*ppe_fp_scale=*/nullptr, /*ppe_fp_bias=*/nullptr);
}
