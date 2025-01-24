//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEAttr ppeAttr) {
    build(builder, state, ppeAttr, /*ppe_fp_scale=*/nullptr, /*ppe_fp_bias=*/nullptr);
}
