//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUVariantGeneral(VPUASM::DPUVariantOp origVarOp,
                                                                     mlir::OpBuilder& builder,
                                                                     const Logger& /*logger*/) {
    if (auto forceInvReadAttr = origVarOp.getForceInvReadAttr()) {
        builder.create<VPUIPDPU::ForceInvReadOp>(origVarOp.getLoc());
    }

    return mlir::success();
}
