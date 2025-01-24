//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/task_rewriter.hpp"

#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"

namespace vpux::vpuip2vpumi40xx {

mlir::LogicalResult VPURTTaskRewriter::matchAndRewrite(VPURT::TaskOp taskOp, OpAdaptor adaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    for (auto executable : taskOp.getOps<VPUMI40XX::ExecutableTaskOpInterface>()) {
        if (auto waitBarriers = adaptor.getWaitBarriers(); !waitBarriers.empty()) {
            executable.waitBarriersMutable().assign(waitBarriers);
        }
        if (auto updateBarriers = adaptor.getUpdateBarriers(); !updateBarriers.empty()) {
            executable.updateBarriersMutable().assign(updateBarriers);
        }
        if (auto enqueueBarrier = adaptor.getEnqueueBarrier(); enqueueBarrier) {
            executable.enqueueBarrierMutable().assign(enqueueBarrier);
        }
    }

    rewriter.inlineBlockBefore(&taskOp.getRegion().getBlocks().front(), taskOp);
    rewriter.eraseOp(taskOp);
    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
