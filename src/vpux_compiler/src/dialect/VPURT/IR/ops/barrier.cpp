//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

mlir::LogicalResult vpux::VPURT::ConfigureBarrierOp::verify() {
    if (!getIsFinalBarrier()) {
        return mlir::success();
    }
    auto barrier = getBarrier();
    auto findConsumerOp = [&]() {
        SmallVector<VPURT::TaskOp> consumerOps;
        for (const auto& user : barrier.getUsers()) {
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(user);
            VPUX_THROW_WHEN(taskOp == nullptr, "VPURT.TaskOp is expected as user for barrier at '{0}'", getLoc());
            auto waitBarriers = taskOp.getWaitBarriers();
            auto iter = llvm::find(waitBarriers, barrier);
            if (iter != waitBarriers.end()) {
                consumerOps.push_back(taskOp);
            }
        }
        return consumerOps;
    };
    auto consumerOps = findConsumerOp();
    if (!consumerOps.empty()) {
        return errorAt(getLoc(), "Final barrier at '{0}' has consumer op '{1}'", getLoc(), consumerOps);
    }
    return mlir::success();
}
