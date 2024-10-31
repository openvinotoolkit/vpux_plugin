//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/STLExtras.h>

using namespace vpux;

namespace {

// enqueue = [var1, var2(brealingPoint), var3, var4, var5]
// we are going to replace it in two enqueues
// lhsEnqueue = [var1, var2] and rhsEnqueue = [var3, var4, var5]

class SplitEnqueueOpsPass : public VPUMI40XX::SplitEnqueueOpsBase<SplitEnqueueOpsPass> {
public:
    explicit SplitEnqueueOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitEnqueueOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto mpi = VPUMI40XX::getMPI(netFunc);
    for (auto enqueueOp : llvm::make_early_inc_range(netFunc.getOps<VPURegMapped::EnqueueOp>())) {
        auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getStart().getDefiningOp());
        auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getEnd().getDefiningOp());
        if (enqueueOp.getStart() == enqueueOp.getEnd()) {
            continue;
        }
        vpux::VPURegMapped::TaskOpInterface breakingPoint = nullptr;
        auto taskOp = end;
        do {
            taskOp = taskOp.getPreviousTask();
            if (taskOp->hasAttr(VPUMI40XX::lastSecondaryTaskInExecutionGroup)) {
                breakingPoint = taskOp;
                break;
            }
        } while (taskOp != start);

        if (breakingPoint == nullptr) {
            continue;
        }

        mlir::OpBuilder builder(enqueueOp);
        builder.setInsertionPointAfter(enqueueOp.getOperation());
        auto lastEnqueue = enqueueOp.getPreviousTaskIdx();
        auto lhsEnqueue = builder.create<VPURegMapped::EnqueueOp>(start.getLoc(), enqueueOp.getType(), lastEnqueue,
                                                                  enqueueOp.getBarrier(), enqueueOp.getTaskType(),
                                                                  start.getResult(), breakingPoint.getResult());

        auto rhsStart = VPUMI40XX::getNextOp(breakingPoint);
        auto rhsEnqueue = builder.create<VPURegMapped::EnqueueOp>(
                start.getLoc(), enqueueOp.getType(), lhsEnqueue.getResult(), enqueueOp.getBarrier(),
                enqueueOp.getTaskType(), rhsStart.getResult(), end.getResult());

        enqueueOp.getResult().replaceUsesWithIf(lhsEnqueue, [](mlir::OpOperand& operand) {
            return mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
        });

        enqueueOp.getResult().replaceUsesWithIf(rhsEnqueue, [](mlir::OpOperand& operand) {
            return !mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
        });

        enqueueOp->erase();
    }
    auto firstEnqu = mpi.getWorkItemTasks();
    auto newCount = VPUMI40XX::reindexEnqueueList(mlir::cast<VPURegMapped::EnqueueOp>(firstEnqu.getDefiningOp()));
    mpi.setWorkItemCount(newCount);
}

}  // namespace

//
// createSplitEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createSplitEnqueueOpsPass(Logger log) {
    return std::make_unique<SplitEnqueueOpsPass>(log);
}
