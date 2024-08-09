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

using namespace vpux;

namespace {
class SplitEnqueuePattern final : public mlir::OpRewritePattern<VPURegMapped::EnqueueOp> {
public:
    SplitEnqueuePattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURegMapped::EnqueueOp>(ctx), _log(log) {
        setDebugName("SplitEnqueuePatternRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPURegMapped::EnqueueOp enqueueOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// enqueue = [var1, var2(brealingPoint), var3, var4, var5]
// we are going to replace it in two enqueues
// lhsEnqueue = [var1, var2] and rhsEnqueue = [var3, var4, var5]

mlir::LogicalResult SplitEnqueuePattern::matchAndRewrite(VPURegMapped::EnqueueOp enqueueOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getStart().getDefiningOp());
    auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getEnd().getDefiningOp());
    if (enqueueOp.getStart() == enqueueOp.getEnd()) {
        return mlir::failure();
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
        return mlir::failure();
    }

    auto lastEnqueue = enqueueOp.getPreviousTaskIdx();
    auto lhsEnqueue = rewriter.create<VPURegMapped::EnqueueOp>(start.getLoc(), enqueueOp.getType(), lastEnqueue,
                                                               enqueueOp.getBarrier(), enqueueOp.getTaskType(),
                                                               start.getResult(), breakingPoint.getResult());

    auto rhsStart = VPUMI40XX::getNextOp(breakingPoint);
    auto rhsEnqueue = rewriter.create<VPURegMapped::EnqueueOp>(
            start.getLoc(), enqueueOp.getType(), lhsEnqueue.getResult(), enqueueOp.getBarrier(),
            enqueueOp.getTaskType(), rhsStart.getResult(), end.getResult());

    enqueueOp.getResult().replaceUsesWithIf(lhsEnqueue, [](mlir::OpOperand& operand) {
        return mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
    });

    enqueueOp.getResult().replaceUsesWithIf(rhsEnqueue, [](mlir::OpOperand& operand) {
        return !mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
    });

    rewriter.eraseOp(enqueueOp.getOperation());
    return mlir::success();
}

class SplitEnqueueOpsPass : public VPUMI40XX::SplitEnqueueOpsBase<SplitEnqueueOpsPass> {
public:
    explicit SplitEnqueueOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitEnqueueOpsPass::safeRunOnFunc() {
    auto ctx = &getContext();
    auto netFunc = getOperation();

    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto firstEnqu = mpi.getWorkItemTasks();
    // if we don't have workItems skip an iteration over the whole IR
    if (!firstEnqu)
        return;

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<SplitEnqueuePattern>(ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(netFunc, std::move(patterns),
                                                        vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    // pattern rewriter may have changed the firstEnqu
    firstEnqu = mpi.getWorkItemTasks();
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
