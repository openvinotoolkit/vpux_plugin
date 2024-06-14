//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

bool hasMultiSwKernelRun(VPUIP::SwKernelOp swKernelOp) {
    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    return std::distance(swKernelRun.begin(), swKernelRun.end()) > 1;
}

SmallVector<mlir::Value> getOuterMostMappingOperand(VPUIP::SwKernelRun swKernelRun) {
    auto swKernelOp = swKernelRun->getParentOfType<VPUIP::SwKernelOp>();
    VPUX_THROW_WHEN(swKernelOp == nullptr, "Cannot find VPUIP.SwKernelOp at '{0}'", swKernelRun->getLoc());

    SmallVector<mlir::Value> outerMostOperands;

    for (auto operand : swKernelRun->getOperands()) {
        auto blockArg = operand.dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(blockArg == nullptr, "Matching argument was not identified");
        auto outerOperand = swKernelOp->getOperand(blockArg.getArgNumber());
        outerMostOperands.push_back(outerOperand);
    }
    return outerMostOperands;
}

bool isOperandFromList(mlir::ValueRange rangeList, mlir::Value operand) {
    return llvm::find(rangeList, operand) != rangeList.end();
}

VPUIP::SwProfilingMetadataAttr getUpdatedSwProfilingMetadataAttr(VPUIP::SwProfilingMetadataAttr attr, size_t tileId,
                                                                 std::optional<size_t> maybeClusterId) {
    const size_t bufferId = attr.getBufferId().getInt();
    const size_t bufferOffset = attr.getBufferOffset().getInt();
    const size_t clusterSize = attr.getClusterSize().getInt();
    const size_t dataIndex = attr.getDataIndex().getInt();
    return vpux::getSwProfilingMetaAttr(attr.getContext(), bufferId, bufferOffset, clusterSize, dataIndex + tileId,
                                        tileId, maybeClusterId);
}

//
// SwKernelRewriter
//

class SwKernelRewriter : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const override;
    SwKernelRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log), _ctx(ctx) {
        setDebugName("SwKernelRewriter");
    }

    bool needUnroll(VPUIP::SwKernelOp swKernelOp) const;
    VPURT::TaskOp createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                  VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter, size_t index) const;

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
};

mlir::LogicalResult SwKernelRewriter::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                      mlir::PatternRewriter& rewriter) const {
    if (!needUnroll(swKernelOp)) {
        return mlir::failure();
    }

    auto vpurtTask = swKernelOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    rewriter.setInsertionPointAfter(vpurtTask);
    auto swKernelRunList = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    for (auto swKernelRunTuple : swKernelRunList | indexed) {
        auto newTaskOp =
                createNewTaskOp(swKernelOp, swKernelRunTuple.value(), vpurtTask, rewriter, swKernelRunTuple.index());
        _log.trace("create new task op: {0}", newTaskOp);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

bool SwKernelRewriter::needUnroll(VPUIP::SwKernelOp swKernelOp) const {
    auto hasMultiSwKernelRunFlag = hasMultiSwKernelRun(swKernelOp);
    return hasMultiSwKernelRunFlag;
}

VPURT::TaskOp SwKernelRewriter::createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                                VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                                size_t index) const {
    auto opLoc = swKernelOp->getLoc();
    auto outerOperand = getOuterMostMappingOperand(swKernelRun);
    auto iter = llvm::find_if(outerOperand, [&](auto operand) {
        return isOperandFromList(swKernelOp.getOutputBuffs(), operand);
    });
    VPUX_THROW_WHEN(iter == outerOperand.end(), "Cannot find operand for output buffer at '{0}'", opLoc);

    auto outBufferStartIndex = std::distance(outerOperand.begin(), iter);
    auto newInputs = SmallVector<mlir::Value>(outerOperand.begin(), outerOperand.begin() + outBufferStartIndex);
    auto newOutBuffers = SmallVector<mlir::Value>(outerOperand.begin() + outBufferStartIndex, outerOperand.end());

    mlir::Value newProfilingBuffer = nullptr;
    VPUIP::SwProfilingMetadataAttr maybeProfMeta = nullptr;
    if (auto profilingBuffer = swKernelOp.getProfilingData()) {
        // In case task has profiling buffer as SwKernelOp operand, each individual SwKernelRun will be given
        // its own chunk of this buffer. Buffer size was properly prepared by act-shave-profiling pass with
        // enough space for each SwKernelRun inside
        VPUX_THROW_WHEN(swKernelOp.getProfilingMetadataAttr() == nullptr, "Missed profilingMetadata for '{0}'",
                        swKernelOp);
        maybeProfMeta =
                getUpdatedSwProfilingMetadataAttr(swKernelOp.getProfilingMetadataAttr(), index, /*clusterId=*/0);
        auto profilingBufferDecl = profilingBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        auto profilingBufferNDType = profilingBuffer.getType().cast<vpux::NDTypeInterface>();

        int64_t numEl =
                VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES / vpux::Byte(profilingBufferNDType.getElemTypeSize()).count();
        if (auto distrProfilingType = profilingBuffer.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            numEl *= distrProfilingType.getDistribution().getNumClusters().getInt();
        }

        const auto newMemType = profilingBufferNDType.changeShape({numEl});
        auto newProfilingBufferDecl = rewriter.create<VPURT::DeclareBufferOp>(
                profilingBufferDecl->getLoc(), newMemType, profilingBufferDecl.getSectionAttr(),
                profilingBufferDecl.getSectionIndexAttr(),
                getIntAttr(_ctx,
                           profilingBufferDecl.getByteOffset() + VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES * index),
                nullptr);

        newProfilingBuffer = newProfilingBufferDecl.getBuffer();
    }
    opLoc = appendLoc(opLoc, "tile_{0}", index);

    auto newSwKernelOp = VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
            rewriter, origTaskOp.getWaitBarriers(), origTaskOp.getUpdateBarriers(), opLoc, newInputs, newOutBuffers,
            newProfilingBuffer, swKernelOp.getKernelFunctionAttr(), swKernelOp.getTileIndexAttr(),
            swKernelOp.getStridesAttr());
    if (maybeProfMeta != nullptr) {
        newSwKernelOp.setProfilingMetadataAttr(maybeProfMeta);
    }
    VPUIP::initSwKernel(newSwKernelOp, swKernelRun, _log);
    return newSwKernelOp->getParentOfType<VPURT::TaskOp>();
}

//
// UnrollSwKernelPass
//

class UnrollSwKernelPass final : public VPUIP::UnrollSwKernelBase<UnrollSwKernelPass> {
public:
    explicit UnrollSwKernelPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSwKernelPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwKernelRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createUnrollSwKernelPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollSwKernelPass(Logger log) {
    return std::make_unique<UnrollSwKernelPass>(log);
}
