//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>
#include <vpux/utils/core/error.hpp>
#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// UngroupBoundedBuffers
//

class UngroupCopyOp final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    UngroupCopyOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupCopyOp::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getInput());
    auto ungroupOutput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getOutputBuff());

    auto copyData = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), ungroupInput.getData(), ungroupOutput.getData());
    auto copyShape = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), ungroupInput.getDynamicShape(),
                                                    ungroupOutput.getDynamicShape());
    rewriter.replaceOpWithNewOp<VPUIP::GroupBoundedBufferOp>(origOp, copyData.getOutput(), copyShape.getOutput());

    return mlir::success();
}

class UngroupSwKernelOp final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    UngroupSwKernelOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupSwKernelOp::matchAndRewrite(VPUIP::SwKernelOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> swKernelOperands;
    SmallVector<mlir::Value> swKernelDynamicInputShapes;
    SmallVector<int32_t> swKernelDynamicInputShapesMap;

    for (auto input : origOp.getInputs()) {
        auto boundedInput = mlir::dyn_cast<VPUIP::BoundedBufferType>(input.getType());
        if (boundedInput != nullptr) {
            auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), input);
            swKernelOperands.push_back(ungroupInput.getData());
            swKernelDynamicInputShapesMap.push_back(swKernelDynamicInputShapes.size());
            swKernelDynamicInputShapes.push_back(ungroupInput.getDynamicShape());
        } else {
            swKernelOperands.push_back(input);
            swKernelDynamicInputShapesMap.push_back(ABSENT_DIMS_FLAG);
        }
    }

    VPUX_THROW_WHEN(origOp.getOutputBuffs().size() != 1 || origOp.getOutputs().size() != 1,
                    "UngroupBoundedBuffers pass supports SwKernelOp with single output for now");

    SmallVector<mlir::Value> swKernelOutputBuffs;
    SmallVector<mlir::Value> swKernelDynamicOutputShapes;
    SmallVector<int32_t> swKernelDynamicOutputShapesMap;

    for (auto outputBuff : origOp.getOutputBuffs()) {
        auto boundedOutputBuff = mlir::dyn_cast<VPUIP::BoundedBufferType>(outputBuff.getType());
        if (boundedOutputBuff != nullptr) {
            auto ungroupOutputBuff = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), outputBuff);
            swKernelOutputBuffs.push_back(ungroupOutputBuff.getData());
            swKernelDynamicOutputShapesMap.push_back(swKernelDynamicOutputShapes.size());
            swKernelDynamicOutputShapes.push_back(ungroupOutputBuff.getDynamicShape());
        } else {
            swKernelOutputBuffs.push_back(outputBuff);
            swKernelDynamicOutputShapesMap.push_back(ABSENT_DIMS_FLAG);
        }
    }

    auto tileIndex = origOp.getTileIndexAttr();
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp->getLoc(), swKernelOperands, swKernelOutputBuffs,
                                                         swKernelDynamicInputShapes, swKernelDynamicInputShapesMap,
                                                         swKernelDynamicOutputShapes, swKernelDynamicOutputShapesMap,
                                                         origOp.getKernelFunction(), tileIndex);

    auto args = kernelArgsRange(origOp);
    initSwKernel(swKernelOp, swKernelOperands, swKernelOutputBuffs, args, _log.nest());

    for (auto result : origOp.getResults()) {
        if (result.getType().isa<VPUIP::BoundedBufferType>()) {
            // if result is BoundedBufferType, then it is guaranteed to have swKernelDynamicOutputShapes not empty
            // Tracking number [E#115679]
            // TODO: SwKernelOp's results 0 and 1 may not always be correct here, only in the case where
            // there is only one bounded result. Similarly, using replaceOp here when there are more than
            // one result will not behave as intended.
            auto groupOp = rewriter.create<VPUIP::GroupBoundedBufferOp>(swKernelOp.getLoc(), swKernelOp.getResult(0),
                                                                        swKernelOp.getResult(1));
            rewriter.replaceOp(origOp, groupOp.getOutput());
        } else {
            rewriter.replaceOp(origOp, swKernelOp.getResults());
        }
    }

    return mlir::success();
}

class UngroupBoundedBuffers final : public VPUIP::UngroupBoundedBuffersBase<UngroupBoundedBuffers> {
public:
    explicit UngroupBoundedBuffers(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UngroupBoundedBuffers::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalCopyOp = [](VPUIP::CopyOp copyOp) {
        bool areBothOperandsBoundedBuffers = copyOp.getInput().getType().isa<VPUIP::BoundedBufferType>() &&
                                             copyOp.getOutput().getType().isa<VPUIP::BoundedBufferType>();

        return !areBothOperandsBoundedBuffers;
    };
    auto isLegalSwKernelOp = [](VPUIP::SwKernelOp op) {
        const auto isBoundedBuffer = [](mlir::Value value) {
            return value.getType().isa<VPUIP::BoundedBufferType>();
        };
        const auto hasDynamicInputs = llvm::any_of(op.getInputs(), isBoundedBuffer);
        const auto hasDynamicOutputs = llvm::any_of(op.getOutputBuffs(), isBoundedBuffer);

        return !hasDynamicInputs && !hasDynamicOutputs;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPUIP::CopyOp>(isLegalCopyOp);
    target.addDynamicallyLegalOp<VPUIP::SwKernelOp>(isLegalSwKernelOp);
    target.addLegalOp<VPUIP::GroupBoundedBufferOp>();
    target.addLegalOp<VPUIP::UngroupBoundedBufferOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UngroupCopyOp>(&ctx, _log);
    patterns.add<UngroupSwKernelOp>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUngroupBoundedBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBoundedBuffersPass(Logger log) {
    return std::make_unique<UngroupBoundedBuffers>(log);
}
