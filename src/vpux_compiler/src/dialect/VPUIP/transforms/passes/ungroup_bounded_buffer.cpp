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

class UngroupConvertDMAOp final : public mlir::OpRewritePattern<VPUIP::ConvertDMAOp> {
public:
    UngroupConvertDMAOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConvertDMAOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConvertDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupConvertDMAOp::matchAndRewrite(VPUIP::ConvertDMAOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getInput());
    auto ungroupOutput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getOutputBuff());

    auto copyData =
            rewriter.create<VPUIP::ConvertDMAOp>(origOp->getLoc(), ungroupInput.getData(), ungroupOutput.getData());
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

    SmallVector<mlir::Value> newResults;
    for (auto i : irange(origOp.getNumResults())) {
        if (mlir::isa<VPUIP::BoundedBufferType>(origOp.getResult(i).getType())) {
            auto groupOp = rewriter.create<VPUIP::GroupBoundedBufferOp>(swKernelOp.getLoc(), swKernelOp.getResult(i),
                                                                        swKernelOp.getDynamicOutputShapes()[i]);
            newResults.push_back(groupOp.getOutput());
        } else {
            newResults.push_back(swKernelOp.getResult(i));
        }
    }
    rewriter.replaceOp(origOp, newResults);

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
    auto isLegalConvertDMAOp = [](VPUIP::ConvertDMAOp ConvertDMAOp) {
        bool areBothOperandsBoundedBuffers = mlir::isa<VPUIP::BoundedBufferType>(ConvertDMAOp.getInput().getType()) &&
                                             mlir::isa<VPUIP::BoundedBufferType>(ConvertDMAOp.getOutput().getType());

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
    target.addDynamicallyLegalOp<VPUIP::ConvertDMAOp>(isLegalConvertDMAOp);
    target.addDynamicallyLegalOp<VPUIP::SwKernelOp>(isLegalSwKernelOp);
    target.addLegalOp<VPUIP::GroupBoundedBufferOp>();
    target.addLegalOp<VPUIP::UngroupBoundedBufferOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UngroupCopyOp>(&ctx, _log);
    patterns.add<UngroupSwKernelOp>(&ctx, _log);
    patterns.add<UngroupConvertDMAOp>(&ctx, _log);

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
