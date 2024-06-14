//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// RemoveU16FakeQuantizeRewriter
//

class RemoveU16FakeQuantizeRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    RemoveU16FakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("RemoveU16FakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveU16FakeQuantizeRewriter::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto levels = origOp.getLevels();

    // Maximum number of levels that don't exceeds I8/U8 storage type
    if (levels <= MAX_LEVELS) {
        return mlir::failure();
    }

    // In case the FakeQuantize is per tensor and the input and output low is equal to 0 it is replaced with a ReLu
    // activation function otherwise the FakeQuantize is completely removed
    if (IE::isPerTensorFQ({origOp})) {
        const auto inLowValue = IE::getConst(origOp.getInputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto outLowValue = IE::getConst(origOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
        const auto inHighValue = IE::getConst(origOp.getInputHigh().getDefiningOp<Const::DeclareOp>())[0];
        const auto outHighValue = IE::getConst(origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];
        if (isFloatEqual(inLowValue, outLowValue) && isFloatEqual(inHighValue, outHighValue) &&
            isFloatEqual(inLowValue, 0.0f)) {
            rewriter.replaceOpWithNewOp<IE::ReLUOp>(origOp, origOp.getInput());
            return mlir::success();
        }
    }

    rewriter.replaceOp(origOp, origOp.getInput());
    return mlir::success();
}

//
// HandleU16FakeQuantizePass
//

class HandleU16FakeQuantizePass final : public IE::HandleU16FakeQuantizeBase<HandleU16FakeQuantizePass> {
public:
    explicit HandleU16FakeQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleU16FakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveU16FakeQuantizeRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleU16FakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleU16FakeQuantizePass(Logger log) {
    return std::make_unique<HandleU16FakeQuantizePass>(log);
}
