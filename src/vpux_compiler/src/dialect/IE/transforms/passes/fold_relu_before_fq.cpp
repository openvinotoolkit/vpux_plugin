//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FoldReLUBeforeFQ
//

class FoldReLUBeforeFQ final : public mlir::OpRewritePattern<IE::ReLUOp> {
public:
    FoldReLUBeforeFQ(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReLUOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReLUOp reluOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FoldReLUBeforeFQ::matchAndRewrite(IE::ReLUOp reluOp, mlir::PatternRewriter& rewriter) const {
    for (auto user : reluOp.getResult().getUsers()) {
        auto fakeQuantOp = mlir::dyn_cast<IE::FakeQuantizeOp>(user);
        if (fakeQuantOp == nullptr) {
            return mlir::failure();
        }

        auto inputLowConst = fakeQuantOp.getInputLow().getDefiningOp<Const::DeclareOp>();
        if (inputLowConst == nullptr) {
            return mlir::failure();
        }

        auto inputLowContent = inputLowConst.getContent();
        auto inputLowValues = inputLowContent.getValues<float>();

        auto hasNegativeInputLowVals = std::any_of(inputLowValues.begin(), inputLowValues.end(), [](float val) {
            return val < 0;
        });
        if (hasNegativeInputLowVals) {
            return mlir::failure();
        }
    }

    _log.nest().trace("Folded ReLU at '{0}'", reluOp.getLoc());
    rewriter.replaceOp(reluOp, reluOp.getInput());

    return mlir::success();
}

//
// FoldReLUBeforeFQPass
//

class FoldReLUBeforeFQPass final : public IE::FoldReLUBeforeFQBase<FoldReLUBeforeFQPass> {
public:
    explicit FoldReLUBeforeFQPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FoldReLUBeforeFQPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldReLUBeforeFQ>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFoldReLUBeforeFQPass(Logger log) {
    return std::make_unique<FoldReLUBeforeFQPass>(log);
}
