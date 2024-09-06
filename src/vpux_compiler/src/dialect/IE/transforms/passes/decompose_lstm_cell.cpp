//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// LSTMCellRewriter
//

class LSTMCellRewriter final : public mlir::OpRewritePattern<IE::LSTMCellOp> {
public:
    LSTMCellRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::LSTMCellOp>(ctx), _log(log) {
        this->setDebugName("LSTMCellRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewriter::matchAndRewrite(IE::LSTMCellOp lstmCell, mlir::PatternRewriter& rewriter) const {
    if (VPU::LSTMCellOp::isSupported(lstmCell)) {
        return mlir::failure();
    }
    _log.trace("Got op {0} at {1}", lstmCell->getName(), lstmCell->getLoc());

    auto matMulInputOp = rewriter.create<IE::MatMulOp>(takeOpLoc(lstmCell, "in_mul"), lstmCell.getInputData(),
                                                       lstmCell.getWeights(), false, true);
    auto matMulHiddenStateOp =
            rewriter.create<IE::MatMulOp>(takeOpLoc(lstmCell, "mul_hid"), lstmCell.getInitialHiddenState(),
                                          lstmCell.getRecurrenceWeights(), false, true);

    auto biasesAddOp = rewriter.create<IE::AddOp>(
            takeOpLoc(lstmCell, "bias"), matMulInputOp.getOutput(), lstmCell.getBiases(),
            IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY), nullptr, nullptr);
    auto lstmGatesInputOp = rewriter.create<IE::AddOp>(
            takeOpLoc(lstmCell, "gates"), biasesAddOp.getOutput(), matMulHiddenStateOp.getOutput(),
            IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT), nullptr, nullptr);

    rewriter.replaceOpWithNewOp<IE::LSTMGatesOp>(lstmCell, lstmGatesInputOp.getOutput(),
                                                 lstmCell.getInitialCellState());

    return mlir::success();
}

//
// DecomposeLSTMCellPass
//

class DecomposeLSTMCellPass final : public IE::DecomposeLSTMCellBase<DecomposeLSTMCellPass> {
public:
    explicit DecomposeLSTMCellPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeLSTMCellPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LSTMCellRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeLSTMCellPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeLSTMCellPass(Logger log) {
    return std::make_unique<DecomposeLSTMCellPass>(log);
}
