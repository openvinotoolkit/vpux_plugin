//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// DequantizeConst
//

class DequantizeConst final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeConst(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp dCastOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeConst::matchAndRewrite(IE::DequantizeOp dCastOp, mlir::PatternRewriter& rewriter) const {
    auto inputConst = dCastOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (inputConst == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got DequantizeCast Operation '{0}' with Constant input '{1}'", dCastOp->getLoc(), inputConst.getLoc());

    const auto qType = inputConst.getType().cast<vpux::NDTypeInterface>();
    const auto qElemType = qType.getElementType().cast<mlir::quant::QuantizedType>();

    const auto outType = dCastOp.getType().cast<vpux::NDTypeInterface>();
    const auto newConstType = outType.changeElemType(qElemType.getExpressedType());
    const auto newConstAttr = inputConst.getContentAttr().dequantize();
    rewriter.replaceOpWithNewOp<Const::DeclareOp>(dCastOp, newConstType, newConstAttr)->setLoc(inputConst->getLoc());

    return mlir::success();
}

//
// DequantizeConstPass
//

class DequantizeConstPass final : public IE::DequantizeConstBase<DequantizeConstPass> {
public:
    explicit DequantizeConstPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DequantizeConstPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeConst>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDequantizeConstPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDequantizeConstPass(Logger log) {
    return std::make_unique<DequantizeConstPass>(log);
}
