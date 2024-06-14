//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FuseFQAndMul
//

class FuseFQAndMul final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    FuseFQAndMul(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isLhsActivation(IE::MultiplyOp multiplyOp) {
    const auto lhsType = multiplyOp.getInput1().getType().cast<mlir::ShapedType>();
    const auto outShapeRes = multiplyOp.getOutput().getType().cast<mlir::ShapedType>();

    return (lhsType == outShapeRes);
}

// This pass comes from the front-end ngrap pass. Openvion's mo will convert some FakeQuantizeOps on Weight in the
// original model into FQ + ADD + MUL. This pass is used to merge these MULs into FQ, so only the case of const weight
// FQ is handled. And currently, only per channel FQ is handled, because the FQ of per element is not yet supported
// #E73317

bool isLegalToFuse(IE::MultiplyOp multiplyOp) {
    bool lhsIsActivation = isLhsActivation(multiplyOp);

    auto fakeQuantOp = lhsIsActivation ? multiplyOp.getInput1().getDefiningOp<IE::FakeQuantizeOp>()
                                       : multiplyOp.getInput2().getDefiningOp<IE::FakeQuantizeOp>();
    if (fakeQuantOp == nullptr) {
        return false;
    }
    if (!fakeQuantOp->hasOneUse()) {
        return false;
    }
    auto constWeigthOp = fakeQuantOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (constWeigthOp == nullptr) {
        return false;
    }

    auto mulConstOp = lhsIsActivation ? multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>()
                                      : multiplyOp.getInput1().getDefiningOp<Const::DeclareOp>();

    if (mulConstOp == nullptr) {
        return false;
    }

    auto mulConstShape = getShape(mulConstOp.getOutput());

    // Only handle per-channel weight case.
    if (mulConstShape[Dims4D::Filter::IC] != 1 || mulConstShape[Dims4D::Filter::KX] != 1 ||
        mulConstShape[Dims4D::Filter::KY] != 1) {
        return false;
    }

    return true;
}

/*
      data  in_L in_H out_L out_H
        |    |    |     |     |
        |    |    |     |     |                data  in_L in_H  out_L * C  out_H * C
        v    v    v     v     v                  |    |    |        |          |
      +-------------------------+                |    |    |        |          |
      |       FakeQuantize      |                v    v    v        v          v
      +-------------------------+             +-----------------------------------+
                   |                =====>    |            FakeQuantize           |
                   v                          +-----------------------------------+
              +----------+                                      |
              | Multiply | <--- C                               v
              +----+-----+
                   |
                   v
*/

mlir::LogicalResult FuseFQAndMul::matchAndRewrite(IE::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const {
    if (!isLegalToFuse(multiplyOp)) {
        return mlir::failure();
    }
    bool lhsIsActivation = isLhsActivation(multiplyOp);

    auto fakeQuantOp = lhsIsActivation ? multiplyOp.getInput1().getDefiningOp<IE::FakeQuantizeOp>()
                                       : multiplyOp.getInput2().getDefiningOp<IE::FakeQuantizeOp>();
    auto mulConstOp = lhsIsActivation ? multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>()
                                      : multiplyOp.getInput1().getDefiningOp<Const::DeclareOp>();
    auto mulConstShape = getShape(mulConstOp.getOutput());

    auto mulConstVal = IE::getConst(mulConstOp);

    auto inLowConst = fakeQuantOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fakeQuantOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fakeQuantOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fakeQuantOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        _log.trace("Got non constant parameters of FakeQuantize '{0}'", fakeQuantOp->getLoc());
        return mlir::failure();
    }

    _log.trace("Fuse mul '{0}' into FQ '{1}'", multiplyOp->getLoc(), fakeQuantOp->getLoc());

    const auto outLowContent = outLowConst.getContent();
    auto outLowVals = SmallVector<float>(outLowContent.getValues<float>());
    const auto outHighContent = outHighConst.getContent();
    auto outHighVals = SmallVector<float>(outHighContent.getValues<float>());

    auto fqOutValsCount = outLowVals.size();

    for (size_t idx = 0; idx < fqOutValsCount; idx++) {
        outHighVals[idx] *= mulConstShape[Dims4D::Filter::OC] == 1 ? mulConstVal[0] : mulConstVal[idx];
        outLowVals[idx] *= mulConstShape[Dims4D::Filter::OC] == 1 ? mulConstVal[0] : mulConstVal[idx];
    }

    auto newOutHighConst = VPU::createFloatConst(outHighConst.getType().cast<mlir::RankedTensorType>(),
                                                 ArrayRef(outHighVals), fakeQuantOp.getLoc(), rewriter);
    auto newOutLowConst = VPU::createFloatConst(outLowConst.getType().cast<mlir::RankedTensorType>(),
                                                ArrayRef(outLowVals), fakeQuantOp.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(multiplyOp, fakeQuantOp.getInput(), fakeQuantOp.getInputLow(),
                                                    fakeQuantOp.getInputHigh(), newOutLowConst, newOutHighConst,
                                                    fakeQuantOp.getLevelsAttr(), fakeQuantOp.getLowFpTypeAttr(),
                                                    fakeQuantOp.getAutoBroadcastAttr());

    return mlir::success();
}

//
// FuseFQAndMulPass
//

class FuseFQAndMulPass final : public IE::FuseFQAndMulBase<FuseFQAndMulPass> {
public:
    explicit FuseFQAndMulPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseFQAndMulPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseFQAndMul>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseFQAndMulPass(Logger log) {
    return std::make_unique<FuseFQAndMulPass>(log);
}
