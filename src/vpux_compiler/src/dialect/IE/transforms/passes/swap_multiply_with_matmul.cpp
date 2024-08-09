//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

//
// SwapMultiplyWithMatmul
//

//                  (1x32x1024x80)           (1x1x1x1)
//                              \               /
//      (1x32x1x80)         IE.Multiply (1x32x1024x80)
//               \            /
//            IE.Matmul(1x32x1x1024)

// To

//        (1x32x1x80)    1x32x1024x80)
//              \            /
//             IE.Matmul(1x32x1x1024)       (1x1x1x1)
//                         \                   /
//                        IE.Multiply (1x32x1x1024)

class SwapMultiplyWithMatmul final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    SwapMultiplyWithMatmul(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("SwapMultiplyWithMatmul");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isBeneficalToSwap(IE::MultiplyOp multiplyOp, IE::MatMulOp matmulOp) {
    auto multiplyShape = getShape(multiplyOp.getOutput());
    auto matmulShape = getShape(matmulOp.getOutput());
    return vpux::details::calcTotalShapeSize(multiplyShape.raw()) >
           vpux::details::calcTotalShapeSize(matmulShape.raw());
}

mlir::Value getSingleDataInput(IE::MultiplyOp multiplyOp) {
    for (auto operand : multiplyOp->getOperands()) {
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(operand.getDefiningOp())) {
            if (IE::isBaseContentSplat(constOp)) {
                return operand;
            }
            return nullptr;
        }
        auto shape = getShape(operand);
        if (vpux::details::calcTotalShapeSize(shape.raw()) == 1) {
            return operand;
        }
    }
    return nullptr;
}

mlir::LogicalResult SwapMultiplyWithMatmul::matchAndRewrite(IE::MultiplyOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got multiply layer at '{1}'", origOp->getName(), origOp->getLoc());
    if (!origOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "multiply has more than one user");
    }

    if (origOp.getPostOpAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "multiply has post op attr");
    }

    if (origOp.getClampAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "multiply has clamp attr");
    }

    auto singleDataInput = getSingleDataInput(origOp);
    if (singleDataInput == nullptr) {
        return matchFailed(rewriter, origOp, "multiply doesn't have single data input");
    }

    const auto nonSingleDataOperand = origOp.getInput1() == singleDataInput ? origOp.getInput2() : origOp.getInput1();

    auto matmulOp = mlir::dyn_cast<IE::MatMulOp>(*origOp.getOutput().getUsers().begin());
    if (matmulOp == nullptr) {
        return matchFailed(rewriter, origOp, "multiply user is not a matmul");
    }

    if (!isBeneficalToSwap(origOp, matmulOp)) {
        return matchFailed(rewriter, origOp, "not benefical to swap multiply with matmul");
    }

    rewriter.setInsertionPoint(matmulOp);
    auto matmulInput1 = matmulOp.getInput1().getDefiningOp() == origOp ? nonSingleDataOperand : matmulOp.getInput1();
    auto matmulInput2 = matmulOp.getInput2().getDefiningOp() == origOp ? nonSingleDataOperand : matmulOp.getInput2();
    auto newMatMul = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), matmulInput1, matmulInput2,
                                                   matmulOp.getTransposeA(), matmulOp.getTransposeB());

    auto multiplyInput1 = origOp.getInput1() == singleDataInput ? origOp.getInput1() : newMatMul.getOutput();
    auto multiplyInput2 = origOp.getInput2() == singleDataInput ? origOp.getInput2() : newMatMul.getOutput();

    auto newMultiply = rewriter.create<IE::MultiplyOp>(origOp->getLoc(), multiplyInput1, multiplyInput2,
                                                       origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
                                                       origOp.getClampAttr());
    rewriter.replaceOp(matmulOp, newMultiply.getOutput());
    _log.trace("Successfully swap multiply with matmul");

    return mlir::success();
}

//
// SwapMultiplyWithMatmulPass
//

class SwapMultiplyWithMatmulPass final : public IE::SwapMultiplyWithMatmulBase<SwapMultiplyWithMatmulPass> {
public:
    explicit SwapMultiplyWithMatmulPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SwapMultiplyWithMatmulPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapMultiplyWithMatmul>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapMultiplyWithMatmulPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMultiplyWithMatmulPass(Logger log) {
    return std::make_unique<SwapMultiplyWithMatmulPass>(log);
}
