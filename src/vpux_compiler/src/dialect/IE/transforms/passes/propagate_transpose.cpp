//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// MoveThroughSoftmax
//

class MoveThroughSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    MoveThroughSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto transposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto softmaxInputRank = origOp.getInput().getType().dyn_cast<NDTypeInterface>().getRank();
    const auto softmaxAxisInd = getPositiveAxisInd(origOp.getAxisIndAttr(), softmaxInputRank);

    const auto transposePerm = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    const auto newSoftmaxAxisInd = transposePerm.dimAt(softmaxAxisInd).ind();

    auto newSoftmaxOp =
            rewriter.create<IE::SoftMaxOp>(origOp.getLoc(), transposeOp.getInput().getType(), transposeOp.getInput(),
                                           getIntAttr(getContext(), newSoftmaxAxisInd), origOp.getPadSizeAttr());
    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newSoftmaxOp.getOutput(),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    origOp.replaceAllUsesWith(newTransposeOp.getOutput());

    return mlir::success();
}

//
// MoveThroughEltwiseGeneric
//

template <class ConcreteOp>
class MoveThroughEltwiseGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughEltwiseGeneric(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughEltwiseGeneric");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult MoveThroughEltwiseGeneric<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(origOp->getNumResults() == 1 && origOp->getNumOperands() == 1,
                      "Not a single input & output operation");

    auto transposeOp = origOp.getInput().template getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto transposeOrder = transposeOp.getOrderValue();
    if (!transposeOrder.has_value()) {
        return matchFailed(_log, rewriter, origOp, "Found invalid TransposeOp");
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperand(0), transposeOp.getInput());
    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newOp->getResult(0),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    rewriter.replaceOp(origOp, newTransposeOp.getOutput());

    return mlir::success();
}

//
// MoveTransposeAffineReshapeThroughAdd
//

/* Rewrite the pattern from:

        Transpose
            |
      AffineReshape
            |
           Add
            |
      (QuantizeCast)

    to:
           Add
            |
      (QuantizeCast)
            |
        Transpose
            |
      AffineReshape
 */

class MoveTransposeAffineReshapeThroughAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    MoveTransposeAffineReshapeThroughAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        this->setDebugName("MoveTransposeAffineReshapeThroughAdd");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveTransposeAffineReshapeThroughAdd::matchAndRewrite(IE::AddOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto input1Op = origOp.getInput1().getDefiningOp();
    auto input2Op = origOp.getInput2().getDefiningOp();

    if (input1Op == nullptr || input2Op == nullptr || getShape(origOp.getInput1()) != getShape(origOp.getInput2())) {
        return mlir::failure();
    }

    auto getAffineReshapeOp = [&](mlir::Operation* op1, mlir::Operation* op2) -> std::optional<IE::AffineReshapeOp> {
        if (mlir::isa<IE::AffineReshapeOp>(op1)) {
            if (mlir::isa<Const::DeclareOp>(op2) && op1->hasOneUse()) {
                return mlir::cast<IE::AffineReshapeOp>(op1);
            }

            // Two inputs of add are from the same affineReshape
            if (op1 == op2 && static_cast<size_t>(std::distance(op1->getUsers().begin(), op1->getUsers().end())) == 2) {
                return mlir::cast<IE::AffineReshapeOp>(op1);
            }
        }

        if (mlir::isa<IE::AffineReshapeOp>(op2) && mlir::isa<Const::DeclareOp>(op1) && op2->hasOneUse()) {
            return mlir::cast<IE::AffineReshapeOp>(op2);
        }

        return std::nullopt;
    };

    auto affineReshapeOpt = getAffineReshapeOp(input1Op, input2Op);
    if (!affineReshapeOpt.has_value()) {
        return mlir::failure();
    }
    auto affineReshapeOp = affineReshapeOpt.value();

    const auto reshapeInput = getShape(affineReshapeOp.getInput());
    const auto addInput = getShape(origOp.getInput1());
    if (reshapeInput.size() != addInput.size()) {
        return mlir::failure();
    }

    auto transposeOp = affineReshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return mlir::failure();
    }

    auto inputType = transposeOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    if (inputShape[Dims4D::Act::C] % alignment || inputShape[Dims4D::Act::N] > 1) {
        return mlir::failure();
    }

    auto orderValueAttr = mlir::AffineMapAttr::get(
            vpux::getPermutationFromOrders(DimsOrder::fromAffineMap(transposeOp.getOrderValueAttr().getValue()),
                                           DimsOrder::NCHW, rewriter.getContext()));
    auto getInputValue = [&](mlir::Operation* op) -> mlir::Value {
        if (!mlir::isa<Const::DeclareOp>(op)) {
            return transposeOp.getInput();
        }
        auto constReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), op->getResult(0), nullptr, false,
                                                                 getIntArrayAttr(rewriter.getContext(), reshapeInput));
        return rewriter.create<IE::TransposeOp>(origOp.getLoc(), constReshape, nullptr, orderValueAttr).getResult();
    };

    auto input1 = getInputValue(input1Op);
    auto input2 = getInputValue(input2Op);
    auto origOutputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto outElemType = origOutputType.getElementType();
    auto newAddOutType = mlir::RankedTensorType::get(inputShape, outElemType).cast<NDTypeInterface>();
    newAddOutType = newAddOutType.changeDimsOrder(origOutputType.getDimsOrder());
    auto outputVal =
            rewriter.create<IE::AddOp>(origOp.getLoc(), newAddOutType, input1, input2, origOp.getAutoBroadcastAttr(),
                                       origOp.getPostOpAttr(), origOp.getClampAttr())
                    .getOutput();

    auto postQuantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*origOp.getOutput().user_begin());
    if (postQuantizeCastOp != nullptr && origOp->hasOneUse()) {
        outputVal =
                rewriter.create<IE::QuantizeCastOp>(
                                postQuantizeCastOp.getLoc(), outputVal,
                                postQuantizeCastOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType())
                        .getOutput();
    }

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), outputVal, transposeOp.getOrder(),
                                                           transposeOp.getOrderValueAttr());
    auto newReshapeOp = rewriter.create<IE::AffineReshapeOp>(affineReshapeOp.getLoc(), newTransposeOp.getOutput(),
                                                             affineReshapeOp.getDimMappingAttr(),
                                                             affineReshapeOp.getShapeValueAttr());

    if (postQuantizeCastOp == nullptr) {
        origOp.replaceAllUsesWith(newReshapeOp.getOutput());
    } else {
        postQuantizeCastOp.replaceAllUsesWith(newReshapeOp.getOutput());
        rewriter.eraseOp(postQuantizeCastOp);
    }

    if (origOp) {
        rewriter.eraseOp(origOp);
    }
    if (affineReshapeOp) {
        rewriter.eraseOp(affineReshapeOp);
    }
    if (transposeOp) {
        rewriter.eraseOp(transposeOp);
    }

    _log.trace("[{0}] Replaced with 'IE::AffineReshapeOp'", getDebugName());
    return mlir::success();
}

//
// PropagateTransposePass
//

class PropagateTransposePass final : public IE::PropagateTransposeBase<PropagateTransposePass> {
public:
    explicit PropagateTransposePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateTransposePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughSoftmax>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::GeluOp>>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::SwishOp>>(&ctx, _log);
    patterns.add<MoveTransposeAffineReshapeThroughAdd>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateTransposePass(Logger log) {
    return std::make_unique<PropagateTransposePass>(log);
}
