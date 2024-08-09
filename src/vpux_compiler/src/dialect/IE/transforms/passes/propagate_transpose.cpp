//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"

#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"

#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/passes.hpp"
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
// MoveThroughSlice
//

void updateSliceAttributes(mlir::ArrayAttr& staticSizes, mlir::ArrayAttr& staticOffsets, mlir::AffineMap permutation,
                           DimsOrder inOrder) {
    VPUX_THROW_UNLESS(permutation.isPermutation(), "Incorrect permutation");
    const auto order = DimsOrder::fromAffineMap(permutation);
    const auto dimsPermutation = order.toPermutation();

    VPUX_THROW_WHEN((staticSizes == nullptr) || (staticOffsets == nullptr), "Incorrect Slice parameters");
    const auto oldOffsets = parseIntArrayAttr<int64_t>(staticOffsets);
    const auto oldSizes = parseIntArrayAttr<int64_t>(staticSizes);
    SmallVector<int64_t> newOffsets;
    SmallVector<int64_t> newSizes;
    newOffsets.resize(oldOffsets.size(), 0);
    newSizes.resize(oldSizes.size(), 0);

    for (auto ind : irange(oldOffsets.size())) {
        const auto inDim = Dim(inOrder.dimAt(ind).ind());
        const auto outDim = dimsPermutation[inDim.ind()];

        newOffsets[outDim.ind()] = oldOffsets[inDim.ind()];
        newSizes[outDim.ind()] = oldSizes[inDim.ind()];
    }

    staticOffsets = getIntArrayAttr(staticOffsets.getContext(), newOffsets);
    staticSizes = getIntArrayAttr(staticSizes.getContext(), newSizes);
}

class MoveThroughSlice final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    MoveThroughSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughSlice::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto isLegalSliceOp = [&](IE::SliceOp op) -> bool {
        auto prevTranspose = op.getSource().getDefiningOp<IE::TransposeOp>();
        if (prevTranspose == nullptr) {
            return true;
        }

        auto orderAttr = prevTranspose.getOrderValueAttr();
        auto order = DimsOrder::fromAffineMap(orderAttr.getValue());
        for (auto user : prevTranspose.getOutput().getUsers()) {
            if (!mlir::isa<IE::SliceOp>(user)) {
                return true;
            }

            auto slice = mlir::cast<IE::SliceOp>(user);
            auto inShape = getShape(slice.getSource());
            auto outShape = getShape(slice.getResult());
            if (inShape.size() != 4 || outShape.size() != 4) {
                return true;
            }

            auto sliceIdx = -1;
            for (auto i : irange(4)) {
                if (inShape[vpux::Dim(i)] != outShape[vpux::Dim(i)]) {
                    sliceIdx = i;
                }
            }
            if (sliceIdx == -1) {
                return true;
            }
            // Only slice on conv channel can be fused into conv
            if (order.dimAt(sliceIdx) != Dims4D::Act::C) {
                return true;
            }
        }

        auto prevConv = prevTranspose.getInput().getDefiningOp<IE::ConvolutionOp>();
        if (prevConv == nullptr || !prevConv->hasOneUse()) {
            return true;
        }

        _log.trace("Find illegal SliceOp {0}", op);

        return false;
    };

    if (isLegalSliceOp(origOp)) {
        return mlir::failure();
    }

    auto origTransposeOp = origOp.getSource().getDefiningOp<IE::TransposeOp>();
    auto orderAttr = origTransposeOp.getOrderValueAttr();
    const auto origPermuteInputOrder = DimsOrder::fromValue(origTransposeOp.getInput());

    for (auto op : origTransposeOp.getOutput().getUsers()) {
        if (auto slice = mlir::dyn_cast_or_null<IE::SliceOp>(op)) {
            auto staticSizes = slice.getStaticSizesAttr();
            auto staticOffsets = slice.getStaticOffsetsAttr();

            updateSliceAttributes(staticSizes, staticOffsets, orderAttr.getValue(), origPermuteInputOrder);

            auto newSliceOp = rewriter.create<IE::SliceOp>(origOp->getLoc(), origTransposeOp.getInput(), staticOffsets,
                                                           staticSizes);

            rewriter.replaceOpWithNewOp<IE::TransposeOp>(slice, newSliceOp.getResult(), nullptr,
                                                         origTransposeOp.getOrderValueAttr());
        }
    }

    rewriter.eraseOp(origTransposeOp);

    _log.trace("Swap slice success");
    return mlir::success();
}

//
// MoveThroughEltwiseGeneric
//

using VerifyCb = FuncRef<bool(mlir::Operation*)>;

template <class ConcreteOp>
class MoveThroughEltwiseGeneric final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughEltwiseGeneric(mlir::MLIRContext* ctx, Logger log, VerifyCb verifyFunc = nullptr)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _verifyFunc(verifyFunc) {
        this->setDebugName("MoveThroughEltwiseGeneric");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VerifyCb _verifyFunc;
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

    if ((_verifyFunc) && !_verifyFunc(origOp.getOperation())) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperand(0), transposeOp.getInput());
    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newOp->getResult(0),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    rewriter.replaceOp(origOp, newTransposeOp.getOutput());

    return mlir::success();
}

class MoveTransposeThroughMultiply final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MoveTransposeThroughMultiply(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("MoveTransposeThroughMultiply");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveTransposeThroughMultiply::matchAndRewrite(IE::MultiplyOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (getShape(origOp.getInput1()) != getShape(origOp.getInput2())) {
        return mlir::failure();
    }

    auto transpose1Op = origOp.getInput1().getDefiningOp<IE::TransposeOp>();
    auto transpose2Op = origOp.getInput2().getDefiningOp<IE::TransposeOp>();
    if (transpose1Op == nullptr || !transpose1Op->hasOneUse() || transpose2Op == nullptr ||
        !transpose2Op->hasOneUse() || transpose1Op.getOrder() != nullptr || transpose2Op.getOrder() != nullptr) {
        // Only constant input is supported when order is set. In this case transpose can be fused into constant
        return mlir::failure();
    }

    if (transpose1Op.getOrderValue() != transpose2Op.getOrderValue()) {
        return mlir::failure();
    }

    auto input1 = transpose1Op.getInput();
    auto input2 = transpose2Op.getInput();
    auto origOutputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto newOutputType = origOutputType.changeShape(getShape(input1));
    auto newMultiplyOp = rewriter.create<IE::MultiplyOp>(origOp.getLoc(), newOutputType, input1, input2,
                                                         origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
                                                         origOp.getClampAttr());

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, newMultiplyOp.getOutput(), transpose1Op.getOrder(),
                                                 transpose1Op.getOrderValueAttr());
    rewriter.eraseOp(transpose1Op);
    rewriter.eraseOp(transpose2Op);
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

    const auto verifyAvgPool = [](mlir::Operation* op) {
        auto avgPoolOp = mlir::dyn_cast<IE::AvgPoolOp>(op);
        if ((avgPoolOp == nullptr) || (!IE::isEltwisePooling<IE::AvgPoolOp>(avgPoolOp))) {
            return false;
        }

        auto transposeOp = avgPoolOp.getInput().getDefiningOp<IE::TransposeOp>();
        if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
            return false;
        }

        auto inputType = transposeOp.getInput().getType().cast<vpux::NDTypeInterface>();
        const auto inputShape = inputType.getShape();
        auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(avgPoolOp.getOperation());
        const auto alignment = iface.getInputChannelAlignment();
        // It's more safe to move transpose after avgPool when input satisfy avgPool requirement
        if (inputShape[Dims4D::Act::C] % alignment || inputShape[Dims4D::Act::N] > 1) {
            return false;
        }

        // Not to swap for model input to transpose case, to avoid introduce nce.permute op before avgPool
        if (transposeOp.getInput().getDefiningOp() == nullptr) {
            return false;
        }

        return true;
    };

    const auto verifyConvert = [](mlir::Operation* op) {
        if (auto convertOp = mlir::dyn_cast<IE::ConvertOp>(op)) {
            auto transposeOp = convertOp.getInput().getDefiningOp<IE::TransposeOp>();
            if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
                return false;
            }
            const auto inOrder = DimsOrder::fromValue(transposeOp.getInput());
            const auto inShape = getShape(transposeOp.getInput());
            const auto inMemShape = inOrder.toMemoryOrder(inShape);
            const auto perm = transposeOp.getOrderValue();
            if (perm.has_value() && isTrivialPermute(inMemShape, perm.value())) {
                return true;
            }

            const auto srcType = convertOp.getInput().getType();
            const auto dstElemType = convertOp.getDstElemType();
            if (getElemTypeSize(srcType) < getElemTypeSize(dstElemType)) {
                return false;
            }

            // If ConvertOp is the last op to return, there is no benefit to move transpose through it
            return !llvm::any_of(convertOp->getUsers(), [](auto user) {
                return mlir::isa<mlir::func::ReturnOp>(user);
            });
        }
        return false;
    };

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughSoftmax>(&ctx, _log);
    patterns.add<MoveThroughSlice>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::GeluOp>>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::SwishOp>>(&ctx, _log);
    patterns.add<MoveThroughEltwiseGeneric<IE::AvgPoolOp>>(&ctx, _log, verifyAvgPool);
    patterns.add<MoveThroughEltwiseGeneric<IE::ConvertOp>>(&ctx, _log, verifyConvert);
    patterns.add<IE::MoveTransposeAffineReshapeThroughAdd>(&ctx, vpux::benefitHigh, _log);
    patterns.add<MoveTransposeThroughMultiply>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateTransposePass(Logger log) {
    return std::make_unique<PropagateTransposePass>(log);
}
