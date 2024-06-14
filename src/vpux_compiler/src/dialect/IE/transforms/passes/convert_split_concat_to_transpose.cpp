//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<mlir::Operation*> getConcatOpConsumer(mlir::Operation* op, bool requireAffineReshape) {
    if (op == nullptr || op->getUsers().empty()) {
        return mlir::failure();
    }

    mlir::Operation* concatOp = nullptr;

    for (auto user : op->getUsers()) {
        mlir::Operation* operation = user;

        if (requireAffineReshape) {
            if (!mlir::isa<IE::AffineReshapeOp>(operation) || operation->getUsers().empty() ||
                !operation->hasOneUse()) {
                return mlir::failure();
            }
            operation = *(operation->getUsers().begin());
        }

        if (!mlir::isa<IE::ConcatOp>(operation)) {
            return mlir::failure();
        }

        if (concatOp == nullptr) {
            concatOp = operation;
            continue;
        } else if (concatOp != operation) {
            return mlir::failure();
        }
    }

    return concatOp;
}

// Check the split dim size after splitOp is 1 to make it feasible to convert into TransposeOp
mlir::FailureOr<vpux::Dim> getSplitDimToShape1(IE::SplitOp splitOp) {
    const auto splitInputShape = getShape(splitOp.getInput());
    const auto splitDim = Dim(splitOp.getAxisValue().value());
    const auto splitNum = splitOp.getNumSplits();

    if (splitInputShape[splitDim] != splitNum) {
        return mlir::failure();
    }

    return splitDim;
}

// Check the concat dim input size is 1 to make it feasible to convert into TransposeOp
mlir::FailureOr<SmallVector<Dim>> getConcatDimWithShape1(IE::ConcatOp concatOp, bool supportAdjacentDims) {
    const auto concatStaticOffsets = concatOp.getStaticOffsets().value();
    if (concatStaticOffsets.size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto concatInputType = concatOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto concatOutputType = concatOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto concatInShape = concatInputType.getShape();
    const auto concatOutShape = concatOutputType.getShape();
    if (concatInShape.size() != concatOutShape.size()) {
        return mlir::failure();
    }

    SmallVector<Dim> concatDims;
    for (const auto& idx : irange(concatInShape.size())) {
        if (concatInShape[Dim(idx)] != concatOutShape[Dim(idx)]) {
            concatDims.push_back(Dim(idx));
        }
    }

    if (concatDims.empty() || concatDims.size() > 1) {
        return mlir::failure();
    }

    for (const auto& input : concatOp.getInputs()) {
        const auto inputShape = getShape(input);
        if (inputShape[concatDims[0]] != 1 && !supportAdjacentDims) {
            return mlir::failure();
        } else {
            SmallVector<Dim> adjustDims;
            if (concatDims[0].ind() - 1 > 0) {
                adjustDims.push_back(Dim(concatDims[0].ind() - 1));
            }
            if (concatDims[0].ind() + 1 < checked_cast<int32_t>(concatInShape.size())) {
                adjustDims.push_back(Dim(concatDims[0].ind() + 1));
            }

            for (auto dim : adjustDims) {
                if (inputShape[dim] == 1) {
                    concatDims[0] = dim;
                    break;
                }
            }

            if (inputShape[concatDims[0]] != 1) {
                return mlir::failure();
            }
        }
    }

    return concatDims;
}

bool checkAffineReshapeDimMapping(IE::SplitOp splitOp) {
    auto userOp = splitOp.getOutputs()[0].getUsers().begin();
    auto affineReshapeOp = mlir::cast<IE::AffineReshapeOp>(*userOp);
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());

    for (const auto& res : splitOp.getOutputs()) {
        auto curAffineReshape = mlir::cast<IE::AffineReshapeOp>(*res.getUsers().begin());
        const auto curDimMapping = parseIntArrayOfArrayAttr<int64_t>(curAffineReshape.getDimMapping());
        if (curDimMapping != dimMapping) {
            return false;
        }
    }

    const auto affineInShape = getShape(affineReshapeOp.getInput());
    const auto affineOutShape = getShape(affineReshapeOp.getOutput());
    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];
        for (auto outIdx : mappedDim) {
            // merge case: N x 1 -> N
            // merge case: 1 x N -> N
            if (inIdx > 0 && mappedDim == dimMapping[inIdx - 1]) {
                if (affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)] && affineInShape[Dim(inIdx)] != 1) {
                    return false;
                } else if (affineInShape[Dim(inIdx - 1)] != affineOutShape[Dim(outIdx)] &&
                           affineInShape[Dim(inIdx - 1)] != 1) {
                    return false;
                }
            }

            // split case: N -> N x 1
            // split case: N -> 1 x N
            if (mappedDim.size() > 1) {
                if (affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)] && affineOutShape[Dim(outIdx)] != 1) {
                    return false;
                }
            }
        }
    }

    return true;
}

//
// SplitAffineReshapeConcatRewriter
//

//
//               |
//            SplitOp
//          /         \                                |
//   AffineReshape  AffineReshape       ->         Transpose
//          \         /                                |
//            ConcatOp
//               |
//

class SplitAffineReshapeConcatRewriter final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitAffineReshapeConcatRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("SplitAffineReshapeConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitAffineReshapeConcatRewriter::matchAndRewrite(IE::SplitOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Split operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto getConsumerResult = getConcatOpConsumer(origOp, true);
    if (mlir::failed(getConsumerResult)) {
        return mlir::failure();
    }

    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(getConsumerResult.value());
    VPUX_THROW_WHEN(concatOp == nullptr, "Not a Concat operation");

    if (origOp.getOutputs().size() != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto concatInputType = concatOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto concatInShape = concatInputType.getShape();

    // Supported case for splitOp: split the dim to shape 1
    auto getSplitDim = getSplitDimToShape1(origOp);
    if (mlir::failed(getSplitDim)) {
        return mlir::failure();
    }

    const auto splitDim = getSplitDim.value();

    // Supported case for concatOp: concat the dim with shape 1
    auto getconcatDims = getConcatDimWithShape1(concatOp, false);
    if (mlir::failed(getconcatDims)) {
        return mlir::failure();
    }

    const auto concatDims = getconcatDims.value();

    // affineReshapeOp dim_mapping supported cases:
    // merge case: N x 1 -> N, 1 x N -> N
    // split case: N -> N x 1, N -> 1 x N
    if (!checkAffineReshapeDimMapping(origOp)) {
        return mlir::failure();
    }

    // Create new transposeOp
    SmallVector<unsigned> transPerm(getShape(origOp.getInput()).size(), 0);
    for (const auto& idx : irange(concatInShape.size())) {
        if (Dim(idx) == splitDim) {
            transPerm[idx] = concatDims[0].ind();
        } else if (Dim(idx) == concatDims[0]) {
            transPerm[idx] = splitDim.ind();
        } else {
            transPerm[idx] = idx;
        }
    }

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, rewriter.getContext()));
    auto newTransposeOp = rewriter.create<IE::TransposeOp>(origOp->getLoc(), origOp.getInput(), nullptr, orderAttr);
    concatOp.replaceAllUsesWith(newTransposeOp.getOutput());

    _log.trace("[{0}] Replaced with 'IE::TransposeOp'", getDebugName());

    return mlir::success();
}

//
// SplitConcatRewriter
//

//
//               |
//            SplitOp
//              | |                                    |
//            ConcatOp          ->              AffineReshapeOp
//               |                                     |

class SplitConcatRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    SplitConcatRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("SplitConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitConcatRewriter::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite ConcatOp operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto splitOp = origOp.getOperand(0).getDefiningOp<IE::SplitOp>();
    if (splitOp == nullptr) {
        return mlir::failure();
    }

    auto getConsumerResult = getConcatOpConsumer(splitOp, false);
    if (mlir::failed(getConsumerResult)) {
        return mlir::failure();
    }

    VPUX_THROW_WHEN(mlir::dyn_cast_or_null<IE::ConcatOp>(getConsumerResult.value()) == nullptr,
                    "Not a Concat operation");

    if (splitOp.getOutputs().size() != origOp.getInputs().size()) {
        return mlir::failure();
    }

    // Supported case for splitOp: split the dim to shape 1
    auto getSplitDim = getSplitDimToShape1(splitOp);
    if (mlir::failed(getSplitDim)) {
        return mlir::failure();
    }

    // Supported case for concatOp: axis dim or adjust dims of concat with shape 1
    auto getconcatDims = getConcatDimWithShape1(origOp, true);
    if (mlir::failed(getconcatDims)) {
        return mlir::failure();
    }
    const auto concatDims = getconcatDims.value();

    const auto origOutputShape = getShape(origOp.getOutput());
    const auto reassociationMap =
            vpux::IE::getReassociationMap(getShape(splitOp.getInput()).raw(), origOutputShape.raw());
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    auto affineReshape = rewriter.create<IE::AffineReshapeOp>(
            origOp->getLoc(), splitOp.getInput(), getIntArrayOfArray(getContext(), reassociationMap.value()),
            getIntArrayAttr(rewriter.getContext(), origOutputShape));
    rewriter.replaceOp(origOp, affineReshape.getOutput());

    _log.trace("[{0}] Replaced with 'IE::AffineReshapeOp'", getDebugName());

    return mlir::success();
}

//
// ConvertSplitConcatToTransposePass
//

class ConvertSplitConcatToTransposePass final :
        public IE::ConvertSplitConcatToTransposeBase<ConvertSplitConcatToTransposePass> {
public:
    explicit ConvertSplitConcatToTransposePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSplitConcatToTransposePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SplitAffineReshapeConcatRewriter>(&ctx, _log);
    patterns.insert<SplitConcatRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSplitConcatToTransposePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSplitConcatToTransposePass(Logger log) {
    return std::make_unique<ConvertSplitConcatToTransposePass>(log);
}
