//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// TileSliceRewriter
//

class TileSliceRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    TileSliceRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("TileSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileSliceRewriter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Slice operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto ctx = rewriter.getContext();
    auto reshapeOp = origOp.getSource().getDefiningOp<IE::AffineReshapeOp>();

    bool hasReshape = reshapeOp != nullptr;
    auto tileOp = hasReshape ? reshapeOp.getInput().getDefiningOp<IE::TileOp>()
                             : origOp.getSource().getDefiningOp<IE::TileOp>();

    if (tileOp == nullptr) {
        return mlir::failure();
    }
    const auto repeatsValue = parseIntArrayAttr<int64_t>(tileOp.getRepeatsValuesAttr());
    const auto nonOneRepeatsValueNum = llvm::count_if(repeatsValue, [](auto repeat) {
        return repeat != 1;
    });
    if (nonOneRepeatsValueNum != 1) {
        return mlir::failure();
    }
    auto tileAxis = IE::getSingleDiffAxis(getShape(tileOp.getInput()), getShape(tileOp.getOutput()));
    if (!tileAxis.has_value()) {
        return mlir::failure();
    }
    auto sliceAxis = IE::getSingleDiffAxis(getShape(origOp.getSource()), getShape(origOp.getResult()));
    if (!sliceAxis.has_value()) {
        return mlir::failure();
    }

    const auto sliceOffset = parseIntArrayAttr<int64_t>(origOp.getStaticOffsetsAttr());
    const auto sliceSize = parseIntArrayAttr<int64_t>(origOp.getStaticSizesAttr());
    const auto nonZeroOffset = llvm::any_of(sliceOffset, [](auto offset) {
        return offset > 0;
    });

    if (nonZeroOffset) {
        return mlir::failure();
    }

    int64_t newRepeatValue = 0;
    if (!hasReshape) {
        newRepeatValue = sliceSize[sliceAxis.value().ind()];
    } else {
        const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());
        auto mappedDim = dimMapping[tileAxis.value().ind()];

        if (mappedDim.size() > 1 || mappedDim[0] != sliceAxis.value().ind()) {
            return mlir::failure();
        }
        auto reshapeInputShape = getShape(reshapeOp.getInput());
        SmallVector<int64_t> mergedDim;
        for (int64_t index = 0; index < checked_cast<int64_t>(reshapeInputShape.size()); index++) {
            if (index == tileAxis.value().ind()) {
                continue;
            }
            auto localMappedDim = dimMapping[index];
            if (localMappedDim.size() == 1 && localMappedDim[0] == mappedDim[0]) {
                mergedDim.push_back(index);
            }
        }

        if (mergedDim.empty()) {
            newRepeatValue = sliceSize[sliceAxis.value().ind()];
        } else if (mergedDim.size() == 1) {
            if (sliceSize[sliceAxis.value().ind()] % reshapeInputShape[Dim(mergedDim[0])] == 0) {
                newRepeatValue = sliceSize[sliceAxis.value().ind()] / reshapeInputShape[Dim(mergedDim[0])];
            } else {
                return mlir::failure();
            }
        } else {
            return mlir::failure();
        }
    }

    auto repeatsOnNewShape = repeatsValue;
    repeatsOnNewShape[tileAxis.value().ind()] = newRepeatValue;
    const auto outputType = tileOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto tileOpInputShape = getShape(tileOp.getInput());
    Shape newShape(tileOpInputShape.raw());
    newShape[tileAxis.value()] = newRepeatValue;
    const auto newOutputType = outputType.changeShape(newShape);
    auto newTileOp = rewriter.create<IE::TileOp>(origOp->getLoc(), newOutputType, tileOp.getInput(), nullptr,
                                                 getIntArrayAttr(ctx, repeatsOnNewShape));
    if (hasReshape) {
        const auto sliceOutShape = getShape(origOp.getResult()).raw();
        rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, newTileOp.getOutput(), reshapeOp.getDimMappingAttr(),
                                                         getIntArrayAttr(ctx, sliceOutShape));
    } else {
        rewriter.replaceOp(origOp, newTileOp);
    }
    return mlir::success();
}

//
// ConcatSliceRewriter
//

class ConcatSliceRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ConcatSliceRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ConcatSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatSliceRewriter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Slice operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto reshapeOp = origOp.getSource().getDefiningOp<IE::AffineReshapeOp>();
    bool hasReshape = reshapeOp != nullptr;
    auto concatOp = hasReshape ? reshapeOp.getInput().getDefiningOp<IE::ConcatOp>()
                               : origOp.getSource().getDefiningOp<IE::ConcatOp>();

    if (concatOp == nullptr || !concatOp.getStaticOffsetsAttr()) {
        return mlir::failure();
    }

    auto sliceOffset = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    const auto sliceOffsetShape = Shape(sliceOffset);
    const auto sliceOutShape = getShape(origOp.getResult());
    auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsetsAttr());
    const auto inputs = concatOp.getInputs();
    SmallVector<ShapeRef> newInputShapes;
    SmallVector<SmallVector<int64_t>> newInputShapesVec;

    if (hasReshape) {
        const auto affineOutShape = getShape(reshapeOp.getOutput());
        const auto modifiedAxes = IE::getConcatModifiedAxis(concatOp);
        for (const auto& input : inputs) {
            const SmallVector<int64_t> newShapeVec =
                    IE::calculateInputShapeAfterSwitchConcatAndAffineReshape(input, concatOp, reshapeOp);
            newInputShapesVec.emplace_back(newShapeVec);
        }

        for (const auto& vector : newInputShapesVec) {
            newInputShapes.emplace_back(ShapeRef(vector));
        }

        auto newOffsetsAttr =
                IE::getNewConcatOffsetsParameters(concatOp.getStaticOffsetsAttr(), reshapeOp.getDimMapping(), inputs,
                                                  newInputShapes, affineOutShape, modifiedAxes);
        concatOffsets = parseIntArrayOfArrayAttr<int64_t>(newOffsetsAttr);
    } else {
        for (const auto& input : inputs) {
            newInputShapes.push_back(getShape(input));
        }
    }

    for (const auto& p : zip(inputs, concatOffsets, newInputShapes)) {
        auto curVal = std::get<0>(p);
        const auto curOffset = std::get<1>(p);
        const auto curShape = std::get<2>(p);
        const auto curOffsetShape = ShapeRef(curOffset);
        auto isSubTensor = [&]() -> bool {
            for (const auto ind : irange(sliceOutShape.size())) {
                const auto d = Dim(ind);
                if ((sliceOffsetShape[d] < curOffsetShape[d]) ||
                    (curOffsetShape[d] + curShape[d] < sliceOffsetShape[d] + sliceOutShape[d])) {
                    return false;
                }
            }
            return true;
        };

        if (!isSubTensor()) {
            continue;
        }

        _log.trace("ConcatSliceRewriter '{0}' at '{1}', {2}->{3}, {4}->{5}", origOp->getName(), origOp->getLoc(),
                   sliceOffsetShape, curOffsetShape, sliceOutShape, curShape);

        if (hasReshape) {
            curVal = rewriter.create<IE::AffineReshapeOp>(reshapeOp.getLoc(), curVal, reshapeOp.getDimMapping(),
                                                          getIntArrayAttr(rewriter.getContext(), curShape));
        }

        for (auto i : irange(sliceOffset.size())) {
            sliceOffset[i] -= curOffset[i];
        }

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, curVal, getIntArrayAttr(getContext(), sliceOffset),
                                                 origOp.getStaticSizes());

        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeOpSlicePass
//

class OptimizeOpSlicePass final : public IE::OptimizeOpSliceBase<OptimizeOpSlicePass> {
public:
    explicit OptimizeOpSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeOpSlicePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConcatSliceRewriter>(&ctx, _log);
    patterns.insert<TileSliceRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeOpSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeOpSlicePass(Logger log) {
    return std::make_unique<OptimizeOpSlicePass>(log);
}
