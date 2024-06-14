//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {

enum class FuseMode { CONVERT_TO_SLICE, CONVERT_TO_EXPAND };

FuseMode getFuseMode(ShapeRef patternInShape, ShapeRef patternOutShape);

// If FuseMode is 'CONVERT_TO_SLICE',  the 'firstShape' is 'newSliceOffsets'; the 'secondShape' is 'newSliceStaticSizes'
// If FuseMode is 'CONVERT_TO_EXPAND', the 'firstShape' is 'newPadsBegin'; the 'secondShape' is 'newPadsEnd'
mlir::FailureOr<std::tuple<Shape, Shape, FuseMode>> getSliceExpandFusedParameters(IE::SliceOp sliceOp,
                                                                                  IE::ExpandOp expandOp);
mlir::FailureOr<std::tuple<Shape, Shape, FuseMode>> getExpandSliceFusedParameters(IE::ExpandOp expandOp,
                                                                                  IE::SliceOp sliceOp);

mlir::LogicalResult genericOptimizeSliceImplicitExpand(IE::ExpandOp layerOp, mlir::Operation* implicitOp,
                                                       bool hasCalculationCost, mlir::PatternRewriter& rewriter,
                                                       Logger innerLog);

mlir::LogicalResult genericOptimizeSliceImplicitShapeCastExpand(IE::ExpandOp layerOp, IE::ShapeCastOp shapeCastOp,
                                                                mlir::Operation* implicitOp,
                                                                mlir::PatternRewriter& rewriter, Logger innerLog);

//
// OptimizeSliceImplicitExpand
//

template <class ImplicitLayer>
class OptimizeSliceImplicitExpand : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceImplicitExpand(mlir::MLIRContext* ctx, Logger log, bool hasCalculationCost)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log), _hasCalculationCost(hasCalculationCost) {
        setDebugName("OptimizeSliceImplicitExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        const auto innerLog = _log.nest();

        auto implicitOp = origOp.getInput().getDefiningOp<ImplicitLayer>();
        if (implicitOp == nullptr) {
            return mlir::failure();
        }
        return genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(), _hasCalculationCost, rewriter,
                                                  innerLog);
    }

private:
    Logger _log;
    bool _hasCalculationCost;
};

//
// OptimizeSliceConcatExpand
//

class OptimizeSliceConcatExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceConcatExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceConcatExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeSliceTwoConcatsExpand
//

class OptimizeSliceTwoConcatsExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceTwoConcatsExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceTwoConcatsExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeSliceExpand
//

class OptimizeSliceExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeExpandSlice
//

class OptimizeExpandSlice final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeExpandSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeExpandSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// OptimizeSliceShapeCastExpand
//

template <class EltwiseLikeSwOp>
class OptimizeSliceShapeCastExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceShapeCastExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceShapeCastExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        const auto innerLog = _log.nest();

        auto shapeCastOp = origOp.getInput().getDefiningOp<IE::ShapeCastOp>();
        if (shapeCastOp == nullptr) {
            innerLog.trace("Expand '{0}' input is not 'shapeCastOp'", origOp->getLoc());
            return mlir::failure();
        }
        if (!shapeCastOp->hasOneUse()) {
            return mlir::failure();
        }

        auto swOp = shapeCastOp.getSource().getDefiningOp<EltwiseLikeSwOp>();
        if (swOp == nullptr) {
            innerLog.trace("ShapeCastOp '{0}' input is not SW op", shapeCastOp->getLoc());
            return mlir::failure();
        }
        if (!swOp->hasOneUse()) {
            return mlir::failure();
        }

        return genericOptimizeSliceImplicitShapeCastExpand(origOp, shapeCastOp, swOp.getOperation(), rewriter,
                                                           innerLog);
    }

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
