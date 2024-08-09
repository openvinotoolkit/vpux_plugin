//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// Common Utils
//

Dim getLowestDim(ShapeRef shape, const DimsOrder& order) {
    const auto rank = order.numDims();
    auto lowestDim = order.dimAt(rank - 1);
    for (auto idx : irange(rank)) {
        auto dim = order.dimAt(idx);
        if (shape[dim] > 1) {
            lowestDim = dim;
        }
    }
    return lowestDim;
}

int64_t getTotalSizeBeforeDim(ShapeRef shape, const DimsOrder& order, const Dim& dim) {
    int64_t totalSize = 1;
    for (auto idx : irange(order.dimPos(dim))) {
        totalSize *= shape[order.dimAt(idx)];
    }
    return totalSize;
}

// Adjusts shape of Softmax to leverage the optimized softmax kernel implementation
// for axis 0 (the last dim in compiler scope)
// Examples:
//   - Softmax(shape=[1, 16, 24, 1], axisInd=2, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 16, 1, 24], axisInd=3, layout=NCHW)
//   - Softmax(shape=[1, 1, 24, 16], axisInd=3, layout=NHWC) is adjusted to
//     Softmax(shape=[1, 16, 24, 1], axisInd=1, layout=NHWC)
// Note that these adjustments should not change the real data in memory, so this pattern
// will only be applied when axis dim is the lowest dim in memory
mlir::LogicalResult adjustForAxisZeroOpt(Shape& shape, int64_t& axisInd, const DimsOrder& order) {
    const auto axisDim = Dim(axisInd);
    const auto lowestDim = getLowestDim(shape, order);
    const auto lastDimInMem = order.dimAt(shape.size() - 1);

    if (axisDim != lowestDim || axisDim == lastDimInMem) {
        return mlir::failure();
    }

    // swap lowest dim with the last memdim
    shape[lastDimInMem] = shape[lowestDim];
    shape[lowestDim] = 1;
    // axis becomes the last memdim
    axisInd = lastDimInMem.ind();

    return mlir::success();
}

//
// AdjustShapeForSoftmax
//
// This rewritter adjusts shape of softmax for optimized kernel implementations
// Supported Optimizations:
//   - Kernel optimization for softmax with axis=0 (last memdim in compiler scope)
//   - Gather dimensions on the tile dim for multishave optimizations
class AdjustShapeForSoftmax final : public mlir::OpRewritePattern<VPU::SoftMaxOp> {
public:
    AdjustShapeForSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForSoftmax");
    }

private:
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, int64_t& axisInd, const DimsOrder& order,
                                               const int64_t numActShaves) const;
    mlir::LogicalResult matchAndRewrite(VPU::SoftMaxOp softmaxOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Adjusts the shape of Softmax to leverage as much shave engines as possible by gather
// dimensions on tile dimension
// Or fuses batch dim to tile dim, as SplitOverBatch is not supported yet and non-outermost dim tiling will introduce
// strided copy
// Examples: (Assume 4 shave engines)
// * Leverage all shaves
//   - Softmax(shape=[1, 2, 16, 24], axisInd=3, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 4, 8, 24], axisInd=3, layout=NCHW)
//   - Softmax(shape=[1, 24, 2, 16], axisInd=1, layout=NHWC) is adjusted to
//     Softmax(shape=[1, 24, 4, 8], axisInd=1, layout=NHWC)
// * Fuse batch dim
//   - Softmax(shape=[16, 4, 16, 24], axisInd=3, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 64, 16, 24], axisInd=3, layout=NCHW)
// Note that these adjustments should not change the real data in memory, and the axis dim
// should not be the tile dim
mlir::LogicalResult AdjustShapeForSoftmax::adjustForMultiShaveOpt(Shape& shape, int64_t& axisInd,
                                                                  const DimsOrder& order,
                                                                  const int64_t numActShaves) const {
    const auto axisDim = Dim(axisInd);

    // only support NCHW and NHWC layout
    if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
        return mlir::failure();
    }

    // NCHW tile at C, NHWC tile at H
    const auto tileDim = order.dimAt(1);

    // Fuse batch dim to tile dim
    const auto batchDim = order.dimAt(0);

    // the axis dim on or before the tile dim is not supported
    if (order.dimPos(tileDim) >= order.dimPos(axisDim)) {
        return mlir::failure();
    }

    // no need to adjust if the tile dim is large enough or
    // equal to the max possible dim shape
    const auto maxPossibleDimShape = getTotalSizeBeforeDim(shape, order, axisDim);
    if ((shape[tileDim] >= numActShaves && shape[batchDim] == 1) || shape[tileDim] == maxPossibleDimShape) {
        return mlir::failure();
    }

    const auto nextDim = order.dimAt(2);
    const auto totalSizeBeforeNextDim = getTotalSizeBeforeDim(shape, order, nextDim);
    // gather shape on the tile dim
    for (auto idx : irange(order.dimPos(nextDim))) {
        auto dim = order.dimAt(idx);
        shape[dim] = dim == tileDim ? totalSizeBeforeNextDim : 1;
    }

    if (shape[tileDim] >= numActShaves) {
        return mlir::success();
    }

    // Find the smallest factor which can satisfy multi-shave requirement
    if (nextDim != axisDim) {
        int64_t tileDimShape = shape[tileDim];
        int64_t nextDimShape = shape[nextDim];
        for (auto factor = 2; factor < nextDimShape; factor++) {
            if ((nextDimShape % factor == 0) && (tileDimShape * factor >= numActShaves)) {
                shape[nextDim] = nextDimShape / factor;
                shape[tileDim] = tileDimShape * factor;
                return mlir::success();
            }
        }
    }

    return mlir::failure();
}

mlir::LogicalResult AdjustShapeForSoftmax::matchAndRewrite(VPU::SoftMaxOp softmaxOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", softmaxOp->getName(), softmaxOp->getLoc());

    const auto ctx = getContext();

    const auto inType = softmaxOp.getInput().getType().cast<NDTypeInterface>();
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape();

    auto shape = inShape.toValues();
    auto axisInd = softmaxOp.getAxisInd();

    const auto axisZeroOpt = adjustForAxisZeroOpt(shape, axisInd, inOrder);
    if (mlir::succeeded(axisZeroOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for AxisZeroOpt", shape, axisInd);
    }

    const auto numActShaves = IE::getTotalNumOfActShaveEngines(softmaxOp->getParentOfType<mlir::ModuleOp>());
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, axisInd, inOrder, numActShaves);
    if (mlir::succeeded(multiShaveOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for MultiShaveOpt", shape, axisInd);
    }

    if (mlir::failed(axisZeroOpt) && mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType.changeShape(shape),
                                                         softmaxOp.getInput(), getIntArrayAttr(ctx, shape));
    auto newSoftmaxOp = rewriter.create<VPU::SoftMaxOp>(softmaxOp.getLoc(), reshapeInOp.getResult(),
                                                        getIntAttr(ctx, axisInd), softmaxOp.getPadSizeAttr(), nullptr);
    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType, newSoftmaxOp.getOutput(),
                                                          getIntArrayAttr(ctx, inShape));

    softmaxOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(softmaxOp);

    return mlir::success();
}

std::optional<Dim> getHighestNonOneDim(ShapeRef shape, DimsOrder order) {
    for (auto i : irange(order.numDims())) {
        auto dim = order.dimAt(i);
        if (shape[dim] > 1) {
            return dim;
        }
    }
    return std::nullopt;
}

mlir::LogicalResult adjustForMultiShaveOptGeneric(Shape& shape, const DimsOrder& order, const int64_t numActShaves) {
    // only support NCHW and NHWC layout
    if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
        return mlir::failure();
    }

    const auto origTileDim = getHighestNonOneDim(shape, order);
    // impossible to adjust for shape 1x1x1x1
    if (!origTileDim.has_value()) {
        return mlir::failure();
    }

    // NCHW tile at C, NHWC tile at H
    const auto tileDim = order.dimAt(1);
    const auto dimN = order.dimAt(0);

    // always adjust shape when dim N is not 1 to prevent Clustering strategy
    // no need to adjust if the original tile dim is large enough or equal to the max possible dim shape
    const auto maxPossibleDimShape = shape.totalSize();
    const auto shapeAtTileDim = shape[origTileDim.value()];
    if (shape[dimN] == 1 && (shapeAtTileDim >= numActShaves || shapeAtTileDim == maxPossibleDimShape)) {
        return mlir::failure();
    }

    // gather shape on the tile dim
    for (size_t idx = 0; idx < shape.size(); idx++) {
        auto dim = order.dimAt(idx);
        shape[dim] = dim == tileDim ? maxPossibleDimShape : 1;
    }

    return mlir::success();
}

//
// AdjustShapeForGelu
//
// This rewritter adjusts shape of Gelu by gathering dimensions on the tiling dim for multi-Clusters and multi-SHAVEs
// optimization:
// 1. Shape is adjusted when SW layer has batch, otherwise Clustering strategy would be assigned.
// 2. Shape is adjusted to ensure the dim size of the highest dimension is enough for SHAVEs engines.
class AdjustShapeForGelu final : public mlir::OpRewritePattern<VPU::GeluOp> {
public:
    AdjustShapeForGelu(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GeluOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForGelu");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::GeluOp geluOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order, const int64_t numActShaves) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForGelu::adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order,
                                                               const int64_t numActShaves) const {
    return adjustForMultiShaveOptGeneric(shape, order, numActShaves);
}

mlir::LogicalResult AdjustShapeForGelu::matchAndRewrite(VPU::GeluOp geluOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", geluOp->getName(), geluOp->getLoc());

    const auto ctx = getContext();

    const auto origIOType = geluOp.getOutput().getType().cast<NDTypeInterface>();
    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    const auto numActShaves = IE::getTotalNumOfActShaveEngines(geluOp->getParentOfType<mlir::ModuleOp>());
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, origIOOrder, numActShaves);
    if (mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for MultiShaveOpt at {1}", shape, geluOp->getLoc());

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(geluOp->getLoc(), origIOType.changeShape(shape),
                                                         geluOp.getInput(), getIntArrayAttr(ctx, shape));

    auto newGeluOp = rewriter.create<VPU::GeluOp>(geluOp->getLoc(), reshapeInOp.getResult(), nullptr);

    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(geluOp->getLoc(), origIOType, newGeluOp.getOutput(),
                                                          getIntArrayAttr(ctx, origIOShape));

    geluOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(geluOp);

    return mlir::success();
}

//
// AdjustShapeForMultiply
//
// This rewritter adjusts shape of Multiply by gathering dimensions on the tiling dim for multi-cluster and multi-SHAVEs
// optimization
// 1. Shape is adjusted when SW layer has batch, otherwise Clustering strategy would be assigned.
// 2. Shape is adjusted to ensure the dim size of the highest dimension is enough for SHAVEs engines.
class AdjustShapeForMultiply final : public mlir::OpRewritePattern<VPU::MultiplyOp> {
public:
    AdjustShapeForMultiply(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForMultiply");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::MultiplyOp multiplyOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order, const int64_t numActShaves) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForMultiply::adjustForMultiShaveOpt(Shape& shape, const DimsOrder& order,
                                                                   const int64_t numActShaves) const {
    return adjustForMultiShaveOptGeneric(shape, order, numActShaves);
}

mlir::LogicalResult AdjustShapeForMultiply::matchAndRewrite(VPU::MultiplyOp multiplyOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", multiplyOp->getName(), multiplyOp->getLoc());

    if (multiplyOp.getInput1().getType() != multiplyOp.getInput2().getType()) {
        return mlir::failure();
    }

    auto isConstInput = [](mlir::Value input) {
        return input.getDefiningOp<Const::DeclareOp>() != nullptr;
    };
    if (isConstInput(multiplyOp.getInput1()) || isConstInput(multiplyOp.getInput2())) {
        return mlir::failure();
    }

    const auto ctx = getContext();

    const auto origIOType = multiplyOp.getOutput().getType().cast<NDTypeInterface>();
    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    const auto numActShaves = IE::getTotalNumOfActShaveEngines(multiplyOp->getParentOfType<mlir::ModuleOp>());
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, origIOOrder, numActShaves);
    if (mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for MultiShaveOpt at {1}", shape, multiplyOp->getLoc());

    auto reshapeIn1Op = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origIOType.changeShape(shape),
                                                          multiplyOp.getInput1(), getIntArrayAttr(ctx, shape));

    auto reshapeIn2Op = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origIOType.changeShape(shape),
                                                          multiplyOp.getInput2(), getIntArrayAttr(ctx, shape));

    auto newMultiplyOp =
            rewriter.create<VPU::MultiplyOp>(multiplyOp->getLoc(), reshapeIn1Op.getResult(), reshapeIn2Op.getResult(),
                                             multiplyOp.getAutoBroadcastAttr(), multiplyOp.getPostOpAttr(), nullptr);

    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(multiplyOp->getLoc(), origIOType, newMultiplyOp.getOutput(),
                                                          getIntArrayAttr(ctx, origIOShape));

    multiplyOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(multiplyOp);

    return mlir::success();
}

//
// AdjustShapeForMVN
//

// This rewritter adjusts shape of MVN with batch size larger than one, otherwise Clustering strategy would be assigned
class AdjustShapeForMVN final : public mlir::OpRewritePattern<VPU::MVNOp> {
public:
    AdjustShapeForMVN(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::MVNOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForMVN");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::MVNOp mvnOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, bool& isAcrossChannels, const DimsOrder order) const;

private:
    Logger _log;
};

mlir::LogicalResult AdjustShapeForMVN::adjustForMultiShaveOpt(Shape& shape, bool& isAcrossChannels,
                                                              const DimsOrder order) const {
    const auto N = shape[Dims4D::Act::N];
    if (order != DimsOrder::NCHW || N == 1) {
        return mlir::failure();
    }

    if (isAcrossChannels) {
        shape[Dims4D::Act::H] = shape.totalSize() / N;
        shape[Dims4D::Act::W] = 1;
        shape[Dims4D::Act::C] = N;
        isAcrossChannels = false;
    } else {
        shape[Dims4D::Act::C] = N * shape[Dims4D::Act::C];
    }
    shape[Dims4D::Act::N] = 1;

    return mlir::success();
}

mlir::LogicalResult AdjustShapeForMVN::matchAndRewrite(VPU::MVNOp mvnOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", mvnOp->getName(), mvnOp->getLoc());

    const auto ctx = getContext();

    const auto origIOType = mvnOp.getOutput().getType().cast<NDTypeInterface>();
    const auto origIOOrder = origIOType.getDimsOrder();
    const auto origIOShape = origIOType.getShape();

    auto shape = origIOShape.toValues();

    auto isAcrossChannels = mvnOp.getAcrossChannels();
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, isAcrossChannels, origIOOrder);
    if (mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape {0} to {1} for MultiShaveOpt at {1}", origIOShape, shape, mvnOp->getLoc());

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(mvnOp->getLoc(), origIOType.changeShape(shape),
                                                         mvnOp.getInput(), getIntArrayAttr(ctx, shape));

    auto newMVNOp = rewriter.create<VPU::MVNOp>(mvnOp->getLoc(), reshapeInOp.getResult(),
                                                mlir::BoolAttr::get(ctx, isAcrossChannels),
                                                mvnOp.getNormalizeVarianceAttr(), mvnOp.getEpsAttr());

    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(mvnOp->getLoc(), origIOType, newMVNOp.getOutput(),
                                                          getIntArrayAttr(ctx, origIOShape));

    mvnOp.replaceAllUsesWith(reshapeOutOp.getResult());
    rewriter.eraseOp(mvnOp);

    return mlir::success();
}

//
// AdjustShapeForReduce
//

template <class ReduceOp>
class AdjustShapeForReduce final : public mlir::OpRewritePattern<ReduceOp> {
public:
    AdjustShapeForReduce(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ReduceOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForReduce");
    }

private:
    mlir::LogicalResult matchAndRewrite(ReduceOp reduceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ReduceOp>
mlir::LogicalResult AdjustShapeForReduce<ReduceOp>::matchAndRewrite(ReduceOp reduceOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", reduceOp->getName(), reduceOp->getLoc());

    const auto inType = reduceOp.getInput().getType().template cast<NDTypeInterface>();
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape();

    // For the Reduce Op like something below
    //     VPU.ReduceMin{axes_value = [1], keep_dims} : tensor<1x245760x1x1xf16> -> tensor<1x1x1x1xf16>
    // this op can't be tiled, or split for MC or MS so far. So put non-1 shape on memory inner dimension
    // to make Shave happy.
    //     VPU.ReduceMin{axes_value = [3], keep_dims} : tensor<1x1x1x245760xf16> -> tensor<1x1x1x1xf16>
    //
    const auto hasSingleNonOneDim = llvm::count_if(inShape, [](const auto dim) {
                                        return dim > 1;
                                    }) == 1;
    if (!hasSingleNonOneDim) {
        return mlir::failure();
    }

    const auto axesValue = parseIntArrayAttr<int64_t>(reduceOp.getAxesValue());
    if (axesValue.size() != 1) {
        return mlir::failure();
    }
    auto axisInd = axesValue[0];

    if (inShape[Dim(axisInd)] == 1) {
        _log.nest(1).trace("axis {0} is not matched to non-1 dimension", axisInd);
        return mlir::failure();
    }

    auto targetShape = inShape.toValues();
    const auto axisZeroOpt = adjustForAxisZeroOpt(targetShape, axisInd, inOrder);
    if (mlir::failed(axisZeroOpt)) {
        _log.nest(1).trace("Failed to do AxisZeroOpt with shape {0} and axisInd {1}", targetShape, axisInd);
        return mlir::failure();
    }

    _log.nest(1).trace("Adjusted shape to {0} for zero axis at {1}", targetShape, reduceOp->getLoc());

    const auto outType = reduceOp.getOutput().getType().template cast<NDTypeInterface>();
    const auto ctx = rewriter.getContext();
    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(reduceOp.getLoc(), inType.changeShape(targetShape),
                                                         reduceOp.getInput(), getIntArrayAttr(ctx, targetShape));
    auto newReduceOp =
            rewriter.create<ReduceOp>(reduceOp.getLoc(), reshapeInOp.getResult(),
                                      getIntArrayAttr(ctx, SmallVector<int64_t>{axisInd}), reduceOp.getKeepDimsAttr());
    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(reduceOp.getLoc(), outType, newReduceOp.getOutput(),
                                                          getIntArrayAttr(ctx, outType.getShape()));

    rewriter.replaceOp(reduceOp, reshapeOutOp.getResult());
    return mlir::success();
}

//
// AdjustForOptimizedSwKernelPass
//

// Currently, this pass inserts ShapeCast ops to adjust tensor shape for SW layers to fully utilize SHAVEs
// It would be better to adjust tensor shape with considering sub-graph optimization as well
// See E#119868 for details.
class AdjustForOptimizedSwKernelPass final :
        public VPU::arch37xx::AdjustForOptimizedSwKernelBase<AdjustForOptimizedSwKernelPass> {
public:
    explicit AdjustForOptimizedSwKernelPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void AdjustForOptimizedSwKernelPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustShapeForSoftmax>(&ctx, _log);
    patterns.add<AdjustShapeForGelu>(&ctx, _log);
    patterns.add<AdjustShapeForMultiply>(&ctx, _log);
    patterns.add<AdjustShapeForMVN>(&ctx, _log);

    patterns.add<AdjustShapeForReduce<VPU::ReduceMinOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceMaxOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceMeanOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceSumOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceProdOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceL1Op>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceL2Op>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceLogicalAndOp>>(&ctx, _log);
    patterns.add<AdjustShapeForReduce<VPU::ReduceLogicalOrOp>>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createAdjustForOptimizedSwKernelPass(Logger log) {
    return std::make_unique<AdjustForOptimizedSwKernelPass>(log);
}
