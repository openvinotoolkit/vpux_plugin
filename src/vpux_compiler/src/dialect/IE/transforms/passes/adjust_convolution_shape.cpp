//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {

//
// FoldConvStrideKernel
//

class FoldConvStrideKernel final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FoldConvStrideKernel(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// This pass want to fold the Convolution's stride attribute to 1 in DimW
//  through adjust the input shape and kernel shape.
//  In this way, it will decrease the expand channels to decrease DMA data copy
//  For example:
//          N  H   W  C             N  H   W  C
//    Input 1x128x128x8             1x128x64x16
//          OC Y X IC       =>     OC Y X IC
//    Kernel 2x1x2x8                2x1x1x16
//    Stride   1 2                    1 1
//  In the ExpandActivation pass, it doesn't need expand the input channel
//
mlir::LogicalResult FoldConvStrideKernel::matchAndRewrite(IE::ConvolutionOp convOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto strides = Shape(parseIntArrayAttr<int64_t>(convOp.getStrides()));
    auto filter = convOp.getFilter();
    // Don't need to consider bias, the function not change the output shape.

    auto inNDInterface = convOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto inDimOrder = inNDInterface.getDimsOrder();
    if (DimsOrder::NHWC != inDimOrder) {
        return mlir::failure();
    }

    auto filterConst = filter.getDefiningOp<Const::DeclareOp>();
    if (filterConst == nullptr) {
        return mlir::failure();
    }

    auto filterShape = vpux::getShape(filter);
    if ((1 == strides[Dims4D::Strides::X]) || (filterShape[Dims4D::Filter::KX] > strides[Dims4D::Strides::X])) {
        return mlir::failure();
    }
    auto inputShape = inNDInterface.getShape();
    if (inputShape[Dims4D::Act::W] % strides[Dims4D::Strides::X]) {
        return mlir::failure();
    }

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedChannel = iface.getInputChannelAlignment();
    if ((filterShape[Dims4D::Filter::IC] % alignedChannel) == 0) {
        // Already aligned
        return mlir::failure();
    }
    Shape newShape(inputShape.raw());
    newShape[Dims4D::Act::W] /= strides[Dims4D::Strides::X];
    newShape[Dims4D::Act::C] *= strides[Dims4D::Strides::X];
    const auto ctx = rewriter.getContext();
    const auto dstType = inNDInterface.changeShape(newShape);
    const auto targetShapeAttr = getIntArrayAttr(ctx, newShape.raw());
    auto inputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(convOp.getLoc(), dstType, convOp.getInput(), targetShapeAttr);

    Shape newFilterShape(filterShape.raw());
    newFilterShape[Dims4D::Filter::IC] *= newFilterShape[Dims4D::Filter::KX];
    newFilterShape[Dims4D::Filter::KX] = 1;
    auto cstContentAttrFilter = filterConst.getContentAttr();
    cstContentAttrFilter = cstContentAttrFilter.reshape(newFilterShape);
    if (newShape[Dims4D::Act::C] != newFilterShape[Dims4D::Filter::IC]) {
        int64_t padding = newShape[Dims4D::Act::C] - newFilterShape[Dims4D::Filter::IC];
        cstContentAttrFilter = cstContentAttrFilter.padWithZero({0, 0, 0, 0}, {0, padding, 0, 0});
    }
    auto newFilter =
            rewriter.create<Const::DeclareOp>(convOp.getLoc(), cstContentAttrFilter.getType(), cstContentAttrFilter);

    auto newStride = strides;
    newStride[Dims4D::Strides::X] = 1;
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            convOp, convOp.getType(), inputShapeCastOp, newFilter, convOp.getBias(),
            getIntArrayAttr(ctx, newStride.raw()), convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(),
            convOp.getDilationsAttr(), convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr());
    return mlir::success();
}

//
// AdjustConvShape
//

class AdjustConvShape final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    AdjustConvShape(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value reshapeBias(mlir::PatternRewriter& rewriter, mlir::Value bias, ShapeRef outShape) {
    if (bias == nullptr) {
        return nullptr;
    }
    auto cst = bias.getDefiningOp<Const::DeclareOp>();
    auto biasShape = getShape(bias);
    auto biasCxW = biasShape[Dims4D::Act::C] * biasShape[Dims4D::Act::W];
    auto outCxW = outShape[Dims4D::Act::C] * outShape[Dims4D::Act::W];
    if (biasCxW == 1) {
        return bias;
    }
    auto contentAttr = cst.getContentAttr();
    Shape newOutSahpe(biasShape.raw());
    newOutSahpe[Dims4D::Act::C] = outShape[Dims4D::Act::C];
    newOutSahpe[Dims4D::Act::W] = outShape[Dims4D::Act::W];
    if (biasCxW != outCxW) {
        auto dimValue = outShape[Dims4D::Act::C];
        auto broadCastDim = Dims4D::Act::C;
        if (outShape[Dims4D::Act::C] % biasShape[Dims4D::Act::C]) {
            dimValue = outCxW / biasShape[Dims4D::Act::C];
            broadCastDim = Dims4D::Act::W;
        } else {
            newOutSahpe[Dims4D::Act::W] = biasShape[Dims4D::Act::W];
        }
        contentAttr = contentAttr.broadcast(broadCastDim, dimValue);
    }
    contentAttr = contentAttr.reshape(newOutSahpe);
    return rewriter.create<Const::DeclareOp>(bias.getLoc(), contentAttr.getType(), contentAttr);
}

//
// Avoid expand though adjust the Convolution's Shape
// For example:
//          N  H  W C       N  H  W C
//   Input  1 16 16 3 -+
//                     |-> 1 16 16 3
//   Kernel 3  1  1 3 -+
//             |
//             V
//          N  H  W C        N  H  W C     N  H  W C
//   Input  1 16  1 48 -+
//                      |->  1 16  1 48 -> 1 16 16 3
//   Kernel 48 1  1 48 -+
//
mlir::LogicalResult AdjustConvShape::matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), convOp->getName(), convOp->getLoc());

    auto filter = convOp.getFilter();
    auto filterShape = vpux::getShape(filter);
    auto padBegin = Shape(parseIntArrayAttr<int64_t>(convOp.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(convOp.getPadsEnd()));
    auto inNDInterface = convOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto inputShape = inNDInterface.getShape();
    auto outNDInterface = convOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto outDimOrder = outNDInterface.getDimsOrder();
    const auto ctx = rewriter.getContext();
    const auto strides = Shape(parseIntArrayAttr<int64_t>(convOp.getStrides()));

    const auto adjustConvShapeParameters =
            getAdjustConvShapeParameters(convOp, convOp.getFilter(), Shape(outNDInterface.getShape()), _log);
    if (mlir::failed(adjustConvShapeParameters)) {
        return mlir::failure();
    }

    const auto adjustConvShapeParametersVal = adjustConvShapeParameters.value();
    auto newFilterShape = adjustConvShapeParametersVal.filterShape;
    auto newInputShape = adjustConvShapeParametersVal.inputShape;
    auto newOutputShape = adjustConvShapeParametersVal.outputShape;
    auto borrowFactor = adjustConvShapeParametersVal.borrowFactor;
    auto leftPading = adjustConvShapeParametersVal.filterPading;
    auto padNum = adjustConvShapeParametersVal.padNum;

    auto newFilterICxKX = newFilterShape[Dims4D::Filter::IC] * newFilterShape[Dims4D::Filter::KX];
    auto oldFilterICxKX = filterShape[Dims4D::Filter::IC] * filterShape[Dims4D::Filter::KX];
    Shape middleFilterShape = {filterShape[Dims4D::Filter::OC], oldFilterICxKX, 1, filterShape[Dims4D::Filter::KY]};
    auto cstContentAttrFilter = filter.getDefiningOp<Const::DeclareOp>().getContentAttr();
    const auto totalPading = newFilterICxKX - oldFilterICxKX;
    SmallVector<mlir::Value> filterConst;
    //
    // Construct the new filter
    // For a NHWC layout Conv:
    //        N  H  W C       N  H  W C
    // Input  1 16 16 3 -+
    //                   |-> 1 16 16 3
    // Kernel 3  1  1 3 -+
    //           |
    //           V
    //        N  H  W C        N  H  W C     N  H  W C
    // Input  1 16  1 48 -+
    //                    |-> 1 16  1 48 -> 1 16 16 3
    // Kernel 48 1  1 48 -+
    //
    // The borrowFactor = 16
    // The new kernel:
    //   Padding 0 in input channel to (3x1x1x48)
    //   Concat in output channel to (48x1x1x48)
    //
    for (int64_t i = 0; i < borrowFactor; i++) {
        auto newCstContent = cstContentAttrFilter.reshape(middleFilterShape);
        auto newLeftPading = (leftPading > 0) ? leftPading : 0;
        auto newRightPading = (totalPading > leftPading) ? (totalPading - leftPading) : 0;
        Shape cstPadBegin = {0, newLeftPading, 0, 0};
        Shape cstPadEnd = {0, newRightPading, 0, 0};
        newCstContent = newCstContent.padWithZero(cstPadBegin, cstPadEnd);
        if (newLeftPading + newRightPading > totalPading) {
            Shape offset = {0, (leftPading > 0) ? 0 : -leftPading, 0, 0};
            Shape viewShape(middleFilterShape.raw());
            viewShape[Dims4D::Filter::IC] += totalPading;
            newCstContent = newCstContent.subview(offset, viewShape);
        }
        auto temp = rewriter.create<Const::DeclareOp>(convOp.getLoc(), newCstContent.getType(), newCstContent);
        filterConst.push_back(temp);
        leftPading += filterShape[Dims4D::Filter::IC] * strides[Dims4D::Strides::X];
    }
    auto newFilterConcatOp = rewriter.create<IE::ConcatOp>(convOp.getLoc(), filterConst, Dims4D::Filter::OC);
    auto newFilterType = filter.getType().dyn_cast<vpux::NDTypeInterface>().changeShape(newFilterShape);
    auto newFilter = rewriter.create<IE::ShapeCastOp>(convOp.getLoc(), newFilterType, newFilterConcatOp.getOutput(),
                                                      getIntArrayAttr(ctx, newFilterShape.raw()));

    // Pading on the Dim W already handled by the const construct
    auto newBeginAttr = convOp.getPadsBeginAttr();
    auto padBVect = parseIntArrayAttr<int64_t>(newBeginAttr);
    padBVect[Dims4D::PadsBegin::Left.ind()] = padBegin[Dims4D::PadsBegin::Left] > 0 ? 1 : 0;

    auto newEndAttr = convOp.getPadsEndAttr();
    auto padEVect = parseIntArrayAttr<int64_t>(newEndAttr);
    padEVect[Dims4D::PadsEnd::Right.ind()] = padEnd[Dims4D::PadsEnd::Right] > 0 ? 1 : 0;

    // New Stride
    auto newStride = strides;
    newStride[Dims4D::Strides::X] = 1;

    auto newBias = reshapeBias(rewriter, convOp.getBias(), newOutputShape);

    const auto dstType = inNDInterface.changeShape(newInputShape);
    const auto targetShapeAttr = getIntArrayAttr(ctx, newInputShape.raw());
    auto maybePaddedInput = convOp.getInput();
    if (padNum) {
        // Do the padding
        auto constShape = SmallVector<int64_t>(inputShape.raw());
        constShape[Dims4D::Act::W.ind()] = padNum;
        SmallVector<mlir::Value> valueRange;
        valueRange.push_back(convOp.getInput());
        valueRange.push_back(
                vpux::IE::createPaddingConstForConcat(constShape, convOp->getLoc(), inNDInterface, 0.0f, rewriter));
        maybePaddedInput = rewriter.create<IE::ConcatOp>(convOp.getLoc(), valueRange, Dims4D::Act::W.ind()).getOutput();
    }
    auto inputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(convOp.getLoc(), dstType, maybePaddedInput, targetShapeAttr);
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            convOp.getLoc(), inputShapeCastOp, newFilter, newBias, getIntArrayAttr(ctx, newStride),
            getIntArrayAttr(ctx, padBVect), getIntArrayAttr(ctx, padEVect), convOp.getDilationsAttr(),
            convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr());
    changeDimsOrder(newConvOp, outDimOrder, _log.nest());
    const auto outShapeAttr = getIntArrayAttr(ctx, outNDInterface.getShape().raw());
    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(convOp, outNDInterface, newConvOp.getOutput(), outShapeAttr);
    _log.trace("Successfully adjusted convolution shape");
    return mlir::success();
}

//
// AdjustConvolutionShapePass
//

class AdjustConvolutionShapePass final : public IE::AdjustConvolutionShapeBase<AdjustConvolutionShapePass> {
public:
    explicit AdjustConvolutionShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldConvStrideKernel>(&ctx, _log);
    patterns.add<AdjustConvShape>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAdjustConvolutionShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionShapePass>(log);
}
