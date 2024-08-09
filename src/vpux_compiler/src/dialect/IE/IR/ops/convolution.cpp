//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

#include <openvino/op/convolution.hpp>

using namespace vpux;

//
// FuseConvAndSlice
//

namespace {

class FuseConvAndSlice final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::ConvolutionOp>::OpRewritePattern;

    void initialize() {
        setDebugName("FuseConvAndSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;
};

//
//     SliceOp
//        |           =>    ConvolutionOp
//    ConvolutionOp
//
// Only support the Slice on DimC
//
mlir::LogicalResult FuseConvAndSlice::matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    auto sliceOp = convOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return matchFailed(rewriter, convOp, "Convolution doesn't have Slice input");
    }
    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    auto sliceSize = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    auto outNDInterface = convOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto outDimOrder = outNDInterface.getDimsOrder();
    auto inNDInterface = convOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto inDimOrder = inNDInterface.getDimsOrder();
    if (inNDInterface.getElementType() != outNDInterface.getElementType() || !inNDInterface.getElementType().isF16()) {
        return matchFailed(rewriter, convOp, "Only handle FP16 case");
    }
    // The channel align interface will return 1 if layout is NCHW
    // Add this condition to promise the channel align interface get valid value
    if (outDimOrder != DimsOrder::NHWC || inDimOrder != DimsOrder::NHWC) {
        return matchFailed(rewriter, convOp, "Only handle NHWC layout");
    }

    auto sliceInput = sliceOp.getSource();
    auto sliceInputShape = vpux::getShape(sliceInput);
    for (size_t index = 0; index < sliceSize.size(); index++) {
        if (static_cast<int64_t>(index) != Dims4D::Act::C.ind() && (sliceSize[index] != sliceInputShape[Dim(index)])) {
            return matchFailed(rewriter, sliceOp, "Only support slice from DimC");
        }
    }

    auto filter = convOp.getFilter();
    auto filterCst = filter.getDefiningOp<Const::DeclareOp>();
    if (filterCst == nullptr) {
        return mlir::failure();
    }

    auto filterShape = vpux::getShape(filter);
    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedChannel = iface.getInputChannelAlignment();
    auto expandSize = vpux::alignValUp(filterShape[Dims4D::Filter::IC], alignedChannel);
    if (sliceInputShape[Dims4D::Act::C] > expandSize) {
        return matchFailed(rewriter, convOp, "Folding cost greater than expand");
    }

    auto cstContentAttrFilter = filterCst.getContentAttr();
    auto dimCPaddingEnd =
            sliceInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC] - sliceOffset[Dims4D::Act::C.ind()];
    Shape cstPadBegin = {0, sliceOffset[Dims4D::Act::C.ind()], 0, 0};
    Shape cstPadEnd = {0, dimCPaddingEnd, 0, 0};
    auto newCstContent = cstContentAttrFilter.padWithZero(cstPadBegin, cstPadEnd);
    auto newFilterConst = rewriter.create<Const::DeclareOp>(convOp.getLoc(), newCstContent.getType(), newCstContent);
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            convOp.getLoc(), outNDInterface, sliceInput, newFilterConst, convOp.getBias(), convOp.getStridesAttr(),
            convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(), convOp.getDilationsAttr(), convOp.getPostOpAttr(),
            convOp.getClampAttr(), convOp.getStaticScaleAttr());

    rewriter.replaceOp(convOp, newConvOp->getOpResults());

    return mlir::success();
}

}  // namespace

//
// Convolution
//

mlir::LogicalResult vpux::IE::ConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvolutionOpAdaptor conv(operands, attrs, prop);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = conv.getInput().getType().cast<mlir::ShapedType>().getShape();
    const auto inType = conv.getInput().getType().cast<mlir::ShapedType>().getElementType();
    const auto filterShape = conv.getFilter().getType().cast<mlir::ShapedType>().getShape();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.getDilations());

    static const auto ChanDim = Dim(1);
    if (inShape[ChanDim.ind()] != filterShape[ChanDim.ind()]) {
        return errorAt(loc, "Channels count of input tensor shape and filter shape must be the same: {0} != {1}",
                       inShape[ChanDim.ind()], filterShape[ChanDim.ind()]);
    }

    const auto op = ov::op::v1::Convolution(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(inShape.begin(), inShape.end())),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                    ov::Shape(filterShape.begin(), filterShape.end())),
            ov::Strides(windowStrides.begin(), windowStrides.end()),
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ov::Strides(windowDilations.begin(), windowDilations.end()));

    const auto& outputShape = op.get_output_partial_shape(0);
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

void vpux::IE::ConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<FuseConvAndBias>(context);
    patterns.add<FuseConvAndSlice>(context);
}

//
// GroupConvolution
//

mlir::LogicalResult vpux::IE::GroupConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupConvolutionOpAdaptor conv(operands, attrs, prop);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = to_small_vector(conv.getInput().getType().cast<mlir::ShapedType>().getShape());
    const auto inType = conv.getInput().getType().cast<mlir::ShapedType>().getElementType();
    auto filterShape = to_small_vector(conv.getFilter().getType().cast<mlir::ShapedType>().getShape());

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.getDilations());

    int64_t groups = 0;
    if (conv.getGroups().value_or(0) != 0) {
        if (filterShape.size() != inShape.size()) {
            return errorAt(loc, "Input size '{0}' does not match filter size '{1}'. (groups != 0)", inShape.size(),
                           filterShape.size());
        }

        groups = conv.getGroups().value();
    } else {
        if (filterShape.size() != inShape.size() + 1) {
            return errorAt(loc, "Input size '{0}' does not match filter size '{1}'. (groups == 0)", inShape.size() + 1,
                           filterShape.size());
        }

        groups = filterShape[0];

        // we need to adjust filters_shape to reuse helpers for normal convolution
        filterShape[1] *= groups;
        filterShape.erase(filterShape.begin());
    }

    inShape[1] /= groups;

    auto op = ov::op::v1::Convolution(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(inShape.begin(), inShape.end())),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                    ov::Shape(filterShape.begin(), filterShape.end())),
            ov::Strides(windowStrides.begin(), windowStrides.end()),
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ov::Strides(windowDilations.begin(), windowDilations.end()));

    const auto& outputShape = op.get_output_partial_shape(0);
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));

    const auto inputType = conv.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outDesc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());

    inferredReturnShapes.emplace_back(shapeI64, inType, outDesc);

    return mlir::success();
}

namespace {

class GroupsToAttr final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::GroupConvolutionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult GroupsToAttr::matchAndRewrite(IE::GroupConvolutionOp convOp,
                                                  mlir::PatternRewriter& rewriter) const {
    if (convOp.getGroups().has_value()) {
        return mlir::failure();
    }

    auto filter = convOp.getFilter();
    auto filterShape = to_small_vector(filter.getType().cast<mlir::ShapedType>().getShape());
    const auto groups = filterShape[0];

    auto getNewShapeValue = [&](mlir::Value input) -> mlir::Value {
        auto shape = to_small_vector(getShape(input).raw());
        shape[1] *= shape[0];
        shape.erase(shape.begin());
        const auto shapeAttr = getIntArrayAttr(getContext(), shape);
        return rewriter.createOrFold<IE::ReshapeOp>(convOp->getLoc(), input, nullptr, false, shapeAttr);
    };

    mlir::Value newFilter = filter;
    if (auto weightsFQ = filter.getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto weightsCst = weightsFQ.getInput().getDefiningOp<Const::DeclareOp>()) {
            auto newWeights = getNewShapeValue(weightsCst);
            auto newInputLow = getNewShapeValue(weightsFQ.getInputLow());
            auto newInputHigh = getNewShapeValue(weightsFQ.getInputHigh());
            auto newOutputLow = getNewShapeValue(weightsFQ.getOutputLow());
            auto newOutputHigh = getNewShapeValue(weightsFQ.getOutputHigh());

            newFilter =
                    rewriter.create<IE::FakeQuantizeOp>(weightsFQ->getLoc(), newWeights, newInputLow, newInputHigh,
                                                        newOutputLow, newOutputHigh, weightsFQ.getLevelsAttr(),
                                                        weightsFQ.getLowFpTypeAttr(), weightsFQ.getAutoBroadcastAttr())
                            .getOutput();
        }
    } else {
        newFilter = getNewShapeValue(filter);
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            convOp, convOp.getInput(), newFilter, convOp.getBias(), convOp.getStridesAttr(), convOp.getPadsBeginAttr(),
            convOp.getPadsEndAttr(), convOp.getDilationsAttr(), getIntAttr(convOp.getContext(), groups),
            convOp.getPostOpAttr(), convOp.getClampAttr());

    return mlir::success();
}

}  // namespace

void vpux::IE::GroupConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.add<FuseConvAndBias>(context);
    patterns.add<GroupsToAttr>(context);
}
