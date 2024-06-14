//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

#include <openvino/op/convolution.hpp>

using namespace vpux;

namespace {

// TODO: needs find suitable implict reshape value. Ticket: E#78751
constexpr int64_t CONVOLUTION_INPUT_SHAPE_ALIGNMENT = 4;

//
// ReshapeSingleConstDWConvInput
//

class ReshapeSingleConstDWConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeSingleConstDWConvInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value getReshapedConst(Const::DeclareOp constOp, ShapeRef shape, mlir::PatternRewriter& rewriter) {
    auto contentAttr = constOp.getContentAttr();
    auto constOutputType = constOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto offset = Shape(shape.size(), 0);
    contentAttr = contentAttr.subview(offset, shape);
    constOutputType = constOutputType.changeShape(shape);
    return rewriter.create<Const::DeclareOp>(constOp.getLoc(), constOutputType, contentAttr).getOutput();
}

mlir::LogicalResult ReshapeSingleConstDWConvInput::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    // Adjust from C to H and W like [1, C, 1, 1] -> [1, C/16, 4, 4]

    const auto ctx = origOp->getContext();
    if (!IE::groupConvIsEltwise(origOp)) {
        return matchFailed(rewriter, origOp, "Not a valid groupConv");
    }

    const auto inputShape = getShape(origOp.getInput());
    const auto filterShape = getShape(origOp.getFilter());
    const auto filterConst = mlir::cast<Const::DeclareOp>(origOp.getFilter().getDefiningOp());

    const auto inputElementType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto filterElementType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outputElementType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();

    // Could not support adjust from C to H or W for quantize type since each channel may have different quantize
    // parameter
    if (inputElementType.isa<mlir::quant::QuantizedType>() || filterElementType.isa<mlir::quant::QuantizedType>() ||
        outputElementType.isa<mlir::quant::QuantizedType>()) {
        return matchFailed(rewriter, origOp, "Could not support quantize");
    }

    if (origOp.getPostOpAttr() != nullptr) {
        return matchFailed(rewriter, origOp, "Could not support grouConv with post op");
    }

    auto outputLayout = origOp.getOutput().getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    if (outputLayout != DimsOrder::NHWC) {
        return matchFailed(rewriter, origOp, "Could not support the output order");
    }

    const auto padStart = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto nonZeroPadStart = llvm::any_of(padStart, [](auto pad) {
        return pad > 0;
    });
    const auto nonZeroPadEnd = llvm::any_of(padEnd, [](auto pad) {
        return pad > 0;
    });
    if (nonZeroPadStart || nonZeroPadEnd) {
        return matchFailed(rewriter, origOp, "Could no support pad");
    }

    // Current logic only works with input and filter shape with 4 dimensions
    const int rank4D = 4;
    if (inputShape.size() != rank4D || filterShape.size() != rank4D) {
        return matchFailed(rewriter, origOp, "Input shape size is not 4");
    }

    auto calcShapeAligned = [&](int64_t divider, int64_t dimValue) {
        if (dimValue == 1) {
            divider *= CONVOLUTION_INPUT_SHAPE_ALIGNMENT;
            dimValue = CONVOLUTION_INPUT_SHAPE_ALIGNMENT;
        }
        return std::make_tuple(divider, dimValue);
    };

    int64_t divider = 1;
    int64_t widthReshaped, heightReshaped, channelReshaped;
    std::tie(divider, widthReshaped) = calcShapeAligned(divider, inputShape[Dims4D::Act::W]);
    std::tie(divider, heightReshaped) = calcShapeAligned(divider, inputShape[Dims4D::Act::H]);
    if (divider == 1) {
        return matchFailed(rewriter, origOp, "Don't need to align");
    }

    if (inputShape[Dims4D::Act::C] % divider != 0) {
        return matchFailed(rewriter, origOp, "Input shape could not be divided {0}", inputShape[Dims4D::Act::C]);
    }

    channelReshaped = inputShape[Dims4D::Act::C] / divider;
    auto alignIface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    if (channelReshaped % alignIface.getInputChannelAlignment() != 0) {
        return matchFailed(rewriter, origOp, "Remaining channel is not aligned.");
    }

    // Reshape Input
    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], channelReshaped, heightReshaped,
                                             widthReshaped};
    auto inShapeCast =
            rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getInput(), getIntArrayAttr(ctx, newInShape));

    // Reshape Filter
    Shape newFilterShape{channelReshaped, filterShape[Dims4D::Filter::IC], filterShape[Dims4D::Filter::KY],
                         filterShape[Dims4D::Filter::KX]};
    auto newConstFilter = getReshapedConst(filterConst, newFilterShape, rewriter);

    // Reshape Bias
    auto bias = origOp.getBias();
    if (bias != nullptr) {
        auto biasConst = mlir::cast<Const::DeclareOp>(bias.getDefiningOp());
        auto biasShape = getShape(bias);
        Shape newBiasShape{biasShape[Dims4D::Act::N], channelReshaped, biasShape[Dims4D::Act::H],
                           biasShape[Dims4D::Act::W]};
        bias = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                getReshapedConst(biasConst, newBiasShape, rewriter));
    }

    auto newGroupAttr = getIntAttr(ctx, channelReshaped);
    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            origOp->getLoc(), inShapeCast.getResult(), newConstFilter, bias, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEnd(), origOp.getDilationsAttr(), newGroupAttr,
            origOp.getPostOpAttr(), origOp.getClampAttr());
    auto origOutputType = origOp.getType().cast<vpux::NDTypeInterface>();
    newGroupConv.getOutput().setType(
            mlir::cast<mlir::RankedTensorType>(origOutputType.changeShape(getShape(newGroupConv.getOutput()))));

    auto outShape = getShape(origOp.getOutput()).raw();
    auto outShapeCast = rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), newGroupConv.getOutput(),
                                                         getIntArrayAttr(ctx, outShape));

    rewriter.replaceOp(origOp, outShapeCast.getResult());

    return mlir::success();
}

//
// ReshapeConvInput
//

template <typename ConcreteOp>
class ReshapeConvInput final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReshapeConvInput(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult ReshapeConvInput<ConcreteOp>::matchAndRewrite(ConcreteOp convOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    /*
        Convert 1x1 convolution from
            input          filter                    input               filter
        [1, C, H, 1]     [OC, C, 1, 1]             [1, C, H, 1]        [OC, C, 1, 1]
              \             /                =>        |                   |
                   Conv                            AffineReshape           |
               [1, OC, H, 1]                     [1, C, H/4, 4]            |
                                                       \                  /
                                                              Conv
                                                        [1, OC, H/4, 4]
                                                               |
                                                          AffineReshape
                                                          [1, OC, H, 1]
    */
    auto ctx = convOp->getContext();
    const auto inputShape = getShape(convOp.getInput());
    const auto filterShape = getShape(convOp.getFilter());

    // Current logic only works with input and filter shape with 4 dimensions
    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    // check suitable 1x1 convolution with input width = 1, strides = [1, 1]
    if (inputShape[Dims4D::Act::W] != 1 || filterShape[Dims4D::Filter::KX] != 1 ||
        filterShape[Dims4D::Filter::KY] != 1) {
        return mlir::failure();
    }

    const auto strides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    auto stridesEqualToOne = llvm::all_of(strides, [](const int64_t elem) {
        return elem == 1;
    });
    if (!stridesEqualToOne) {
        return mlir::failure();
    }

    int64_t convolutionInputShapeAlignment = CONVOLUTION_INPUT_SHAPE_ALIGNMENT;
    // Find another factor if input height is not divisible by 4
    if (inputShape[Dims4D::Act::H] % CONVOLUTION_INPUT_SHAPE_ALIGNMENT != 0) {
        int64_t val = inputShape[Dims4D::Act::H];
        int64_t factor = sqrt(val);
        for (; factor > 1; factor--) {
            if (val % factor == 0) {
                convolutionInputShapeAlignment = factor;
                break;
            }
        }

        if (factor == 1) {
            return mlir::failure();
        }
    }

    _log.trace("Adjust input shape for convolution at '{0}'", convOp->getLoc());
    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                             inputShape[Dims4D::Act::H] / convolutionInputShapeAlignment,
                                             convolutionInputShapeAlignment};

    const auto inputShapeAttr = getIntArrayAttr(convOp->getContext(), newInShape);
    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind()},
                                                   {Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                                                   {Dims4D::Act::W.ind()}};
    auto newInput = rewriter.create<IE::AffineReshapeOp>(convOp->getLoc(), convOp.getInput(),
                                                         getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);
    mlir::IRMapping mapper;
    mapper.map(convOp.getInput(), newInput.getOutput());
    auto newConvOp = mlir::dyn_cast<ConcreteOp>(rewriter.clone(*convOp, mapper));

    auto outputShape = getShape(convOp.getOutput());
    auto newOutputShape = Shape(SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C],
                                                     outputShape[Dims4D::Act::H] / convolutionInputShapeAlignment,
                                                     outputShape[Dims4D::Act::W] * convolutionInputShapeAlignment});

    auto newOutputType = newConvOp.getOutput().getType().template cast<vpux::NDTypeInterface>();
    newOutputType = newOutputType.changeShape(newOutputShape);
    newConvOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(newOutputType));
    const auto outShape = getShape(convOp.getOutput()).raw();
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);

    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(convOp, newConvOp.getOutput(),
                                                     getIntArrayOfArray(ctx, outDimMapping), outShapeAttr);

    return mlir::success();
}

//
// ReshapeExpandDWConvInput
//

class ReshapeExpandDWConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeExpandDWConvInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReshapeExpandDWConvInput::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());

    // Only support GroupConvolution with constant filter
    // Kernel size must be 1x1, and must be a depthwise convolution.
    auto kernelShape = getShape(origOp.getFilter());
    if (kernelShape[Dims4D::Filter::KX] != 1 || kernelShape[Dims4D::Filter::KX] != 1 ||
        kernelShape[Dims4D::Filter::OC] != origOp.getGroups().value()) {
        return mlir::failure();
    }
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };
    if (!VPU::NCEDepthConvolutionOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }
    // Check stride
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    if (strides[Dims4D::Strides::X.ind()] > 1 || strides[Dims4D::Strides::Y.ind()] == 1) {
        return mlir::failure();
    }
    auto parentExpandOp = origOp.getInput().getDefiningOp<IE::ExpandOp>();
    if (parentExpandOp == nullptr) {
        return mlir::failure();
    }
    if (!parentExpandOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    if (iface == nullptr) {
        return mlir::failure();
    }
    const auto alignment = iface.getInputChannelAlignment();

    const auto unExpandedInput = parentExpandOp.getInput();
    const auto unExpandedType = unExpandedInput.getType().cast<vpux::NDTypeInterface>();
    auto unExpandedShape = Shape(unExpandedType.getShape().toValues());

    auto IN = unExpandedShape[Dims4D::Act::N];
    auto IC = unExpandedShape[Dims4D::Act::C];
    auto IH = unExpandedShape[Dims4D::Act::H];
    auto IW = unExpandedShape[Dims4D::Act::W];

    if (IC % alignment == 0) {
        _log.trace("Channel is already aligned");
        return mlir::failure();
    }
    // Check if can align
    if (IC * IW % alignment != 0) {
        _log.trace("Channel cannot be aligned");
        return mlir::failure();
    }

    auto constInput = origOp.getFilter().getDefiningOp<Const::DeclareOp>();
    auto realDataSizeResult = vpux::IE::getBaseContentNumElements(constInput);
    auto activationDataSize =
            std::accumulate(unExpandedShape.begin(), unExpandedShape.end(), int64_t(1), std::multiplies<int64_t>());
    if (mlir::failed(realDataSizeResult) ||
        (realDataSizeResult.value() != 1 && realDataSizeResult.value() != activationDataSize)) {
        _log.trace("Unsupported const input {0} at {1}", constInput->getName(), constInput->getLoc());
        return mlir::failure();
    }

    // Insert ShapeCast to align input shape
    // For example: 1x3x640x640 -> 1x48x640x40
    auto newIC = alignment * IC;
    auto newIW = IC * IW / newIC;
    auto alignedInShape = Shape({IN, newIC, IH, newIW});
    auto alignedInputShapeAttr = getIntArrayAttr(rewriter.getContext(), alignedInShape);
    const auto dstType = unExpandedType.changeShape(ShapeRef(alignedInShape));

    auto shapeCastInputOp =
            rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), dstType, unExpandedInput, alignedInputShapeAttr);

    auto contentAttr = constInput.getContentAttr();
    auto baseContent = contentAttr.getBaseContent();
    auto dataShape = getShape(constInput.getOutput()).toValues();

    Const::ContentAttr newContentAttr = Const::ContentAttr::get(baseContent);
    Shape realDataShape = baseContent.getShapedType().getShape();

    auto newConstOutputType = constInput.getOutput().getType().cast<vpux::NDTypeInterface>();
    newContentAttr = newContentAttr.broadcast(Dims4D::Act::N, alignedInShape[Dims4D::Act::C]);
    auto newConstantShape = Shape(newConstOutputType.getShape().size(), int64_t(1));
    newConstantShape[Dims4D::Act::N] = alignedInShape[Dims4D::Act::C];
    newConstOutputType = newConstOutputType.changeShape(newConstantShape);
    newContentAttr = newContentAttr.reshape(newConstantShape);

    for (auto& attr : contentAttr.getTransformations()) {
        if (attr.isa<Const::PadWithZeroAttr>() || attr.isa<Const::BroadcastAttr>()) {
            // skip the attributes that the contentAttr already contains
            continue;
        }
        if (attr.isa<Const::ReshapeAttr>()) {
            // Only remain the reshape attribute when it's used for dimension expansion to 4D,
            // and for dimension shrink from 5D to 4D
            // e.g., from [1x512] to [1x1x1x512]
            auto reshapeAttr = attr.cast<Const::ReshapeAttr>();
            auto reshapeShape = Shape(parseIntArrayAttr<int64_t>(reshapeAttr.getShape()));
            if (vpux::IE::isNotDimExpansionReshape(realDataShape, reshapeShape) &&
                vpux::IE::isNotDimShrinkReshape(realDataShape, reshapeShape)) {
                continue;
            }
        }
        newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
    }

    auto newConstInput = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newConstOutputType, newContentAttr);

    // Infer group conv output shape
    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    auto convInShape = to_small_vector(alignedInShape.raw());
    convInShape[1] /= newIC;
    auto filterShape = to_small_vector(newConstOutputType.getShape().raw());

    const auto op =
            ov::op::v1::Convolution(std::make_shared<ov::op::v0::Parameter>(
                                            ov::element::i32, ov::Shape(convInShape.begin(), convInShape.end())),
                                    std::make_shared<ov::op::v0::Parameter>(
                                            ov::element::i32, ov::Shape(filterShape.begin(), filterShape.end())),
                                    ov::Strides(windowStrides.begin(), windowStrides.end()),
                                    ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                    ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                    ov::Strides(windowDilations.begin(), windowDilations.end()));

    const auto& outputShape = op.get_output_partial_shape(0);
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    const auto origOutType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto convOutType = unExpandedType.changeShapeElemType(Shape(shapeI64), origOutType);
    auto groupsAttr = getIntAttr(rewriter, newIC);
    auto grpConv = rewriter.create<IE::GroupConvolutionOp>(
            origOp->getLoc(), convOutType, shapeCastInputOp, newConstInput, origOp.getBias(), origOp.getStridesAttr(),
            origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilationsAttr(), groupsAttr,
            /*post_opAttr=*/nullptr, /*clamp=*/nullptr);

    // Insert ShapeCast to reshape the output to original outShape
    auto unExpandedOutShape = Shape({IN, IC, convOutType.getShape()[Dims4D::Act::H], IW});
    auto shapeCastOutputAttr = getIntArrayAttr(rewriter.getContext(), unExpandedOutShape);
    auto shapeCastOutputOp = rewriter.create<IE::ShapeCastOp>(
            origOp->getLoc(), convOutType.changeShape(unExpandedOutShape), grpConv, shapeCastOutputAttr);

    auto newOutputExpandOp = rewriter.create<IE::ExpandOp>(
            origOp->getLoc(), shapeCastOutputOp, parentExpandOp.getPadsBeginAttr(), parentExpandOp.getPadsEndAttr());

    // Replace with new sub graph
    rewriter.replaceOp(origOp, newOutputExpandOp->getResult(0));

    return mlir::success();
}

//
// AdjustConvolutionInputShape
//

class AdjustConvolutionInputShapePass final :
        public IE::AdjustConvolutionInputShapeBase<AdjustConvolutionInputShapePass> {
public:
    explicit AdjustConvolutionInputShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionInputShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // Adjust between H and W like [1, C, H, W]  -> [1, C, H/4, W*4]
    patterns.add<ReshapeConvInput<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ReshapeConvInput<IE::GroupConvolutionOp>>(&ctx, _log);

    // Adjust from C to H and W like [1, C, 1, 1] -> [1, C/16, 4, 4]
    patterns.add<ReshapeSingleConstDWConvInput>(&ctx, _log);

    // Adust between C and H/W like [1, C, H, W] -> [1, C*4, H, W/4]
    // Also need stride[H] > 1
    patterns.add<ReshapeExpandDWConvInput>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertFCToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionInputShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionInputShapePass>(log);
}
