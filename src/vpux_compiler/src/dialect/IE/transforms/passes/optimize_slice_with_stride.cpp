//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

int64_t calculateAlignmentFactor(const vpux::NDTypeInterface sliceInType) {
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(sliceInType.getElementType());

    auto calcFactor = [channelAlignment](int64_t channel) {
        auto leastAlignedChannel = std::lcm(channel, channelAlignment);
        return (leastAlignedChannel / channel);
    };

    return calcFactor(sliceInType.getShape()[Dims4D::Act::C]);
}

IE::ShapeCastOp reshapeConvInput(mlir::Location loc, mlir::Value input, const int64_t channelAlignment,
                                 mlir::PatternRewriter& rewriter) {
    const auto origShape = getShape(input);
    const auto batch = origShape[Dims4D::Act::N];
    const auto channels = origShape[Dims4D::Act::C] * channelAlignment;
    const auto height = origShape[Dims4D::Act::H];
    const auto width = origShape[Dims4D::Act::W] / channelAlignment;

    const SmallVector<int64_t> targetShape = {batch, channels, height, width};

    const auto reshapedLoc = appendLoc(loc, "reshape input for DPU slice");
    return vpux::IE::buildShapeCast(reshapedLoc, input, ArrayRef(targetShape), rewriter);
}

IE::ShapeCastOp reshapeConvOutput(IE::SliceOp origOp, mlir::Value convOutput, mlir::PatternRewriter& rewriter) {
    const Shape origShape = getShape(origOp.getResult()).toValues();
    const SmallVector<int64_t> targetShape = origShape.raw();

    const auto reshapedLoc = appendLoc(origOp.getLoc(), "reshape output for DPU slice");
    return vpux::IE::buildShapeCast(reshapedLoc, convOutput, ArrayRef(targetShape), rewriter);
}

//
// SliceOpConverter
//

class SliceOpConverter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    SliceOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    IE::ConvolutionOp createConvolution(IE::SliceOp origOp, mlir::Value weights, mlir::Value activation,
                                        mlir::Type convOutElemType, mlir::PatternRewriter& rewriter) const;
    mlir::Value composeWeights(IE::SliceOp origOp, const mlir::Type convolutionInputType,
                               const int64_t convolutionAlignment, mlir::PatternRewriter& rewriter) const;
    bool isBeneficialToConvert(IE::SliceOp origOp) const;

    Logger _log;
};

bool SliceOpConverter::isBeneficialToConvert(IE::SliceOp sliceOp) const {
    const auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
    if (sliceOffset.size() != 4) {
        _log.trace("Slice at {0} has {1}-d start padding. Only 4-d shapes are supported", sliceOp.getLoc(),
                   sliceOffset.size());
        return false;
    }

    const auto sliceInType = sliceOp.getSource().getType().cast<vpux::NDTypeInterface>();
    const auto sliceOutType = sliceOp.getResult().getType().cast<vpux::NDTypeInterface>();
    const auto sliceInShape = sliceInType.getShape();
    const auto supportedLayout = DimsOrder::NHWC;
    const auto sliceInLayout = sliceInType.getDimsOrder();
    const auto inputN = sliceInShape[Dims4D::Act::N];
    const auto inputW = sliceInShape[Dims4D::Act::W];

    if (sliceInLayout != supportedLayout) {
        _log.trace("Slice at {0} has {1} input layout, expected {2}", sliceOp.getLoc(), sliceInLayout, supportedLayout);
        return false;
    }

    const auto inputShape = getShape(sliceOp.getSource()).raw();
    const auto outputShape = getShape(sliceOp.getResult()).raw();
    // Only slice on the lowest dim(channel, NHWC layout) will be converted
    for (auto i : irange(inputShape.size())) {
        if (inputShape[i] != outputShape[i] && i != checked_cast<uint32_t>(Dims4D::Act::C.ind())) {
            _log.trace("Slice at {1} is not slice on channel", sliceOp.getLoc());
            return false;
        }
    }

    // Check if the output of slice op cannot fit CMX, it ensure a DDR->DDR copy
    // E#103384::IE dialect should be HW-agnostic as much as possible, here should not depend on VPU/VPUIP dialect.
    // An option is to use interfaces like registerLayerWithPostOpModelInterface for vpux::VPU::getTotalCMXSize(op)
    // for such op which need to this function to check memory size in IE dialect.
    if (sliceOutType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(sliceOp)) {
        _log.trace("Slice at {0} is not a DDR -> DDR copy", sliceOp.getLoc());
        return false;
    }

    // Currently not support quantized slice and we can remove this for quantized slice
    if (sliceInType.getElementType().isa_and_nonnull<mlir::quant::QuantizedType>()) {
        return false;
    }

    if (inputN != 1) {
        _log.trace("Slice at {0} has batch {1}. Expected to have 1", sliceOp.getLoc(), inputN);
        return false;
    }

    // For quantized input, we can remove this and add another rewrite pattern for composed weights and activation.
    if (!sliceInType.getElementType().isF16()) {
        _log.trace("Slice at {0} has {1} element type. Only float16 types are supported", sliceOp.getLoc(),
                   sliceInType.getElementType());
        return false;
    }

    const auto convolutionAlignment = calculateAlignmentFactor(sliceInType);
    const int64_t kernelOutputChannels = getShape(sliceOp.getResult())[Dims4D::Act::C] * convolutionAlignment;
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(sliceInType.getElementType());
    // Here we need to ensure we can borrow factor from W for channel alignment. And if a factor borrowed from W which
    // still cannot satisfy output channel alignment, we need a bigger factor from W to C. It's unefficient because the
    // Conv's channel will be very big
    if (inputW % convolutionAlignment != 0 || kernelOutputChannels % channelAlignment != 0) {
        _log.trace("Slice at {0} cannot borrow suitable factor from W for alignment", sliceOp.getLoc());
        return false;
    }

    // Currently only convert slice from PermuteCastOp otherwise it may lead to performance regression. For example, if
    // slice from a NCE op's output, in optimize copies pass, it may eliminate this DDR to DDR copy. If any other ops
    // need to support, we can extend here.
    auto parentOp = sliceOp.getSource().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::PermuteCastOp, IE::ConvertOp>(parentOp)) {
        return false;
    }

    return true;
}

IE::ConvolutionOp SliceOpConverter::createConvolution(IE::SliceOp origOp, mlir::Value weights, mlir::Value activation,
                                                      mlir::Type convOutElemType,
                                                      mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    // IE::ConvolutionOp output type inference sets NCHW output order.
    // Specify convolution output type explicitly.
    const auto origOutType = origOp.getResult().getType().cast<vpux::NDTypeInterface>();
    const auto weightsShape = getShape(weights);
    const auto outChannels = weightsShape[Dims4D::Filter::OC];
    const Shape convInShape = getShape(activation).toValues();
    const Shape convOutShape = {convInShape[Dims4D::Act::N], outChannels, convInShape[Dims4D::Act::H],
                                convInShape[Dims4D::Act::W]};

    const auto convOutType = origOutType.changeShape(convOutShape).changeElemType(convOutElemType);

    return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), convOutType, activation, weights,
                                              /*bias=*/nullptr, strides, kernelPadsBegin, kernelPadsEnd, dilations,
                                              /*postOp=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr);
}

mlir::Value SliceOpConverter::composeWeights(IE::SliceOp origOp, const mlir::Type convolutionInputType,
                                             const int64_t convolutionAlignment,
                                             mlir::PatternRewriter& rewriter) const {
    const auto origInShape = getShape(origOp.getSource());
    const auto origOutShape = getShape(origOp.getResult());
    const auto origInputChannel = origInShape[Dims4D::Act::C];

    const int64_t kernelOutputChannels = origOutShape[Dims4D::Act::C] * convolutionAlignment;
    const int64_t kernelInputChannels = origInShape[Dims4D::Act::C] * convolutionAlignment;
    const int64_t kernelY = 1;
    const int64_t kernelX = 1;
    const auto weightShape = Shape{kernelOutputChannels, kernelInputChannels, kernelY, kernelX};

    const auto origChannelOffset = parseIntArrayAttr<int64_t>(origOp.getStaticOffsetsAttr())[Dims4D::Act::C.ind()];
    std::vector<vpux::type::float16> weightValues(weightShape.totalSize(), checked_cast<vpux::type::float16>(0.f));

    // For example, Slice:1x9x1088x1920->1x3x1088x1920, offset[0, 1, 0, 0]
    // we can constract weights with shape 3x9x1x1 like this:
    // origChannelOffset
    //   |
    // 0 1 0 0 0 0 0 0 0
    // 0 0 1 0 0 0 0 0 0
    // 0 0 0 1 0 0 0 0 0
    // After we reshaped input for channel alignment, we also need construct new weights.
    // For the tensor in the example, reshaped input to 1x144x1088x120, the kernel with shape 48x144x1x1 will be like:
    // 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...                       |
    // 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...                       | <- blockSize
    // 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...  <- blockRow 0        |
    // 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    // 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    // 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ...  <- blockRow 1
    // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ...
    // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 ...
    // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 ...
    // ...
    // |                 |
    // inChanOffset 0    |
    //              inChanOffset 1
    // Resulting weights can be split into the number of blocks defined by convolutionAlignment.
    // In the example, we have 16 blocks.
    const auto& blockSize = origOutShape[Dims4D::Act::C];
    for (int64_t blockRow = 0; blockRow < convolutionAlignment; blockRow++) {
        const auto blockOffsetIndex = blockRow * blockSize * kernelInputChannels;
        auto inChanOffset = origInputChannel * blockRow;
        for (int64_t i = 0; i < blockSize; i++) {
            const auto index = blockOffsetIndex + inChanOffset + origChannelOffset + i * kernelInputChannels + i;
            weightValues[index] = 1.f;
        }
    }

    const auto ctx = rewriter.getContext();
    const auto weightStorageType = mlir::RankedTensorType::get(weightShape.raw(), mlir::Float16Type::get(ctx));
    const auto weightStorageAttr = mlir::DenseElementsAttr::get(weightStorageType, ArrayRef(weightValues));
    const auto weightContentAttr = Const::ContentAttr::get(weightStorageAttr);
    const auto declLoc = appendLoc(origOp.getLoc(), "weights for DPU slice");

    const auto weightExpressedType = mlir::RankedTensorType::get(weightShape.raw(), convolutionInputType);
    auto declOp = rewriter.create<Const::DeclareOp>(declLoc, weightExpressedType, weightContentAttr);

    const auto reorderLoc = appendLoc(origOp.getLoc(), "reorder weights for DPU slice");
    const auto weightTypeNCHW = declOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto reorderType = weightTypeNCHW.changeDimsOrder(DimsOrder::OYXI);
    const auto orderMap = DimsOrder::OYXI.toAffineMap(ctx);
    auto reorderOut = rewriter.createOrFold<IE::ReorderOp>(reorderLoc, reorderType, declOp.getOutput(), orderMap);

    return reorderOut;
}

// Replace 'Slice' with 'Convolution' which meet the requirements:
// 1. Cannot optimize in previous passes
// 2. DDR->DDR strided copy
// 3. Can use 'ShapeCast' for channel alignment instead of 'Expand'
mlir::LogicalResult SliceOpConverter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Slice op at {0}", origOp->getLoc());

    if (!isBeneficialToConvert(origOp)) {
        _log.trace("Cannot or is not beneficial to convert Slice to Conv");
        return mlir::failure();
    }

    const auto sliceInput = origOp.getSource();
    const auto sliceInType = sliceInput.getType().cast<vpux::NDTypeInterface>();
    const auto convolutionAlignment = calculateAlignmentFactor(sliceInType);

    auto reshapeIn = reshapeConvInput(origOp.getLoc(), sliceInput, convolutionAlignment, rewriter);
    auto weights = composeWeights(origOp, sliceInType.getElementType(), convolutionAlignment, rewriter);
    auto convOp = createConvolution(origOp, weights, reshapeIn.getResult(), sliceInType.getElementType(), rewriter);
    auto reshapeOut = reshapeConvOutput(origOp, convOp.getOutput(), rewriter);

    rewriter.replaceOp(origOp, reshapeOut.getResult());

    _log.trace("Successfully convert IE::SliceOp at {0} to IE::ConvolutionOp", origOp->getLoc());
    return mlir::success();
}

//
// SliceConcatRewriter
//

class SliceConcatRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    SliceConcatRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("SliceConcatRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool doesSliceConcatMeetRequirement(IE::SliceOp sliceOp, IE::ConcatOp concatOp) {
    auto sliceInShape = getShape(sliceOp.getSource());
    auto sliceOutShape = getShape(sliceOp.getResult());
    auto concatOutShape = getShape(concatOp.getOutput());
    if (sliceInShape.size() != 4 || sliceOutShape.size() != 4) {
        return false;
    }

    if (concatOp.getInputs().size() != 2) {
        return false;
    }

    auto inConvOp = sliceOp.getSource().getDefiningOp<IE::ConvolutionOp>();
    if (inConvOp == nullptr) {
        return false;
    }

    // If convolution has other users, slice cannot fuse into convolution.
    if (!inConvOp.getOutput().hasOneUse()) {
        return false;
    }

    Const::DeclareOp constInput = nullptr;
    for (auto input : concatOp.getInputs()) {
        constInput = input.getDefiningOp<Const::DeclareOp>();
        if (constInput != nullptr) {
            break;
        }
    }
    if (constInput == nullptr) {
        return false;
    }

    const auto sliceInputOrder = DimsOrder::fromValue(sliceOp.getSource());
    if (sliceInputOrder != DimsOrder::NHWC) {
        return false;
    }

    // Only slice on channel(lowest dim) will be fused into previous convolution.
    if (sliceInShape[Dims4D::Act::N] != sliceOutShape[Dims4D::Act::N] ||
        sliceInShape[Dims4D::Act::H] != sliceOutShape[Dims4D::Act::H] ||
        sliceInShape[Dims4D::Act::W] != sliceOutShape[Dims4D::Act::W]) {
        return false;
    }

    // If slice's input or concat's output is not aligned, not convert slice-concat pattern due to it may introduce
    // extra DMA for alignment.
    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(sliceOp.getSource().getDefiningOp());
    if (iface == nullptr) {
        return false;
    }
    const auto alignment = iface.getOutputChannelAlignment();
    return sliceInShape[Dims4D::Act::C] % alignment == 0 && concatOutShape[Dims4D::Act::C] % alignment == 0;
}

// Search the pattern like below:
//    Convolution
//         |
//       Slice  Constant
//          \      /
//           Concat
// Fuse the Slice into previous Convolution and convert Concat to Eltwise Add to reduce DMA:
//    Convolution
//         |   Constant(broadcast)
//          \     /
//            Add
mlir::LogicalResult SliceConcatRewriter::matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Concat '{1}' at '{2}'", getDebugName(), concatOp->getName(), concatOp->getLoc());

    auto concatInputs = concatOp.getInputs();
    SmallVector<IE::SliceOp> sliceOps;
    for (auto input : concatInputs) {
        if (input.getDefiningOp<IE::SliceOp>() != nullptr) {
            sliceOps.push_back(input.getDefiningOp<IE::SliceOp>());
        }
    }
    if (sliceOps.size() != 1) {
        return matchFailed(rewriter, concatOp, "Concat is expected to have exactly one sliced input");
    }

    IE::SliceOp sliceOp = sliceOps[0];
    if (!doesSliceConcatMeetRequirement(sliceOp, concatOp)) {
        return matchFailed(rewriter, concatOp, "Slice-Concat does not meet requirements");
    }

    Const::DeclareOp constInput = nullptr;
    for (auto input : concatInputs) {
        constInput = input.getDefiningOp<Const::DeclareOp>();
        if (constInput != nullptr) {
            break;
        }
    }

    // For example:
    //   Input(1x1024x32x32) Filter(512x1024x3x3)            Input(1x1024x32x32)  New_Filter(512x1024x3x3)
    //          \            /                                           \        /
    //         Convolution(1x512x32x32)                     To    Convolution(1x512x32x32) New_Constant(1x512x32x32)
    //                  |                                                              \           /
    //            Slice(1x511x32x32)   Constant(1x1x32x32)                            Add(1x512x32x32)
    //                  \                 /
    //                 Concat(1x512x32x32)
    // The 'New_Filter' is from 'Filter' with attr: #const.SubView<[0, 0, 0, 0], [511, 1024, 3, 3]>,
    // #const.PadWithZero<[0, 0, 0, 0], [1, 0, 0, 0]>]. Here we reconstruct Convolution's weights to let last output
    // channel all zero, which is make valid data consistent with the result after Slice.
    // The 'New_Constant' is from 'Constant' with attr: #const.PadWithZero<[0, 511, 0, 0], [0, 0, 0, 0]>]. After
    // broadcast the 'Constant' and add it with Convolution's output tensor to get original concat result.

    // Fuse Slice into previous Convolution.
    auto inConvOp = sliceOp.getSource().getDefiningOp<IE::ConvolutionOp>();
    auto filter = inConvOp.getFilter();
    auto filterShape = vpux::getShape(filter);
    auto filterCst = filter.getDefiningOp<Const::DeclareOp>();
    if (filterCst == nullptr) {
        return mlir::failure();
    }
    const auto sliceOutputShape = vpux::getShape(sliceOp.getResult());
    auto cstContentAttrFilter = filterCst.getContentAttr();
    const auto subviewOffsets = Shape{0, 0, 0, 0};
    const auto subviewStaticShape = Shape{sliceOutputShape[Dims4D::Act::C], filterShape[Dims4D::Filter::IC],
                                          filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]};
    cstContentAttrFilter = cstContentAttrFilter.subview(subviewOffsets, subviewStaticShape);
    Shape cstPadBegin = {0, 0, 0, 0};
    Shape cstPadEnd = {filterShape[Dims4D::Filter::OC] - sliceOutputShape[Dims4D::Act::C], 0, 0, 0};
    cstContentAttrFilter = cstContentAttrFilter.padWithZero(cstPadBegin, cstPadEnd);
    auto newFilter =
            rewriter.create<Const::DeclareOp>(filterCst.getLoc(), cstContentAttrFilter.getType(), cstContentAttrFilter);
    auto newConvOp = rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            inConvOp, inConvOp.getOutput().getType(), inConvOp.getInput(), newFilter, inConvOp.getBias(),
            inConvOp.getStrides(), inConvOp.getPadsBegin(), inConvOp.getPadsEnd(), inConvOp.getDilations(),
            inConvOp.getPostOpAttr(), inConvOp.getClampAttr(), inConvOp.getStaticScaleAttr());
    sliceOp.getResult().replaceAllUsesWith(sliceOp.getSource());

    // Replace Concat with Add.
    auto constInputAttr = constInput.getContentAttr();
    Shape addCstPadBegin = {0, sliceOutputShape[Dims4D::Act::C], 0, 0};
    Shape addCstPadEnd = {0, 0, 0, 0};
    constInputAttr = constInputAttr.padWithZero(addCstPadBegin, addCstPadEnd);
    auto newConstantInput =
            rewriter.create<Const::DeclareOp>(constInput.getLoc(), constInputAttr.getType(), constInputAttr);
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    rewriter.replaceOpWithNewOp<IE::AddOp>(concatOp, concatOp.getOutput().getType(), newConvOp.getOutput(),
                                           newConstantInput, broadcastType, nullptr, nullptr);
    return mlir::success();
}

//
// FuseSliceWithConvRewriter
//
class FuseSliceWithConvRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    FuseSliceWithConvRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("FuseSliceWithConvRewriter");
    }
    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseSliceWithConvRewriter::matchAndRewrite(IE::SliceOp sliceOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Slice '{1}' at '{2}'", getDebugName(), sliceOp->getName(), sliceOp->getLoc());
    auto inConvOp = sliceOp.getSource().getDefiningOp<IE::ConvolutionOp>();
    if (inConvOp == nullptr) {
        return mlir::failure();
    }
    auto outputLayerUsers = sliceOp.getResult().getUsers();
    auto anyUserIsConcat = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
        return mlir::isa<IE::ConcatOp>(user);
    });

    // If slice's user is concat op, it will processed by 'SliceConcatRewriter'.
    if (anyUserIsConcat) {
        return mlir::failure();
    }
    // If convolution has other users, slice cannot fuse into convolution.
    if (!inConvOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto filter = inConvOp.getFilter();
    auto filterShape = vpux::getShape(filter);
    auto filterCst = filter.getDefiningOp<Const::DeclareOp>();
    if (filterCst == nullptr) {
        return mlir::failure();
    }
    const auto sliceOutputShape = vpux::getShape(sliceOp.getResult());
    auto cstContentAttrFilter = filterCst.getContentAttr();
    const auto subviewOffsets = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets()));
    const auto subviewStaticShape = Shape{sliceOutputShape[vpux::Dims4D::Act::C], filterShape[vpux::Dims4D::Filter::IC],
                                          filterShape[vpux::Dims4D::Filter::KY], filterShape[vpux::Dims4D::Filter::KX]};
    cstContentAttrFilter = cstContentAttrFilter.subview(subviewOffsets, subviewStaticShape);
    auto newFilter =
            rewriter.create<Const::DeclareOp>(filterCst.getLoc(), cstContentAttrFilter.getType(), cstContentAttrFilter);

    // If fused conv cannot be optimized by AdjustConvolutionShape, it will result in performance regession, skip the
    // case.
    const auto adjustConvShapeParameters =
            getAdjustConvShapeParameters(inConvOp, newFilter, Shape(vpux::getShape(sliceOp.getResult())), _log);
    if (mlir::failed(adjustConvShapeParameters)) {
        _log.trace("Not suitable to fuse slice due to not efficient");
        rewriter.eraseOp(newFilter);
        return mlir::failure();
    }

    const auto origOutType = inConvOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto newOutType = origOutType.changeShape(vpux::getShape(sliceOp.getResult()));
    auto newConv = rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            inConvOp, newOutType, inConvOp.getInput(), newFilter, inConvOp.getBias(), inConvOp.getStrides(),
            inConvOp.getPadsBegin(), inConvOp.getPadsEnd(), inConvOp.getDilations(), inConvOp.getPostOpAttr(),
            inConvOp.getClampAttr(), inConvOp.getStaticScaleAttr());
    sliceOp.getResult().replaceAllUsesWith(newConv.getOutput());

    _log.trace("[{0}] Successfully Fuse Slice into Conv '{1}' at '{2}'", getDebugName(), newConv->getName(),
               newConv->getLoc());

    return mlir::success();
}

//
// OptimizeSliceWithStridePass
//

class OptimizeSliceWithStridePass final : public IE::OptimizeSliceWithStrideBase<OptimizeSliceWithStridePass> {
public:
    explicit OptimizeSliceWithStridePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeSliceWithStridePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SliceOpConverter>(&ctx, _log);
    patterns.add<SliceConcatRewriter>(&ctx, _log);
    patterns.add<FuseSliceWithConvRewriter>(&ctx, _log);

    auto func = getOperation();

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createOptimizeSliceWithStridePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeSliceWithStridePass(Logger log) {
    return std::make_unique<OptimizeSliceWithStridePass>(log);
}
