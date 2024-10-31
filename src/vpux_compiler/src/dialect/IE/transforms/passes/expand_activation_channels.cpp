//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace vpux {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape) {
    Shape padsEnd(origShape.size());

    for (auto i : irange(origShape.size())) {
        const auto d = Dim(i);
        padsEnd[d] = extendedShape[d] - origShape[d];
    }

    return padsEnd;
}

Shape calcPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    extendedShape[Dims4D::Act::C] = alignValUp(origShape[Dims4D::Act::C], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

Shape getInputPads(vpux::NDTypeInterface origType, const bool transposeA, const int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    // IE.MatMul {transposeA = false} in1 = [B, ..., N, IC], in2 = [B, ..., IC, OC]
    // IC is the last dimension in in1 shape
    // IE.MatMul {transposeA = true}  in1 = [B, ..., IC, N], in2 = [B, ..., IC, OC]
    // IC is the penultimate dimension in in1 shape
    const auto rank = origShape.size();
    VPUX_THROW_UNLESS(rank >= 2, "Matrix must have rows and columns. Got rank {0}", rank);
    const auto expandDim = transposeA ? Dim(rank - 2) : Dim(rank - 1);
    extendedShape[expandDim] = alignValUp(origShape[expandDim], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

std::pair<Shape, Shape> getFilterPads(vpux::NDTypeInterface origType, const bool transposeB,
                                      const int64_t inChannelAlignment, const int64_t outChannelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    // IE.MatMul {transposeB = false} in1 = [B, ..., N, IC], in2 = [B, ..., IC, OC]
    // IC is the penultimate dimension in in2 shape, OC is the last
    // IE.MatMul {transposeB = true}  in1 = [B, ..., N, IC], in2 = [B, ..., OC, IC]
    // IC is the last dimension in in2 shape, OC is the penultimate
    const auto rank = origShape.size();
    VPUX_THROW_UNLESS(rank >= 2, "Matrix must have rows and columns. Got rank {0}", rank);
    const auto dimIC = transposeB ? Dim(rank - 1) : Dim(rank - 2);
    const auto dimOC = transposeB ? Dim(rank - 2) : Dim(rank - 1);
    extendedShape[dimIC] = alignValUp(origShape[dimIC], inChannelAlignment);
    auto padsEndIC = calcPadsEnd(origShape, extendedShape);
    auto extendedShapeOc = extendedShape;
    extendedShapeOc[dimOC] = alignValUp(origShape[dimOC], outChannelAlignment);

    return std::make_pair(padsEndIC, calcPadsEnd(extendedShape, extendedShapeOc));
}

bool needsPadding(const int64_t dim) {
    return dim != 0;
}

SmallVector<int64_t> extractMeaningfulOutput(mlir::Operation* origOp, ShapeRef outPadsEnd) {
    SmallVector<int64_t> offsets(outPadsEnd.size(), 0);
    auto sliceOp = origOp->getOperand(0).template getDefiningOp<IE::SliceOp>();
    if (sliceOp != nullptr) {
        auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        const auto sliceChannelOffset = sliceOffsets[Dims4D::Act::C.ind()];
        if (sliceChannelOffset < outPadsEnd[Dims4D::Act::C]) {
            offsets[Dims4D::Act::C.ind()] = sliceChannelOffset;
        } else {
            offsets[Dims4D::Act::C.ind()] = outPadsEnd[Dims4D::Act::C];
        }
    }

    return offsets;
}

mlir::Value expandWithOffset(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, IE::SliceOp sliceOp,
                             mlir::Value expandValue, ShapeRef inPadsEnd, size_t expandDim) {
    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    auto padBegin = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    auto padEnd = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    auto sliceOffsets = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    if (sliceOp != nullptr) {
        sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    }
    const auto sliceChannelOffset = sliceOffsets[Dims4D::Act::C.ind()];

    if (sliceChannelOffset < inPadsEnd[Dim(expandDim)]) {
        padBegin[expandDim] = sliceChannelOffset;
        padEnd[expandDim] = inPadsEnd[Dim(expandDim)] - sliceChannelOffset;
    } else {
        padBegin[expandDim] = inPadsEnd[Dim(expandDim)];
    }

    return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), expandValue,
                                               getIntArrayAttr(rewriter, ArrayRef(padBegin)),
                                               getIntArrayAttr(rewriter, ArrayRef(padEnd)));
}

mlir::Value paddingChannel(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                           ShapeRef padsEnd, size_t expandDim) {
    auto sliceOp = origOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), expandValue, std::nullopt, padsEnd);
    }

    return expandWithOffset(rewriter, origOp, sliceOp, expandValue, padsEnd, expandDim);
}

mlir::Value paddingFilter(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                          Shape padsEnd) {
    auto sliceOp = origOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), expandValue, std::nullopt, ShapeRef(padsEnd));
    }
    auto firstExpand = expandWithOffset(rewriter, origOp, sliceOp, expandValue, padsEnd, Dims4D::Act::C.ind());

    padsEnd[Dims4D::Filter::IC] = 0;
    return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), firstExpand, std::nullopt, ShapeRef(padsEnd));
}

mlir::Value concatWithZeroConst(mlir::Location loc, mlir::Value filter, ShapeRef subInput, int64_t sliceChannelOffset,
                                mlir::PatternRewriter& rewriter) {
    const auto filterType = filter.getType().cast<vpux::NDTypeInterface>();

    auto firstPadShape = to_small_vector(filterType.getShape());
    auto secondPadShape = to_small_vector(filterType.getShape());

    if (sliceChannelOffset <= subInput[Dims4D::Filter::IC]) {
        firstPadShape[Dims4D::Filter::IC.ind()] = sliceChannelOffset;
        secondPadShape[Dims4D::Filter::IC.ind()] = subInput[Dims4D::Filter::IC] - sliceChannelOffset;
    } else {
        firstPadShape[Dims4D::Filter::IC.ind()] = subInput[Dims4D::Filter::IC];
        secondPadShape[Dims4D::Filter::IC.ind()] = 0;
    }

    auto const generateZeroConst = [&](ShapeRef padShape) {
        const auto padType = filterType.changeShape(ShapeRef(padShape));
        const auto eleType = padType.getElementType();

        const auto getEleStorageType = [&]() {
            if (const auto quantizedType = eleType.dyn_cast<mlir::quant::QuantizedType>()) {
                return normalizeQuantStorageType(quantizedType);
            } else {
                return eleType;
            }
        };
        const auto storageElementType = getEleStorageType();

        auto outputBuffer = Const::Content::allocTempBuffer(padType, storageElementType, false);
        outputBuffer.fillWithZero();

        const auto dataType = padType.changeElemType(storageElementType).cast<mlir::RankedTensorType>();
        mlir::DenseElementsAttr eleAttr;
        const auto getDataAttr = [&](auto buffer) {
            eleAttr = mlir::DenseElementsAttr::get(dataType, buffer);
        };
        outputBuffer.mutate(getDataAttr);

        return rewriter.create<Const::DeclareOp>(loc, padType, Const::ContentAttr::get(eleAttr)).getOutput();
    };

    SmallVector<mlir::Value> concatInput;
    if (sliceChannelOffset > 0) {
        concatInput.push_back(generateZeroConst(ShapeRef(firstPadShape)));
    }
    concatInput.push_back(filter);
    if (secondPadShape[Dims4D::Filter::IC.ind()] != 0) {
        concatInput.push_back(generateZeroConst(ShapeRef(secondPadShape)));
    }
    auto concatOp = rewriter.create<IE::ConcatOp>(loc, concatInput, Dims4D::Filter::IC);

    return concatOp.getOutput();
}

mlir::Value padConvFilter(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, const int64_t inChanPadEnd,
                          const int64_t outChanPadEnd, const Logger& log) {
    auto filterOperand = origOp->getOperand(1);
    if (inChanPadEnd == 0 && outChanPadEnd == 0) {
        return filterOperand;
    }

    auto inputOperand = origOp->getOperand(0);
    auto inputSliceOp = inputOperand.getDefiningOp<IE::SliceOp>();

    auto filterShape = filterOperand.getType().cast<vpux::NDTypeInterface>().getShape();
    Shape filterPadsEnd(filterShape.size(), 0);
    filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;
    filterPadsEnd[Dims4D::Filter::IC] = inChanPadEnd;

    auto filterOp = filterOperand.getDefiningOp();

    bool isConstFilter = mlir::isa_and_nonnull<Const::DeclareOp>(filterOp);
    if (!isConstFilter) {
        if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(filterOp)) {
            const auto fqInputConstOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
            isConstFilter = fqInputConstOp != nullptr;
        }
    }

    // E#72287: Convert ExpandOp to const Concat in VPUIP, ExpandOp is preferred in IE for optimization.
    const auto expandTensor = [&](mlir::Value filter, ShapeRef pad, IE::SliceOp sliceOp) {
        auto sliceOffsets = mlir::SmallVector<int64_t>(pad.size(), 0);
        auto sliceChannelOffset = 0;
        if (sliceOp != nullptr) {
            sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            sliceChannelOffset = sliceOffsets[Dims4D::Act::C.ind()];
        }
        return concatWithZeroConst(origOp->getLoc(), filter, pad, sliceChannelOffset, rewriter);
    };

    mlir::Value paddedFilter;
    if (!isConstFilter && inChanPadEnd != 0 && outChanPadEnd == 0) {
        // 1 dim expand for non-const filter
        log.trace("Pad non-const filter in IC at '{0}'", origOp->getLoc());
        paddedFilter = expandTensor(filterOperand, filterPadsEnd, inputSliceOp);

    } else if (!isConstFilter && inChanPadEnd != 0 && outChanPadEnd != 0) {
        // 2 dims expand for non-const filter
        log.trace("Pad non-const filter in IC & OC at '{0}'", origOp->getLoc());

        mlir::Value paddedFilter1;
        Shape filterPadsEnd1(filterShape.size(), 0);
        filterPadsEnd1[Dims4D::Filter::IC] = inChanPadEnd;
        paddedFilter1 = expandTensor(filterOperand, filterPadsEnd1, inputSliceOp);

        Shape filterPadsEnd2(filterShape.size(), 0);
        filterPadsEnd2[Dims4D::Filter::OC] = outChanPadEnd;
        paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), paddedFilter1, std::nullopt,
                                                           ShapeRef(filterPadsEnd2));
    } else {
        // Const filter expand or expand on OC only
        paddedFilter = paddingFilter(origOp, rewriter, filterOperand, std::move(filterPadsEnd));
    }
    return paddedFilter;
}

//
// generalRewrite
//

// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input
// calcOutputSliceOffset - function, calcualte output slice offset, it's different for Conv and per-channel ops
//

mlir::LogicalResult generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator,
                                   FuncRef<SmallVector<int64_t>(mlir::Operation*, Shape)> calcOutputSliceOffset,
                                   Logger log) {
    auto* ctx = origOp->getContext();

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp);

    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    const auto inPadsEnd = calcPadsEnd(inputType, iface.getInputChannelAlignment());
    const auto outPadsEnd = calcPadsEnd(outputType, iface.getOutputChannelAlignment());

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    constexpr std::string_view outChanAttrName = "output_channels";
    constexpr std::string_view inChanAttrName = "input_channels";

    if (inPadsEnd[Dims4D::Act::C] == 0 && outPadsEnd[Dims4D::Act::C] == 0) {
        if (VPU::hasOnlyOutPadding(getModuleOp(origOp)) && !origOp->hasAttr(outChanAttrName)) {
            const auto outChanBeforeAttr =
                    vpux::getIntAttr(origOp->getContext(), outputType.getShape()[Dims4D::Act::C]);
            origOp->setAttr(mlir::StringAttr::get(origOp->getContext(), outChanAttrName), outChanBeforeAttr);
        }
        if (VPU::hasOnlyInPadding(getModuleOp(origOp)) && !origOp->hasAttr(inChanAttrName)) {
            const auto inChanBeforeAttr = vpux::getIntAttr(origOp->getContext(), inputType.getShape()[Dims4D::Act::C]);
            origOp->setAttr(mlir::StringAttr::get(origOp->getContext(), inChanAttrName), inChanBeforeAttr);
        }
        return matchFailed(log, rewriter, origOp, "Both input and output channels are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::C] == 0) {
        if (VPU::hasOnlyInPadding(getModuleOp(origOp)) && !origOp->hasAttr(inChanAttrName)) {
            const auto inChanBeforeAttr = vpux::getIntAttr(origOp->getContext(), inputType.getShape()[Dims4D::Act::C]);
            origOp->setAttr(mlir::StringAttr::get(origOp->getContext(), inChanAttrName), inChanBeforeAttr);
        }
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput = paddingChannel(origOp, rewriter, origOp->getOperand(0), inPadsEnd, Dims4D::Act::C.ind());
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, outPadsEnd[Dims4D::Act::C]);

    if (outPadsEnd[Dims4D::Act::C] == 0) {
        if (VPU::hasOnlyOutPadding(getModuleOp(origOp)) && !origOp->hasAttr(outChanAttrName)) {
            const auto outChanBeforeAttr =
                    vpux::getIntAttr(origOp->getContext(), outputType.getShape()[Dims4D::Act::C]);
            origOp->setAttr(mlir::StringAttr::get(origOp->getContext(), outChanAttrName), outChanBeforeAttr);
        }
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        auto offsets = calcOutputSliceOffset(origOp, outPadsEnd);

        auto sliceOp =
                rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp->getResult(0).getType(), newOp->getResult(0),
                                                         getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));
        extendOpLoc(sliceOp, "slice_out");
    }

    return mlir::success();
}

}  // namespace vpux

//
// MaxPoolRewriter
//

mlir::LogicalResult IE::MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.getKernelSize(),
                                              origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(),
                                              origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                                              origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, extractMeaningfulOutput, _log.nest());
}

//
// ConvolutionRewriter
//

mlir::LogicalResult IE::ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto filterShape = getShape(origOp.getFilter());

        const auto newInputShape = getShape(expandedInput);
        const auto inChanPadEnd = newInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];

        const auto paddedFilter = padConvFilter(rewriter, origOp, inChanPadEnd, outChanPadEnd, _log);

        mlir::Value paddedBiases;
        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.getBias(), std::nullopt,
                                                                   ShapeRef(biasPadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), newOutputType, expandedInput, paddedFilter,
                                                  paddedBiases, origOp.getStrides(), origOp.getPadsBegin(),
                                                  origOp.getPadsEnd(), origOp.getDilations(), origOp.getPostOpAttr(),
                                                  origOp.getClampAttr(), origOp.getStaticScaleAttr(),
                                                  origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, Shape outPadsEnd) -> SmallVector<int64_t> {
        SmallVector<int64_t> offsets(outPadsEnd.size(), 0);

        return offsets;
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}

//
// MatMulRewriter
//

mlir::LogicalResult IE::MatMulRewriter::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MatMul layer at '{1}'", getDebugName(), origOp->getLoc());
    const auto inputType = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();

    auto alignedChannelOpInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    auto inputAlignment = alignedChannelOpInterface.getInputChannelAlignment();
    auto outputAlignment = alignedChannelOpInterface.getOutputChannelAlignment();
    const auto inputPads = getInputPads(inputType, origOp.getTransposeA(), inputAlignment);
    const auto [filterPadsIC, filterPadsOC] =
            getFilterPads(filterType, origOp.getTransposeB(), inputAlignment, outputAlignment);
    const auto padsBegin = SmallVector<int64_t>(inputType.getShape().size(), 0);
    mlir::Value expandedInput = origOp.getInput1();
    if (std::any_of(inputPads.begin(), inputPads.end(), needsPadding)) {
        expandedInput = rewriter.createOrFold<IE::ExpandOp>(origOp.getLoc(), origOp.getInput1(),
                                                            getIntArrayAttr(rewriter, ArrayRef(padsBegin)),
                                                            getIntArrayAttr(rewriter, inputPads.raw()));
    }

    mlir::Value expandedFilter = origOp.getInput2();
    if (std::any_of(filterPadsIC.begin(), filterPadsIC.end(), needsPadding)) {
        expandedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp.getLoc(), origOp.getInput2(),
                                                             getIntArrayAttr(rewriter, ArrayRef(padsBegin)),
                                                             getIntArrayAttr(rewriter, filterPadsIC.raw()));
    }
    if (std::any_of(filterPadsOC.begin(), filterPadsOC.end(), needsPadding)) {
        expandedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp.getLoc(), expandedFilter,
                                                             getIntArrayAttr(rewriter, ArrayRef(padsBegin)),
                                                             getIntArrayAttr(rewriter, filterPadsOC.raw()));
    }

    auto newOp = rewriter.create<IE::MatMulOp>(origOp.getLoc(), expandedInput, expandedFilter, origOp.getTransposeA(),
                                               origOp.getTransposeB());

    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outShape = outputType.getShape();
    const auto offsets = SmallVector<int64_t>(outShape.size(), 0);

    auto ctx = rewriter.getContext();
    rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, newOp->getResult(0), getIntArrayAttr(ctx, offsets),
                                             getIntArrayAttr(ctx, outShape));
    return mlir::success();
}

//
// GroupConvolutionRewriter
//

mlir::LogicalResult IE::GroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolutionOp layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        const auto filterShape = getShape(origOp.getFilter());

        mlir::Value paddedFilter;

        if (outChanPadEnd == 0) {
            paddedFilter = origOp.getFilter();
        } else {
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;

            paddedFilter = paddingChannel(origOp, rewriter, origOp.getFilter(), std::move(filterPadsEnd),
                                          Dims4D::Filter::OC.ind());
        }

        mlir::Value paddedBiases;

        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = paddingChannel(origOp, rewriter, origOp.getBias(), biasPadsEnd, Dims4D::Act::C.ind());
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);
        const auto newConvOutShape = newOutputType.getShape().toValues();

        return rewriter.create<IE::GroupConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, paddedBiases, origOp.getStrides(),
                origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                getIntAttr(getContext(), newConvOutShape[Dims4D::Act::C]), origOp.getPostOpAttr(),
                origOp.getClampAttr(), origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, extractMeaningfulOutput, _log.nest());
}

//
// InterpolateRewriter
//

mlir::LogicalResult IE::InterpolateRewriter::matchAndRewrite(IE::InterpolateOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Interpolate layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto sizesInput = origOp.getSizes();
        auto sizesAttr = origOp.getSizesAttrAttr();
        const auto calcModeAttr = origOp.getAttr().getShapeCalcMode();
        if (calcModeAttr != nullptr && calcModeAttr.getValue() == IE::InterpolateCalcMode::SIZES) {
            const auto inType = origOp.getInput().getType().cast<NDTypeInterface>();
            const auto axesVal =
                    IE::getInterpAxesVal(origOp.getLoc(), origOp.getAxes(), origOp.getAxesAttrAttr(), inType);

            SmallVector<int64_t> newSizesVal(axesVal.size());
            const auto outputShape = newOutputType.getShape();
            for (const auto idx : irange(axesVal.size())) {
                newSizesVal[idx] = outputShape[Dim(axesVal[idx])];
            }
            sizesAttr = getIntArrayAttr(origOp.getContext(), newSizesVal);
            sizesInput = nullptr;
        }

        return rewriter.create<IE::InterpolateOp>(
                origOp.getLoc(), newOutputType, expandedInput, sizesInput, origOp.getScales(), origOp.getAxes(),
                sizesAttr, origOp.getScalesAttrAttr(), origOp.getAxesAttrAttr(), origOp.getTileOffsetAttrAttr(),
                origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttrAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, Shape outPadsEnd) -> SmallVector<int64_t> {
        SmallVector<int64_t> offsets(outPadsEnd.size(), 0);

        return offsets;
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}

//
// TransposedConvolutionRewriter
//

mlir::LogicalResult IE::TransposedConvolutionRewriter::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Transposed Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        const auto newInputShape = getShape(expandedInput);
        const auto filterShape = getShape(origOp.getFilter());
        const auto inChanPadEnd = newInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];
        auto paddedFilter = padConvFilter(rewriter, origOp, inChanPadEnd, outChanPadEnd, _log);

        mlir::Value paddedBiases;

        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = paddingChannel(origOp, rewriter, origOp.getBias(), biasPadsEnd, Dims4D::Act::C.ind());
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto outputType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = outputType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::TransposedConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, origOp.getOutputShape(), paddedBiases,
                origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                origOp.getOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, Shape outPadsEnd) -> SmallVector<int64_t> {
        return SmallVector<int64_t>(outPadsEnd.size(), 0);
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}

//
// PadRewriter
//

mlir::LogicalResult IE::PadRewriter::matchAndRewrite(IE::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Pad layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::PadOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.getPadsBegin(),
                                          origOp.getPadsEnd(), origOp.getPadValue(), origOp.getPadsBeginAttrAttr(),
                                          origOp.getPadsEndAttrAttr(), origOp.getPadValueAttrAttr(),
                                          origOp.getModeAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, Shape outPadsEnd) -> SmallVector<int64_t> {
        return SmallVector<int64_t>(outPadsEnd.size(), 0);
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}
