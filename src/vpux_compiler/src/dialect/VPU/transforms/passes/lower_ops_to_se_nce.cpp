//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/utils/loop.hpp"

using namespace vpux;

namespace {

mlir::Value createWeightsConstantImpl(vpux::NDTypeInterface inputType, float weightsValue, ArrayRef<int64_t> kernelSize,
                                      mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc) {
    const auto channels = inputType.getShape()[Dims4D::Act::C];
    auto weightShape =
            Shape({channels, channels, kernelSize[Dims4D::Kernel::Y.ind()], kernelSize[Dims4D::Kernel::X.ind()]});

    mlir::Type elemType = mlir::Float16Type::get(ctx);
    const auto inputElemType = inputType.getElementType();
    if (auto qInputElemType = inputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        double quantScale = static_cast<double>(weightsValue);
        weightsValue = 1.0f;
        elemType = mlir::quant::UniformQuantizedType::get(
                /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                /*scale=*/quantScale, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    }
    const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    const auto weightsType =
            mlir::RankedTensorType::get(weightShape.raw(), elemType, tensorAttr).cast<vpux::NDTypeInterface>();
    const auto order = weightsType.getDimsOrder();

    const auto weightsNumElems = weightsType.getNumElements();
    SmallVector<float> content(weightsNumElems, 0.0f);

    const auto kernelSizeCount = weightShape[Dims4D::Filter::KY] * weightShape[Dims4D::Filter::KX];
    const auto eachWeightSizeCount = weightShape[Dims4D::Filter::IC] * kernelSizeCount;
    loop_2d(LoopExecPolicy::Parallel, ctx, channels, kernelSizeCount, [&](int64_t channelIdx, int64_t kernelSizeIdx) {
        const auto beginOffset = channelIdx * kernelSizeCount;
        const auto contentIdx = channelIdx * eachWeightSizeCount + beginOffset + kernelSizeIdx;
        content[contentIdx] = weightsValue;
    });

    const auto dataStorageType = mlir::RankedTensorType::get(weightShape.raw(), mlir::Float32Type::get(ctx));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(content));

    auto contentAttr = Const::ContentAttr::get(dataAttr);
    if (auto qElemType = elemType.dyn_cast<mlir::quant::QuantizedType>()) {
        contentAttr = contentAttr.convertElemType(getUInt8Type(ctx));
        contentAttr = contentAttr.quantCast(qElemType);
    } else if (elemType.isa<mlir::Float16Type>()) {
        contentAttr = contentAttr.convertElemType(mlir::Float16Type::get(ctx));
    }
    if (order != DimsOrder::fromNumDims(weightShape.size())) {
        contentAttr = contentAttr.reorder(order);
    }

    auto weightsConstOp = rewriter.create<Const::DeclareOp>(loc, weightsType, contentAttr);
    return weightsConstOp.getOutput();
}

mlir::Value convertOpToConv(mlir::Operation* origOp, mlir::Value weights, mlir::Value sparseInput, VPU::ArchKind arch,
                            mlir::PatternRewriter& rewriter) {
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto ppeTaskAttr = nullptr;

    const auto weightsTableVec =
            VPU::createWeightsTableData(origOp->getOperand(0), origOp->getResult(0), weights, /*bias=*/nullptr, OC,
                                        ppeTaskAttr, arch, /*postOpAttr=*/nullptr, /*constScale=*/nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto stridesAttr = getIntArrayAttr(origOp->getContext(), SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(origOp->getContext(), PadInfo(0, 0, 0, 0));
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(weights));

    return rewriter
            .create<VPU::NCEConvolutionOp>(
                    origOp->getLoc(), outputType, sparseInput, weights, weightsTable, /*activationWindow=*/nullptr,
                    /*instructionListTable=*/nullptr, stridesAttr, padAttr, ppeTaskAttr, rawFilterShape,
                    /*activationWindowChannelLength=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr)
            .getResult();
}

//
// InterpolateToNCE
//

class InterpolateToNCE final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    InterpolateToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::InterpolateOp>(ctx), _arch(arch), _log(log) {
        setDebugName("InterpolateToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createSparseInput(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                  VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const;
    mlir::Value createWeightsConstant(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                      ArrayRef<int64_t> kernelSize) const;

    VPU::ArchKind _arch;
    Logger _log;
};

// Creates a sparse input whose sparsity map and storage element table have the following `H x W` shapes:
//   [factorH * inputH + padTop + padBottom] x [factorW * inputW + padLeft + padRight]
// The sparsity map constant has all bits set to 1.
// The storage element table operation and the resulting sparse tensor have a SEInterpolateAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value InterpolateToNCE::createSparseInput(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                                VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const {
    auto ctx = origOp.getContext();
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    auto inputDimsOrder = inputType.getDimsOrder();

    // Create the SEInterpolateAttr
    auto coordModeAttr = origOp.getAttr().getCoordMode();
    VPUX_THROW_WHEN(coordModeAttr == nullptr, "Missing coordinate transformation mode");
    IE::InterpolateNearestModeAttr nearestModeAttr = nullptr;
    if (modeAttr != nullptr && modeAttr.getValue() == VPU::NCEInterpolateMode::NEAREST) {
        nearestModeAttr = origOp.getAttr().getNearestMode();
        VPUX_THROW_WHEN(nearestModeAttr == nullptr, "Missing nearest mode");
    }

    mlir::ArrayAttr initialInputShapeAttr = nullptr;
    mlir::ArrayAttr initialOutputShapeAttr = nullptr;
    if (coordModeAttr.getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) {
        initialInputShapeAttr = getIntArrayAttr(ctx, inputShape.raw());
        initialOutputShapeAttr = getIntArrayAttr(ctx, outputShape.raw());
    }
    auto scalesAttr = getFPArrayAttr(ctx, scales);
    auto seInterpolateAttr = VPU::SEInterpolateAttr::get(ctx, modeAttr, coordModeAttr, scalesAttr, nearestModeAttr,
                                                         /*offsets=*/nullptr, /*sizes=*/nullptr, initialInputShapeAttr,
                                                         initialOutputShapeAttr);
    auto seAttr = seInterpolateAttr.cast<VPU::SEAttr>();

    // Create the StorageElementTable operation
    auto arch = VPU::getArch(origOp);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(origOp->getLoc(), inputShape.raw(),
                                                                 inputType.getElementType(), seSize, seDepth, seAttr);

    // Create the sparsity map constant
    auto smShape = to_small_vector(seTableOp.getType().cast<vpux::NDTypeInterface>().getShape());
    smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);
    auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
    auto smElemCount =
            std::accumulate(smShape.begin(), smShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    SmallVector<uint8_t> smContent(smElemCount, 1);
    const auto baseAttr = mlir::DenseElementsAttr::get(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr).reorder(inputDimsOrder).convertElemType(smElemType);
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp.getLoc(), smType, contentAttr);

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConstOp.getOutput(),
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
// The kernels have the following configuration, where one single input channel will be populated for each kernel:
//   KernelSizeH x KernelSizeW with value 1 / (KernelSizeH * KernelSizeW)
mlir::Value InterpolateToNCE::createWeightsConstant(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                                    ArrayRef<int64_t> kernelSize) const {
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto weightsValue =
            1.0f / checked_cast<float>(kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()]);
    return createWeightsConstantImpl(inputType, weightsValue, kernelSize, rewriter, origOp.getContext(),
                                     origOp.getLoc());
}

mlir::LogicalResult InterpolateToNCE::matchAndRewrite(VPU::InterpolateOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto modeAttr = VPU::getNCEInterpolateModeAttr(origOp.getAttr().getMode());
    auto potentialScales = VPU::getNCEInterpolateScales(inputType, outputType, origOp.getAttr().getCoordMode());
    VPUX_THROW_UNLESS(potentialScales.has_value(), "Cannot get scales of NCE Interpolate");
    const auto scales = potentialScales.value();
    const auto kernelSize = VPU::getNCEInterpolateKernelSize(scales, modeAttr, origOp.getAttr().getCoordMode());

    const auto sparseInput = createSparseInput(origOp, rewriter, modeAttr, scales);
    const auto weights = createWeightsConstant(origOp, rewriter, kernelSize);
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto rawFilterShape = getIntArrayAttr(rewriter, weightsShape);

    const auto ppeTaskAttr = nullptr;

    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), weights, nullptr,
                                                             OC, ppeTaskAttr, _arch, nullptr, nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto strides = VPU::getNCEInterpolateStrides(scales, modeAttr, origOp.getAttr().getCoordMode());
    auto stridesAttr = getIntArrayAttr(rewriter, strides);
    auto nceOp = rewriter.create<VPU::NCEInterpolateOp>(origOp->getLoc(), outputType, sparseInput, weights,
                                                        weightsTable, stridesAttr, ppeTaskAttr, rawFilterShape,
                                                        /*multi_cluster_strategyAttr=*/nullptr, modeAttr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// TransposedConvolutionToNCE
//

class TransposedConvolutionToNCE final : public mlir::OpRewritePattern<VPU::TransposedConvolutionOp> {
public:
    TransposedConvolutionToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::TransposedConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("TransposedConvolutionToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<uint8_t> createSparsityMapContent(ArrayRef<int64_t> shape, ArrayRef<int64_t> padding,
                                                  const int64_t factorH, const int64_t factorW) const;
    mlir::Value createSparseInput(VPU::TransposedConvolutionOp origOp, mlir::PatternRewriter& rewriter) const;

    VPU::ArchKind _arch;
    Logger _log;
};

SmallVector<uint8_t> TransposedConvolutionToNCE::createSparsityMapContent(ArrayRef<int64_t> shape,
                                                                          ArrayRef<int64_t> padding,
                                                                          const int64_t factorH,
                                                                          const int64_t factorW) const {
    const auto elemCount = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    const auto channels = shape[Dims4D::Act::C.ind()];
    const auto height = shape[Dims4D::Act::H.ind()];
    const auto width = shape[Dims4D::Act::W.ind()];

    const auto padLeft = padding[VPU::SE_PAD_LEFT];
    const auto padTop = padding[VPU::SE_PAD_TOP];
    const auto padRight = padding[VPU::SE_PAD_RIGHT];
    const auto padBottom = padding[VPU::SE_PAD_BOTTOM];

    SmallVector<uint8_t> content(elemCount, 0);
    for (int64_t h = padTop; h < height - padBottom; h += (factorH + 1)) {
        for (int64_t w = padLeft; w < width - padRight; w += (factorW + 1)) {
            for (int64_t c = 0; c < channels; ++c) {
                const auto index = c * height * width + h * width + w;
                content[index] = 1;
            }
        }
    }
    return content;
}

// Creates a sparse input containing a sparsity map and a storage element table.
// The storage element table operation and the resulting sparse tensor have a SEUpsamplingAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value TransposedConvolutionToNCE::createSparseInput(VPU::TransposedConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto filterShape = filterType.getShape();

    // Create the SEUpsamplingAttr
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto factorH = strides[Dims4D::Strides::Y.ind()] - 1;
    const auto factorW = strides[Dims4D::Strides::X.ind()] - 1;
    const auto factorsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{factorH, factorW});

    auto outputPadding = parseIntArrayAttr<int64_t>(origOp.getOutputPadding());
    if (outputPadding.empty()) {
        outputPadding = SmallVector<int64_t>({0, 0});
    }
    const auto outputPaddingH = outputPadding[Dims4D::PadsOutput::Y.ind()];
    const auto outputPaddingW = outputPadding[Dims4D::PadsOutput::X.ind()];
    const auto origPads = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());
    const auto padLeft = filterShape[Dims4D::Filter::KX] - origPads.left - 1;
    const auto padTop = filterShape[Dims4D::Filter::KY] - origPads.top - 1;
    const auto padRight = filterShape[Dims4D::Filter::KX] - origPads.right - 1 + outputPaddingW;
    const auto padBottom = filterShape[Dims4D::Filter::KY] - origPads.bottom - 1 + outputPaddingH;
    if (origPads.left < 0 || origPads.top < 0 || origPads.right < 0 || origPads.bottom < 0) {
        _log.nest().trace("Negative padding is unsupported: {0}, {1}, {2}, {3}", padLeft, padTop, padRight, padBottom);
        return nullptr;
    }
    const SmallVector<int64_t> padding{padLeft, padTop, padRight, padBottom};
    const auto paddingAttr = getIntArrayAttr(ctx, padding);

    auto seUpsamplingAttr =
            VPU::SEUpsamplingAttr::get(ctx, factorsAttr, paddingAttr, /*offsets=*/nullptr, /*sizes=*/nullptr);
    auto seAttr = seUpsamplingAttr.cast<VPU::SEAttr>();

    // Create the StorageElementTable operation
    auto arch = VPU::getArch(origOp);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(origOp->getLoc(), inputShape.raw(),
                                                                 inputType.getElementType(), seSize, seDepth, seAttr);

    // Create the sparsity map constant
    auto smShape = to_small_vector(seTableOp.getType().cast<vpux::NDTypeInterface>().getShape());
    smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);
    auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
    const auto smContent = createSparsityMapContent(smShape, padding, factorH, factorW);
    const auto baseAttr = mlir::DenseElementsAttr::get(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr).reorder(inputDimsOrder).convertElemType(smElemType);
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, contentAttr);

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConstOp.getOutput(),
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

mlir::LogicalResult TransposedConvolutionToNCE::matchAndRewrite(VPU::TransposedConvolutionOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto sparseInput = createSparseInput(origOp, rewriter);
    if (sparseInput == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to create sparse input");
    }
    const auto weights = origOp.getFilter();
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto rawFilterShape = getIntArrayAttr(rewriter, weightsShape);

    const auto stridesAttr = getIntArrayAttr(origOp.getContext(), SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(0, 0, 0, 0));

    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto OC = outputType.getShape()[Dims4D::Act::C];

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(biasConstOp == nullptr, "VPU::TransposedConvolutionOp bias input is not constant");
        bias = biasConstOp.getContentAttr();
    }
    const auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);

    const auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), weights, bias, OC,
                                                             ppeTaskAttr, _arch, origOp.getPostOpAttr(), nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(
            origOp->getLoc(), outputType, sparseInput, weights, weightsTable, /*activationWindow=*/nullptr,
            /*instructionListTable=*/nullptr, stridesAttr, padAttr, ppeTaskAttr, rawFilterShape,
            /*activationWindowChannelLength=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// PadToNCE
//

class PadToNCE final : public mlir::OpRewritePattern<VPU::PadOp> {
public:
    PadToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::PadOp>(ctx), _arch(arch), _log(log) {
        setDebugName("PadToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<uint8_t> createSparsityMapContent(IE::PadMode padMode, ArrayRef<int64_t> shape,
                                                  ArrayRef<int64_t> padding) const;
    mlir::Value createSparseInput(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const;
    mlir::Value createWeightsConstant(VPU::PadOp origOp, mlir::PatternRewriter& rewriter,
                                      ArrayRef<int64_t> kernelSize) const;
    mlir::Value convertPadToConv(VPU::PadOp origOp, mlir::Value sparseInput, mlir::PatternRewriter& rewriter) const;

    VPU::ArchKind _arch;
    Logger _log;
};

SmallVector<uint8_t> PadToNCE::createSparsityMapContent(IE::PadMode padMode, ArrayRef<int64_t> shape,
                                                        ArrayRef<int64_t> padding) const {
    const auto elemCount = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    const auto channels = shape[Dims4D::Act::C.ind()];
    const auto height = shape[Dims4D::Act::H.ind()];
    const auto width = shape[Dims4D::Act::W.ind()];

    const auto padLeft = padding[VPU::SE_PAD_LEFT];
    const auto padTop = padding[VPU::SE_PAD_TOP];
    const auto padRight = padding[VPU::SE_PAD_RIGHT];
    const auto padBottom = padding[VPU::SE_PAD_BOTTOM];

    if (padMode != IE::PadMode::CONSTANT) {
        return SmallVector<uint8_t>(elemCount, 1);
    }

    SmallVector<uint8_t> content(elemCount, 0);
    for (int64_t h = padTop; h < height - padBottom; ++h) {
        for (int64_t w = padLeft; w < width - padRight; ++w) {
            for (int64_t c = 0; c < channels; ++c) {
                const auto index = c * height * width + h * width + w;
                content[index] = 1;
            }
        }
    }

    return content;
}

// Creates a sparse input containing a sparsity map and a storage element table.
// The storage element table operation and the resulting sparse tensor have a SEPaddingAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value PadToNCE::createSparseInput(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto padMode = origOp.getMode();

    // Create the SEPaddingAttr
    auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr().value());
    auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr().value());
    const SmallVector<int64_t> padding{padsBegin[Dims4D::Act::W.ind()], padsBegin[Dims4D::Act::H.ind()],
                                       padsEnd[Dims4D::Act::W.ind()], padsEnd[Dims4D::Act::H.ind()]};
    auto sePaddingAttr = VPU::SEPaddingAttr::get(ctx, origOp.getModeAttr(), getIntArrayAttr(ctx, padding),
                                                 /*offsets=*/nullptr, /*sizes=*/nullptr);
    auto seAttr = sePaddingAttr.cast<VPU::SEAttr>();

    // Create the StorageElementTable operation
    auto arch = VPU::getArch(origOp);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(origOp->getLoc(), inputShape.raw(),
                                                                 inputType.getElementType(), seSize, seDepth, seAttr);

    // Create the sparsity map constant
    auto smShape = to_small_vector(seTableOp.getType().cast<vpux::NDTypeInterface>().getShape());
    smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);
    auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
    const auto smContent = createSparsityMapContent(padMode, smShape, padding);
    const auto baseAttr = mlir::DenseElementsAttr::get(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr).reorder(inputDimsOrder).convertElemType(smElemType);
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, contentAttr);

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConstOp.getOutput(),
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
mlir::Value PadToNCE::createWeightsConstant(VPU::PadOp origOp, mlir::PatternRewriter& rewriter,
                                            ArrayRef<int64_t> kernelSize) const {
    auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    auto weightsValue = 1.0f;
    return createWeightsConstantImpl(inputType, weightsValue, kernelSize, rewriter, origOp.getContext(),
                                     origOp.getLoc());
}

mlir::Value PadToNCE::convertPadToConv(VPU::PadOp origOp, mlir::Value sparseInput,
                                       mlir::PatternRewriter& rewriter) const {
    const auto weights = createWeightsConstant(origOp, rewriter, /*kernelSize=*/SmallVector<int64_t>{1, 1});
    return convertOpToConv(origOp, weights, sparseInput, _arch, rewriter);
}

mlir::LogicalResult PadToNCE::matchAndRewrite(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto sparseInput = createSparseInput(origOp, rewriter);
    if (sparseInput == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to create sparse input");
    }

    auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(*origOp.getResult().getUsers().begin());
    auto isLegalFusedIntoConv =
            convOp && origOp.getResult().hasOneUse() && !mlir::isa<VPU::SparseTensorType>(convOp.getInput().getType());
    if (isLegalFusedIntoConv) {
        convOp.setOperand(0, sparseInput);
        rewriter.eraseOp(origOp);
        return mlir::success();
    }

    auto nceOp = convertPadToConv(origOp, sparseInput, rewriter);

    rewriter.replaceOp(origOp, nceOp);
    return mlir::success();
}

//
// RollToNCE
//

class RollToNCE final : public mlir::OpRewritePattern<VPU::RollOp> {
public:
    RollToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::RollOp>(ctx), _arch(arch), _log(log) {
        setDebugName("RollToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::RollOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createWeightsConstant(VPU::RollOp origOp, mlir::PatternRewriter& rewriter,
                                      ArrayRef<int64_t> kernelSize) const;
    mlir::Value createSparseInput(VPU::RollOp origOp, SmallVector<int64_t> axes, SmallVector<int64_t> shift,
                                  mlir::PatternRewriter& rewriter) const;

    VPU::ArchKind _arch;
    Logger _log;
};

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
mlir::Value RollToNCE::createWeightsConstant(VPU::RollOp origOp, mlir::PatternRewriter& rewriter,
                                             ArrayRef<int64_t> kernelSize) const {
    auto inputType = origOp.getData().getType().cast<vpux::NDTypeInterface>();
    auto weightsValue = 1.0f;
    return createWeightsConstantImpl(inputType, weightsValue, kernelSize, rewriter, origOp.getContext(),
                                     origOp.getLoc());
}

// Creates a sparse input containing a sparsity map and a storage element table.
// The storage element table operation and the resulting sparse tensor have a SERollAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value RollToNCE::createSparseInput(VPU::RollOp origOp, SmallVector<int64_t> axes, SmallVector<int64_t> shift,
                                         mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto inputType = origOp.getData().getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();

    // Create the SERollAttr
    auto seRollAttr = VPU::SERollAttr::get(ctx, getIntArrayAttr(ctx, shift), getIntArrayAttr(ctx, axes),
                                           /*offsets=*/nullptr, /*sizes=*/nullptr);
    auto seAttr = seRollAttr.cast<VPU::SEAttr>();

    // Create the StorageElementTable operation
    auto sparsityConstraint = VPU::getSparsityConstraint(_arch);
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(origOp->getLoc(), inputShape.raw(),
                                                                 inputType.getElementType(), seSize, seDepth, seAttr);

    // Create the sparsity map constant
    auto smShape = to_small_vector(seTableOp.getType().cast<vpux::NDTypeInterface>().getShape());
    smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);
    auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);

    const auto elemCount = std::accumulate(smShape.begin(), smShape.end(), int64_t(1), std::multiplies<int64_t>());
    SmallVector<uint8_t> smContent(elemCount, 1);

    const auto baseAttr = mlir::DenseElementsAttr::get(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr).reorder(inputDimsOrder).convertElemType(smElemType);
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, contentAttr);

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getData(), smConstOp.getOutput(),
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

mlir::LogicalResult RollToNCE::matchAndRewrite(VPU::RollOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inputType = origOp.getData().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    auto shiftAndAxesOrFail =
            IE::getShiftAndAxesForRollOp(origOp.getLoc(), origOp.getShift(), origOp.getAxes(), inputShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return mlir::failure();
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto shift = shiftAndAxes.shift;
    const auto axes = shiftAndAxes.axes;

    const auto sparseInput = createSparseInput(origOp, std::move(axes), std::move(shift), rewriter);
    if (sparseInput == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to create sparse input");
    }

    const auto weights = createWeightsConstant(origOp, rewriter, /*kernelSize=*/SmallVector<int64_t>{1, 1});
    auto nceOpOutput = convertOpToConv(origOp, weights, sparseInput, _arch, rewriter);
    rewriter.replaceOp(origOp, nceOpOutput);

    return mlir::success();
}

//
// LowerOpsToSENCEPass
//

class LowerOpsToSENCEPass final : public VPU::LowerOpsToSENCEBase<LowerOpsToSENCEPass> {
public:
    explicit LowerOpsToSENCEPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

mlir::LogicalResult LowerOpsToSENCEPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }
    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

void LowerOpsToSENCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::InterpolateOp>([&](VPU::InterpolateOp op) {
        return !(_seOpsEnabled &&
                 VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true));
    });
    target.addDynamicallyLegalOp<VPU::TransposedConvolutionOp>([&](VPU::TransposedConvolutionOp op) {
        return !(_seOpsEnabled && VPU::isSupportedSEPTransposedConv(op, logCb, /*checkLayout=*/false,
                                                                    /*checkChannelAlignment=*/false));
    });
    target.addDynamicallyLegalOp<VPU::PadOp>([&](VPU::PadOp op) {
        return !(_seExperimentalOpsEnabled && VPU::isSupportedSEPPadOp(op, logCb, /*checkLayout=*/true,
                                                                       /*checkChannelAlignment=*/true));
    });
    target.addDynamicallyLegalOp<VPU::RollOp>([&](VPU::RollOp op) {
        return !(_seExperimentalOpsEnabled && VPU::isSupportedSEPRoll(op, logCb, /*checkLayout=*/true,
                                                                      /*checkChannelAlignment=*/true));
    });
    target.addLegalOp<VPU::NCEInterpolateOp>();
    target.addLegalOp<VPU::NCEConvolutionOp>();
    target.addLegalOp<VPU::StorageElementTableOp>();
    target.addLegalOp<VPU::GroupSparseTensorOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);

    if (_seOpsEnabled) {
        patterns.add<InterpolateToNCE>(&ctx, arch, _log);
        patterns.add<TransposedConvolutionToNCE>(&ctx, arch, _log);
    }
    if (_seExperimentalOpsEnabled) {
        patterns.add<PadToNCE>(&ctx, arch, _log);
        patterns.add<RollToNCE>(&ctx, arch, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerOpsToSENCEPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createLowerOpsToSENCEPass(const bool seOpsEnabled,
                                                                 const bool seExperimentalOpsEnabled, Logger log) {
    return std::make_unique<LowerOpsToSENCEPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}
