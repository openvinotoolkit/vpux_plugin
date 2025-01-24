//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <cstdint>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

mlir::Value createWeightsConstantImpl(vpux::NDTypeInterface inputType, float weightsValue, ArrayRef<int64_t> kernelSize,
                                      mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc) {
    const auto channels = inputType.getShape()[Dims4D::Act::C];
    auto weightShape =
            Shape({channels, channels, kernelSize[Dims4D::Kernel::Y.ind()], kernelSize[Dims4D::Kernel::X.ind()]});

    mlir::Type elemType = mlir::Float16Type::get(ctx);
    const auto inputElemType = inputType.getElementType();
    if (const auto qInputElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputElemType)) {
        // The weightsValue might not be representable on quantized type, thus the weights tensor is populated with 1's
        // and later scaled (under high-precision) to obtain the desired weightsValue.
        const auto quantScale = static_cast<double>(weightsValue);
        weightsValue = 1.0f;

        if (vpux::isFloat8Quantized(qInputElemType)) {
            elemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/qInputElemType.getStorageType(),
                    /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/quantScale, /*zeroPoint=*/0, /*storageTypeMin=*/qInputElemType.getStorageTypeMin(),
                    /*storageTypeMax=*/qInputElemType.getStorageTypeMax());

        } else if (qInputElemType.getStorageType().isInteger(8)) {
            elemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/quantScale, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);

        } else {
            VPUX_THROW("Unsupported quantized storage type: {0}", qInputElemType.getStorageType());
        }
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
    const auto dataAttr = Const::createConstContent(dataStorageType, ArrayRef(content));

    Const::ContentSetup contentAttrSetup(dataStorageType);

    if (const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(qElemType);
    } else if (mlir::isa<mlir::Float16Type>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(mlir::Float16Type::get(ctx));
    }
    if (order != DimsOrder::fromNumDims(weightShape.size())) {
        contentAttrSetup = contentAttrSetup.reorder(order);
    }

    auto weightsConstOp = rewriter.create<Const::DeclareOp>(
            loc, weightsType, Const::ContentAttr::get(dataAttr, std::move(contentAttrSetup)));
    return weightsConstOp.getOutput();
}

mlir::Value convertOpToConv(mlir::Operation* origOp, mlir::Value weights, mlir::Value sparseInput, VPU::ArchKind arch,
                            mlir::PatternRewriter& rewriter) {
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp, arch);

    auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
    auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
    const auto weightsTableVec =
            VPU::createWeightsTableData(origOp->getOperand(0), origOp->getResult(0), weights, /*bias=*/{}, OC,
                                        ppeConverter, biasConverter, /*constScale=*/nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto stridesAttr = getIntArrayAttr(origOp->getContext(), SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(origOp->getContext(), PadInfo(0, 0, 0, 0));
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(weights));

    auto outputChannelsAttr = origOp->hasAttr(VPU::outChanAttrName)
                                      ? origOp->getAttr(VPU::outChanAttrName).cast<mlir::IntegerAttr>()
                                      : nullptr;

    return rewriter
            .create<VPU::NCEConvolutionOp>(origOp->getLoc(), outputType, sparseInput, weights, weightsTable,
                                           stridesAttr, padAttr, ppeAttr, mpeEngineAttr, rawFilterShape,
                                           /*multi_cluster_strategyAttr=*/nullptr, outputChannelsAttr)
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
    const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(uint8_t(1)));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(smContentType).reorder(inputDimsOrder).castElemType(smElemType));
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp.getLoc(), smType, std::move(contentAttr));

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

    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    const auto OC = outputType.getShape()[Dims4D::Act::C];
    auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch);
    auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch);
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), weights, {}, OC,
                                                             ppeConverter, biasConverter, nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto strides = VPU::getNCEInterpolateStrides(scales, modeAttr, origOp.getAttr().getCoordMode());
    auto stridesAttr = getIntArrayAttr(rewriter, strides);
    auto nceOp = rewriter.create<VPU::NCEInterpolateOp>(origOp->getLoc(), outputType, sparseInput, weights,
                                                        weightsTable, stridesAttr, ppeAttr, rawFilterShape,
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
    // Calculate the pad based on whether the outputShape is specified. For example:
    // Input: 1x16x128x128xf16    Weights: 32x16x2x2xf16    OutputShape: 2xsi32 = dense<128>
    //                   \                |               /
    //                   TransposedConv: 1x32x128x128xf16
    //                   (strides = [2, 2], output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64, 64])
    // The padL/padR/padT/padB should be non-negative integer, here will be set to 0 instead of -63.
    // Then the Upsampling output shape will be 1x32x255x255xf16 with strides = [2, 2].
    auto padLeft = filterShape[Dims4D::Filter::KX] - origPads.left - 1;
    auto padTop = filterShape[Dims4D::Filter::KY] - origPads.top - 1;
    auto padRight = filterShape[Dims4D::Filter::KX] - origPads.right - 1 + outputPaddingW;
    auto padBottom = filterShape[Dims4D::Filter::KY] - origPads.bottom - 1 + outputPaddingH;
    if (origOp.getOutputShape() != nullptr) {
        padLeft = std::max<int64_t>(padLeft, 0);
        padTop = std::max<int64_t>(padTop, 0);
        padRight = std::max<int64_t>(padRight, 0);
        padBottom = std::max<int64_t>(padBottom, 0);
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
    const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(smContentType).reorder(inputDimsOrder).castElemType(smElemType));
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr));

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

    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto outputShape = outputType.getShape();
    const auto sparseInType = sparseInput.getType().dyn_cast<VPU::SparseTensorType>();

    // In case the outputShape is specified, update the convOp outputType. For example:
    // Input: 1x16x128x128xf16      Weights: 32x16x2x2xf16      OutputShape: 2xsi32 = dense<128>
    //                      \               |                  /
    //                       TransposedConv: 1x32x128x128xf16
    //                       (strides = [2, 2], output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64, 64])
    // The SEUpsampling output shape will be 1x32x255x255xf16.
    // So the NCEConv output shape should be updated to: 1x32x254x254xf16 instead of original output shape
    // 1x32x128x128xf16, and then sliceOp will be added for crop to 1x32x128x128xf16.
    if (origOp.getOutputShape() != nullptr) {
        const auto sparseInShape = sparseInType.cast<vpux::NDTypeInterface>().getShape();
        const auto OH = sparseInShape[Dims4D::Act::H] - weightsShape[Dims4D::Filter::KY] + 1;
        const auto OW = sparseInShape[Dims4D::Act::W] - weightsShape[Dims4D::Filter::KX] + 1;
        auto convOutputShape = SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], OH, OW};
        outputType = outputType.changeShape(Shape(convOutputShape));
    }

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(biasConstOp == nullptr, "VPU::TransposedConvolutionOp bias input is not constant");
        bias = biasConstOp.getContentAttr();
    }

    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp, _arch);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch);
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), weights, bias, OC,
                                                             ppeConverter, biasConverter, nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(
            origOp->getLoc(), outputType, sparseInput, weights, weightsTable, stridesAttr, padAttr, ppeAttr,
            mpeEngineAttr, rawFilterShape,
            /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputChannelsAttr());

    // In case the outputShape is specified, create sliceOp for crop
    const auto nceOutputShape = nceOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape();
    if (origOp.getOutputShape() != nullptr && nceOutputShape != outputShape) {
        const auto seUpsamplingAttr = sparseInType.getSeAttr().dyn_cast_or_null<VPU::SEUpsamplingAttr>();
        const auto seUpsamplingAttrPadding = parseIntArrayAttr<int64_t>(seUpsamplingAttr.getPadding());
        const auto origPadLeft = weightsShape[Dims4D::Filter::KX] - 1;
        const auto origPadTop = weightsShape[Dims4D::Filter::KY] - 1;
        const auto reducedPadLeft = origPadLeft - seUpsamplingAttrPadding[VPU::SE_PAD_LEFT];
        const auto reducedPadTop = origPadTop - seUpsamplingAttrPadding[VPU::SE_PAD_TOP];
        const auto padsBeginVector = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
        auto offsets = SmallVector<int64_t>(outputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(outputShape.begin(), outputShape.end());
        offsets[Dims4D::Act::H.ind()] = padsBeginVector[Dims4D::PadsBegin::Top] - reducedPadTop;
        offsets[Dims4D::Act::W.ind()] = padsBeginVector[Dims4D::PadsBegin::Left] - reducedPadLeft;

        auto sliceOp = rewriter.create<VPU::SliceOp>(origOp->getLoc(), nceOp.getOutput(),
                                                     getIntArrayAttr(getContext(), offsets),
                                                     getIntArrayAttr(getContext(), sizes));

        rewriter.replaceOp(origOp, sliceOp);
        return mlir::success();
    }

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// DilatedConvolutionToNCE
//

class DilatedConvolutionToNCE final : public mlir::OpRewritePattern<VPU::GroupConvolutionOp> {
public:
    DilatedConvolutionToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("DilatedConvolutionToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<uint8_t> createSparsityMapContent(ArrayRef<int64_t> shape) const;
    mlir::Value createSparseInput(Logger log, VPU::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter,
                                  const int64_t row, const int64_t column) const;

    VPU::ArchKind _arch;
    Logger _log;
};

SmallVector<uint8_t> DilatedConvolutionToNCE::createSparsityMapContent(ArrayRef<int64_t> shape) const {
    const auto elemCount = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    SmallVector<uint8_t> content(elemCount, 1);
    return content;
}

// Creates a sparse input containing a sparsity map and a storage element table.
mlir::Value DilatedConvolutionToNCE::createSparseInput(Logger log, VPU::GroupConvolutionOp origOp,
                                                       mlir::PatternRewriter& rewriter, int64_t x, int64_t y) const {
    log.trace("DilatedConvolutionToNCE::createSparseInput");

    auto ctx = origOp.getContext();

    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());

    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto filterShape = filterType.getShape();

    auto [dilateY, dilateX] = VPU::DilationUtils::extractDilationFactors(origOp.getDilations());
    auto [strideY, strideX] = VPU::DilationUtils::extractDilationStrides(origOp.getStrides());

    strideX = strideX % dilateX == 0 ? strideX / dilateX : strideX;
    strideY = strideY % dilateY == 0 ? strideY / dilateY : strideY;

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{strideX, strideY});

    auto kernelSizeAttr = getIntArrayAttr(
            ctx, SmallVector<int64_t>{filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]});

    auto dataOffset = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, y, x});

    const auto rowCount = inputShape[Dims4D::Act::H];
    const auto colCount = inputShape[Dims4D::Act::W];
    const auto dataColCount = (colCount - x + dilateX - 1) / dilateX;
    const auto dataRowCount = (rowCount - y + dilateY - 1) / dilateY;
    const auto resultSizes =
            SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], dataRowCount, dataColCount};

    auto dataSizes = SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                          inputShape[Dims4D::Act::H] - y, inputShape[Dims4D::Act::W] - x};

    auto seDilatedConvAttr = VPU::SEDilatedConvAttr::get(ctx, origOp.getDilations(), strides, kernelSizeAttr,
                                                         dataOffset, getIntArrayAttr(ctx, dataSizes),
                                                         /* offsets = */ nullptr, /* sizes = */ nullptr);
    auto seAttr = mlir::cast<VPU::SEAttr>(seDilatedConvAttr);

    // Create the StorageElementTable operation
    const auto arch = VPU::getArch(origOp);
    const auto sparsityConstraint = VPU::getSparsityConstraint(arch);

    // Depthwise limitation WL size is limited by 64
    constexpr int64_t depthwiseWorkloadLimit = 64;

    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint, depthwiseWorkloadLimit);

    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;

    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(origOp->getLoc(), inputShape.raw(),
                                                                 inputType.getElementType(), seSize, seDepth, seAttr);

    // Create the sparsity map constant
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);

    auto smContentType = mlir::RankedTensorType::get(resultSizes, smContentElemType);
    auto smContent = createSparsityMapContent(resultSizes);

    auto baseAttr = mlir::DenseElementsAttr::get(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);

    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(resultSizes, smElemType, tensorAttr);

    auto contentAttr =
            Const::ContentAttr::get(baseAttr).transform().reorder(inputDimsOrder).castElemType(smElemType).get();
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr));
    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConstOp.getResult(),
                                                             seTableOp.getOutput(), seAttr);

    return groupOp.getOutput();
}

mlir::LogicalResult DilatedConvolutionToNCE::matchAndRewrite(VPU::GroupConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp->getContext();

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto innerLog = _log.nest();

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto outputShape = outputType.getShape();
    const auto outputChannels = outputShape[Dims4D::Act::C];

    const auto filter = origOp.getFilter();
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(filter.getType());
    const auto filterShape = filterType.getShape();

    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto [dilateY, dilateX] = VPU::DilationUtils::extractDilationFactors(origOp.getDilations());
    auto [strideY, strideX] = VPU::DilationUtils::extractDilationStrides(origOp.getStrides());

    const auto subConvCountX = dilateX;
    const auto subConvCountY = dilateY;

    strideX = strideX % dilateX == 0 ? strideX / dilateX : strideX;
    strideY = strideY % dilateY == 0 ? strideY / dilateY : strideY;

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{strideY, strideX});

    auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));

    padStart[Dims4D::PadsBegin::Top] = std::max<int64_t>(padStart[Dims4D::PadsBegin::Top] - dilateY + 1, 0l);
    padStart[Dims4D::PadsBegin::Left] = std::max<int64_t>(padStart[Dims4D::PadsBegin::Left] - dilateX + 1, 0l);
    padEnd[Dims4D::PadsEnd::Bottom] = std::max<int64_t>(padEnd[Dims4D::PadsEnd::Bottom] - dilateY + 1, 0l);
    padEnd[Dims4D::PadsEnd::Right] = std::max<int64_t>(padEnd[Dims4D::PadsEnd::Right] - dilateX + 1, 0l);

    auto padding = PadInfo(padStart[Dims4D::PadsBegin::Left], padEnd[Dims4D::PadsEnd::Right],
                           padStart[Dims4D::PadsBegin::Top], padEnd[Dims4D::PadsEnd::Bottom]);

    auto padAttr = VPU::getPaddingAttr(getContext(), padding);

    // Create weights table
    Const::ContentAttr bias;

    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(biasConstOp == nullptr, "VPU::GroupConvolutionOp bias input is not constant");
        bias = biasConstOp.getContentAttr();
    }

    const auto ppeOpaqueAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch);
    auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch);

    auto alignedWeights = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    const auto weightsTableVec =
            VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedWeights, bias, outputChannels,
                                        ppeConverter, biasConverter, /* constScale = */ nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    // Generate sub-convolutions
    SmallVector<mlir::Value> subConvolutions;
    SmallVector<Shape> offsets;

    int64_t outputOffsetX = 0;
    int64_t outputOffsetY = 0;

    auto subConvLog = innerLog.nest();
    auto subConvOutputHeight{0};
    for (auto y : irange(subConvCountY)) {
        for (auto x : irange(subConvCountX)) {
            // Create sub-convolution.
            auto sparseInput = createSparseInput(subConvLog, origOp, rewriter, x, y);

            auto nceDepthConvolutionOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
                    vpux::appendLoc(origOp->getLoc(), "_subconv_y_{0}_x_{1}", y, x), sparseInput, alignedWeights,
                    weightsTable, strides, padAttr, ppeOpaqueAttr, rawFilterShape,
                    /* multiClusterStrategyAttr = */ nullptr, origOp.getOutputChannelsAttr());
            auto originalLayout = origOp.getResult().getType().cast<vpux::NDTypeInterface>().getDimsOrder();

            auto convType = nceDepthConvolutionOp.getResult().getType().cast<vpux::NDTypeInterface>();
            nceDepthConvolutionOp.getResult().setType(convType.changeDimsOrder(originalLayout));
            auto outputShape = convType.getShape();
            auto subConvOutputWidth = outputShape[Dims4D::Act::W];
            subConvOutputHeight = outputShape[Dims4D::Act::H];
            offsets.emplace_back(Shape{0, 0, outputOffsetY, outputOffsetX});

            outputOffsetX += subConvOutputWidth;
            subConvolutions.emplace_back(nceDepthConvolutionOp.getResult());
        }

        outputOffsetX = 0;
        outputOffsetY += subConvOutputHeight;
    }

    // This concat produces inaccurate result, strided concat is being developed to interleave subconvolutions: E#87431
    auto concatOp = rewriter.create<VPU::ConcatOp>(vpux::appendLoc(origOp->getLoc(), "_concat"),
                                                   origOp.getOutput().getType(), mlir::ValueRange(subConvolutions),
                                                   getIntArrayOfArray(rewriter.getContext(), offsets));
    rewriter.replaceOp(origOp, concatOp.getResult());
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
    const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(smContentType).reorder(inputDimsOrder).castElemType(smElemType));
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr));

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

    const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(uint8_t(1)));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(smContentType).reorder(inputDimsOrder).castElemType(smElemType));
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr));

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
                 VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true,
                                                    /*checkBatch=*/true));
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
    target.addDynamicallyLegalOp<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp op) {
        return !(VPU::isSupportedSEPDilatedConv(op, logCb, /*checkLayout=*/true,
                                                /*checkChannelAlignment=*/true) &&
                 VPU::isDilatedGroupConv(op));
    });
    target.addLegalOp<VPU::NCEInterpolateOp>();
    target.addLegalOp<VPU::NCEConvolutionOp>();
    target.addLegalOp<VPU::NCEDepthConvolutionOp>();
    target.addLegalOp<VPU::StorageElementTableOp>();
    target.addLegalOp<VPU::GroupSparseTensorOp>();
    target.addLegalOp<VPU::ConcatOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<VPU::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);

    if (_seOpsEnabled) {
        patterns.add<InterpolateToNCE>(&ctx, arch, _log);
        patterns.add<TransposedConvolutionToNCE>(&ctx, arch, _log);
    }
    if (_seExperimentalOpsEnabled) {
        patterns.add<PadToNCE>(&ctx, arch, _log);
        patterns.add<RollToNCE>(&ctx, arch, _log);
    }
    patterns.add<DilatedConvolutionToNCE>(&ctx, arch, _log);

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
