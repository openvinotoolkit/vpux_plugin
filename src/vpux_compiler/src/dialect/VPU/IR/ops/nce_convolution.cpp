//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <openvino/op/convolution.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

bool vpux::VPU::NCEConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                             vpux::NDTypeInterface output, Byte reservedMem) {
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // These depend on a particular tile
    const auto OC = output.getShape()[Dims4D::Act::C];
    const auto IC = input.getShape()[Dims4D::Act::C];

    const auto inOrder = input.getDimsOrder();

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    if (inOrder == DimsOrder::NHWC) {
        // do nothing
    } else if (inOrder == DimsOrder::NCHW) {
        const auto kernelSize = Shape{KY, KX};

        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
        const auto strideW = kernelStrides[Dims4D::Strides::X];

        auto activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::CM_CONV, kernelSize,
                                                                         strideW, input.getElementType(), IC);

        buffers.push_back(activationWindowSize * 1_Byte);
    } else {
        VPUX_THROW("[{0}] Unsupported input layout '{1}'", getLoc(), inOrder);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCEConvolutionOp::isSupported(IE::ConvolutionOp op, LogCb logCb, bool checkLayout,
                                              bool checkChannelAlignment) {
    return VPU::isSupportedConv(op, logCb, checkLayout, checkChannelAlignment);
}

//
// verify
//

static mlir::LogicalResult verifyConv(mlir::Location loc, VPU::ArchKind arch, VPU::NCEConvolutionOpAdaptor op,
                                      mlir::Value output) {
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.getRawFilterShape()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto padAttr = op.getPad();
    const auto weightsTableShape = getShape(op.getWeightsTable());

    return VPU::verifyConvUtil(loc, arch, filterShape, kernelStrides, padAttr, weightsTableShape, output);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verify() {
    auto op = getOperation();
    const auto arch = getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    const NCEConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(), op->getPropertiesStorage(),
                                              op->getRegions());
    if (mlir::failed(verifyConv(getOperation()->getLoc(), arch, convAdaptor, getOutput()))) {
        return mlir::failure();
    }

    const auto inputOrder = DimsOrder::fromValue(getInput());

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto inputType = getInput().getType().cast<NDTypeInterface>();
    const auto outputType = getOutput().getType().cast<NDTypeInterface>();
    const auto filterType = getFilter().getType().cast<NDTypeInterface>();

    const auto alignedFilterShape = filterType.getShape();
    const auto expectedAlignedFilterShape = inferAlignedFilterShape(inputType, outputType, filterType);

    if (alignedFilterShape != expectedAlignedFilterShape) {
        return errorAt(op, "Got wrong shape for NCE Convolution 'filter' '{0}', expected '{1}'", alignedFilterShape,
                       expectedAlignedFilterShape);
    }

    if (inputOrder == DimsOrder::NHWC) {
        if (getActivationWindow() != nullptr) {
            return errorAt(op, "'activationWindow' should not be used with Z-Major NCE Convolution");
        }
        if (getActivationWindowChannelLength().has_value()) {
            return errorAt(op, "'activation_window_channel_length' should not be used with Z-Major NCE Convolution");
        }
    } else {
        if (getActivationWindow() == nullptr) {
            return errorAt(op, "Missing 'activationWindow' operand for C-Major NCE Convolution");
        }
        if (!getActivationWindowChannelLength().has_value()) {
            return errorAt(op, "Missing 'activation_window_channel_length' operand for C-Major NCE Convolution");
        }

        const auto IC = inputType.getShape()[Dims4D::Act::C];

        const auto kernelSize = Shape{KY, KX};

        const auto activationWindowShape = getShape(getActivationWindow());
        const auto expectedActivationWindowShape = NCESparsity::inferActivationWindowShape(
                NCESparsity::Mode::CM_CONV, kernelSize, SX, inputType.getElementType(), IC);

        if (activationWindowShape != expectedActivationWindowShape) {
            return errorAt(op, "Got wrong shape for 'activationWindow' '{0}', expected '{1}'", activationWindowShape,
                           expectedActivationWindowShape);
        }

        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::CM_CONV, kernelSize, SX,
                                                                        inputType.getElementType(), IC);

        if (getActivationWindowChannelLength().value() != bitPatternSize) {
            return errorAt(op, "Got wrong value for 'activation_window_channel_length' '{0}', expected '{1}'",
                           getActivationWindowChannelLength(), bitPatternSize);
        }
    }

    return mlir::success();
}

Shape vpux::VPU::NCEConvolutionOp::inferAlignedFilterShape(NDTypeInterface input, NDTypeInterface output,
                                                           NDTypeInterface filter) {
    const auto rawFilterShape = Shape(parseIntArrayAttr<int64_t>(this->getRawFilterShape()));
    const auto KY = rawFilterShape[Dims4D::Filter::KY];
    const auto KX = rawFilterShape[Dims4D::Filter::KX];

    const auto IC = input.getShape()[Dims4D::Act::C];
    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto alignment = NCEInvariant::getAlignment(filter.getElementType());

    const auto remainder = (IC * KY * KX) % alignment;

    if (remainder == 0) {
        return Shape{OC, IC, KY, KX};
    }

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

    return Shape{OC, 1, 1, IC * KY * KX + padding};
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEConvolutionOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.getInput());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.getRawFilterShape()));

    if (inShape[Dims4D::Act::C] != filterShape[Dims4D::Filter::IC]) {
        return errorAt(loc, "Input tensor channels and filter shape must be the same");
    }

    const auto windowStrides = parseIntArrayAttr<int64_t>(op.getStrides());
    const auto windowDilations = ov::Strides({1, 1});

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = ov::CoordinateDiff({padTop, padLeft});
    const auto dataPaddingAbove = ov::CoordinateDiff({padBottom, padRight});

    const auto conv = ov::op::v1::Convolution(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape(inShape.begin(), inShape.end())),
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                    ov::Shape(filterShape.begin(), filterShape.end())),
            ov::Strides(windowStrides.begin(), windowStrides.end()), dataPaddingBelow, dataPaddingAbove,
            windowDilations);

    const auto& outputShapeNG = conv.get_output_partial_shape(0);

    const auto outputShape = to_small_vector(outputShapeNG.get_shape() | transformed([](size_t val) {
                                                 return checked_cast<int64_t>(val);
                                             }));

    auto inputType = op.getInput().getType();
    if (auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        inputType = sparseInputType.getData();
    }
    const auto outputType = inputType.cast<vpux::NDTypeInterface>().changeShape(Shape(outputShape));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto origPadding = toPadInfo(getPad());

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    auto inputTiling =
            backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    // Remove bias input tile if present
    if (inputTiling.tiles.size() > 2) {
        // Drop the bias tile
        inputTiling.tiles.pop_back();
    }

    // Adjust filter tile for the aligned filter
    inputTiling.tiles[1].shape = getShape(getFilter()).toValues();
    inputTiling.tiles[1].shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    inputTiling.tiles.push_back(VPU::getWeightsTableTile(this, outputTile));

    if (getInstructionListTable() != nullptr) {
        inputTiling.tiles.push_back(VPU::getInstructionListTableTile(this, outputTile));
    }

    if (getActivationWindow() != nullptr) {
        inputTiling.tiles.push_back(VPU::getActivationWindowTile(this, outputTile));
    }

    return inputTiling;
}

void vpux::VPU::NCEConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEConvolutionOp::getKernelSizeVal() {
    const auto kernelShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto KY = kernelShape[Dims4D::Filter::KY];
    const auto KX = kernelShape[Dims4D::Filter::KX];
    return {KY, KX};
}

SmallVector<int64_t> vpux::VPU::NCEConvolutionOp::getStridesVal() {
    return parseIntArrayAttr<int64_t>(getStrides());
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto arch = VPU::getArch(getOperation());
    const bool isCMajor =
            VPU::NCEInvariant::isChannelMajorCompatible(arch, getInput().getType().cast<vpux::NDTypeInterface>());
    if (isCMajor) {
        return strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
               strategy == VPU::MultiClusterStrategy::Clustering;
    }

    auto nceOp = mlir::cast<NCEConvolutionOp>(getOperation());
    const auto isCompatible = VPU::isSEPConvCompatibleWithClusterStrategy(nceOp, strategy);
    if (isCompatible.has_value()) {
        return isCompatible.value();
    }

    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outputDimsOrder = outputType.getDimsOrder();
    // Unsupported to broadcast the lowest dimension
    // Track E#120804
    if (outputDimsOrder.dimAt(outputDimsOrder.numDims() - 1) == Dims4D::Act::H) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEConvolutionOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getNCEExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                                    distributionMode, numTiles, numClusters, alignment,
                                                    uniformDistributedSegments, overlapParams);
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool VPU::NCEConvolutionOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    auto outputShape = ShapeRef(oriOutputTile.shape);
    auto offset = ShapeRef(oriOutputTile.offsets);
    auto axis = ShapeRef(oriOutputTile.axis);
    if (outputShape == ShapeRef()) {
        outputShape = getShape(getOutput());
    }
    vpux::TileInfo outputTile{outputShape, offset, axis, oriOutputTile.isCompletedTile};
    if (!VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile)) {
        return false;
    }

    auto nceOp = mlir::cast<NCEConvolutionOp>(getOperation());
    Shape inputShape = getShape(nceOp.getInput()).toValues();
    auto inputType = nceOp.getInput().getType().cast<NDTypeInterface>();
    // If has custom output shape, infer the input shape
    if (outputShape != getShape(nceOp.getOutput())) {
        VPUX_THROW_UNLESS(offset != ShapeRef() && axis != ShapeRef(),
                          "Offsets and axis must have value when create TileInfo. Loc: {0}", nceOp->getLoc());
        outputTile.isCompletedTile = true;
        auto computerShape = nceOp.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles.front().shape;
        auto inputOffset = computerShape.tiles.front().offsets;
        inputType = inputType.extractDenseTile(inputOffset, inputShape);
    }

    auto moduleOp = nceOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = IE::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    return isSOHSupportedByDPU(inputType, inputShape, numTiles, false, VPU::getArch(nceOp.getOperation()));
}

bool VPU::NCEConvolutionOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEConvolutionOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEConvolutionOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEConvolutionOp>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape()[Dims4D::Act::C], strategy);
    return fitIntoCMX(getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy),
                      getDistributedFilterTypeFromOp(nceOp, nceOp.getFilter().getType(), numClusters, strategy),
                      getDistributedOutputTypeFromOp(nceOp, nceOp.getOutput().getType(), numClusters, strategy),
                      reservedMem);
}

bool VPU::NCEConvolutionOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCEConvolutionOp>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, nceOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    auto distributedInputType =
            getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOp, nceOp.getFilter().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyChannels() {
    auto arch = VPU::getArch(*this);
    return mlir::success(
            vpux::VPU::NCEInvariant::isInputActTypeSupported(arch, getInput().getType().cast<vpux::NDTypeInterface>(),
                                                             getInputChannelAlignment()) &&
            vpux::VPU::NCEInvariant::isOutputActTypeSupported(getOutput().getType().cast<vpux::NDTypeInterface>(),
                                                              getOutputChannelAlignment()));
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment()));
}

bool vpux::VPU::NCEConvolutionOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

DimArr vpux::VPU::NCEConvolutionOp::restrictedFusionAxes() {
    return {Dims4D::Act::C};
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEConvolutionOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto arch = getArch(getOperation());
    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(arch, outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    switch (arch) {
    case VPU::ArchKind::NPU30XX: {
        // Weights sparsity is only supported for ZMajor Convolutions
        const auto inputType = getInput().getType().cast<vpux::NDTypeInterface>();
        const auto sparsityMode = inputType.getDimsOrder() == DimsOrder::NHWC ? VPU::SparsitySupport::SPARSE_WEIGHTS
                                                                              : VPU::SparsitySupport::NONE;
        return sparsityMode & excludeMode;
    }
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return NCESparsity::FULLY_SUPPORTED_SPARSITY_MODE & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyKernel(IE::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.getInput().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return NCEInvariant::verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch,
                                      log);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyKernel(IE::TransposedConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (mlir::failed(IE::canConvertTransposedConvToConv(origOp))) {
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto SY = 1;
    const auto SX = 1;

    const auto padTop = 0;
    const auto padBottom = 0;
    const auto padLeft = 0;
    const auto padRight = 0;

    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());

    return NCEInvariant::verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch,
                                      log);
}

mlir::LogicalResult vpux::VPU::NCEConvolutionOp::verifyConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                               vpux::NDTypeInterface inputType,
                                                               vpux::NDTypeInterface filterType,
                                                               vpux::NDTypeInterface outputType,
                                                               mlir::ArrayAttr kernelStrides, Logger log) {
    log.setName("NCEInvariant");

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    if (OC % alignment != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  alignment, OC);
        return mlir::failure();
    }

    const auto inOrder = inputType.getDimsOrder();

    Byte requiredCMX;
    if (inOrder == DimsOrder::NHWC) {
        requiredCMX = VPU::getRequiredCMXSizeForNCEOps({inputType, filterType, outputType}, OC);
    } else if (inOrder == DimsOrder::NCHW) {
        const auto remainder = (IC * KY * KX) % alignment;
        VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

        const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

        const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
        const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());

        const auto kernelSize = Shape{KY, KX};
        const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

        const auto activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
                VPU::NCESparsity::Mode::CM_CONV, kernelSize, kernelStridesVals[Dims4D::Strides::X],
                inputType.getElementType(), IC);

        requiredCMX = VPU::getRequiredCMXSizeForNCEOps({inputType, alignedFilterType, outputType}, OC) +
                      activationWindowSize * 1_Byte;
    } else {
        log.debug("[{0}] Unsupported input layout '{1}'", loc, inOrder);
        return mlir::failure();
    }

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Convolution, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}
