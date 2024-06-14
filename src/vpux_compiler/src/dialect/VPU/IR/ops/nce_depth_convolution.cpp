//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <openvino/op/convolution.hpp>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output, Byte reservedMem) {
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto OC = output.getShape()[Dims4D::Act::C];

    const Shape kernelSize{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
    const auto strideW = kernelStrides[Dims4D::Strides::X];

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), filter.getTotalAllocSize(), output.getTotalAllocSize(),
                                 NCEInvariant::getWeightsTableSize(OC)};

    if (getActivationWindow() != nullptr) {
        int64_t activationWindowSize = NCESparsity::getActivationWindowSize(NCESparsity::Mode::DW_CONV, kernelSize,
                                                                            strideW, input.getElementType(), 1);
        buffers.push_back(activationWindowSize * 1_Byte);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    auto arch = getArch(getOperation());
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEDepthConvolutionOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface filter,
                                                  vpux::NDTypeInterface output) {
    return fitIntoCMX(input, filter, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEDepthConvolutionOp::isSupported(IE::GroupConvolutionOp op, LogCb logCb, bool checkLayout,
                                                   bool checkChannelAlignment) {
    if (op.getType().getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }
    if (getShape(op.getFilter()).size() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return false;
    }

    const auto inputShape = getShape(op.getInput());
    const auto IC = inputShape[Dims4D::Act::C];

    const auto filterShape = getShape(op.getFilter());
    const auto fIC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!op.getGroups().has_value()) {
        logCb(formatv("Grouped convolution does not have groups attribute"));
        return false;
    }
    if (op.getGroups().value() != OC) {
        logCb(formatv("Unsupported group size: '{0}' expected '{1}'", op.getGroups(), OC));
        return false;
    }
    if (fIC != 1) {
        logCb(formatv("Group Convolution with more than one filter per input channel is not supported"));
        return false;
    }
    if (OC != IC) {
        logCb(formatv("Group Convolution has '{0}' groups, expected '{1}'", OC, IC));
        return false;
    }

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.getPadsBegin(), op.getPadsEnd());

    if (!NCEInvariant::isAttrsSupported(VPU::getArch(op), KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right,
                                        logCb)) {
        return false;
    }

    const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(getArch(op), inputType, getInputChannelAlignmentImpl(inputType),
                                                   false) ||
            !NCEInvariant::isOutputActTypeSupported(outputType, getOutputChannelAlignmentImpl(outputType))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto arch = getArch(op);
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 2, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult verifyDepthConv(mlir::Location loc, VPU::ArchKind arch, VPU::NCEDepthConvolutionOpAdaptor op,
                                    mlir::Value output) {
    const auto logCb = [loc](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    const auto outputShape = getShape(output);
    const auto OC = outputShape[Dims4D::Act::C];

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    if (!VPU::NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    const auto weightsTableShape = getShape(op.getWeightsTable());
    const auto expectedWeightsTableShape = VPU::NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape != expectedWeightsTableShape) {
        return errorAt(loc, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape,
                       expectedWeightsTableShape);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verify() {
    const auto op = getOperation();
    const auto arch = getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    const NCEDepthConvolutionOpAdaptor convAdaptor(op->getOperands(), op->getAttrDictionary(),
                                                   op->getPropertiesStorage(), op->getRegions());
    if (mlir::failed(verifyDepthConv(op->getLoc(), arch, convAdaptor, getOutput()))) {
        return mlir::failure();
    }

    const auto inputType = getInput().getType().cast<NDTypeInterface>();
    const auto IC = inputType.getShape()[Dims4D::Act::C];

    const auto outputType = getOutput().getType().cast<NDTypeInterface>();
    const auto filterType = getFilter().getType().cast<NDTypeInterface>();

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));

    if (getActivationWindow() != nullptr) {
        const auto KY = filterShape[Dims4D::Filter::KY];
        const auto KX = filterShape[Dims4D::Filter::KX];
        const auto kernelSize = Shape{KY, KX};

        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
        const auto SX = kernelStrides[Dims4D::Strides::X];

        const auto activationWindowShape = getShape(getActivationWindow());
        const auto expectedActivationWindowShape = NCESparsity::inferActivationWindowShape(
                NCESparsity::Mode::DW_CONV, kernelSize, SX, inputType.getElementType(), 1);

        if (activationWindowShape != expectedActivationWindowShape) {
            return errorAt(op, "Got wrong shape for 'activationWindow' '{0}', expected '{1}'", activationWindowShape,
                           expectedActivationWindowShape);
        }

        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize, SX,
                                                                        inputType.getElementType(), IC);

        if (getActivationWindowChannelLength() != bitPatternSize) {
            return errorAt(op, "Got wrong value for 'activation_window_channel_length' '{0}', expected '{1}'",
                           getActivationWindowChannelLength(), bitPatternSize);
        }
    }

    const auto alignedFilterShape = filterType.getShape();
    const auto expectedAlignedFilterShape = inferAlignedFilterShape(outputType, filterType);

    if (alignedFilterShape != expectedAlignedFilterShape) {
        return errorAt(op, "Got wrong shape for 'filter' '{0}', expected '{1}'", alignedFilterShape,
                       expectedAlignedFilterShape);
    }

    return mlir::success();
}

Shape vpux::VPU::NCEDepthConvolutionOp::inferAlignedFilterShape(NDTypeInterface output, NDTypeInterface filter) {
    const auto rawFilterShape = Shape(parseIntArrayAttr<int64_t>(this->getRawFilterShape()));
    const auto KY = rawFilterShape[Dims4D::Filter::KY];
    const auto KX = rawFilterShape[Dims4D::Filter::KX];

    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto alignment = NCEInvariant::getAlignment(filter.getElementType());

    const auto remainder = (KY * KX) % alignment;

    if (remainder == 0) {
        return Shape{OC, 1, KY, KX};
    }

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;

    return Shape{OC, KY * KX + padding, 1, 1};
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEDepthConvolutionOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.getRawFilterShape()));
    const auto fIC = filterShape[Dims4D::Filter::IC];

    if (fIC != 1) {
        return errorAt(loc, "Non depthwize convolution case");
    }

    // Adjust input shape to reuse helpers for standard convolution
    auto inShape = getShape(op.getInput()).toValues();
    inShape[Dims4D::Act::C] = 1;

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

vpux::InputTiling vpux::VPU::NCEDepthConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                      vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origPadding = toPadInfo(getPad());
    const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));

    // This op incorporates bias values in WeightsTable
    const auto origBiasShape = ShapeRef();

    auto inputTiling = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, getStrides(),
                                              origPadding, origInputShape[Dims4D::Act::C]);
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

    if (getActivationWindow() != nullptr) {
        inputTiling.tiles.push_back(VPU::getActivationWindowTile(this, outputTile));
    }

    return inputTiling;
}

void vpux::VPU::NCEDepthConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    VPU::adjustPaddings(this, inputTiling);
    VPU::adjustRawFilterShape(this, outputTile);

    const auto inputType = getInput().getType().cast<NDTypeInterface>();
    const auto IC = inputTiling.tiles[0].shape[Dims4D::Act::C];

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto kernelSize = Shape{KY, KX};

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
    const auto SX = kernelStrides[Dims4D::Strides::X];

    if (getActivationWindow() != nullptr) {
        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize, SX,
                                                                        inputType.getElementType(), IC);

        setActivationWindowChannelLengthAttr(getIntAttr(getContext(), bitPatternSize));
    }
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEDepthConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEDepthConvolutionOp::getKernelSizeVal() {
    const auto kernelShape = Shape(parseIntArrayAttr<int64_t>(getRawFilterShape()));
    const auto KY = kernelShape[Dims4D::Filter::KY];
    const auto KX = kernelShape[Dims4D::Filter::KX];
    return {KY, KX};
}

SmallVector<int64_t> vpux::VPU::NCEDepthConvolutionOp::getStridesVal() {
    return parseIntArrayAttr<int64_t>(getStrides());
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEDepthConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEDepthConvolutionOp::getExplicitDistributedTensorAttr(
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
bool VPU::NCEDepthConvolutionOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
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

    auto nceOp = mlir::cast<NCEDepthConvolutionOp>(getOperation());
    Shape inputShape = getShape(nceOp.getInput()).toValues();
    auto inputType = nceOp.getInput().getType().cast<NDTypeInterface>();
    // If has custom output shape, infer the input shape
    if (outputShape != getShape(nceOp->getResult(0))) {
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

    return isSOHSupportedByDPU(inputType, inputShape, numTiles, true, VPU::getArch(nceOp.getOperation()));
}

bool VPU::NCEDepthConvolutionOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset,
                                                                     ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEDepthConvolutionOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset,
                                                                      ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEDepthConvolutionOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEDepthConvolutionOp>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape()[Dims4D::Act::C], strategy);
    return fitIntoCMX(getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy),
                      getDistributedFilterTypeFromOp(nceOp, nceOp.getFilter().getType(), numClusters, strategy),
                      getDistributedOutputTypeFromOp(nceOp, nceOp.getOutput().getType(), numClusters, strategy),
                      reservedMem);
}

bool VPU::NCEDepthConvolutionOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCEDepthConvolutionOp>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, nceOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    auto distributedInputType =
            getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOp, nceOp.getFilter().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

bool vpux::VPU::NCEDepthConvolutionOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEDepthConvolutionOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto arch = getArch(getOperation());
    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(arch, outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    switch (arch) {
    case VPU::ArchKind::NPU30XX:
        return VPU::SparsitySupport::NONE;
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verifyKernel(IE::GroupConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.getInput().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }
    if (origOp.getFilter().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (!origOp.getGroups().has_value()) {
        log.trace("[{0}] Grouped convolution does not have groups", origOp->getLoc());
        return mlir::failure();
    }
    if (origOp.getGroups().value() != OC) {
        log.trace("[{0}] Unsupported group size: '{1}' expected '{2}'", origOp->getLoc(), origOp.getGroups(), OC);
        return mlir::failure();
    }
    if (filtersPerInChan != 1) {
        log.trace("[{0}] Group Convolution with more than one filter per channel is not supported", origOp->getLoc());
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.getInput());
    const auto IC = inputShape[Dims4D::Act::C];
    if (OC != IC) {
        log.trace("[{0}] Group Convolution has {1} groups, expected {2}", origOp->getLoc(), OC, IC);
        return mlir::failure();
    }

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

mlir::LogicalResult vpux::VPU::NCEDepthConvolutionOp::verifyGroupConvCMX(
        mlir::Location loc, mlir::ModuleOp module, vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
        vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelStrides, bool hasActivationWindow, Logger log) {
    log.setName("NCEInvariant");

    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelStrides.size());

    const auto filterShape = filterType.getShape();
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    if (OC % alignment != 0) {
        log.debug("[{0}] Output channels count of depthwise convolution must be a multiple of {1}, got {2}", loc,
                  alignment, OC);
        return mlir::failure();
    }

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    const auto padding = (remainder > 0) ? (alignment - remainder) : 0;
    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, filtersPerInChan * KY * KX + padding};
    const auto alignedFilterType = mlir::RankedTensorType::get(alignedWeightShape, filterType.getElementType());

    int64_t activationWindowSize = 0;
    if (hasActivationWindow) {
        const Shape kernelSizeVals{KY, KX};
        const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));
        activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
                VPU::NCESparsity::Mode::DW_CONV, kernelSizeVals, kernelStridesVals[Dims4D::Strides::X],
                inputType.getElementType(), 1);
    }

    const auto requiredCMX = VPU::getRequiredCMXSizeForNCEOps({inputType, alignedFilterType, outputType}, OC) +
                             activationWindowSize * 1_Byte;

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Depthwise Convolution, available '{1}', required '{2}'", loc,
                  cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}
