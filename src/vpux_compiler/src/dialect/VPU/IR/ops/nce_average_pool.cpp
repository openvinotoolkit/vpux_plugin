//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEAveragePoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(input, output, Byte(0));
}

bool vpux::VPU::NCEAveragePoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                             Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCEAveragePoolOp::isSupported(IE::AvgPoolOp op, LogCb logCb, bool checkLayout,
                                              bool checkChannelAlignment) {
    if (op.getType().getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (op.getRoundingType() != IE::RoundingType::FLOOR) {
        logCb(formatv("Unsupported rounding mode '{0}'", op.getRoundingType()));
        return false;
    }

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(op.getKernelSize()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.getPadsBegin(), op.getPadsEnd());

    // NCE could not support exclude pad, exclude pad take effect when there is padding
    if (op.getExcludePads() && pads.enabled()) {
        return false;
    }

    if (!NCEInvariant::isAttrsSupported(VPU::getArch(op), KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right,
                                        logCb)) {
        return false;
    }

    const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();

    if (inputType.getElementType().isSignedInteger() || outputType.getElementType().isSignedInteger() ||
        inputType.getElementType().isUnsignedInteger() || outputType.getElementType().isUnsignedInteger()) {
        return false;
    }

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
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::verify() {
    const auto op = getOperation();
    const auto arch = getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == VPU::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    const auto logCb = [op](const formatv_object_base& msg) {
        (void)errorAt(op, "{0}", msg.str());
    };

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(getKernelSize()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = getPad().getTop().getValue().getSExtValue();
    const auto padBottom = getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = getPad().getLeft().getValue().getSExtValue();
    const auto padRight = getPad().getRight().getValue().getSExtValue();

    if (!NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEAveragePoolOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.getInput()).raw();

    const auto windowShape = parseIntArrayAttr<int64_t>(op.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(op.getStrides());

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = SmallVector<int64_t>({padTop, padLeft});
    const auto dataPaddingAbove = SmallVector<int64_t>({padBottom, padRight});

    const auto shapeI64 =
            inferAvgPoolOutputShape(inShape, windowStrides, dataPaddingBelow, dataPaddingAbove, windowShape);

    auto inputType = op.getInput().getType();
    if (auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        inputType = sparseInputType.getData();
    }
    const auto outputType = inputType.cast<vpux::NDTypeInterface>().changeShape(Shape(shapeI64));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEAveragePoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origPadding = toPadInfo(getPad());

    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, getKernelSize(), getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    return inputTiling;
}

void vpux::VPU::NCEAveragePoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    VPU::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEAveragePoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEAveragePoolOp::getKernelSizeVal() {
    return parseIntArrayAttr<int64_t>(getKernelSize());
}

SmallVector<int64_t> vpux::VPU::NCEAveragePoolOp::getStridesVal() {
    return parseIntArrayAttr<int64_t>(getStrides());
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEAveragePoolOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEAveragePoolOp::getExplicitDistributedTensorAttr(
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
bool VPU::NCEAveragePoolOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
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

    auto nceOp = mlir::cast<NCEAveragePoolOp>(getOperation());
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

bool VPU::NCEAveragePoolOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEAveragePoolOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEAveragePoolOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEAveragePoolOp>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape()[Dims4D::Act::C], strategy);
    return fitIntoCMX(getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy),
                      getDistributedOutputTypeFromOp(nceOp, nceOp.getOutput().getType(), numClusters, strategy),
                      reservedMem);
}

bool VPU::NCEAveragePoolOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCEAveragePoolOp>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, nceOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    auto distributedInputType =
            getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, newDistributedTensorType);
}

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

bool vpux::VPU::NCEAveragePoolOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEAveragePoolOp::sparsitySupport() {
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
        VPUX_THROW("NCEAveragePoolOp is not supported for {0}", arch);
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX:
        return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::verifyKernel(IE::AvgPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (origOp.getInput().getType().cast<vpux::NDTypeInterface>().getRank() != 4) {
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];
    const auto pads = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());

    // NCE could not support exclude pad, exclude pad take effect when there is padding
    if (origOp.getExcludePads() && pads.enabled()) {
        return mlir::failure();
    }
    const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
    return NCEInvariant::verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, arch,
                                      log);
}
