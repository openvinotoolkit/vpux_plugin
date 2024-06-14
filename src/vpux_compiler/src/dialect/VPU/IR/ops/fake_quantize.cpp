//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FakeQuantizeOp::verify() {
    const auto levels = getLevels();
    const auto lowFpType = getLowFpType();

    if (!levels.has_value()) {
        if (!lowFpType.has_value()) {
            return errorAt(*this, "Missing both levels and low precision floating type");
        }
        if (!lowFpType->isa<mlir::Float8E4M3FNType>() && !lowFpType->isa<mlir::Float8E5M2Type>()) {
            return errorAt(*this, "Unsupported low floating point type {0}", *lowFpType);
        }
    } else {
        if (lowFpType.has_value()) {
            return errorAt(*this,
                           "Contradicting attributes, both levels and low precision floating type were provided");
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::FakeQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FakeQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputLowType = quantize.getInputLow().getType().cast<vpux::NDTypeInterface>();
    const auto inputHighType = quantize.getInputHigh().getType().cast<vpux::NDTypeInterface>();
    const auto outputLowType = quantize.getOutputLow().getType().cast<vpux::NDTypeInterface>();
    const auto outputHighType = quantize.getOutputHigh().getType().cast<vpux::NDTypeInterface>();
    const auto autob = quantize.getAutoBroadcast();

    const auto outShapeOrResult = IE::broadcastEltwiseShape(
            {inputType.getShape().raw(), inputLowType.getShape().raw(), inputHighType.getShape().raw(),
             outputLowType.getShape().raw(), outputHighType.getShape().raw()},
            autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        const auto outType = inputType.changeShape(Shape(outShapeOrResult.value()));
        inferredReturnTypes.push_back(outType);
    }

    return outShapeOrResult;
}

void vpux::VPU::FakeQuantizeOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value input, ::mlir::Value input_low, ::mlir::Value input_high,
                                      ::mlir::Value output_low, ::mlir::Value output_high,
                                      /*optional*/ mlir::IntegerAttr levels, /*optional*/ ::mlir::TypeAttr low_fp_type,
                                      vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(odsBuilder, odsState, input, input_low, input_high, output_low, output_high, levels, low_fp_type,
          auto_broadcast, {});
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::FakeQuantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::FakeQuantizeOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::FakeQuantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::FakeQuantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::FakeQuantizeOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

bool VPU::FakeQuantizeOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto fqOp = mlir::cast<VPU::FakeQuantizeOp>(getOperation());
    const auto outputShape = getShape(fqOp.getOutput());
    auto numClusters = VPU::getOptimalNumClusters(fqOp, outputShape[Dims4D::Act::C], strategy);

    SmallVector<vpux::NDTypeInterface> ioBuffers;
    ioBuffers.push_back(getDistributedOutputTypeFromOp(fqOp, fqOp.getOutput().getType(), numClusters, strategy));

    for (const auto input : fqOp->getOperands()) {
        ioBuffers.push_back(getDistributedActivationTypeFromOp(fqOp, input.getType(), numClusters, strategy));
    }

    return fitIntoCMX(std::move(ioBuffers), reservedMem);
}

//
// SWOpInterface
//

bool vpux::VPU::FakeQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 6,
                      "FakeQuantizeOp requires 5 input and 1 output, but the number of buffer is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::FakeQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::FakeQuantizeOp::supportCycleCostCalculation() {
    return false;
}
