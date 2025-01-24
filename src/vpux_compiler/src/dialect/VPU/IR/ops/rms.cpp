//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RMSOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RMSOpAdaptor rms(operands, attrs, prop);
    if (mlir::failed(rms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = rms.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto gammaType = rms.getGamma().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const auto gammaShape = gammaType.getShape().raw();
    const auto inputRank = inputShape.size();
    const auto gammaRank = gammaShape.size();

    if ((inputRank < 3) || (inputRank > 4)) {
        return errorAt(loc, "Input tensor rank should be 3 or 4. Got {0}D tensor.", inputRank);
    }

    if (inputShape[inputRank - 1] != gammaShape[gammaRank - 1]) {
        return errorAt(loc, "Input width should be the same as gamma. Got input width = {0} and gamma width = {1}",
                       inputShape[inputRank - 1], gammaShape[0]);
    }

    inferredReturnTypes.push_back(inType);
    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::RMSOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

void vpux::VPU::RMSOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
                             ::mlir::Value gamma, ::mlir::FloatAttr epsilon) {
    build(odsBuilder, odsState, input, gamma, epsilon, {});
}

vpux::VPU::DistributionInfo vpux::VPU::RMSOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::RMSOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "RMSOp requires 2 inputs and 1 output, but the number of buffer is {0}",
                      buffers.size());

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

bool vpux::VPU::RMSOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::RMSOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::RMSOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    TileInfo gammaTile(getShape(getGamma()));
    auto inTile = outputTile;
    gammaTile.shape[Dim(Dims4D::Act::W)] = inTile.shape[Dim(Dims4D::Act::W)];

    return TilingInfo{{std::move(inTile), std::move(gammaTile)}};
}

void vpux::VPU::RMSOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::RMSOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outputRank = outputType.getShape().size();
    SmallVector<int64_t> maxNumTiles;
    maxNumTiles = getMaxNumTilesWithAxesExclusion(op, /*axis:*/ {checked_cast<int64_t>(outputRank - 1)});
    return vpux::getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}
