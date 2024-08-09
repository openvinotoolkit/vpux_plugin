//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AccumulateOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AccumulateOpAdaptor accumulate(operands, attrs, prop);
    if (mlir::failed(accumulate.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = accumulate.getLhs().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = accumulate.getRhs().getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(in1Type == in2Type, "Types of operands of VPU.Accumulate don't match: {0} vs {1}", in1Type,
                      in2Type);
    const auto outType = in1Type;
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::AccumulateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::AccumulateOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    // No attributes - do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::AccumulateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::AccumulateOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t /*numTiles*/) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributedTensorNative vpux::VPU::AccumulateOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributedTensorNative(mlir::cast<VPU::SWOpInterface>(getOperation()), shape,
                                                     distributionMode, numTiles, numClusters, alignment,
                                                     uniformDistributedSegments, overlapParams);
}

bool vpux::VPU::AccumulateOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto accumulateOp = mlir::cast<VPU::AccumulateOp>(getOperation());
    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto numClusters = VPU::getOptimalNumClusters(accumulateOp, outputType.getShape(), strategy);

    SmallVector<Byte> buffersSize{
            VPU::getTotalAllocSizeWithDistribution(getLhs().getType(),
                                                   getActivationDistributionAttrFromOp(accumulateOp, getLhs().getType(),
                                                                                       numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(getRhs().getType(),
                                                   getActivationDistributionAttrFromOp(accumulateOp, getRhs().getType(),
                                                                                       numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getLhsScale().getType(), getActivationDistributionAttrFromOp(accumulateOp, getLhsScale().getType(),
                                                                                 numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getRhsScale().getType(), getActivationDistributionAttrFromOp(accumulateOp, getRhsScale().getType(),
                                                                                 numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(getOutput().getType(),
                                                   getOutputDistributionAttrFromOp(accumulateOp, getOutput().getType(),
                                                                                   numClusters.getInt(), strategy))};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// SWOpInterface
//

bool vpux::VPU::AccumulateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 5,
                      "AccumulateOp requires 4 inputs and 1 output, but the number of buffers is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    llvm::transform(buffers, std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::AccumulateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::AccumulateOp::supportCycleCostCalculation() {
    return false;
}
