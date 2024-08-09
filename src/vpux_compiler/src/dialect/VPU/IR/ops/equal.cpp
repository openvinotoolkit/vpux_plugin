//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EqualOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::EqualOpAdaptor equal(operands, attrs, prop);
    if (mlir::failed(equal.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = equal.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = equal.getInput2().getType().cast<vpux::NDTypeInterface>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       equal.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        auto outType =
                mlir::RankedTensorType::get(outShapeRes.value(), getBool8Type(ctx), createTensorAttrFromType(in1Type));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::EqualOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributedTensorNative vpux::VPU::EqualOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributedTensorNative(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                     distributionMode, numTiles, numClusters, alignment,
                                                     uniformDistributedSegments, overlapParams);
}

bool VPU::EqualOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto equalOp = mlir::cast<VPU::EqualOp>(getOperation());
    const auto outputType = equalOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(equalOp, outputType.getShape(), strategy);

    SmallVector<Byte> buffersSize{
            VPU::getTotalAllocSizeWithDistribution(getInput1().getType(),
                                                   getActivationDistributionAttrFromOp(equalOp, getInput1().getType(),
                                                                                       numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(getInput2().getType(),
                                                   getActivationDistributionAttrFromOp(equalOp, getInput2().getType(),
                                                                                       numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(),
                    getOutputDistributionAttrFromOp(equalOp, getOutput().getType(), numClusters.getInt(), strategy))};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// SWOpInterface
//

bool vpux::VPU::EqualOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "EqualOp requires 2 inputs and 1 outputs, but the number of buffer is {0}",
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

bool vpux::VPU::EqualOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::EqualOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::EqualOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                               ::mlir::Value input2, vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(builder, state, input1, input2, auto_broadcast, {});
}
