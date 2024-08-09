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

mlir::LogicalResult vpux::VPU::MaximumOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MaximumOpAdaptor maximum(operands, attrs, prop);
    if (mlir::failed(maximum.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = maximum.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = maximum.getInput2().getType().cast<vpux::NDTypeInterface>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       maximum.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = mlir::RankedTensorType::get(outShapeRes.value(), in1Type.getElementType(),
                                                         createTensorAttrFromType(in1Type));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

bool vpux::VPU::MaximumOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && inShape[Dims4D::Act::H] > 1) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && inShape[Dims4D::Act::C] > 1) {
        return true;
    }

    return false;
}

bool VPU::MaximumOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto maximumOp = mlir::cast<VPU::MaximumOp>(getOperation());
    const auto outputType = maximumOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(maximumOp, outputType.getShape(), strategy);

    SmallVector<Byte> buffersSize{
            VPU::getTotalAllocSizeWithDistribution(getInput1().getType(),
                                                   getActivationDistributionAttrFromOp(maximumOp, getInput1().getType(),
                                                                                       numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(getInput2().getType(),
                                                   getActivationDistributionAttrFromOp(maximumOp, getInput2().getType(),
                                                                                       numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(),
                    getOutputDistributionAttrFromOp(maximumOp, getOutput().getType(), numClusters.getInt(), strategy))};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

vpux::VPU::DistributedTensorNative vpux::VPU::MaximumOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributedTensorNative(mlir::cast<VPU::SWOpInterface>(getOperation()), shape,
                                                     distributionMode, numTiles, numClusters, alignment,
                                                     uniformDistributedSegments, overlapParams);
}

bool vpux::VPU::MaximumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "MaximumOp requires 2 inputs and 1 outputs, but the number of buffer is {0}",
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

bool vpux::VPU::MaximumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MaximumOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MaximumOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                 ::mlir::Value input2, vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(builder, state, input1, input2, auto_broadcast, {});
}
