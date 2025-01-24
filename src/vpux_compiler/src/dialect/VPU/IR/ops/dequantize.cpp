//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DequantizeOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DequantizeOpAdaptor dequantize(operands, attrs, prop);
    if (mlir::failed(dequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dequantize.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = dequantize.getDstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

void vpux::VPU::DequantizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                    mlir::TypeAttr dstElemType) {
    build(builder, state, input, dstElemType, nullptr);
}

bool vpux::VPU::DequantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::DequantizeOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

bool VPU::DequantizeOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef, ShapeRef) {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(getResult());
    }
    auto numOfCluster = getNumTiles(*this);
    //  Currently dequantize is used for filters of convolutions which are tiled on OC for SOK
    auto OC = outputShape[Dims4D::Filter::OC];
    return OC >= NCEInvariant::VPU_CHANNEL_ALIGNMENT * numOfCluster;
}

//
// SWOpInterface
//

bool vpux::VPU::DequantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "DequantizeOp requires 1 inputs and 1 outputs, but the number of buffers is {0}", buffers.size());

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

bool vpux::VPU::DequantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DequantizeOp::supportCycleCostCalculation() {
    return false;
}

bool vpux::VPU::DequantizeOp::isVFSupported() {
    return true;
}
