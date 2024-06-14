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

mlir::LogicalResult vpux::VPU::SelectOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SelectOpAdaptor select(operands, attrs);
    if (mlir::failed(select.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = select.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = select.getInput2().getType().cast<vpux::NDTypeInterface>();
    const auto in3Type = select.getInput3().getType().cast<vpux::NDTypeInterface>();

    const auto outShapeRes =
            IE::broadcastEltwiseShape({in1Type.getShape().raw(), in2Type.getShape().raw(), in3Type.getShape().raw()},
                                      select.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = in2Type.changeShape(Shape(outShapeRes.value()));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::SelectOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::SelectOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

bool VPU::SelectOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto selectOp = mlir::cast<VPU::SelectOp>(getOperation());
    const auto outputShape = getShape(selectOp.getOutput());
    auto numClusters = VPU::getOptimalNumClusters(selectOp, outputShape[Dims4D::Act::C], strategy);

    SmallVector<vpux::NDTypeInterface> ioBuffers;
    ioBuffers.push_back(
            getDistributedOutputTypeFromOp(selectOp, selectOp.getOutput().getType(), numClusters, strategy));

    for (const auto input : selectOp->getOperands()) {
        ioBuffers.push_back(getDistributedActivationTypeFromOp(selectOp, input.getType(), numClusters, strategy));
    }

    return fitIntoCMX(std::move(ioBuffers), reservedMem);
}

//
// SWOpInterface
//

void vpux::VPU::SelectOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input1,
                                ::mlir::Value input2, ::mlir::Value input3,
                                vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(odsBuilder, odsState, input1, input2, input3, auto_broadcast, {});
}

bool vpux::VPU::SelectOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 4, "SelectOp requires 3 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::SelectOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::SelectOp::supportCycleCostCalculation() {
    return false;
}
