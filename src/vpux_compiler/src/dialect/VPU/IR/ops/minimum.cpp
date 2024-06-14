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

mlir::LogicalResult vpux::VPU::MinimumOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MinimumOpAdaptor minimum(operands, attrs);
    if (mlir::failed(minimum.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = minimum.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = minimum.getInput2().getType().cast<vpux::NDTypeInterface>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       minimum.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = in1Type.changeShape(Shape(outShapeRes.value()));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

bool vpux::VPU::MinimumOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
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

bool VPU::MinimumOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto minimumOp = mlir::cast<VPU::MinimumOp>(getOperation());
    const auto outputType = minimumOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(minimumOp, outputType.getShape()[Dims4D::Act::C], strategy);
    auto distInput1Type =
            getDistributedActivationTypeFromOp(minimumOp, minimumOp.getInput1().getType(), numClusters, strategy);
    auto distInput2Type =
            getDistributedActivationTypeFromOp(minimumOp, minimumOp.getInput2().getType(), numClusters, strategy);
    auto distOutputType =
            getDistributedOutputTypeFromOp(minimumOp, minimumOp.getOutput().getType(), numClusters, strategy);
    return fitIntoCMX({distInput1Type, distInput2Type, distOutputType}, reservedMem);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::MinimumOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return vpux::VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                         distributionMode, numTiles, numClusters, alignment,
                                                         uniformDistributedSegments);
}

bool vpux::VPU::MinimumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "MinimumOp requires 2 inputs and 1 outputs, but the number of buffer is {0}",
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

bool vpux::VPU::MinimumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MinimumOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MinimumOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                 ::mlir::Value input2, vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(builder, state, input1, input2, auto_broadcast, {});
}
