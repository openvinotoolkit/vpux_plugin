//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN1MeanVarOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN1MeanVarOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto iType = op.getSum().getType().cast<vpux::NDTypeInterface>();
    const auto iShape = iType.getShape().raw();
    const auto iOrder = iType.getDimsOrder();
    const auto inN = iShape[0];

    // For default order NxCxW shape, expecting data in memory as NxWxC (i.e C-minor)
    // for alignment with original MvnOp main tensor NHWC layout.
    // The (0,1,2,3) -> (0,2,3,1) permutation is available via 'DimsOrder::NHWC'
    VPUX_THROW_UNLESS(iOrder == DimsOrder::NHWC, "Expecting NHWC layout, got {0}", iOrder);

    const auto fullShape = parseIntArrayAttr<int64_t>(op.getOrigShape());
    const auto fullC = fullShape[Dims4D::Act::C.ind()];
    const auto fullN = fullShape[Dims4D::Act::N.ind()];

    VPUX_THROW_UNLESS(inN == fullN, "Mismatch N: {0} != {1}", inN, fullN);

    const auto outC = (op.getAcrossChannels() ? 1 : fullC);
    const auto outW = op.getNormalizeVariance() ? 2 : 1;  // {mean, var} or {mean}

    SmallVector<int64_t> oShape{inN, outC, 1, outW};
    auto oShapeType = iType.changeShape(Shape(oShape));
    auto oType = oShapeType.changeElemType(op.getOutputType());
    inferredReturnTypes.push_back(oType);

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MVN1MeanVarOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::MVN1MeanVarOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

//
// SWOpInterface
//

bool vpux::VPU::MVN1MeanVarOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "MVN1MeanVarOp requires 1 input and 1 output, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::MVN1MeanVarOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MVN1MeanVarOp::supportCycleCostCalculation() {
    return false;
}
//
// build
//

void vpux::VPU::MVN1MeanVarOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value sum,
                                     ::mlir::ArrayAttr orig_shape, bool across_channels, bool normalize_variance,
                                     ::mlir::APFloat eps, ::mlir::Type output_type) {
    build(builder, state, sum, orig_shape, across_channels, normalize_variance, eps, output_type, {});
}
