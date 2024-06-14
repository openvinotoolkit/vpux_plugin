//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::VPU::PReluOp::verify() {
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto slopeType = getNegativeSlope().getType().cast<vpux::NDTypeInterface>();
    const auto slopeShape = slopeType.getShape().raw();

    if (slopeShape.size() != 4 || inShape.size() != 4) {
        return errorAt(*this, "Tiling restrictions require slope to have a 4D shape, got size of {0}",
                       slopeShape.size());
    }

    if (inShape[Dims4D::Act::C.ind()] != slopeShape[Dims4D::Act::C.ind()] ||
        slopeShape[Dims4D::Act::C.ind()] != slopeType.getShape().totalSize()) {
        return errorAt(*this,
                       "4D slope shape should have the last dim equal to the channel input dim, as broadcast with "
                       "numpy values is not supported: {0} != {1}",
                       inShape[Dims4D::Act::C.ind()], slopeShape[Dims4D::Act::C.ind()]);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::PReluOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PReluOpAdaptor prelu(operands, attrs);
    if (mlir::failed(prelu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = prelu.getInput().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::PReluOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    TileInfo inputTile(getShape(getInput()));
    TileInfo slopeTile(getShape(getNegativeSlope()));
    inputTile = outputTile;
    if (outputTile.shape[Dims4D::Act::C] != slopeTile.shape[Dims4D::Act::C]) {
        VPUX_THROW("Tiling per channel output is not supported for now, proposed {0} channel shape does not match the "
                   "slope value {1}.",
                   outputTile.shape[Dims4D::Act::C], slopeTile.shape[Dims4D::Act::C]);
    }

    return TilingInfo{{std::move(inputTile), std::move(slopeTile)}};
}

void vpux::VPU::PReluOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing here
}

mlir::FailureOr<OutputTiling> vpux::VPU::PReluOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// build
//

void vpux::VPU::PReluOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input1,
                               ::mlir::Value input2) {
    build(odsBuilder, odsState, input1, input2, nullptr);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::PReluOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::PReluOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

bool VPU::PReluOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto preluOp = mlir::cast<VPU::PReluOp>(getOperation());
    const auto outputType = preluOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(preluOp, outputType.getShape()[Dims4D::Act::C], strategy);
    auto distInput1Type =
            getDistributedActivationTypeFromOp(preluOp, preluOp.getInput().getType(), numClusters, strategy);
    auto distInput2Type =
            getDistributedActivationTypeFromOp(preluOp, preluOp.getNegativeSlope().getType(), numClusters, strategy);
    auto distOutputType = getDistributedOutputTypeFromOp(preluOp, preluOp.getOutput().getType(), numClusters, strategy);
    return fitIntoCMX({distInput1Type, distInput2Type, distOutputType}, reservedMem);
}

//
// SWOpInterface
//

bool vpux::VPU::PReluOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "PReluOp requires 2 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::PReluOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::PReluOp::supportCycleCostCalculation() {
    return false;
}
