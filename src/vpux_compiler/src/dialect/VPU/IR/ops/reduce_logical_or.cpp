//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReduceLogicalOrOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReduceLogicalOrOpAdaptor reduceLogicalOr(operands, attrs);
    if (mlir::failed(reduceLogicalOr.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceLogicalOr.getInput();
    const auto keepDims = reduceLogicalOr.getKeepDims();

    auto axesValue = parseIntArrayAttr<int64_t>(reduceLogicalOr.getAxesValue());

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::ReduceLogicalOrOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::ReduceLogicalOrOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto inputType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();
    const auto axesVec = parseIntArrayAttr<int64_t>(getAxesValueAttr());
    return checkStrategyCompatibilityReduce(strategy, numTiles, inShape, axesVec);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::ReduceLogicalOrOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return vpux::VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                         distributionMode, numTiles, numClusters, alignment,
                                                         uniformDistributedSegments);
}

bool vpux::VPU::ReduceLogicalOrOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    return fitIntoCMXReduce(getOperation(), buffers, reservedMem);
}

bool vpux::VPU::ReduceLogicalOrOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMXReduce(getOperation(), buffers);
}

bool vpux::VPU::ReduceLogicalOrOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ReduceLogicalOrOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                  vpux::Logger /*log*/) {
    const auto inShape = getInput().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto axesValue = getAxesValue();
    const auto keepDims = getKeepDims();

    return backInferReduceTile(outputTile, inShape, axesValue, keepDims);
}

void vpux::VPU::ReduceLogicalOrOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::ReduceLogicalOrOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();
    const auto keepDims = getKeepDims();
    SmallVector<int64_t> maxNumTiles;

    if (keepDims) {
        const auto axes = parseIntArrayAttr<int64_t>(getAxesValueAttr());
        maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);
    } else {
        const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
        const auto outputShape = outputType.getShape();
        maxNumTiles = to_small_vector(outputShape);
    }

    return vpux::getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}

//
// build
//

void vpux::VPU::ReduceLogicalOrOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                         ::mlir::ArrayAttr axes_value, ::mlir::UnitAttr keep_dims) {
    build(builder, state, input, axes_value, keep_dims, {});
}
