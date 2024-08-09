//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NormalizeL2Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NormalizeL2OpAdaptor normalizeL2(operands, attrs, prop);
    if (mlir::failed(normalizeL2.verify(loc))) {
        return mlir::failure();
    }
    const auto axes = parseIntArrayAttr<int64_t>(normalizeL2.getAxesValueAttr());
    if (axes.empty()) {
        return mlir::failure();
    }

    const auto inType = normalizeL2.getData().getType().cast<vpux::NDTypeInterface>();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NormalizeL2Op::verify() {
    const auto inRank = getData().getType().cast<vpux::NDTypeInterface>().getRank();
    auto axesVec = parseIntArrayAttr<int64_t>(getAxesValueAttr());

    for (auto& axis : axesVec) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(*this, "Axes values should be unique");
    }

    return mlir::success();
}

bool vpux::VPU::NormalizeL2Op::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = getData().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inputType.getShape();

    auto axesVec = parseIntArrayAttr<int64_t>(getAxesValueAttr());
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    auto isCompatibleStrategy{[&](auto strategyToCheck, auto dimensionToCheck) {
        return strategy == strategyToCheck && inShape[dimensionToCheck] > 1 &&
               std::find(axesVec.begin(), axesVec.end(), dimensionToCheck.ind()) == axesVec.end();
    }};

    if (isCompatibleStrategy(VPU::MultiClusterStrategy::SplitOverHeight, Dims4D::Act::H)) {
        return true;
    }

    if (isCompatibleStrategy(VPU::MultiClusterStrategy::SplitOverKernel, Dims4D::Act::C)) {
        return true;
    }

    return false;
}

bool VPU::NormalizeL2Op::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto NormalizeL2Op = mlir::cast<VPU::NormalizeL2Op>(getOperation());
    const auto outputType = NormalizeL2Op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(NormalizeL2Op, outputType.getShape(), strategy);

    SmallVector<Byte> buffersSize{
            VPU::getTotalAllocSizeWithDistribution(
                    getData().getType(), getActivationDistributionAttrFromOp(NormalizeL2Op, getData().getType(),
                                                                             numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(getOutput().getType(),
                                                   getOutputDistributionAttrFromOp(NormalizeL2Op, getOutput().getType(),
                                                                                   numClusters.getInt(), strategy))};
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

vpux::VPU::DistributedTensorNative vpux::VPU::NormalizeL2Op::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributedTensorNative(mlir::cast<VPU::SWOpInterface>(getOperation()), shape,
                                                     distributionMode, numTiles, numClusters, alignment,
                                                     uniformDistributedSegments, overlapParams);
}

bool vpux::VPU::NormalizeL2Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "NormalizeL2Op requires 1 inputs and 1 outputs, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::NormalizeL2Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::NormalizeL2Op::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::NormalizeL2Op::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return InputTiling{{outputTile}};
}

void vpux::VPU::NormalizeL2Op::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::NormalizeL2Op::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return tileShape[dimToTile] < maxNumTiles[dimToTile.ind()];
    };

    const auto axesValues = parseIntArrayAttr<int64_t>(getAxesValueAttr());
    const auto isSpecifiedAxis = [&](const vpux::Dim* tiledDim) -> bool {
        // cases when there is no chance to skip dims, all is specified => default tiling mode
        if (axesValues.size() == outputShape.size()) {
            return false;
        } else {
            // cases when there is chance to tile other dims than specified
            for (int64_t axis : axesValues) {
                if (axis == static_cast<int64_t>(tiledDim->ind())) {
                    return true;
                }
            }
        }
        return false;
    };

    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while (((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim))) ||
               isSpecifiedAxis(tileDimIter)) {
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW_WHEN(tilingMode == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                                op->getLoc());
            }
        }
        ++nTilesOnDim[dimToTile];
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outputShape);
}
