//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::InterpolateOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    auto outShape = IE::calcOutputShapes(interpolate, loc, Logger::global(), ctx);

    const auto inType = interpolate.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::InterpolateOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    if (strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return true;
    }

    return false;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::InterpolateOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& /*overlapParams*/) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

void vpux::VPU::InterpolateOp::build(
        ::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
        /*optional*/ ::mlir::Value sizes, /*optional*/ ::mlir::Value scales, /*optional*/ ::mlir::Value axes,
        /*optional*/ ::mlir::ArrayAttr sizes_attr, /*optional*/ ::mlir::ArrayAttr scales_attr,
        /*optional*/ ::mlir::ArrayAttr axes_attr, /*optional*/ ::mlir::ArrayAttr tile_offset_attr,
        /*optional*/ ::mlir::ArrayAttr initial_input_dims_attr, /*optional*/ ::mlir::ArrayAttr initial_output_dims_attr,
        vpux::IE::InterpolateAttr attr) {
    build(odsBuilder, odsState, input, sizes, scales, axes, sizes_attr, scales_attr, axes_attr, tile_offset_attr,
          initial_input_dims_attr, initial_output_dims_attr, nullptr, nullptr, nullptr, attr);
}

//
// SWOpInterface
//

bool vpux::VPU::InterpolateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 2 && buffers.size() <= 5,
                      "InterpolateOp requires 1 input, 3 optional attribution inputs and 1 output, but the "
                      "number of buffer is {0}",
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

bool vpux::VPU::InterpolateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::InterpolateOp::supportCycleCostCalculation() {
    return false;
}

InputTiling vpux::VPU::InterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origAxes = IE::extractIntVector(getLoc(), getAxes(), getAxesAttrAttr());
    VPUX_THROW_UNLESS(mlir::succeeded(origAxes), "InterpolateOp::backInferTileInfo failed to extract axes");

    auto iShape = getInitialInputDimsAttr().has_value() ? parseIntArrayAttr<int64_t>(getInitialInputDimsAttr().value())
                                                        : to_small_vector(getShape(getInput()));
    auto oShape = getInitialOutputDimsAttr().has_value()
                          ? parseIntArrayAttr<int64_t>(getInitialOutputDimsAttr().value())
                          : to_small_vector(getShape(getOutput()));
    auto initialInputOffsets = getInitialInputOffsetAttr().has_value()
                                       ? parseIntArrayAttr<int64_t>(getInitialInputOffsetAttr().value())
                                       : SmallVector<int64_t>(getShape(getInput()).size(), 0);

    auto initialOutputOffsets = getInitialOutputOffsetAttr().has_value()
                                        ? parseIntArrayAttr<int64_t>(getInitialOutputOffsetAttr().value())
                                        : SmallVector<int64_t>(getShape(getOutput()).size(), 0);

    mlir::Builder builder(*this);
    if (!getInitialInputDimsAttr().has_value()) {
        auto newInitialInputDims = builder.getI64ArrayAttr(iShape);
        setInitialInputDimsAttrAttr(newInitialInputDims);
    }
    if (!getInitialOutputDimsAttr().has_value()) {
        auto newInitialOutputDims = builder.getI64ArrayAttr(oShape);
        setInitialOutputDimsAttrAttr(newInitialOutputDims);
    }

    SmallVector<double> tileOffset(iShape.size(), 0.f);
    auto newTileOffset = builder.getF64ArrayAttr(tileOffset);
    setTileOffsetAttrAttr(newTileOffset);

    const auto axesVal = origAxes.value();
    vpux::Scales fwdScales;
    // Compute scale-factors based on full I/O resolution ratio
    SmallVector<double> backwardScale;
    for (size_t i = 0; i < axesVal.size(); i++) {
        backwardScale.push_back(static_cast<double>(iShape[axesVal[i]]) / oShape[axesVal[i]]);
    }

    SmallVector<int64_t> beginPads(iShape.size(), 0);
    SmallVector<int64_t> endPads(iShape.size(), 0);

    mlir::FailureOr<SmallVector<int64_t>> inferedInputTile;
    auto coordMode = getAttr().getCoordMode().getValue();
    auto interpolateMode = getAttr().getMode().getValue();
    auto nearestMode = getAttr().getNearestMode().getValue();
    auto currentInputShape = to_small_vector(getShape(getInput()));
    auto inTiles = vpux::backInferInterpolateTile(outputTile, iShape, oShape, initialInputOffsets, initialOutputOffsets,
                                                  currentInputShape, interpolateMode, coordMode, nearestMode, log);
    auto newInputOffset = to_small_vector(inTiles.tiles[0].offsets);

    // Recalculate the backward scale based on the new input/output shape
    for (size_t i = 0; i < axesVal.size(); i++) {
        fwdScales.push_back(static_cast<double>(outputTile.shape[Dim(axesVal[i])]) /
                            inTiles.tiles[0].shape[Dim(axesVal[i])]);
    }

    auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;
    auto forwardInferedShape = IE::inferInterpOutShape(
            getLoc(), axesVal, inTiles.tiles[0].shape, {beginPads}, {endPads}, shapeCalcMode,
            IE::extractIntVector(getLoc(), getSizes(), getSizesAttr().value_or<mlir::ArrayAttr>({})), {fwdScales},
            mlir::Float32Type::get(getContext()), log);

    // TODO: E#36319 we counting only endpads - begin pad might matter for offsets not for dims
    auto shapeArray = to_small_vector(outputTile.shape);
    if (endPads.size() == shapeArray.size()) {
        for (auto shapeOrig : shapeArray | indexed) {
            endPads[shapeOrig.index()] = shapeOrig.value() - forwardInferedShape[shapeOrig.index()];
        }
    }
    auto convertShape = [](::mlir::Value plainShape) {
        auto origShape = getShape(plainShape);
        TileInfo tileShape{origShape.size()};
        tileShape.shape = getShape(plainShape).toValues();
        return tileShape;
    };

    if (auto size = getSizes()) {
        inTiles.tiles.push_back(convertShape(size));
    }
    if (auto scale = getScales()) {
        inTiles.tiles.push_back(convertShape(scale));
    }
    if (auto axis = getAxes()) {
        inTiles.tiles.push_back(convertShape(axis));
    }
    inTiles.pads = {0, endPads[2], 0, endPads[3]};
    return inTiles;
}

void vpux::VPU::InterpolateOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outTile) {
    if (!inputTiling.pads.has_value()) {
        return;
    }
    mlir::Builder builder(*this);

    TileInfo inputTile = inputTiling.tiles.begin()[0];

    const auto origInputDims = IE::extractIntVector(getLoc(), getAxes(), getAxesAttrAttr());
    const auto initialInputDims = parseIntArrayAttr<int64_t>(getInitialInputDimsAttrAttr());
    const auto initialOutputDims = parseIntArrayAttr<int64_t>(getInitialOutputDimsAttrAttr());

    const auto initialInputOffset = builder.getI64ArrayAttr(to_small_vector(inputTiling.tiles[0].offsets));
    const auto initialOutputOffset = builder.getI64ArrayAttr(to_small_vector(outTile.offsets));
    setInitialInputOffsetAttrAttr(initialInputOffset);
    setInitialOutputOffsetAttrAttr(initialOutputOffset);

    const auto numDims = initialInputDims.size();

    SmallVector<double> tileOffset(numDims, 0.f);
    auto newTileOffset = builder.getF64ArrayAttr(tileOffset);
    setTileOffsetAttrAttr(newTileOffset);

    SmallVector<int64_t> endPads(numDims, 0);
    SmallVector<int64_t> beginPads(numDims, 0);

    endPads[2] = inputTiling.pads.value().right;
    endPads[3] = inputTiling.pads.value().bottom;

    auto newEndPads = builder.getI64ArrayAttr(endPads);
    auto newBeginPads = builder.getI64ArrayAttr(beginPads);

    // forcing scales calculation mode
    auto calcModeAttr = vpux::IE::InterpolateCalcModeAttr::get(this->getContext(), IE::InterpolateCalcMode::SCALES);

    auto newAttrs = IE::InterpolateAttr::get(getContext(), getAttr().getMode(), calcModeAttr, getAttr().getCoordMode(),
                                             getAttr().getNearestMode(), getAttr().getAntialias(), newBeginPads,
                                             newEndPads, getAttr().getCubeCoeff());

    auto axesValue = IE::extractIntVector(getLoc(), getAxes(), getAxesAttr().value_or<mlir::ArrayAttr>({})).value();
    auto scale = SmallVector<double>(axesValue.size(), 1);
    // Recompute SCALE attribute based on new input output tiling
    for (auto axis : axesValue | indexed) {
        const auto axisDim = Dim(axis.value());
        scale[axis.index()] = static_cast<double>(outTile.shape[axisDim]) / inputTiling.tiles[0].shape[axisDim];
    }

    // set pads begin + end attrs
    setAttrAttr(newAttrs);
    setScalesAttrAttr(builder.getF64ArrayAttr(scale));
}

// Generates a list of tiles of the output tensor that satisfy the CMX memory constrains and don't alter the original
// scaling factors. When generating the tiles the following characteristics are taken into consideration:
// 1. The data layout, in order to find an optimal manner of transferring the data between CMX and DDR
//    NCHW -> the tiling dim order is C, H, W
//    NHWC -> the tiling dim order is H, W, C
// 2. Scaling factor, based on value of the scaling factor is in one axis, the following decisions are taken:
//    If the greatest common factor between in dim and out dim is 1 then remove axis from tiling axis
//    If the greatest common factor > 1 then choose the smallest common factor that satisfies constrains
//    If in and out dim are equal (scaling factor == 1) then iteratively increase the number of tiles on this axis since
//    the constrains are satisfied or the number of tiles is equal with minimum value between in and out dim
mlir::FailureOr<OutputTiling> vpux::VPU::InterpolateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto origOp = mlir::dyn_cast<VPU::InterpolateOp>(op);
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    const auto inType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeArray = inType.getShape().raw();
    const auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outShape = outType.getShape();
    const auto outShapeArray = outType.getShape().raw();
    const auto axes = IE::extractIntVector(op->getLoc(), origOp.getAxes(), origOp.getAxesAttr());
    const auto axesVal = axes.value();
    Shape nTilesOnDim(outShapeArray.size(), 1);

    auto getTileDimOrder = [&]() {
        VPUX_THROW_UNLESS(outType.getDimsOrder() == DimsOrder::NCHW || outType.getDimsOrder() == DimsOrder::NHWC,
                          "Interpolate Op only support NCHW and NHWC layout, but got '{0}'", outType.getDimsOrder());

        auto dimOrder = outType.getDimsOrder().toPermutation();
        // Get greatest common factor between two values
        auto getGreaterCommonFactor = [&](int64_t firstVal, int64_t secondVal) {
            while (firstVal != secondVal) {
                if (firstVal > secondVal) {
                    firstVal = firstVal - secondVal;
                } else {
                    secondVal = secondVal - firstVal;
                }
            }
            return firstVal;
        };

        for (auto axis : axesVal | indexed) {
            const auto axisDim = Dim(axis.value());
            // Check if greatest common factor between current in and out dim is 1 and if yes remove the axis from the
            // list of tiling dims
            if (getGreaterCommonFactor(inShapeArray[axisDim.ind()], outShapeArray[axisDim.ind()]) == 1) {
                llvm::erase_if(dimOrder, [axisDim](Dim dim) {
                    return (dim == axisDim);
                });
            }
        }

        log.nest(2).debug("getTilingStrategy: dimOrder = {0}", dimOrder);
        return dimOrder;
    };

    const auto tileDimOrder = getTileDimOrder();

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outShape, log](ShapeRef nTilesOnDim,
                                                                      TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };
    // Compute next common factor of two integer values greater or equal then currFactor
    // if it doesn't exist getNextCommonFactor returns 0
    auto getNextCommonFactor = [&](int64_t firstVal, int64_t secondVal, int64_t currFactor) -> int64_t {
        auto minVal = std::min(firstVal, secondVal);
        while (currFactor < minVal) {
            currFactor++;
            if (firstVal % currFactor == 0 && secondVal % currFactor == 0) {
                return currFactor;
            }
        }
        return 0;
    };

    // Get an feasible isolated tiling strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while (tileDimIter < tileDimOrder.end() &&
               nTilesOnDim[dimToTile] >=
                       std::min(inShapeArray[(*tileDimIter).ind()], outShapeArray[(*tileDimIter).ind()])) {
            dimToTile = *(++tileDimIter);
        }
        VPUX_THROW_WHEN(tileDimIter == tileDimOrder.end(), "Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
        // Check if current tiling axis is an interpolation axis and if yes increase the number of tiles to
        // next common factor between in and out dim on current axis
        auto axesIt = llvm::find(axesVal, (*tileDimIter).ind());
        if (axesIt != axesVal.end() && inShapeArray[*axesIt] != outShapeArray[*axesIt]) {
            // If getNextCommonFactor returns > 0 then updates the number of tiles in current axis to returned value
            // else move to next axis considered for tiling
            auto nextFactor =
                    getNextCommonFactor(inShapeArray[*axesIt], outShapeArray[*axesIt], nTilesOnDim[dimToTile]);
            if (nextFactor > 0) {
                nTilesOnDim[Dim((*axesIt))] = nextFactor;
            } else {
                dimToTile = *(++tileDimIter);
            }
        } else {
            ++nTilesOnDim[dimToTile];
        }
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outShape);
}
