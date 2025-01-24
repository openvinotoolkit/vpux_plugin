//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherDMAOp::inferReturnTypes(mlir::MLIRContext*, std::optional<mlir::Location>,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPU::GatherDMAOpAdaptor gatherDMAOp(operands, attrs, prop);
    const auto indicesType = gatherDMAOp.getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();

    if (!gatherDMAOp.getAxisValue().has_value()) {
        return mlir::failure();
    }
    const auto axis = gatherDMAOp.getAxisValue().value();

    const auto inputType = gatherDMAOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    auto outputShape = inputShape.toValues();
    outputShape[Dim(axis)] = indicesShape[Dim(axis)];

    auto outType = inputType.changeShape(Shape(std::move(outputShape)));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GatherDMAOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getShape(getInput());
    const auto origIndicesShape = getShape(getIndices());
    bool hasAxisTensor = false;

    int64_t axisValue = 0;

    if (getAxisValueAttr() != nullptr) {
        axisValue = getAxisValueAttr().cast<mlir::IntegerAttr>().getValue().getSExtValue();
    }
    if (getAxis() != nullptr) {
        auto axisConst = getAxis().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(axisConst != nullptr, "Only constant input is supported for axis");
        VPUX_THROW_UNLESS(axisConst.getContentAttr().isSplat(), "Axis value must be a scalar");
        const auto axisContent = axisConst.getContent();
        axisValue = axisContent.getSplatValue<int64_t>();
        hasAxisTensor = true;
    }

    return vpux::backInferGatherDMATile(outputTile, origInputShape, origIndicesShape, axisValue, hasAxisTensor, log);
}

void vpux::VPU::GatherDMAOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::GatherDMAOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for Gather currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    VPUX_THROW_WHEN(getAxisValueAttr() == nullptr, "Miss axis value, for op {0} at '{1}'", baseOp->getName(), getLoc());

    auto axisValue = getAxisValueAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto outputType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    const auto inputType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputSize = inputType.getCompactAllocSize();
    const auto indicesType = getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesSize = indicesType.getCompactAllocSize();
    const auto outputRank = static_cast<int64_t>(outputShape.size());

    SmallVector<int64_t> dataBeforeAxisRange, indicesRange, dataAfterAxisRange;
    for (int64_t i = 0; i < outputRank; ++i) {
        if (i < axisValue) {
            dataBeforeAxisRange.push_back(i);
        } else if (axisValue == i) {
            indicesRange.push_back(i);
        } else {
            dataAfterAxisRange.push_back(i);
        }
    }

    SmallVector<int64_t> tileDimOrder;
    if (inputSize > indicesSize) {
        // TileDimOrder: {dataBeforeAxisRange, dataAfterAxisRange, indicesRange}.
        tileDimOrder.insert(tileDimOrder.end(), dataBeforeAxisRange.begin(), dataBeforeAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataAfterAxisRange.begin(), dataAfterAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), indicesRange.begin(), indicesRange.end());
    } else {
        // TileDimOrder: {indicesRange, dataBeforeAxisRange, dataAfterAxisRange}.
        tileDimOrder.insert(tileDimOrder.end(), indicesRange.begin(), indicesRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataBeforeAxisRange.begin(), dataBeforeAxisRange.end());
        tileDimOrder.insert(tileDimOrder.end(), dataAfterAxisRange.begin(), dataAfterAxisRange.end());
    }

    auto nTilesOnDimforGather = getSupportedNTilesOnDimforGather(tileDimOrder, baseOp, tilingMode, log);

    log.trace("Isolated tiling strategy: {0}", nTilesOnDimforGather);
    return fillDividedTiles(baseOp, nTilesOnDimforGather, outputShape);
}
