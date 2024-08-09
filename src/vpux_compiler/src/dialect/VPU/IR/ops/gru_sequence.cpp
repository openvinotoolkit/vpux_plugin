//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceOpAdaptor gru(operands, attrs, prop);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto initialStateType = gru.getInitialHiddenState().getType().cast<vpux::NDTypeInterface>();
    const auto outputStateType = initialStateType;
    const auto outputStateShape = outputStateType.getShape().raw();
    const auto seqLength = gru.getSeqLength();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};
    const auto middleStateType = initialStateType.changeShape(Shape(middleStateShape));

    inferredReturnShapes.push_back(middleStateType);
    inferredReturnShapes.push_back(outputStateType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GRUSequenceOp::backInferTileInfo(const vpux::TileInfo& outputTileY, vpux::Logger) {
    const auto origInputShape = getShape(getInputData());
    const auto origInitialHiddenStateShape = getShape(getInitialHiddenState());
    const auto origWShape = getShape(getWeights());
    const auto origRShape = getShape(getRecurrenceWeights());
    const auto origBShape = getShape(getBiases());

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo wTile(origWShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];
    inputTile.shape[Dim(1)] = outputTileY.shape[Dim(2)];
    inputTile.offsets[Dim(1)] = outputTileY.offsets[Dim(2)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];
    initialHiddenStateTile.shape[Dim(1)] = outputTileY.shape[Dim(1)];
    initialHiddenStateTile.offsets[Dim(1)] = outputTileY.offsets[Dim(1)];

    wTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    wTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    rTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    rTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    bTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    bTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    return InputTiling{{inputTile, initialHiddenStateTile, wTile, std::move(rTile), bTile}};
}

void vpux::VPU::GRUSequenceOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputYTile) {
    auto* ctx = this->getContext();
    auto inputTileInfo = inputTiling.tiles[0];
    VPUX_THROW_UNLESS(inputTileInfo.shape[Dim(1)] == outputYTile.shape[Dim(2)],
                      "seq_length dimension in input tile is incompatible with output tile, seq_length dimension of "
                      "input tile is {0}, but it's {1} in output tile",
                      inputTileInfo.shape[Dim(1)], outputYTile.shape[Dim(2)]);
    auto origSeqLength = getSeqLengthAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    auto tiledSeqLength = inputTileInfo.shape[Dim(1)];
    if (origSeqLength != tiledSeqLength) {
        const auto newSeqLength = getIntAttr(ctx, tiledSeqLength);
        this->setSeqLengthAttr(newSeqLength);
    }
    // The num_direction dimension of output can be tiled when direction is BIDIRECTONAL.
    // GRUSequence with BIDIRECTIONAL attribute will be split into two GRUSequence, GRUSequence with FORWARD attribute
    // and GRUSequence with REVERSE attribute when num_direction dimension was tiled.
    auto origDirection = getDirectionAttr().getValue();
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL && outputYTile.shape[Dim(1)] == 1 &&
        outputYTile.offsets[Dim(1)] == 0) {
        const auto newDirectionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD);
        this->setDirectionAttr(newDirectionAttr);
    }
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL && outputYTile.shape[Dim(1)] == 1 &&
        outputYTile.offsets[Dim(1)] == 1) {
        const auto newDirectionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::REVERSE);
        this->setDirectionAttr(newDirectionAttr);
    }
}

mlir::FailureOr<OutputTiling> vpux::VPU::GRUSequenceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GRUSequence currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    // There are two outputs of GRUSequence, the namse Y and Ho are from OpenVINO doc.
    // The shape of Y is [batch_size, num_directions, seq_len, hidden_size],
    // and the shape of Ho is [batch_size, num_directions, hidden_size].
    // Ho-tiles can be inferred by Y-tiles.
    const auto outputYType = baseOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputYShape = outputYType.getShape();
    Shape nTilesOnDimForOutputY(outputYShape.size(), 1);

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputYShape, log](ShapeRef nTilesOnDim,
                                                                              TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputYShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t tileDim = 0;
    while (!isSupportedTileSize(nTilesOnDimForOutputY, tilingMode)) {
        // Dimension of hidden_size can't be tiled.
        if (tileDim >= 3) {
            log.trace("can't get feasible tiling strategy for GRUSequence.");
            return mlir::failure();
        }
        if (nTilesOnDimForOutputY[Dim(tileDim)] >= outputYShape[Dim(tileDim)]) {
            ++tileDim;
        } else {
            ++nTilesOnDimForOutputY[Dim(tileDim)];
        }
    }
    log.trace("Isolated tiling strategy: {0}", nTilesOnDimForOutputY);

    auto origTilesY = fillDividedTiles(baseOp, nTilesOnDimForOutputY, outputYShape);
    return origTilesY;
}

//
// reifyTileGRUSequence
//

// There are some reasons for custom applyTileStrategy and reifyTile :
// 1, There are two outputs of GRUSequence, names Y and Ho from OpenVINO doc, it's a little
// different from TopK because the shapes of two TopK outputs are same. The shape of Y is
// [batch_size, num_directions, seq_len, hidden_size], and the shape of Ho is [batch_size,
// num_directions, hidden_size]. And Ho-tiles can be inferred by Y-tiles. Besides, a
// inferoutputHoTile logic is needed.
// 2, These tiles GRUSequence aren't independent of each other when seq_length dimension is tiled.
// So, output of previous tile GRUSequence is needed to create current tile GRUSequence.
// And the logic is different according to direction attribute.
// 3, The function to reverse tiles order for REVERSE mode is also necessary.
OutputTiling inferOutputHoTile(const OutputTiling& tilesY) {
    // The rank of outputHo equals 3.
    OutputTiling tilesHo;
    for (const auto& outputYTile : tilesY) {
        TileInfo outputHoTile(3);
        outputHoTile.shape[Dim(0)] = outputYTile.shape[Dim(0)];
        outputHoTile.shape[Dim(1)] = outputYTile.shape[Dim(1)];
        outputHoTile.shape[Dim(2)] = outputYTile.shape[Dim(3)];
        outputHoTile.offsets[Dim(0)] = outputYTile.offsets[Dim(0)];
        outputHoTile.offsets[Dim(1)] = outputYTile.offsets[Dim(1)];
        outputHoTile.offsets[Dim(2)] = outputYTile.offsets[Dim(3)];
        tilesHo.push_back(outputHoTile);
    }
    return tilesHo;
}

void reverseTilesOrderForReverseMode(OutputTiling& tilesY, int64_t seqLengthTile,
                                     IE::RNNSequenceDirection origDirection) {
    const auto reverse = [seqLengthTile, &tilesY](size_t i) {
        for (size_t j = 0; j < size_t(seqLengthTile / 2); ++j) {
            std::swap(tilesY[i + j], tilesY[i + seqLengthTile - 1 - j]);
        }
    };
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL) {
        for (size_t i = 0; i < tilesY.size(); i += seqLengthTile) {
            if (tilesY[i].offsets[Dim(1)] == 1) {
                reverse(i);
            }
        }
    }
    if (origDirection == IE::RNNSequenceDirection::REVERSE) {
        for (size_t i = 0; i < tilesY.size(); i += seqLengthTile) {
            reverse(i);
        }
    }
}

OutputTiling vpux::VPU::GRUSequenceOp::getOutputTiling(const vpux::TileInfo& firstOutputTile, vpux::Logger /*log*/) {
    return GRUSequenceOutputTiling(firstOutputTile);
}
