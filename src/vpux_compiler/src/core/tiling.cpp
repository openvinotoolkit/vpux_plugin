//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ThreadPool.h>
#include <algorithm>
#include <cmath>
#include <optional>

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

//
// TileInfo
//

// Imagine shape [8, 8, 9] and divisor [2, 3, 2].
// We'll end up with the following shapes and offsets.

// Shapes   {[4, 3, 5], [4, 3, 5], [4, 3, 5], [4, 3, 5], [4, 2, 5], [4, 2, 5],
//           [4, 3, 4], [4, 3, 4], [4, 3, 4], [4, 3, 4], [4, 2, 4], [4, 2, 4]}
// Offsets  {[0, 0, 0], [4, 0, 0], [0, 3, 0], [4, 3, 0], [0, 6, 0], [4, 6, 0],
//           [0, 0, 5], [4, 0, 5], [0, 3, 5], [4, 3, 5], [0, 6, 0], [4, 6, 5]}

//
// Divide tiles and return the size and interval per current dimension.
//
// The size of the tiles can contain at most two distinct values.
// As for the above case, take the dimension 1 for example:
// shape 8 is divided into 3 tiles with tile size [3, 3, 2]
// we return a tuple containing two values of tile sizes (3 and 2) and the interval between them (which is 2)
//
// Returned tuple contains three values:
// tileSize - the first kind of value of divided tile sizes (3 in the example above)
// remainderTileSize - the second kind of value of divided tile sizes (2 in the example above)
// tileSizeInterval - the numbers of tiles are divided as tileSize (2 in the example above)
std::optional<std::tuple<int64_t, int64_t, size_t>> divideTileSizeAndInterval(Dim dimension, ShapeRef divisors,
                                                                              ShapeRef shape,
                                                                              ArrayRef<int64_t> alignment,
                                                                              Logger log = Logger::global()) {
    const auto shapeVal = shape[dimension];
    const auto divisorVal = divisors[dimension];
    const auto alignmentVal = alignment[dimension.ind()];

    if (shapeVal < divisorVal) {
        // Indivisible when the shape size is smaller than the divisor
        return std::nullopt;
    }

    int64_t tileSize, remainderTileSize;
    size_t tileSizeInterval;
    if (alignmentVal > 1) {
        // Whenever there is alignment, all N-1 tiles need to be multiple
        // of said align value.
        // The remainder shape is admitted to not be a mutiple of align value,
        // since this code is tasked to simply tile the original shape, not also align it.
        tileSize = alignValUp(divUp(shapeVal, divisorVal), alignmentVal);
        remainderTileSize = shapeVal - tileSize * (divisorVal - 1);

        if (remainderTileSize <= 0) {
            log.trace("DivideTiles can't meet the request: ShapeVal = {0}, divisorVal = {1}, alignmentTileSize = {2}",
                      shapeVal, divisorVal, tileSize);
            return std::nullopt;
        }

        tileSizeInterval = divisorVal - 1;

    } else {
        // When there is no alignment needed, we prefer to distribute the remainder in an
        // equal way across the first tiles.
        // For example 17 tiled 4 ways can be done as:
        // A) [5, 5, 5, 2] when we take the ceil value of the division
        // and leave the remainder as the last tile.
        // B) [5, 4, 4, 4] when we take the floor of the division and distribute
        // the remainder across the first tiles.
        // In any of the two cases, we'll have just 2 distinct values in the shape array.
        tileSize = shapeVal / divisorVal;
        remainderTileSize = shapeVal % divisorVal;

        if (remainderTileSize) {
            tileSizeInterval = remainderTileSize;
            remainderTileSize = tileSize;
            tileSize++;
        } else {
            tileSizeInterval = divisorVal;
        }
    }

    return std::make_tuple(tileSize, remainderTileSize, tileSizeInterval);
}

// Arguments:
// dividedTiles - final array of computed tiles
// divisors - array with the tile divisors for each dimension
// shape - original shape to tile
// alignment - array with alignments for each dimension
// dimensionIndex - current dimension index to be processed
// ongoingTile - individual tile solution which we construct and push to dividedTiles array
// unrollSpatialFirst - unroll in the order of NHWC if it is true
mlir::LogicalResult divideTiles(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape,
                                ArrayRef<int64_t> alignment, size_t dimensionIndex, vpux::TileInfo& ongoingTile,
                                bool unrollSpatialFirst) {
    // If spatial first, unroll in order of NHWC
    // else, follow the default order NCHW
    const auto spatialFirstOrder = DimsOrder::NHWC;
    const auto dimension = unrollSpatialFirst ? spatialFirstOrder.dimAt(dimensionIndex) : Dim(dimensionIndex);

    auto tileSizeIntervalResult = divideTileSizeAndInterval(dimension, divisors, shape, alignment);
    if (!tileSizeIntervalResult) {
        return mlir::failure();
    }
    int64_t tileSize = std::get<0>(*tileSizeIntervalResult);
    int64_t remainderTileSize = std::get<1>(*tileSizeIntervalResult);
    size_t tileSizeInterval = std::get<2>(*tileSizeIntervalResult);

    // Iterate and backtrack on the current list of shapes and offsets
    const size_t totalTileSize = divisors[dimension];
    int64_t tileOffset = 0;
    for (auto tileIndex : irange(totalTileSize)) {
        int64_t tileShape = tileIndex < tileSizeInterval ? tileSize : remainderTileSize;
        ongoingTile.shape[dimension] = tileShape;
        ongoingTile.offsets[dimension] = tileOffset;
        ongoingTile.axis[dimension] = totalTileSize;
        tileOffset += tileShape;

        // Full dividedTile is created so need to register the solution
        if (dimensionIndex == (divisors.size() - 1)) {
            dividedTiles.push_back(ongoingTile);
        } else {
            auto isSuccessful = divideTiles(dividedTiles, divisors, shape, alignment, dimensionIndex + 1, ongoingTile,
                                            unrollSpatialFirst);
            if (mlir::failed(isSuccessful)) {
                return mlir::failure();
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult divideTilesYuvToRgbOp(OutputTiling& dividedTiles, ShapeRef divisors, ShapeRef shape,
                                          vpux::TileInfo& ongoingTile) {
    // N C H W. Tile on C and H dimensions, minimum granularity is 2
    const auto dimC = Dim(Dims4D::Act::C);
    const auto dimH = Dim(Dims4D::Act::H);
    ongoingTile.shape[Dim(Dims4D::Act::N)] = shape[Dim(Dims4D::Act::N)];
    ongoingTile.shape[Dim(Dims4D::Act::W)] = shape[Dim(Dims4D::Act::W)];

    ongoingTile.axis[Dim(Dims4D::Act::N)] = divisors[Dim(Dims4D::Act::N)];
    ongoingTile.axis[Dim(Dims4D::Act::C)] = divisors[Dim(Dims4D::Act::C)];
    ongoingTile.axis[Dim(Dims4D::Act::H)] = divisors[Dim(Dims4D::Act::H)];
    ongoingTile.axis[Dim(Dims4D::Act::W)] = divisors[Dim(Dims4D::Act::W)];

    const auto shapeValC = shape[dimC];
    auto divisorValC = divisors[dimC];

    size_t tileSizeInitC, tileSizeC, remainderTileSizeC;

    tileSizeInitC = shapeValC / divisorValC;
    tileSizeC = tileSizeInitC + (tileSizeInitC % 2);
    divisorValC = shapeValC / tileSizeC;
    remainderTileSizeC = shapeValC % tileSizeC;

    ongoingTile.shape[dimC] = tileSizeC;
    for (int i = 0; i < divisorValC; ++i) {
        ongoingTile.offsets[dimC] = tileSizeC * i;

        const auto shapeValH = shape[dimH];
        auto divisorValH = divisors[dimH];
        size_t tileSizeInitH, tileSizeH, remainderTileSizeH;

        tileSizeInitH = shapeValH / divisorValH;
        tileSizeH = tileSizeInitH + (tileSizeInitH % 2);
        divisorValH = shapeValH / tileSizeH;
        remainderTileSizeH = shapeValH % tileSizeH;
        ongoingTile.shape[dimH] = tileSizeH;

        for (int j = 0; j < divisorValH; ++j) {
            ongoingTile.offsets[dimH] = tileSizeH * j;
            dividedTiles.push_back(ongoingTile);
        }

        if (remainderTileSizeH) {
            ongoingTile.shape[dimH] = remainderTileSizeH;
            ongoingTile.offsets[dimH] = tileSizeH * divisorValH;
            dividedTiles.push_back(ongoingTile);
        }
    }

    if (remainderTileSizeC) {
        ongoingTile.shape[dimC] = remainderTileSizeC;
        ongoingTile.offsets[dimC] = tileSizeC * divisorValC;

        const auto shapeValH = shape[dimH];
        auto divisorValH = divisors[dimH];
        size_t tileSizeInitH, tileSizeH, remainderTileSizeH;

        tileSizeInitH = shapeValH / divisorValH;
        tileSizeH = tileSizeInitH + (tileSizeInitH % 2);
        divisorValH = shapeValH / tileSizeH;
        remainderTileSizeH = shapeValH % tileSizeH;
        ongoingTile.shape[dimH] = tileSizeH;

        for (int j = 0; j < divisorValH; ++j) {
            ongoingTile.offsets[dimH] = tileSizeH * j;
            dividedTiles.push_back(ongoingTile);
        }

        if (remainderTileSizeH) {
            ongoingTile.shape[dimH] = remainderTileSizeH;
            ongoingTile.offsets[dimH] = tileSizeH * divisorValH;
            dividedTiles.push_back(ongoingTile);
        }
    }

    return mlir::success();
}

mlir::FailureOr<OutputTiling> fillDividedTilesYuvToRgbOp(ShapeRef divisors, ShapeRef shape) {
    OutputTiling dividedTiles;
    size_t totalTileNum = 1;
    for (auto divVal : divisors) {
        totalTileNum *= divVal;
    }
    dividedTiles.reserve(totalTileNum);

    auto ongoingTile = vpux::TileInfo(divisors.size());
    ongoingTile.isCompletedTile = true;

    auto isSuccessful = divideTilesYuvToRgbOp(dividedTiles, divisors, shape, ongoingTile);
    if (mlir::failed(isSuccessful)) {
        return mlir::failure();
    }

    return dividedTiles;
}

mlir::FailureOr<OutputTiling> vpux::fillDividedTiles(ShapeRef divisors, ShapeRef shape,
                                                     std::optional<ArrayRef<int64_t>> alignment,
                                                     bool unrollSpatialFirst) {
    OutputTiling dividedTiles;
    size_t totalTileNum = 1;
    for (auto divVal : divisors) {
        totalTileNum *= divVal;
    }
    dividedTiles.reserve(totalTileNum);

    auto ongoingTile = vpux::TileInfo(divisors.size());
    ongoingTile.isCompletedTile = true;

    auto alignmentShape = SmallVector<int64_t>(shape.size(), 1);
    auto alignmentShapeRef = ArrayRef(alignmentShape);
    if (alignment.has_value()) {
        alignmentShapeRef = alignment.value();
    }

    auto isSuccessful =
            divideTiles(dividedTiles, divisors, shape, alignmentShapeRef, 0, ongoingTile, unrollSpatialFirst);
    if (mlir::failed(isSuccessful)) {
        return mlir::failure();
    }

    return dividedTiles;
}

/*
 * Consider memory sharing and compare the DMA cost of different loop orders
 * Spatial first - Unroll the op in the order of NHWC
 * Non spatial first - Unroll the op in the order of NCHW
 * If the inputMemSize * divisors[C] > filterMemSize * divisors[H] * divisors[W], spatial first
 * else tiling over C first
 */
bool isSpatialFirstNestedTiling(mlir::Operation* op, ShapeRef divisors) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
        // Only use spatial first for operations with weights
        // The memory sharing has no difference if the op has no weights
        return false;
    }
    if (getNonOneDim(divisors).size() <= 1) {
        return false;
    }
    auto inputMemSize = getTotalSize(nceOp->getOperand(0));
    auto filterMemSize = getTotalSize(nceOp.getWeightsOperand());
    return inputMemSize * divisors[Dims4D::Act::C] >
           filterMemSize * divisors[Dims4D::Act::H] * divisors[Dims4D::Act::W];
}

std::optional<SmallVector<int64_t>> getAlignment(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    std::optional<SmallVector<int64_t>> optionalAlignment = std::nullopt;

    const auto getSubByteAlignmentFactor = [&op] {
        int64_t alignmentFactor = 1;

        auto setFactor = [&alignmentFactor](mlir::Value value) {
            const auto elemSize = vpux::getElemTypeSize(value.getType());
            if (elemSize.count() < CHAR_BIT) {
                alignmentFactor = std::max(alignmentFactor, CHAR_BIT / elemSize.count());
            }
        };

        // check all operands
        for (auto operand : op->getOperands()) {
            setFactor(operand);
        }

        // check output
        setFactor(op->getResult(0));

        return alignmentFactor;
    };

    auto alignment = SmallVector<int64_t>(shape.size(), 1);

    if (op->hasTrait<VPU::EltwiseOp>()) {
        if (auto factor = getSubByteAlignmentFactor(); factor > 1) {
            std::transform(divisors.begin(), divisors.end(), alignment.begin(), [&factor](int x) {
                return x > 1 ? factor : x;
            });
            optionalAlignment = alignment;
        }
    }

    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        alignment[vpux::Dims4D::Act::W.ind()] = VPU::NCEInvariant::getAlignment(outputType.getElementType());
        optionalAlignment = std::move(alignment);
    } else if (auto tilingIface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        alignment[vpux::Dims4D::Act::C.ind()] =
                std::lcm(tilingIface.getOutputChannelAlignment(), alignment[vpux::Dims4D::Act::C.ind()]);
        optionalAlignment = std::move(alignment);
    }

    return optionalAlignment;
}

std::optional<SmallVector<int64_t>> calculateAlignmentLCM(ArrayRef<SmallVector<int64_t>> alignments) {
    if (alignments.empty()) {
        return std::nullopt;
    }

    const auto numDimensions = alignments[0].size();
    for (auto alignment : alignments) {
        VPUX_THROW_UNLESS(alignment.size() == numDimensions, "All alignment must have the same size");
    }

    SmallVector<int64_t> alignmentLCM(numDimensions, 1);
    for (size_t dim = 0; dim < numDimensions; ++dim) {
        for (auto alignment : alignments) {
            alignmentLCM[dim] = std::lcm(alignmentLCM[dim], alignment[dim]);
        }
    }

    return alignmentLCM;
}

mlir::FailureOr<OutputTiling> vpux::fillDividedTiles(mlir::Operation* op, ShapeRef divisors, ShapeRef shape) {
    if (mlir::isa<VPU::YuvToRgbOp>(op)) {
        return fillDividedTilesYuvToRgbOp(divisors, shape);
    }

    std::optional<SmallVector<int64_t>> optionalAlignment = std::nullopt;
    if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(op)) {
        SmallVector<SmallVector<int64_t>> alignments;
        for (auto& innerOp : vfOp.getBody()->without_terminator()) {
            const auto outShape = getShape(innerOp.getResult(0));
            auto currentAlignment = getAlignment(&innerOp, divisors, outShape);
            if (currentAlignment.has_value()) {
                alignments.emplace_back(currentAlignment.value());
            }
        }
        optionalAlignment = calculateAlignmentLCM(alignments);
    } else {
        optionalAlignment = getAlignment(op, divisors, shape);
    }

    auto unrollSpatialFirst = isSpatialFirstNestedTiling(op, divisors);
    return vpux::fillDividedTiles(divisors, shape, optionalAlignment, unrollSpatialFirst);
}

//
// PadInfo
//

PadInfo vpux::backInferPadsTile(const TileInfo& outputTile, ShapeRef inShape, const PadInfo& origPads,
                                ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides) {
    const std::array<int64_t, Dims4D::Act::numSpatialDims> origPadsBegin = {origPads.top, origPads.left};
    const std::array<int64_t, Dims4D::Act::numSpatialDims> origPadsEnd = {origPads.bottom, origPads.right};

    SmallVector<int64_t> tilePadsBegin(Dims4D::Act::numSpatialDims);
    SmallVector<int64_t> tilePadsEnd(Dims4D::Act::numSpatialDims);

    for (auto ind : irange(Dims4D::Act::numSpatialDims)) {
        const auto spatialDim = Dims4D::Act::getSpatialDim(ind);

        const auto outSize = outputTile.shape[spatialDim];
        const auto outOffset = outputTile.offsets[spatialDim];

        const DimRange inputRange(0, inShape[spatialDim]);
        const DimRange tileRange(outOffset, outOffset + outSize);

        std::tie(std::ignore, tilePadsBegin[ind], tilePadsEnd[ind]) = inputForOutputDim(
                tileRange, kernel[ind], strides[ind], inputRange, origPadsBegin[ind], origPadsEnd[ind]);
    }

    return PadInfo(tilePadsBegin[1], tilePadsEnd[1], tilePadsBegin[0], tilePadsEnd[0]);
}

//
// Common tiling utilities
//

namespace {

struct PlaneTile final {
    DimRange width;
    DimRange height;

    int64_t area() const {
        return width.length() * height.length();
    }

    // Checks if rhs located completely in this.
    bool contains(const PlaneTile& other) const {
        return width.contains(other.width) && height.contains(other.height);
    }

    // Returns new `PlaneTile` which represents `other` as ROI of this.
    PlaneTile asROI(const PlaneTile& other) const {
        return {width.asROI(other.width), height.asROI(other.height)};
    }

    bool operator==(const PlaneTile& other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const PlaneTile& other) const {
        return !(*this == other);
    }

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PlaneTile [width tile = {0}, height tile = {1}]", width, height);
    }
};

struct PlaneTileSolution final {
    // Input tile which meets HW requirements in terms of alignment.
    PlaneTile inputTile;

    // Padding which should be applied to input tile in order to calculate output tile.
    // Meets HW requirements in terms of size and symmetry.
    PadInfo inputPad;

    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "PlaneTileSolution [inputTile = {0}, inputPad = {1}]", inputTile, inputPad);
    }
};

// Return input tile and padding required to calculate the output tile.
// Padding should be applied to the input tile. It could be asymmetric, or doesn't meet HW requirements in terms of its
// size.
// * initialInputDims - Dims of the whole input tensor (not of specific tile).
// * initialPad - padding which should be applied to the whole input tensor (not to specific tile).
std::tuple<PlaneTile, PadInfo> inputForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY,
                                                  int64_t strideX, int64_t strideY, ShapeRef initialInputDims,
                                                  const PadInfo& initialPad) {
    PlaneTile inputTile = {{0, 0}, {0, 0}};
    PadInfo pad = {0, 0, 0, 0};

    std::tie(inputTile.height, pad.top, pad.bottom) = inputForOutputDim(
            output.height, kernelY, strideY, {0, initialInputDims[Dims4D::Act::H]}, initialPad.top, initialPad.bottom);

    std::tie(inputTile.width, pad.left, pad.right) = inputForOutputDim(
            output.width, kernelX, strideX, {0, initialInputDims[Dims4D::Act::W]}, initialPad.left, initialPad.right);

    return std::make_tuple(inputTile, pad);
}

PlaneTileSolution solutionForOutputTile(const PlaneTile& output, int64_t kernelX, int64_t kernelY, int64_t strideX,
                                        int64_t strideY, ShapeRef initialInputDims, const PadInfo& initialPad) {
    PlaneTileSolution solution;
    std::tie(solution.inputTile, solution.inputPad) =
            inputForOutputTile(output, kernelX, kernelY, strideX, strideY, initialInputDims, initialPad);

    return solution;
}

}  // namespace

// inputTile planar H/W size should keep the same with original input H/W when no tiling over those axis.
// However the back inferring size may become smaller, e.g., OutputTile 7x7, Kernel 1x1, Stride 2x2.
// The inferring inputTile planar shape is 13x13 however original planar input shape may be 14x14, which will cause
// a redundant data slice from input. Here is to restore original input planar shape to avoid extra copies.
void restorePlanarShapeForInputTile(TileInfo& inputTile, ShapeRef origInputShape, vpux::Dim planarDim) {
    if (planarDim != Dims4D::Act::H && planarDim != Dims4D::Act::W) {
        VPUX_THROW("Invalid planar dim {0}", planarDim);
    }
    if (inputTile.shape[planarDim] > origInputShape[planarDim]) {
        VPUX_THROW("Invalid back inferring size {0} over dim {1}", inputTile.shape[planarDim], planarDim);
    }

    inputTile.shape[planarDim] = origInputShape[planarDim];
}

//
// Convolution tiling
//

InputTiling vpux::backInferConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                    ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding) {
    PlaneTile output;
    output.height.begin = outputTile.offsets[Dims4D::Act::H];
    output.height.end = outputTile.offsets[Dims4D::Act::H] + outputTile.shape[Dims4D::Act::H];
    output.width.begin = outputTile.offsets[Dims4D::Act::W];
    output.width.end = outputTile.offsets[Dims4D::Act::W] + outputTile.shape[Dims4D::Act::W];

    const auto strideY = strides[Dims4D::Strides::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[Dims4D::Strides::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution =
            solutionForOutputTile(output, origFilterShape[Dims4D::Filter::KX], origFilterShape[Dims4D::Filter::KY],
                                  strideX, strideY, origInputShape, origPadding);

    TileInfo inputTile(origInputShape);
    TileInfo filterTile(origFilterShape);
    TileInfo biasTile(origBiasShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.axis = outputTile.axis;

    inputTile.offsets[Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[Dims4D::Act::W] = solution.inputTile.width.length();

    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::H] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::H);
    }
    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::W] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::W);
    }

    filterTile.shape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];
    filterTile.offsets[Dims4D::Filter::OC] = outputTile.offsets[Dims4D::Act::C];

    if (!biasTile.shape.empty()) {
        biasTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        biasTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
        return TilingInfo{{inputTile, filterTile, biasTile}, solution.inputPad};
    }
    return TilingInfo{{inputTile, filterTile}, solution.inputPad};
}

InputTiling vpux::backInferGroupConvTile(const TileInfo& outputTile, ShapeRef origInputShape, ShapeRef origFilterShape,
                                         ShapeRef origBiasShape, mlir::ArrayAttr strides, const PadInfo& origPadding,
                                         int64_t groups) {
    auto res = backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides, origPadding);

    const auto inputTileIdx = 0;
    auto& inputTiles = res.tiles[inputTileIdx];

    // For GroupConv, the weights' OC dim is the product of num_group * num_channels_per_group
    const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / groups;

    // To correctly compute input tile when tiling is done over out channels, we need to determine
    // the start group for the tile and the number of groups it spans.
    // Based on them, we can back-infer the necessary input tile.
    // E.g. GroupConv groups = 6; in channels = 12; out channels = 18; filter = (groups * 3 out ch) x 2 in ch
    //      w/ tiling = [1, 3, 1, 1]
    // The resulting tiled GroupConvs are:
    //      Tile 0: GC w/ groups = 2 (group 0 & 1 of orig GC): out channels 0 - 5, in channels 0 - 3
    //      Tile 1: GC w/ groups = 2 (group 2 & 3 of orig GC): out channels 6 - 11, in channels 4 - 7
    //      Tile 2: GC w/ groups = 2 (group 4 & 5 of orig GC): out channels 12 - 17, in channels 8 - 11
    const auto startGroupForTile = outputTile.offsets[Dims4D::Act::C] / numOutChannelsPerGroup;
    const auto numGroupsForTile = divUp(outputTile.shape[Dims4D::Act::C], numOutChannelsPerGroup);

    inputTiles.offsets[Dims4D::Act::C] = startGroupForTile * origFilterShape[Dims4D::Filter::IC];
    inputTiles.shape[Dims4D::Act::C] = numGroupsForTile * origFilterShape[Dims4D::Filter::IC];

    return res;
}

//
// Pooling tiling
//

InputTiling vpux::backInferPoolTile(const TileInfo& outputTile, ShapeRef origInputShape, mlir::ArrayAttr kernel_size,
                                    mlir::ArrayAttr strides, const PadInfo& origPadding) {
    PlaneTile output;
    output.height.begin = outputTile.offsets[Dims4D::Act::H];
    output.height.end = outputTile.offsets[Dims4D::Act::H] + outputTile.shape[Dims4D::Act::H];
    output.width.begin = outputTile.offsets[Dims4D::Act::W];
    output.width.end = outputTile.offsets[Dims4D::Act::W] + outputTile.shape[Dims4D::Act::W];

    const auto kernelY = kernel_size[Dims4D::Kernel::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto kernelX = kernel_size[Dims4D::Kernel::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto strideY = strides[Dims4D::Strides::Y.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto strideX = strides[Dims4D::Strides::X.ind()].cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto solution =
            solutionForOutputTile(output, kernelX, kernelY, strideX, strideY, origInputShape, origPadding);

    TileInfo inputTile(origInputShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];

    inputTile.offsets[Dims4D::Act::H] = solution.inputTile.height.begin;
    inputTile.shape[Dims4D::Act::H] = solution.inputTile.height.length();

    inputTile.offsets[Dims4D::Act::W] = solution.inputTile.width.begin;
    inputTile.shape[Dims4D::Act::W] = solution.inputTile.width.length();

    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::H] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::H);
    }
    if (outputTile.isCompletedTile && outputTile.axis[Dims4D::Act::W] == 1) {
        restorePlanarShapeForInputTile(inputTile, origInputShape, Dims4D::Act::W);
    }

    return TilingInfo{{inputTile}, solution.inputPad};
}

InputTiling vpux::backInferReduceTile(const vpux::TileInfo& outputTile, ShapeRef inShape, mlir::ArrayAttr axesAttr,
                                      bool keepDims) {
    SmallVector<TileInfo> inputTiles;

    const auto axesValue = parseIntArrayAttr<int64_t>(axesAttr);
    const auto tiledOutputAxis = outputTile.axis.raw();
    const auto tiledOutputShape = outputTile.shape.raw();
    const auto tiledOutputOffsets = outputTile.offsets.raw();

    // Adding tiling case when keep dims is false and the axes are reduced from outputShape
    if (keepDims == false) {
        Shape newInput, newAxis, newOffset;
        std::copy(tiledOutputShape.begin(), tiledOutputShape.end(), std::back_inserter(newInput));
        std::copy(tiledOutputAxis.begin(), tiledOutputAxis.end(), std::back_inserter(newAxis));
        std::copy(tiledOutputOffsets.begin(), tiledOutputOffsets.end(), std::back_inserter(newOffset));

        for (auto axesInd : axesValue) {
            // Adjusting the new input based on tiled output
            newInput.insert(newInput.begin() + axesInd, inShape[Dim(axesInd)]);
            newAxis.insert(newAxis.begin() + axesInd, 1);
            newOffset.insert(newOffset.begin() + axesInd, 0);
        }

        TileInfo inTile(newInput, newOffset, newAxis);

        return TilingInfo{{std::move(inTile)}};
    }

    auto inTile = outputTile;
    for (auto axesInd : axesValue) {
        inTile.shape[Dim(axesInd)] = inShape[Dim(axesInd)];
    }

    return TilingInfo{{std::move(inTile)}};
}

namespace {

// Transform the coordinate in the resized tensor to the coordinate in the original tensor.
// It is from Interpolate-4 document at OpenVINO.
// scale = input_shape / output_shape
double inferInCoord(IE::InterpolateCoordMode coordMode, int64_t outCoord, int64_t origInSize, int64_t origOutSize,
                    double scale) {
    double inCoord = 0;
    switch (coordMode) {
    case IE::InterpolateCoordMode::HALF_PIXEL:
        inCoord = scale * (outCoord + 0.5) - 0.5;
        break;
    case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL:
        inCoord = origOutSize == 1 ? 0.0f : scale * (outCoord + 0.5) - 0.5;
        break;
    case IE::InterpolateCoordMode::ASYMMETRIC:
        inCoord = outCoord * scale;
        break;
    case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN:
        inCoord = (outCoord + 0.5) * scale;
        break;
    case IE::InterpolateCoordMode::ALIGN_CORNERS:
        inCoord = origOutSize == 1 ? 0.0 : outCoord * (origInSize - 1.0) / (origOutSize - 1.0);
        break;
    default:
        VPUX_THROW("Doesn't support coordMode: {0}", coordMode);
    }
    return inCoord;
};

// Get the integer input coordinate from the float input coordinate according to the interpolate attributes
int64_t getNearestCoord(IE::InterpolateMode interpolateMode, IE::InterpolateNearestMode nearestMode, double inCoord,
                        double scale, bool roundUp) {
    int64_t nearestDim = 0;
    if (interpolateMode == IE::InterpolateMode::LINEAR || interpolateMode == IE::InterpolateMode::LINEAR_ONNX) {
        nearestDim = roundUp ? std::ceil(inCoord) : std::floor(inCoord);
    } else if (interpolateMode == IE::InterpolateMode::CUBIC) {
        nearestDim = roundUp ? std::floor(inCoord) + 2 : std::floor(inCoord) - 1;
    } else if (interpolateMode == IE::InterpolateMode::NEAREST) {
        switch (nearestMode) {
        case IE::InterpolateNearestMode::ROUND_PREFER_FLOOR:
            if (isDoubleEqual(inCoord, std::floor(inCoord) + 0.5)) {
                nearestDim = std::floor(inCoord);
            } else {
                nearestDim = std::round(inCoord);
            }
            break;
        case IE::InterpolateNearestMode::ROUND_PREFER_CEIL:
            nearestDim = std::round(inCoord);
            break;
        case IE::InterpolateNearestMode::FLOOR:
            nearestDim = std::floor(inCoord);
            break;
        case IE::InterpolateNearestMode::CEIL:
            nearestDim = std::ceil(inCoord);
            break;
        case IE::InterpolateNearestMode::SIMPLE:
            if (scale > 1.0) {
                nearestDim = std::ceil(inCoord);
            } else {
                nearestDim = std::floor(inCoord);
            }
            break;
        default:
            VPUX_THROW("Doesn't support nearestMode: {0}", nearestMode);
        }
    } else {
        VPUX_THROW("Doesn't support interpolateMode: {0}", interpolateMode);
    }

    return nearestDim;
};

SmallVector<int64_t> propagateOffsetForInterpolate(
        ArrayRef<int64_t> axes, ArrayRef<int64_t> offset, ArrayRef<int64_t> initialInputDims,
        ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
        ArrayRef<int64_t> initialOutputOffsets, ArrayRef<int64_t> currentInputDims,
        vpux::IE::InterpolateCalcMode calcMode, vpux::IE::InterpolateMode interpolateMode,
        vpux::IE::InterpolateCoordMode coordMode, vpux::IE::InterpolateNearestMode nearestMode, ArrayRef<int64_t> sizes,
        ArrayRef<double> scales, bool roundUp, SmallVector<int64_t>&& tiledIndices, vpux::Logger log) {
    log.trace("Interp propagate offset: input = {0}", offset);

    SmallVector<int64_t> inferedOffset(offset.begin(), offset.end());
    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        VPUX_THROW_WHEN(sizes.size() != axes.size(),
                        "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                        sizes.size(), axes.size());
        auto sizesIter = sizes.begin();
        for (const auto& i : axes) {
            log.trace("Interp sizes - axis: {0}", i);
            inferedOffset[i] = *sizesIter++;
        }
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        VPUX_THROW_WHEN(scales.size() != axes.size(),
                        "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                        scales.size(), axes.size());
        auto scalesIter = scales.begin();
        for (const auto& i : axes) {
            log.trace("Interp scales - axis: {0}", i);

            if (std::find(tiledIndices.begin(), tiledIndices.end(), i) == tiledIndices.end()) {
                inferedOffset[i] = roundUp ? currentInputDims[i] - 1 : 0;
                scalesIter++;
            } else {
                double inCoord = inferInCoord(coordMode, offset[i] + initialOutputOffsets[i], initialInputDims[i],
                                              initialOutputDims[i], *scalesIter) -
                                 initialInputOffsets[i];
                int64_t inCoordInt = getNearestCoord(interpolateMode, nearestMode, inCoord, *scalesIter, roundUp);

                inferedOffset[i] = std::clamp(inCoordInt, static_cast<int64_t>(0), currentInputDims[i] - 1);
                scalesIter++;
            }
        }
    } else {
        VPUX_THROW("Doesn't support shape_calculation_mode: {0}", calcMode);
    }

    log.trace("Interp propagate offset: output = {0}", inferedOffset);
    return inferedOffset;
}

SmallVector<int64_t> backInferOffsetForInterpolate(
        ArrayRef<int64_t> offset, IE::InterpolateMode interpolateMode, IE::InterpolateCoordMode coordMode,
        IE::InterpolateNearestMode nearestMode, ArrayRef<int64_t> initialInputDims, ArrayRef<int64_t> initialOutputDims,
        ArrayRef<int64_t> initialInputOffsets, ArrayRef<int64_t> initialOutputOffsets,
        ArrayRef<int64_t> currentInputDims, bool roundUp, SmallVector<int64_t>&& tiledIndices, Logger log) {
    SmallVector<int64_t> axes;
    for (auto i : irange(initialInputDims.size())) {
        if (initialInputDims[i] != initialOutputDims[i]) {
            axes.push_back(i);
        }
    }

    // Compute scale-factors based on full I/O resolution ratio
    SmallVector<int64_t> fullOutSize;
    SmallVector<double> backwardScale;
    for (size_t i = 0; i < axes.size(); i++) {
        backwardScale.push_back(static_cast<double>(initialInputDims[axes[i]]) / initialOutputDims[axes[i]]);
        fullOutSize.push_back(initialOutputDims[axes[i]]);
    }

    // TODO: E#36318 how to deal with calc-mode = size if scales missed - recalc them somewhere:
    auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;
    return propagateOffsetForInterpolate(axes, offset, initialInputDims, initialOutputDims, initialInputOffsets,
                                         initialOutputOffsets, currentInputDims, shapeCalcMode, interpolateMode,
                                         coordMode, nearestMode, fullOutSize, backwardScale, roundUp,
                                         std::move(tiledIndices), log);
}
}  // namespace

//
// Interpolate tiling
//

InputTiling vpux::backInferInterpolateTile(const vpux::TileInfo& outputTile, ArrayRef<int64_t> initialInputDims,
                                           ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> initialInputOffsets,
                                           ArrayRef<int64_t> initialOutputOffsets, ArrayRef<int64_t> currentInputDims,
                                           std::optional<ArrayRef<int64_t>> coordinatesDims,
                                           std::optional<ArrayRef<int64_t>> lambdasDims,
                                           vpux::IE::InterpolateMode interpolateMode,
                                           vpux::IE::InterpolateCoordMode coordMode,
                                           vpux::IE::InterpolateNearestMode nearestMode, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Interpolate, output tile: {0}", outputTile);

    auto outputOffsetBegin = to_small_vector(outputTile.offsets);
    SmallVector<int64_t> outputOffsetEnd(outputOffsetBegin.size());
    for (size_t ind = 0; ind < outputOffsetEnd.size(); ind++) {
        outputOffsetEnd[ind] = outputOffsetBegin[ind] + outputTile.shape[Dim(ind)] - 1;
    }

    SmallVector<int64_t> tiledIndices;
    for (auto i : irange(outputTile.axis.size())) {
        if (outputTile.axis[Dim(i)] > 1) {
            tiledIndices.push_back(i);
        }
    }

    auto inferedInputOffsetBegin = backInferOffsetForInterpolate(
            outputOffsetBegin, interpolateMode, coordMode, nearestMode, initialInputDims, initialOutputDims,
            initialInputOffsets, initialOutputOffsets, currentInputDims, false, std::move(tiledIndices), log);
    auto inferedInputOffsetEnd = backInferOffsetForInterpolate(
            outputOffsetEnd, interpolateMode, coordMode, nearestMode, initialInputDims, initialOutputDims,
            initialInputOffsets, initialOutputOffsets, currentInputDims, true, std::move(tiledIndices), log);

    SmallVector<int64_t> inferedInputShape(inferedInputOffsetEnd.size(), 0);
    for (size_t ind = 0; ind < inferedInputOffsetEnd.size(); ind++) {
        inferedInputShape[ind] = inferedInputOffsetEnd[ind] - inferedInputOffsetBegin[ind] + 1;
    }

    TileInfo inputTile(Shape(inferedInputShape), Shape(inferedInputOffsetBegin), outputTile.axis);
    SmallVector<TileInfo> tiles({std::move(inputTile)});
    if (coordinatesDims.has_value()) {
        tiles.emplace_back(Shape(coordinatesDims.value()));
    }
    if (lambdasDims.has_value()) {
        tiles.emplace_back(Shape(lambdasDims.value()));
    }
    return InputTiling{tiles};
}

//
// Gather tiling
//

InputTiling vpux::backInferGatherTile(const vpux::TileInfo& outputTile, const ShapeRef& origInputShape,
                                      const ShapeRef& origIndicesShape, int64_t axisValue, int64_t batchDims,
                                      bool hasAxisTensor, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Gather, output tile: {0}", outputTile);
    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);

    auto inputRank = origInputShape.size();
    auto indicesRank = origIndicesShape.size();

    for (int64_t i = 0; i < static_cast<int64_t>(inputRank); ++i) {
        if (i < axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else if (i == axisValue) {
            continue;
        } else {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i + indicesRank - batchDims - 1)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + indicesRank - batchDims - 1)];
        }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(indicesRank); ++i) {
        if (i < batchDims) {
            indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        } else {
            indicesTile.shape[Dim(i)] = outputTile.shape[Dim(i + axisValue - batchDims)];
            indicesTile.offsets[Dim(i)] = outputTile.offsets[Dim(i + axisValue - batchDims)];
        }
    }

    if (hasAxisTensor) {
        return InputTiling{{std::move(inputTile), std::move(indicesTile), TileInfo(ShapeRef({1}))}};
    }

    return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
}

//
// GatherDMA tiling
//

InputTiling vpux::backInferGatherDMATile(const vpux::TileInfo& outputTile, ShapeRef origInputShape,
                                         ShapeRef origIndicesShape, int64_t axisValue, bool hasAxisTensor,
                                         vpux::Logger log) {
    log.trace("Try to back infer input tiling for Gather-DMA, output tile: {0}", outputTile);
    TileInfo inputTile(origInputShape);
    TileInfo indicesTile(origIndicesShape);

    auto inputRank = origInputShape.size();

    for (int64_t i = 0; i < static_cast<int64_t>(inputRank); ++i) {
        if (i != axisValue) {
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)];
        }
    }

    indicesTile.shape[Dim(axisValue)] = outputTile.shape[Dim(axisValue)];
    indicesTile.offsets[Dim(axisValue)] = outputTile.offsets[Dim(axisValue)];

    if (hasAxisTensor) {
        return InputTiling{{std::move(inputTile), std::move(indicesTile), TileInfo(ShapeRef({1}))}};
    }

    return InputTiling{{std::move(inputTile), std::move(indicesTile)}};
}

//
// Pad tiling
//

InputTiling vpux::backInferPadTile(const vpux::TileInfo& outputTile, const ShapeRef inShape, const ShapeRef outShape,
                                   const ShapeRef origPadsBegin, const ShapeRef origPadsEnd, vpux::Logger log) {
    log.trace("Try to back infer input tiling for Pad, output tile: {0}", outputTile);
    const auto padBegins = origPadsBegin;
    const auto padEnds = origPadsEnd;
    auto curTile = outputTile;

    for (auto ind : irange(inShape.size())) {
        auto idx = Dim(ind);

        if (curTile.axis[idx] == 1) {
            curTile.shape[idx] = inShape[idx];
        } else {
            curTile.shape[idx] = outputTile.shape[idx];
            if (outputTile.offsets[idx] == 0) {
                curTile.shape[idx] -= padBegins[idx];
            }
            if (outputTile.offsets[idx] + outputTile.shape[idx] == outShape[idx]) {
                curTile.shape[idx] -= padEnds[idx];
            }
        }
        VPUX_THROW_UNLESS(curTile.shape[idx] > 0, "Unsupported tile shape : '{0}'. Must be grater than 0.",
                          curTile.shape[idx]);

        if (outputTile.offsets[idx] != 0) {
            curTile.offsets[idx] = outputTile.offsets[idx] - padBegins[idx];
        } else {
            curTile.offsets[idx] = outputTile.offsets[idx];
        }
        curTile.axis[idx] = outputTile.axis[idx];
        VPUX_THROW_UNLESS(curTile.offsets[idx] < inShape[idx],
                          "Tile offset '{0}' must be smaller than input shape '{1}'.", curTile.offsets[idx],
                          inShape[idx]);
    }

    return TilingInfo{curTile};
}

void vpux::updatePadOpAttrsAfterTiling(const ShapeRef outShape, const TileInfo& outTile,
                                       SmallVector<int64_t>& padsBegin, SmallVector<int64_t>& padsEnd) {
    for (auto ind : irange(outShape.size())) {
        if (outTile.axis[Dim(ind)] == 1) {
            continue;
        }
        if (outTile.offsets[Dim(ind)] < padsBegin[ind]) {
            padsBegin[ind] = padsBegin[ind] - outTile.offsets[Dim(ind)];
        } else {
            padsBegin[ind] = 0;
        }
        if (outTile.offsets[Dim(ind)] + outTile.shape[Dim(ind)] != outShape[Dim(ind)]) {
            padsEnd[ind] = 0;
        }
    }
}

//
// DepthToSpace tiling
//

InputTiling vpux::backInferDepthToSpaceTile(const vpux::TileInfo& outputTile, ShapeRef origInputShape,
                                            int64_t blockSize, vpux::Logger) {
    VPUX_THROW_WHEN(blockSize == 0, "BlockSize is zero and used as a divisor");
    VPUX_THROW_WHEN(origInputShape.size() != 4, "Unsupported shape rank: {0}", origInputShape.size());

    TileInfo inputTile(origInputShape);
    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.shape[Dims4D::Act::W] = outputTile.shape[Dims4D::Act::W] / blockSize;
    inputTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H] / blockSize;

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C] * (blockSize * blockSize);
    inputTile.offsets[Dims4D::Act::W] = outputTile.offsets[Dims4D::Act::W] / blockSize;
    inputTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H] / blockSize;

    return InputTiling{inputTile};
}

/// @brief Infer output window size OH X OW from input window size IH X IW
std::optional<std::pair<int64_t, int64_t>> vpux::spatialOutputForInputWindowSize(
        const std::pair<int64_t, int64_t>& inputHW, ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
        const PadInfo& pads) {
    VPUX_THROW_WHEN(kernel.size() != 2, "Expected kernel size to be 2. Got '{0}'", kernel.size());
    const auto KY = kernel[Dims4D::Kernel::Y.ind()];
    const auto KX = kernel[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(strides.size() != 2, "Expected strides size to be 2. Got '{0}'", strides.size());
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    const auto padTop = pads.top;
    const auto padBottom = pads.bottom;
    const auto padLeft = pads.left;
    const auto padRight = pads.right;
    if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
        VPUX_THROW("Invalid pads: top '{0}', bottom '{1}', left '{2}', right '{3}'", padTop, padBottom, padLeft,
                   padRight);
    }

    const auto outputHeight = (inputHW.first - KY + padTop + padBottom) / SY + 1;
    const auto outputWidth = (inputHW.second - KX + padLeft + padRight) / SX + 1;

    if (outputHeight <= 0 || outputWidth <= 0) {
        return std::nullopt;
    }

    return std::make_pair(outputHeight, outputWidth);
}

//
// Tiling utils
//

std::tuple<DimRange, int64_t, int64_t> vpux::inputForOutputDim(const DimRange& output, int64_t kernel, int64_t stride,
                                                               const DimRange& initialInputRange, int64_t padBefore,
                                                               int64_t padAfter) {
    VPUX_THROW_UNLESS(output.length() > 0, "Wrong output tile '{0}'", output);
    VPUX_THROW_UNLESS(initialInputRange.length() > 0, "Wrong initial input range '{0}'", initialInputRange);
    VPUX_THROW_UNLESS(kernel > 0, "Wrong kernel '{0}'", kernel);
    VPUX_THROW_UNLESS(stride > 0, "Wrong stride '{0}'", stride);
    VPUX_THROW_UNLESS(padBefore >= 0, "Wrong padBefore '{0}'", padBefore);
    VPUX_THROW_UNLESS(padAfter >= 0, "Wrong padAfter '{0}'", padAfter);

    DimRange input = {0, 0};
    int64_t before = 0;
    int64_t after = 0;

    input.begin = output.begin * stride - padBefore;

    if (input.begin < initialInputRange.begin) {
        VPUX_THROW_UNLESS(initialInputRange.begin - input.begin <= padBefore,
                          "Input tile '{0}' and padBefore '{1}' doesn't match to initial range '{2}'", input, padBefore,
                          initialInputRange);

        before = std::min(initialInputRange.begin - input.begin, padBefore);
        input.begin = initialInputRange.begin;
    }

    VPUX_THROW_UNLESS(input.begin < initialInputRange.end, "Input tile '{0}' doesn't match to initial range '{1}'",
                      input, initialInputRange);

    input.end = (output.end - 1) * stride + kernel - padBefore;

    if (input.end > initialInputRange.end) {
        VPUX_THROW_UNLESS(input.end - initialInputRange.end <= padAfter,
                          "Input tile '{0}' and padAfter '{1}' doesn't match to initial range '{2}'", input, padAfter,
                          initialInputRange);

        after = std::min(input.end - initialInputRange.end, padAfter);
        input.end = initialInputRange.end;
    }

    VPUX_THROW_UNLESS(input.end > initialInputRange.begin, "Input tile '{0}' doesn't match to initial range '{1}'",
                      input, initialInputRange);
    VPUX_THROW_UNLESS(input.length() > 0, "Input tile '{0}' doesn't match to initial range '{1}'", input,
                      initialInputRange);

    return std::make_tuple(input, before, after);
}

// @brief Following function computes new strides based on the new tensor shape.
// @warning The new shape can be a result of tiling or aligning or something else.
SmallVector<Strides> vpux::adaptStrides(ShapeRef origShape, StridesRef origStrides, ArrayRef<Shape> adaptedShapes,
                                        DimsOrder dimsOrder) {
    auto adaptedStrides = SmallVector<Strides>();
    const auto memShape = dimsOrder.toMemoryOrder(origShape);
    const auto memStrides = dimsOrder.toMemoryOrder(origStrides);

    for (const auto& adaptedShape : adaptedShapes) {
        const auto adaptedMemShape = dimsOrder.toMemoryOrder(Shape(adaptedShape));

        SmallVector<Bit> adaptedMemStrides(memStrides.raw());
        // Automatically adaptedMemStrides.back() is equal to the element type size
        for (int i = static_cast<int>(memStrides.size()) - 2; i >= 0; --i) {
            // Compute the ration between consecutive strides.
            // This tells us how many elements were accounted for in the original
            // strides and by using this, we incrementally construct the new adapted strides.
            const auto currStride = memStrides[MemDim(i)].count();
            const auto prevStride = memStrides[MemDim(i + 1)].count();
            const auto prevAdaptedStride = adaptedMemStrides[i + 1].count();

            auto adaptedStride = prevAdaptedStride * currStride / prevStride;

            if (adaptedStride != (int)adaptedStride) {
                vpux::Logger log("VPUX Adapt Strides Tiling method", vpux::LogLevel::Error);
                log.error("Adapted strides has decimals and may cause problems");
            }

            const auto strideRatio = currStride / prevStride;
            // If there is a change between the original and the new shape,
            // we favor striding with the new shape size instead of the previous stride ratio.
            if (memShape[MemDim(i + 1)] != adaptedMemShape[MemDim(i + 1)]) {
                // In the case of multiclustering, all such scenarios like H|K cluster tiling
                // with H|K prefetch tiling should be concatenated in DDR as simple tensors.

                // Long story to why we don't allow strides and tiling on same axis:
                // Mostly it's unclear how we should handle correctly such a case, because the nature
                // of strides can be very multifaceted, and we don't have explicit knowledge of the
                // scope for that stride.
                //
                // Let's take a case like 24 dimension strided to 32.
                // You may do this to either stride to 32 to fit in a concat over the specific axis
                // Or you may do this for alignment reasons, such that each pixel starts at a 16 byte
                // aligned address.
                //
                // So if we tile 24 by 2, and have 12. How should the strides be adapted?
                // Should we keep them as 32 to satisfy the concat or should we readjust them and align
                // to next value multiple of 16, which will be 16.
                // It's this lack of information and very context dependent reason why we avoid to
                // tackle this case.
                //
                // Without having a solid and functional infrastructure, to do everything in full knowledge
                // of context it can easily lead to a lot of problems and instabilities in the future.

                VPUX_THROW_WHEN(strideRatio != memShape[MemDim(i + 1)],
                                "Can't have both stride ratio '{0}' != shape '{1}' and also adapted shape '{2}' on "
                                "same axis '{3}'.",
                                strideRatio, memShape[MemDim(i + 1)], adaptedMemShape[MemDim(i + 1)], i + 1);
                adaptedStride = adaptedMemShape[MemDim(i + 1)] * prevAdaptedStride;
            }

            adaptedMemStrides[i] = Bit(adaptedStride);
        }
        adaptedStrides.emplace_back(dimsOrder.toLogicalOrder(MemStrides(adaptedMemStrides)));
    }

    return adaptedStrides;
}

DimArr vpux::getTileDimOrderND(MemShape memShape, DimsOrder dimOrder) {
    // Function calculates tile dim order from memory shape and dimOrder
    // It prioritize dim order depending on dim size and dimsOrder
    // Ex: MemShape: 3x80x80x40x80  DimOrder: NCDHW (0x12345)
    //      the return will be {1, 2, 4, 3, 0}
    //            equivalent:  {C, D, W, H, N}
    auto outputMemShape = memShape.raw();
    auto outputSortShape = memShape.raw();
    const auto outputDimOrderVec = dimOrder.toPermutation();

    std::sort(outputSortShape.begin(), outputSortShape.end(), std::greater<int64_t>());

    DimArr returntileDimOrder;

    for (auto it : outputSortShape) {
        if (it > 1) {
            // find the first value that match
            auto dimIt = std::find(outputMemShape.begin(), outputMemShape.end(), it);
            // extract the DimOrder
            returntileDimOrder.push_back(outputDimOrderVec[dimIt - outputMemShape.begin()]);
            // set the value to 0 to avoid geting the same index if more values are equals
            *dimIt = 0;
        }
    }

    return returntileDimOrder;
}

DimArr getTileDimOrderByShape(mlir::Operation* op, Dim filterDimToCompare, Dim actInputDimToCompare) {
    const auto preferTilingOrder = VPU::getSEPConvTilingOrder(op);
    if (preferTilingOrder.has_value()) {
        return preferTilingOrder.value();
    }
    VPUX_THROW_WHEN(op->getOperands().size() < 2,
                    "Only support multi-operand ops to get tile dim order by shape, but got '{0}'",
                    op->getOperands().size());
    const auto activationType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto filterType = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = getShape(op->getResult(0));
    const auto isChannelValid = VPU::doesNCEOpChannelSatisfyWorkload(op, TileInfo(outputShape));
    const auto isFilterLargerToTile =
            (filterType.getShape()[filterDimToCompare] > activationType.getShape()[actInputDimToCompare]) ||
            !isChannelValid;
    return isFilterLargerToTile ? DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}
                                : DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
}

DimArr vpux::getTileDimOrder(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    // Compare the Activation and Filter channels
    // if filter channels > activation channels
    //      First tile at C
    // else tile at H

    auto tileDimOrder =
            llvm::TypeSwitch<mlir::Operation*, DimArr>(op)
                    .Case<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return getTileDimOrderByShape(op, Dims4D::Filter::IC, Dims4D::Act::C);
                    })
                    .Case<VPU::NCEDepthConvolutionOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return getTileDimOrderByShape(op, Dims4D::Filter::OC, Dims4D::Act::H);
                    })
                    .Case<VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputShape = getShape(op->getResult(0));
                        const auto isChannelValid = VPU::doesNCEOpChannelSatisfyWorkload(op, TileInfo(outputShape));
                        if (isChannelValid) {
                            return DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W};
                        } else {
                            return DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                        }
                    })
                    .Case<VPU::MVNOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        auto mvn1 = mlir::dyn_cast<VPU::MVNOp>(op);
                        auto dims = mvn1.getNonNormDims();
                        VPUX_THROW_UNLESS(dims.size(), "Could not find non-norm axes");
                        return dims;
                    })
                    .Case<VPU::MVN6Op>([&](mlir::Operation* op) {
                        auto mvn6 = mlir::dyn_cast<VPU::MVN6Op>(op);
                        auto dims = mvn6.getNonNormDims();
                        VPUX_THROW_UNLESS(dims.size(), "Could not find non-norm axes");
                        return dims;
                    })
                    .Case<VPU::QuantizeOp>([&](mlir::Operation*) {
                        // Not splitting over C, to keep aligned with number of Scales in qType
                        // and so avoid 'validateQuantElemType' fail
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::DequantizeOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::DetectionOutputDecodeBoxesOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::C, Dims4D::Act::H};  // [1, numLocClasses, numPriors, 4]
                    })
                    .Case<VPU::DetectionOutputSortOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::H};  // [1, 1, numClasses, numPriors]
                    })
                    .Case<VPU::DetectionOutputNmsCaffeOp>([&](mlir::Operation*) {
                        return DimArr{Dims4D::Act::H};  // [1, 1, numClasses, topK]
                    })
                    .Case<VPU::NCEEltwiseOp>([&](mlir::Operation* op) {
                        const auto outputShape = getShape(op->getResult(0));
                        return outputShape[Dims4D::Act::C] / VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT <
                                               outputShape[Dims4D::Act::H]
                                       ? DimArr{Dims4D::Act::H, Dims4D::Act::C, Dims4D::Act::W}
                                       : DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                    })
                    .Case<VPU::SoftMaxOp>([&](mlir::Operation* op) {
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                        auto softMaxOp = mlir::cast<VPU::SoftMaxOp>(op);
                        auto axis = softMaxOp.getAxisIndAttr().getValue().getSExtValue();
                        auto dimIt = std::find(tileDimOrder.begin(), tileDimOrder.end(), Dim(axis));
                        if (dimIt != tileDimOrder.end()) {
                            // Tiling along SoftMax operation axis is not supported
                            log.nest(2).trace("Removing axis dim {0} for SoftMax {1}", *dimIt, tileDimOrder);
                            tileDimOrder.erase(dimIt);
                        }
                        return tileDimOrder;
                    })
                    .Case<VPU::DepthToSpaceOp>([&](mlir::Operation* op) {
                        auto origOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(op);
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
                        VPUX_THROW_UNLESS(outputType.getDimsOrder() == DimsOrder::NCHW ||
                                                  outputType.getDimsOrder() == DimsOrder::NHWC,
                                          "DepthToSpace Op only support NCHW and NHWC layout, but got '{0}'",
                                          outputType.getDimsOrder());

                        // It is better to tile DepthToSpace Op at the highest dimension
                        // to avoid stride concat that is inefficient
                        if (origOp.getMode() == IE::DepthToSpaceMode::DEPTH_FIRST) {
                            return outputType.getDimsOrder() == DimsOrder::NHWC
                                           ? SmallVector<Dim>{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C}
                                           : SmallVector<Dim>{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
                        }

                        // It is illegal to tile DepthToSpace Op at channel when it is the BLOCKS_FIRST mode
                        // If that, the output will be a discontinuous memory buffer and will cause accuracy issue
                        if (origOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST) {
                            return SmallVector<Dim>{Dims4D::Act::H, Dims4D::Act::W};
                        }

                        VPUX_THROW("Unknown DepthToSpaceMode. BLOCKS_FIRST and DEPTH_FIRST methods are supported only");
                    })
                    .Case<VPU::NCEPermuteOp>([&](mlir::Operation* op) {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        return DimArr{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
                    })
                    .Default([&](mlir::Operation* op) -> DimArr {
                        log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
                        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

                        return getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
                    });

    // For prefetching mode, only weights can be pre-fetched to the parent op
    if (tilingMode == TilingMode::PREFETCHING) {
        tileDimOrder = SmallVector<Dim>({Dims4D::Act::C});
    }
    return tileDimOrder;
}

bool vpux::isMultiClusterCompatibleForTiling(mlir::Operation* op, const OutputTiling& tiles, Logger log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    VPUX_THROW_WHEN(op == nullptr, "Operation '{0}' doesn't implement ClusteredOpInterface", op->getName());
    if (!clusteredOp->hasAttr(VPU::multiClusterStrategy)) {
        return true;
    }

    // Instead of checking strategy compatible shapes for all tiles, we only check the tiles have unique shapes
    // to reduce compilation time as isStrategyCompatibleShape only cares about tiled shape
    auto tileCandidates = VPU::getUniqueShapeTilingCandidates(op, tiles, log);

    auto isStrategyCompatibleWithTile = [&](const TileInfo& outputTile) {
        return VPU::isStrategyCompatibleShape(clusteredOp, outputTile, clusteredOp.getMultiClusterStrategy().value(),
                                              log);
    };

    return llvm::all_of(tileCandidates, isStrategyCompatibleWithTile);
}

bool canSWLayerBeEvenlyUnrolled(mlir::Operation* op, const OutputTiling& tiles, Dim targetDim, Logger) {
    auto tileOp = IE::getTileExecutor(getModuleOp(op));
    int64_t shaveActCount = 1;
    if (auto shaveActExec = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT)) {
        shaveActCount = shaveActExec.getCount();
    }

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

    std::set<TileInfo, VPU::TileShapeCompare> uniqueShapeTiles(tiles.begin(), tiles.end());

    auto canOutputTiledShapeBeEvenlyDivided = [&](const TileInfo& outputTile) {
        // assume the worst case: ACT SHAVE kernel tiling in VPUIP and SW layer tiling in VPU are performed on the same
        // dimension
        int64_t factor = shaveActCount;

        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
        const auto outputTileShape = outputTileType.getShape();

        if (op->hasAttr(vpux::VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            op->getLoc());

            const auto numClusters =
                    clusteredOp.getOptimalNumClusters(outputTileShape, clusteredOp.getMultiClusterStrategy().value());
            auto outDistributedType = VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters);
            auto dimIdx = VPUIP::getTilingDimIndex(outDistributedType);
            if (dimIdx.has_value() && dimIdx == targetDim.ind()) {
                factor *= numClusters.getInt();
            }
        }

        Shape nTilesOnDim(outputTileShape.size(), 1);
        nTilesOnDim[targetDim] = factor;
        auto tiles = fillDividedTiles(nTilesOnDim, outputTileShape);
        if (mlir::failed(tiles)) {
            return false;
        }

        // Currently, a simple heuristic to decide if it's even unrolling or not is comparing tiled shape size
        // on tiling dimension:
        // First SHAVE's tile has the largest shape size
        // If any other tile's shape size is smaller than half of the first tile's shape size on tiling dimension,
        // it's considered to be unevenly unrolled
        for (auto tile : tiles.value()) {
            if (tile.shape[targetDim] <= tiles.value().front().shape[targetDim] / 2) {
                return false;
            }
        }

        return true;
    };

    return llvm::all_of(uniqueShapeTiles, canOutputTiledShapeBeEvenlyDivided);
}

// Compute the maximum of tile number for each dimension to make sure:
// the tiling numbers are compatible for each dimension
// (Height) DPUs are fully used - at least one line for each DPU
// (Channel) No extra channel alignment - output channel for each cluster should be larger than minChannelSize
SmallVector<int64_t> vpux::getMaxNumTiles(mlir::Operation* op) {
    const auto outputShape = getShape(op->getResult(0));
    auto maxNumTiles = SmallVector<int64_t>(outputShape.begin(), outputShape.end());

    if (!mlir::isa<VPU::MemPermuteOp>(op)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());
    }

    int64_t minChannelSize = 1;
    // NCEPermute operation requires alignment only for width
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported shape rank: {0}", outputShape.size());

        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        auto minWidthSize = VPU::NCEInvariant::getAlignment(outputType.getElementType());
        const auto maxWidthTiles = outputShape[Dims4D::Act::W] / minWidthSize;
        maxNumTiles[Dims4D::Act::W.ind()] = maxWidthTiles;
    } else {
        if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                              "Unsupported shape rank: {0}", outputShape.size());
            minChannelSize = channelsInfo.getOutputChannelAlignment();
        }

        // Consider supported channels for DW ops
        if (auto channelAlignedIface = mlir::dyn_cast<VPU::AlignedWorkloadChannelsOpInterface>(op)) {
            const auto supportedChannels = channelAlignedIface.getSupportedWorkLoadChannels();
            const auto minSupportedChannel = supportedChannels.back();
            if (minChannelSize < minSupportedChannel) {
                minChannelSize = minSupportedChannel;
            }
        }

        const auto channelDim =
                outputShape.size() == DimsGroups5D::Act::numDims ? DimsGroups5D::Act::C : Dims4D::Act::C;
        const auto maxChannelTiles = outputShape[channelDim] / minChannelSize;
        maxNumTiles[channelDim.ind()] = maxChannelTiles;
    }

    if (op->hasAttr(VPU::multiClusterStrategy)) {
        VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                          "Unsupported shape rank: {0}", outputShape.size());

        auto strategy = op->getAttrOfType<VPU::MultiClusterStrategyAttr>(VPU::multiClusterStrategy).getValue();
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto tileOp = IE::getTileExecutor(module);
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            // To make sure the SOH MultiCluster strategy still compatible after tiling,
            // Each cluster should compute at least one output line
            // e.g., 4 cluster compilation, when tiling a layer with output height = 16
            // the tile number for height should be <= 16/4 = 4
            maxNumTiles[Dims4D::Act::H.ind()] = outputShape[Dims4D::Act::H] / tileOp.getCount();
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            // To make sure the SOK MultiCluster strategy still compatible after tiling,
            // each cluster should compute at least minChannelSize(=16) output channels.
            // For SOK, we can use less than the specified number of clusters, to avoid the requirement to align output
            int64_t minNumClustersForSOK = tileOp.getCount();
            while (outputShape[Dims4D::Act::C] % (minChannelSize * minNumClustersForSOK) != 0 &&
                   minNumClustersForSOK > 0) {
                --minNumClustersForSOK;
            }
            maxNumTiles[Dims4D::Act::C.ind()] = outputShape[Dims4D::Act::C] / (minChannelSize * minNumClustersForSOK);
        }
    }

    return maxNumTiles;
}

InputTiling vpux::backInferEltwiseTile(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    SmallVector<TileInfo> inputTiles;
    for (auto& origInput : op->getOpOperands()) {
        const auto curShape = getShape(origInput.get());
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile eltwise operation '{0}' at '{1}', which has operands with different rank",
                          op->getName(), op->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

bool isSupportedTileSizeForHWLayer(mlir::Operation* op, ShapeRef nTilesOnDim, TilingMode tilingMode, Logger log) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    const auto outputShape = getShape(op->getResult(0));

    const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    if (mlir::failed(tiles)) {
        return false;
    }

    return isMultiClusterCompatibleForTiling(op, tiles.value(), log) &&
           tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
}

std::shared_future<bool> checkSupportedTilingAsync(mlir::Operation* op, ShapeRef nTilesOnDim, TilingMode tilingMode,
                                                   llvm::ThreadPool& threadPool, Logger log) {
    return threadPool.async([op, nTilesOnDim, tilingMode, log]() {
        log.trace("[Async] Check tiling {0} for op {1}", nTilesOnDim, op->getLoc());
        return isSupportedTileSizeForHWLayer(op, nTilesOnDim, tilingMode, log);
    });
}

// SWLayer

mlir::FailureOr<OutputTiling> vpux::getSWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                             DimArrRef tileDimOrder, Logger log,
                                                                             ArrayRef<int64_t> maxTilesPerDim) {
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

    SmallVector<int64_t> maxNumTiles(maxTilesPerDim.begin(), maxTilesPerDim.end());
    if (maxTilesPerDim.empty()) {
        maxNumTiles = tilingBuilder.getMaxNumTiles();
    }

    // Step1. get an feasible isolated tiling strategy
    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while ((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim, maxNumTiles, dimToTile))) {
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW_WHEN(tilingMode == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                                op->getLoc());
            }
        }
        ++nTilesOnDim[dimToTile];
    }

    auto isolatedTiles = fillDividedTiles(op, nTilesOnDim, outputShape);

    auto tilingDims = getNonOneDim(nTilesOnDim);
    if (VPUIP::isLegalConvertToDMA(op, log, /*checkCMXSize*/ false) || tilingDims.size() != 1 ||
        canSWLayerBeEvenlyUnrolled(op, isolatedTiles.value(), *tilingDims.begin(), log)) {
        log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
        return isolatedTiles;
    }

    // Step2. continue to increase on the dimension of isolated tiling to get a output shape that can be evenly
    // distributed on ACT SHAVEs
    const auto targetDim = *tilingDims.begin();
    Shape evenUnrollingTilesOnDim = nTilesOnDim;
    while (isDimLeftToTile(evenUnrollingTilesOnDim, maxNumTiles, targetDim) &&
           // Prevent long compilation time caused by excessive tiling
           evenUnrollingTilesOnDim[targetDim] <= MAX_EXCESSIVE_TILING_TIME * nTilesOnDim[targetDim]) {
        ++evenUnrollingTilesOnDim[targetDim];

        auto evenUnrollingTiles = fillDividedTiles(op, evenUnrollingTilesOnDim, outputShape);
        if (mlir::succeeded(evenUnrollingTiles) &&
            canSWLayerBeEvenlyUnrolled(op, evenUnrollingTiles.value(), targetDim, log)) {
            log.trace("Found better isolated tiling strategy: {0} for even unrolling", evenUnrollingTilesOnDim);
            return evenUnrollingTiles;
        }
    }

    log.trace("Fallback to isolated tiling strategy: {0}", nTilesOnDim);
    return isolatedTiles;
}

mlir::FailureOr<OutputTiling> vpux::getSWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log,
                                                             ArrayRef<int64_t> maxTilesPerDim) {
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);
    return getSWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log, maxTilesPerDim);
}

// Compute the maximum of tile number for each dimension with respect of not tile specified axes. No other restriction
// apply, rest of maximum tiles reflect output shape.
SmallVector<int64_t> vpux::getMaxNumTilesWithAxesExclusion(mlir::Operation* op, ArrayRef<int64_t> axes) {
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    SmallVector<int64_t> maxNumTiles(outputShape.begin(), outputShape.end());
    const auto tileDimOrder = getTileDimOrderND(outputType.getMemShape(), outputType.getDimsOrder());
    for (const auto dimVal : tileDimOrder) {
        if (std::find(axes.begin(), axes.end(), dimVal.ind()) != axes.end()) {
            // not tile over axis, not alowed
            maxNumTiles[dimVal.ind()] = 1;
        }
    }
    return maxNumTiles;
}

// HWLayer

mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                             DimArrRef tileDimOrder, Logger log) {
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                      "Unsupported operation '{0}' at '{1}', it has non 4D/5D result", op->getName(), op->getLoc());

    const auto dimAlignInfo = getAlignDimAndSize(op);
    auto dimToAlign = dimAlignInfo.first;
    auto dimAlignment = dimAlignInfo.second;

    Shape nTilesOnDim(outputShape.size(), 1);

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    // Allow uneven tiling over OC, such as OC = 80 can be tiled as three tiles [32, 32, 16]
    const auto isSupportedAlignedDivision = [](int64_t dimSize, int64_t tiles, int64_t alignment) {
        auto base = vpux::divUp(dimSize, tiles);
        auto alignedBase = alignValUp(base, alignment);
        auto remainder = dimSize - alignedBase * (tiles - 1);
        return remainder > 0;
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();

    // In case of pipelining, an isolated tiling strategy is first created
    // Then the tiling number would be increased to get a pipelining tiling strategy
    // If no feasible pipelining tiling could be found, fallback to isolated tiling strategy
    const auto tilingModeToCheck = tilingMode == TilingMode::PIPELINING ? TilingMode::ISOLATED : tilingMode;

    auto dimPlus = [&](Shape& nTilesOnDim, Dim dimToTile) -> void {
        if (dimToTile == dimToAlign && dimAlignment != 1) {
            do {
                ++nTilesOnDim[dimToTile];
            } while (!isSupportedAlignedDivision(outputShape[dimToTile], nTilesOnDim[dimToTile], dimAlignment) &&
                     isDimLeftToTile(nTilesOnDim, maxNumTiles, dimToTile));
        } else {
            ++nTilesOnDim[dimToTile];
        }
        log.nest().trace("dimPlus: nTilesOnDim - {0}", nTilesOnDim);
    };

    auto dimMinus = [&](Shape& nTilesOnDim, Dim dimToTile) -> void {
        --nTilesOnDim[dimToTile];
        // Skip the tiling numbers which are not aligned
        while (dimToTile == dimToAlign && dimAlignment != 1 && nTilesOnDim[dimToTile] > 1 &&
               !isSupportedAlignedDivision(outputShape[dimToTile], nTilesOnDim[dimToTile], dimAlignment)) {
            --nTilesOnDim[dimToTile];
        }
        log.nest().trace("dimMinus: nTilesOnDim - {0}", nTilesOnDim);
    };

    // TODO E#107313: refactor to use the appropriate amount of threads
    auto checkSupportedTilingMultithreaded = [op, dimPlus, log](Shape& nTilesOnDim, Dim dimToTile,
                                                                ArrayRef<int64_t> maxNumTiles,
                                                                TilingMode tilingMode) -> bool {
        constexpr int64_t MAX_CONCURRENT_CHECKERS = 2;
        SmallVector<std::shared_future<bool>> pendingCheckers;
        SmallVector<Shape> tilingCandidates;
        auto currTilesOnDim = nTilesOnDim;

        auto ctx = op->getContext();
        auto& threadPool = ctx->getThreadPool();

        // check supported tiling asynchronously
        for (size_t i = 0; i < MAX_CONCURRENT_CHECKERS; i++) {
            // number of tiles provided by dimPlus() might be larger than maxNumTiles
            // so we only enqueue checker when tiles number is legal
            if (currTilesOnDim[dimToTile] <= maxNumTiles[dimToTile.ind()]) {
                tilingCandidates.push_back(currTilesOnDim);
                pendingCheckers.push_back(
                        checkSupportedTilingAsync(op, tilingCandidates.back(), tilingMode, threadPool, log));
            }

            dimPlus(currTilesOnDim, dimToTile);
        }

        // wait for all checking threads to complete
        for (auto& thread : pendingCheckers) {
            thread.wait();
        }

        // synchronize the check results
        bool isSupported = false;
        for (size_t i = 0; i < pendingCheckers.size(); i++) {
            nTilesOnDim = tilingCandidates[i];
            isSupported = pendingCheckers[i].get();
            if (isSupported) {
                // supported tiling is found, break and return true
                break;
            }
        }

        return isSupported;
    };

    auto checkSupportedTiling = [op, checkSupportedTilingMultithreaded, log](Shape& nTilesOnDim, Dim dimToTile,
                                                                             ArrayRef<int64_t> maxNumTiles,
                                                                             TilingMode tilingMode) -> bool {
        if (op->getContext()->isMultithreadingEnabled()) {
            return checkSupportedTilingMultithreaded(nTilesOnDim, dimToTile, maxNumTiles, tilingMode);
        } else {
            return isSupportedTileSizeForHWLayer(op, nTilesOnDim, tilingMode, log);
        }
    };

    // If input or filter is too big, the operation can't be fit into cmx even all the first dim tiled.
    // So decrease the next dim firstly to make first dim can be tiled exactly.
    auto feasibleNextDim = [&](Shape& nTilesOnDim, Dim dimToTile) -> void {
        if (tileDimOrder.size() <= 1) {
            return;
        }

        auto newTilesOnDim = nTilesOnDim;
        newTilesOnDim[dimToTile] = maxNumTiles[dimToTile.ind()];
        log.trace("feasibleNextDim: maxNumTiles - {0}, newTilesOnDim init value - {1}, nTilesOnDim init value - {2}",
                  maxNumTiles, newTilesOnDim, nTilesOnDim);

        while (newTilesOnDim[dimToTile] > 1) {
            auto tiles = fillDividedTiles(op, newTilesOnDim, outputShape);
            if (!mlir::failed(tiles)) {
                if (isMultiClusterCompatibleForTiling(op, tiles.value(), log)) {
                    break;
                }
            }
            dimMinus(newTilesOnDim, dimToTile);
        }

        auto nextTileDimIter = tileDimIter;
        ++nextTileDimIter;
        while ((nextTileDimIter < tileDimOrder.end()) &&
               (!checkSupportedTiling(newTilesOnDim, *nextTileDimIter, maxNumTiles, tilingModeToCheck))) {
            if (!isDimLeftToTile(newTilesOnDim, maxNumTiles, *nextTileDimIter)) {
                nTilesOnDim[*nextTileDimIter] = newTilesOnDim[*nextTileDimIter];
                ++nextTileDimIter;
                continue;
            }

            dimPlus(newTilesOnDim, *nextTileDimIter);
        }

        if (nextTileDimIter != tileDimOrder.end()) {
            nTilesOnDim[*nextTileDimIter] = newTilesOnDim[*nextTileDimIter];
        }
    };

    // Step1. get an feasible isolated tiling strategy or prefetching strategy
    // feasibleNextDim to firstly fix the second dim with a proper tile number to reduce time
    feasibleNextDim(nTilesOnDim, dimToTile);
    log.trace("feasibleNextDim: final feasible nTilesOnDim - {0}", nTilesOnDim);
    while (!checkSupportedTiling(nTilesOnDim, dimToTile, maxNumTiles, tilingModeToCheck)) {
        // Move to next dim if current dim can not go on
        // TODO: remove or refactor it as below while logic hardly get into
        while ((tileDimIter < tileDimOrder.end()) && (!isDimLeftToTile(nTilesOnDim, maxNumTiles, dimToTile))) {
            // If the current tiling dimension is not supported because of multicluster strategy
            // decrease the current dimension tiling size until the multicluster strategy is compatible again
            auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
            while (nTilesOnDim[dimToTile] > 1) {
                if (!mlir::failed(tiles)) {
                    if (isMultiClusterCompatibleForTiling(op, tiles.value(), log)) {
                        break;
                    }
                }
                dimMinus(nTilesOnDim, dimToTile);
                tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
            }
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                if (tilingModeToCheck == TilingMode::ISOLATED) {
                    log.nest(1).trace("Failed to tile {0} at '{1}'", op->getName(), op->getLoc());
                    return mlir::failure();
                }
                // If still not find the tiling strategy in PREFETCHING, fall back to neutral tiling
                auto neutralTiling = Shape(outputShape.size(), 1);
                log.nest(1).trace("Fallback to neutral tiling while attempting prefetching: {0}", neutralTiling);
                return fillDividedTiles(op, neutralTiling, outputShape);
            }
        }

        // Tile current dim to find a proper number
        dimPlus(nTilesOnDim, dimToTile);
    }

    auto getDimsToTile = [](const Shape& nTilesOnDim) -> SmallVector<Dim> {
        SmallVector<Dim> res;
        for (unsigned i = 0; i < nTilesOnDim.size(); i++) {
            if (nTilesOnDim[Dim(i)] > 1)
                res.emplace_back(Dim(i));
        }
        return res;
    };
    auto dimsToTile = getDimsToTile(nTilesOnDim);
    auto isolatedTiles = fillDividedTiles(op, nTilesOnDim, outputShape);

    if (tilingMode != TilingMode::PIPELINING) {
        // return isolated tiling when getting nested tiles.
        log.nest(1).trace("Return isolated strategy: {0}", nTilesOnDim);
        return isolatedTiles;
    }

    if (dimsToTile.size() > 1) {
        log.nest(1).trace("Fallback to isolated strategy due to nested tiling: {0}", nTilesOnDim);
        return isolatedTiles;
    }

    // Step2. For pipelining, continue to increase on the dimension of isolated tiling
    //        or on the channel dimension in case of neutral tiling to cover cases with large constants
    const auto targetDim = dimsToTile.size() == 0 ? Dims4D::Act::C : dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    log.trace("Attempting to generate tiling strategy for pipelining");
    while (!isSupportedTileSizeForHWLayer(op, prefetchableTilesOnDim, TilingMode::PIPELINING, log)) {
        if (prefetchableTilesOnDim[targetDim] >= MAX_PREFETCH_TILING_TIME * nTilesOnDim[targetDim] ||
            !isDimLeftToTile(prefetchableTilesOnDim, maxNumTiles, targetDim)) {
            log.nest(3).trace("Fallback to isolated strategy: {0}", nTilesOnDim);
            return isolatedTiles;
        }
        if (targetDim == dimToAlign && dimAlignment != 1) {
            do {
                ++prefetchableTilesOnDim[targetDim];
                if (!isDimLeftToTile(prefetchableTilesOnDim, maxNumTiles, targetDim)) {
                    return isolatedTiles;
                }
            } while (!isSupportedAlignedDivision(outputShape[targetDim], prefetchableTilesOnDim[targetDim],
                                                 dimAlignment));
        } else {
            ++prefetchableTilesOnDim[targetDim];
        }
    }

    log.trace("Pipelining tiling strategy: {0}", prefetchableTilesOnDim);
    return fillDividedTiles(op, prefetchableTilesOnDim, outputShape);
}

mlir::FailureOr<OutputTiling> vpux::getHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    log.nest(2).trace("Tile Dim order is {0}", tileDimOrder);
    return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
}

bool vpux::isDimLeftToTile(ShapeRef curNumTiles, ArrayRef<int64_t> maxNumTiles, Dim testTileDim) {
    return curNumTiles[testTileDim] < maxNumTiles[testTileDim.ind()];
}

mlir::FailureOr<OutputTiling> vpux::isSupportedTileSize(mlir::Operation* op, ShapeRef nTilesOnDim,
                                                        TilingMode tilingMode, Logger log) {
    const auto outputShape = getShape(op->getResult(0));
    const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    if (tilingInfo == nullptr) {
        return mlir::failure();
    }

    if (isMultiClusterCompatibleForTiling(op, tiles.value(), log) &&
        tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log)) {
        return tiles;
    }

    return mlir::failure();
}

std::pair<Dim, int64_t> vpux::getAlignDimAndSize(mlir::Operation* op) {
    int64_t dimAlignment = 1;
    auto dimToAlign = Dims4D::Act::C;

    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        dimAlignment = channelsInfo.getOutputChannelAlignment();
    }

    // For NCE Permute operation we must have alignment over width because
    // in following passes a Reorder layer will be added that will generate NWCH order
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        dimToAlign = Dims4D::Act::W;
        dimAlignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    }

    return std::make_pair(dimToAlign, dimAlignment);
}

// Allow uneven tiling over OC, such as OC = 80 can be tiled as three tiles [32, 32, 16]
bool vpux::isSupportedAlignedDivision(int64_t dimSize, int64_t tiles, int64_t alignment) {
    auto base = vpux::divUp(dimSize, tiles);
    auto alignedBase = alignValUp(base, alignment);
    auto remainder = dimSize - alignedBase * (tiles - 1);
    return remainder > 0;
}

SmallVector<Dim> vpux::getNonOneDim(ShapeRef inputShape) {
    SmallVector<Dim> nonOneDims;
    for (auto index : irange(inputShape.size())) {
        if (inputShape[Dim(index)] != 1) {
            nonOneDims.push_back(Dim(index));
        }
    }
    return nonOneDims;
}

mlir::FailureOr<Shape> vpux::getNextTiling(Dim targetDim, Dim dimToAlign, int64_t dimAlignment, Shape nTilesOnDim,
                                           ArrayRef<int64_t> maxNumTiles, ShapeRef outputShape) {
    if (!isDimLeftToTile(nTilesOnDim, maxNumTiles, targetDim)) {
        return mlir::failure();
    }
    if (targetDim == dimToAlign && dimAlignment != 1) {
        do {
            ++nTilesOnDim[targetDim];
            if (!isDimLeftToTile(nTilesOnDim, maxNumTiles, targetDim)) {
                return mlir::failure();
            }
        } while (!isSupportedAlignedDivision(outputShape[targetDim], nTilesOnDim[targetDim], dimAlignment));
    } else {
        ++nTilesOnDim[targetDim];
    }
    return nTilesOnDim;
}

std::optional<Dim> vpux::getMaxNonOneDim(ShapeRef inputShape) {
    Dim maxNonOneDim;
    int64_t maxShape = 0;
    for (auto index : irange(inputShape.size())) {
        if (inputShape[Dim(index)] != 1 && inputShape[Dim(index)] > maxShape) {
            maxNonOneDim = Dim(index);
            maxShape = inputShape[Dim(index)];
        }
    }
    if (maxShape <= 1) {
        return std::nullopt;
    }
    return maxNonOneDim;
}
