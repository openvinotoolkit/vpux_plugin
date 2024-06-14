//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;

VPU::DistributedTensorAttr VPU::getSWExplicitDistributedTensorAttr(
        VPU::SWOpInterface swOp, ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments) {
    VPUX_THROW_WHEN(swOp == nullptr, "Cannot get SW DistributedTensorAttr, is not a SW op");
    auto ctx = swOp.getContext();
    const auto actTensorDistrModeAttr = VPU::DistributionModeAttr::get(ctx, distributionMode);

    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        const auto outShape = getShape(swOp->getResult(0));
        std::optional<ArrayRef<int64_t>> alignmentValue =
                alignment == nullptr ? std::nullopt
                                     : std::optional<ArrayRef<int64_t>>(parseIntArrayAttr<int64_t>(alignment));

        const auto tiles = fillDividedTiles(Shape(parseIntArrayAttr<int64_t>(numTiles)), outShape, alignmentValue);
        VPUX_THROW_WHEN(mlir::failed(tiles), "Incorrect tiles at {0}", swOp.getLoc());
        const auto outTiles = tiles.value();

        auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilder == nullptr, "Cannot cast op to TilingBuilderOpInterface at {0}", swOp.getLoc());
        SmallVector<InputTiling> inputTiles;
        for (const auto& outTile : outTiles) {
            inputTiles.push_back(tilingBuilder.backInferTileInfo(outTile, Logger::global()));
            VPUX_THROW_UNLESS(inputTiles.back().tiles.size() == 1, "Unexpected input operands size {0}",
                              inputTiles.back().tiles.size());
        }

        SmallVector<SmallVector<int64_t>> inputPerClusterShape;
        SmallVector<SmallVector<int64_t>> inputPerClusterOffset;
        for (auto i : irange(outTiles.size())) {
            inputPerClusterShape.push_back(to_small_vector(inputTiles[i].tiles.front().shape));
            inputPerClusterOffset.push_back(to_small_vector(inputTiles[i].tiles.front().offsets));
        }

        auto perClusterShapeAttr = vpux::getIntArrayOfArray(ctx, inputPerClusterShape);
        auto perClusterOffsetAttr = vpux::getIntArrayOfArray(ctx, inputPerClusterOffset);
        return VPU::DistributedTensorAttr::get(ctx, actTensorDistrModeAttr, numTiles, nullptr, nullptr, nullptr,
                                               numClusters, alignment, uniformDistributedSegments, perClusterShapeAttr,
                                               perClusterOffsetAttr, perClusterShapeAttr, perClusterOffsetAttr,
                                               nullptr);
    }

    return getNonOverlappedDistributedAttr(shape, actTensorDistrModeAttr, numTiles, numClusters, alignment,
                                           uniformDistributedSegments, ctx);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::getNCEExplicitDistributedTensorAttr(
        VPU::NCEOpInterface nceOp, ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    VPUX_THROW_WHEN(nceOp == nullptr, "Cannot get HW DistributedTensorAttr, is not a HW op");
    auto ctx = nceOp.getContext();

    const auto distrModeAttr = DistributionModeAttr::get(ctx, distributionMode);
    if (distributionMode == DistributionMode::OVERLAPPED) {
        VPUX_THROW_WHEN((overlapParams.memoryShapes == nullptr) || (overlapParams.memoryOffsets == nullptr) ||
                                (overlapParams.computeShapes == nullptr) || (overlapParams.computeOffsets == nullptr),
                        "memoryShapes, memoryOffsets, computeShapes, computeOffsets cannot be nullptr.");
        return vpux::VPU::DistributedTensorAttr::get(ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr,
                                                     numClusters, alignment, uniformDistributedSegments,
                                                     overlapParams.computeShapes, overlapParams.computeOffsets,
                                                     overlapParams.memoryShapes, overlapParams.memoryOffsets, nullptr);
    }

    DistributedTensorAttr distributedTensorAttr =
            DistributedTensorAttr::get(ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, alignment,
                                       uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensorAttr);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensorAttr);
    auto perClusterMemoryShapes = vpux::getIntArrayOfArray(ctx, optionalClusterMemoryShapes.value());
    auto perClusterMemoryOffsets =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensorAttr));

    auto perClusterComputeShapes =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapes(shape, distributedTensorAttr));
    auto perClusterComputeOffsets =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapeOffsets(shape, distributedTensorAttr));

    return VPU::DistributedTensorAttr::get(ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           alignment, uniformDistributedSegments, perClusterComputeShapes,
                                           perClusterComputeOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                                           nullptr);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::getConcatExplicitDistributedAttr(
        ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
        mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams, mlir::MLIRContext* ctx) {
    const auto distrModeAttr = DistributionModeAttr::get(ctx, distributionMode);
    if (distributionMode == DistributionMode::OVERLAPPED) {
        VPUX_THROW_WHEN((overlapParams.memoryShapes == nullptr) || (overlapParams.memoryOffsets == nullptr),
                        "memoryShapes and memoryOffsets cannot be nullptr.");
        return vpux::VPU::DistributedTensorAttr::get(ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr,
                                                     numClusters, alignment, uniformDistributedSegments,
                                                     overlapParams.memoryShapes, overlapParams.memoryOffsets,
                                                     overlapParams.memoryShapes, overlapParams.memoryOffsets, nullptr);
    }

    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, alignment, uniformDistributedSegments,
            nullptr, nullptr, nullptr, nullptr, nullptr);

    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensorAttr);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensorAttr);
    auto perClusterMemoryShapes = getIntArrayOfArray(ctx, optionalClusterMemoryShapes.value());
    auto perClusterMemoryOffsets =
            getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensorAttr));

    return VPU::DistributedTensorAttr::get(
            ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, alignment, uniformDistributedSegments,
            perClusterMemoryShapes, perClusterMemoryOffsets, perClusterMemoryShapes, perClusterMemoryOffsets, nullptr);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::getConcatExplicitDistributedAttrForNewShape(
        VPU::DistributedTensorAttr originDistribution, vpux::ShapeRef newShape, mlir::MLIRContext* ctx) {
    // For non-overlapped mode, use already existing methods that compute per cluster shapes/methods
    if (originDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return VPU::getConcatExplicitDistributedAttr(
                newShape, originDistribution.getMode().getValue(), originDistribution.getNumTiles(),
                originDistribution.getNumClusters(), originDistribution.getAlignment(),
                originDistribution.getUniformDistributedSegments(), VPU::OverlapDistributionParams(), ctx);
    }

    const auto numTiles = vpux::parseIntArrayAttr<int64_t>(originDistribution.getNumTiles());
    auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(originDistribution.getMemoryShapes());

    // For overlapped mode, on the clustering dim, the shapes are taken from the initial distribution, while the rest of
    // the dims will take values from the new shape; this works as long as the concat axis != clustering axis, which is
    // a prerequisite of Distributed Concat
    for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
        for (size_t dim = 0; dim < numTiles.size(); dim++) {
            if (numTiles[dim] == 1) {
                memoryShapes[cluster][dim] = newShape[Dim(dim)];
            }
        }
    }

    auto memoryShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes);
    return VPU::DistributedTensorAttr::get(
            ctx, originDistribution.getMode(), originDistribution.getNumTiles(), originDistribution.getKernel(),
            originDistribution.getPads(), originDistribution.getStrides(), originDistribution.getNumClusters(),
            originDistribution.getAlignment(), originDistribution.getUniformDistributedSegments(), memoryShapesAttr,
            originDistribution.getMemoryOffsets(), memoryShapesAttr, originDistribution.getMemoryOffsets(),
            originDistribution.getEqualMemoryAndComputeView());
}

/// @param distributionWithProperAlignment The original alignment may need be updated to get valid perClusterShapesAttr
/// for slice ops.
/// E.g., C=64, T=4, Alignment=16, then perClusterShape is [16, 16, 16, 16]. For sliceShape C = 32,
/// perClusterShape should be [8, 8, 8, 8], thus original alignment must be changed
VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSliceLikeOps(
        VPU::DistributedTensorAttr distributionWithProperAlignment, ArrayRef<int64_t> sliceShape,
        ArrayRef<int64_t> originShape, mlir::MLIRContext* ctx) {
    const auto mode = distributionWithProperAlignment.getMode().getValue();

    // Explicit DistributedAttr can be inferred for Slice in SEGMENTED case or in any case that has full tensor
    // in all cluster (i.e. if mode contains DUPLICATED or SEGMENTED).
    VPUX_THROW_WHEN(
            (mode != VPU::DistributionMode::SEGMENTED) && (mode != VPU::DistributionMode::OVERLAPPED) &&
                    !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
                    !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED),
            "Cannot apply Slice-like Op on input with explicit memory/compute shapes/offsets with DistributionMode {0}",
            distributionWithProperAlignment.getMode());

    // For Overlapped, if slice axis is clustering axis, per cluster shapes/offsets need to be computed taking into
    // consideration Slice/Subview's neighbour ops, which cannot be done with information available here; the calling
    // pass should fill the correct information in this scenario
    VPUX_THROW_WHEN(
            mode == VPU::DistributionMode::OVERLAPPED &&
                    VPU::isSegmentedOverlappedAxisSameAsSliceAxis(distributionWithProperAlignment.getNumTiles(),
                                                                  originShape, sliceShape),
            "Overlapped clustering axis is the same as Slice/Subview axis; cannot infer per cluster shapes/offsets "
            "without compute op information");

    const auto getDistribution = [&](mlir::ArrayAttr perClusterShapesAttr,
                                     mlir::ArrayAttr perClusterOffsetsAttr) -> VPU::DistributedTensorAttr {
        // Slice/SubviewOp is not a "compute" op, so compute shapes/offsets have no reason to be different
        // from memory shapes/offsets
        return VPU::DistributedTensorAttr::get(
                ctx, distributionWithProperAlignment.getMode(), distributionWithProperAlignment.getNumTiles(),
                distributionWithProperAlignment.getKernel(), distributionWithProperAlignment.getPads(),
                distributionWithProperAlignment.getStrides(), distributionWithProperAlignment.getNumClusters(),
                distributionWithProperAlignment.getAlignment(),
                distributionWithProperAlignment.getUniformDistributedSegments(), perClusterShapesAttr,
                perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr,
                distributionWithProperAlignment.getEqualMemoryAndComputeView());
    };

    if (mode == VPU::DistributionMode::OVERLAPPED) {
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(distributionWithProperAlignment.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < originShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != originShape[dim]) {
                    memoryShapes[cluster][dim] = sliceShape[dim];
                }
            }
        }

        return getDistribution(vpux::getIntArrayOfArray(ctx, memoryShapes),
                               distributionWithProperAlignment.getMemoryOffsets());
    }

    const auto memoryShapes = VPU::getPerClusterMemoryShapes(Shape(sliceShape), distributionWithProperAlignment);
    VPUX_THROW_WHEN(
            !memoryShapes.has_value(),
            "Cannot compute memory shapes for the shape of Slice/Subview's output; shape = {0}, distribution ={1}",
            sliceShape, distributionWithProperAlignment);

    auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes.value());
    auto perClusterOffsetsAttr = vpux::getIntArrayOfArray(
            ctx, VPU::getPerClusterMemoryShapeOffsets(Shape(sliceShape), distributionWithProperAlignment));
    auto distribution = getDistribution(perClusterShapesAttr, perClusterOffsetsAttr);
    return distribution;
}

/**
 * @brief  Get Explicit DistAttr by provided explicit shapes. The function is used to get the last slice of a segmented
 distributed type. E.g., C=128, T=6, Alignment=16, then perClusterShape is [32, 32, 16, 16, 16, 16]. For sliceShape
 C=80, with offset=48, perClusterShape should be [24, 24, 8, 8, 8, 8], which is unable to be infered by the original
 dist attr.
 * @param distribution the src distribution of slice like op
 * @param sliceOutputShape The output shape of the slice like op
 * @param explicitShapes The expected output shapes on all clusters
 */
VPU::DistributedTensorAttr vpux::VPU::getSegmentedExplicitDistrAttrForSliceLikeOps(
        VPU::DistributedTensorAttr distribution, ArrayRef<int64_t> sliceOutputShape, mlir::ArrayAttr explicitShapes,
        mlir::MLIRContext* ctx) {
    const auto mode = distribution.getMode().getValue();
    // Explicit DistributedAttr can be inferred in any case that has full tensor
    // in all cluster (i.e. if mode contains DUPLICATED or MULTICASTED).
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED,
                      "Cannot get SEGMENTED explicit distribution for Slice-like op with DistributionMode {0}",
                      distribution.getMode());
    const auto shapes = parseIntArrayOfArrayAttr<int64_t>(explicitShapes);
    auto hasSameDimSize = llvm::all_of(shapes, [&](const auto& shape) {
        return shape.size() == sliceOutputShape.size();
    });
    VPUX_THROW_UNLESS(hasSameDimSize, "Explicit shapes have different dim num: shapes {0}, slice output {1}",
                      explicitShapes, sliceOutputShape);

    int64_t sliceDim = -1;
    for (auto i : irange(sliceOutputShape.size())) {
        auto sliceOnCurrentDim = llvm::all_of(shapes, [&](const auto& shape) {
            return shape[i] != sliceOutputShape[i];
        });
        if (sliceOnCurrentDim) {
            VPUX_THROW_UNLESS(sliceDim == -1, "Only support explicit shapes on single dim");
            auto sumDimSize =
                    std::accumulate(shapes.begin(), shapes.end(), 0, [&](const int64_t sum, const auto& shape) {
                        return sum + shape[i];
                    });
            VPUX_THROW_UNLESS(sumDimSize == sliceOutputShape[i], "explicit shapes {0} don't match shape {1}", shapes,
                              sliceOutputShape);
            sliceDim = i;
        }
    }
    VPUX_THROW_WHEN(sliceDim == -1, "All explicit shapes have same shape. Explicit shapes are not necessary");

    SmallVector<SmallVector<int64_t>> offsets;
    int64_t offsetVal = 0;
    for (auto& shape : shapes) {
        SmallVector<int64_t> offset(sliceOutputShape.size(), 0);
        offset[sliceDim] = offsetVal;
        offsets.emplace_back(std::move(offset));
        offsetVal += shape[sliceDim];
    }
    auto explicitOffsets = getIntArrayOfArray(ctx, offsets);

    // Create DistributedTensorAttr with provided shapes and offsets. Since alignment is unnecessary, remove it from the
    // new attr
    return VPU::DistributedTensorAttr::get(
            ctx, distribution.getMode(), distribution.getNumTiles(), distribution.getKernel(), distribution.getPads(),
            distribution.getStrides(), distribution.getNumClusters(),
            /*alignmentAttr*/ nullptr, distribution.getUniformDistributedSegments(), explicitShapes, explicitOffsets,
            explicitShapes, explicitOffsets, distribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getNonOverlappedDistributedAttr(
        ShapeRef shape, VPU::DistributionModeAttr distrModeAttr, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        mlir::MLIRContext* ctx) {
    VPUX_THROW_WHEN(distrModeAttr.getValue() == VPU::DistributionMode::OVERLAPPED,
                    "getNonOverlappedDistributedAttr: distribution mode is OVERLAPPED");
    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, alignment, uniformDistributedSegments,
            nullptr, nullptr, nullptr, nullptr, nullptr);
    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensorAttr);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensorAttr);
    auto perClusterMemoryShapes = getIntArrayOfArray(ctx, optionalClusterMemoryShapes.value());
    auto perClusterMemoryOffsets =
            getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensorAttr));
    auto perClusterComputeShapes =
            getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapes(shape, distributedTensorAttr));
    auto perClusterComputeOffsets =
            getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapeOffsets(shape, distributedTensorAttr));

    return VPU::DistributedTensorAttr::get(ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           alignment, uniformDistributedSegments, perClusterComputeShapes,
                                           perClusterComputeOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                                           nullptr);
}

NDTypeInterface vpux::VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(NDTypeInterface buff, ShapeRef shape,
                                                                              mlir::Type elemType) {
    auto distributedBuff = mlir::dyn_cast<VPUIP::DistributedBufferType>(buff);
    VPUX_THROW_WHEN(distributedBuff == nullptr,
                    "changeShapeElemTypeForNonOverlappedDistributedBuffers: buff is not DistributedBufferType = {0}",
                    buff);

    auto distribution = distributedBuff.getDistribution();
    VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "DistributedBuffer has mode different from DUPLICATED after unrolling");
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedBuff.getDistribution())) {
        auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                shape, distribution.getMode(), nullptr, distribution.getNumClusters(), nullptr,
                distribution.getUniformDistributedSegments(), distributedBuff.getContext());
        return distributedBuff.changeShapeElemTypeForExplicitDistribution(shape, elemType, newDistribution);
    }

    return distributedBuff.changeShapeElemType(shape, elemType);
};

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSparseData(
        VPU::DistributedTensorAttr denseDataDistribution, ShapeRef dataShape, VPU::SEAttr seAttr,
        mlir::MLIRContext* ctx) {
    if (seAttr == nullptr) {
        return denseDataDistribution;
    }

    SmallVector<int64_t> seAttrOffsets(dataShape.size(), 0);
    if (auto tileInfo = seAttr.getTileInfo(); tileInfo.has_value() && tileInfo->offsets != nullptr) {
        seAttrOffsets = parseIntArrayAttr<int64_t>(tileInfo->offsets);
    }

    auto getDataShapesOffsets =
            [&](mlir::ArrayAttr denseDataShapesAttr,
                mlir::ArrayAttr denseDataOffsetsAttr) -> std::pair<mlir::ArrayAttr, mlir::ArrayAttr> {
        const auto denseDataShapes = parseIntArrayOfArrayAttr<int64_t>(denseDataShapesAttr);
        const auto denseDataOffsets = parseIntArrayOfArrayAttr<int64_t>(denseDataOffsetsAttr);
        const auto clusterNum = denseDataShapes.size();
        auto dataShapesVec = SmallVector<Shape>(clusterNum, Shape(denseDataShapes[0]));
        auto dataOffsetsVec = SmallVector<Shape>(clusterNum, Shape(denseDataOffsets[0]));

        for (size_t clusterIdx = 0; clusterIdx < clusterNum; ++clusterIdx) {
            const auto denseDataShape = Shape(denseDataShapes[clusterIdx]);
            const auto denseDataOffset = Shape(denseDataOffsets[clusterIdx]);

            Shape startOffsets(denseDataShape.size());
            std::transform(denseDataOffset.begin(), denseDataOffset.end(), seAttrOffsets.begin(), startOffsets.begin(),
                           [](const int64_t dataOffset, const int64_t seOffset) {
                               return dataOffset + seOffset;
                           });

            seAttr.extractTile(startOffsets, denseDataShape, dataShape, dataOffsetsVec[clusterIdx],
                               dataShapesVec[clusterIdx]);
        }

        return {getIntArrayOfArray(ctx, dataShapesVec), getIntArrayOfArray(ctx, dataOffsetsVec)};
    };

    const auto computeView =
            getDataShapesOffsets(denseDataDistribution.getComputeShapes(), denseDataDistribution.getComputeOffsets());
    const auto memoryView =
            getDataShapesOffsets(denseDataDistribution.getMemoryShapes(), denseDataDistribution.getMemoryOffsets());

    return VPU::DistributedTensorAttr::get(ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(),
                                           nullptr, nullptr, nullptr, denseDataDistribution.getNumClusters(),
                                           /*alignment*/ nullptr, denseDataDistribution.getUniformDistributedSegments(),
                                           computeView.first, computeView.second, memoryView.first, memoryView.second,
                                           denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSparsityMap(
        VPU::DistributedTensorAttr denseDataDistribution, ShapeRef sparsityMapShape, mlir::UnitAttr isWeights,
        mlir::MLIRContext* ctx) {
    if (isWeights == nullptr) {
        return denseDataDistribution;
    }

    auto isValidDistributionForWeights = [&]() -> bool {
        if (denseDataDistribution.getNumTiles() == nullptr) {
            return true;
        }

        const auto numTiles = parseIntArrayAttr<int64_t>(denseDataDistribution.getNumTiles());
        if (numTiles.size() == 4 && numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] == 1 &&
            numTiles[Dims4D::Act::W.ind()] == 1) {
            return true;
        }

        return false;
    };

    VPUX_THROW_WHEN(!isValidDistributionForWeights(),
                    "Weights should be segmented only over OC dim, distributed attr = {0}", denseDataDistribution);

    auto getWeightsShapes = [&](mlir::ArrayAttr shapesAttr) -> mlir::ArrayAttr {
        auto shapesVec = parseIntArrayOfArrayAttr<int64_t>(shapesAttr);

        for (auto& shapes : shapesVec) {
            shapes[Dims4D::Filter::IC.ind()] = sparsityMapShape[Dims4D::Filter::IC];
            shapes[Dims4D::Filter::KY.ind()] = sparsityMapShape[Dims4D::Filter::KY];
            shapes[Dims4D::Filter::KX.ind()] = sparsityMapShape[Dims4D::Filter::KX];
        }

        return getIntArrayOfArray(ctx, shapesVec);
    };

    return VPU::DistributedTensorAttr::get(
            ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(), nullptr, nullptr, nullptr,
            denseDataDistribution.getNumClusters(), denseDataDistribution.getAlignment(),
            denseDataDistribution.getUniformDistributedSegments(),
            getWeightsShapes(denseDataDistribution.getComputeShapes()), denseDataDistribution.getComputeOffsets(),
            getWeightsShapes(denseDataDistribution.getMemoryShapes()), denseDataDistribution.getMemoryOffsets(),
            denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSETable(VPU::DistributedTensorAttr denseDataDistribution,
                                                                     const size_t seSize, mlir::MLIRContext* ctx) {
    auto getSETableShapesOffsets = [&](mlir::ArrayAttr shapesOffsetsAttr,
                                       const bool isOffset = false) -> mlir::ArrayAttr {
        auto shapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(shapesOffsetsAttr);

        for (auto& shapesOffsets : shapesOffsetsVec) {
            // In cases where tensor is SEGMENTED over C, SETable depth per cluster must be adjusted
            shapesOffsets[Dims4D::Act::C.ind()] =
                    isOffset ? shapesOffsets[Dims4D::Act::C.ind()] / static_cast<int64_t>(seSize)
                             : divUp(shapesOffsets[Dims4D::Act::C.ind()], static_cast<int64_t>(seSize));
        }

        return getIntArrayOfArray(ctx, shapesOffsetsVec);
    };

    auto seTableAlignmentAttr = denseDataDistribution.getAlignment();
    if (seTableAlignmentAttr != nullptr) {
        auto seTableAlignment = parseIntArrayAttr<int64_t>(seTableAlignmentAttr);
        seTableAlignment[Dims4D::Act::C.ind()] = 1;
        seTableAlignmentAttr = getIntArrayAttr(ctx, seTableAlignment);
    }

    return VPU::DistributedTensorAttr::get(ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(),
                                           nullptr, nullptr, nullptr, denseDataDistribution.getNumClusters(),
                                           seTableAlignmentAttr, denseDataDistribution.getUniformDistributedSegments(),
                                           getSETableShapesOffsets(denseDataDistribution.getComputeShapes()),
                                           getSETableShapesOffsets(denseDataDistribution.getComputeOffsets(), true),
                                           getSETableShapesOffsets(denseDataDistribution.getMemoryShapes()),
                                           getSETableShapesOffsets(denseDataDistribution.getMemoryOffsets(), true),
                                           denseDataDistribution.getEqualMemoryAndComputeView());
}
