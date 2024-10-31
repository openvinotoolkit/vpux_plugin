//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace VPU {

OverlapDistributionParams getExplicitOverlapParamsForSWOpInput(SWOpInterface swOp, ShapeRef outShape,
                                                               ArrayRef<int64_t> numTiles, ArrayRef<int64_t> alignment);

DistributionInfoAttr getSWExplicitDistributionInfoAttr(SWOpInterface swOp, ShapeRef shape,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
                                                       mlir::UnitAttr uniformDistributedSegments,
                                                       const vpux::VPU::OverlapDistributionParams& overlapParams);
DistributionInfoAttr getNCEExplicitDistributionInfoAttr(NCEOpInterface nceOp, ShapeRef shape,
                                                        VPU::DistributionMode distributionMode,
                                                        mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                        mlir::ArrayAttr alignment,
                                                        mlir::UnitAttr uniformDistributedSegments,
                                                        const vpux::VPU::OverlapDistributionParams& overlapParams);
DistributionInfoAttr getConcatExplicitDistributedAttr(ShapeRef shape, VPU::DistributionMode distributionMode,
                                                      mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                      mlir::ArrayAttr alignment,
                                                      mlir::UnitAttr uniformDistributedSegments,
                                                      const vpux::VPU::OverlapDistributionParams& overlapParams,
                                                      mlir::MLIRContext* ctx);
DistributionInfoAttr getConcatExplicitDistributedAttrForNewShape(VPU::DistributionInfoAttr originDistribution,
                                                                 ShapeRef newShape, mlir::MLIRContext* ctx);
DistributionInfoAttr getExplicitDistrAttrForSliceLikeOps(VPU::DistributionInfoAttr distributionWithProperAlignment,
                                                         ArrayRef<int64_t> sliceShape, ArrayRef<int64_t> originShape,
                                                         mlir::MLIRContext* ctx);
DistributionInfoAttr getSegmentedExplicitDistrAttrForSliceLikeOps(VPU::DistributionInfoAttr distributionAttr,
                                                                  ArrayRef<int64_t> sliceOutputShape,
                                                                  mlir::ArrayAttr explicitOutputShapes,
                                                                  mlir::MLIRContext* ctx);
DistributionInfoAttr getNonOverlappedDistributedAttr(ShapeRef shape, VPU::DistributionModeAttr distrModeAttr,
                                                     mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                     mlir::ArrayAttr alignment,
                                                     mlir::UnitAttr uniformDistributedSegments, mlir::MLIRContext* ctx);
NDTypeInterface changeShapeElemTypeForDuplicatedDistributedBuffers(NDTypeInterface buff, ShapeRef shape,
                                                                   mlir::Type elemType);

DistributionInfoAttr getExplicitDistrAttrForSparseData(VPU::DistributionInfoAttr denseDataDistribution,
                                                       ShapeRef dataShape, VPU::SEAttr seAttr, mlir::MLIRContext* ctx);
DistributionInfoAttr getExplicitDistrAttrForSparsityMap(VPU::DistributionInfoAttr denseDataDistribution,
                                                        ShapeRef sparsityMapShape, mlir::UnitAttr isWeights,
                                                        mlir::MLIRContext* ctx);
DistributionInfoAttr getExplicitDistrAttrForSETable(VPU::DistributionInfoAttr denseDataDistribution,
                                                    const size_t seSize, mlir::MLIRContext* ctx);

//
DistributionInfo getSWExplicitDistributionInfo(VPU::SWOpInterface swOp, ShapeRef shape,
                                               VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                               const int64_t numClusters, ArrayRef<int64_t> alignment,
                                               bool uniformDistributedSegments,
                                               const vpux::VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo getNCEExplicitDistributionInfo(VPU::NCEOpInterface nceOp, ShapeRef shape,
                                                     VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                     const int64_t numClusters, ArrayRef<int64_t> alignment,
                                                     bool uniformDistributedSegments,
                                                     const vpux::VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo getConcatExplicitDistributedNative(ShapeRef shape, VPU::DistributionMode distributionMode,
                                                         ArrayRef<int64_t> numTiles, int64_t numClusters,
                                                         ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                         const vpux::VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo getExplicitDistrNativeForSliceLikeOps(
        const VPU::DistributionInfo& distributionWithProperAlignment, ArrayRef<int64_t> sliceShape,
        ArrayRef<int64_t> originShape);

VPU::DistributionInfo getSegmentedExplicitDistrNativeForSliceLikeOps(const VPU::DistributionInfo& distribution,
                                                                     ArrayRef<int64_t> sliceOutputShape,
                                                                     ArrayRef<SmallVector<int64_t>> explicitShapes);

DistributionInfo getNonOverlappedDistributedNative(ShapeRef shape, VPU::DistributionMode distrMode,
                                                   ArrayRef<int64_t> numTiles, int64_t numClusters,
                                                   ArrayRef<int64_t> alignment, bool uniformDistributedSegments);

VPU::DistributionInfo getConcatExplicitDistributedNativeForNewShape(const VPU::DistributionInfo& originDistribution,
                                                                    vpux::ShapeRef newShape);

template <typename T,
          std::enable_if_t<or_<std::is_same<VPU::SparseTensorType, T>, std::is_same<VPUIP::SparseBufferType, T>>::value,
                           bool> = true>
DistributionInfoAttr getExplicitDistrAttrForActualDataFromSparseType(T origType) {
    VPUX_THROW_WHEN(!mlir::isa<VPU::DistributedTypeInterface>(origType),
                    "getExplicitDistrAttrForActualDataFromSparseType: type is not distributed");

    auto ctx = origType.getContext();

    auto getDistribution = [](mlir::Type componentType) -> DistributionInfoAttr {
        if (auto distributedTensor = componentType.dyn_cast<VPU::DistributedTensorType>()) {
            return distributedTensor.getDistribution();
        } else if (auto distributedBuffer = componentType.dyn_cast<VPUIP::DistributedBufferType>()) {
            return distributedBuffer.getDistribution();
        }

        VPUX_THROW("Sparse type's component is not distributed, component type = {0}", componentType);
    };

    auto patchDistributionChannels = [&](mlir::ArrayAttr data, mlir::ArrayAttr seTable) -> mlir::ArrayAttr {
        const auto dataShapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(data);
        auto actualShapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(seTable);

        std::transform(dataShapesOffsetsVec.begin(), dataShapesOffsetsVec.end(), actualShapesOffsetsVec.begin(),
                       actualShapesOffsetsVec.begin(),
                       [](const SmallVector<int64_t>& dataShapesOffsets, SmallVector<int64_t> actualShapesOffsets) {
                           actualShapesOffsets[Dims4D::Act::C.ind()] = dataShapesOffsets[Dims4D::Act::C.ind()];
                           return actualShapesOffsets;
                       });

        return getIntArrayOfArray(ctx, actualShapesOffsetsVec);
    };

    auto seTable = origType.getStorageElementTable();
    auto dataType = origType.getData();
    const auto dataDistribution = getDistribution(dataType);

    VPUX_THROW_WHEN(!isDistributedAttrWithExplicitShapesAndOffsets(dataDistribution),
                    "Distribution for SparseType is not explicit, data distribution = {0}", dataDistribution);

    if (seTable == nullptr) {
        return dataDistribution;
    }

    auto seTableDistribution = getDistribution(seTable);
    mlir::ArrayAttr computeShapes =
            patchDistributionChannels(dataDistribution.getComputeShapes(), seTableDistribution.getComputeShapes());
    mlir::ArrayAttr computeOffsets =
            patchDistributionChannels(dataDistribution.getComputeOffsets(), seTableDistribution.getComputeOffsets());
    mlir::ArrayAttr memoryShapes =
            patchDistributionChannels(dataDistribution.getMemoryShapes(), seTableDistribution.getMemoryShapes());
    mlir::ArrayAttr memoryOffsets =
            patchDistributionChannels(dataDistribution.getMemoryOffsets(), seTableDistribution.getMemoryOffsets());

    return DistributionInfoAttr::get(ctx, seTableDistribution.getMode(), seTableDistribution.getNumTiles(), nullptr,
                                     nullptr, nullptr, seTableDistribution.getNumClusters(),
                                     seTableDistribution.getAlignment(),
                                     seTableDistribution.getUniformDistributedSegments(), computeShapes, computeOffsets,
                                     memoryShapes, memoryOffsets, seTableDistribution.getEqualMemoryAndComputeView());
}

}  // namespace VPU
}  // namespace vpux
