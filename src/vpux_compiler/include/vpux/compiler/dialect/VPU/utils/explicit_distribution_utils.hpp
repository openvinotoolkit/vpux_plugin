//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace VPU {

DistributedTensorAttr getSWExplicitDistributedTensorAttr(SWOpInterface swOp, ShapeRef shape,
                                                         DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                         mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
                                                         mlir::UnitAttr uniformDistributedSegments);
DistributedTensorAttr getNCEExplicitDistributedTensorAttr(NCEOpInterface nceOp, ShapeRef shape,
                                                          VPU::DistributionMode distributionMode,
                                                          mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                          mlir::ArrayAttr alignment,
                                                          mlir::UnitAttr uniformDistributedSegments,
                                                          const vpux::VPU::OverlapDistributionParams& overlapParams);
DistributedTensorAttr getConcatExplicitDistributedAttr(ShapeRef shape, VPU::DistributionMode distributionMode,
                                                       mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                       mlir::ArrayAttr alignment,
                                                       mlir::UnitAttr uniformDistributedSegments,
                                                       const vpux::VPU::OverlapDistributionParams& overlapParams,
                                                       mlir::MLIRContext* ctx);
DistributedTensorAttr getConcatExplicitDistributedAttrForNewShape(VPU::DistributedTensorAttr originDistribution,
                                                                  ShapeRef newShape, mlir::MLIRContext* ctx);
DistributedTensorAttr getExplicitDistrAttrForSliceLikeOps(VPU::DistributedTensorAttr distributionWithProperAlignment,
                                                          ArrayRef<int64_t> sliceShape, ArrayRef<int64_t> originShape,
                                                          mlir::MLIRContext* ctx);
DistributedTensorAttr getSegmentedExplicitDistrAttrForSliceLikeOps(VPU::DistributedTensorAttr distribution,
                                                                   ArrayRef<int64_t> sliceOutputShape,
                                                                   mlir::ArrayAttr explicitOutputShapes,
                                                                   mlir::MLIRContext* ctx);
DistributedTensorAttr getNonOverlappedDistributedAttr(ShapeRef shape, VPU::DistributionModeAttr distrModeAttr,
                                                      mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                      mlir::ArrayAttr alignment,
                                                      mlir::UnitAttr uniformDistributedSegments,
                                                      mlir::MLIRContext* ctx);
NDTypeInterface changeShapeElemTypeForDuplicatedDistributedBuffers(NDTypeInterface buff, ShapeRef shape,
                                                                   mlir::Type elemType);

DistributedTensorAttr getExplicitDistrAttrForSparseData(VPU::DistributedTensorAttr denseDataDistribution,
                                                        ShapeRef dataShape, VPU::SEAttr seAttr, mlir::MLIRContext* ctx);
DistributedTensorAttr getExplicitDistrAttrForSparsityMap(VPU::DistributedTensorAttr denseDataDistribution,
                                                         ShapeRef sparsityMapShape, mlir::UnitAttr isWeights,
                                                         mlir::MLIRContext* ctx);
DistributedTensorAttr getExplicitDistrAttrForSETable(VPU::DistributedTensorAttr denseDataDistribution,
                                                     const size_t seSize, mlir::MLIRContext* ctx);

template <typename T,
          std::enable_if_t<or_<std::is_same<VPU::SparseTensorType, T>, std::is_same<VPUIP::SparseBufferType, T>>::value,
                           bool> = true>
DistributedTensorAttr getExplicitDistrAttrForActualDataFromSparseType(T origType) {
    VPUX_THROW_WHEN(!mlir::isa<VPU::DistributedTypeInterface>(origType),
                    "getExplicitDistrAttrForActualDataFromSparseType: type is not distributed");

    auto ctx = origType.getContext();

    auto getDistribution = [](mlir::Type componentType) -> DistributedTensorAttr {
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

    return DistributedTensorAttr::get(
            ctx, seTableDistribution.getMode(), seTableDistribution.getNumTiles(), nullptr, nullptr, nullptr,
            seTableDistribution.getNumClusters(), seTableDistribution.getAlignment(),
            seTableDistribution.getUniformDistributedSegments(), computeShapes, computeOffsets, memoryShapes,
            memoryOffsets, seTableDistribution.getEqualMemoryAndComputeView());
}

}  // namespace VPU
}  // namespace vpux
