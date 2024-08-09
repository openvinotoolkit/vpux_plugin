//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distributed_tensor_native.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux {
namespace VPU {
vpux::VPU::DistributedTensorNative vpux::VPU::DistributedTensorNative::getClassFromAttr(
        vpux::VPU::DistributedTensorAttr distributionAttr) {
    if (distributionAttr == nullptr) {
        return {};
    }

    auto mode = distributionAttr.getMode().getValue();
    auto numClusters = distributionAttr.getNumClusters().getInt();

    auto numTiles = distributionAttr.getNumTiles() ? parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles())
                                                   : SmallVector<int64_t>{};
    auto kernel = distributionAttr.getKernel() ? parseIntArrayAttr<int64_t>(distributionAttr.getKernel())
                                               : SmallVector<int64_t>{};
    auto strides = distributionAttr.getStrides() ? parseIntArrayAttr<int64_t>(distributionAttr.getStrides())
                                                 : SmallVector<int64_t>{};
    auto pad = distributionAttr.getPads() ? vpux::VPU::Padding::getClassFromAttr(distributionAttr.getPads())
                                          : std::optional<vpux::VPU::Padding>(std::nullopt);
    auto alignment = distributionAttr.getAlignment() ? parseIntArrayAttr<int64_t>(distributionAttr.getAlignment())
                                                     : SmallVector<int64_t>{};
    auto uniformDistributedSegments = distributionAttr.getUniformDistributedSegments() ? true : false;
    auto computeShapes = distributionAttr.getComputeShapes()
                                 ? parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getComputeShapes())
                                 : SmallVector<SmallVector<int64_t>>{};
    auto computeOffsets = distributionAttr.getComputeOffsets()
                                  ? parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getComputeOffsets())
                                  : SmallVector<SmallVector<int64_t>>{};
    auto memoryShapes = distributionAttr.getMemoryShapes()
                                ? parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getMemoryShapes())
                                : SmallVector<SmallVector<int64_t>>{};
    auto memoryOffsets = distributionAttr.getMemoryOffsets()
                                 ? parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getMemoryOffsets())
                                 : SmallVector<SmallVector<int64_t>>{};
    auto equalMemoryAndComputeView = distributionAttr.getEqualMemoryAndComputeView() ? true : false;

    return vpux::VPU::DistributedTensorNative(mode, numTiles, kernel, strides, pad, numClusters, alignment,
                                              uniformDistributedSegments, computeShapes, computeOffsets, memoryShapes,
                                              memoryOffsets, equalMemoryAndComputeView);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::DistributedTensorNative::getAttrFromClass(
        mlir::MLIRContext* ctx, const vpux::VPU::DistributedTensorNative& distribution) {
    auto modeAttr = vpux::VPU::DistributionModeAttr::get(ctx, distribution.getDistributionMode());
    auto numClustersAttr = vpux::getIntAttr(ctx, distribution.getNumClusters());
    auto padAttr = distribution.getPadding().has_value()
                           ? vpux::VPU::Padding::getAttrFromClass(ctx, distribution.getPadding().value())
                           : nullptr;

    mlir::ArrayAttr numTilesAttr =
            distribution.getNumTiles().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getNumTiles());
    mlir::ArrayAttr kernelAttr =
            distribution.getKernel().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getKernel());
    mlir::ArrayAttr stridesAttr =
            distribution.getStrides().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getStrides());
    mlir::ArrayAttr alignmentAttr =
            distribution.getAlignment().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getAlignment());
    mlir::UnitAttr uniformDistributedSegmentsAttr =
            distribution.hasUniformDistributedSegments() ? mlir::UnitAttr::get(ctx) : nullptr;
    mlir::ArrayAttr computeShapesAttr = distribution.getComputeShapes().empty()
                                                ? nullptr
                                                : vpux::getIntArrayOfArray(ctx, distribution.getComputeShapes());
    mlir::ArrayAttr computeOffsetsAttr = distribution.getComputeOffsets().empty()
                                                 ? nullptr
                                                 : vpux::getIntArrayOfArray(ctx, distribution.getComputeOffsets());
    mlir::ArrayAttr memoryShapesAttr = distribution.getMemoryShapes().empty()
                                               ? nullptr
                                               : vpux::getIntArrayOfArray(ctx, distribution.getMemoryShapes());
    mlir::ArrayAttr memoryOffsetsAttr = distribution.getMemoryOffsets().empty()
                                                ? nullptr
                                                : vpux::getIntArrayOfArray(ctx, distribution.getMemoryOffsets());
    mlir::UnitAttr equalMemoryAndComputeViewAttr =
            distribution.hasEqualMemoryAndComputeView() ? mlir::UnitAttr::get(ctx) : nullptr;

    return vpux::VPU::DistributedTensorAttr::get(ctx, modeAttr, numTilesAttr, kernelAttr, padAttr, stridesAttr,
                                                 numClustersAttr, alignmentAttr, uniformDistributedSegmentsAttr,
                                                 computeShapesAttr, computeOffsetsAttr, memoryShapesAttr,
                                                 memoryOffsetsAttr, equalMemoryAndComputeViewAttr);
}
}  // namespace VPU
}  // namespace vpux
