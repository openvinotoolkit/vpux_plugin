//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distributed_tensor_native.hpp"
#include "vpux/utils/core/small_vector.hpp"

using MLIR_DistributedTensorCpp = MLIR_UnitBase;
using namespace vpux;

TEST_F(MLIR_DistributedTensorCpp, PaddingTest) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    int64_t top = 1;
    int64_t bottom = 2;
    int64_t left = 3;
    int64_t right = 4;

    auto topAttr = getIntAttr(&ctx, top);
    auto bottomAttr = getIntAttr(&ctx, bottom);
    auto leftAttr = getIntAttr(&ctx, left);
    auto rightAttr = getIntAttr(&ctx, right);

    auto paddingAttr = VPU::PaddingAttr::get(&ctx, leftAttr, rightAttr, topAttr, bottomAttr);

    auto padClass = VPU::Padding(left, right, top, bottom);
    auto attrFromClass = VPU::Padding::getAttrFromClass(&ctx, padClass);
    EXPECT_EQ(paddingAttr, attrFromClass);

    auto padFromAttr = VPU::Padding::getClassFromAttr(paddingAttr);
    EXPECT_EQ(padClass, padFromAttr);
}

TEST_F(MLIR_DistributedTensorCpp, DistributedTensorNativeTest) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    // common values
    auto distributionMode = VPU::DistributionMode::NONE;
    auto numTiles = SmallVector<int64_t>{1, 1, 1, 1};
    auto kernel = SmallVector<int64_t>{1, 1, 1, 2};
    auto pad = std::optional<VPU::Padding>(VPU::Padding(1, 2, 3, 4));
    auto strides = SmallVector<int64_t>{1, 1, 1, 3};
    int64_t numClusters = 0;
    auto alignment = SmallVector<int64_t>{1, 1, 1, 4};
    bool uniformDistributedSegments = true;
    auto computeShapes = SmallVector<SmallVector<int64_t>>{{1, 2}, {1, 2}, {1, 2}, {1, 5}};
    auto computeOffsets = SmallVector<SmallVector<int64_t>>{{2, 2}, {2, 2}, {2, 2}, {2, 5}};
    auto memoryShapes = SmallVector<SmallVector<int64_t>>{{3, 2}, {3, 2}, {3, 2}, {3, 5}};
    auto memoryOffsets = SmallVector<SmallVector<int64_t>>{{4, 2}, {4, 2}, {4, 2}, {4, 5}};
    bool equalMemoryAndComputeView = false;

    // Attr
    auto distributionModeAttr = VPU::DistributionModeAttr::get(&ctx, distributionMode);
    auto numClustersAttr = getIntAttr(&ctx, numClusters);
    mlir::ArrayAttr numTilesAttr = numTiles.empty() ? nullptr : getIntArrayAttr(&ctx, numTiles);
    mlir::ArrayAttr kernelAttr = kernel.empty() ? nullptr : getIntArrayAttr(&ctx, kernel);
    mlir::ArrayAttr stridesAttr = strides.empty() ? nullptr : getIntArrayAttr(&ctx, strides);
    mlir::ArrayAttr alignmentAttr = alignment.empty() ? nullptr : getIntArrayAttr(&ctx, alignment);
    mlir::UnitAttr uniformDistributedSegmentsAttr = uniformDistributedSegments ? mlir::UnitAttr::get(&ctx) : nullptr;
    mlir::ArrayAttr computeShapesAttr = computeShapes.empty() ? nullptr : getIntArrayOfArray(&ctx, computeShapes);
    mlir::ArrayAttr computeOffsetsAttr = computeOffsets.empty() ? nullptr : getIntArrayOfArray(&ctx, computeOffsets);
    mlir::ArrayAttr memoryShapesAttr = memoryShapes.empty() ? nullptr : getIntArrayOfArray(&ctx, memoryShapes);
    mlir::ArrayAttr memoryOffsetsAttr = memoryOffsets.empty() ? nullptr : getIntArrayOfArray(&ctx, memoryOffsets);
    mlir::UnitAttr equalMemoryAndComputeViewAttr = equalMemoryAndComputeView ? mlir::UnitAttr::get(&ctx) : nullptr;

    auto padAttr = pad.has_value() ? VPU::Padding::getAttrFromClass(&ctx, pad.value()) : nullptr;

    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            &ctx, distributionModeAttr, numTilesAttr, kernelAttr, padAttr, stridesAttr, numClustersAttr, alignmentAttr,
            uniformDistributedSegmentsAttr, computeShapesAttr, computeOffsetsAttr, memoryShapesAttr, memoryOffsetsAttr,
            equalMemoryAndComputeViewAttr);

    // CPP class
    auto distributedTensorStruct = VPU::DistributedTensorNative(
            distributionMode, numTiles, kernel, strides, pad, numClusters, alignment, uniformDistributedSegments,
            computeShapes, computeOffsets, memoryShapes, memoryOffsets, equalMemoryAndComputeView);

    auto attrFromClass = VPU::DistributedTensorNative::getAttrFromClass(&ctx, distributedTensorStruct);
    EXPECT_EQ(distributedTensorAttr, attrFromClass);

    auto fromAttr = VPU::DistributedTensorNative::getClassFromAttr(attrFromClass);
    EXPECT_EQ(fromAttr, distributedTensorStruct);
}
