//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;
using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;

using MLIR_DistributedTypesIfMethodsForExplicitDistribution = MLIR_UnitBase;

namespace {
constexpr vpux::StringRef CMX_NAME = "CMX_NN";

void compareExplicitDistributedAttrs(VPU::DistributedTypeInterface distributedTypes,
                                     ArrayRef<VPU::DistributionInfoAttr> expectedDistributions) {
    ASSERT_NE(distributedTypes, nullptr);
    EXPECT_EQ(distributedTypes.getDistributedTypes().size(), expectedDistributions.size());

    for (const auto& p : zip(distributedTypes.getDistributedTypes(), expectedDistributions)) {
        const auto type = std::get<0>(p);
        const auto expectedDistribution = std::get<1>(p);

        const bool isDistributedReq = mlir::isa<VPU::DistributedTensorType, VPUIP::DistributedBufferType>(type);
        EXPECT_TRUE(isDistributedReq);

        if (auto distributedBuff = type.dyn_cast<VPUIP::DistributedBufferType>()) {
            EXPECT_EQ(distributedBuff.getDistribution(), expectedDistribution);
        } else if (auto distributedTensor = type.dyn_cast<VPU::DistributedTensorType>()) {
            EXPECT_EQ(distributedTensor.getDistribution(), expectedDistribution);
        }
    }
}
}  // namespace

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, DistributedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 64, 8, 16}, SmallVector<int64_t>{1, 64, 6, 16}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto distributedTypeIf =
            VPUIP::DistributedBufferType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr)
                    .dyn_cast<VPU::DistributedTypeInterface>();

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 8});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 8}, SmallVector<int64_t>{1, 32, 6, 8}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .extractViewTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                ShapeRef(tileElemStrides), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }
}

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, DistributedTensorType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 64, 8, 16}, SmallVector<int64_t>{1, 64, 6, 16}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto distributedTypeIf =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr)
                    .dyn_cast<VPU::DistributedTypeInterface>();

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPU::DistributedTensorType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPU::DistributedTensorType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 8});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 8}, SmallVector<int64_t>{1, 32, 6, 8}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPU::DistributedTensorType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPU::DistributedTensorType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        EXPECT_ANY_THROW(distributedTypeIf.extractViewTileForExplicitDistribution(
                ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides), distributedAttrForNewShape));
    }
}

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, DistributedTensorType5D) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = SmallVector<int64_t>({3, 1, 64, 13, 16});
    const auto elemType = mlir::Float16Type::get(&ctx);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::GNHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 1, 1, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 1, 64, 13, 16}, SmallVector<int64_t>{2, 1, 64, 13, 16}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0, 0}, SmallVector<int64_t>{1, 0, 0, 0, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto distributedTypeIf =
            VPU::DistributedTensorType::get(&ctx, shape, elemType, dimsOrder, dimsSpace, distributedAttr)
                    .dyn_cast<VPU::DistributedTypeInterface>();

    {
        const auto newShape = SmallVector<int64_t>({2, 1, 64, 13, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 1, 64, 13, 16}, SmallVector<int64_t>{1, 1, 64, 13, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0, 0}, SmallVector<int64_t>{1, 0, 0, 0, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        ASSERT_NE(newType, nullptr);

        const auto newDistribution =
                newType.getDistributedTypes().front().cast<VPU::DistributedTensorType>().getDistribution();
        EXPECT_EQ(newDistribution, distributedAttrForNewShape);
    }
}

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, SparseBufferTypeDataAndSparsityMap) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto dataElemType = mlir::Float16Type::get(&ctx);
    const auto sparsityMapElemType = mlir::IntegerType::get(&ctx, 1);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 64, 8, 16}, SmallVector<int64_t>{1, 64, 6, 16}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto data =
            VPUIP::DistributedBufferType::get(&ctx, shape, dataElemType, dimsOrder, dimsSpace, distributedAttr);
    const auto sparsityMap =
            VPUIP::DistributedBufferType::get(&ctx, shape, sparsityMapElemType, dimsOrder, dimsSpace, distributedAttr);

    auto distributedTypeIf = VPUIP::SparseBufferType::get(data, sparsityMap, nullptr, nullptr, nullptr, nullptr)
                                     .cast<VPU::DistributedTypeInterface>();

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 8});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 8}, SmallVector<int64_t>{1, 32, 6, 8}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .extractViewTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                ShapeRef(tileElemStrides), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }
}

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, SparseBufferTypeWeights) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const auto shape = SmallVector<int64_t>({64, 16, 3, 3});
    const auto sparseMapShape = SmallVector<int64_t>({64, 1, 1, 256});
    const auto dataElemType = mlir::Float16Type::get(&ctx);
    const auto sparsityMapElemType = mlir::IntegerType::get(&ctx, 1);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({2, 1, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterDataShapes(
            {SmallVector<int64_t>{48, 16, 3, 3}, SmallVector<int64_t>{16, 16, 3, 3}});
    const PerClusterShapesOffsetsVec perClusterDataOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{48, 0, 0, 0}});
    const auto perClusterDataShapesAttr = getIntArrayOfArray(&ctx, perClusterDataShapes);
    const auto perClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, perClusterDataOffsets);

    const auto dataDistrAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterDataShapesAttr, perClusterDataOffsetsAttr,
                                           perClusterDataShapesAttr, perClusterDataOffsetsAttr, nullptr);

    const PerClusterShapesOffsetsVec perClusterSMapShapes(
            {SmallVector<int64_t>{48, 1, 1, 256}, SmallVector<int64_t>{16, 1, 1, 256}});
    const PerClusterShapesOffsetsVec perClusterSMapOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{48, 0, 0, 0}});
    const auto perClusterSMapShapesAttr = getIntArrayOfArray(&ctx, perClusterSMapShapes);
    const auto perClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, perClusterSMapOffsets);

    const auto smapDistrAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterSMapShapesAttr, perClusterSMapOffsetsAttr,
                                           perClusterSMapShapesAttr, perClusterSMapOffsetsAttr, nullptr);

    const auto data = VPUIP::DistributedBufferType::get(&ctx, shape, dataElemType, dimsOrder, dimsSpace, dataDistrAttr);
    const auto sparsityMap = VPUIP::DistributedBufferType::get(&ctx, sparseMapShape, sparsityMapElemType, dimsOrder,
                                                               dimsSpace, smapDistrAttr);

    const auto isWeights = mlir::UnitAttr::get(&ctx);
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const auto sparsityCompression = VPUIP::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                         numElemsAttr, getIntAttr(&ctx, alignment));

    auto distributedTypeIf = VPUIP::SparseBufferType::get(data, sparsityMap, nullptr, isWeights, sparsityCompression)
                                     .cast<VPU::DistributedTypeInterface>();

    {
        const SmallVector<int64_t> newShape({32, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(
                {SmallVector<int64_t>{16, 16, 3, 3}, SmallVector<int64_t>{16, 16, 3, 3}});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{16, 0, 0, 0}});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForNewDataShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewDataShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(
                {SmallVector<int64_t>{16, 1, 1, 256}, SmallVector<int64_t>{16, 1, 1, 256}});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{16, 0, 0, 0}});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForNewSMapShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForNewDataShape,
                                                                                 distributedAttrForNewSMapShape};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const SmallVector<int64_t> newShape({32, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(
                {SmallVector<int64_t>{16, 16, 3, 3}, SmallVector<int64_t>{16, 16, 3, 3}});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{16, 0, 0, 0}});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForNewDataShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewDataShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(
                {SmallVector<int64_t>{16, 1, 1, 256}, SmallVector<int64_t>{16, 1, 1, 256}});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{16, 0, 0, 0}});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForNewSMapShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        ASSERT_NE(newType, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForNewDataShape,
                                                                                 distributedAttrForNewSMapShape};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const SmallVector<int64_t> newShape({32, 16, 3, 3});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(
                {SmallVector<int64_t>{16, 16, 3, 3}, SmallVector<int64_t>{16, 16, 3, 3}});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{16, 0, 0, 0}});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForNewDataShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewDataShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(
                {SmallVector<int64_t>{16, 1, 1, 256}, SmallVector<int64_t>{16, 1, 1, 256}});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{16, 0, 0, 0}});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForNewSMapShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForNewDataShape,
                                                                                 distributedAttrForNewSMapShape};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const SmallVector<int64_t> tileOffset({16, 0, 0, 0});
        const SmallVector<int64_t> tileShape({48, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(
                {SmallVector<int64_t>{32, 16, 3, 3}, SmallVector<int64_t>{16, 16, 3, 3}});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{32, 0, 0, 0}});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForDataTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForDataTile)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(
                {SmallVector<int64_t>{32, 1, 1, 256}, SmallVector<int64_t>{16, 1, 1, 256}});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{32, 0, 0, 0}});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForSMapTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForDataTile,
                                                                                 distributedAttrForSMapTile};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const SmallVector<int64_t> tileOffset({16, 0, 0, 0});
        const SmallVector<int64_t> tileShape({48, 16, 3, 3});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(
                {SmallVector<int64_t>{32, 16, 3, 3}, SmallVector<int64_t>{16, 16, 3, 3}});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{32, 0, 0, 0}});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForDataTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .extractViewTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                ShapeRef(tileElemStrides), distributedAttrForDataTile)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(
                {SmallVector<int64_t>{32, 1, 1, 256}, SmallVector<int64_t>{16, 1, 1, 256}});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{32, 0, 0, 0}});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForSMapTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForDataTile,
                                                                                 distributedAttrForSMapTile};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }
}

/*
    Test case: SparseBuffer type uses SEAttr specific for Interp Nearest with 2x upscale
    Initial Sparse buffer has the following components:
       - data: [1, 64, 32, 32]
       - sparsity_map: [1, 64, 64, 64]
       - storage_element_table [1, 1, 64, 64]

    Effective output type will have shape: [1, 64, 64, 64]

    Distribution is OVERLAPPED and each component will be be distributed as follows:
     - data:
        cluster 0: start (0, 0, 0, 0) -> end (0, 63, 17, 31)
        cluster 1: start (0, 0, 15, 0) -> end (0, 63, 31, 31)
     - sparsity map:
        cluster 0: start (0, 0, 0, 0) -> end (0, 63, 35, 63)
        cluster 1: start (0, 0, 30, 0) -> end (0, 63, 63, 63)
     - storage_element_table:
        cluster 0: start (0, 0, 0, 0) -> end (0, 0, 35, 63)
        cluster 1: start (0, 0, 30, 0) -> end (0, 0, 63, 63)
*/
TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, SparseBufferTypeWithSETable) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape seTableShape{1, 1, 64, 64};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);
    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);
    const SmallVector<int64_t> sizes({1, 64, 64, 64});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    const auto dataElemType = mlir::Float16Type::get(&ctx);
    const auto sparsityMapElemType = mlir::IntegerType::get(&ctx, 1);
    const auto seTableElemType = mlir::IntegerType::get(&ctx, 32);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterDataShapes(
            {SmallVector<int64_t>{1, 64, 18, 32}, SmallVector<int64_t>{1, 64, 17, 32}});
    const PerClusterShapesOffsetsVec perClusterDataOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});
    const auto perClusterDataShapesAttr = getIntArrayOfArray(&ctx, perClusterDataShapes);
    const auto perClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, perClusterDataOffsets);
    const auto distributedDataAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterDataShapesAttr, perClusterDataOffsetsAttr,
                                           perClusterDataShapesAttr, perClusterDataOffsetsAttr, nullptr);

    const PerClusterShapesOffsetsVec perClusterSeTableShapes(
            {SmallVector<int64_t>{1, 1, 36, 64}, SmallVector<int64_t>{1, 1, 34, 64}});
    const PerClusterShapesOffsetsVec perClusterSeTableOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});
    const auto perClusterSeTableShapesAttr = getIntArrayOfArray(&ctx, perClusterSeTableShapes);
    const auto perClusterSeTableOffsetsAttr = getIntArrayOfArray(&ctx, perClusterSeTableOffsets);
    const auto distributedSeTableAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterSeTableShapesAttr, perClusterSeTableOffsetsAttr,
                                           perClusterSeTableShapesAttr, perClusterSeTableOffsetsAttr, nullptr);

    const auto data = VPUIP::DistributedBufferType::get(&ctx, shape.raw(), dataElemType, dimsOrder, dimsSpace,
                                                        distributedDataAttr);
    const auto sparsityMap = VPUIP::DistributedBufferType::get(&ctx, shape.raw(), sparsityMapElemType, dimsOrder,
                                                               dimsSpace, distributedDataAttr);
    const auto seTable = VPUIP::DistributedBufferType::get(&ctx, seTableShape.raw(), seTableElemType, dimsOrder,
                                                           dimsSpace, distributedSeTableAttr);

    auto distributedTypeIf =
            VPUIP::SparseBufferType::get(data, sparsityMap, seTable, nullptr, nullptr, SEInterpolateAttr)
                    .cast<VPU::DistributedTypeInterface>();

    {
        /*
            Applying changeShape on initial SparseBuffer -> new effective shape: [1, 64, 64, 32]
                => Resulting component shapes:
                    - data: [1, 64, 32, 16]
                    - sparsity_map: [1, 64, 64, 32]
                    - storage_element_table [1, 1, 64, 32]

            Distribution is OVERLAPPED and each component will be be distributed as follows:
            - data:
                cluster 0: start (0, 0, 0, 0) -> end (0, 63, 17, 15)
                cluster 1: start (0, 0, 15, 0) -> end (0, 63, 31, 15)
            - sparsity map:
                cluster 0: start (0, 0, 0, 0) -> end (0, 63, 35, 31)
                cluster 1: start (0, 0, 30, 0) -> end (0, 63, 63, 31)
            - storage_element_table:
                cluster 0: start (0, 0, 0, 0) -> end (0, 0, 35, 31)
                cluster 1: start (0, 0, 30, 0) -> end (0, 0, 63, 31)
        */
        const auto newShape = SmallVector<int64_t>({1, 64, 64, 32});
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 36, 32}, SmallVector<int64_t>{1, 64, 34, 32}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForNewShape =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 18, 16}, SmallVector<int64_t>{1, 64, 17, 16}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 36, 32}, SmallVector<int64_t>{1, 64, 34, 32}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 36, 32}, SmallVector<int64_t>{1, 1, 34, 32}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        // Applying changeShapeElemTyp on initial SparseBuffer -> new effective shape: [1, 64, 64, 32]
        // Same test case as for changeShape, only element type gets modified as well
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const auto newShape = SmallVector<int64_t>({1, 64, 64, 32});
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 36, 32}, SmallVector<int64_t>{1, 64, 34, 32}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForNewShape =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 18, 16}, SmallVector<int64_t>{1, 64, 17, 16}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 36, 32}, SmallVector<int64_t>{1, 64, 34, 32}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 36, 32}, SmallVector<int64_t>{1, 1, 34, 32}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        // Applying changeTypeComponents on initial SparseBuffer -> new effective shape: [1, 64, 64, 32]
        // Same test case as for changeShape; only type component that changes is the shape
        const auto newShape = SmallVector<int64_t>({1, 64, 64, 32});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 36, 32}, SmallVector<int64_t>{1, 64, 34, 32}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForNewShape =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 18, 16}, SmallVector<int64_t>{1, 64, 17, 16}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 36, 32}, SmallVector<int64_t>{1, 64, 34, 32}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 36, 32}, SmallVector<int64_t>{1, 1, 34, 32}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 30, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        /*
            Applying extractDenseTile on initial SparseBuffer -> new effective shape: [1, 64, 32, 64]
            Initial effective buffer (0, 0, 0, 0) -> (0, 63, 63, 63)
            Extracted tile (0, 0, 15, 0) -> (0, 63, 46, 63)
                => Resulting component shapes:
                    - data: [1, 64, 17, 32]
                    - sparsity_map: [1, 64, 32, 64]
                    - storage_element_table [1, 1, 32, 64]

            Distribution is OVERLAPPED and each component will be be distributed as follows:
            - data:
                cluster 0: start (0, 0, 0, 0) -> end (0, 63, 10, 31)
                cluster 1: start (0, 0, 8, 0) -> end (0, 63, 16, 31)
            - sparsity map:
                cluster 0: start (0, 0, 0, 0) -> end (0, 63, 19, 63)
                cluster 1: start (0, 0, 15, 0) -> end (0, 63, 31, 63)
            - storage_element_table:
                cluster 0: start (0, 0, 0, 0) -> end (0, 0, 19, 63)
                cluster 1: start (0, 0, 15, 0) -> end (0, 0, 31, 63)
        */
        const SmallVector<int64_t> tileOffset({0, 0, 15, 0});
        const SmallVector<int64_t> tileShape({1, 64, 32, 64});
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 20, 64}, SmallVector<int64_t>{1, 64, 17, 64}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForTile =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForTile)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        auto newSparseType = newType.dyn_cast<VPUIP::SparseBufferType>();
        ASSERT_NE(newSparseType, nullptr);

        auto tiledSeAttr = newSparseType.getSeAttr().cast<VPU::SEInterpolateAttr>();
        auto expectedSeOffsets = getIntArrayAttr(&ctx, SmallVector<int64_t>{0, 0, 1, 0});
        auto expectedSeSizes = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 64, 32, 64});
        EXPECT_EQ(tiledSeAttr.getOffsets(), expectedSeOffsets);
        EXPECT_EQ(tiledSeAttr.getSizes(), expectedSeSizes);

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 11, 32}, SmallVector<int64_t>{1, 64, 9, 32}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 20, 64}, SmallVector<int64_t>{1, 64, 17, 64}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 20, 64}, SmallVector<int64_t>{1, 1, 17, 64}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        /*
            Applying extractViewTile on initial SparseBuffer -> new effective shape: [1, 64, 32, 64]
            Initial effective buffer (0, 0, 0, 0) -> (0, 63, 63, 63)
            Extracted tile (0, 0, 16, 0) -> (0, 63, 47, 63)
                => Resulting component shapes:
                    - data: [1, 64, 16, 32]
                    - sparsity_map: [1, 64, 32, 64]
                    - storage_element_table [1, 1, 32, 64]

            Distribution is OVERLAPPED and each component will be be distributed as follows:
            - data:
                cluster 0: start (0, 0, 0, 0) -> end (0, 63, 9, 31)
                cluster 1: start (0, 0, 7, 0) -> end (0, 63, 15, 31)
            - sparsity map:
                cluster 0: start (0, 0, 0, 0) -> end (0, 63, 19, 63)
                cluster 1: start (0, 0, 14, 0) -> end (0, 63, 31, 63)
            - storage_element_table:
                cluster 0: start (0, 0, 0, 0) -> end (0, 0, 19, 63)
                cluster 1: start (0, 0, 14, 0) -> end (0, 0, 31, 63)
        */
        const SmallVector<int64_t> tileOffset({0, 0, 16, 0});
        const SmallVector<int64_t> tileShape({1, 64, 32, 64});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});

        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 20, 64}, SmallVector<int64_t>{1, 64, 18, 64}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 14, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForTile =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .extractViewTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                ShapeRef(tileElemStrides), distributedAttrForTile)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        auto newSparseType = newType.dyn_cast<VPUIP::SparseBufferType>();
        ASSERT_NE(newSparseType, nullptr);

        auto tiledSeAttr = newSparseType.getSeAttr().cast<VPU::SEInterpolateAttr>();
        auto expectedSeOffsets = getIntArrayAttr(&ctx, SmallVector<int64_t>{0, 0, 0, 0});
        auto expectedSeSizes = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 64, 32, 64});
        EXPECT_EQ(tiledSeAttr.getOffsets(), expectedSeOffsets);
        EXPECT_EQ(tiledSeAttr.getSizes(), expectedSeSizes);

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 10, 32}, SmallVector<int64_t>{1, 64, 9, 32}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 20, 64}, SmallVector<int64_t>{1, 64, 18, 64}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 14, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 20, 64}, SmallVector<int64_t>{1, 1, 18, 64}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 14, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }
}

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, SparseTensorTypeDataAndSparsityMap) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = SmallVector<int64_t>({1, 64, 13, 16});
    const auto dataElemType = mlir::Float16Type::get(&ctx);
    const auto sparsityMapElemType = mlir::IntegerType::get(&ctx, 1);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterShapes(
            {SmallVector<int64_t>{1, 64, 8, 16}, SmallVector<int64_t>{1, 64, 6, 16}});
    const PerClusterShapesOffsetsVec perClusterOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});
    const auto perClusterShapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    const auto perClusterOffsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto distributedAttr = VPU::DistributionInfoAttr::get(
            &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
            perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr, nullptr);

    const auto data = VPU::DistributedTensorType::get(&ctx, shape, dataElemType, dimsOrder, dimsSpace, distributedAttr);
    const auto sparsityMap =
            VPU::DistributedTensorType::get(&ctx, shape, sparsityMapElemType, dimsOrder, dimsSpace, distributedAttr);

    auto distributedTypeIf = VPU::SparseTensorType::get(data, sparsityMap, nullptr, nullptr, nullptr, nullptr)
                                     .cast<VPU::DistributedTypeInterface>();

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 16});
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 16}, SmallVector<int64_t>{1, 32, 6, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const auto newShape = SmallVector<int64_t>({1, 32, 13, 8});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 8, 8}, SmallVector<int64_t>{1, 32, 6, 8}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 7, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        compareExplicitDistributedAttrs(newType, SmallVector<VPU::DistributionInfoAttr>{distributedAttrForNewShape,
                                                                                        distributedAttrForNewShape});
    }

    {
        const SmallVector<int64_t> tileOffset({0, 0, 6, 0});
        const SmallVector<int64_t> tileShape({1, 64, 5, 16});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 2, 16}, SmallVector<int64_t>{1, 64, 4, 16}});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 1, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto distributedAttrForNewShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        EXPECT_ANY_THROW(distributedTypeIf.extractViewTileForExplicitDistribution(
                ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides), distributedAttrForNewShape));
    }
}

TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, SparseTensorTypeWeights) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto shape = SmallVector<int64_t>({64, 16, 3, 3});
    const auto sparseMapShape = SmallVector<int64_t>({64, 1, 1, 256});
    const auto dataElemType = mlir::Float16Type::get(&ctx);
    const auto sparsityMapElemType = mlir::IntegerType::get(&ctx, 1);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
    const int64_t numClusters = 2;
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    const PerClusterShapesOffsetsVec perClusterDataShapes(numClusters, SmallVector<int64_t>{64, 16, 3, 3});
    const PerClusterShapesOffsetsVec perClusterDataOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});
    const auto perClusterDataShapesAttr = getIntArrayOfArray(&ctx, perClusterDataShapes);
    const auto perClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, perClusterDataOffsets);

    const auto dataDistrAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr,
                                           nullptr, nullptr, perClusterDataShapesAttr, perClusterDataOffsetsAttr,
                                           perClusterDataShapesAttr, perClusterDataOffsetsAttr, nullptr);

    const PerClusterShapesOffsetsVec perClusterSMapShapes(numClusters, SmallVector<int64_t>{64, 1, 1, 256});
    const PerClusterShapesOffsetsVec perClusterSMapOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});
    const auto perClusterSMapShapesAttr = getIntArrayOfArray(&ctx, perClusterSMapShapes);
    const auto perClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, perClusterSMapOffsets);

    const auto smapDistrAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr,
                                           nullptr, nullptr, perClusterSMapShapesAttr, perClusterSMapOffsetsAttr,
                                           perClusterSMapShapesAttr, perClusterSMapOffsetsAttr, nullptr);

    const auto data = VPU::DistributedTensorType::get(&ctx, shape, dataElemType, dimsOrder, dimsSpace, dataDistrAttr);
    const auto sparsityMap = VPU::DistributedTensorType::get(&ctx, sparseMapShape, sparsityMapElemType, dimsOrder,
                                                             dimsSpace, smapDistrAttr);

    const auto isWeights = mlir::UnitAttr::get(&ctx);
    const int64_t compressionAxis = 0;
    const int64_t alignment = 16;
    SmallVector<int64_t> numElems(64);
    std::iota(numElems.begin(), numElems.end(), 0);
    const auto numElemsType = mlir::RankedTensorType::get({64}, getInt64Type(&ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numElems));
    const auto sparsityCompression = VPU::SparsityCompressionAttr::get(&ctx, getIntAttr(&ctx, compressionAxis),
                                                                       numElemsAttr, getIntAttr(&ctx, alignment));

    auto distributedTypeIf = VPU::SparseTensorType::get(data, sparsityMap, nullptr, isWeights, sparsityCompression)
                                     .cast<VPU::DistributedTypeInterface>();

    {
        const SmallVector<int64_t> newShape({32, 16, 3, 3});
        const SmallVector<int64_t> newSMShape({32, 1, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(numClusters, SmallVector<int64_t>{32, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForNewDataShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewDataShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(numClusters, SmallVector<int64_t>{32, 1, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForNewSMapShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForNewDataShape,
                                                                                 distributedAttrForNewSMapShape};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const SmallVector<int64_t> newShape({32, 16, 3, 3});
        const SmallVector<int64_t> newSMShape({32, 1, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(numClusters, SmallVector<int64_t>{32, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForNewDataShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewDataShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(numClusters, SmallVector<int64_t>{32, 1, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForNewSMapShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        ASSERT_NE(newType, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForNewDataShape,
                                                                                 distributedAttrForNewSMapShape};

        for (const auto& p : zip(newType.getDistributedTypes(), expectedDistributedAttrs)) {
            const auto buffType = std::get<0>(p);
            const auto expectedDistribution = std::get<1>(p);

            const auto newDistribution = buffType.cast<VPU::DistributedTensorType>().getDistribution();
            EXPECT_EQ(newDistribution, expectedDistribution);
        }
    }

    {
        const SmallVector<int64_t> newShape({32, 16, 3, 3});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(numClusters, SmallVector<int64_t>{32, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForNewDataShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewDataShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(numClusters, SmallVector<int64_t>{32, 1, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForNewSMapShape = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForNewDataShape,
                                                                                 distributedAttrForNewSMapShape};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const SmallVector<int64_t> tileOffset({16, 0, 0, 0});
        const SmallVector<int64_t> tileShape({48, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(numClusters, SmallVector<int64_t>{48, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForDataTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForDataTile)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec expectedPerClusterSMapShapes(numClusters, SmallVector<int64_t>{48, 1, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterSMapOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterSMapShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapShapes);
        const auto expectedPerClusterSMapOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterSMapOffsets);

        const auto distributedAttrForSMapTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterSMapShapesAttr, expectedPerClusterSMapOffsetsAttr, expectedPerClusterSMapShapesAttr,
                expectedPerClusterSMapOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributedAttrs = {distributedAttrForDataTile,
                                                                                 distributedAttrForSMapTile};

        compareExplicitDistributedAttrs(newType, expectedDistributedAttrs);
    }

    {
        const SmallVector<int64_t> tileOffset({16, 0, 0, 0});
        const SmallVector<int64_t> tileShape({48, 16, 3, 3});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});
        const PerClusterShapesOffsetsVec expectedPerClusterDataShapes(numClusters, SmallVector<int64_t>{48, 16, 3, 3});
        const PerClusterShapesOffsetsVec expectedPerClusterDataOffsets(numClusters, SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterDataShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataShapes);
        const auto expectedPerClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterDataOffsets);

        const auto distributedAttrForDataTile = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                expectedPerClusterDataShapesAttr, expectedPerClusterDataOffsetsAttr, expectedPerClusterDataShapesAttr,
                expectedPerClusterDataOffsetsAttr, nullptr);

        EXPECT_ANY_THROW(distributedTypeIf.extractViewTileForExplicitDistribution(
                ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides), distributedAttrForDataTile));
    }
}

/*
    Test case: SparseTensor type uses SEAttr specific for Interp Nearest with 2x upscale and SeSize = 32
    Initial Sparse buffer has the following components:
       - data: [1, 64, 32, 32]
       - sparsity_map: [1, 64, 64, 64]
       - storage_element_table [1, 2, 64, 64]

    Effective output type will have shape: [1, 64, 64, 64]

    Distribution is SEGMENTED over C and each component will be be distributed as follows:
     - data:
        cluster 0: start (0, 0, 0, 0) -> end (0, 31, 31, 31)
        cluster 1: start (0, 32, 0, 0) -> end (0, 63, 31, 31)
     - sparsity map:
        cluster 0: start (0, 0, 0, 0) -> end (0, 31, 63, 63)
        cluster 1: start (0, 32, 0, 0) -> end (0, 63, 63, 63)
     - storage_element_table:
        cluster 0: start (0, 0, 0, 0) -> end (0, 0, 63, 63)
        cluster 1: start (0, 1, 0, 0) -> end (0, 1, 63, 63)
*/
TEST_F(MLIR_DistributedTypesIfMethodsForExplicitDistribution, SparseTensorTypeWithSETable) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const Shape shape{1, 64, 32, 32};
    const Shape seTableShape{1, 2, 64, 64};

    const SmallVector<float> scale({1, 1, 2, 2});
    const auto scaleAttr = getFPArrayAttr(&ctx, scale);
    const SmallVector<int64_t> offsets({0, 0, 0, 0});
    const auto offsetsAttr = getIntArrayAttr(&ctx, offsets);
    const SmallVector<int64_t> sizes({1, 64, 64, 64});
    const auto sizesAttr = getIntArrayAttr(&ctx, sizes);

    const auto dataElemType = mlir::Float16Type::get(&ctx);
    const auto sparsityMapElemType = mlir::IntegerType::get(&ctx, 1);
    const auto seTableElemType = mlir::IntegerType::get(&ctx, 32);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);

    const auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, VPU::NCEInterpolateMode::NEAREST);
    const auto coordTransformModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, IE::InterpolateCoordMode::ASYMMETRIC);
    const auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, IE::InterpolateNearestMode::FLOOR);
    const auto SEInterpolateAttr = VPU::SEInterpolateAttr::get(
            &ctx, modeAttr, coordTransformModeAttr, scaleAttr, nearestModeAttr, offsetsAttr, sizesAttr,
            /*initialInputShapeAttr=*/nullptr, /*initialOutputShapeAttr=*/nullptr);

    const auto distributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);

    const PerClusterShapesOffsetsVec perClusterDataShapes(
            {SmallVector<int64_t>{1, 32, 32, 32}, SmallVector<int64_t>{1, 32, 32, 32}});
    const PerClusterShapesOffsetsVec perClusterDataOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});
    const auto perClusterDataShapesAttr = getIntArrayOfArray(&ctx, perClusterDataShapes);
    const auto perClusterDataOffsetsAttr = getIntArrayOfArray(&ctx, perClusterDataOffsets);
    const auto distributedDataAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterDataShapesAttr, perClusterDataOffsetsAttr,
                                           perClusterDataShapesAttr, perClusterDataOffsetsAttr, nullptr);

    const PerClusterShapesOffsetsVec perClusterSeTableShapes(
            {SmallVector<int64_t>{1, 1, 64, 64}, SmallVector<int64_t>{1, 1, 64, 64}});
    const PerClusterShapesOffsetsVec perClusterSeTableOffsets(
            {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 1, 0, 0}});
    const auto perClusterSeTableShapesAttr = getIntArrayOfArray(&ctx, perClusterSeTableShapes);
    const auto perClusterSeTableOffsetsAttr = getIntArrayOfArray(&ctx, perClusterSeTableOffsets);
    const auto distributedSeTableAttr =
            VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           nullptr, nullptr, perClusterSeTableShapesAttr, perClusterSeTableOffsetsAttr,
                                           perClusterSeTableShapesAttr, perClusterSeTableOffsetsAttr, nullptr);

    const auto data =
            VPU::DistributedTensorType::get(&ctx, shape.raw(), dataElemType, dimsOrder, dimsSpace, distributedDataAttr);
    const auto sparsityMap = VPU::DistributedTensorType::get(&ctx, shape.raw(), sparsityMapElemType, dimsOrder,
                                                             dimsSpace, distributedDataAttr);
    const auto seTable = VPU::DistributedTensorType::get(&ctx, seTableShape.raw(), seTableElemType, dimsOrder,
                                                         dimsSpace, distributedSeTableAttr);

    auto distributedTypeIf = VPU::SparseTensorType::get(data, sparsityMap, seTable, nullptr, nullptr, SEInterpolateAttr)
                                     .cast<VPU::DistributedTypeInterface>();

    {
        /*
            Applying changeShape on initial SparseTensor -> new effective shape: [1, 64, 64, 32]
                => Resulting component shapes:
                    - data: [1, 64, 32, 16]
                    - sparsity_map: [1, 64, 64, 32]
                    - storage_element_table [1, 1, 64, 32]

            Distribution is SEGMENTED over C and each component will be be distributed as follows:
            - data:
                cluster 0: start (0, 0, 0, 0) -> end (0, 31, 31, 15)
                cluster 1: start (0, 32, 0, 0) -> end (0, 63, 31, 15)
            - sparsity map:
                cluster 0: start (0, 0, 0, 0) -> end (0, 31, 63, 31)
                cluster 1: start (0, 32, 0, 0) -> end (0, 63, 63, 31)
            - storage_element_table:
                cluster 0: start (0, 0, 0, 0) -> end (0, 0, 63, 31)
                cluster 1: start (0, 1, 0, 0) -> end (0, 1, 63, 31)
        */
        const auto newShape = SmallVector<int64_t>({1, 64, 64, 32});
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 64, 32}, SmallVector<int64_t>{1, 32, 64, 32}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForNewShape =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf.changeShapeForExplicitDistribution(ShapeRef(newShape), distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 32, 16}, SmallVector<int64_t>{1, 32, 32, 16}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 64, 32}, SmallVector<int64_t>{1, 32, 64, 32}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 64, 32}, SmallVector<int64_t>{1, 1, 64, 32}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 1, 0, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        // Applying changeShapeElemTyp on initial SparseTensor -> new effective shape: [1, 64, 64, 32]
        // Same test case as for changeShape, only element type gets modified as well
        const auto newElemType = mlir::IntegerType::get(&ctx, 8);
        const auto newShape = SmallVector<int64_t>({1, 64, 64, 32});
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 64, 32}, SmallVector<int64_t>{1, 32, 64, 32}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForNewShape =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .changeShapeElemTypeForExplicitDistribution(ShapeRef(newShape), newElemType,
                                                                           distributedAttrForNewShape)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 32, 16}, SmallVector<int64_t>{1, 32, 32, 16}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 64, 32}, SmallVector<int64_t>{1, 32, 64, 32}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 64, 32}, SmallVector<int64_t>{1, 1, 64, 32}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 1, 0, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        // Applying changeTypeComponents on initial SparseTensor -> new effective shape: [1, 64, 64, 32]
        // Same test case as for changeShape; only type component that changes is the shape
        const auto newShape = SmallVector<int64_t>({1, 64, 64, 32});
        const auto newTypeComponents = TypeComponents().setShape(ShapeRef(newShape));
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 64, 32}, SmallVector<int64_t>{1, 32, 64, 32}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForNewShape =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType =
                distributedTypeIf
                        .changeTypeComponentsForExplicitDistribution(newTypeComponents, distributedAttrForNewShape)
                        .dyn_cast<VPU::DistributedTypeInterface>();

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 32, 16}, SmallVector<int64_t>{1, 32, 32, 16}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 64, 32}, SmallVector<int64_t>{1, 32, 64, 32}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 64, 32}, SmallVector<int64_t>{1, 1, 64, 32}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 1, 0, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        /*
            Applying extractDenseTile on initial SparseTensor -> new effective shape: [1, 64, 32, 64]
            Initial effective buffer (0, 0, 0, 0) -> (0, 63, 63, 63)
            Extracted tile (0, 0, 16, 0) -> (0, 63, 47, 63)
                => Resulting component shapes:
                    - data: [1, 64, 16, 32]
                    - sparsity_map: [1, 64, 32, 64]
                    - storage_element_table [1, 2, 32, 64]

            Distribution is SEGMENTED over C and each component will be be distributed as follows:
            - data:
                cluster 0: start (0, 0, 0, 0) -> end (0, 31, 15, 15)
                cluster 1: start (0, 32, 0, 0) -> end (0, 63, 15, 15)
            - sparsity map:
                cluster 0: start (0, 0, 0, 0) -> end (0, 31, 31, 31)
                cluster 1: start (0, 32, 0, 0) -> end (0, 63, 31, 31)
            - storage_element_table:
                cluster 0: start (0, 0, 0, 0) -> end (0, 0, 31, 31)
                cluster 1: start (0, 1, 0, 0) -> end (0, 1, 31, 31)
        */
        const SmallVector<int64_t> tileOffset({0, 0, 16, 0});
        const SmallVector<int64_t> tileShape({1, 64, 32, 64});
        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 32, 64}, SmallVector<int64_t>{1, 32, 32, 64}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForTile =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        auto newType = distributedTypeIf
                               .extractDenseTileForExplicitDistribution(ShapeRef(tileOffset), ShapeRef(tileShape),
                                                                        distributedAttrForTile)
                               .dyn_cast<VPU::DistributedTypeInterface>();

        auto newSparseType = newType.dyn_cast<VPU::SparseTensorType>();
        ASSERT_NE(newSparseType, nullptr);

        auto tiledSeAttr = newSparseType.getSeAttr().cast<VPU::SEInterpolateAttr>();
        auto expectedSeOffsets = getIntArrayAttr(&ctx, SmallVector<int64_t>{0, 0, 0, 0});
        auto expectedSeSizes = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 64, 32, 64});
        EXPECT_EQ(tiledSeAttr.getOffsets(), expectedSeOffsets);
        EXPECT_EQ(tiledSeAttr.getSizes(), expectedSeSizes);

        const PerClusterShapesOffsetsVec dataPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 16, 32}, SmallVector<int64_t>{1, 32, 16, 32}});
        const PerClusterShapesOffsetsVec dataPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto dataPerClusterShapesAttr = getIntArrayOfArray(&ctx, dataPerClusterShapes);
        const auto dataPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, dataPerClusterOffsets);

        const auto expectedDataDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, dataPerClusterShapesAttr, dataPerClusterOffsetsAttr,
                                               dataPerClusterShapesAttr, dataPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec sMapPerClusterShapes(
                {SmallVector<int64_t>{1, 32, 32, 64}, SmallVector<int64_t>{1, 32, 32, 64}});
        const PerClusterShapesOffsetsVec sMapPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 32, 0, 0}});

        const auto sMapPerClusterShapesAttr = getIntArrayOfArray(&ctx, sMapPerClusterShapes);
        const auto sMapPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, sMapPerClusterOffsets);

        const auto expectedSMapDistrAttr =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr,
                                               sMapPerClusterShapesAttr, sMapPerClusterOffsetsAttr, nullptr);

        const PerClusterShapesOffsetsVec seTablePerClusterShapes(
                {SmallVector<int64_t>{1, 1, 32, 64}, SmallVector<int64_t>{1, 1, 32, 64}});
        const PerClusterShapesOffsetsVec seTablePerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 1, 0, 0}});

        const auto seTablePerClusterShapesAttr = getIntArrayOfArray(&ctx, seTablePerClusterShapes);
        const auto seTablePerClusterOffsetsAttr = getIntArrayOfArray(&ctx, seTablePerClusterOffsets);

        const auto expectedSeTableDistrAttr = VPU::DistributionInfoAttr::get(
                &ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                seTablePerClusterShapesAttr, seTablePerClusterOffsetsAttr, seTablePerClusterShapesAttr,
                seTablePerClusterOffsetsAttr, nullptr);

        const SmallVector<VPU::DistributionInfoAttr> expectedDistributions = {
                expectedDataDistrAttr, expectedSMapDistrAttr, expectedSeTableDistrAttr};

        compareExplicitDistributedAttrs(newType, expectedDistributions);
    }

    {
        // extractViewTile not applicable for Tensor type
        const SmallVector<int64_t> tileOffset({0, 0, 16, 0});
        const SmallVector<int64_t> tileShape({1, 64, 32, 64});
        const SmallVector<int64_t> tileElemStrides({1, 1, 1, 1});

        const PerClusterShapesOffsetsVec newPerClusterShapes(
                {SmallVector<int64_t>{1, 64, 20, 64}, SmallVector<int64_t>{1, 64, 19, 64}});
        const PerClusterShapesOffsetsVec newPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 15, 0}});

        const auto newPerClusterShapesAttr = getIntArrayOfArray(&ctx, newPerClusterShapes);
        const auto newPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, newPerClusterOffsets);

        const auto distributedAttrForTile =
                VPU::DistributionInfoAttr::get(&ctx, distributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
                                               nullptr, nullptr, newPerClusterShapesAttr, newPerClusterOffsetsAttr,
                                               newPerClusterShapesAttr, newPerClusterOffsetsAttr, nullptr);

        EXPECT_ANY_THROW(distributedTypeIf.extractViewTileForExplicitDistribution(
                ShapeRef(tileOffset), ShapeRef(tileShape), ShapeRef(tileElemStrides), distributedAttrForTile));
    }
}
