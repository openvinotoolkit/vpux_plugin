//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using PerClusterShapesOffsetsVec = SmallVector<SmallVector<int64_t>>;
constexpr StringRef CMX_NAME = "CMX_NN";

void testDistributedAttr(llvm::StringLiteral inputIR, vpux::VPU::DistributedTensorType inputType,
                         vpux::VPU::DistributedTensorType expectedDistributedType, mlir::MLIRContext* ctx) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto checkElementType = [&](mlir::Type res, mlir::Type exp) {
        auto resQuant = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(res);
        auto expQuant = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(exp);
        if (resQuant != nullptr && expQuant != nullptr) {
            EXPECT_TRUE(resQuant.getExpressedType() == expQuant.getExpressedType());
            EXPECT_EQ(resQuant.getZeroPoint(), expQuant.getZeroPoint());
            EXPECT_EQ(resQuant.getScale(), expQuant.getScale());
            return;
        }

        ASSERT_EQ(res, exp);
    };

    for (auto& op : func.getOps()) {
        if (auto distributedCastOp = mlir::dyn_cast<vpux::VPU::DistributedCastOpInterface>(op)) {
            auto resDistributedType = distributedCastOp.inferCastedDistOutput(inputType);

            // Could not infer output distribution
            if (expectedDistributedType == nullptr) {
                ASSERT_EQ(mlir::failed(resDistributedType), true);
            } else {
                ASSERT_EQ(mlir::succeeded(resDistributedType), true);

                auto resultType = resDistributedType.value().cast<vpux::VPU::DistributedTensorType>();
                EXPECT_EQ(resultType.getShape(), expectedDistributedType.getShape());
                EXPECT_EQ(resultType.getDimsOrder(), expectedDistributedType.getDimsOrder());
                checkElementType(resultType.getElementType(), expectedDistributedType.getElementType());
                EXPECT_EQ(resultType.getMemSpace(), expectedDistributedType.getMemSpace());
                EXPECT_EQ(resultType.getDistribution(), expectedDistributedType.getDistribution());
            }
        }
    }
}

using MLIR_DistributedCastOpInterfaceTest = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_DistributedCastOpInterfaceTest, QuantizeCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        !qElemType = !quant.uniform<u8:f16, 0.01:32>
        !qElemType1 = !quant.uniform<u8:f16, 0.02:64>
        module @test {
            func.func @main(%arg0: tensor<1x128x16x16x!qElemType>) -> tensor<1x128x16x16x!qElemType1> {
                %0 = VPU.QuantizeCast(%arg0) {dstElemType = !qElemType1}
                        : tensor<1x128x16x16x!qElemType> -> tensor<1x128x16x16x!qElemType1>
                return %0 : tensor<1x128x16x16x!qElemType1>
            }
        }
    )";
    const vpux::Shape shape = {1, 128, 16, 16};

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClusters = getIntAttr(&ctx, 2);
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto dimsOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(&ctx));
    const auto inQuantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx), mlir::Float16Type::get(&ctx),
                                                                    0.01, 32, 0, 255);
    const auto outQuantType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(&ctx),
                                                                     mlir::Float16Type::get(&ctx), 0.02, 64, 0, 255);

    // Distribution does not change between input and output for QuantizeCast
    auto getInputTypeAndTest = [&](VPU::DistributedTensorAttr distribution) {
        const auto inputDistributedType = vpux::VPU::DistributedTensorType::get(&ctx, shape.raw(), inQuantType,
                                                                                dimsOrder, memSpace, distribution);
        const auto outputDistributedType = inputDistributedType.cast<NDTypeInterface>()
                                                   .changeElemType(outQuantType)
                                                   .cast<VPU::DistributedTensorType>();

        testDistributedAttr(inputIR, inputDistributedType, outputDistributedType, &ctx);
    };

    {
        const auto overlappedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);

        const auto overlappedDistributedAtr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, overlappedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        getInputTypeAndTest(overlappedDistributedAtr);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        getInputTypeAndTest(duplicatedDistributedAtr);
    }
}

TEST_F(MLIR_DistributedCastOpInterfaceTest, AffineReshape) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        module @test {
            func.func @main(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x128x1x256xf16, {order = #NWCH}> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 128, 1, 256]}
                        : tensor<1x128x16x16xf16, {order = #NHWC}> -> tensor<1x128x1x256xf16, {order = #NWCH}>
                return %0 : tensor<1x128x1x256xf16, {order = #NWCH}>
            }
        }
    )";
    const vpux::Shape shape = {1, 128, 16, 16};
    const vpux::Shape newShape = {1, 128, 1, 256};

    const auto numClusters = getIntAttr(&ctx, 2);
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto inDimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto outDimsOrder = mlir::AffineMapAttr::get(DimsOrder::NWCH.toAffineMap(&ctx));
    auto fp16Type = mlir::Float16Type::get(&ctx);

    // Only DUPLICATED distribution is legal for AffineReshape
    auto getInputTypeAndTest = [&](VPU::DistributedTensorAttr inDistribution,
                                   VPU::DistributedTensorAttr outDistribution) {
        const auto inputDistributedType = vpux::VPU::DistributedTensorType::get(&ctx, shape.raw(), fp16Type,
                                                                                inDimsOrder, memSpace, inDistribution);

        if (outDistribution == nullptr) {
            testDistributedAttr(inputIR, inputDistributedType, nullptr, &ctx);
            return;
        }

        const auto outputDistributedType = vpux::VPU::DistributedTensorType::get(
                &ctx, newShape.raw(), fp16Type, outDimsOrder, memSpace, outDistribution);

        testDistributedAttr(inputIR, inputDistributedType, outputDistributedType, &ctx);
    };

    {
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto segmentedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters.getInt(), SmallVector<int64_t>{1, 128, 8, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 8, 0}});

        const auto inPerClusterShapesAttr = getIntArrayOfArray(&ctx, inPerClusterShapes);
        const auto inPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterOffsets);

        const auto segDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, segmentedMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                inPerClusterShapesAttr, inPerClusterOffsetsAttr, inPerClusterShapesAttr, inPerClusterOffsetsAttr,
                nullptr);

        getInputTypeAndTest(segDistributedAttr, nullptr);
    }

    {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED);
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters.getInt(), SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(numClusters.getInt(), SmallVector<int64_t>{0, 0, 0, 0});

        const auto inPerClusterShapesAttr = getIntArrayOfArray(&ctx, inPerClusterShapes);
        const auto inPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterOffsets);

        const auto inDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                inPerClusterShapesAttr, inPerClusterOffsetsAttr, inPerClusterShapesAttr, inPerClusterOffsetsAttr,
                nullptr);

        const PerClusterShapesOffsetsVec expectedPerClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 128, 1, 256});
        const PerClusterShapesOffsetsVec expectedPerClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto expectedPerClusterShapesAttr = getIntArrayOfArray(&ctx, expectedPerClusterShapes);
        const auto expectedPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, expectedPerClusterOffsets);
        const auto duplicatedDistributedAtr = VPU::DistributedTensorAttr::get(
                &ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                expectedPerClusterShapesAttr, expectedPerClusterOffsetsAttr, expectedPerClusterShapesAttr,
                expectedPerClusterOffsetsAttr, nullptr);

        getInputTypeAndTest(inDistributedAttr, duplicatedDistributedAtr);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));

        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters.getInt(),
                                                                   SmallVector<int64_t>{1, 64, 16, 16});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inPerMemoryClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 128, 16, 16});
        const PerClusterShapesOffsetsVec inPerMemoryClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto inPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, inPerClusterComputeShapes);
        const auto inPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterComputeOffsets);
        const auto inPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, inPerMemoryClusterShapes);
        const auto inPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, inPerMemoryClusterOffsets);

        const auto inDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, segDupMode, numTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                inPerClusterComputeShapesAttr, inPerClusterComputeOffsetsAttr, inPerClusterMemoryShapesAttr,
                inPerClusterMemoryOffsetsAttr, nullptr);

        getInputTypeAndTest(inDistributedAttr, nullptr);
    }
}

TEST_F(MLIR_DistributedCastOpInterfaceTest, PermuteCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
        module @test {
            func.func @main(%arg0: tensor<1x256x40x1xf16, {order = #NHWC}>) -> tensor<1x40x1x256xf16, {order = #NCHW}> {
                %0 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW}
                        : tensor<1x256x40x1xf16, {order = #NHWC}> -> tensor<1x40x1x256xf16, {order = #NCHW}>
                return %0 : tensor<1x40x1x256xf16, {order = #NCHW}>
            }
        }
    )";
    const vpux::Shape shape = {1, 256, 40, 1};
    const vpux::Shape newShape = {1, 40, 1, 256};

    const auto numClusters = getIntAttr(&ctx, 2);
    const auto memSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    const auto inDimsOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto outDimsOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(&ctx));
    auto fp16Type = mlir::Float16Type::get(&ctx);

    auto getTypesAndTest = [&](VPU::DistributedTensorAttr inDistribution, VPU::DistributedTensorAttr outDistribution) {
        const auto inputDistributedType = vpux::VPU::DistributedTensorType::get(&ctx, shape.raw(), fp16Type,
                                                                                inDimsOrder, memSpace, inDistribution);

        if (outDistribution == nullptr) {
            testDistributedAttr(inputIR, inputDistributedType, nullptr, &ctx);
            return;
        }

        const auto outputDistributedType = vpux::VPU::DistributedTensorType::get(
                &ctx, newShape.raw(), fp16Type, outDimsOrder, memSpace, outDistribution);

        testDistributedAttr(inputIR, inputDistributedType, outputDistributedType, &ctx);
    };

    {
        const auto inNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto ovrMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const PerClusterShapesOffsetsVec inPerClusterShapes(numClusters.getInt(), SmallVector<int64_t>{1, 256, 20, 1});
        const PerClusterShapesOffsetsVec inPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 20, 0}});

        const auto inPerClusterShapesAttr = getIntArrayOfArray(&ctx, inPerClusterShapes);
        const auto inPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterOffsets);

        const auto ovrDistributedAttr =
                vpux::VPU::DistributedTensorAttr::get(&ctx, ovrMode, inNumTiles, nullptr, nullptr, nullptr, numClusters,
                                                      nullptr, nullptr, inPerClusterShapesAttr, inPerClusterOffsetsAttr,
                                                      inPerClusterShapesAttr, inPerClusterOffsetsAttr, nullptr);

        const auto outNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));
        const auto segMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED);
        const PerClusterShapesOffsetsVec outPerClusterShapes(numClusters.getInt(), SmallVector<int64_t>{1, 20, 1, 256});
        const PerClusterShapesOffsetsVec outPerClusterOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 20, 0, 0}});

        const auto outPerClusterShapesAttr = getIntArrayOfArray(&ctx, outPerClusterShapes);
        const auto outPerClusterOffsetsAttr = getIntArrayOfArray(&ctx, outPerClusterOffsets);

        const auto segDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, segMode, outNumTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                outPerClusterShapesAttr, outPerClusterOffsetsAttr, outPerClusterShapesAttr, outPerClusterOffsetsAttr,
                nullptr);

        getTypesAndTest(ovrDistributedAttr, segDistributedAttr);
    }

    {
        const auto inNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
        const auto ovrMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters.getInt(),
                                                                   SmallVector<int64_t>{1, 256, 20, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 20, 0}});
        const auto inPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, inPerClusterComputeShapes);
        const auto inPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterComputeOffsets);

        const PerClusterShapesOffsetsVec inPerClusterMemoryShapes(
                {SmallVector<int64_t>{1, 256, 22, 1}, SmallVector<int64_t>{1, 256, 21, 1}});
        const PerClusterShapesOffsetsVec inPerClusterMemoryOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 19, 0}});

        const auto inPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, inPerClusterMemoryShapes);
        const auto inPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterMemoryOffsets);

        const auto ovrDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, ovrMode, inNumTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                inPerClusterComputeShapesAttr, inPerClusterComputeOffsetsAttr, inPerClusterMemoryShapesAttr,
                inPerClusterMemoryOffsetsAttr, nullptr);

        // output axis is C and Overlapped is not equivalent to Segmented => cannot infer distribution
        getTypesAndTest(ovrDistributedAttr, nullptr);
    }

    {
        const auto segDupMode = VPU::DistributionModeAttr::get(
                &ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
        const auto inNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 2, 1, 1}));

        const PerClusterShapesOffsetsVec inPerClusterComputeShapes(numClusters.getInt(),
                                                                   SmallVector<int64_t>{1, 64, 40, 1});
        const PerClusterShapesOffsetsVec inPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 64, 0, 0}});
        const PerClusterShapesOffsetsVec inPerMemoryClusterShapes(numClusters.getInt(),
                                                                  SmallVector<int64_t>{1, 128, 40, 1});
        const PerClusterShapesOffsetsVec inPerMemoryClusterOffsets(numClusters.getInt(),
                                                                   SmallVector<int64_t>{0, 0, 0, 0});

        const auto inPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, inPerClusterComputeShapes);
        const auto inPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, inPerClusterComputeOffsets);
        const auto inPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, inPerMemoryClusterShapes);
        const auto inPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, inPerMemoryClusterOffsets);

        const auto inDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, segDupMode, inNumTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                inPerClusterComputeShapesAttr, inPerClusterComputeOffsetsAttr, inPerClusterMemoryShapesAttr,
                inPerClusterMemoryOffsetsAttr, nullptr);

        const auto outNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 1, 2}));

        const PerClusterShapesOffsetsVec outPerClusterComputeShapes(numClusters.getInt(),
                                                                    SmallVector<int64_t>{1, 40, 1, 64});
        const PerClusterShapesOffsetsVec outPerClusterComputeOffsets(
                {SmallVector<int64_t>{0, 0, 0, 0}, SmallVector<int64_t>{0, 0, 0, 64}});
        const PerClusterShapesOffsetsVec outPerMemoryClusterShapes(numClusters.getInt(),
                                                                   SmallVector<int64_t>{1, 40, 1, 128});
        const PerClusterShapesOffsetsVec outPerMemoryClusterOffsets(numClusters.getInt(),
                                                                    SmallVector<int64_t>{0, 0, 0, 0});

        const auto outPerClusterComputeShapesAttr = getIntArrayOfArray(&ctx, outPerClusterComputeShapes);
        const auto outPerClusterComputeOffsetsAttr = getIntArrayOfArray(&ctx, outPerClusterComputeOffsets);
        const auto outPerClusterMemoryShapesAttr = getIntArrayOfArray(&ctx, outPerMemoryClusterShapes);
        const auto outPerClusterMemoryOffsetsAttr = getIntArrayOfArray(&ctx, outPerMemoryClusterOffsets);

        const auto outDistributedAttr = vpux::VPU::DistributedTensorAttr::get(
                &ctx, segDupMode, outNumTiles, nullptr, nullptr, nullptr, numClusters, nullptr, nullptr,
                outPerClusterComputeShapesAttr, outPerClusterComputeOffsetsAttr, outPerClusterMemoryShapesAttr,
                outPerClusterMemoryOffsetsAttr, nullptr);

        getTypesAndTest(inDistributedAttr, outDistributedAttr);
    }
}
