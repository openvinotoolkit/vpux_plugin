//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/init.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

void testDType(mlir::MLIRContext* ctx, VPU::ClusteredOpInterface clusteredOp,
               VPU::DistributedTensorAttr expectedDistributedAttr, mlir::IntegerAttr numClusters, bool isAct,
               NDTypeInterface tiledInput = nullptr, NDTypeInterface tiledOutput = nullptr) {
    auto inputType = tiledInput != nullptr ? tiledInput : clusteredOp->getOperand(0).getType().cast<NDTypeInterface>();
    auto outputType =
            tiledOutput != nullptr ? tiledOutput : clusteredOp->getResult(0).getType().cast<NDTypeInterface>();

    auto distributedIf =
            isAct ? VPU::getDistributedActivationTypeFromOp(clusteredOp, inputType, numClusters, outputType)
                  : VPU::getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters, inputType);
    auto distributedType = distributedIf.getDistributedTypes().front().cast<VPU::DistributedTensorType>();

    const auto memSpace = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    auto order = isAct ? mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx))
                       : mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(ctx));
    auto expectedType =
            isAct ? VPU::DistributedTensorType::get(ctx, inputType.getShape().raw(), inputType.getElementType(), order,
                                                    memSpace, expectedDistributedAttr)
                  : VPU::DistributedTensorType::get(ctx, outputType.getShape().raw(), outputType.getElementType(),
                                                    order, memSpace, expectedDistributedAttr);

    EXPECT_EQ(distributedType, expectedType);
}

using MLIR_GetDistributedTypeFromOpSOKAlignmentTest = MLIR_UnitBase;

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            IE.TileResource 3 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
                %cst2 = const.Declare tensor<144x16x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x16x1x1xf16, {order = #NHWC}>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPETask<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64>,
                    rawFilterShape = [144, 144, 1, 1],
                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %1 = VPU.MVN(%0) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %2 = VPU.NCE.DepthConvolution(%1, %cst2, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPETask<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64>,
                    rawFilterShape = [144, 1, 1, 1],
                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                return %2 : tensor<1x144x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    const vpux::Shape offsets({0, 0, 0, 0});
    const vpux::Shape size({1, 50, 16, 16});

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedDistribution = VPU::DistributedTensorAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr);
    auto expectedTiledDistribution = VPU::DistributedTensorAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr);

    func.walk([&](VPU::SWOpInterface op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type

        auto inputType = clusteredOp->getOperand(0).getType().cast<NDTypeInterface>();
        auto outputType = clusteredOp->getResult(0).getType().cast<NDTypeInterface>();

        const auto inputTileType = inputType.extractDenseTile(offsets, size);
        const auto outputTileType = outputType.extractDenseTile(offsets, size);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, true, inputTileType,
                  outputTileType);  // test tiled activation distributed type
        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, false, inputTileType,
                  outputTileType);  // test tiled output distributed type
    });
}

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentAfterSlice1) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            IE.TileResource 3 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<128x128x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPETask<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64>,
                    rawFilterShape = [128, 128, 1, 1],
                    strides = [1, 1]}
                        -> tensor<1x128x16x16xf16, {order = #NHWC}>
                %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 64, 16, 16]
                    : tensor<1x128x16x16xf16, {order = #NHWC}> to tensor<1x64x16x16xf16, {order = #NHWC}>
                %2 = VPU.MVN(%1) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x64x16x16xf16, {order = #NHWC}>
                        -> tensor<1x64x16x16xf16, {order = #NHWC}>
                %3 = VPU.HSwish(%2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                    : tensor<1x64x16x16xf16, {order = #NHWC}>
                        -> tensor<1x64x16x16xf16, {order = #NHWC}>
                return %3 : tensor<1x64x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);
    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));

    for (auto& op : func.getOps()) {
        if (mlir::isa<VPU::SWOpInterface>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            auto expectedDistribution = VPU::DistributedTensorAttr::get(
                    &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
                    nullptr, nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr,
                    nullptr, nullptr, nullptr);

            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        }

        // In the above subgraph, there will always be a spill due to the Slice over K after the Conv.
        // Therefore, it would be better to have SEGMENTED | DUPLICATED @ Conv output. That does not happen currently.
        // Leaving this test here to track behaviour of this scenario.
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            auto expectedDistribution = VPU::DistributedTensorAttr::get(
                    &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
                    nullptr, nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr,
                    nullptr, nullptr, nullptr);

            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        }
    }
}

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentAfterSlice2) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            IE.TileResource 3 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x160x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<160x160x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<160x160x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<160x1x1x4xsi32> = dense<1> : tensor<160x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPETask<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64>,
                    rawFilterShape = [160, 160, 1, 1],
                    strides = [1, 1]}
                        -> tensor<1x160x16x16xf16, {order = #NHWC}>
                %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 144, 16, 16]
                    : tensor<1x160x16x16xf16, {order = #NHWC}> to tensor<1x144x16x16xf16, {order = #NHWC}>
                %2 = VPU.MVN(%1) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %3 = VPU.HSwish(%2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                return %3 : tensor<1x144x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedAlignedDistribution = VPU::DistributedTensorAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr);

    for (auto& op : func.getOps()) {
        if (mlir::isa<VPU::MVNOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      true);  // test activation distributed type

            // At the moment if the SW op's input has alignment, the output should have alignment to avoid invaild
            // tiling
            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      false);  // test output distributed type
        }

        if (mlir::isa<VPU::HSwishOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      true);  // test activation distributed type
            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      false);  // test output distributed type
        }

        // In the above subgraph, there will always be a spill due to the Slice over K after the Conv.
        // Therefore, it would be better to have SEGMENTED | DUPLICATED @ Conv output. That does not happen currently.
        // Leaving this test here to track behaviour of this scenario.
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      false);  // test output distributed type
        }
    }
}

struct DistributedTypeFromSOKOpParams {
    llvm::StringLiteral inputIR;
    bool isSwOpOutputDistributionAligned;
};

class GetDistributedTypeFromSOKOpTests : public testing::TestWithParam<DistributedTypeFromSOKOpParams> {};

TEST_P(GetDistributedTypeFromSOKOpTests, SegmentedOverChannelsDistribution) {
    const auto params = GetParam();
    const llvm::StringLiteral inputIR = params.inputIR;
    const bool isSwOpOutputDistributionAligned = params.isSwOpOutputDistributionAligned;

    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);
    vpux::registerCommonInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedAlignedDistribution = VPU::DistributedTensorAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr);

    auto expectedUnalignedDistribution = VPU::DistributedTensorAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, /*alignment*/ nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr);

    auto expectedSwOpDistribution =
            isSwOpOutputDistributionAligned ? expectedAlignedDistribution : expectedUnalignedDistribution;

    func.walk([&](VPU::ClusteredOpInterface op) {
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            testDType(&ctx, op, expectedAlignedDistribution, numClusters, false);  // test output distributed type
        }
        if (mlir::isa<VPU::NCEAveragePoolOp>(op)) {
            testDType(&ctx, op, expectedAlignedDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, op, expectedAlignedDistribution, numClusters, false);  // test output distributed type
        }
        if (mlir::isa<VPU::SWOpInterface>(op.getOperation())) {
            testDType(&ctx, op, expectedSwOpDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, op, expectedSwOpDistribution, numClusters, false);  // test output distributed type
        }
    });
}

// clang-format off

std::vector<DistributedTypeFromSOKOpParams> verticalFusionWrappingParams = {
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        IE.TileResource 3 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}>
        {
            %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
            %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
            %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x144x16x16xf16, {order = #NHWC}>,
                                     %cst0 as %arg2: tensor<144x144x1x1xf16, {order = #NHWC}>,
                                     %cst1 as %arg3: tensor<144x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2,
                                     1]}
                -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPETask<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64>,
                    rawFilterShape = [144, 144, 1, 1],
                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                VPU.Yield %0
            }
            %1 = VPU.VerticalFusion (%0 as %arg4: tensor<1x144x16x16xf16, {order = #NHWC}>)
                attributes {tilingStrategy = [1, 1, 2, 1]}
                -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %2 = VPU.HSwish(%arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                VPU.Yield %2
            }
            return %1 : tensor<1x144x16x16xf16, {order = #NHWC}>
        }
    })", false},
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        IE.TileResource 3 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}>
        {
            %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
            %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
            %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x144x16x16xf16, {order = #NHWC}>,
                                     %cst0 as %arg2: tensor<144x144x1x1xf16, {order = #NHWC}>,
                                     %cst1 as %arg3: tensor<144x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2,
                                     1]}
                -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPETask<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64>,
                    rawFilterShape = [144, 144, 1, 1],
                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                VPU.Yield %0
            }
            %1 = VPU.MVN(%0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            return %1 : tensor<1x144x16x16xf16, {order = #NHWC}>
        }
    })", false}};

std::vector<DistributedTypeFromSOKOpParams> segmentedAvgPoolParams = {
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        IE.TileResource 3 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x8x16xf16, {order = #NHWC}> {
            %0 = VPU.MVN(%arg0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 144, 8, 16]
                : tensor<1x144x16x16xf16, {order = #NHWC}> to tensor<1x144x8x16xf16, {order = #NHWC}>
            %2 = VPU.NCE.AveragePool(%1) {
                kernel_size = [1, 1],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LPRELU>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1311 : i64, lrelu_shift = 17 : i64,
                quant_scale = [1.000000e+00],
                fp_prelu_alpha = 0.01000213623046875 : f64>,
                strides = [1, 1]}
                    -> tensor<1x144x8x16xf16, {order = #NHWC}>
            return %2 : tensor<1x144x8x16xf16, {order = #NHWC}>
        }
    })", true},
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        IE.TileResource 3 of @NCE at 6.000000e+02 MHz
        func.func @main(%arg0: tensor<1x48x16x16xf16, {order = #NHWC}>)
            -> (tensor<1x96x8x16xf16, {order = #NHWC}>, tensor<1x96x8x16xf16, {order = #NHWC}>) {
            %0 = VPU.MVN(%arg0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x48x16x16xf16, {order = #NHWC}>
                    -> tensor<1x48x16x16xf16, {order = #NHWC}>
            %1 = VPU.MVN(%arg0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x48x16x16xf16, {order = #NHWC}>
                    -> tensor<1x48x16x16xf16, {order = #NHWC}>
            %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
                : tensor<1x48x16x16xf16, {order = #NHWC}>, tensor<1x48x16x16xf16, {order = #NHWC}>
                -> tensor<1x96x16x16xf16, {order = #NHWC}>
            %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 96, 8, 16]
                : tensor<1x96x16x16xf16, {order = #NHWC}> to tensor<1x96x8x16xf16, {order = #NHWC}>
            %4 = VPU.NCE.AveragePool(%3) {
                kernel_size = [1, 1],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LPRELU>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1311 : i64, lrelu_shift = 17 : i64,
                quant_scale = [1.000000e+00],
                fp_prelu_alpha = 0.01000213623046875 : f64>,
                strides = [1, 1]}
                    -> tensor<1x96x8x16xf16, {order = #NHWC}>
            %5 = VPU.Slice %2 [0, 0, 8, 0] [1, 96, 8, 16]
                : tensor<1x96x16x16xf16, {order = #NHWC}> to tensor<1x96x8x16xf16, {order = #NHWC}>
            %6 = VPU.NCE.AveragePool(%5) {
                kernel_size = [1, 1],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPETask<mode = <LPRELU>,
                clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                lrelu_mult = 1311 : i64, lrelu_shift = 17 : i64,
                quant_scale = [1.000000e+00],
                fp_prelu_alpha = 0.01000213623046875 : f64>,
                strides = [1, 1]}
                    -> tensor<1x96x8x16xf16, {order = #NHWC}>
            return %4, %6 : tensor<1x96x8x16xf16, {order = #NHWC}>, tensor<1x96x8x16xf16, {order = #NHWC}>
        }
    })", true}
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(VFWrappedSOKConvWithSOKSWOpConsumer, GetDistributedTypeFromSOKOpTests,
                         testing::ValuesIn(verticalFusionWrappingParams));
INSTANTIATE_TEST_SUITE_P(SOKNCEAvgPoolWithSOKSWOpProducer, GetDistributedTypeFromSOKOpTests,
                         testing::ValuesIn(segmentedAvgPoolParams));
