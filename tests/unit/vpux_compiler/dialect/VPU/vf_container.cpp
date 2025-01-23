//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using vpux::VPU::ArchKind;
using namespace vpux;

using MLIR_VPU_VFPipelineContainer = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_VFPipelineContainer, VF_ContainerCost) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
       func.func @main(%arg0: tensor<1x48x16x16xf16, {order = #NHWC}>) -> tensor<1x1024x16x16xf16, {order = #NHWC}> {
            %cst = const.Declare tensor<1024x48x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1024x48x1x1xf16>, [#const.Reorder<#NHWC>]
            %cst_0 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>

            %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x48x16x16xf16, {order = #NHWC}>, %cst as %arg3: tensor<1024x48x1x1xf16, {order = #NHWC}>,
            %cst_0 as %arg4: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 2]}
                -> tensor<1x1024x16x16xf16, {order = #NHWC}> {
            %1 = VPU.NCE.Convolution(%arg2, %arg3, %arg4)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [1024, 48, 1, 1], strides = [1, 1]}
                -> tensor<1x1024x16x16xf16, {order = #NHWC}>
            %2 = VPU.SoftMax(%1)
                {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x16x16xf16, {order = #NHWC}>
                -> tensor<1x1024x16x16xf16, {order = #NHWC}>
                VPU.Yield %2
            }
            return %0 : tensor<1x1024x16x16xf16, {order = #NHWC}>
       }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU40XX, VPU::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto container = VPU::VFPipelineContainer();
    auto layerCost = std::make_unique<VPU::LayerVPUNNCost>(func);

    auto operationStorage = std::make_unique<VPU::TilingOperationStorage>();

    func->walk([&](VPU::VerticalFusionOp vfOp) {
        restoreTilingRegions(vfOp, vpux::Logger::global(), operationStorage);
    });

    int index = 0;
    VPU::StrategyCost summary = 0;
    VPU::StrategyCost weightsCost = 0;
    func->walk([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface, VPU::SWOpInterface>(op)) {
            return;
        }

        auto tilingInfo = operationStorage->get(op, index);

        VPU::VPUNNCostParameters params(VPU::MultiClusterStrategy::Clustering, {tilingInfo.value().second},
                                        TilingMode::ISOLATED, {tilingInfo.value().first.tiles}, false);

        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            if (nceOp.getWeightsOperand() != nullptr) {
                weightsCost = layerCost->getSpillingReadCost(op, params, nceOp.getWeightsOperand());
                EXPECT_TRUE(container.addDMA(index, weightsCost));
                EXPECT_FALSE(container.addDMA(index, 0));
                summary += weightsCost;
            }
        }
        auto operCost = layerCost->getStrategyCost(op, params);
        EXPECT_TRUE(container.addOperation(op, index, operCost));
        summary += operCost;
    });

    EXPECT_EQ(container.maxCost(), summary);
    EXPECT_EQ(container.getPrefetchAvailability(), summary - weightsCost);
}
