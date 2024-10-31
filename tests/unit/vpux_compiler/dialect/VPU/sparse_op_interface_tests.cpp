//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using vpux::VPU::ArchKind;

void testSparsitySupport(llvm::StringLiteral inputIR, ArchKind arch, bool supportInputSparsity,
                         bool supportOutputSparsity, bool supportWeightSparsity) {
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = vpux::VPU::InitCompilerOptions(arch, vpux::VPU::CompilationMode::DefaultHW);

    vpux::VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto sparseOp = mlir::dyn_cast<vpux::VPU::SparseOpInterface>(op)) {
            ASSERT_EQ(vpux::VPU::supportsSparseInputs(&op), supportInputSparsity);
            ASSERT_EQ(vpux::VPU::supportsSparseOutputs(&op), supportOutputSparsity);
            ASSERT_EQ(vpux::VPU::supportsSparseWeights(&op), supportWeightSparsity);
        }
    }
}

TEST(MLIR_VPU_Sparsity, NCEZMajorConvSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
                %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
                %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                        opaque_ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        rawFilterShape = [16, 16, 1, 1],
                        strides = [1, 1]
                    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

                return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::NPU37XX, /*input=*/true, /*output=*/true, /*weights=*/true);
}

TEST(MLIR_VPU_Sparsity, NCEEltwiseSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Eltwise(%arg0, %arg0) {op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x16x16x16xf16, {order = #NHWC}>
                return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::NPU37XX, /*input=*/false, /*output=*/true, /*weights=*/false);
}

TEST(MLIR_VPU_Sparsity, NCEDepthconvSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func.func @main(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
                    dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]
                %cst1 = const.Declare tensor<16x1x1x4xsi32> =
                    dense<1> : tensor<16x1x1x4xsi32>

                %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %cst1) {
                        opaque_ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                        rawFilterShape = [16, 1, 4, 8],
                        strides = [1, 1]
                    } -> tensor<1x16x37x73xf16, {order = #NHWC}>
                return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::NPU37XX, /*input=*/false, /*output=*/true, /*weights=*/false);
}

TEST(MLIR_VPU_Sparsity, NCEMaxpoolSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func.func @main(%arg0: tensor<16x16x16x16xf16, {order = #NHWC}>) -> tensor<16x16x16x16xf16, {order = #NHWC}> {
                %0 = VPU.MaxPool(%arg0) {
                    kernel_size = [3, 3],
                    opaque_ppe = #VPU.PPEStub<>,
                    pads_begin = [1, 1],
                    pads_end = [1, 1],
                    rounding_type = #IE.rounding_type<FLOOR>,
                    strides = [1, 1]
                } : tensor<16x16x16x16xf16, {order = #NHWC}> -> tensor<16x16x16x16xf16, {order = #NHWC}>
                return %0 : tensor<16x16x16x16xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::NPU37XX, /*input=*/false, /*output=*/true, /*weights=*/false);
}

TEST(MLIR_VPU_Sparsity, NCEAvgpoolSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func.func @main(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
                %0 = VPU.NCE.AveragePool(%arg0) {
                        kernel_size = [3, 3],
                        opaque_ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
                        strides = [1, 1]
                    } -> tensor<1x16x4x4xf16, {order = #NHWC}>

                return %0 : tensor<1x16x4x4xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::NPU37XX, /*input=*/false, /*output=*/true, /*weights=*/false);
}
