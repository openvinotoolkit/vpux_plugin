//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

namespace mlir {
/// Generates a new symbol reference attribute with a new leaf reference.
static SymbolRefAttr generateNewRefAttr(SymbolRefAttr oldAttr, SymbolRefAttr newLeafAttr) {
    if (oldAttr.isa<FlatSymbolRefAttr>()) {
        return newLeafAttr;
    }
    auto nestedRefs = llvm::to_vector<2>(oldAttr.getNestedReferences());
    nestedRefs.back() = FlatSymbolRefAttr::get(newLeafAttr.getRootReference());

    nestedRefs.append(newLeafAttr.getNestedReferences().begin(), newLeafAttr.getNestedReferences().end());

    return SymbolRefAttr::get(oldAttr.getRootReference(), nestedRefs);
}
}  // namespace mlir

using MLIR_SymbolTable = MLIR_UnitBase;

TEST_F(MLIR_SymbolTable, CheckGenerateNewRefAttr) {
    mlir::MLIRContext ctx(registry);

    llvm::StringLiteral root = "root";
    llvm::StringLiteral alpha = "alpha";
    llvm::StringLiteral beta = "beta";
    llvm::StringLiteral gamma = "gamma";
    llvm::StringLiteral theta = "theta";
    mlir::SymbolRefAttr test;

    auto rootAttr = mlir::StringAttr::get(&ctx, root);
    auto alphaAttr = mlir::FlatSymbolRefAttr::get(&ctx, alpha);
    auto betaAttr = mlir::FlatSymbolRefAttr::get(&ctx, beta);
    auto gammaAttr = mlir::FlatSymbolRefAttr::get(&ctx, gamma);
    auto thetaAttr = mlir::FlatSymbolRefAttr::get(&ctx, theta);

    test = mlir::generateNewRefAttr(alphaAttr, betaAttr);
    ASSERT_EQ(test, betaAttr);

    auto ralpha = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr});
    auto rbeta = mlir::SymbolRefAttr::get(rootAttr, {betaAttr});
    test = mlir::generateNewRefAttr(ralpha, betaAttr);
    ASSERT_EQ(test, rbeta);

    auto rabg = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, betaAttr, gammaAttr});
    auto rabt = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, betaAttr, thetaAttr});
    test = mlir::generateNewRefAttr(rabg, thetaAttr);
    ASSERT_EQ(test, rabt);

    auto rab = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, betaAttr});
    auto gt = mlir::SymbolRefAttr::get(gammaAttr.getAttr(), {thetaAttr});
    auto ragt = mlir::SymbolRefAttr::get(rootAttr, {alphaAttr, gammaAttr, thetaAttr});
    test = mlir::generateNewRefAttr(rab, gt);
    ASSERT_EQ(test, ragt);
}

TEST_F(MLIR_SymbolTable, ReplaceAllSymbolUses) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
        ELF.Main @ELFMain {

        ELF.CreateSection @taskBuff aligned(64) secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {
        }

        ELF.CreateSection @buffers aligned(64) secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {
        }

        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer<"NetworkInput" [0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer<"NetworkOutput" [0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
        VPUASM.NNDMA @DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(0) clean_after(0) descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)

        }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto elfMain = module.get().lookupSymbol<vpux::ELF::MainOp>("ELFMain");
    ASSERT_TRUE(elfMain != nullptr);

    auto taskBuffSection = elfMain.lookupSymbol<vpux::ELF::DataSectionOp>("taskBuff");
    ASSERT_TRUE(taskBuffSection != nullptr);

    auto taskBuff = elfMain.lookupSymbol<vpux::VPUASM::DeclareTaskBufferOp>("DeclareTaskBuffer_DMA_0");
    ASSERT_TRUE(taskBuff != nullptr);

    auto taskBuffRef = mlir::FlatSymbolRefAttr::get(taskBuff.getNameAttr());
    auto taskBuffName = mlir::SymbolRefAttr::get(taskBuffSection.getNameAttr(), {taskBuffRef});
    auto res = mlir::SymbolTable::replaceAllSymbolUses(taskBuff, taskBuffName, elfMain);
    taskBuff.getOperation()->moveBefore(taskBuffSection.getBody(), taskBuffSection.getBody()->end());

    ASSERT_TRUE(res.succeeded());

    auto bufferSection = elfMain.lookupSymbol<vpux::ELF::DataSectionOp>("buffers");
    ASSERT_TRUE(bufferSection != nullptr);

    mlir::SymbolRefAttr newBuffNames[2];

    for (int i = 0; i < 2; ++i) {
        std::string buffName = "DeclareBuffer" + std::to_string(i);
        auto buffer = elfMain.lookupSymbol<vpux::VPUASM::DeclareBufferOp>(buffName);
        ASSERT_TRUE(buffer != nullptr);

        auto bufferRef = mlir::FlatSymbolRefAttr::get(buffer.getNameAttr());
        newBuffNames[i] = mlir::SymbolRefAttr::get(bufferSection.getNameAttr(), {bufferRef});
        res = mlir::SymbolTable::replaceAllSymbolUses(buffer, newBuffNames[i], elfMain);
        buffer.getOperation()->moveBefore(bufferSection.getBody(), bufferSection.getBody()->end());

        ASSERT_TRUE(res.succeeded());
    }

    auto dmaOp = elfMain.lookupSymbol<vpux::VPUASM::NNDMAOp>("DMA_0_0_0");

    ASSERT_EQ(dmaOp.getTaskLocationAttr(), taskBuffName);
    ASSERT_EQ(dmaOp.getInputAttr(), newBuffNames[0]);
    ASSERT_EQ(dmaOp.getOutputBuffsAttr()[0], newBuffNames[1]);
}
