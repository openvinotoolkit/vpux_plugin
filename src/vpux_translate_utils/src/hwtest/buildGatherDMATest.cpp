//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

/*
    There 3 operations :
    Input tensor stays in DDR
    Move indices to CMX -> Do GatherDMA -> Move GatherDMAOutput to DDR (funcOutput)
*/

void buildGatherDMA(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                    Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    const auto input = testDesc.getInputLayerList().front();
    const auto dmaParams = testDesc.getDMAparams();
    const auto output = testDesc.getOutputLayers().front();
    const auto indicesDesc = testDesc.getInputGatherIndices();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());
    SmallVector<int64_t> indicesShape(indicesDesc.shape.begin(), indicesDesc.shape.end());

    if (testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX) {
        VPUX_THROW_UNLESS(dmaParams.engine == 0, "buildGatherDMA: DMA on NPU40XX should have 1 engine");
    }

    VPUX_THROW_UNLESS(!inShape.empty(), "buildGatherDMA: Input rank is 0");
    VPUX_THROW_WHEN(inShape == outShape, "buildGatherDMA: in_shape and out_shape are same, unneccessary gather");

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    const auto sectionIdx = 0;
    const auto elemType = getInt64Type(ctx);

    const auto indicesType = getMemRefType(VPURT::BufferSection::NetworkInput, sectionIdx, ShapeRef(indicesShape),
                                           elemType, DimsOrder::NHWC);

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(inType);
    inputTypes.push_back(indicesType);
    inputTypes.push_back(outType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outType);

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(),
            printToString("dma_from_{0}_{1}_to_{2}_{3}_gather", nb::to_string(dmaParams.srcLocation), inputType,
                          nb::to_string(dmaParams.dstLocation), outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());
    int barrierNumber = 0;
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    auto barrier2 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

    auto indicesArg = func.getArgument(1);
    auto inputCmxOffset = 0;
    auto indicesCmxType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, ShapeRef(indicesShape), elemType, DimsOrder::NHWC);
    auto indicesCmxTensorOp =
            createDeclareTensorOp(funcbuilder, indicesCmxType, VPURT::BufferSection::CMX_NN, 0, inputCmxOffset);
    inputCmxOffset += totalTensorSize(indicesShape, elemType);
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(),
                                                mlir::ValueRange(barrier1.getBarrier()), funcbuilder.getUnknownLoc(),
                                                indicesArg, getTensorResult(indicesCmxTensorOp), 0);

    auto outputCmxType =
            getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, ShapeRef(outShape), outputType, DimsOrder::NHWC);
    auto outputCmxTensorOp =
            createDeclareTensorOp(funcbuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, inputCmxOffset);
    inputCmxOffset += totalTensorSize(outShape, outputType);

    VPURT::wrapIntoTaskOp<VPUIP::GatherDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(barrier2.getBarrier()), builder.getUnknownLoc(),
                                              func.getArgument(0), getTensorResult(indicesCmxTensorOp),
                                              getTensorResult(outputCmxTensorOp), 0, 0, 0);

    auto funcOutput = func.getArgument(2);
    // finalBarrier passed as production barrier to last DMA task
    auto barrier3 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier2.getBarrier()),
                                                mlir::ValueRange(barrier3.getBarrier()), funcbuilder.getUnknownLoc(),
                                                getTensorResult(outputCmxTensorOp), funcOutput, 0);

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutput);

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
    buildCNNOp(builder, func.getName(),
               {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(indicesShape), indicesType.getElementType(), DimsOrder::NHWC, nullptr)},
               getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr));
}

}  // namespace hwtest
}  // namespace vpux
