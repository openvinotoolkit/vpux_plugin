//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace hwtest {

void buildReadAfterWriteDMAACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto loc = builder.getUnknownLoc();

    const auto input = testDesc.getInputLayerList().front();
    const auto output = testDesc.getOutputLayers().front();
    const auto iterationCount = testDesc.getIterationCount();
    const auto cluster = testDesc.getClusterNumber();

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildReadAfterWriteDMAACTTest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildReadAfterWriteDMAACTTest: Got empty outputShape");

    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputShape, inputType);
    const auto outputDMACMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);
    const auto outputACTCMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);

    const auto rewritable_bytes = 1;
    const auto INPUT_CMX_OFFSET = 0;
    auto OUTPUT_ACT_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize - rewritable_bytes;
    auto OUTPUT_DMA_CMX_OFFSET = OUTPUT_ACT_CMX_OFFSET + outputACTCMXSize;

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);
    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(inputParamType);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("read_after_write_dma_act_{0}_{1}", inputType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    SmallVector<vpux::VPURT::DeclareBufferOp> inputCMXVec;
    inputCMXVec.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                                DimsOrder::NHWC, cluster, INPUT_CMX_OFFSET));
    auto outputDMACMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                              DimsOrder::NHWC, cluster, OUTPUT_DMA_CMX_OFFSET);
    auto outputACTCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                              DimsOrder::NHWC, cluster, OUTPUT_ACT_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, functionInput, inputCMXVec[0].getOperation()->getResult(0), 0);

    auto waitBarrier = updateBarrier;
    auto startIter = freeBarrierId++;
    for (std::size_t i = startIter; i + 1 < iterationCount; i += 2) {
        if (i != startIter) {
            inputCMXVec[0] = outputDMACMX;
            OUTPUT_ACT_CMX_OFFSET = OUTPUT_DMA_CMX_OFFSET + outputDMACMXSize - rewritable_bytes;
            outputACTCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                                 DimsOrder::NHWC, cluster, OUTPUT_ACT_CMX_OFFSET);

            OUTPUT_DMA_CMX_OFFSET = OUTPUT_ACT_CMX_OFFSET + outputACTCMXSize;
            outputDMACMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                                 DimsOrder::NHWC, cluster, OUTPUT_DMA_CMX_OFFSET);
        }
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                              mlir::ValueRange(updateBarrier.getBarrier()), loc,
                                              inputCMXVec[0].getOperation()->getResult(0),
                                              outputDMACMX.getOperation()->getResult(0), 0,
                                              testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX ? 0 : cluster);

        waitBarrier = updateBarrier;
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i + 1);

        buildActShaveTask(testDesc, module, functionBuilder, log, ArrayRef(inputTypes), inputCMXVec, outputACTCMX,
                          /*profilingDataCMX*/ nullptr, mlir::ValueRange(waitBarrier.getBarrier()),
                          mlir::ValueRange(updateBarrier.getBarrier()), cluster);

        waitBarrier = updateBarrier;
    }

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, iterationCount, testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputDMACMX.getOperation()->getResult(0), functionOutput, 0);

    functionBuilder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{functionOutput});

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.numberOfDMAPorts = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
