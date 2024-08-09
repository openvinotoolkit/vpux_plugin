//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"

using namespace vpux;

namespace vpux {
namespace hwtest {

namespace {

using opBuilderCallback = std::function<void(const nb::TestCaseJsonDescriptor&, mlir::ModuleOp, mlir::OpBuilder,
                                             Logger&, ArrayRef<mlir::Type>, SmallVector<vpux::VPURT::DeclareBufferOp>&,
                                             vpux::VPURT::DeclareBufferOp, vpux::VPURT::DeclareBufferOp,
                                             mlir::ValueRange, mlir::ValueRange, size_t, size_t)>;
opBuilderCallback getOpBuilder(nb::CaseType caseType) {
    switch (caseType) {
    case nb::CaseType::ActShave:
        return &buildActShaveTask;
    default:
        VPUX_THROW("ActShave case is only supported for generic RaceCondition");
    }
}

}  // namespace

/*
//                       [input]
//           (dma to CMX0)     (dma to CMX1)
//                |                 |
//              /   \             /   \
//            |       |         |       |
//          (op)    (op)       (op)    (op)
//            |       |         |       |
//          (op)    (op)       (op)    (op)
//            |       |         |       |
//        ... (loop with conv ops and barriers)
//            |       |         |       |
//          (op)    (op)       (op)    (op)
//            |       |         |       |
//       [output0]  [output1] [output2]  [output3]
*/

void buildRaceConditionTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto raceConditionParams = testDesc.getRaceConditionParams();
    auto testDescUnderlyingOp = testDesc.getUnderlyingOp();
    VPUX_THROW_WHEN(testDescUnderlyingOp == nullptr, "underlyingOp is nullptr for CaseType::RaceCondition");

    auto underlyingOpBuilder = getOpBuilder(testDescUnderlyingOp->getCaseType());

    auto input = testDescUnderlyingOp->getInputLayerList().front();
    auto output = testDescUnderlyingOp->getOutputLayers().front();
    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    SmallVector<mlir::Type> inputTypes;
    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    inputTypes.push_back(inputParamType);
    SmallVector<mlir::Type> outputTypes;
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    const auto outputsCount = raceConditionParams.requestedClusters * raceConditionParams.requestedUnits;
    inputTypes.insert(inputTypes.end(), outputsCount, outputParamType);
    outputTypes.insert(outputTypes.end(), outputsCount, outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), ArrayRef(outputTypes));

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(),
            printToString("race_condition_{0}_{1}_{2}", testDescUnderlyingOp->getCaseStr(), inputType, outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    auto funcInput = func.getArgument(0);

    const auto inputCMXOffset = 0;
    const auto outputCMXOffset = inputCMXOffset + totalTensorSize(inShape, inputType);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::ConfigureBarrierOp waitBarrier;
    size_t barrierNumber = freeBarrierId++;
    SmallVector<mlir::Value> waitBarriers;
    SmallVector<vpux::VPURT::DeclareBufferOp> outputs;
    SmallVector<mlir::Type> cnnOpOutputs;
    for (size_t cluster = 0; cluster < raceConditionParams.requestedClusters; ++cluster) {
        auto inputDataDMA = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), barrierNumber++);

        auto inputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, cluster, inShape, inputType, DimsOrder::NHWC);
        auto outputCMXType =
                getMemRefType(VPURT::BufferSection::CMX_NN, cluster, outShape, outputType, DimsOrder::NHWC);

        SmallVector<vpux::VPURT::DeclareBufferOp> inputCMXVec;
        inputCMXVec.push_back(createDeclareTensorOp(funcBuilder, inputCMXType, VPURT::BufferSection::CMX_NN,
                                                    static_cast<int>(cluster), inputCMXOffset));

        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                funcBuilder, waitWLMBarrier, mlir::ValueRange(inputDataDMA.getBarrier()), builder.getUnknownLoc(),
                funcInput, getTensorResult(inputCMXVec[0]), 0,
                testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX ? 0 : cluster);

        auto localOutputCMXOffset = outputCMXOffset;
        for (size_t unit = 0; unit < raceConditionParams.requestedUnits; ++unit) {
            auto outputCMX = createDeclareTensorOp(funcBuilder, outputCMXType, VPURT::BufferSection::CMX_NN,
                                                   static_cast<int>(cluster), localOutputCMXOffset);
            outputs.emplace_back(outputCMX);
            cnnOpOutputs.emplace_back(getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr));
            localOutputCMXOffset += totalTensorSize(outShape, outputType);
            for (size_t iter = 0; iter < raceConditionParams.iterationsCount; ++iter) {
                auto updateBarrier =
                        funcBuilder.create<VPURT::ConfigureBarrierOp>(funcBuilder.getUnknownLoc(), barrierNumber++);
                underlyingOpBuilder(*testDescUnderlyingOp, module, funcBuilder, log, inputTypes, inputCMXVec, outputCMX,
                                    /* profilingData */ nullptr,
                                    iter == 0 ? mlir::ValueRange(inputDataDMA.getBarrier())
                                              : mlir::ValueRange(waitBarrier.getBarrier()),
                                    mlir::ValueRange(updateBarrier.getBarrier()), cluster, unit);
                waitBarrier = updateBarrier;
            }
            waitBarriers.emplace_back(waitBarrier.getBarrier());
        }
    }

    SmallVector<mlir::BlockArgument> returnOps;
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcBuilder.create<VPURT::ConfigureBarrierOp>(funcBuilder.getUnknownLoc(), barrierNumber++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);
    for (size_t i = 0; i < outputs.size(); ++i) {
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(llvm::ArrayRef(waitBarriers)),
                                                    mlir::ValueRange(finalBarrier.getBarrier()),
                                                    builder.getUnknownLoc(), getTensorResult(outputs[i]),
                                                    func.getArgument(static_cast<unsigned int>(i + 1)), 0);
        returnOps.emplace_back(func.getArgument(static_cast<unsigned int>(i + 1)));
    }

    funcBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                             mlir::ValueRange(ArrayRef<mlir::BlockArgument>(returnOps)));

    //  Pass Manager
    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::ReferenceHW);
    initCompilerOptions.numberOfDPUGroups = static_cast<int>(raceConditionParams.requestedClusters);
    initCompilerOptions.numberOfDMAPorts = static_cast<int>(raceConditionParams.requestedClusters);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    //  CNN Operation
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               cnnOpOutputs);
}

}  // namespace hwtest
}  // namespace vpux
