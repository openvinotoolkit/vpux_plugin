//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildRaceConditionDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayerList().front();
    auto output = testDesc.getOutputLayers().front();
    auto iterationCount = testDesc.getIterationCount();
    const auto numClusters = testDesc.getNumClusters();

    // set runtime resources
    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = numClusters;
    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildRaceConditionDMATest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildRaceConditionDMATest: Got empty outputShape");

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    SmallVector<mlir::Type> inputTypes(numClusters, outType);
    inputTypes.insert(inputTypes.begin(), inType);

    SmallVector<mlir::Type> outputTypes(numClusters, outType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), ArrayRef(outputTypes));

    auto func = builder.create<mlir::func::FuncOp>(
            loc, printToString("race_condition_dma_{0}_{1}", inputType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcInput = func.getArgument(0);
    SmallVector<mlir::BlockArgument> funcOutputs;

    for (unsigned int idx = 1; idx <= static_cast<unsigned int>(numClusters); ++idx) {
        funcOutputs.push_back(func.getArgument(idx));
    }

    SmallVector<mlir::MemRefType> outputCMXTypes;
    SmallVector<VPURT::DeclareBufferOp> outputs;

    for (std::size_t idx = 0; idx < numClusters; ++idx) {
        outputCMXTypes.push_back(
                getMemRefType(VPURT::BufferSection::CMX_NN, idx, outShape, outputType, DimsOrder::NHWC));
        outputs.push_back(funcBuilder.create<VPURT::DeclareBufferOp>(
                loc, outputCMXTypes[idx], VPURT::BufferSection::CMX_NN, idx, /*byteOffset=*/0));
    }

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    mlir::Value lastBarrier;

    auto startIter = freeBarrierId++;
    for (std::size_t i = startIter; i < iterationCount - 1; ++i) {
        auto updateBarrier = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, i).getBarrier();

        for (std::size_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx += 2) {
            vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder,
                                                        i == startIter ? mlir::ValueRange(waitWLMBarrier) : lastBarrier,
                                                        mlir::ValueRange(updateBarrier), loc, funcInput,
                                                        outputs[clusterIdx].getOperation()->getResult(0), 0);

            if (clusterIdx + 1 < numClusters) {
                vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                        funcBuilder, i == startIter ? mlir::ValueRange(waitWLMBarrier) : lastBarrier, updateBarrier,
                        loc, funcInput, outputs[clusterIdx + 1].getOperation()->getResult(0),
                        testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX ? 0 : 1);
            }
        }

        lastBarrier = updateBarrier;
    }

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcBuilder
                                .create<vpux::VPURT::ConfigureBarrierOp>(loc, iterationCount - 1,
                                                                         testDesc.getWLMParams().isWLMPartialEnabled)
                                .getBarrier();

    for (std::size_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx += 2) {
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, lastBarrier, finalBarrier, loc,
                                                    outputs[clusterIdx].getOperation()->getResult(0),
                                                    funcOutputs[clusterIdx], 0);

        if (clusterIdx + 1 < numClusters) {
            vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                    funcBuilder, lastBarrier, finalBarrier, loc, outputs[clusterIdx].getOperation()->getResult(0),
                    funcOutputs[clusterIdx + 1], testDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX ? 0 : 1);
        }
    }

    auto outputsRef = ArrayRef(funcOutputs);
    funcBuilder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange(outputsRef));

    SmallVector<mlir::Type> userOutputs(numClusters,
                                        getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr));
    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               ArrayRef(userOutputs));
}

}  // namespace hwtest
}  // namespace vpux
