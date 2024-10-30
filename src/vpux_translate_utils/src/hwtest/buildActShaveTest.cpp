//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

using namespace vpux;

namespace vpux {
namespace hwtest {

void buildActShave(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, const SmallVector<mlir::Type>& inputTypes, mlir::Type outputType) {
    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::ReferenceHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.numberOfDMAPorts = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    auto* ctx = builder.getContext();

    //  Input/Output -----------------------------------------------------------
    auto inputList = testDesc.getInputLayerList();
    auto output = testDesc.getOutputLayers().front();
    auto profilingParams = testDesc.getProfilingParams();

    SmallVector<SmallVector<int64_t>> inShapes;
    SmallVector<mlir::Type> funcInputTypes;
    SmallVector<mlir::Type> funcOutputTypes;

    const auto profOutputTypeUI64 = getUInt64Type(ctx);
    SmallVector<int64_t> profDataShapeUI64{HWP_ACTSHAVE_BYTES_PER_ENTRY / 8};

    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        SmallVector<int64_t> inShape(inputList[idx].shape.begin(), inputList[idx].shape.end());
        VPUX_THROW_UNLESS(!inShape.empty(), "buildActShave: Got empty input '{0}' shape ", idx);
        inShapes.push_back(inShape);

        auto inputParamType =
                getMemRefType(VPURT::BufferSection::NetworkInput, inShapes[idx], inputTypes[idx], DimsOrder::NHWC);
        funcInputTypes.push_back(inputParamType);
    }

    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());
    VPUX_THROW_UNLESS(!outShape.empty(), "buildActShave: Got empty outputShape");

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    funcInputTypes.push_back(outputParamType);
    funcOutputTypes.push_back(outputParamType);

    if (profilingParams.profilingEnabled()) {
        auto profParamType = getMemRefType(VPURT::BufferSection::ProfilingOutput, profDataShapeUI64, profOutputTypeUI64,
                                           DimsOrder::C);
        funcInputTypes.push_back(profParamType);
        funcOutputTypes.push_back(profParamType);
    }
    // Build Function ---------------------------------------------------------------

    const auto funcType = builder.getFunctionType(ArrayRef(funcInputTypes), ArrayRef(funcOutputTypes));

    std::string funcOpName = "actshave_";
    for (auto iType : funcInputTypes) {
        funcOpName += printToString("{0}", iType);
    }
    funcOpName += printToString("{0}", outputType);

    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcOpName, funcType,
                                                   builder.getStringAttr("private"), /*arg_attrs=*/nullptr,
                                                   /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    SmallVector<mlir::Value> funcinputs;
    for (unsigned int idx = 0; idx < static_cast<unsigned int>(inputList.size()); idx++) {
        auto funcinput = func.getArgument(idx);
        funcinputs.push_back(funcinput);
    }
    auto funcoutput = func.getArgument(static_cast<unsigned int>(inputList.size()));
    auto profoutput = profilingParams.profilingEnabled()
                              ? func.getArgument(static_cast<unsigned int>(inputList.size()) + 1)
                              : nullptr;

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    //  Build main function: barriers
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    //  Build main function: input/output cmx
    SmallVector<vpux::VPURT::DeclareBufferOp> inputCmxVec;
    size_t inputCmxOffset = 0;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        auto inputCmxType =
                getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShapes[idx], inputTypes[idx], DimsOrder::NHWC);
        inputCmxVec.push_back(
                createDeclareTensorOp(funcbuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, inputCmxOffset));
        inputCmxOffset += totalTensorSize(inShapes[idx], inputTypes[idx]);
    }

    const auto outputCmxOffset = inputCmxOffset;
    auto outputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    vpux::VPURT::DeclareBufferOp outputCmx =
            createDeclareTensorOp(funcbuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, outputCmxOffset);

    VPURT::DeclareBufferOp profOutputCmx;
    VPURT::DeclareBufferOp profOutputDdr;
    if (profilingParams.profilingEnabled()) {
        const auto profCmxOffset = outputCmxOffset + totalTensorSize(outShape, outputType);
        SmallVector<int64_t> profoutputcmxShapeUI32 = {HWP_ACTSHAVE_BYTES_PER_ENTRY / 4};
        auto profOutputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, profoutputcmxShapeUI32,
                                               getUInt32Type(ctx), DimsOrder::C);
        profOutputCmx =
                createDeclareTensorOp(funcbuilder, profOutputCmxType, VPURT::BufferSection::CMX_NN, 0, profCmxOffset);
        profOutputDdr = createDeclareTensorOp(funcbuilder,
                                              getMemRefType(VPURT::BufferSection::ProfilingOutput,
                                                            profoutputcmxShapeUI32, getUInt32Type(ctx), DimsOrder::C),
                                              VPURT::BufferSection::ProfilingOutput, 0, 0);
    }

    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        //  Build main function: DMA func input -> CMX input
        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier,
                                                    mlir::ValueRange(barrier0.getBarrier()), builder.getUnknownLoc(),
                                                    funcinputs[idx], getTensorResult(inputCmxVec[idx]), 0);
    }

    //  Build main function: Call operation builder
    buildActShaveTask(testDesc, module, funcbuilder, log, ArrayRef(funcInputTypes), inputCmxVec, outputCmx,
                      profOutputCmx, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()));

    //  Build main function: DMA CMX output -> func output
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                                mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                                getTensorResult(outputCmx), funcoutput, 0);

    if (profilingParams.swProfilingEnabled) {
        // copy profiling data into DDR
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              getTensorResult(profOutputCmx), getTensorResult(profOutputDdr), 0);
    }

    //  Build main function: returnOp
    mlir::SmallVector<mlir::Value> funcOutputs;
    funcOutputs.push_back(funcoutput);
    if (profilingParams.profilingEnabled()) {
        funcOutputs.push_back(profoutput);
    }
    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutputs);

    SmallVector<mlir::Type> inputTensorTypeVec;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        auto inputTensorType = getTensorType(ShapeRef(inShapes[idx]), inputTypes[idx], DimsOrder::NHWC, nullptr);
        inputTensorTypeVec.push_back(inputTensorType);
    }
    auto outputTensorType = getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr);

    //  CNN Operation

    mlir::SmallVector<ProfilingDataSection> profilingDataSections;
    if (profilingParams.profilingEnabled()) {
        size_t offset = 0;
        if (profilingParams.swProfilingEnabled) {
            profilingDataSections.push_back({HWP_SW_SECTION_EXEC_TYPE, offset, HWP_ACTSHAVE_BYTES_PER_ENTRY});
            offset += HWP_ACTSHAVE_BYTES_PER_ENTRY;
        }
    }
    buildCNNOp(builder, func.getName(), inputTensorTypeVec, outputTensorType, profilingDataSections);
}  // namespace hwtest

}  // namespace hwtest
}  // namespace vpux
