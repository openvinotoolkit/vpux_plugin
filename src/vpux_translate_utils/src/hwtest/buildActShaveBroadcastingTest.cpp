//
// Copyright (C) 2024 Intel Corporation.
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

void buildActShaveBroadcast(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, const SmallVector<mlir::Type>& inputTypes, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto testArchitecture = testDesc.getArchitecture();

    //  Input/Output -----------------------------------------------------------
    auto inputList = testDesc.getInputLayerList();
    auto output = testDesc.getOutputLayers().front();
    auto broadcastingParams = testDesc.getActShaveBroadcastingParams();

    nb::MemoryLocation maxAllowedCMXTileIdx = nb::MemoryLocation::CMX0;
    if (testArchitecture == vpux::VPU::ArchKind::NPU40XX) {
        maxAllowedCMXTileIdx = nb::MemoryLocation::CMX5;
    } else {
        VPUX_THROW("buildActShaveBroadcast: unsupported architecture: {0}", VPU::stringifyArchKind(testArchitecture));
    }
    SmallVector<int64_t> dstBroadcastCmxTiles;
    for (const auto memLocation : broadcastingParams.dstLocations) {
        if (memLocation > maxAllowedCMXTileIdx) {
            VPUX_THROW("buildActShaveBroadcast: unsupported memory configured for ActShave broadcast destination: {0}",
                       nb::to_string(memLocation));
        }
        dstBroadcastCmxTiles.push_back(static_cast<int64_t>(memLocation));
    }
    const auto numBroadcastTiles = dstBroadcastCmxTiles.size();
    VPUX_THROW_UNLESS(numBroadcastTiles != 0,
                      "buildActShaveBroadcast: no valid destination location for ActShave broadcast");

    auto maxTileIdxIter = std::max_element(dstBroadcastCmxTiles.begin(), dstBroadcastCmxTiles.end());
    VPUX_THROW_UNLESS(maxTileIdxIter != dstBroadcastCmxTiles.end(),
                      "buildActShaveBroadcast: failure establishing the max destination CMX tile idx");
    int64_t maxTileIdx = *maxTileIdxIter;
    if (broadcastingParams.srcLocation >= nb::MemoryLocation::CMX0 &&
        broadcastingParams.srcLocation <= maxAllowedCMXTileIdx) {
        if (static_cast<int64_t>(broadcastingParams.srcLocation) > maxTileIdx) {
            maxTileIdx = static_cast<int64_t>(broadcastingParams.srcLocation);
        }
    } else if (broadcastingParams.srcLocation != nb::MemoryLocation::DDR) {
        VPUX_THROW("buildActShaveBroadcast: unsupported memory configured for ActShave broadcast source: {0}",
                   nb::to_string(broadcastingParams.srcLocation));
    }

    SmallVector<SmallVector<int64_t>> inShapes;
    SmallVector<mlir::Type> funcInputTypes;
    SmallVector<mlir::Type> funcOutputTypes;

    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        SmallVector<int64_t> inShape(inputList[idx].shape.begin(), inputList[idx].shape.end());
        VPUX_THROW_UNLESS(!inShape.empty(), "buildActShaveBroadcast: Got empty input '{0}' shape ", idx);
        inShapes.push_back(inShape);

        auto inputParamType =
                getMemRefType(VPURT::BufferSection::NetworkInput, inShapes[idx], inputTypes[idx], DimsOrder::NHWC);
        funcInputTypes.push_back(inputParamType);
    }

    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());
    VPUX_THROW_UNLESS(!outShape.empty(), "buildActShaveBroadcast: Got empty outputShape");

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    funcInputTypes.append(numBroadcastTiles, outputParamType);
    funcOutputTypes.append(numBroadcastTiles, outputParamType);

    // set runtime resources ------------------------------------------------------
    std::optional<int> numTiles = static_cast<int>(maxTileIdx) + 1;
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    //  Pass Manager
    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);

    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::ReferenceHW);
    initCompilerOptions.numberOfDMAPorts = 1;
    initCompilerOptions.setNumberOfDPUGroups(numTiles);
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Init Compilation failed");

    // Build Function ---------------------------------------------------------------

    const auto funcType = builder.getFunctionType(ArrayRef(funcInputTypes), ArrayRef(funcOutputTypes));

    std::string dstMemLocationsNames;
    for (auto memoryLoc : broadcastingParams.dstLocations) {
        dstMemLocationsNames.append(nb::to_string(memoryLoc));
        dstMemLocationsNames.append("_");
    }
    std::string funcOpName = "actshave_";
    for (auto iType : funcInputTypes) {
        funcOpName += printToString("{0}", iType);
    }
    funcOpName += printToString("from_{0}_to_{1}{2}", nb::to_string(broadcastingParams.srcLocation),
                                dstMemLocationsNames, outputType);

    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcOpName, funcType,
                                                   builder.getStringAttr("private"), /*arg_attrs=*/nullptr,
                                                   /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    SmallVector<mlir::Value> funcinputs;
    for (unsigned int idx = 0; idx < static_cast<unsigned int>(inputList.size()); idx++) {
        auto funcinput = func.getArgument(idx);
        funcinputs.push_back(funcinput);
    }

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    //  Build main function: barriers
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    //  Build main function: input/output cmx
    //  Build main function: DMA func input -> CMX input
    SmallVector<vpux::VPURT::DeclareBufferOp> inputCmxVec;
    size_t inputCmxOffset = 0;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        const auto sectionIdx = static_cast<int>(broadcastingParams.srcLocation);
        auto inputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, inShapes[idx], inputTypes[idx],
                                          DimsOrder::NHWC);
        inputCmxVec.push_back(createDeclareTensorOp(funcbuilder, inputCmxType, VPURT::BufferSection::CMX_NN, sectionIdx,
                                                    inputCmxOffset));
        inputCmxOffset += totalTensorSize(inShapes[idx], inputTypes[idx]);

        vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier,
                                                    mlir::ValueRange(barrier0.getBarrier()), builder.getUnknownLoc(),
                                                    funcinputs[idx], getTensorResult(inputCmxVec[idx]), 0);
    }

    // Create the distributed buffer necessary to broadcast the data to all the configured CMX tiles
    const auto outputCmxOffset = inputCmxOffset;
    auto cmxOutputMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    const auto tensorTypeIf = cmxOutputMemRefType.cast<vpux::NDTypeInterface>();
    const auto orderAttr = mlir::AffineMapAttr::get(tensorTypeIf.getDimsOrder().toAffineMap(ctx));
    const auto elemStrides = to_small_vector(tensorTypeIf.getStrides() | transformed([&](Bit stride) {
                                                 return stride.count() / tensorTypeIf.getElemTypeSize().count();
                                             }));
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr, /*allocSize=*/nullptr, ctx);
    const auto dimsSpace = IndexedSymbolAttr::get(ctx, stringifyMemoryKind(tensorTypeIf.getMemoryKind()));
    const auto duplicatedDistrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    const auto numClustersAttr = getIntAttr(ctx, numBroadcastTiles);
    auto distrTensorAttr = VPU::DistributionInfoAttr::get(ctx, duplicatedDistrModeAttr, nullptr, nullptr, nullptr,
                                                          nullptr, numClustersAttr, nullptr, nullptr, nullptr, nullptr,
                                                          nullptr, nullptr, nullptr);

    auto distributedCMXOutputType = VPUIP::DistributedBufferType::get(ctx, outShape, tensorTypeIf.getElementType(),
                                                                      layout, dimsSpace, distrTensorAttr);

    vpux::VPURT::DeclareBufferOp outputCmx = createDeclareTensorOp(
            funcbuilder, distributedCMXOutputType, VPURT::BufferSection::CMX_NN, dstBroadcastCmxTiles, outputCmxOffset);

    //  Build main function: Call operation builder
    const auto clusterId = static_cast<size_t>(broadcastingParams.srcLocation);
    buildActShaveTask(testDesc, module, funcbuilder, log, ArrayRef(funcInputTypes), inputCmxVec, outputCmx,
                      /*profilingDataCMX=*/nullptr, mlir::ValueRange(barrier0.getBarrier()),
                      mlir::ValueRange(barrier1.getBarrier()), clusterId);

    //  Build main function: DMA CMX output -> func output
    mlir::SmallVector<mlir::Value> funcOutputs;
    for (std::size_t idx = 0; idx < numBroadcastTiles; ++idx) {
        auto funcoutput = func.getArgument(inputList.size() + idx);
        funcOutputs.push_back(funcoutput);
        auto singleOutputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, dstBroadcastCmxTiles[idx], outShape,
                                                 outputType, DimsOrder::NHWC);
        auto singleOutputCMXBuffer =
                createDeclareTensorOp(funcbuilder, singleOutputCMXType, VPURT::BufferSection::CMX_NN,
                                      dstBroadcastCmxTiles[idx], outputCmxOffset);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              singleOutputCMXBuffer.getOperation()->getResult(0), funcoutput, 0);
    }

    //  Build main function: returnOp
    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutputs);

    SmallVector<mlir::Type> inputTensorTypeVec;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        auto inputTensorType = getTensorType(ShapeRef(inShapes[idx]), inputTypes[idx], DimsOrder::NHWC, nullptr);
        inputTensorTypeVec.push_back(inputTensorType);
    }

    SmallVector<mlir::Type> outputTensorTypesVec(
            numBroadcastTiles, getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr));

    //  CNN Operation
    mlir::SmallVector<ProfilingDataSection> profilingDataSections;
    buildCNNOp(builder, func.getName(), inputTensorTypeVec, outputTensorTypesVec, profilingDataSections);
}

}  // namespace hwtest
}  // namespace vpux
