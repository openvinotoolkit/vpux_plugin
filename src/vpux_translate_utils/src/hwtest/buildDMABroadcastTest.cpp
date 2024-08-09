//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"

namespace vpux {
namespace hwtest {

/*DMA broadcast builder supports the test configurations described below.
CMX can be configured on any valid tile (NPU40XX: 0->5).


[Use case 1] DMA source - DDR, DMA destination - multiple CMX:

                 DMAop (distributed buffer)                 DMAop
DDR(FuncInput) ---------------------------> multiple CMX --------> DDR(FuncOutput 0)
                                                            DMAop
                                                          --------> DDR(FuncOutput n)


[Use case 2] DMA source - CMX, DMA destination - multiple CMX:

                 DMAop          DMAop (distributed buffer)                 DMAop
DDR(FuncInput) --------> CMX ---------------------------> multiple CMX --------> DDR(FuncOutput 0)
                                                                           DMAop
                                                                        --------> DDR(FuncOutput n)
*/

void buildDMABroadcast(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                       Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto testArchitecture = testDesc.getArchitecture();
    VPUX_THROW_UNLESS(!testDesc.getInputLayerList().empty(), "buildDMABroadcast: no input layer");
    VPUX_THROW_UNLESS(!testDesc.getOutputLayers().empty(), "buildDMABroadcast: no output layer");

    auto input = testDesc.getInputLayerList().front();
    auto dmaParams = testDesc.getDMAparams();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(dmaParams.engine == 0, "buildDMABroadcast: DMA on NPU40XX should have 1 engine");
    VPUX_THROW_UNLESS(!inShape.empty(), "buildDMABroadcast: input rank is 0");
    VPUX_THROW_UNLESS(inShape == outShape, "buildDMABroadcast: in_shape and out_shape don't match");
    VPUX_THROW_UNLESS(inputType == outputType, "buildDMABroadcast: inputType and outputType don't match");

    nb::MemoryLocation maxAllowedCMXTileIdx = nb::MemoryLocation::CMX0;
    if (testArchitecture == vpux::VPU::ArchKind::NPU40XX) {
        maxAllowedCMXTileIdx = nb::MemoryLocation::CMX5;
    } else {
        VPUX_THROW("buildDMABroadcast: unsupported architecture: {0}", VPU::stringifyArchKind(testArchitecture));
    }
    SmallVector<int64_t> dstBroadcastCmxTiles;
    for (const auto memLocation : dmaParams.dstLocations) {
        if (memLocation > maxAllowedCMXTileIdx) {
            VPUX_THROW("buildDMABroadcast: unsupported memory configured for DMA broadcast destination: {0}",
                       nb::to_string(memLocation));
        }
        dstBroadcastCmxTiles.push_back(static_cast<int64_t>(memLocation));
    }
    const auto numBroadcastTiles = dstBroadcastCmxTiles.size();
    VPUX_THROW_UNLESS(numBroadcastTiles != 0, "buildDMABroadcast: no valid destination location for DMA broadcast");

    auto maxTileIdxIter = std::max_element(dstBroadcastCmxTiles.begin(), dstBroadcastCmxTiles.end());
    VPUX_THROW_UNLESS(maxTileIdxIter != dstBroadcastCmxTiles.end(),
                      "buildDMABroadcast: failure establishing the max destination CMX tile idx");
    int64_t maxTileIdx = *maxTileIdxIter;
    if (dmaParams.srcLocation >= nb::MemoryLocation::CMX0 && dmaParams.srcLocation <= maxAllowedCMXTileIdx) {
        if (static_cast<int64_t>(dmaParams.srcLocation) > maxTileIdx) {
            maxTileIdx = static_cast<int64_t>(dmaParams.srcLocation);
        }
    } else if (dmaParams.srcLocation != nb::MemoryLocation::DDR) {
        VPUX_THROW("buildDMABroadcast: unsupported memory configured for DMA broadcast source: {0}",
                   nb::to_string(dmaParams.srcLocation));
    }

    // Handle mlir::func::FuncOp

    const auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    auto argTypes = SmallVector<mlir::Type>({inputParamType});

    const auto outputParamType =
            getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    argTypes.append(numBroadcastTiles, outputParamType);
    SmallVector<mlir::Type> returnTypes(numBroadcastTiles, outputParamType);

    const auto funcType = builder.getFunctionType(argTypes, returnTypes);

    std::string dstMemLocationsNames;
    for (auto memoryLoc : dmaParams.dstLocations) {
        dstMemLocationsNames.append(nb::to_string(memoryLoc));
        dstMemLocationsNames.append("_");
    }
    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(),
            printToString("dma_from_{0}_{1}_to_{2}{3}", nb::to_string(dmaParams.srcLocation), inputType,
                          dstMemLocationsNames, outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops

    auto funcInput = func.getArgument(0);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled, true);

    // Handle the input (configuration for memory source location - either DDR or CMX). If it is CMX, create a
    // DMA task to copy it from the function input argument into CMX
    mlir::Value dmaInput;
    size_t cmxBroadcastOffset = 0;
    auto inputDMABarrier = waitWLMBarrier;
    if (dmaParams.srcLocation == nb::MemoryLocation::DDR) {
        dmaInput = funcInput;
    } else {
        const auto sectionIdx = static_cast<int>(dmaParams.srcLocation);
        auto inputCMXtype =
                getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, inShape, inputType, DimsOrder::NHWC);
        auto inputCMX = createDeclareTensorOp(funcbuilder, inputCMXtype, VPURT::BufferSection::CMX_NN, sectionIdx, 0);

        auto srcInDstTilesIter = std::find(dstBroadcastCmxTiles.begin(), dstBroadcastCmxTiles.end(), sectionIdx);
        if (srcInDstTilesIter != dstBroadcastCmxTiles.end()) {
            cmxBroadcastOffset += totalTensorSize(inShape, inputType);
        }

        inputDMABarrier.clear();
        inputDMABarrier.emplace_back(
                funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc()).getBarrier());

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier, inputDMABarrier, builder.getUnknownLoc(),
                                              funcInput, inputCMX.getOperation()->getResult(0), dmaParams.engine);

        dmaInput = inputCMX.getOperation()->getResult(0);
    }

    // Create the distributed buffer necessary to broadcast the data to all the configured CMX tiles

    auto cmxOutputMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, outShape, outputType, DimsOrder::NHWC);
    const auto tensorTypeIf = cmxOutputMemRefType.cast<vpux::NDTypeInterface>();

    mlir::MLIRContext* ctx = funcbuilder.getContext();
    const auto orderAttr = mlir::AffineMapAttr::get(tensorTypeIf.getDimsOrder().toAffineMap(ctx));

    const auto elemStrides = to_small_vector(tensorTypeIf.getStrides() | transformed([&](Bit stride) {
                                                 return stride.count() / tensorTypeIf.getElemTypeSize().count();
                                             }));
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, ctx);

    const auto dimsSpace = IndexedSymbolAttr::get(ctx, stringifyMemoryKind(tensorTypeIf.getMemoryKind()));
    const auto duplicatedDistrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    const auto numClustersAttr = getIntAttr(ctx, numBroadcastTiles);

    auto distrTensorAttr = VPU::DistributedTensorAttr::get(ctx, duplicatedDistrModeAttr, nullptr, nullptr, nullptr,
                                                           nullptr, numClustersAttr, nullptr, nullptr, nullptr, nullptr,
                                                           nullptr, nullptr, nullptr);

    auto distributedCMXOutputType = VPUIP::DistributedBufferType::get(ctx, outShape, tensorTypeIf.getElementType(),
                                                                      layout, dimsSpace, distrTensorAttr);

    VPURT::DeclareBufferOp cmxDistributedOutputBuffer =
            createDeclareTensorOp(funcbuilder, distributedCMXOutputType, VPURT::BufferSection::CMX_NN,
                                  dstBroadcastCmxTiles, cmxBroadcastOffset);

    auto broadcastDMABarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc()).getBarrier();

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, inputDMABarrier, broadcastDMABarrier, builder.getUnknownLoc(),
                                          dmaInput, cmxDistributedOutputBuffer, dmaParams.engine);

    auto finalBarrier = funcbuilder
                                .create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc(),
                                                                        testDesc.getWLMParams().isWLMPartialEnabled)
                                .getBarrier();
    // Create CMX2DDR DMAs to move outputs from each CMX broadcast tile into to DDR (function output arguments)
    SmallVector<mlir::Value> funcOutputs;
    for (std::size_t i = 0; i < numBroadcastTiles; ++i) {
        auto funcOutput = func.getArgument(1 + i);
        funcOutputs.push_back(funcOutput);

        auto singleOutputCMXType = getMemRefType(VPURT::BufferSection::CMX_NN, dstBroadcastCmxTiles[i], outShape,
                                                 outputType, DimsOrder::NHWC);
        auto singleOutputCMXBuffer =
                createDeclareTensorOp(funcbuilder, singleOutputCMXType, VPURT::BufferSection::CMX_NN,
                                      dstBroadcastCmxTiles[i], cmxBroadcastOffset);

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, broadcastDMABarrier, finalBarrier, builder.getUnknownLoc(),
                                              singleOutputCMXBuffer.getOperation()->getResult(0), funcOutput,
                                              dmaParams.engine);
    }

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutputs);

    // Set runtime resources
    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    std::optional<int> numTiles = static_cast<int>(maxTileIdx) + 1;
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    auto initCompilerOptions = VPU::InitCompilerOptions(testArchitecture, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.setNumberOfDPUGroups(numTiles);
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    // Assign physical barriers instead of virtual barriers
    pm.addPass(VPURT::createAssignPhysicalBarriersPass(false, log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    SmallVector<mlir::Type> outputTensorTypes(
            numBroadcastTiles, getTensorType(ShapeRef(outShape), outputType, vpux::DimsOrder::NHWC, nullptr));

    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               outputTensorTypes);
}

}  // namespace hwtest
}  // namespace vpux
