//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

/*DMA builder config, depending on input/output configuration:

                 DMAop                            DMAop                             DMAop
DDR(FuncInput) --------> DDR(FuncOutput) / CMX* --------> DDR(FuncOutput) / CMX* --------> DDR(FunOutput)
                        stop execution |                 stop execution |                stop execution |

*CMX data can be stored either on tile 0 or tile 1


Testing Memory Side Cache, (initial pipeline remains valid, for accuracy check):
At least one from inputMSC/outputMSC buffers must be in DDR

DDR(FuncInput) \+                                                                      ----> ... (Initial pipepile)
                \ DMAop                                                               /
                 \                                                                   /
              inputMSC(DDRHeap/CMX)          inputMSC(DDRHeap/CMX)                  /
                            \                 /         \           ...
                             \ DMAop         /DMAop      \ Dmaop    /     .......
                              \             /             \        /
                          outputMSC(DDRHeap/CMX)           ........
                           transaction1                  transaction2  ----> transactionN

For non-trashing tests, the buffers are located at the same address.
For trashing tests, the input buffer of the previous DMAOp changes address before another DMAOp, so caching wouldn't
be possible:

    inputMSC(DDRHeap/CMX)    inputMSC(DDRHeap/CMX)
             \                 /                   \ offset(outputMSC)++
              \ DmaOp         / DMAop               \ DMAop
               \             / offset(inputMSC)++    \..
            outputMSC(DDRHeap/CMX)                   ....
*/

void buildDMA(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder, Logger& log,
              mlir::Type inputType, mlir::Type outputType) {
    auto input = testDesc.getInputLayerList().front();
    auto dmaParams = testDesc.getDMAparams();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    auto testArchitecture = testDesc.getArchitecture();
    if (testArchitecture == vpux::VPU::ArchKind::NPU40XX) {
        VPUX_THROW_UNLESS(dmaParams.engine == 0, "buildDMA: DMA on NPU40XX should have 1 engine");
    }

    VPUX_THROW_UNLESS(!inShape.empty(), "buildDMA: Input rank is 0");
    VPUX_THROW_UNLESS(inShape == outShape, "buildDMA: in_shape and out_shape don't match");
    if (!dmaParams.doConvert) {
        VPUX_THROW_UNLESS(inputType == outputType, "buildDMA: inputType and outputType don't match");
    }

    auto inputTotalSize = totalTensorSize(inShape, inputType);
    auto outputTotalSize = totalTensorSize(outShape, outputType);

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC));

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(),
            printToString("dma_from_{0}_{1}_to_{2}_{3}", nb::to_string(dmaParams.srcLocation), inputType,
                          nb::to_string(dmaParams.dstLocation), outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcInput0 = func.getArgument(0);
    auto funcOutput = func.getArgument(1);

    // input - output tensors

    llvm::SmallVector<mlir::Value> waitBarriers;
    mlir::Value DMAinput;
    size_t CMX0_AVAILABLE_OFFSET = 0;
    size_t CMX1_AVAILABLE_OFFSET = 0;
    VPURT::DeclareVirtualBarrierOp DMAtaskBarrier;
    if (dmaParams.srcLocation == nb::MemoryLocation::DDR) {
        DMAinput = funcInput0;
    } else if (dmaParams.srcLocation == nb::MemoryLocation::CMX0 || dmaParams.srcLocation == nb::MemoryLocation::CMX1) {
        const auto sectionIdx = dmaParams.srcLocation == nb::MemoryLocation::CMX0 ? 0 : 1;
        auto inputCMXtype =
                getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, inShape, inputType, DimsOrder::NHWC);
        auto inputCMX = createDeclareTensorOp(
                funcbuilder, inputCMXtype, VPURT::BufferSection::CMX_NN, sectionIdx,
                dmaParams.srcLocation == nb::MemoryLocation::CMX0 ? CMX0_AVAILABLE_OFFSET : CMX1_AVAILABLE_OFFSET);
        if (dmaParams.srcLocation == nb::MemoryLocation::CMX0) {
            CMX0_AVAILABLE_OFFSET += inputTotalSize;
        } else {
            CMX1_AVAILABLE_OFFSET += inputTotalSize;
        }

        DMAtaskBarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc());

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(),
                                              mlir::ValueRange(DMAtaskBarrier.getBarrier()), builder.getUnknownLoc(),
                                              funcInput0, inputCMX.getOperation()->getResult(0), dmaParams.engine);

        waitBarriers.emplace_back(DMAtaskBarrier.getBarrier());
        DMAinput = inputCMX.getOperation()->getResult(0);
    } else {
        VPUX_THROW("Unsupported src memory location {0}", nb::to_string(dmaParams.srcLocation));
    }

    // test memory side cache -> 1000 DMA circular transactions
    if (dmaParams.testMemSideCache) {
        size_t srcOffset = 0;
        size_t dstOffset = 0;
        size_t DDR_AVAILABLE_OFFSET = 0;
        auto srcSectionIdx = 0;
        auto dstSectionIdx = 0;
        VPURT::BufferSection srcBufferSection, dstBufferSection;
        mlir::MemRefType inputMSCBufferType, outputMSCBufferType;
        vpux::VPURT::DeclareBufferOp inputMSC, outputMSC;

        auto waitBarrier = DMAtaskBarrier;
        DMAtaskBarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc());

        // setup input/output buffers
        if (dmaParams.srcLocation == nb::MemoryLocation::DDR) {
            srcBufferSection = VPURT::BufferSection::DDR;
            srcOffset = DDR_AVAILABLE_OFFSET;
            DDR_AVAILABLE_OFFSET += inputTotalSize;

            inputMSCBufferType = getMemRefType(srcBufferSection, inShape, inputType, DimsOrder::NHWC);
            inputMSC = createDeclareTensorOp(funcbuilder, inputMSCBufferType, srcBufferSection, srcOffset);
        } else {
            srcBufferSection = VPURT::BufferSection::CMX_NN;
            if (dmaParams.srcLocation == nb::MemoryLocation::CMX0) {
                srcOffset = CMX0_AVAILABLE_OFFSET;
                CMX0_AVAILABLE_OFFSET += inputTotalSize;
                srcSectionIdx = 0;
            } else {
                srcOffset = CMX1_AVAILABLE_OFFSET;
                CMX1_AVAILABLE_OFFSET += inputTotalSize;
                srcSectionIdx = 1;
            }

            inputMSCBufferType = getMemRefType(srcBufferSection, srcSectionIdx, inShape, inputType, DimsOrder::NHWC);
            inputMSC =
                    createDeclareTensorOp(funcbuilder, inputMSCBufferType, srcBufferSection, srcSectionIdx, srcOffset);
        }

        if (dmaParams.dstLocation == nb::MemoryLocation::DDR) {
            dstBufferSection = VPURT::BufferSection::DDR;
            dstOffset = DDR_AVAILABLE_OFFSET;
            DDR_AVAILABLE_OFFSET += inputTotalSize;

            outputMSCBufferType = getMemRefType(dstBufferSection, inShape, inputType, DimsOrder::NHWC);
            outputMSC = createDeclareTensorOp(funcbuilder, outputMSCBufferType, dstBufferSection, dstOffset);
        } else {
            dstBufferSection = VPURT::BufferSection::CMX_NN;
            if (dmaParams.dstLocation == nb::MemoryLocation::CMX0) {
                dstOffset = CMX0_AVAILABLE_OFFSET;
                CMX0_AVAILABLE_OFFSET += inputTotalSize;
                dstSectionIdx = 0;
            } else {
                dstOffset = CMX1_AVAILABLE_OFFSET;
                CMX1_AVAILABLE_OFFSET += inputTotalSize;
                dstSectionIdx = 1;
            }

            outputMSCBufferType = getMemRefType(dstBufferSection, dstSectionIdx, inShape, inputType, DimsOrder::NHWC);
            outputMSC =
                    createDeclareTensorOp(funcbuilder, outputMSCBufferType, dstBufferSection, dstSectionIdx, dstOffset);
        }

        mlir::Value inputBufferMSC;
        mlir::Value outputBufferMSC;

        // setup input as function inputs
        if (dmaParams.srcLocation == nb::MemoryLocation::DDR) {
            inputBufferMSC = inputMSC;
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(),
                                                  mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                                                  builder.getUnknownLoc(), DMAinput, inputBufferMSC, dmaParams.engine);

        } else {
            inputBufferMSC = inputMSC.getOperation()->getResult(0);
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                                  mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                                                  builder.getUnknownLoc(), DMAinput, inputBufferMSC, dmaParams.engine);
        }

        if (dmaParams.dstLocation == nb::MemoryLocation::DDR) {
            outputBufferMSC = outputMSC;
        } else {
            outputBufferMSC = outputMSC.getOperation()->getResult(0);
        }

        for (auto dmaTransactionCount = 0; dmaTransactionCount < 10; ++dmaTransactionCount) {
            waitBarrier = DMAtaskBarrier;
            DMAtaskBarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc());

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                                  mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                                                  builder.getUnknownLoc(), inputBufferMSC, outputBufferMSC,
                                                  dmaParams.engine);
            waitBarrier = DMAtaskBarrier;
            DMAtaskBarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc());

            if (dmaParams.cacheTrashing) {
                if (dmaParams.srcLocation == nb::MemoryLocation::DDR) {
                    inputBufferMSC = createDeclareTensorOp(funcbuilder, inputMSCBufferType, srcBufferSection,
                                                           DDR_AVAILABLE_OFFSET);

                    DDR_AVAILABLE_OFFSET =
                            DDR_AVAILABLE_OFFSET > 4 * inputTotalSize ? 0 : DDR_AVAILABLE_OFFSET + inputTotalSize;
                }
            }

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                                  mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                                                  builder.getUnknownLoc(), outputBufferMSC, inputBufferMSC,
                                                  dmaParams.engine);

            if (dmaParams.cacheTrashing) {
                if (dmaParams.dstLocation == nb::MemoryLocation::DDR) {
                    outputBufferMSC = createDeclareTensorOp(funcbuilder, outputMSCBufferType, dstBufferSection,
                                                            DDR_AVAILABLE_OFFSET);
                    DDR_AVAILABLE_OFFSET =
                            DDR_AVAILABLE_OFFSET > 4 * inputTotalSize ? 0 : DDR_AVAILABLE_OFFSET + inputTotalSize;
                }
            }
        }
    }

    mlir::Value DMAoutput;
    if (dmaParams.dstLocation == nb::MemoryLocation::DDR) {
        DMAoutput = funcOutput;
    } else if (dmaParams.dstLocation == nb::MemoryLocation::CMX0 || dmaParams.dstLocation == nb::MemoryLocation::CMX1) {
        const auto sectionIdx = dmaParams.dstLocation == nb::MemoryLocation::CMX0 ? 0 : 1;
        auto outputCMXtype =
                getMemRefType(VPURT::BufferSection::CMX_NN, sectionIdx, outShape, outputType, DimsOrder::NHWC);
        auto outputCMX = createDeclareTensorOp(
                funcbuilder, outputCMXtype, VPURT::BufferSection::CMX_NN, sectionIdx,
                dmaParams.dstLocation == nb::MemoryLocation::CMX0 ? CMX0_AVAILABLE_OFFSET : CMX1_AVAILABLE_OFFSET);
        if (dmaParams.dstLocation == nb::MemoryLocation::CMX0) {
            CMX0_AVAILABLE_OFFSET += outputTotalSize;
        } else {
            CMX1_AVAILABLE_OFFSET += outputTotalSize;
        }
        DMAoutput = outputCMX.getOperation()->getResult(0);
    } else {
        VPUX_THROW("Unsupported dst memory location {0}", nb::to_string(dmaParams.dstLocation));
    }

    if (dmaParams.dstLocation == nb::MemoryLocation::CMX0 || dmaParams.dstLocation == nb::MemoryLocation::CMX1) {
        if (dmaParams.testMemSideCache) {
            waitBarriers.emplace_back(DMAtaskBarrier.getBarrier());
        }
        DMAtaskBarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc());
    }

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcbuilder.create<VPURT::DeclareVirtualBarrierOp>(builder.getUnknownLoc());
    if (!dmaParams.doConvert) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(llvm::ArrayRef<mlir::Value>(waitBarriers)),
                                              dmaParams.dstLocation == nb::MemoryLocation::DDR
                                                      ? mlir::ValueRange(finalBarrier.getBarrier())
                                                      : mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                                              builder.getUnknownLoc(), DMAinput, DMAoutput, dmaParams.engine);
    } else {
        VPURT::wrapIntoTaskOp<VPUIP::ConvertDMAOp>(
                funcbuilder, mlir::ValueRange(llvm::ArrayRef<mlir::Value>(waitBarriers)),
                dmaParams.dstLocation == nb::MemoryLocation::DDR ? mlir::ValueRange()
                                                                 : mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                builder.getUnknownLoc(), DMAinput, DMAoutput, dmaParams.engine);
    }

    if (dmaParams.dstLocation == nb::MemoryLocation::CMX0 || dmaParams.dstLocation == nb::MemoryLocation::CMX1) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(DMAtaskBarrier.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              DMAoutput, funcOutput, dmaParams.engine);
    }

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutput);
    // set runtime resources
    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    std::optional<int> numTiles = std::nullopt;
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    if (testArchitecture == vpux::VPU::ArchKind::NPU40XX) {
        // E#77729
        numTiles = 2;
    }

    auto initCompilerOptions = VPU::InitCompilerOptions(testArchitecture, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.setNumberOfDPUGroups(numTiles);
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    // assign physical barriers instead of virtual barriers
    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
