//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <functional>
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

//
//   Run 4 DMA ops in parallel:
//      2 x CMX source:
//          DMA port 0 channel 0
//          DMA port 1 channel 0
//      2 x DDR source:
//          DMA port 0 channel 1
//          DMA port 1 channel 1
//
//                                 [input]
//                                    |
//   (DMAop, CMX->CMX, port 0) -- (barrier) -- (DMAop, DDR->DDR, port 1)
//                                |       |
//          (DMAop, CMX->CMX, port 1)    (DMAop, DDR->DDR, port 0)
//                              \ |       | /
//                                (barrier)
//                      |          |     |           |
//                [output0]  [output1]  [output2]  [output3]
//

void buildDualChannelDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                             Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayerList().front();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildRaceConditionDMATest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildRaceConditionDMATest: Got empty outputShape");

    size_t CMX_0_OFFSET = 0;
    size_t CMX_1_OFFSET = 0;

    auto inputTotalSize = totalTensorSize(inShape, inputType);

    const auto inType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    const auto outType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);

    const auto funcType =
            builder.getFunctionType(ArrayRef(std::vector<mlir::Type>{inType, outType, outType, outType, outType}),
                                    ArrayRef(std::vector<mlir::Type>{outType, outType, outType, outType}));

    auto func = builder.create<mlir::func::FuncOp>(
            loc, printToString("race_condition_dma_{0}_{1}", inputType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    const auto funcInput = func.getArgument(0);
    const auto funcOutput_0 = func.getArgument(1);
    const auto funcOutput_1 = func.getArgument(2);
    const auto funcOutput_2 = func.getArgument(3);
    const auto funcOutput_3 = func.getArgument(4);

    auto inputCMXtype0 = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShape, inputType, DimsOrder::NHWC);
    auto inputCMX_0 = funcBuilder.create<VPURT::DeclareBufferOp>(loc, inputCMXtype0, VPURT::BufferSection::CMX_NN, 0,
                                                                 CMX_0_OFFSET);

    auto inputCMXtype1 = getMemRefType(VPURT::BufferSection::CMX_NN, 1, inShape, inputType, DimsOrder::NHWC);
    auto inputCMX_1 = funcBuilder.create<VPURT::DeclareBufferOp>(loc, inputCMXtype1, VPURT::BufferSection::CMX_NN, 1,
                                                                 CMX_1_OFFSET);

    CMX_0_OFFSET += inputTotalSize;
    CMX_1_OFFSET += inputTotalSize;

    const auto outputCMXType0 = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    auto outputCMX_0 = funcBuilder.create<VPURT::DeclareBufferOp>(loc, outputCMXType0, VPURT::BufferSection::CMX_NN, 0,
                                                                  CMX_0_OFFSET);

    const auto outputCMXType1 = getMemRefType(VPURT::BufferSection::CMX_NN, 1, outShape, outputType, DimsOrder::NHWC);
    auto outputCMX_1 = funcBuilder.create<VPURT::DeclareBufferOp>(loc, outputCMXType1, VPURT::BufferSection::CMX_NN, 1,
                                                                  CMX_1_OFFSET);

    auto barrier_0 = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, 0);
    auto barrier_1 = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, 1);
    // finalBarrier passed as production barrier to last DMA task
    auto barrier_2 = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, 2);

    // transactions from ProgrammableInput to CMX
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(),
                                                mlir::ValueRange(barrier_0.getBarrier()), loc, funcInput,
                                                inputCMX_0.getOperation()->getResult(0), 0);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(),
                                                mlir::ValueRange(barrier_0.getBarrier()), loc, funcInput,
                                                inputCMX_1.getOperation()->getResult(0), 1);

    // 4 DMA ops in parallel
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcBuilder, mlir::ValueRange(barrier_0.getBarrier()), mlir::ValueRange(barrier_1.getBarrier()), loc,
            inputCMX_0.getOperation()->getResult(0), outputCMX_0.getOperation()->getResult(0), 0);  // CMX->CMX (port 0)

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcBuilder, mlir::ValueRange(barrier_0.getBarrier()), mlir::ValueRange(barrier_1.getBarrier()), loc,
            inputCMX_1.getOperation()->getResult(0), outputCMX_1.getOperation()->getResult(0), 1);  // CMX->CMX (port 1)

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier_0.getBarrier()),
                                                mlir::ValueRange(barrier_1.getBarrier()), loc, funcInput, funcOutput_2,
                                                0);  // DDR->DDR (port 0)

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier_0.getBarrier()),
                                                mlir::ValueRange(barrier_1.getBarrier()), loc, funcInput, funcOutput_3,
                                                1);  // DDR->DDR (port 1)

    // transactions from CMX to ProgrammableOutput
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier_1.getBarrier()),
                                                mlir::ValueRange(barrier_2.getBarrier()), loc,
                                                outputCMX_0.getOperation()->getResult(0), funcOutput_0, 0);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier_1.getBarrier()),
                                                mlir::ValueRange(barrier_2.getBarrier()), loc,
                                                outputCMX_1.getOperation()->getResult(0), funcOutput_1, 1);

    funcBuilder.create<mlir::func::ReturnOp>(loc,
                                             mlir::ValueRange{funcOutput_0, funcOutput_1, funcOutput_2, funcOutput_3});

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 2;
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
