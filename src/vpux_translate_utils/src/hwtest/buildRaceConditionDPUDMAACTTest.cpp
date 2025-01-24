//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

/*
//                  [input]
//                    |
//                 (barrier)
//               /    |     \
//              /     |      \
//             /      |       \
//            |       |        |
//         (conv)  (DMAop)   (ACTshave)
//            |       |        |
//       (barrier) (barrier) (barrier)
//            |       |        |
//        ... (loop with conv ops, DMA ops and barriers)
//            |       |        |
//         (conv)  (DMAop)   (ACTshave)
//            |       |        |
//             \      |       /
//              \     |      /
//               \    |     /
//                (barrier)
//            |       |        |
//       [output0]  [output1]  [output2]
//
*/

void buildRaceConditionDPUDMAACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                     mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                     mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();
    const auto iterationCount = testDesc.getIterationCount();
    const auto numClusters = testDesc.getNumClusters();

    // set runtime resources
    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::ReferenceHW);
    initCompilerOptions.numberOfDPUGroups = numClusters;
    initCompilerOptions.numberOfDMAPorts = 1;
    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    const llvm::SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const llvm::SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    VPUX_THROW_UNLESS(inputShape.size() >= 4, "buildRaceConditionDPUDMAACT: Input rank is less than 4");
    VPUX_THROW_UNLESS(outputShape.size() >= 4, "buildRaceConditionDPUDMAACT: Output rank is less than 4");
    VPUX_THROW_UNLESS(weightsShape.size() >= 4, "buildRaceConditionDPUDMAACT: Output rank is less than 4");

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    /////////////////////////////////////
    const auto alignmentRequirement = 16;

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto output0CMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);  // result of DPU op
    const auto output1CMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);    // result of DMA op
    const auto output2CMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);    // result of ActShave op
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET % alignment == 0, "WEIGHTS_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_CMX_OFFSET);

    const auto OUTPUT_CMX_OFFSET_0 = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_0 % alignment == 0, "OUTPUT_CMX_OFFSET_0 must be multiple of {0}, got {1} ",
                      alignment, OUTPUT_CMX_OFFSET_0);
    const auto OUTPUT_CMX_OFFSET_1 = OUTPUT_CMX_OFFSET_0 + output0CMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_1 % alignment == 0, "OUTPUT_CMX_OFFSET_1 must be multiple of {0}, got {1} ",
                      alignment, OUTPUT_CMX_OFFSET_1);
    const auto OUTPUT_CMX_OFFSET_2 = OUTPUT_CMX_OFFSET_1 + output1CMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_2 % alignment == 0, "OUTPUT_CMX_OFFSET_2 must be multiple of {0}, got {1} ",
                      alignment, OUTPUT_CMX_OFFSET_2);
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET_2 + output2CMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType0 =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);
    const auto outputParamType1 =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType2 =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, inputShape, inputType, DimsOrder::NHWC);

    const auto funcType = builder.getFunctionType(
            SmallVector<mlir::Type>{inputParamType, outputParamType0, outputParamType1, outputParamType2},
            SmallVector<mlir::Type>{outputParamType0, outputParamType1, outputParamType2});

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("race_condition_dpu_dma_act_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput_0 = function.getArgument(1);
    auto functionOutput_1 = function.getArgument(2);
    auto functionOutput_2 = function.getArgument(3);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = generateDefaultWeightsAttr(weightsValues, weightsType);

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                            vpux::DimsOrder::OYXI, 0, WEIGHTS_CMX_OFFSET);

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, std::move(weightsAttribute));

    auto outputCMX_0 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                             DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET_0);  // DPU result
    auto outputCMX_1 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                             DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET_1);  // DMA result
    auto outputCMX_2 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                             DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET_2);  // ActShave result

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto sparsityPtrStep = 0;
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(testDesc.getArchitecture());
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(testDesc.getArchitecture());
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
            static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
            VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
            output.shape[1], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::ArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues,
                                          Const::ContentSetup(weightsTableDDRType).reorder(vpux::DimsOrder::NHWC)));

    auto weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                 int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    VPURT::ConfigureBarrierOp waitBarrier;

    SmallVector<vpux::VPURT::DeclareBufferOp> inputCMXVec;
    inputCMXVec.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                                vpux::DimsOrder::NHWC, 0, INPUT_CMX_OFFSET));

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, functionInput, inputCMXVec[0].getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()), loc,
            weightsDDR.getOperation()->getResult(0), weightsCMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()), loc,
            weightsTableDDR.getOperation()->getResult(0), weightsTableCMX.getOperation()->getResult(0), 0);

    waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(inputParamType);
    inputTypes.push_back(outputParamType2);

    auto startIter = freeBarrierId++;
    for (std::size_t i = startIter; i < iterationCount; ++i) {
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i + 1);
        // DPU task
        auto nceTask_0 = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                mlir::ValueRange(updateBarrier.getBarrier()), loc, inputCMXVec[0].getBuffer(), weightsCMX.getBuffer(),
                weightsTableCMX.getBuffer(), /*spr_lookup_table=*/nullptr, inputCMXVec[0].getBuffer(),
                outputCMX_0.getBuffer(), outputCMX_0.getBuffer(), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides,
                kernelPaddings, nullptr, nullptr);

        const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
        const auto outEnd = getIntArrayAttr(
                ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
        const auto inEnd = getIntArrayAttr(
                ctx, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
        const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                             paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
        nceTask_0.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);
        // DMA
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                              mlir::ValueRange(updateBarrier.getBarrier()), loc,
                                              inputCMXVec[0].getOperation()->getResult(0),
                                              outputCMX_1.getOperation()->getResult(0), 0);
        // ActShave
        buildActShaveTask(testDesc, module, functionBuilder, log, ArrayRef(inputTypes), inputCMXVec, outputCMX_2,
                          /*profilingDataCMX*/ nullptr, mlir::ValueRange(waitBarrier.getBarrier()),
                          mlir::ValueRange(updateBarrier.getBarrier()));
        waitBarrier = updateBarrier;
    }

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, iterationCount + 1, testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputCMX_0.getOperation()->getResult(0), functionOutput_0, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputCMX_1.getOperation()->getResult(0), functionOutput_1, 0);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputCMX_2.getOperation()->getResult(0), functionOutput_2, 0);

    functionBuilder.create<mlir::func::ReturnOp>(
            loc, mlir::ValueRange{functionOutput_0, functionOutput_1, functionOutput_2});

    mlir::PassManager pmBuilderEnd(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (conv.compress) {
        pmBuilderEnd.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderEnd.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
