//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

//
//       [input]
//          |
//       (conv_0) --- (conv_1)
//                       |
//                    [output]
//

void buildContinuedConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    using namespace VPUIP;

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    auto* ctx = builder.getContext();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape[0], weights.shape[1], weights.shape[2],
                                                       weights.shape[3]};

    VPUX_THROW_UNLESS(inputShape.size() >= 4, "buildContinuedConv: Got inputShape with rank less than 4");
    VPUX_THROW_UNLESS(outputShape.size() >= 4, "buildContinuedConv: Got outputShape with rank less than 4");
    VPUX_THROW_UNLESS(weightsShape.size() >= 4, "buildContinuedConv: Got weightsShape with rank less than 4");

    const auto streamsOverC = 2;
    const llvm::SmallVector<std::int64_t> inputPartialShape(
            {inputShape[0], inputShape[1] / streamsOverC, inputShape[2], inputShape[3]});
    const llvm::SmallVector<std::int64_t> weightsPartialShape(
            {weightsShape[0], weightsShape[1] / streamsOverC, weightsShape[2], weightsShape[3]});
    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsPartialShape[0], 1, 1, 4};

    const char* weightsFileName = "weights.dat";

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto OUTPUT_CONV_0_CMX_OFFSET = OUTPUT_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto OUTPUT_CONV_1_CMX_OFFSET = OUTPUT_CONV_0_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto INPUT_CMX_OFFSET = OUTPUT_CONV_1_CMX_OFFSET + totalTensorSize(outputShape, outputType);
    const auto WEIGHTSTABLE_0_CMX_OFFSET = INPUT_CMX_OFFSET + totalTensorSize(inputShape, inputType);
    const auto WEIGHTSTABLE_1_CMX_OFFSET = WEIGHTSTABLE_0_CMX_OFFSET + 4 * weightsTableShape[0] * weightsTableShape[3];
    const auto WEIGHTS_PARTIAL_0_CMX_OFFSET =
            WEIGHTSTABLE_1_CMX_OFFSET + 4 * weightsTableShape[0] * weightsTableShape[3];
    const auto WEIGHTS_PARTIAL_1_CMX_OFFSET =
            WEIGHTS_PARTIAL_0_CMX_OFFSET + totalTensorSize(weightsPartialShape, weightsType);
    const auto INPUT_CONV_0_CMX_OFFSET = INPUT_CMX_OFFSET;
    const auto INPUT_CONV_1_CMX_OFFSET = INPUT_CONV_0_CMX_OFFSET + totalTensorSize(inputShape, inputType) / 2;

    const auto getMemRef = [](ArrayRef<std::int64_t> shape, mlir::Type elemType, VPU::MemoryKind memKind) {
        return vpux::getMemRefType(ShapeRef(shape), elemType, DimsOrder::NHWC, memKind);
    };

    const auto outputParamType = getMemRef(outputShape, outputType, VPU::MemoryKind::DDR);

    llvm::SmallVector<mlir::Type, 3> inputTypes;
    inputTypes.push_back(getMemRef(inputShape, inputType, VPU::MemoryKind::DDR));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::ArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("continued_conv_{0}_{1}_{2}", inputType, weightsType, outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    const auto getCMXTensor = [&builder, &functionBuilder](ArrayRef<int64_t> shape, mlir::Type elemType,
                                                           std::size_t offset) {
        const auto CMXType = hwtest::getMemRefType(VPURT::BufferSection::CMX_NN, 0, shape, elemType, DimsOrder::NHWC);
        return functionBuilder.create<vpux::VPURT::DeclareBufferOp>(builder.getUnknownLoc(), CMXType,
                                                                    VPURT::BufferSection::CMX_NN, 0, offset);
    };

    const auto getMACAccTensor = [&builder, &functionBuilder, getMemRef](const llvm::SmallVector<std::int64_t>& shape,
                                                                         mlir::Type type, std::size_t offset) {
        const auto MACAccType = getMemRef(shape, type, VPU::MemoryKind::Register);
        return functionBuilder.create<vpux::VPURT::DeclareBufferOp>(builder.getUnknownLoc(), MACAccType,
                                                                    VPURT::BufferSection::MAC_Accumulators, 0, offset);
    };

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    // weights data
    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);

    // Weights partial 0
    const auto weightsPartialParamType = getMemRef(weightsPartialShape, weightsType, vpux::VPU::MemoryKind::DDR);
    auto weightsPartial0Values = splitWeightsOverC(weightsValues, weightsShape, weightsType, builder.getContext(),
                                                   /*startC*/ 0, /*endC*/ weightsPartialShape[1]);
    Const::ContentSetup weightsPartial0AttributeSetup(weightsPartial0Values.getType());
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        weightsPartial0AttributeSetup = weightsPartial0AttributeSetup.castElemType(qty);
    }

    auto weightsPartial0Attribute =
            Const::ContentAttr::get(weightsPartial0Values, weightsPartial0AttributeSetup.reorder(DimsOrder::NHWC));
    auto weightsPartial0DDR = functionBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightsPartialParamType,
                                                                       std::move(weightsPartial0Attribute));

    // Weights partial 1
    auto weightsPartial1Values =
            splitWeightsOverC(weightsValues, weightsShape, weightsType, ctx,
                              /*startC*/ weightsPartialShape[1], /*endC*/ 2 * weightsPartialShape[1]);
    Const::ContentSetup weightsPartial1AttributeSetup(weightsPartial1Values.getType());
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        weightsPartial1AttributeSetup = weightsPartial1AttributeSetup.castElemType(qty);
    }

    auto weightsPartial1Attribute =
            Const::ContentAttr::get(weightsPartial1Values, weightsPartial1AttributeSetup.reorder(DimsOrder::NHWC));
    auto weightsPartial1DDR = functionBuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightsPartialParamType,
                                                                       std::move(weightsPartial1Attribute));

    auto inputCMX = getCMXTensor(inputShape, inputType, INPUT_CMX_OFFSET);

    // Tensors - NCE_0
    auto inputPartial0CMX = getCMXTensor(inputPartialShape, inputType, INPUT_CONV_0_CMX_OFFSET);
    auto weightsPartial0CMX = getCMXTensor(weightsPartialShape, weightsType, WEIGHTS_PARTIAL_0_CMX_OFFSET);
    auto output0CMX = getMACAccTensor(outputShape, outputType, OUTPUT_CONV_0_CMX_OFFSET);

    // Tensors - NCE_1
    auto inputPartial1CMX = getCMXTensor(inputPartialShape, inputType, INPUT_CONV_1_CMX_OFFSET);
    auto weightsPartial1CMX = getCMXTensor(weightsPartialShape, weightsType, WEIGHTS_PARTIAL_1_CMX_OFFSET);
    auto output1CMX = getCMXTensor(outputShape, outputType, OUTPUT_CONV_1_CMX_OFFSET);

    // weights table 0
    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto sparsityPtrStep = 0;
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(testDesc.getArchitecture());
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(testDesc.getArchitecture());
    const auto weightsTable0 = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_PARTIAL_0_CMX_OFFSET),
            static_cast<std::int32_t>(weightsPartialShape[1] * weightsPartialShape[2] * weightsPartialShape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
            outputShape[1], weightsType);

    const auto weightsTableDDRMemRef = getMemRef(weightsTableShape, int32, VPU::MemoryKind::DDR);
    const auto weightsTable0Values =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::ArrayRef<std::int32_t>(weightsTable0));
    auto weightsTable0DDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTable0Values,
                                          Const::ContentSetup(weightsTableDDRType).reorder(vpux::DimsOrder::NHWC)));
    auto weightsTable0CMX = getCMXTensor(weightsTableShape, int32, WEIGHTSTABLE_0_CMX_OFFSET);

    // weights table 1

    const auto weightsTable1 = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_PARTIAL_1_CMX_OFFSET),
            static_cast<std::int32_t>(weightsPartialShape[1] * weightsPartialShape[2] * weightsPartialShape[3] *
                                      getElemTypeSize(weightsType).count() / 8),
            VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
            outputShape[1], weightsType);

    const auto weightsTable1Values =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::ArrayRef<std::int32_t>(weightsTable1));
    auto weightsTable1DDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTable1Values,
                                          Const::ContentSetup(weightsTableDDRType).reorder(vpux::DimsOrder::NHWC)));
    auto weightsTable1CMX = getCMXTensor(weightsTableShape, int32, WEIGHTSTABLE_1_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    // Barriers
    std::vector<mlir::Value> barriers;
    auto isFinalBarrier = false;
    auto startIter = freeBarrierId++;
    auto num_barriers = testDesc.getWLMParams().isWLMPartialEnabled == true ? 5 : 3;
    for (auto i = static_cast<int>(startIter); i <= num_barriers; ++i) {
        if (i == num_barriers)
            isFinalBarrier = testDesc.getWLMParams().isWLMPartialEnabled;
        auto barrier =
                functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), i, isFinalBarrier);
        barriers.push_back(barrier.getBarrier());
    }

    // Input DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier, barriers[0], builder.getUnknownLoc(),
                                          functionInput, inputCMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsPartial0DDR.getOperation()->getResult(0),
                                          weightsPartial0CMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsPartial1DDR.getOperation()->getResult(0),
                                          weightsPartial1CMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsTable0DDR.getOperation()->getResult(0),
                                          weightsTable0CMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), barriers[0], builder.getUnknownLoc(),
                                          weightsTable1DDR.getOperation()->getResult(0),
                                          weightsTable1CMX.getOperation()->getResult(0), 0);

    // NCE params
    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);
    const auto isContinued = mlir::UnitAttr::get(ctx);

    // NCE Task 0
    auto nceTask_0 = VPURT::wrapIntoTaskOp<NCEClusterTaskOp>(
            functionBuilder, barriers[0], barriers[1], builder.getUnknownLoc(), inputPartial0CMX.getBuffer(),
            weightsPartial0CMX.getBuffer(), weightsTable0CMX.getBuffer(),
            /*spr_lookup_table*/ nullptr, inputPartial0CMX.getBuffer(), output0CMX.getBuffer(), output0CMX.getBuffer(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings, isContinued, /*sp_pattern*/ nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd = getIntArrayAttr(
            ctx,
            std::vector<std::int64_t>{inputPartialShape[3] - 1, inputPartialShape[2] - 1, inputPartialShape[1] - 1});

    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask_0.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, VPU::MPEMode::CUBOID_16x16);

    // NCE Task 1
    auto nceTask_1 = VPURT::wrapIntoTaskOp<NCEClusterTaskOp>(
            functionBuilder, barriers[1], barriers[2], builder.getUnknownLoc(), inputPartial1CMX.getBuffer(),
            weightsPartial1CMX.getBuffer(), weightsTable1CMX.getBuffer(),
            /*spr_lookup_table*/ nullptr, inputPartial1CMX.getBuffer(), output1CMX.getBuffer(), output1CMX.getBuffer(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    nceTask_1.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, VPU::MPEMode::CUBOID_16x16);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, barriers[2], barriers[3], builder.getUnknownLoc(),
                                          output1CMX.getOperation()->getResult(0), functionOutput, 0);

    functionBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
