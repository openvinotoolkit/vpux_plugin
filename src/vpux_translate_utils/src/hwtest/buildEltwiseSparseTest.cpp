//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildEltwiseSparse(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto arch = testDesc.getArchitecture();

    auto input = testDesc.getInputLayerList().front();
    auto inputSM = testDesc.getInputSMList().front();

    auto weight = testDesc.getWeightLayers().front();
    auto weightsSM = testDesc.getWeightSMs().front();

    auto output = testDesc.getOutputLayers().front();
    auto seSize = testDesc.getEltwiseLayer().seSize;
    auto eltwiseMode = testDesc.getEltwiseLayer().mode;
    VPUX_THROW_UNLESS(seSize != 0, "buildEltwiseSparse: Storage Element size is 0");

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> inSMShape(inputSM.shape.begin(), inputSM.shape.end());
    SmallVector<int64_t> weightsShape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> weightsSMShape(weightsSM.shape.begin(), weightsSM.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(inShape.size() >= 4, "buildEltwiseSparse: Input rank is less than 4");
    VPUX_THROW_UNLESS(weightsShape.size() >= 4, "buildEltwiseSparse: Weights rank is less than 4");
    VPUX_THROW_UNLESS(outShape.size() >= 4, "buildEltwiseSparse: Output rank is less than 4");

    auto smElemType = mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Signless);
    auto outputTotalsize = totalTensorSize(outShape, outputType);
    auto inputTotalsize = totalTensorSize(inShape, inputType);
    auto inputSMTotalsize = totalTensorSize(inSMShape, smElemType);
    auto weightsSMTotalsize = totalTensorSize(weightsSMShape, smElemType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_SM_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputTotalsize;
    const auto INPUT_CMX_OFFSET = INPUT_SM_CMX_OFFSET + inputSMTotalsize;
    const auto WEIGHTS_SM_CMX_OFFSET = INPUT_CMX_OFFSET + inputTotalsize;
    const auto WEIGHTS_CMX_OFFSET = WEIGHTS_SM_CMX_OFFSET + weightsSMTotalsize;

    SmallVector<mlir::Type> inputTypes;
    auto inputDataType = getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC);
    auto inputSMType = inputDataType.cast<vpux::NDTypeInterface>().changeElemType(smElemType).cast<mlir::MemRefType>();
    inputTypes.push_back(inputDataType);
    inputTypes.push_back(inputSMType);

    auto weightsDataType =
            getMemRefType(VPURT::BufferSection::NetworkInput, weightsShape, weightsType, DimsOrder::NHWC);
    auto weightsSMType =
            weightsDataType.cast<vpux::NDTypeInterface>().changeElemType(smElemType).cast<mlir::MemRefType>();
    inputTypes.push_back(weightsDataType);
    inputTypes.push_back(weightsSMType);

    // output
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(),
            printToString("eltwise_{0}_{1}_{2}_{3}_{4}", inputType, smElemType, weightsType, smElemType, outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    VPUX_THROW_UNLESS(func.getNumArguments() == 5, "buildEltwiseSparse: number of arguments != 5");
    auto funcInput = func.getArgument(0);
    auto funcInputSM = func.getArgument(1);
    auto funcWeights = func.getArgument(2);
    auto funcWeightsSm = func.getArgument(3);
    auto funcOutput = func.getArgument(4);

    auto inputSMCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShape, smElemType, DimsOrder::NHWC);
    auto inputSMCmx =
            createDeclareTensorOp(funcBuilder, inputSMCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT_SM_CMX_OFFSET);

    auto inputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShape, inputType, DimsOrder::NHWC);
    auto inputcmx = createDeclareTensorOp(funcBuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);

    auto weightsSMCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, weightsShape, smElemType, DimsOrder::NHWC);
    auto weightsSMCmx = createDeclareTensorOp(funcBuilder, weightsSMCmxType, VPURT::BufferSection::CMX_NN, 0,
                                              WEIGHTS_SM_CMX_OFFSET);

    auto weightsCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, weightsShape, weightsType, DimsOrder::NHWC);
    auto weightscmx =
            createDeclareTensorOp(funcBuilder, weightsCmxType, VPURT::BufferSection::CMX_NN, 0, WEIGHTS_CMX_OFFSET);

    auto outputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    auto outputCmx =
            createDeclareTensorOp(funcBuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parentInputCmx =
            createDeclareTensorOp(funcBuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parentOutputCmx =
            createDeclareTensorOp(funcBuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 2);

    // DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), funcInputSM, inputSMCmx.getOperation()->getResult(0),
                                          0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), funcInput, inputcmx.getOperation()->getResult(0), 0);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), funcWeightsSm,
                                          weightsSMCmx.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), funcWeights, weightscmx.getOperation()->getResult(0),
                                          0);

    // NCE Task
    mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcBuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()),
            builder.getUnknownLoc(),
            /*input=*/inputcmx.getOperation()->getResult(0),
            /*input_sparsity_map=*/inputSMCmx.getOperation()->getResult(0),
            /*input_storage_element_table=*/nullptr,
            /*weights=*/weightscmx.getOperation()->getResult(0),
            /*weights_sparsity_map=*/weightsSMCmx.getOperation()->getResult(0),
            /*weightsTable=*/nullptr,
            /*instruction_table_list=*/nullptr,
            /*spr_lookup_table*/ nullptr,
            /*activation_window=*/nullptr,
            /*parent_input=*/parentInputCmx.getOperation()->getResult(0),
            /*parent_input_sparsity_map=*/inputSMCmx.getOperation()->getResult(0),
            /*parent_input_storage_element_table=*/nullptr,
            /*parent_output=*/parentOutputCmx.getOperation()->getResult(0),
            /*parent_output_sparsity_map=*/nullptr,
            /*output_buff=*/outputCmx.getOperation()->getResult(0),
            /*output_sparsity_map_buff=*/nullptr,
            /*profiling_data=*/nullptr, VPUIP::NCETaskType::ELTWISE,
            /*kernel_size=*/mlir::ArrayAttr(), /*kernel_strides*/ mlir::ArrayAttr(),
            /*kernel_padding=*/VPU::PaddingAttr(),
            /*activation_window_channel_length=*/actChannelLength,
            /*is_continued=*/nullptr, /*cm_sp_pattern=*/nullptr, /*is_segmented=*/nullptr,
            /*out_channel_offset=*/nullptr, /*input_channels_compression*/ nullptr,
            /*is_superdense=*/nullptr,
            /*is_inplace=*/nullptr,
            /*input_se_size=*/getIntAttr(ctx, static_cast<int32_t>(seSize)),
            /*output_se_size=*/nullptr);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t bypassMult = 1;
    int64_t bypassShift = 0;

    if (auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }

    // Since Eltwise operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto eltwiseQuantScale =
            VPU::calculateQuantScaleVectorForEltwise(inputCmxType, weightsCmxType, outputCmxType, arch, false);
    // Scale approximation is required for quantized inputs.
    if (inputCmxType.getElementType().isa<mlir::FloatType>()) {
        // It is intentional to apply int32 limits for floating point clamping.
        // See E#50875 for details.
        nceTask.addPPETask(funcBuilder, eltwiseMode, clampLow, clampHigh, bypassMult, bypassShift, bypassMult,
                           bypassShift, bypassShift, eltwiseQuantScale);
    } else {
        const auto scaleApproximation = QuantizationApproximation(arch, eltwiseQuantScale);
        nceTask.addPPETask(funcBuilder, eltwiseMode, clampLow, clampHigh, bypassMult, bypassShift,
                           scaleApproximation.mult(), scaleApproximation.shift());
    }

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.getVariants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outShape[3] - 1, outShape[2] - 1, outShape[1] - 1});
    const auto inEnd = getIntArrayAttr(ctx, std::vector<std::int64_t>{inShape[3] - 1, inShape[2] - 1, inShape[1] - 1});
    auto pad = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);

    // NB For eltwise operations, NTHW_NTK=(8, 8) is the only mode supported by
    // the hardware; this corresponds to CUBOID_8x16.
    nceTask.addDPUTask(variantbuilder, start, outEnd, start, inEnd, pad, VPU::MPEMode::CUBOID_8x16);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                          outputCmx.getOperation()->getResult(0), funcOutput, 0);

    funcBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutput);

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(),
               {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(inShape), smElemType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(weightsShape), weightsType, DimsOrder::NHWC, nullptr),
                getTensorType(ShapeRef(weightsShape), smElemType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}
}  // namespace hwtest
}  // namespace vpux
