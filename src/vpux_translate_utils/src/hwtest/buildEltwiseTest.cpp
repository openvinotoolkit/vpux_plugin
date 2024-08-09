//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

void buildEltwise(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto arch = testDesc.getArchitecture();

    auto input = testDesc.getInputLayerList().front();
    auto weight = testDesc.getWeightLayers().front();
    auto output = testDesc.getOutputLayers().front();
    auto eltwiseMode = testDesc.getEltwiseLayer().mode;

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> weightsShape(weight.shape.begin(), weight.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(inShape.size() >= 4, "buildEltwise: Input rank is less than 4");
    VPUX_THROW_UNLESS(outShape.size() >= 4, "buildEltwise: Output rank is less than 4");
    VPUX_THROW_UNLESS(weightsShape.size() >= 4, "buildEltwise: Weights rank is less than 4");

    auto outputTotalSize = totalTensorSize(outShape, outputType);
    auto inputTotalSize = totalTensorSize(inShape, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputTotalSize;
    const auto INPUT1_CMX_OFFSET = INPUT0_CMX_OFFSET + inputTotalSize;

    if (arch == vpux::VPU::ArchKind::NPU40XX || arch == vpux::VPU::ArchKind::NPU37XX) {
        VPUX_THROW_UNLESS((inputType == weightsType), "Eltwise expects inputs of same type");
    }

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, weightsShape, weightsType, DimsOrder::NHWC));

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("eltwise_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcweights = func.getArgument(1);
    auto funcoutput = func.getArgument(2);

    // input - output cmx tensors
    auto inputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShape, inputType, DimsOrder::NHWC);
    auto inputCmx =
            createDeclareTensorOp(funcbuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);

    auto weightsCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, weightsShape, weightsType, DimsOrder::NHWC);
    auto weightsCmx =
            createDeclareTensorOp(funcbuilder, weightsCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT1_CMX_OFFSET);

    auto outputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    auto outputCmx =
            createDeclareTensorOp(funcbuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parentInputCmx =
            createDeclareTensorOp(funcbuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);
    auto parentOutputCmx =
            createDeclareTensorOp(funcbuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcbuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcbuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    // DMAs
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, waitWLMBarrier, mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), funcinput, inputCmx.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), funcweights, weightsCmx.getOperation()->getResult(0),
                                          0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                          outputCmx.getOperation()->getResult(0), funcoutput, 0);

    mlir::Value wtTblValue;
    const auto qPerChType = outputType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
    if (qPerChType) {
        const auto WEIGHTSTABLE_CMX_OFFSET = INPUT1_CMX_OFFSET + inputTotalSize;

        // weights table ddr tensor
        SmallVector<int64_t> wtTblDataShape{output.shape[1], 1, 1, 4};
        auto wtTblDataDdrType = getMemRefType(VPURT::BufferSection::DDR, wtTblDataShape,
                                              builder.getIntegerType(32, true), DimsOrder::NHWC);
        const auto wtTblDataDdrValueType =
                mlir::RankedTensorType::get(wtTblDataShape, builder.getIntegerType(32, /*isSigned=*/true));

        const std::vector<int32_t> wtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(0),
                /*sparsityPtr*/ std::nullopt, static_cast<int32_t>(0), arch, output.shape[1]);

        auto wtTblDataValues = ArrayRef<int32_t>(wtTblDataValuesVec);
        auto wtTblDataVals = mlir::DenseElementsAttr::get(wtTblDataDdrValueType, wtTblDataValues);
        auto wtTblDataDdr =
                funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), wtTblDataDdrType,
                                                     Const::ContentAttr::get(wtTblDataVals).reorder(DimsOrder::NHWC));

        // weights table cmx tensor
        auto wtTblCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, wtTblDataShape,
                                          builder.getIntegerType(32, true), DimsOrder::NHWC);
        auto wtTblCmx = createDeclareTensorOp(funcbuilder, wtTblCmxType, VPURT::BufferSection::CMX_NN, 0,
                                              WEIGHTSTABLE_CMX_OFFSET);
        wtTblValue = wtTblCmx.getOperation()->getResult(0);

        // weights table dma ddr->cmx
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                              builder.getUnknownLoc(), wtTblDataDdr.getOperation()->getResult(0),
                                              wtTblCmx.getOperation()->getResult(0), 0);
    }

    // NCE Task
    mlir::IntegerAttr actChannelLength = builder.getI32IntegerAttr(0);
    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()),
            builder.getUnknownLoc(), outputCmxType, inputCmx.getOperation()->getResult(0),
            weightsCmx.getOperation()->getResult(0), wtTblValue,
            /*instruction_table_list=*/nullptr, /*spr_lookup_table*/ nullptr, /*activation_window=*/nullptr,
            parentInputCmx.getOperation()->getResult(0), parentOutputCmx.getOperation()->getResult(0),
            outputCmx.getOperation()->getResult(0), VPUIP::NCETaskType::ELTWISE, mlir::ArrayAttr(), mlir::ArrayAttr(),
            VPU::PaddingAttr(), actChannelLength,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    auto eltwiseQuantScale = qPerChType ? 0
                                        : VPU::calculateQuantScaleVectorForEltwise(inputCmxType, weightsCmxType,
                                                                                   outputCmxType, arch, false);

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
    // Scale approximation is required for quantized inputs.
    if (inputCmxType.getElementType().isa<mlir::FloatType>()) {
        // It is intentional to apply int32 limits for floating point clamping.
        // See E#50875 for details.
        nceTask.addPPETask(funcbuilder, eltwiseMode, clampLow, clampHigh, bypassMult, bypassShift, bypassMult,
                           bypassShift, bypassShift, eltwiseQuantScale);
    } else {
        const auto scaleApproximation = QuantizationApproximation(arch, eltwiseQuantScale);
        nceTask.addPPETask(funcbuilder, eltwiseMode, clampLow, clampHigh, bypassMult, bypassShift,
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

    funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcoutput);

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
                getTensorType(ShapeRef(weightsShape), weightsType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(inShape), outputType, DimsOrder::NHWC, nullptr)});
}
}  // namespace hwtest
}  // namespace vpux
