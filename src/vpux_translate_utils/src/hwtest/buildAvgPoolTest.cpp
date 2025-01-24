//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <functional>
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

#include <climits>

namespace vpux {
namespace hwtest {

void buildAvgpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto arch = testDesc.getArchitecture();

    auto input = testDesc.getInputLayerList().front();
    auto poolOp = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inShape.empty(), "buildAvgpool: Got empty inputShape");
    VPUX_THROW_UNLESS(!outShape.empty(), "buildAvgpool: Got empty outputShape");

    std::vector<int64_t> filterSize{poolOp.kernel_shape.at(0), poolOp.kernel_shape.at(1)};
    std::vector<int64_t> strideVec(poolOp.stride.begin(), poolOp.stride.end());
    std::vector<int64_t> paddingVec = convertNBPadtoNCETaskPad(poolOp.pad);

    auto inputTotalSize = totalTensorSize(inShape, inputType);
    auto outputTotalSize = totalTensorSize(outShape, outputType);

    auto scaleValue = 1 / double(poolOp.kernel_shape.at(0) * poolOp.kernel_shape.at(1));

    mlir::Type weightsType = inputType;
    if (auto qtype = inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        auto inputStorageType = mlir::quant::QuantizedType::castToStorageType(qtype);
        int64_t zeroPoint = 0;

        if (inputStorageType.isUnsignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else if (inputStorageType.isSignedInteger(8)) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 getSInt8Type(ctx), builder.getF32Type(), scaleValue,
                                                                 zeroPoint, 0, 1);
        } else if (inputStorageType.isFloat8E5M2()) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 builder.getFloat8E5M2Type(), builder.getF32Type(),
                                                                 scaleValue, zeroPoint, 0, 1);
        } else if (inputStorageType.isFloat8E4M3FN()) {
            weightsType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::FlagValue::Signed,
                                                                 builder.getFloat8E4M3FNType(), builder.getF32Type(),
                                                                 scaleValue, zeroPoint, 0, 1);
        } else {
            VPUX_THROW("Unsupported storage type for input quantized type. I8 or U8 or FP8 is supported only");
        }
    }

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputTotalSize;

    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inShape, inputType, DimsOrder::NHWC));
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto func = builder.create<mlir::func::FuncOp>(loc, printToString("avgPool_{0}_{1}", inputType, outputType),
                                                   funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr,
                                                   /*res_attrs=*/nullptr);

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcInput = func.getArgument(0);
    auto funcOutput = func.getArgument(1);

    // input - output cmx tensors

    auto inputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inShape, inputType, DimsOrder::NHWC);
    auto inputCmx = createDeclareTensorOp(funcBuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outShape, outputType, DimsOrder::NHWC);
    auto outputCmx =
            createDeclareTensorOp(funcBuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parentInputCmx =
            createDeclareTensorOp(funcBuilder, inputCmxType, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parentOutputCmx =
            createDeclareTensorOp(funcBuilder, outputCmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    // barrier config
    auto barrier0 = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    auto barrier1 = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcBuilder.create<VPURT::ConfigureBarrierOp>(loc, freeBarrierId++,
                                                                      testDesc.getWLMParams().isWLMPartialEnabled);

    // DMA input-->cmx
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, waitWLMBarrier, mlir::ValueRange(barrier0.getBarrier()),
                                                loc, funcInput, inputCmx.getOperation()->getResult(0), 0);

    mlir::Value wtTblValue;
    auto qPerChType = outputType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
    if (qPerChType) {
        const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + inputTotalSize;

        // weights table ddr tensor
        SmallVector<int64_t> wtTblDataShape{output.shape[1], 1, 1, 4};
        auto wtTblDataDdrType = getMemRefType(VPURT::BufferSection::DDR, wtTblDataShape,
                                              builder.getIntegerType(32, true), DimsOrder::NHWC);
        const auto wtTblDataDdrValueType =
                mlir::RankedTensorType::get(wtTblDataShape, builder.getIntegerType(32, /*isSigned=*/true));
        mlir::FloatAttr constScale = nullptr;
        if (!mlir::dyn_cast<mlir::quant::QuantizedType>(inputType)) {
            constScale = builder.getF32FloatAttr(scaleValue);
        }

        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
        const std::vector<int32_t> wtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(0),
                /*sparsityPtr*/ std::nullopt, static_cast<int32_t>(0), ppeConverter, biasConverter, output.shape[1],
                weightsType, {}, constScale);

        auto wtTblDataValues = ArrayRef<int32_t>(wtTblDataValuesVec);
        auto wtTblDataVals = mlir::DenseElementsAttr::get(wtTblDataDdrValueType, wtTblDataValues);
        auto wtTblDataDdr = funcBuilder.create<Const::DeclareOp>(
                builder.getUnknownLoc(), wtTblDataDdrType,
                Const::ContentAttr::get(wtTblDataVals,
                                        Const::ContentSetup(wtTblDataDdrValueType).reorder(DimsOrder::NHWC)));

        // weights table cmx tensor

        auto wtTblCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, wtTblDataShape,
                                          builder.getIntegerType(32, true), DimsOrder::NHWC);
        auto wtTblCmx = createDeclareTensorOp(funcBuilder, wtTblCmxType, VPURT::BufferSection::CMX_NN, 0,
                                              WEIGHTSTABLE_CMX_OFFSET);
        wtTblValue = wtTblCmx.getOperation()->getResult(0);

        // weights table dma ddr->cmx
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                              builder.getUnknownLoc(), wtTblDataDdr.getOperation()->getResult(0),
                                              wtTblCmx.getOperation()->getResult(0), 0);
    }

    // NCE Task
    auto filterSizeAttr = getIntArrayAttr(funcBuilder, filterSize);
    auto strides = getIntArrayAttr(funcBuilder, strideVec);
    auto kernelPadding = VPU::getPaddingAttr(ctx, paddingVec[PAD_NCETASK_LEFT], paddingVec[PAD_NCETASK_RIGHT],
                                             paddingVec[PAD_NCETASK_TOP], paddingVec[PAD_NCETASK_BOTTOM]);

    auto nceTask = vpux::VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcBuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()), loc,
            outputCmxType, inputCmx.getOperation()->getResult(0), mlir::Value(), wtTblValue,
            /*spr_lookup_table*/ nullptr, parentInputCmx.getOperation()->getResult(0),
            parentOutputCmx.getOperation()->getResult(0), outputCmx.getOperation()->getResult(0),
            VPUIP::NCETaskType::AVEPOOL, filterSizeAttr, strides, kernelPadding,
            /*actChannelLength*/ nullptr, /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    // Since AvgPool operation doesn't have weights table it requires final quantization scaling
    // to be part of output tensor description. Scale vector will be placed in PPE block and
    // later used during NCE task serialization
    auto avgPoolScale = qPerChType ? 0 : VPU::computeAvgPoolQuantScale(inputCmxType, outputCmxType, filterSize);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    if (auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;

        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }
    int64_t bypassMult = 1;
    int64_t bypassShift = 0;
    if (inputCmxType.getElementType().isa<mlir::FloatType>()) {
        // Scale approximation is required for quantized inputs.
        // It is intentional to apply int32 limits for floating point clamping.
        // See E#50875 for details.
        auto ppeAttr = VPU::PPEIntAttr::get(ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP),
                                            vpux::getIntAttr(ctx, clampLow), vpux::getIntAttr(ctx, clampHigh),
                                            vpux::getIntAttr(ctx, bypassMult), vpux::getIntAttr(ctx, bypassShift),
                                            vpux::getFPArrayAttr(ctx, ArrayRef<double>{avgPoolScale}),
                                            vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{bypassMult}),
                                            vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{bypassShift}),
                                            vpux::getIntAttr(ctx, bypassShift), /* in1QuantMult = */ nullptr,
                                            /* in2QuantMult = */ nullptr,
                                            /* fpPreluAlpha = */ nullptr);
        nceTask.addPPETask(funcBuilder, ppeAttr);
    } else {
        const auto scaleApproximation = QuantizationApproximation(avgPoolScale);
        auto ppeAttr = VPU::PPEIntAttr::get(
                ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP), vpux::getIntAttr(ctx, clampLow),
                vpux::getIntAttr(ctx, clampHigh), vpux::getIntAttr(ctx, bypassMult), vpux::getIntAttr(ctx, bypassShift),
                /* quantScale = */ nullptr, vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.mult()}),
                vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.shift()}),
                vpux::getIntAttr(ctx, scaleApproximation.postShift()), /* in1QuantMult = */ nullptr,
                /* in2QuantMult = */ nullptr,
                /* fpPreluAlpha = */ nullptr);
        nceTask.addPPETask(funcBuilder, ppeAttr);
    }

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.getVariants().front(), funcBuilder.getListener());

    // NB For pooling operations, NTHW_NTK=(16, 4) is the only mode supported by
    // the hardware; this corresponds to CUBOID_16x16.
    createDPUTaskOp(funcBuilder, variantbuilder, outShape, inShape, paddingVec, VPU::MPEMode::CUBOID_16x16);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                                mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                                outputCmx.getOperation()->getResult(0), funcOutput, 0);

    funcBuilder.create<mlir::func::ReturnOp>(loc, funcOutput);

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outShape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
