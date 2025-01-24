// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

// usual case (only weights sparsity):
//
//                  [input]   [sparse weights*]
//                     |     /
//                   (conv)
//                     |
//                  [output]
//
//
// activation + weights sparsity case:
//
//   [input]   [input_sparsity_map]   [sparse weights*]
//       \             |                   /
//                  (conv)
//                     |
//                  [output]
//
// *weights are in sparse format, without sparsity map(SM), but the SM is produced in builder

vpux::VPU::PPEMode getPPEMode(nb::ActivationType activationType) {
    switch (activationType) {
    case nb::ActivationType::LeakyReLU:
        return vpux::VPU::PPEMode::LPRELU;
        break;
    case nb::ActivationType::ReLUX:
        return vpux::VPU::PPEMode::LRELUX;
        break;
    default:
        VPUX_THROW("Encountered unsupported activation type '{0}'", nb::to_string(activationType));
    }
}

void buildSparseZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);
    const auto arch = testDesc.getArchitecture();

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildSparseZMajorConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildSparseZMajorConv: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildSparseZMajorConv: Got empty weightsShape");
    VPUX_THROW_UNLESS(!weightsTableShape.empty(), "buildSparseZMajorConv: Got empty weightsTableShape");

    SmallVector<std::int64_t> inputSMShape;
    if (conv.act_sparsity) {
        inputSMShape = inputShape;
        VPUX_THROW_UNLESS(!inputSMShape.empty(), "buildSparseZMajorConv: Got empty inputSMShape");
    }

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    const auto alignmentRequirement = 16;

    const auto sparsityElementType = mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Signless);

    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);
    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto weightsSMCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, sparsityElementType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);

    std::size_t inputSMTotalsize = 0;
    if (conv.act_sparsity) {
        inputSMTotalsize = vpux::hwtest::totalTensorSize(inputSMShape, sparsityElementType);
    }

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET % alignment == 0, "WEIGHTS_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_CMX_OFFSET);

    const auto OUTPUT_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);

    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    size_t INPUT_SM_CMX_OFFSET = 0;
    if (conv.act_sparsity) {
        INPUT_SM_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
        VPUX_THROW_UNLESS(INPUT_SM_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                          alignment, INPUT_SM_CMX_OFFSET);
    }

    const auto WEIGHTS_SM_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize + inputSMTotalsize;
    VPUX_THROW_UNLESS(WEIGHTS_SM_CMX_OFFSET % alignment == 0, "WEIGHTS_SM_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_SM_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = WEIGHTS_SM_CMX_OFFSET + weightsSMCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    SmallVector<mlir::Type> inputTypes;
    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    inputTypes.push_back(inputParamType);
    if (conv.act_sparsity) {
        auto inputSMType = inputParamType.cast<vpux::NDTypeInterface>()
                                   .changeElemType(sparsityElementType)
                                   .cast<mlir::MemRefType>();
        inputTypes.push_back(inputSMType);
    }

    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::func::FuncOp>(
            loc,
            llvm::formatv("sparse_zm_conv_{0}_{1}_{2}_{3}", inputType, sparsityElementType, weightsType, outputType)
                    .str(),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    mlir::BlockArgument functionInputSM;
    mlir::BlockArgument functionOutput;
    if (conv.act_sparsity) {
        functionInputSM = function.getArgument(1);
        functionOutput = function.getArgument(2);
    } else {
        functionOutput = function.getArgument(1);
    }

    auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>();
    if (qty != nullptr) {
        VPUX_THROW_UNLESS(qty.getStorageType().getIntOrFloatBitWidth() >= CHAR_BIT,
                          "buildSparseZMajorConv: types with sizeof less than BYTE is not supported for weights");
    }
    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    Const::ContentSetup weightsAttributeSetup(weightsValues.getType());
    weightsAttributeSetup = weightsAttributeSetup.reorder(vpux::DimsOrder::OYXI);

    if (qty != nullptr) {
        weightsAttributeSetup = weightsAttributeSetup.castElemType(qty);
    }

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsSparsityMap = Const::ContentAttr::get(weightsValues, weightsAttributeSetup.clone().getSparsityMap());

    auto weightsAttribute = Const::ContentAttr::get(weightsValues, weightsAttributeSetup);
    auto numNonSparseElements = vpux::countNonSparseElementsPerOC(weightsAttribute.fold(), weightsType);
    const auto numNonSparseElementsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(numNonSparseElements.size())}, getInt64Type(ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numNonSparseElementsType, ArrayRef(numNonSparseElements));
    weightsAttributeSetup = weightsAttributeSetup.sparsify(true, numElemsAttr);
    auto compressedContentType =
            Const::inferFinalTypeAndSplat(weightsValues, weightsAttributeSetup.getTransformations()).first;
    const auto compressedWeightsTensorType = compressedContentType;
    const auto compressedWeightsDDRType =
            getMemRefType(VPURT::BufferSection::DDR, compressedWeightsTensorType.getShape().raw(),
                          compressedWeightsTensorType.getElementType(), compressedWeightsTensorType.getDimsOrder());

    const auto weightsSMShape = weightsSparsityMap.getType().getShape().raw();
    const auto weightsSMDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsSMShape, sparsityElementType, DimsOrder::NCHW);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = vpux::getStrides(functionInput);

    auto weightsCMX = createDeclareTensorOp(
            functionBuilder, VPURT::BufferSection::CMX_NN, compressedWeightsDDRType.getShape(),
            compressedWeightsDDRType.getElementType(), vpux::DimsOrder::OYXI, StridesRef(), 0, WEIGHTS_CMX_OFFSET);
    auto weightsDenseViewCMX =
            createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                  vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);
    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);

    mlir::Value inputSMCmxBuffer = nullptr;
    vpux::VPURT::DeclareBufferOp inputSMCmx;
    if (conv.act_sparsity) {
        auto inputSMCmxType =
                getMemRefType(VPURT::BufferSection::CMX_NN, 0, inputShape, sparsityElementType, DimsOrder::NHWC);
        inputSMCmx = createDeclareTensorOp(functionBuilder, inputSMCmxType, VPURT::BufferSection::CMX_NN, 0,
                                           INPUT_SM_CMX_OFFSET);
        inputSMCmxBuffer = inputSMCmx.getBuffer();
    }

    auto weightsSMStrides = weightsSMDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto weightsSMCMX =
            createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsSMShape, sparsityElementType,
                                  vpux::DimsOrder::OIYX, weightsSMStrides, 0, WEIGHTS_SM_CMX_OFFSET);

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            loc, compressedWeightsDDRType, Const::ContentAttr::get(weightsValues, std::move(weightsAttributeSetup)));

    auto weightsSMDDR = functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsSMDDRType,
                                                                       std::move(weightsSparsityMap));
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);

    const auto OC = weightsShape[vpux::Dims4D::Filter::OC.ind()];
    const auto IC = weightsShape[vpux::Dims4D::Filter::IC.ind()];
    const auto KY = weightsShape[vpux::Dims4D::Filter::KY.ind()];
    const auto KX = weightsShape[vpux::Dims4D::Filter::KX.ind()];
    const auto workloadSize = IC * KY * KX;
    const auto weightsElemByteSize = checked_cast<int32_t>(getElemTypeSize(weightsType).to<Byte>().count());

    auto weightsPtrOffset = static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET);
    auto sparsityPtrOffset = static_cast<std::int32_t>(WEIGHTS_SM_CMX_OFFSET);
    const auto sparsityPtrStep = Bit(workloadSize).to<Byte>().count();

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffset;
        const auto weightSetSize = (numNonSparseElements[oc] * weightsElemByteSize);
        weightsPtrOffset += alignValUp<int32_t>(static_cast<int32_t>(weightSetSize), alignmentRequirement);

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += static_cast<int32_t>(sparsityPtrStep);
    }

    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(testDesc.getArchitecture());
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(testDesc.getArchitecture());
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, llvm::ArrayRef(weightsPtrs), llvm::ArrayRef(sparsityPtrs), ppeConverter,
            biasConverter, weights.shape[vpux::Dims4D::Filter::OC.ind()], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::ArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues,
                                          Const::ContentSetup(weightsTableDDRType).reorder(vpux::DimsOrder::NHWC)));

    auto weightsTableCMX_0 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                   int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    VPURT::ConfigureBarrierOp waitBarrier;

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, functionInput, inputCMX.getOperation()->getResult(0), 0);
    if (conv.act_sparsity) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                              mlir::ValueRange(updateBarrier.getBarrier()), builder.getUnknownLoc(),
                                              functionInputSM, inputSMCmxBuffer, 0);
    }
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()), loc,
            weightsDDR.getOperation()->getResult(0), weightsCMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()), loc,
            weightsTableDDR.getOperation()->getResult(0), weightsTableCMX_0.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()), builder.getUnknownLoc(),
            weightsSMDDR.getOperation()->getResult(0), weightsSMCMX.getOperation()->getResult(0), 0);
    waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    auto nceTask_0 = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()), mlir::ValueRange(updateBarrier.getBarrier()),
            loc, inputCMX.getBuffer(), /*input_sparsity_map=*/inputSMCmxBuffer,
            /*input_storage_element_table=*/nullptr, weightsDenseViewCMX.getBuffer(), weightsSMCMX.getBuffer(),
            weightsTableCMX_0.getBuffer(), /*spr_lookup_table=*/nullptr, inputCMX.getBuffer(),
            /*parent_input_sparsity_map=*/inputSMCmxBuffer,
            /*parent_input_storage_element_table=*/nullptr, outputCMX.getBuffer(),
            /*parent_output_sparsity_map=*/nullptr, outputCMX.getBuffer(), /*output_sparsity_map=*/nullptr,
            /*profiling_data=*/nullptr, /*max_per_xy=*/nullptr, /*min_per_xy=*/nullptr,
            /*min_max_per_tensor=*/mlir::ValueRange(), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides,
            kernelPaddings, nullptr, nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask_0.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);
    waitBarrier = updateBarrier;

    const auto ppeConfiguration = testDesc.getActivationLayer();

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    const int64_t lreluMult = 1;
    const int64_t lreluShift = 0;
    auto outputScale = 1.0 / output.qp.scale.front();
    const auto scaleApproximation = QuantizationApproximation(outputScale);
    if (const auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        clampLow = outElemQType.getStorageTypeMin();
        clampHigh = outElemQType.getStorageTypeMax();
    }

    if (ppeConfiguration.activationType != nb::ActivationType::None) {
        if (ppeConfiguration.maximum != 0) {
            clampHigh = static_cast<int64_t>(ppeConfiguration.maximum);
        }

        const auto preluScale = ppeConfiguration.alpha;
        const auto alphaApproximation = PReLUApproximation(preluScale);
        auto ppeAttr = VPU::PPEIntAttr::get(
                ctx, VPU::PPEModeAttr::get(ctx, getPPEMode(ppeConfiguration.activationType)),
                vpux::getIntAttr(ctx, clampLow), vpux::getIntAttr(ctx, clampHigh),
                vpux::getIntAttr(ctx, alphaApproximation.mult()), vpux::getIntAttr(ctx, alphaApproximation.shift()),
                /* quantScale = */ nullptr, vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.mult()}),
                vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.shift()}),
                vpux::getIntAttr(ctx, scaleApproximation.postShift()), /* in1QuantMult = */ nullptr,
                /* in2QuantMult = */ nullptr, vpux::getFPAttr(ctx, ppeConfiguration.alpha));
        nceTask_0.addPPETask(functionBuilder, ppeAttr);
    } else {
        if (const auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
            SmallVector<double> quantMults;
            SmallVector<double> quantShifts;
            const auto scalesAndZps = extractScalesAndZeroPoints(outElemQType);
            const auto scales = scalesAndZps.first;
            const auto zps = scalesAndZps.second;

            quantMults.resize(scales.size());
            quantShifts.resize(scales.size());
            for (std::size_t i = 0; i < scales.size(); ++i) {
                const auto quantScaleApproximation = QuantizationApproximation(scales[i]);
                quantMults[i] = quantScaleApproximation.mult();
                quantShifts[i] = quantScaleApproximation.shift();
            }

            auto ppeAttr = VPU::PPEIntAttr::get(
                    ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP), vpux::getIntAttr(ctx, clampLow),
                    vpux::getIntAttr(ctx, clampHigh), vpux::getIntAttr(ctx, lreluMult),
                    vpux::getIntAttr(ctx, lreluShift), getFPArrayAttr(ctx, ArrayRef<double>{outputScale}),
                    vpux::getIntArrayAttr(ctx, quantMults), vpux::getIntArrayAttr(ctx, quantShifts),
                    getIntAttr(ctx, scaleApproximation.postShift()),
                    /* in1QuantMult = */ nullptr, /* in2QuantMult = */ nullptr,
                    /* fpPreluAlpha = */ nullptr);
            nceTask_0.addPPETask(functionBuilder, ppeAttr);
        } else {
            auto ppeAttr = VPU::PPEIntAttr::get(
                    ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP), vpux::getIntAttr(ctx, clampLow),
                    vpux::getIntAttr(ctx, clampHigh), vpux::getIntAttr(ctx, lreluMult),
                    vpux::getIntAttr(ctx, lreluShift), getFPArrayAttr(ctx, ArrayRef<double>{outputScale}),
                    vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.mult()}),
                    vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.shift()}),
                    getIntAttr(ctx, scaleApproximation.postShift()),
                    /* in1QuantMult = */ nullptr, /* in2QuantMult = */ nullptr,
                    /* fpPreluAlpha = */ nullptr);
            nceTask_0.addPPETask(functionBuilder, ppeAttr);
        }
    }

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, freeBarrierId++, testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputCMX.getOperation()->getResult(0), functionOutput, 0);

    functionBuilder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{functionOutput});

    module.dump();

    mlir::PassManager pmBuilderEnd(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (conv.compress) {
        pmBuilderEnd.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }
    if (conv.act_sparsity) {
        pmBuilderEnd.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
    }
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderEnd.run(module)), "Compilation failed");

    if (conv.act_sparsity) {
        buildCNNOp(builder, function.getName(),
                   {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr),
                    getTensorType(ShapeRef(inputSMShape), sparsityElementType, DimsOrder::NHWC, nullptr)},
                   {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
    } else {
        buildCNNOp(builder, function.getName(),
                   {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
                   {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
    }
}

}  // namespace hwtest
}  // namespace vpux
