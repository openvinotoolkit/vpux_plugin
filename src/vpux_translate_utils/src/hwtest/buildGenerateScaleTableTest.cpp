//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzle_transform.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

void buildGenerateScaleTableTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                 mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                 mlir::Type outputType) {
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
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();
    constexpr size_t cluster = 0;

    const auto weightsSwizzlingKey = testDesc.getWeightsSwizzlingKey();
    const auto architecture = testDesc.getArchitecture();

    const SmallVector<int64_t> inputShape(input.shape.begin(), input.shape.end());
    const SmallVector<int64_t> outputShape(output.shape.begin(), output.shape.end());
    const SmallVector<int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const auto& activationLayer = testDesc.getActivationLayer();

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildGenerateScaleTableTest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildGenerateScaleTableTest: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildGenerateScaleTableTest: Got empty weightsShape");
    VPUX_THROW_UNLESS(activationLayer.activationType == nb::ActivationType::PopulateWeightTable,
                      "buildGenerateScaleTableTest: only weight table population on act shaves is supported.");
    VPUX_THROW_UNLESS(activationLayer.weightsOffset.has_value(),
                      "buildGenerateScaleTableTest: weights base offset is undefined.");
    VPUX_THROW_UNLESS(activationLayer.weightsPtrStep.has_value(),
                      "buildGenerateScaleTableTest: weights pointer step is undefined.");

    const SmallVector<int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;
    auto paddedInputCMXShape = inputShape;
    auto paddedWeightsCMXShape = weightsShape;
    auto weightsCMXShape = weightsShape;
    const auto inputChannelsIndex = vpux::Dims4D::Act::C.ind();
    const auto inputChannels = inputShape[inputChannelsIndex];
    const auto inputHeightIndex = vpux::Dims4D::Act::H.ind();
    const auto inputHeight = inputShape[inputHeightIndex];
    const auto inputWidthIndex = vpux::Dims4D::Act::W.ind();
    const auto inputWidth = inputShape[inputWidthIndex];
    const auto outputLayout = oduPermutationToLayout(testDesc.getODUPermutation());
    auto outputCMXShape = outputShape;
    const auto outAlignDim = getInnermostDim(outputLayout);
    const auto outAlignmentInBits = 16 * CHAR_BIT;
    const auto outElSizeInBits = static_cast<vpux::Bit>(getElemTypeSize(outputType)).count();
    const auto outAlignment = std::max<int64_t>(outAlignmentInBits / outElSizeInBits, 16);
    const auto outAlignRemainder = outputCMXShape[outAlignDim.ind()] % outAlignment;
    if (outAlignRemainder != 0) {
        outputCMXShape[outAlignDim.ind()] += (outAlignment - outAlignRemainder);
    }

    const auto alignmentRequirement = 16;
    const auto subLineLength = 4;
    const auto isCompressedFormatEnabled = inputChannels <= subLineLength;
    const auto isInputPaddingRequired = inputChannels < alignmentRequirement;

    const auto swizzlingPaddingAlignment = vpux::getSizeAlignmentForSwizzling(architecture) * 2;
    const auto paddedWeightsSize = vpux::hwtest::totalTensorSize(paddedWeightsCMXShape, weightsType);
    const auto isWeightsPaddingRequired = (weightsSwizzlingKey != nb::SwizzlingKey::key0) &&
                                          (paddedWeightsSize < static_cast<uint64_t>(swizzlingPaddingAlignment));
    const auto isWeightsSwizzlingRequired = weightsSwizzlingKey != nb::SwizzlingKey::key0;
    mlir::UnitAttr inputChannelsCompression = nullptr;

    if (isWeightsPaddingRequired) {
        weightsCMXShape[vpux::Dims4D::Filter::KY.ind()] *= swizzlingPaddingAlignment / paddedWeightsSize;
        paddedWeightsCMXShape = weightsCMXShape;
    }

    if (isInputPaddingRequired) {
        paddedWeightsCMXShape[vpux::Dims4D::Filter::IC.ind()] = alignmentRequirement;
    }

    if (isInputPaddingRequired) {
        inputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedInputCMXShape[inputChannelsIndex] = alignmentRequirement;

        if (isCompressedFormatEnabled) {
            inputChannelsCompression = mlir::UnitAttr::get(builder.getContext());
            inputCMXShape[inputChannelsIndex] = subLineLength;
            inputCMXShape[inputHeightIndex] = 1;
            inputCMXShape[inputWidthIndex] =
                    vpux::alignValUp(inputHeight * inputWidth, static_cast<int64_t>(subLineLength));
        }
    }

    mlir::IntegerAttr swizzlingKeyAttr;
    vpux::VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr;
    const auto swizzlingAligment =
            (isWeightsSwizzlingRequired)
                    ? vpux::getAddressAlignmentForSwizzling(nb::to_underlying(weightsSwizzlingKey), architecture)
                    : 16;

    const auto weightsCMXSize =
            vpux::alignValUp(vpux::hwtest::totalTensorSize(paddedWeightsCMXShape, weightsType),
                             static_cast<uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize = vpux::alignValUp(vpux::hwtest::totalTensorSize(inputCMXShape, inputType),
                                               static_cast<uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));
    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET = 0;
    const auto OUTPUT_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);

    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET =
            vpux::alignValUp(INPUT_CMX_OFFSET + inputCMXSize, static_cast<uint64_t>(swizzlingAligment));
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    auto ndOutputType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();
    const auto outputParamType = ndOutputType.changeDimsOrder(outputLayout);
    SmallVector<mlir::Type, 2> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("populate_weight_table_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttributeSetup = generateDefaultWeightsAttr(weightsValues, weightsType).transform();

    if (isWeightsPaddingRequired) {
        auto kernelShapePaddingDifference =
                weightsCMXShape[vpux::Dims4D::Filter::KY.ind()] - weightsShape[vpux::Dims4D::Filter::KY.ind()];
        weightsAttributeSetup =
                weightsAttributeSetup.padWithZero({0, 0, 0, 0}, {0, 0, kernelShapePaddingDifference, 0});
    }

    const auto weightsDDRType =
            (isWeightsSwizzlingRequired)
                    ? getMemRefType(VPURT::BufferSection::Constant, cluster, weightsCMXShape, weightsType,
                                    DimsOrder::NHWC, StridesRef(), swizzlingSchemeAttr)
                    : getMemRefType(VPURT::BufferSection::Constant, weightsCMXShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = vpux::getStrides(functionInput);

    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    if (isInputPaddingRequired) {
        const auto weightsOutputChannelsStrideInBytes = weightsOutputChannelsStrideInBits.count() / CHAR_BIT;
        const auto weightsElementSizeInBits = getElemTypeSize(weightsType).count();
        const auto weightsElememtSizeInBytes = weightsElementSizeInBits / CHAR_BIT;
        const auto weightsOutputChannelsStrideInElements =
                weightsOutputChannelsStrideInBytes / weightsElememtSizeInBytes;
        const auto alignedWeightsOutputChannelStrideInElements =
                vpux::alignValUp(weightsOutputChannelsStrideInElements, static_cast<int64_t>(alignmentRequirement));
        const auto alignedWeightsOutputChannelsStrideInBits =
                alignedWeightsOutputChannelStrideInElements * weightsElementSizeInBits;
        weightsOutputChannelsStrideInBits = vpux::Bit(alignedWeightsOutputChannelsStrideInBits);
    }
    if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
        weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
    }

    if (isInputPaddingRequired) {
        inputStrides[vpux::Dims4D::Act::W] =
                inputStrides[vpux::Dims4D::Act::C] * (isCompressedFormatEnabled ? subLineLength : alignmentRequirement);
        inputStrides[vpux::Dims4D::Act::H] = inputStrides[vpux::Dims4D::Act::W] * inputShape[inputWidthIndex];
        inputStrides[vpux::Dims4D::Act::N] = inputStrides[vpux::Dims4D::Act::H] * inputShape[inputHeightIndex];
    }

    vpux::VPURT::DeclareBufferOp weightsCMX;
    if (isWeightsSwizzlingRequired) {
        swizzlingKeyAttr = getIntAttr(ctx, nb::to_underlying(weightsSwizzlingKey));
        swizzlingSchemeAttr = createSwizzlingSchemeAttr(ctx, architecture, swizzlingKeyAttr.getInt());

        weightsAttributeSetup = weightsAttributeSetup.swizzleConstant(nb::to_underlying(weightsSwizzlingKey),
                                                                      static_cast<uint64_t>(architecture));
        weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsCMXShape, weightsType,
                                           vpux::DimsOrder::OYXI, weightsStrides, cluster, WEIGHTS_CMX_OFFSET,
                                           swizzlingSchemeAttr);
        weightsCMX.setSwizzlingKeyAttr(vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
    } else {
        weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsCMXShape, weightsType,
                                           vpux::DimsOrder::OYXI, weightsStrides, cluster, WEIGHTS_CMX_OFFSET);
    }

    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, cluster, INPUT_CMX_OFFSET);

    auto paddedInputCMX = inputCMX;
    auto paddedWeightsCMX = weightsCMX;
    if (isInputPaddingRequired) {
        paddedInputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedInputCMXShape,
                                               inputType, DimsOrder::NHWC, cluster, INPUT_CMX_OFFSET);

        if (isWeightsSwizzlingRequired) {
            const auto paddedWeightsDDRType =
                    getMemRefType(VPURT::BufferSection::Constant, cluster, paddedWeightsCMXShape, weightsType,
                                  DimsOrder::NHWC, StridesRef(), swizzlingSchemeAttr);
            const auto paddedWeightsStrides = paddedWeightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
            paddedWeightsCMX = createDeclareTensorOp(
                    functionBuilder, VPURT::BufferSection::CMX_NN, paddedWeightsCMXShape, weightsType, DimsOrder::NHWC,
                    paddedWeightsStrides, cluster, WEIGHTS_CMX_OFFSET, swizzlingSchemeAttr);
        } else {
            paddedWeightsCMX =
                    createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedWeightsCMXShape,
                                          weightsType, DimsOrder::NHWC, cluster, WEIGHTS_CMX_OFFSET);
        }
    }

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, weightsAttributeSetup.get());

    auto outputCMXpadded =
            getMemRefType(VPURT::BufferSection::CMX_NN, cluster, outputCMXShape, outputType, outputLayout);
    auto ndOutputCMXpadded = outputCMXpadded.cast<vpux::NDTypeInterface>();
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           outputLayout, ndOutputCMXpadded.getStrides(), cluster, OUTPUT_CMX_OFFSET);

    const auto sparsityPtrStep = 0;
    auto weightsPtrStep = static_cast<int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT);
    VPUX_THROW_UNLESS(activationLayer.weightsPtrStep.value() == weightsPtrStep,
                      "buildGenerateScaleTableTest: weight pointer step parameter {0} does not match strides {1}.",
                      activationLayer.weightsPtrStep.value(), weightsPtrStep);

    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(testDesc.getArchitecture());
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(testDesc.getArchitecture());
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<int32_t>(WEIGHTS_CMX_OFFSET), weightsPtrStep,
            VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
            output.shape[1], weightsType);

    mlir::MemRefType weightsTableDDRMemRef;
    if (isWeightsSwizzlingRequired) {
        weightsTableDDRMemRef = getMemRefType(VPURT::BufferSection::Constant, 0, weightsTableShape, int32,
                                              DimsOrder::NHWC, StridesRef(), swizzlingSchemeAttr);
    } else {
        weightsTableDDRMemRef =
                getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    }

    auto weightsTableStrides = weightsTableDDRMemRef.cast<vpux::NDTypeInterface>().getStrides();

    vpux::VPURT::DeclareBufferOp weightsTableCMX;
    if (isWeightsSwizzlingRequired) {
        weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                DimsOrder::NHWC, weightsTableStrides, cluster, WEIGHTSTABLE_CMX_OFFSET,
                                                swizzlingSchemeAttr);
        weightsTableCMX.setSwizzlingKeyAttr(
                vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
        paddedWeightsCMX.setSwizzlingKeyAttr(
                vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
    } else {
        weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                DimsOrder::NHWC, cluster, WEIGHTSTABLE_CMX_OFFSET);
    }

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    const int64_t inputCopyBarrierId = freeBarrierId++;
    auto inputCopyBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, inputCopyBarrierId);
    const int64_t actShaveUpdateBarrierId = inputCopyBarrierId + 1;
    auto actShaveUpdateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, actShaveUpdateBarrierId);
    const int64_t dpuUpdateBarrierId = actShaveUpdateBarrierId + 1;
    auto dpuUpdateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, dpuUpdateBarrierId);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier,
                                          mlir::ValueRange(inputCopyBarrier.getBarrier()), loc, functionInput,
                                          inputCMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(inputCopyBarrier.getBarrier()), loc,
            weightsDDR.getOperation()->getResult(0), weightsCMX.getOperation()->getResult(0), 0);

    SmallVector<mlir::Type, 1> actShaveInputTypes;
    actShaveInputTypes.push_back(weightsTableCMX.getType());
    SmallVector<vpux::VPURT::DeclareBufferOp> actShaveInputs = {};
    buildActShaveTask(testDesc, module, functionBuilder, log, ArrayRef(actShaveInputTypes), actShaveInputs,
                      weightsTableCMX, nullptr, mlir::ValueRange(inputCopyBarrier.getBarrier()),
                      mlir::ValueRange(actShaveUpdateBarrier.getBarrier()), cluster);

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    SmallVector<int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    auto sparsityPattern = isInputPaddingRequired ? ((1 << inputChannels) - 1) : 0;

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, actShaveUpdateBarrier.getBarrier(), dpuUpdateBarrier.getBarrier(), loc,
            paddedInputCMX.getBuffer(), paddedWeightsCMX.getBuffer(), weightsTableCMX.getBuffer(),
            /*instruction_table_list*/ nullptr, /*spr_lookup_table*/ nullptr, paddedInputCMX.getBuffer(),
            outputCMX.getBuffer(), outputCMX.getBuffer(), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides,
            kernelPaddings, nullptr, vpux::getIntAttr(builder.getContext(), sparsityPattern), nullptr, nullptr,
            inputChannelsCompression);

    const auto start = getIntArrayAttr(ctx, std::vector<int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});

    const auto inShape = paddedInputCMX.getType().cast<NDTypeInterface>().getShape();
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<int64_t>{inShape[Dims4D::Act::W] - 1, inShape[Dims4D::Act::H] - 1,
                                                      inShape[Dims4D::Act::C] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    const int64_t lreluMult = 1;
    const int64_t lreluShift = 0;
    if (const auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        clampLow = outElemQType.getStorageTypeMin();
        clampHigh = outElemQType.getStorageTypeMax();
    }

    auto outputScale = 1.0 / output.qp.scale.front();
    const auto scaleApproximation = QuantizationApproximation(outputScale);
    auto ppeAttr =
            VPU::PPEIntAttr::get(ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP), vpux::getIntAttr(ctx, clampLow),
                                 vpux::getIntAttr(ctx, clampHigh), vpux::getIntAttr(ctx, lreluMult),
                                 vpux::getIntAttr(ctx, lreluShift), getFPArrayAttr(ctx, ArrayRef<double>{outputScale}),
                                 vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.mult()}),
                                 vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.shift()}),
                                 vpux::getIntAttr(ctx, scaleApproximation.postShift()), /* in1QuantMult = */ nullptr,
                                 /* in2QuantMult = */ nullptr, /*fp_prelu_alpha*/ nullptr);
    nceTask.addPPETask(functionBuilder, ppeAttr);

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, 3, testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(dpuUpdateBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                          outputCMX.getOperation()->getResult(0), functionOutput, 0);

    functionBuilder.create<mlir::func::ReturnOp>(loc, functionOutput);

    module.dump();

    mlir::PassManager pmBuilderEnd(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (conv.compress) {
        pmBuilderEnd.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }
    if (isWeightsSwizzlingRequired) {
        pmBuilderEnd.nest<mlir::func::FuncOp>().addNestedPass<Const::DeclareOp>(Const::createConstantFoldingPass());
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderEnd.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, outputLayout, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
