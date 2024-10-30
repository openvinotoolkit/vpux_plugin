//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

namespace {

vpux::VPU::PPEMode getPPEMode(nb::ActivationType activationType) {
    switch (activationType) {
    case nb::ActivationType::LeakyReLU:
        return vpux::VPU::PPEMode::LPRELU;
    case nb::ActivationType::ReLUX:
        return vpux::VPU::PPEMode::LRELU;
    case nb::ActivationType::Rsqrt:
        return vpux::VPU::PPEMode::RSQRT;
    case nb::ActivationType::Sigmoid:
        return vpux::VPU::PPEMode::SIGMOID;
    case nb::ActivationType::Sin:
        return vpux::VPU::PPEMode::NOOP;
    case nb::ActivationType::Tanh:
        return vpux::VPU::PPEMode::TANH;
    default:
        VPUX_THROW("Encountered unsupported activation type '{0}'", nb::to_string(activationType));
    }
}

unsigned getNumberOfLUTLines(nb::ActivationType activationType) {
    switch (activationType) {
    case nb::ActivationType::Rsqrt:
        return 10;
    case nb::ActivationType::Sigmoid:
        return 38;
    case nb::ActivationType::Sin:
        return 13;
    case nb::ActivationType::Tanh:
        return 14;
    default:
        VPUX_THROW("Encountered unsupported activation type '{0}' when getting Lut lines",
                   nb::to_string(activationType));
    }
}

}  // namespace

//
//       [input]
//          |
//        (conv)
//          |
//       [output]
//

void buildSimpleZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    const auto int32 = builder.getIntegerType(32, true);

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const auto weightsSwizzlingKey = testDesc.getWeightsSwizzlingKey();
    const auto architecture = testDesc.getArchitecture();
    const auto ppeConfiguration = testDesc.getActivationLayer();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildSimpleZMajorConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildSimpleZMajorConv: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildSimpleZMajorConv: Got empty weightsShape");

    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

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
    // ODU data size = Output Z multiple
    // 32 bit        = 16
    // 16 bit        = 16
    // 8 bit         = 16
    // 4 bit         = 32
    // 2 bit         = 64
    // 1 bit         = 128
    const auto outAlignment = std::max<int64_t>(outAlignmentInBits / outElSizeInBits, 16);
    const auto outAlignRemainder = outputCMXShape[outAlignDim.ind()] % outAlignment;
    if (outAlignRemainder != 0) {
        outputCMXShape[outAlignDim.ind()] += (outAlignment - outAlignRemainder);
    }
    auto padAuto = false;

    const auto alignmentRequirement = 16;
    const auto subLineLength = 4;
    const auto isCompressedFormatEnabled = inputChannels <= subLineLength;
    const auto isInputPaddingRequired = inputChannels < alignmentRequirement;

    // Swizzling alignment for some smaller buffers to 1024B as 512B aligned buffer cases fail:
    // E#56079 padding should be updated or removed in case of fix
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

    if (isInputPaddingRequired || padAuto) {
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
                    vpux::alignValUp(inputHeight * inputWidth, static_cast<std::int64_t>(subLineLength));
        }
    }

    mlir::IntegerAttr swizzlingKeyAttr;
    vpux::VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr;
    if (isWeightsSwizzlingRequired) {
        swizzlingKeyAttr = getIntAttr(ctx, nb::to_underlying(weightsSwizzlingKey));
        swizzlingSchemeAttr = createSwizzlingSchemeAttr(ctx, architecture, swizzlingKeyAttr.getInt());
    }
    const auto swizzlingAligment =
            (isWeightsSwizzlingRequired)
                    ? vpux::getAddressAlignmentForSwizzling(nb::to_underlying(weightsSwizzlingKey), architecture)
                    : 16;

    const auto weightsCMXSize =
            vpux::alignValUp(vpux::hwtest::totalWeightsSize(paddedWeightsCMXShape, weightsType),
                             static_cast<std::uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));

    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize =
            vpux::alignValUp(vpux::hwtest::totalTensorSize(inputCMXShape, inputType),
                             static_cast<std::uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));
    const auto wtableCMXSize = vpux::hwtest::totalTensorSize(weightsTableShape, int32);

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
            vpux::alignValUp(INPUT_CMX_OFFSET + inputCMXSize, static_cast<std::uint64_t>(swizzlingAligment));
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    auto ndOutputType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();
    const auto outputParamType = ndOutputType.changeDimsOrder(outputLayout);
    llvm::SmallVector<mlir::Type, 2> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::ArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("zmajor_conv_{0}_{1}_{2}", inputType, weightsType, outputType),
            funcType, builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = generateDefaultWeightsAttr(weightsValues, weightsType);

    if (isWeightsPaddingRequired) {
        auto kernelShapePaddingDifference =
                weightsCMXShape[vpux::Dims4D::Filter::KY.ind()] - weightsShape[vpux::Dims4D::Filter::KY.ind()];
        weightsAttribute =
                weightsAttribute.transform().padWithZero({0, 0, 0, 0}, {0, 0, kernelShapePaddingDifference, 0}).get();
    }

    const auto weightsDDRType =
            (isWeightsSwizzlingRequired)
                    ? getMemRefType(VPURT::BufferSection::Constant, 0, weightsCMXShape, weightsType, DimsOrder::NHWC,
                                    StridesRef(), swizzlingSchemeAttr)
                    : getMemRefType(VPURT::BufferSection::Constant, weightsCMXShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = vpux::getStrides(functionInput);

    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    if (isInputPaddingRequired || padAuto) {
        const auto weightsOutputChannelsStrideInBytes = weightsOutputChannelsStrideInBits.count() / CHAR_BIT;
        const auto weightsElementSizeInBits = getElemTypeSize(weightsType).count();
        const auto weightsElememtSizeInBytes = weightsElementSizeInBits / CHAR_BIT;
        const auto weightsOutputChannelsStrideInElements =
                weightsOutputChannelsStrideInBytes / weightsElememtSizeInBytes;
        const auto alignedWeightsOutputChannelStrideInElements = vpux::alignValUp(
                weightsOutputChannelsStrideInElements, static_cast<std::int64_t>(alignmentRequirement));
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
        weightsAttribute =
                weightsAttribute.transform()
                        .swizzleConstant(nb::to_underlying(weightsSwizzlingKey), static_cast<uint64_t>(architecture))
                        .get();
        weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsCMXShape, weightsType,
                                           vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET,
                                           swizzlingSchemeAttr);
        weightsCMX.setSwizzlingKeyAttr(vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
    } else {
        weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsCMXShape, weightsType,
                                           vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);
    }

    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);

    auto paddedInputCMX = inputCMX;
    auto paddedWeightsCMX = weightsCMX;
    if (isInputPaddingRequired) {
        paddedInputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedInputCMXShape,
                                               inputType, DimsOrder::NHWC, 0, INPUT_CMX_OFFSET);

        if (isWeightsSwizzlingRequired) {
            const auto paddedWeightsDDRType =
                    getMemRefType(VPURT::BufferSection::Constant, 0, paddedWeightsCMXShape, weightsType,
                                  DimsOrder::NHWC, StridesRef(), swizzlingSchemeAttr);
            const auto paddedWeightsStrides = paddedWeightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
            paddedWeightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                     paddedWeightsCMXShape, weightsType, DimsOrder::NHWC,
                                                     paddedWeightsStrides, 0, WEIGHTS_CMX_OFFSET, swizzlingSchemeAttr);
        } else {
            paddedWeightsCMX =
                    createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedWeightsCMXShape,
                                          weightsType, DimsOrder::NHWC, 0, WEIGHTS_CMX_OFFSET);
        }
    }

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType,
                                                                     std::move(weightsAttribute));

    auto outputCMXpadded = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outputCMXShape, outputType, outputLayout);
    auto ndOutputCMXpadded = outputCMXpadded.cast<vpux::NDTypeInterface>();
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           outputLayout, ndOutputCMXpadded.getStrides(), 0, OUTPUT_CMX_OFFSET);

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto sparsityPtrStep = 0;
    auto weightsPtrStep = static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT);

    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(testDesc.getArchitecture());
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(testDesc.getArchitecture());
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET), weightsPtrStep,
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

    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::ArrayRef<std::int32_t>(weightsTable));
    auto weightsTableStrides = weightsTableDDRMemRef.cast<vpux::NDTypeInterface>().getStrides();
    auto weightsTableContentAttrSetup =
            vpux::Const::ContentAttr::transform(weightsTableValues).reorder(vpux::DimsOrder::NHWC);

    vpux::VPURT::DeclareBufferOp weightsTableCMX;
    if (isWeightsSwizzlingRequired) {
        weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                DimsOrder::NHWC, weightsTableStrides, 0, WEIGHTSTABLE_CMX_OFFSET,
                                                swizzlingSchemeAttr);
        weightsTableCMX.setSwizzlingKeyAttr(
                vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
        paddedWeightsCMX.setSwizzlingKeyAttr(
                vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
        weightsTableContentAttrSetup = weightsTableContentAttrSetup.swizzleConstant(
                nb::to_underlying(weightsSwizzlingKey), static_cast<uint64_t>(architecture));
    } else {
        weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);
    }
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef, weightsTableContentAttrSetup.get());

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto barrier0 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto barrier1 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            builder.getUnknownLoc(), freeBarrierId++, testDesc.getWLMParams().isWLMPartialEnabled);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier, mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), functionInput, inputCMX.getOperation()->getResult(0),
                                          0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), weightsDDR.getOperation()->getResult(0),
                                          weightsCMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                                          builder.getUnknownLoc(), weightsTableDDR.getOperation()->getResult(0),
                                          weightsTableCMX.getOperation()->getResult(0), 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                          outputCMX.getOperation()->getResult(0), functionOutput, 0);

    mlir::Value sprLUTBuffer = nullptr;
    if (ppeConfiguration.activationType == nb::ActivationType::Rsqrt ||
        ppeConfiguration.activationType == nb::ActivationType::Sigmoid ||
        ppeConfiguration.activationType == nb::ActivationType::Sin ||
        ppeConfiguration.activationType == nb::ActivationType::Tanh) {
        const auto SPR_LUT_TABLE_CMX_OFFSET =
                vpux::alignValUp(WEIGHTSTABLE_CMX_OFFSET + wtableCMXSize, static_cast<std::uint64_t>(alignment));
        VPUX_THROW_UNLESS(SPR_LUT_TABLE_CMX_OFFSET % alignment == 0,
                          "SPR_LUT_TABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                          SPR_LUT_TABLE_CMX_OFFSET);

        const llvm::SmallVector<std::int64_t> sprLUTShape{1, 1, getNumberOfLUTLines(ppeConfiguration.activationType),
                                                          16};
        const auto sprLUTElementType = mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);

        auto sprLUTContent = computeSprLookupTable(ppeConfiguration.activationType);

        auto sprLUTValues = mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(sprLUTShape, sprLUTElementType),
                                                         llvm::ArrayRef<uint16_t>(sprLUTContent));

        auto sprLUTDDRType =
                getMemRefType(VPURT::BufferSection::Constant, sprLUTShape, sprLUTElementType, DimsOrder::NCHW);
        auto sprLUTStrides = sprLUTDDRType.cast<vpux::NDTypeInterface>().getStrides();

        auto sprLUTConstAttr = vpux::Const::ContentAttr::transform(sprLUTValues).reorder(vpux::DimsOrder::NCHW).get();

        auto sprLUTDDR = functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), sprLUTDDRType,
                                                                        std::move(sprLUTConstAttr));

        auto sprLUTCMX =
                createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, sprLUTShape, sprLUTElementType,
                                      DimsOrder::NCHW, sprLUTStrides, 0, SPR_LUT_TABLE_CMX_OFFSET);

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), builder.getUnknownLoc(),
                sprLUTDDR.getOperation()->getResult(0), sprLUTCMX.getOperation()->getResult(0), 0);

        sprLUTBuffer = sprLUTCMX.getBuffer();
    }

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    auto sparsityPattern = (isInputPaddingRequired || padAuto) ? ((1 << inputChannels) - 1) : 0;

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, barrier0.getBarrier(), barrier1.getBarrier(), builder.getUnknownLoc(),
            paddedInputCMX.getBuffer(), paddedWeightsCMX.getBuffer(), weightsTableCMX.getBuffer(),
            /*instruction_table_list*/ nullptr, /*spr_lookup_table=*/sprLUTBuffer, paddedInputCMX.getBuffer(),
            outputCMX.getBuffer(), outputCMX.getBuffer(), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides,
            kernelPaddings, nullptr, vpux::getIntAttr(builder.getContext(), sparsityPattern), nullptr, nullptr,
            inputChannelsCompression);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});

    const auto inShape = paddedInputCMX.getType().cast<NDTypeInterface>().getShape();
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inShape[Dims4D::Act::W] - 1, inShape[Dims4D::Act::H] - 1,
                                                           inShape[Dims4D::Act::C] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);

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
        nceTask.addPPETask(functionBuilder, ppeAttr);
    } else {
        if (const auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
            SmallVector<int64_t> quantMults;
            SmallVector<int64_t> quantShifts;
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
                    vpux::getIntAttr(ctx, scaleApproximation.postShift()), /* in1QuantMult = */ nullptr,
                    /* in2QuantMult = */ nullptr, /*fp_prelu_alpha*/ nullptr);
            nceTask.addPPETask(functionBuilder, ppeAttr);
        } else {
            auto ppeAttr = VPU::PPEIntAttr::get(
                    ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP), vpux::getIntAttr(ctx, clampLow),
                    vpux::getIntAttr(ctx, clampHigh), vpux::getIntAttr(ctx, lreluMult),
                    vpux::getIntAttr(ctx, lreluShift), getFPArrayAttr(ctx, ArrayRef<double>{outputScale}),
                    vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.mult()}),
                    vpux::getIntArrayAttr(ctx, ArrayRef<int64_t>{scaleApproximation.shift()}),
                    vpux::getIntAttr(ctx, scaleApproximation.postShift()), /* in1QuantMult = */ nullptr,
                    /* in2QuantMult = */ nullptr, /*fp_prelu_alpha*/ nullptr);
            nceTask.addPPETask(functionBuilder, ppeAttr);
        }
    }

    functionBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), functionOutput);

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
