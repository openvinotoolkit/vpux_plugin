//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

//                    ReductionLayer (ReduceMean/ReduceSumSquare)
//
//                       [input]
//                          |
//                  (nce_cluster_task0)
//                          |
//                       [output0]
//                          |
//                   [functionOutput0]
//
//
//
//                ReductionOutputLayer (ReduceMax/ReduceMin)
//
//                         [input]
//                            |
//              _    (nce_cluster_task0)         _
//             |              |                   |
//        [output0]   [reductionOutput0]    [reductionOutput1]
//                            |                   |
//                     [functionOutput0]    [functionOutput1]
//                  |                    |
//     -------------|------------------- | --------------------
//     identity conv| min/max per xy or  | only min&max per tensor
//         output   | min&max per tensor | if multi_tile
//      !NOT USED!  |                    |
//

namespace vpux {
namespace hwtest {

void buildReductionTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log, mlir::Type inputType, mlir::Type outputType) {
    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    auto dpuGroups = 1;
    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    const auto reductionType = testDesc.getReductionType();
    nb::ReduceOutLayer reduceOutLayer;
    // reduceMax & reduceMin  take place on the ODU, so an identity conv is used as a normal NCE task
    // to provide an input for the ODU
    if (reductionType == vpux::VPUIP::NCETaskType::CONV) {
        reduceOutLayer = testDesc.getReduceOutLayer();
    }
    if (reduceOutLayer.isMultiTile) {
        dpuGroups += 1;
    }
    initCompilerOptions.numberOfDPUGroups = dpuGroups;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    auto* ctx = builder.getContext();

    const auto input = testDesc.getInputLayerList().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildReduction: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildReduction: Got empty outputShape");

    const auto inputChannelsIndex = vpux::Dims4D::Act::C.ind();
    const auto inputChannels = inputShape[inputChannelsIndex];
    const auto inputHeightIndex = vpux::Dims4D::Act::H.ind();
    const auto inputHeight = inputShape[inputHeightIndex];
    const auto inputWidthIndex = vpux::Dims4D::Act::W.ind();
    const auto inputWidth = inputShape[inputWidthIndex];

    auto inputCMXShape = inputShape;
    auto outputCMXShape = outputShape;
    auto paddedInputCMXShape = inputShape;

    const auto outputLayout = oduPermutationToLayout(testDesc.getODUPermutation());
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

    auto padAuto16bit = false;
    auto padAuto8bit = false;
    auto padAuto = false;

    auto inputStorageType = inputType;
    if (inputType.dyn_cast<mlir::quant::QuantizedType>()) {
        inputStorageType = mlir::quant::QuantizedType::castToStorageType(inputStorageType);
    }

    auto asIntegerType = inputStorageType.dyn_cast<mlir::IntegerType>();
    auto asFloatType = inputStorageType.dyn_cast<mlir::FloatType>();

    if (asIntegerType) {
        padAuto8bit = (asIntegerType.getWidth() == 8) && inputChannels < 16;
    } else if (asFloatType) {
        padAuto16bit = (asFloatType.getWidth() == 16) && inputChannels < 10;
    }

    padAuto = padAuto16bit || padAuto8bit;

    const auto alignmentRequirement = 16;
    const auto subLineLength = 4;
    const auto isCompressedFormatEnabled = inputChannels <= subLineLength;
    const auto isInputPaddingRequired = padAuto ? false : inputChannels < alignmentRequirement;

    mlir::UnitAttr inputChannelsCompression = nullptr;

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

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    auto finalOutputShape = outputShape;
    nb::OutputLayer reduceOutput;
    llvm::SmallVector<std::int64_t> reduceOutputShape;
    // reduceMax & reduceMin  take place on the ODU, so an identity conv is used as a normal NCE task, to provide an
    // input for the ODU
    if (reductionType == vpux::VPUIP::NCETaskType::CONV) {
        reduceOutput = reduceOutLayer.output.front();
        reduceOutputShape = llvm::SmallVector<std::int64_t>(reduceOutput.shape.begin(), reduceOutput.shape.end());
        finalOutputShape = reduceOutputShape;
    }

    auto ndOutputType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, finalOutputShape, outputType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();
    const auto outputParamType = ndOutputType.changeDimsOrder(outputLayout);
    llvm::SmallVector<mlir::Type, 2> inputTypes;
    std::vector<mlir::Type> outputTypes = {outputParamType};
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(outputParamType);
    if (reduceOutLayer.isMultiTile) {
        inputTypes.push_back(outputParamType);
        outputTypes.push_back(outputParamType);
    }

    const auto funcType = builder.getFunctionType(llvm::ArrayRef(inputTypes), ArrayRef(outputTypes));

    auto function = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("reduce_op_{0}_{1}", inputType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    mlir::BlockArgument functionOutput0, functionOutput1;
    auto functionInput = function.getArgument(0);
    functionOutput0 = function.getArgument(1);
    if (reduceOutLayer.isMultiTile) {
        functionOutput1 = function.getArgument(2);
    }

    auto inputStrides = vpux::getStrides(functionInput);

    if (isInputPaddingRequired) {
        inputStrides[vpux::Dims4D::Act::W] =
                inputStrides[vpux::Dims4D::Act::C] * (isCompressedFormatEnabled ? subLineLength : alignmentRequirement);
        inputStrides[vpux::Dims4D::Act::H] = inputStrides[vpux::Dims4D::Act::W] * inputShape[inputWidthIndex];
        inputStrides[vpux::Dims4D::Act::N] = inputStrides[vpux::Dims4D::Act::H] * inputShape[inputHeightIndex];
    }

    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);

    auto paddedInputCMX = inputCMX;
    if (isInputPaddingRequired) {
        paddedInputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedInputCMXShape,
                                               inputType, DimsOrder::NHWC, 0, INPUT_CMX_OFFSET);
    }
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);

    mlir::Value weightsValue, weightsTableValue;
    vpux::VPURT::DeclareBufferOp weightsCMX, weightsTableCMX, reduceOutputCMX, reduceOutputCMX1;
    mlir::Value maxPerXYValue, minPerXYValue;
    // max & min reduction is done within ODU, so identity CONV task is used before a reduce op
    // for CONV, weights and weights table are created
    vpux::Const::DeclareOp weightsDDR, weightsTableDDR;
    llvm::SmallVector<mlir::Value> minMaxPerTensorValues;
    if (reductionType == vpux::VPUIP::NCETaskType::CONV) {
        const auto REDUCE_OUTPUT_CMX_OFFSET =
                vpux::alignValUp(INPUT_CMX_OFFSET + inputCMXSize, static_cast<std::uint64_t>(alignmentRequirement));
        const auto reduceOutputCMXSize = vpux::hwtest::totalTensorSize(reduceOutputShape, outputType);
        auto reduceOutputMemRef =
                getMemRefType(VPURT::BufferSection::CMX_NN, 0, reduceOutputShape, outputType, outputLayout);
        auto ndReduceOutputCMXpadded = reduceOutputMemRef.cast<vpux::NDTypeInterface>();
        reduceOutputCMX =
                createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, reduceOutputShape, outputType,
                                      outputLayout, ndReduceOutputCMXpadded.getStrides(), 0, REDUCE_OUTPUT_CMX_OFFSET);

        if (reduceOutLayer.isMultiTile) {
            auto reduceOutputMemRef1 =
                    getMemRefType(VPURT::BufferSection::CMX_NN, 1, reduceOutputShape, outputType, outputLayout);
            auto ndReduceOutputCMXpadded1 = reduceOutputMemRef1.cast<vpux::NDTypeInterface>();
            reduceOutputCMX1 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, reduceOutputShape,
                                                     outputType, outputLayout, ndReduceOutputCMXpadded1.getStrides(), 1,
                                                     REDUCE_OUTPUT_CMX_OFFSET);
        }

        const auto weights = testDesc.getWeightLayers().front();
        const llvm::SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
        VPUX_THROW_UNLESS(!weightsShape.empty(), "buildReductionTest: Got empty weightsShape");
        const char* weightsFileName = "weights.dat";
        const auto weightsSize = vpux::hwtest::totalTensorSize(weightsShape, inputStorageType);
        const auto WEIGHTS_CMX_OFFSET =
                vpux::alignValUp(REDUCE_OUTPUT_CMX_OFFSET + reduceOutputCMXSize, static_cast<std::uint64_t>(alignment));

        const auto weightsValues = generateWeights(builder, weightsShape, inputType, ctx, weightsFileName);
        auto weightsAttribute = generateDefaultWeightsAttr(weightsValues, inputType);

        const auto weightsDDRType =
                getMemRefType(VPURT::BufferSection::Constant, weightsShape, inputType, DimsOrder::NHWC);
        auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();

        weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType,
                                                                    std::move(weightsAttribute));

        weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, inputType,
                                           vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);

        const auto WEIGHTSTABLE_CMX_OFFSET =
                vpux::alignValUp(WEIGHTS_CMX_OFFSET + weightsSize, static_cast<std::uint64_t>(alignment));
        VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                          "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                          WEIGHTSTABLE_CMX_OFFSET);
        const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};
        const auto int32 = builder.getIntegerType(32, true);
        auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];
        if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
            weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
        }

        const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
        const auto sparsityPtrStep = 0;
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(testDesc.getArchitecture());
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(testDesc.getArchitecture());

        // using input channels as output channels, as conv output is identical to input
        // actual output channels reflect the reduce operation
        const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
                static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
                inputShape[1], inputType);

        const auto weightsTableDDRMemRef =
                getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
        const auto weightsTableValues =
                mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::ArrayRef<std::int32_t>(weightsTable));
        auto weightsTableContentAttr = vpux::Const::ContentAttr::get(
                weightsTableValues, Const::ContentSetup(weightsTableDDRType).reorder(vpux::DimsOrder::NHWC));
        weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsTableDDRMemRef,
                                                                         std::move(weightsTableContentAttr));
        weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

        mlir::UnitAttr reduceMaxPerXYAttr = nullptr;
        mlir::UnitAttr reduceMinPerXYAttr = nullptr;
        mlir::UnitAttr reduceMinMaxPerTensorAttr = nullptr;

        if (reduceOutLayer.doReduceMaxPerXY) {
            reduceMaxPerXYAttr = mlir::UnitAttr::get(builder.getContext());
            maxPerXYValue = reduceOutputCMX.getBuffer();
        } else if (reduceOutLayer.doReduceMinPerXY) {
            reduceMinPerXYAttr = mlir::UnitAttr::get(builder.getContext());
            minPerXYValue = reduceOutputCMX.getBuffer();
        } else if (reduceOutLayer.doReduceMinMaxPerTensor) {
            reduceMinMaxPerTensorAttr = mlir::UnitAttr::get(builder.getContext());
            minMaxPerTensorValues.push_back(reduceOutputCMX.getBuffer());
            if (reduceOutLayer.isMultiTile) {
                minMaxPerTensorValues.push_back(reduceOutputCMX1.getBuffer());
            }
        }
    }

    auto outputCMXpadded = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outputCMXShape, outputType, outputLayout);
    auto ndOutputCMXpadded = outputCMXpadded.cast<vpux::NDTypeInterface>();
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           outputLayout, ndOutputCMXpadded.getStrides(), 0, OUTPUT_CMX_OFFSET);

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
    if (reductionType == vpux::VPUIP::NCETaskType::CONV) {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), builder.getUnknownLoc(),
                weightsDDR.getOperation()->getResult(0), weightsCMX.getOperation()->getResult(0), 0);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()), builder.getUnknownLoc(),
                weightsTableDDR.getOperation()->getResult(0), weightsTableCMX.getOperation()->getResult(0), 0);
        weightsValue = weightsCMX.getBuffer();
        weightsTableValue = weightsTableCMX.getBuffer();

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              reduceOutputCMX.getOperation()->getResult(0), functionOutput0, 0);
        if (reduceOutLayer.isMultiTile) {
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                                  mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                                  reduceOutputCMX1.getOperation()->getResult(0), functionOutput1, 1);
        }
    } else {
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              outputCMX.getOperation()->getResult(0), functionOutput0, 0);
    }

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    llvm::SmallVector<std::int64_t> kernelShape = {1, 1};
    const auto kernelSize = getIntArrayAttr(ctx, kernelShape);

    auto sparsityPattern = (isInputPaddingRequired || padAuto) ? ((1 << inputChannels) - 1) : 0;

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, barrier0.getBarrier(), barrier1.getBarrier(), builder.getUnknownLoc(),
            paddedInputCMX.getBuffer(), /*input_sparsity_map=*/nullptr, /*input_storage_element_table=*/nullptr,
            /*weights*/ weightsValue, /*weights_sparsity_map=*/nullptr, /*weightsTable*/ weightsTableValue,
            /*spr_lookup_table=*/nullptr, paddedInputCMX.getBuffer(),
            /*parent_input_sparsity_map=*/nullptr, /*parent_input_storage_element_table=*/nullptr,
            outputCMX.getBuffer(), /*parent_output_sparsity_map=*/nullptr, outputCMX.getBuffer(),
            /*output_sparsity_map=*/nullptr, /*profiling_data=*/nullptr,
            /*max_per_xy=*/maxPerXYValue, /*min_per_xy=*/minPerXYValue, /*min_max_per_tensor=*/minMaxPerTensorValues,
            reductionType, kernelSize, strides, kernelPaddings, nullptr,
            vpux::getIntAttr(builder.getContext(), sparsityPattern), nullptr, nullptr, inputChannelsCompression,
            /*is_zero_offset_weights_table*/ nullptr, /*is_superdense*/ nullptr,
            /*is_inplace*/ nullptr, /*input_se_size*/ nullptr, /*output_se_size*/ nullptr,
            /*isPermuteQuantize*/ nullptr, /*isSmallKernelOptimized*/ nullptr, /*mpeEngineAttr*/ nullptr,
            /*eltwiseType*/ nullptr);

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

    double clampLow = std::numeric_limits<float>::lowest();
    double clampHigh = std::numeric_limits<float>::max();
    double zp = 0.0;
    if (const auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;
        zp = zps.front();
        clampLow = static_cast<double>(outElemQType.getStorageTypeMin() - zps.front());
        clampHigh = static_cast<double>(outElemQType.getStorageTypeMax() - zps.front());
    }
    auto ppeAttr = VPU::PPEFpAttr::get(ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP),
                                       vpux::getFPAttr(ctx, clampLow), vpux::getFPAttr(ctx, clampHigh),
                                       /* scale = */ vpux::getFPAttr(ctx, 1.0),
                                       /* pReluAlpha = */ vpux::getFPArrayAttr(ctx, ArrayRef{1.0}),
                                       /* bias = */ vpux::getFPAttr(ctx, 0.0), /* adder = */ vpux::getFPAttr(ctx, zp),
                                       /* in1Mult = */ nullptr,
                                       /* in2Mult = */ nullptr,
                                       /* sprLUT = */ nullptr);
    nceTask.addPPETask(functionBuilder, ppeAttr);

    llvm::SmallVector<mlir::BlockArgument> functionOutputs = {functionOutput0};
    llvm::SmallVector<mlir::Type> functionOutputTypes = {
            getTensorType(ShapeRef(finalOutputShape), outputType, outputLayout, nullptr)};
    if (reduceOutLayer.isMultiTile) {
        functionOutputs.push_back(functionOutput1);
        functionOutputTypes.push_back(getTensorType(ShapeRef(finalOutputShape), outputType, outputLayout, nullptr));
    }
    functionBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange(functionOutputs));
    module.dump();
    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               llvm::ArrayRef<mlir::Type>(functionOutputTypes));
}

}  // namespace hwtest
}  // namespace vpux
