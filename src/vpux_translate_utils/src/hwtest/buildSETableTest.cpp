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
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

namespace {
std::vector<int32_t> computeSeTable(const nb::SETablePattern& seTablePattern, ArrayRef<int64_t> shape,
                                    mlir::Type actType) {
    const auto channels = shape[Dims4D::Act::C.ind()];
    const auto height = shape[Dims4D::Act::H.ind()];
    const auto width = shape[Dims4D::Act::W.ind()];
    auto seTableContent = std::vector<int32_t>(height * width, 0);
    const auto numBytesPerWidth = channels * actType.getIntOrFloatBitWidth() / CHAR_BIT;
    const int64_t SHIFT_FOR_STORAGE_ELEMENT = 9;

    switch (seTablePattern) {
    case nb::SETablePattern::SwitchLines:
        for (int64_t h = 0; h < height; h += 2) {
            for (int64_t w = 0; w < width; w++) {
                const auto offsetLine0 = h * width + w;
                const auto offsetLine1 = (h + 1) * width + w;
                seTableContent[offsetLine0] =
                        static_cast<int32_t>(((offsetLine1 * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT);
                seTableContent[offsetLine1] =
                        static_cast<int32_t>(((offsetLine0 * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT);
            }
        }

        if (height % 2 == 1) {
            for (int64_t w = 0; w < width; w++) {
                const auto elemOffset = (height - 1) * width + w;
                seTableContent[elemOffset] =
                        static_cast<int32_t>(((elemOffset * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT);
            }
        }
        break;
    case nb::SETablePattern::OriginalInput:
        for (int64_t h = 0; h < height; h++) {
            for (int64_t w = 0; w < width; w++) {
                const auto offset = h * width + w;
                seTableContent[offset] =
                        static_cast<int32_t>(((offset * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT);
            }
        }
        break;
    default:
        VPUX_THROW("Wrong Storage Element Table pattern.");
        break;
    }

    return seTableContent;
}
}  // namespace

void buildSETableTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                      Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);

    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    constexpr int64_t CLUSTER_NUM = 0;
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);
    const auto int1 = builder.getI1Type();

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();
    const auto seTableParams = testDesc.getSETableParams();
    const auto seTableElementType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signless);

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildSETableTest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildSETableTest: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildSETableTest: Got empty weightsShape");

    const SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};
    const SmallVector<std::int64_t> seTableShape{1, 1, inputShape[2], inputShape[3]};

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    const auto alignmentRequirement = 16;

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);
    const auto wtableCMXSize = vpux::hwtest::totalTensorSize(weightsTableShape, int32);
    const auto seTableCMXSize = vpux::hwtest::totalTensorSize(seTableShape, int32);

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

    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    const auto SE_TABLE_CMX_OFFSET = WEIGHTSTABLE_CMX_OFFSET + wtableCMXSize;
    VPUX_THROW_UNLESS(SE_TABLE_CMX_OFFSET % alignment == 0, "SE_TABLE_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, SE_TABLE_CMX_OFFSET);

    const auto SP_MAP_CMX_OFFSET = SE_TABLE_CMX_OFFSET + seTableCMXSize;
    VPUX_THROW_UNLESS(SP_MAP_CMX_OFFSET % alignment == 0, "SP_MAP_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, SP_MAP_CMX_OFFSET);

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);

    const auto returnTypesVec = SmallVector<mlir::Type>({outputParamType});
    const auto argTypesVec = SmallVector<mlir::Type>({inputParamType, outputParamType});
    const auto funcType = builder.getFunctionType(argTypesVec, returnTypesVec);

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("se_table_dpu_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto fcnBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());
    auto functionInput = function.getArgument(0);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttributeSetup = vpux::Const::ContentAttr::transform(weightsValues);
    weightsAttributeSetup = weightsAttributeSetup.reorder(vpux::DimsOrder::OYXI);

    auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>();

    if (qty != nullptr) {
        if (qty.getStorageType().isInteger(4)) {
            weightsAttributeSetup = weightsAttributeSetup.bitPack(4);
        }
        weightsAttributeSetup = weightsAttributeSetup.quantCast(qty);
    }

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = functionInput.getType().cast<vpux::NDTypeInterface>().getStrides();

    auto weightsDDR = fcnBuilder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, weightsAttributeSetup.get());

    auto weightsCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                            DimsOrder::OYXI, weightsStrides, CLUSTER_NUM, WEIGHTS_CMX_OFFSET);
    auto inputCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          DimsOrder::NHWC, inputStrides, CLUSTER_NUM, INPUT_CMX_OFFSET);

    // Create sparsity map filled with 1s
    const auto numElems = inputCMX.getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto sparseMapContent = std::vector<char>(numElems / CHAR_BIT, static_cast<char>(0xFF));
    auto sparseMapValues = mlir::DenseElementsAttr::getFromRawBuffer(mlir::RankedTensorType::get(inputShape, int1),
                                                                     llvm::ArrayRef<char>(sparseMapContent));

    auto sparsityMapDDRType = getMemRefType(VPURT::BufferSection::Constant, inputShape, int1, DimsOrder::OIYX);
    auto sparsityMapTypeIf = sparsityMapDDRType.cast<vpux::NDTypeInterface>();

    auto sparsityMapDDR = fcnBuilder.create<vpux::Const::DeclareOp>(loc, sparsityMapDDRType,
                                                                    vpux::Const::ContentAttr::get(sparseMapValues));

    auto sparsityMapCMX =
            createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, sparsityMapTypeIf.getShape().raw(),
                                  sparsityMapTypeIf.getElementType(), sparsityMapTypeIf.getDimsOrder(),
                                  sparsityMapTypeIf.getStrides(), CLUSTER_NUM, SP_MAP_CMX_OFFSET);

    // Create SE table and fill it according to pattern
    auto seTableContent = computeSeTable(seTableParams.seTablePattern, inputShape, inputType);
    auto seTableValues = mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(seTableShape, seTableElementType),
                                                      llvm::ArrayRef<int32_t>(seTableContent));

    auto seTableDDRType =
            getMemRefType(VPURT::BufferSection::Constant, seTableShape, seTableElementType, DimsOrder::NHWC);
    auto seTableStrides = seTableDDRType.cast<vpux::NDTypeInterface>().getStrides();

    auto seTableConstAttr = vpux::Const::ContentAttr::transform(seTableValues).reorder(vpux::DimsOrder::OYXI).get();

    auto seTableDDR = fcnBuilder.create<vpux::Const::DeclareOp>(loc, seTableDDRType, std::move(seTableConstAttr));

    auto seTableCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, seTableShape, seTableElementType,
                                            DimsOrder::NHWC, seTableStrides, CLUSTER_NUM, SE_TABLE_CMX_OFFSET);

    auto inputSeSizeAttr = getIntAttr(ctx, inputShape[Dims4D::Act::C.ind()]);

    // Create weights table
    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
        weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
    }

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
    auto weightsTableDDR = fcnBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::transform(weightsTableValues).reorder(vpux::DimsOrder::NHWC).get());

    auto weightsTableCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                 DimsOrder::NHWC, CLUSTER_NUM, WEIGHTSTABLE_CMX_OFFSET);

    const auto outputMemRefType =
            getMemRefType(VPURT::BufferSection::CMX_NN, outputCMXShape, outputType, DimsOrder::NHWC);
    const auto outputTypeIf = outputMemRefType.cast<vpux::NDTypeInterface>();

    VPURT::DeclareBufferOp outCMXBuffer = createDeclareTensorOp(
            fcnBuilder, VPURT::BufferSection::CMX_NN, outputCMXShape, outputTypeIf.getElementType(),
            outputTypeIf.getDimsOrder(), CLUSTER_NUM, OUTPUT_CMX_OFFSET);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(fcnBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto updateBarrier = fcnBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);

    // Create DMAs for input act, weights, weights table, sparsity map and SE table
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, waitWLMBarrier, mlir::ValueRange(updateBarrier.getBarrier()), loc,
                                          functionInput, inputCMX, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, weightsDDR, weightsCMX, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, weightsTableDDR, weightsTableCMX, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, sparsityMapDDR, sparsityMapCMX, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, seTableDDR, seTableCMX, 0);

    auto waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);
    auto sparsityMap = !seTableParams.seOnlyEn ? sparsityMapCMX.getBuffer() : nullptr;
    // Create NCEClusterTaskOp
    updateBarrier = fcnBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            fcnBuilder, mlir::ValueRange(waitBarrier.getBarrier()), mlir::ValueRange(updateBarrier.getBarrier()), loc,
            inputCMX.getBuffer(), sparsityMap, seTableCMX.getBuffer(), weightsCMX,
            /*weights_sparsity_map=*/nullptr, weightsTableCMX, /*instruction_list_table=*/nullptr,
            /*spr_lookup_table*/ nullptr, inputCMX.getBuffer(), sparsityMapCMX.getBuffer(), seTableCMX.getBuffer(),
            outCMXBuffer, /*parent_output_sparsity_map=*/nullptr, outCMXBuffer,
            /*output_sparsity_map_buff=*/nullptr, /*profiling_data=*/nullptr, vpux::VPUIP::NCETaskType::CONV,
            kernelSize, strides, kernelPaddings,
            /*is_continued=*/nullptr,
            /*cm_sp_pattern=*/nullptr,
            /*is_segmented=*/nullptr,
            /*out_channel_offset=*/nullptr,
            /*input_channels_compression=*/nullptr,
            /*is_superdense=*/nullptr,
            /*is_inplace=*/nullptr,
            /*input_se_size=*/inputSeSizeAttr,
            /*output_se_size=*/nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    nceTask.addDPUTask(fcnBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);

    waitBarrier = updateBarrier;
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = fcnBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++,
                                                                           testDesc.getWLMParams().isWLMPartialEnabled);

    // Create CMX2DDR DMAs from each cluster the output was broadcasted to

    auto functionOutput = function.getArgument(1);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                          mlir::ValueRange(finalBarrier.getBarrier()), loc, outCMXBuffer,
                                          functionOutput, 0);

    fcnBuilder.create<mlir::func::ReturnOp>(loc, SmallVector<mlir::Value>{functionOutput});

    mlir::PassManager pmBuilderEnd(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (conv.compress) {
        pmBuilderEnd.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderEnd.run(module)), "Compilation failed");

    auto outputTensorTypeVec =
            SmallVector<mlir::Type>{getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)};
    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)}, outputTensorTypeVec);
}

}  // namespace hwtest
}  // namespace vpux
