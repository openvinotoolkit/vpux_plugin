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
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

void buildDifferentClustersDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                   mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();
    const auto DPUTaskParams = testDesc.getDPUTaskParams();
    const auto inputCluster = DPUTaskParams.inputCluster;
    const SmallVector<std::int64_t> outputClusters{DPUTaskParams.outputClusters.begin(),
                                                   DPUTaskParams.outputClusters.end()};
    const auto weightsCluster = DPUTaskParams.weightsCluster;
    const auto weightsTableCluster = DPUTaskParams.weightsTableCluster;

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildDifferentClustersDPUTest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildDifferentClustersDPUTest: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildDifferentClustersDPUTest: Got empty weightsShape");
    VPUX_THROW_UNLESS(!weightsTableShape.empty(), "buildDifferentClustersDPUTest: Got empty weightsTableShape");

    // set runtime resources

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    const auto maxClusterOutput = static_cast<size_t>(*std::max_element(outputClusters.begin(), outputClusters.end()));
    const auto numTiles = std::max({inputCluster, weightsCluster, weightsTableCluster, maxClusterOutput}) + 1;
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.setNumberOfDPUGroups(numTiles);
    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    const auto alignmentRequirement = 16;

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);

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

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);

    const auto returnTypesVec = SmallVector<mlir::Type>(outputClusters.size(), outputParamType);
    auto argTypesVec = SmallVector<mlir::Type>({inputParamType});
    argTypesVec.append(returnTypesVec.begin(), returnTypesVec.end());
    const auto funcType = builder.getFunctionType(argTypesVec, returnTypesVec);

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("different_clusters_dpu_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());
    auto functionInput = function.getArgument(0);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = generateDefaultWeightsAttr(weightsValues, weightsType);

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = functionInput.getType().cast<vpux::NDTypeInterface>().getStrides();

    auto weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                            DimsOrder::OYXI, weightsStrides, weightsCluster, WEIGHTS_CMX_OFFSET);
    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          DimsOrder::NHWC, inputStrides, inputCluster, INPUT_CMX_OFFSET);

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, std::move(weightsAttribute));

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
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues,
                                          Const::ContentSetup(weightsTableDDRType).reorder(vpux::DimsOrder::NHWC)));

    auto weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                 int32, DimsOrder::NHWC, weightsTableCluster, WEIGHTSTABLE_CMX_OFFSET);

    const auto outputMemRefType =
            getMemRefType(VPURT::BufferSection::CMX_NN, outputCMXShape, outputType, DimsOrder::NHWC);
    const auto outputTypeIf = outputMemRefType.cast<vpux::NDTypeInterface>();

    SmallVector<vpux::VPURT::DeclareBufferOp> outCMXBufferVec;
    outCMXBufferVec.reserve(outputClusters.size());
    for (std::size_t idx = 0; idx < outputClusters.size(); idx++) {
        outCMXBufferVec.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputCMXShape,
                                                        outputTypeIf.getElementType(), outputTypeIf.getDimsOrder(),
                                                        outputClusters[idx], OUTPUT_CMX_OFFSET));
    }

    auto nceClusterTaskOutBuffer = outCMXBufferVec.front().getBuffer();
    if (outputClusters.size() > 1) {
        // Create distributed buffer for CMX output
        const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        const auto numClustersAttr = getIntAttr(ctx, outputClusters.size());
        const auto distributedAttr = VPU::DistributionInfoAttr::get(ctx, distributionModeAttr, nullptr, nullptr,
                                                                    nullptr, nullptr, numClustersAttr, nullptr, nullptr,
                                                                    nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto orderAttr = mlir::AffineMapAttr::get(outputTypeIf.getDimsOrder().toAffineMap(ctx));
        const auto elemStrides = to_small_vector(outputTypeIf.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / outputTypeIf.getElemTypeSize().count();
                                                 }));
        const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
        const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr, /*allocSize=*/nullptr, ctx);

        const auto dimsSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyMemoryKind(outputTypeIf.getMemoryKind()));

        auto outDistributedCMXType = VPUIP::DistributedBufferType::get(
                ctx, outputCMXShape, outputTypeIf.getElementType(), layout, dimsSpace, distributedAttr);

        auto outDistributedCMX = createDeclareTensorOp(functionBuilder, outDistributedCMXType,
                                                       VPURT::BufferSection::CMX_NN, outputClusters, OUTPUT_CMX_OFFSET);
        nceClusterTaskOutBuffer = outDistributedCMX.getBuffer();
    }

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);

    // Create DMAs for input act, weights and weights table
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, waitWLMBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                          loc, functionInput, inputCMX, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                          mlir::ValueRange(updateBarrier.getBarrier()), loc, weightsDDR, weightsCMX, 0);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                          mlir::ValueRange(updateBarrier.getBarrier()), loc, weightsTableDDR,
                                          weightsTableCMX, 0);

    auto waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    // Create NCEClusterTaskOp
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);
    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()), mlir::ValueRange(updateBarrier.getBarrier()),
            loc, inputCMX.getBuffer(), weightsCMX.getBuffer(), weightsTableCMX.getBuffer(),
            /*spr_lookup_table=*/nullptr, inputCMX.getBuffer(), nceClusterTaskOutBuffer, nceClusterTaskOutBuffer,
            vpux::VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings, nullptr, nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    nceTask.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);

    waitBarrier = updateBarrier;
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, freeBarrierId++, testDesc.getWLMParams().isWLMPartialEnabled);

    // Create CMX2DDR DMAs from each cluster the output was broadcasted to
    auto functionOutputs = SmallVector<mlir::Value>(outputClusters.size());
    for (unsigned int idx = 0; idx < static_cast<unsigned int>(outputClusters.size()); idx++) {
        auto functionOutput = function.getArgument(1 + idx);
        functionOutputs[idx] = functionOutput;
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), loc, outCMXBufferVec[idx],
                                              functionOutput, 0);
    }

    functionBuilder.create<mlir::func::ReturnOp>(loc, functionOutputs);

    module.dump();

    mlir::PassManager pmBuilderEnd(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (conv.compress) {
        pmBuilderEnd.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderEnd.run(module)), "Compilation failed");

    auto outputTensorType = getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr);
    const auto outputTensorTypesVec = SmallVector<mlir::Type>(outputClusters.size(), outputTensorType);
    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)}, outputTensorTypesVec);
}

}  // namespace hwtest
}  // namespace vpux
