//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

namespace {

struct DistributedAttrs {
    VPU::DistributedTensorAttr parentInDistrAttr;
    VPU::DistributedTensorAttr parentOutDistrAttr;
    VPU::DistributedTensorAttr weightsDistrAttr;

    DistributedAttrs(mlir::MLIRContext* ctx, nb::SegmentationType sType, const std::size_t numClusters, bool broadcast,
                     mlir::Type outputType, const int64_t inputWidth) {
        const auto duplicatedDistrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        const auto segmentedDistrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
        const auto numClustersAttr = getIntAttr(ctx, numClusters);

        if (sType == nb::SegmentationType::SOK) {
            // Ensure tiles have the channel num aligned
            const auto alignedChannelNum = VPU::NCEInvariant::getAlignment(outputType);
            mlir::ArrayAttr outNumTiles = getIntArrayAttr(ctx, SmallVector<std::size_t>{1, numClusters, 1, 1});
            mlir::ArrayAttr wNumTiles = getIntArrayAttr(ctx, SmallVector<std::size_t>{numClusters, 1, 1, 1});
            mlir::ArrayAttr outAlignment = getIntArrayAttr(ctx, SmallVector<int64_t>{{1, alignedChannelNum, 1, 1}});
            mlir::ArrayAttr wAlignment = getIntArrayAttr(ctx, SmallVector<int64_t>{alignedChannelNum, 1, 1, 1});

            parentInDistrAttr = VPU::DistributedTensorAttr::get(ctx, duplicatedDistrModeAttr, nullptr, nullptr, nullptr,
                                                                nullptr, numClustersAttr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);

            const VPU::DistributionMode outputMode =
                    (broadcast == true ? VPU::DistributionMode::DUPLICATED : VPU::DistributionMode::NONE) |
                    VPU::DistributionMode::SEGMENTED;
            parentOutDistrAttr = VPU::DistributedTensorAttr::get(
                    ctx, VPU::DistributionModeAttr::get(ctx, outputMode), outNumTiles, nullptr, nullptr, nullptr,
                    numClustersAttr, outAlignment, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            weightsDistrAttr = VPU::DistributedTensorAttr::get(ctx, segmentedDistrModeAttr, wNumTiles, nullptr, nullptr,
                                                               nullptr, numClustersAttr, wAlignment, nullptr, nullptr,
                                                               nullptr, nullptr, nullptr, nullptr);
        }

        if (sType == nb::SegmentationType::SOH) {
            mlir::ArrayAttr numTiles = getIntArrayAttr(ctx, SmallVector<std::size_t>{1, 1, numClusters, 1});
            const auto alignment = getIntArrayAttr(
                    ctx, SmallVector<int64_t>{
                                 1, 1, VPU::getSOHPerClusterHeightAlignment(inputWidth, /*isInputSparse=*/false), 1});
            parentInDistrAttr = VPU::DistributedTensorAttr::get(ctx, segmentedDistrModeAttr, numTiles, nullptr, nullptr,
                                                                nullptr, numClustersAttr, alignment, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr);

            const VPU::DistributionMode outputMode =
                    (broadcast == true ? VPU::DistributionMode::MULTICASTED : VPU::DistributionMode::NONE) |
                    VPU::DistributionMode::SEGMENTED;
            parentOutDistrAttr = VPU::DistributedTensorAttr::get(
                    ctx, VPU::DistributionModeAttr::get(ctx, outputMode), numTiles, nullptr, nullptr, nullptr,
                    numClustersAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

            weightsDistrAttr = VPU::DistributedTensorAttr::get(ctx, duplicatedDistrModeAttr, nullptr, nullptr, nullptr,
                                                               nullptr, numClustersAttr, nullptr, nullptr, nullptr,
                                                               nullptr, nullptr, nullptr, nullptr);
        }
    }
};

VPURT::DeclareBufferOp createParentBuffer(mlir::MLIRContext* ctx, mlir::OpBuilder& builder,
                                          VPU::DistributedTensorAttr distrTensorAttr, mlir::Type tensorType,
                                          ArrayRef<int64_t> tensorShape, const DimsOrder dimsOrder,
                                          ArrayRef<int64_t> clusters, const std::size_t offset) {
    const auto cmxMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, tensorShape, tensorType, dimsOrder);
    const auto tensorTypeIf = cmxMemRefType.cast<vpux::NDTypeInterface>();

    const auto orderAttr = mlir::AffineMapAttr::get(tensorTypeIf.getDimsOrder().toAffineMap(ctx));
    const auto elemStrides = to_small_vector(tensorTypeIf.getStrides() | transformed([&](Bit stride) {
                                                 return stride.count() / tensorTypeIf.getElemTypeSize().count();
                                             }));
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, ctx);

    const auto dimsSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyMemoryKind(tensorTypeIf.getMemoryKind()));

    auto distributedCMXType = VPUIP::DistributedBufferType::get(ctx, tensorShape, tensorTypeIf.getElementType(), layout,
                                                                dimsSpace, distrTensorAttr);

    return createDeclareTensorOp(builder, distributedCMXType, VPURT::BufferSection::CMX_NN, clusters, offset);
}

// Create Weights DDR & CMX buffers and DMAs for them
SmallVector<VPURT::DeclareBufferOp> handleWeights(mlir::OpBuilder& builder, VPURT::DeclareBufferOp parentTensor,
                                                  Const::ContentAttr weightsContent, ArrayRef<int64_t> clusters,
                                                  std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier) {
    const auto numClusters = clusters.size();
    auto loc = builder.getUnknownLoc();

    auto distributedBufferType = parentTensor.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto tensorTypeIf = distributedBufferType.cast<NDTypeInterface>();

    SmallVector<vpux::VPURT::DeclareBufferOp> weightsCMXBufferVec;
    weightsCMXBufferVec.reserve(numClusters);

    // Create CMX buffers for weights in each tile
    const auto perClusterShapes = distributedBufferType.getPerClusterMemoryShapes();
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        const auto weightsCMXMemRefType =
                getMemRefType(VPURT::BufferSection::CMX_NN, clusters[idx], perClusterShapes[idx],
                              tensorTypeIf.getElementType(), tensorTypeIf.getDimsOrder());
        weightsCMXBufferVec.push_back(createDeclareTensorOp(builder, weightsCMXMemRefType, VPURT::BufferSection::CMX_NN,
                                                            clusters[idx], offset));
    }

    // SOH case, weights are duplicated in their entirety in each tile
    const auto dmaDuplicatedBuffers = [&]() {
        // Create DDR buffer for weights
        const auto weightsDDRType =
                getMemRefType(VPURT::BufferSection::Constant, llvm::to_vector(tensorTypeIf.getShape()),
                              tensorTypeIf.getElementType(), tensorTypeIf.getDimsOrder());
        auto weightsDDRBuffer = builder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, weightsContent);

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()),
                                              loc, weightsDDRBuffer, parentTensor, 0);
    };

    // SOK case, weights are split over Output Channels (K) dim, with a slice in each tile
    const auto dmaSegmentedBuffers = [&]() {
        Shape weightsOffset{0, 0, 0, 0};
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            // Create a DDR buffer for each slice of the weights
            const auto weightsDDRMemRefType =
                    getMemRefType(VPURT::BufferSection::Constant, llvm::to_vector(perClusterShapes[idx]),
                                  tensorTypeIf.getElementType(), tensorTypeIf.getDimsOrder());
            // Create weights slice by using subview on the full weights content
            auto weightsDDRBuffer = builder.create<vpux::Const::DeclareOp>(
                    loc, weightsDDRMemRefType, weightsContent.subview(weightsOffset, perClusterShapes[idx]));
            weightsOffset[vpux::Dims4D::Filter::OC] += perClusterShapes[idx][vpux::Dims4D::Filter::OC];

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, weightsDDRBuffer,
                                                  weightsCMXBufferVec[idx], 0);
        }
    };

    switch (distributedBufferType.getDistribution().getMode().getValue()) {
    case VPU::DistributionMode::DUPLICATED:
        dmaDuplicatedBuffers();
        break;
    case VPU::DistributionMode::SEGMENTED:
        dmaSegmentedBuffers();
        break;
    default:
        VPUX_THROW("DistributionMode unsupported for weights");
    };

    offset += distributedBufferType.getTotalAllocSize().count();
    return weightsCMXBufferVec;
}

// Create WeightsTable DDR & CMX buffers and DMAs for them
SmallVector<VPURT::DeclareBufferOp> handleWeightsTable(mlir::MLIRContext* ctx, mlir::OpBuilder& builder,
                                                       VPU::ArchKind arch,
                                                       VPUIP::DistributedBufferType weightsDistrBufferType,
                                                       mlir::Type inputType, mlir::Type outputType,
                                                       mlir::Type wtableElemType, ArrayRef<int64_t> wtableShape,
                                                       ArrayRef<int64_t> clusters, const std::size_t& offset,
                                                       VPURT::ConfigureBarrierOp updateBarrier) {
    const auto numClusters = clusters.size();
    auto loc = builder.getUnknownLoc();
    const auto sparsityPtrStep = 0;
    auto weightsStrides = weightsDistrBufferType.getStrides();
    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

    const auto alignmentRequirement = 16;
    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
        weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
    }

    auto weightsTypeIf = weightsDistrBufferType.cast<NDTypeInterface>();

    SmallVector<vpux::VPURT::DeclareBufferOp> wtableCMXBufferVec;
    wtableCMXBufferVec.reserve(numClusters);

    const auto perClusterShapes = weightsDistrBufferType.getPerClusterMemoryShapes();
    // Create CMX buffers for weights table in each tile
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        const SmallVector<int64_t> shape = {
                perClusterShapes[idx][Dims4D::Filter::OC], wtableShape[Dims4D::Filter::IC.ind()],
                wtableShape[Dims4D::Filter::KY.ind()], wtableShape[Dims4D::Filter::KX.ind()]};
        const auto wtableCMXMemRefType =
                getMemRefType(VPURT::BufferSection::CMX_NN, clusters[idx], shape, wtableElemType, DimsOrder::NHWC);
        wtableCMXBufferVec.push_back(createDeclareTensorOp(builder, wtableCMXMemRefType, VPURT::BufferSection::CMX_NN,
                                                           clusters[idx], offset));
    }

    // For SOH case, weights table are duplicated in each tile
    const auto dmaDuplicatedBuffers = [&]() {
        // Create distributed, duplicated buffer in CMX; will be used in DMA task to move same content to each tile
        auto wtableParentBuffer = createParentBuffer(ctx, builder, weightsDistrBufferType.getDistribution(),
                                                     wtableElemType, wtableShape, DimsOrder::NHWC, clusters, offset);
        const auto wtableDDRType =
                getMemRefType(VPURT::BufferSection::Constant, wtableShape, wtableElemType, DimsOrder::NHWC);

        // Create weights table content
        const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, 0,
                static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, arch,
                wtableShape[vpux::Dims4D::Filter::OC.ind()], weightsTypeIf.getElementType());
        auto wtableTensorType = mlir::RankedTensorType::get(wtableShape, wtableElemType);
        const auto weightsTableValues =
                mlir::DenseElementsAttr::get(wtableTensorType, llvm::ArrayRef<std::int32_t>(weightsTable));

        auto weightsDDRBuffer = builder.create<vpux::Const::DeclareOp>(
                loc, wtableDDRType, vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC));

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.getBarrier()),
                                              loc, weightsDDRBuffer, wtableParentBuffer, 0);
    };

    // For SOK case, each tile gets a weight table for the num of output channels it computes
    const auto dmaSegmentedBuffers = [&]() {
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            // Create weights table DDR buffer for each tile
            const auto wtableTypeIf = wtableCMXBufferVec[idx].getType().cast<NDTypeInterface>();
            const auto wtableDDRType =
                    getMemRefType(VPURT::BufferSection::Constant, llvm::to_vector(wtableTypeIf.getShape()),
                                  wtableElemType, DimsOrder::NHWC);

            // Create weights table content for each weights table chunck
            const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                    inputType, outputType, 0,
                    static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                    VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, arch,
                    wtableTypeIf.getShape()[vpux::Dims4D::Filter::OC], weightsTypeIf.getElementType());
            auto wtableTensorType =
                    mlir::RankedTensorType::get(llvm::to_vector(wtableTypeIf.getShape()), wtableElemType);
            const auto weightsTableValues =
                    mlir::DenseElementsAttr::get(wtableTensorType, llvm::ArrayRef<std::int32_t>(weightsTable));

            auto wtableDDRBuffer = builder.create<vpux::Const::DeclareOp>(
                    loc, wtableDDRType,
                    vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC));

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, wtableDDRBuffer,
                                                  wtableCMXBufferVec[idx], 0);
        }
    };

    switch (weightsDistrBufferType.getDistribution().getMode().getValue()) {
    case VPU::DistributionMode::DUPLICATED:
        dmaDuplicatedBuffers();
        break;
    case VPU::DistributionMode::SEGMENTED:
        dmaSegmentedBuffers();
        break;
    default:
        VPUX_THROW("DistributionMode unsupported for weightsTable");
    };

    return wtableCMXBufferVec;
}

}  // namespace

void buildMultiClustersDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                               mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto outputs = testDesc.getOutputLayers();
    const auto outputLayout = oduPermutationToLayout(testDesc.getODUPermutation());
    const auto multiClusterParams = testDesc.getMultiClusterDPUParams();
    const SmallVector<std::int64_t> taskClusters{multiClusterParams.taskClusters.begin(),
                                                 multiClusterParams.taskClusters.end()};
    const auto numClusters = taskClusters.size();

    auto getParentOutputShape = [&outputs, &multiClusterParams]() -> SmallVector<std::int64_t> {
        SmallVector<int64_t> parentShape(outputs.front().shape.begin(), outputs.front().shape.end());
        // If output is broadcasted, each output entry in test config will have the shape of the whole output
        if (multiClusterParams.broadcast == true) {
            return parentShape;
        }

        // When output is not broadcasted, the parent shape can be obtained by summing up the sizes from each
        // output entry in config, along the axis used for splitting
        const auto dim = static_cast<int64_t>(multiClusterParams.segmentation);
        for (std::size_t outIdx = 1; outIdx < outputs.size(); outIdx++) {
            parentShape[dim] += outputs[outIdx].shape[dim];
        }

        return parentShape;
    };

    const SmallVector<std::int64_t> parentInputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const SmallVector<std::int64_t> weightsTableShape{weightsShape[Dims4D::Filter::OC.ind()], 1, 1, 4};
    const auto parentOutputShape = getParentOutputShape();

    VPUX_THROW_UNLESS(!parentInputShape.empty(), "buildMultiClustersDPUTest: Got empty parentInputShape");
    VPUX_THROW_UNLESS(!parentOutputShape.empty(), "buildMultiClustersDPUTest: Got empty parentOutputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildMultiClustersDPUTest: Got empty weightsShape");
    VPUX_THROW_UNLESS(!weightsTableShape.empty(), "buildMultiClustersDPUTest: Got empty weightsTableShape");

    const char* weightsFileName = "weights.dat";

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, parentInputShape, inputType, DimsOrder::NHWC);

    auto getReturnTypesVec = [](ArrayRef<nb::OutputLayer> outputs, mlir::Type outputType, ArrayRef<int64_t> parentShape,
                                const std::size_t numClusters, const bool broadcast,
                                vpux::DimsOrder outputLayout) -> SmallVector<mlir::Type> {
        if (broadcast == true) {
            const auto outputParamType =
                    getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, parentShape, outputType, outputLayout);
            return SmallVector<mlir::Type>(numClusters, outputParamType);
        }

        SmallVector<mlir::Type> returnTypes;
        returnTypes.reserve(numClusters);
        for (const auto& output : outputs) {
            const auto outputParamType = getMemRefType(
                    vpux::VPURT::BufferSection::NetworkOutput,
                    SmallVector<std::int64_t>(output.shape.begin(), output.shape.end()), outputType, outputLayout);
            returnTypes.push_back(outputParamType);
        }

        return returnTypes;
    };

    const auto returnTypesVec = getReturnTypesVec(outputs, outputType, parentOutputShape, numClusters,
                                                  multiClusterParams.broadcast, outputLayout);
    auto argTypesVec = SmallVector<mlir::Type>({inputParamType});
    argTypesVec.append(returnTypesVec.begin(), returnTypesVec.end());
    const auto funcType = builder.getFunctionType(argTypesVec, returnTypesVec);

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("multiple_clusters_dpu_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());
    auto functionInput = function.getArgument(0);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = Const::ContentAttr::get(weightsValues);
    weightsAttribute = weightsAttribute.reorder(DimsOrder::OYXI);

    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto quantizedType = vpux::changeStorageType(qty, weightsAttribute.getType().getElementType());
        weightsAttribute = weightsAttribute.quantCast(quantizedType);
        if (qty.getStorageType().isInteger(4)) {
            weightsAttribute = weightsAttribute.bitPack(4);
        }
    }

    const auto kernelStrides = getIntArrayAttr(ctx, conv.stride);
    const std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    const SmallVector<std::int64_t> kernel = {weightsShape[Dims4D::Filter::KY.ind()],
                                              weightsShape[Dims4D::Filter::KX.ind()]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    const auto distrBufferParams =
            DistributedAttrs(ctx, multiClusterParams.segmentation, numClusters, multiClusterParams.broadcast,
                             outputType, parentInputShape[Dims4D::Act::W.ind()]);

    std::size_t offsetCMX = 0;

    const auto numClustersAttr = getIntAttr(ctx, numClusters);
    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 0);

    // Weights tensor
    auto weightsParentBuffer = createParentBuffer(ctx, functionBuilder, distrBufferParams.weightsDistrAttr, weightsType,
                                                  weightsShape, DimsOrder::OYXI, taskClusters, offsetCMX);
    auto weightsCMXBufferVec = handleWeights(functionBuilder, weightsParentBuffer, weightsAttribute, taskClusters,
                                             offsetCMX, updateBarrier);

    const auto outAlignment = VPU::NCEInvariant::getAlignment(outputType) *
                              static_cast<vpux::Bit>(getElemTypeSize(outputType)).count() / CHAR_BIT;
    VPUX_THROW_UNLESS(offsetCMX % outAlignment == 0, "CMX offset for output tensor must be multiple of {0}, got {1}",
                      outAlignment, offsetCMX);
    const auto OUTPUT_CMX_OFFSET = offsetCMX;

    // Parent output distributed tensor CMX
    auto outParentDistributedCMX =
            createParentBuffer(ctx, functionBuilder, distrBufferParams.parentOutDistrAttr, outputType,
                               parentOutputShape, outputLayout, taskClusters, offsetCMX);
    auto outParentDistrCMXType = outParentDistributedCMX.getType().cast<VPUIP::DistributedBufferType>();

    SmallVector<VPURT::DeclareBufferOp> outCMXBufferVec;
    outCMXBufferVec.reserve(numClusters);
    const auto outPerClusterShapes = outParentDistrCMXType.getPerClusterComputeShapes();
    const auto outPerClusterOffsets = outParentDistrCMXType.getPerClusterComputeShapeOffsets();
    const auto outputStrides = outParentDistrCMXType.getStrides();
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        const auto shape = llvm::to_vector(outPerClusterShapes[idx]);
        if (multiClusterParams.broadcast) {
            // In SOH & broadcast mode (HKSwitch split case), individual CMX buffers in each tile must have
            // ditributed mode SEGMENTED | MULTICASTED
            // In SOK & broadcast mode, individual CMX buffers should be marked as DUPLICATED to be properly
            // broadcasted in other tiles
            const auto outDistrAttr =
                    (multiClusterParams.segmentation == nb::SegmentationType::SOH)
                            ? distrBufferParams.parentOutDistrAttr
                            : VPU::DistributedTensorAttr::get(
                                      ctx, VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED),
                                      nullptr, nullptr, nullptr, nullptr, numClustersAttr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr, nullptr);
            auto outCMXBufferType =
                    VPUIP::DistributedBufferType::get(ctx, shape, outputType, outParentDistrCMXType.getLayout(),
                                                      outParentDistrCMXType.getMemSpace(), outDistrAttr);

            // In SOK & broadcast mode, output offset in each tile should be the same; runtime computes the address for
            // each slice based on workloads In SOH & broadcast mode, output offset in each tile should be computed so
            // that each slice is put in its proper place in the full output This formula satisfies both these
            // constraints as follows:
            //    - for SOK, outPerClusterOffsets will be [0, offset_k_tile_i, 0, 0], therefore
            //                       outPerClusterOffsets[idx][Dims4D::Act::H] = 0 => outputSliceOffset = offsetCMX
            //    - for SOH,  outPerClusterOffsets will be [0, 0, offset_h_tile_i, 0], therefore
            //                       outputSliceOffset = Byte(offsetCMX) + offset_h_tile_i * stride_per_H
            const Byte outputSliceOffset = Byte(offsetCMX) + outPerClusterOffsets[idx][Dims4D::Act::H] *
                                                                     static_cast<Byte>(outputStrides[Dims4D::Act::H]);

            outCMXBufferVec.push_back(createDeclareTensorOp(functionBuilder, outCMXBufferType,
                                                            VPURT::BufferSection::CMX_NN, taskClusters,
                                                            outputSliceOffset.count()));
            continue;
        }

        auto outCMXMemRefType =
                getMemRefType(VPURT::BufferSection::CMX_NN, taskClusters[idx], shape, outputType, outputLayout);
        outCMXBufferVec.push_back(createDeclareTensorOp(functionBuilder, outCMXMemRefType, VPURT::BufferSection::CMX_NN,
                                                        taskClusters[idx], offsetCMX));
    }

    offsetCMX += outParentDistrCMXType.getTotalAllocSize().count();
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inputType) *
                             static_cast<vpux::Bit>(getElemTypeSize(inputType)).count() / CHAR_BIT;
    VPUX_THROW_UNLESS(offsetCMX % inAlignment == 0, "CMX offset for input tensor must be multiple of {0}, got {1}",
                      inAlignment, offsetCMX);

    // Parent input distributed tensor
    auto inParentDistributedCMX =
            createParentBuffer(ctx, functionBuilder, distrBufferParams.parentInDistrAttr, inputType, parentInputShape,
                               DimsOrder::NHWC, taskClusters, offsetCMX);
    auto inParentDistrCMXType = inParentDistributedCMX.getType().cast<VPUIP::DistributedBufferType>();

    SmallVector<vpux::VPURT::DeclareBufferOp> inCMXBufferVec;
    inCMXBufferVec.reserve(numClusters);
    const auto inPerClusterShapes = inParentDistrCMXType.getPerClusterMemoryShapes();
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        inCMXBufferVec.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN,
                                                       inPerClusterShapes[idx], inParentDistrCMXType.getElementType(),
                                                       inParentDistrCMXType.getDimsOrder(), taskClusters[idx],
                                                       offsetCMX));
    }

    if (multiClusterParams.segmentation == nb::SegmentationType::SOK) {
        // Input is SOK mode is duplicated in each tile, so we can use the distributed parent tensor as CMX dst for
        // NNDMAOp
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                              mlir::ValueRange(updateBarrier.getBarrier()), loc, functionInput,
                                              inParentDistributedCMX, 0);
    } else {
        // Input in SOH mode is split along H axis, therefore we must DMA the each slice to the CMX of the corresponding
        // tile
        const auto inPerClusterOffsets = inParentDistrCMXType.getPerClusterMemoryShapeOffsets();
        const auto inputStrides = inParentDistrCMXType.getStrides();
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            // The entire DDR input resides in one buffer in NetworkInput Section
            // To be able to DMA slices of it, we must create sub-buffers in NetworkInput section, index 0, each
            // sub-buffer having an offset that points to the beginning of the slice.
            // The DDR sub-buffer offsets are computed based on the offsets of their CMX counterparts which are obtained
            // from the input parent distributed buffer.
            const auto inShape = inPerClusterShapes[idx];
            const Byte inSliceOffset =
                    inPerClusterOffsets[idx][Dims4D::Act::H] * static_cast<Byte>(inputStrides[Dims4D::Act::H]);
            auto networkInputBuffer =
                    createDeclareTensorOp(functionBuilder, VPURT::BufferSection::NetworkInput, inPerClusterShapes[idx],
                                          inParentDistrCMXType.getElementType(), inParentDistrCMXType.getDimsOrder(),
                                          /*sectionIdx=*/0, inSliceOffset.count());

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, networkInputBuffer,
                                                  inCMXBufferVec[idx], 0);
        }
    }

    offsetCMX += inParentDistrCMXType.getTotalAllocSize().count();
    const auto wtAlignment =
            VPU::NCEInvariant::getAlignment(int32) * static_cast<vpux::Bit>(getElemTypeSize(int32)).count() / CHAR_BIT;
    VPUX_THROW_UNLESS(offsetCMX % wtAlignment == 0, "CMX offset for weights table must be multiple of {0}, got {1}",
                      wtAlignment, offsetCMX);

    // Weights table
    auto weightsDistrTensorType = weightsParentBuffer.getType().dyn_cast<VPUIP::DistributedBufferType>();

    auto wtableCMXBufferVec =
            handleWeightsTable(ctx, functionBuilder, testDesc.getArchitecture(), weightsDistrTensorType, inputType,
                               outputType, int32, weightsTableShape, taskClusters, offsetCMX, updateBarrier);

    auto waitBarrier = updateBarrier;

    // Create NCEClusterTaskOp
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 1);

    auto outStart = SmallVector<std::int64_t>{0, 0, 0};
    // In SOH mode, when kernel size for H dimension is higher than 1, DPU task will need to read lines from the
    // neighbouring cluster. To signal that to runtime, is_segmented field must be set.
    const auto isSegmented =
            (distrBufferParams.parentInDistrAttr.getMode().getValue() == VPU::DistributionMode::SEGMENTED)
                    ? mlir::UnitAttr::get(ctx)
                    : nullptr;
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        // In SOK mode, runtime will compute slice offsets based on workload sizes. In non broadcast mode,
        // we want the output of each tile to start at OUTPUT_CMX_OFFSET. That is why, when broadcasting is disabled,
        // we must ensure that workload size along channel dimension does not indicate the position of the slice
        // relative to the whole output tensor. Otherwise, runtime will compute output offets as OUTPUT_CMX_OFFSET +
        // k_slice_offset_for_tile_i.
        const auto outputChannels =
                (multiClusterParams.segmentation == nb::SegmentationType::SOK && multiClusterParams.broadcast == false)
                        ? outPerClusterShapes[idx][Dims4D::Act::C]
                        : parentOutputShape[Dims4D::Act::C.ind()];
        const auto fullSize = SmallVector<std::int64_t>{parentOutputShape[Dims4D::Act::W.ind()],
                                                        parentOutputShape[Dims4D::Act::H.ind()], outputChannels};

        auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                mlir::ValueRange(updateBarrier.getBarrier()), loc, inCMXBufferVec[idx].getBuffer(),
                weightsCMXBufferVec[idx].getBuffer(), wtableCMXBufferVec[idx].getBuffer(),
                /*instruction_table_list=*/nullptr, /*spr_lookup_table*/ nullptr,
                /*activation_window=*/nullptr, inParentDistributedCMX.getBuffer(), outParentDistributedCMX.getBuffer(),
                outCMXBufferVec[idx].getBuffer(), vpux::VPUIP::NCETaskType::CONV, kernelSize, kernelStrides,
                kernelPaddings, /*activation_window_channel_length=*/nullptr, /*is_continued=*/nullptr,
                /*cm_sp_pattern=*/nullptr, /*is_segmented=*/isSegmented, /*out_channel_offset=*/nullptr);

        const auto workloadPadding =
                getMulticlusteringPaddings(ctx, idx, numClusters, multiClusterParams.segmentation, kernelPaddings);

        const auto outSizes =
                SmallVector<int64_t>{outPerClusterShapes[idx][Dims4D::Act::W], outPerClusterShapes[idx][Dims4D::Act::H],
                                     outPerClusterShapes[idx][Dims4D::Act::C]};

        const auto outStartAttr = getIntArrayAttr(ctx, outStart);
        const auto outEndAttr = getIntArrayAttr(
                ctx, SmallVector<std::int64_t>{outStart[0] + outSizes[0] - 1, outStart[1] + outSizes[1] - 1,
                                               outStart[2] + outSizes[2] - 1});

        nceTask.addDPUTask(functionBuilder, outStartAttr, outEndAttr, workloadPadding, conv.cube_mode);

        for (std::size_t dim = 0; dim < outStart.size(); dim++) {
            outStart[dim] += (outSizes[dim] < fullSize[dim]) ? outSizes[dim] : 0;
        }
    }

    waitBarrier = updateBarrier;

    // Create CMX2DDR DMAs to move outputs from each cluster to DDR
    auto functionOutputs = SmallVector<mlir::Value>(numClusters);

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, 2, testDesc.getWLMParams().isWLMPartialEnabled);

    for (unsigned int idx = 0; idx < static_cast<unsigned int>(numClusters); idx++) {
        auto functionOutput = function.getArgument(1 + idx);
        functionOutputs[idx] = functionOutput;
        const auto outShape =
                multiClusterParams.broadcast ? parentOutputShape : llvm::to_vector(outPerClusterShapes[idx]);
        const auto outMemRefType =
                getMemRefType(VPURT::BufferSection::CMX_NN, taskClusters[idx], outShape, outputType, outputLayout);
        auto outBuffer = createDeclareTensorOp(functionBuilder, outMemRefType, VPURT::BufferSection::CMX_NN,
                                               taskClusters[idx], OUTPUT_CMX_OFFSET);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), loc, outBuffer->getResult(0),
                                              functionOutput, 0);
    }

    functionBuilder.create<mlir::func::ReturnOp>(loc, functionOutputs);

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = numClusters;
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    SmallVector<mlir::Type> outputTensorTypesVec;
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        const auto outShape =
                multiClusterParams.broadcast ? parentOutputShape : llvm::to_vector(outPerClusterShapes[idx]);
        auto outputTensorType = getTensorType(ShapeRef(outShape), outputType, outputLayout, nullptr);
        outputTensorTypesVec.push_back(outputTensorType);
    }
    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(parentInputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               outputTensorTypesVec);
}

}  // namespace hwtest
}  // namespace vpux
