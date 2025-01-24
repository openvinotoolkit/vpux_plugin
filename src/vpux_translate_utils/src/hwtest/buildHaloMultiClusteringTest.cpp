//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
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
int64_t getOutputSpatialDim(const int64_t inputDim, const int64_t kernelSz, const int64_t padBefore,
                            const int64_t padAfter, const int64_t stride) {
    return (inputDim - kernelSz + padBefore + padAfter) / stride + 1;
};

const auto HW_DPU_PROFILING_SIZE_BYTES = 32;

struct Buffers {
    VPURT::DeclareBufferOp input;
    VPURT::DeclareBufferOp output;
    VPURT::DeclareBufferOp weights;
    VPURT::DeclareBufferOp weightsTable;
    SmallVector<VPURT::DeclareBufferOp> outputIti;
    VPURT::DeclareBufferOp profilingOutputCMX;
    VPURT::DeclareBufferOp profilingOutputDDR;
};

VPURT::DeclareBufferOp createBuffer(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Type tensorType,
                                    ArrayRef<int64_t> tensorShape, const DimsOrder dimsOrder,
                                    ArrayRef<int64_t> clusters, const std::size_t offset) {
    if (clusters.size() != 1) {
        auto cmxMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, tensorShape, tensorType, dimsOrder);

        const auto tensorTypeIf = cmxMemRefType.cast<vpux::NDTypeInterface>();

        const auto orderAttr = mlir::AffineMapAttr::get(tensorTypeIf.getDimsOrder().toAffineMap(ctx));
        const auto elemStrides = to_small_vector(tensorTypeIf.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / tensorTypeIf.getElemTypeSize().count();
                                                 }));
        const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
        const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                                  /*allocSize=*/nullptr, ctx);

        const auto dimsSpace = IndexedSymbolAttr::get(ctx, stringifyMemoryKind(tensorTypeIf.getMemoryKind()));

        const auto duplicatedDistrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);

        const auto numClustersAttr = getIntAttr(ctx, clusters.size());

        auto distrTensorAttr = VPU::DistributionInfoAttr::get(ctx, duplicatedDistrModeAttr, nullptr, nullptr, nullptr,
                                                              nullptr, numClustersAttr, nullptr, nullptr, nullptr,
                                                              nullptr, nullptr, nullptr, nullptr);

        auto distributedCMXType = VPUIP::DistributedBufferType::get(ctx, tensorShape, tensorTypeIf.getElementType(),
                                                                    layout, dimsSpace, distrTensorAttr);

        return createDeclareTensorOp(builder, distributedCMXType, VPURT::BufferSection::CMX_NN, clusters, offset);
    }

    auto cmxMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, clusters[0], tensorShape, tensorType, dimsOrder);

    return createDeclareTensorOp(builder, cmxMemRefType, VPURT::BufferSection::CMX_NN, clusters[0], offset);
}

VPURT::DeclareBufferOp createITIBuffer(mlir::MLIRContext* ctx, mlir::OpBuilder& builder, mlir::Type tensorType,
                                       ArrayRef<int64_t> tensorShape, const DimsOrder dimsOrder, const int64_t cluster,
                                       ArrayRef<VPUIP::HaloRegionAttr> inwardHalos,
                                       ArrayRef<VPUIP::OutwardHaloRegionAttr> outwardHalos, const std::size_t offset) {
    const auto cmxMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, tensorShape, tensorType, dimsOrder);
    const auto tensorTypeIf = cmxMemRefType.cast<vpux::NDTypeInterface>();

    const auto orderAttr = mlir::AffineMapAttr::get(tensorTypeIf.getDimsOrder().toAffineMap(ctx));
    const auto elemStrides = to_small_vector(tensorTypeIf.getStrides() | transformed([&](Bit stride) {
                                                 return stride.count() / tensorTypeIf.getElemTypeSize().count();
                                             }));
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                              /*allocSize=*/nullptr, ctx);

    const auto dimsSpace =
            vpux::IndexedSymbolAttr::get(ctx, stringifyMemoryKind(tensorTypeIf.getMemoryKind()), cluster);

    auto itiBuffType = VPUIP::ITIBufferType::get(ctx, tensorShape, tensorTypeIf.getElementType(), layout, dimsSpace,
                                                 nullptr, inwardHalos, outwardHalos);

    return createDeclareTensorOp(builder, itiBuffType, VPURT::BufferSection::CMX_NN, cluster, offset);
}

class Strategy {
public:
    Strategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters)
            : builder_(builder), clusters_(clusters), opBuffers_(clusters.size()) {
    }

    virtual ~Strategy(){};

    SmallVector<Buffers>& getBuffers() {
        return opBuffers_;
    }

    virtual bool isHaloDim(const llvm::ArrayRef<int64_t> fullShape, llvm::SmallVector<int64_t> haloShape, size_t dim) {
        return haloShape[dim] != fullShape[dim];
    }

    void handleConstants(Const::ContentAttr&& weightsContent, VPU::ArchKind arch, mlir::Type inputType,
                         mlir::Type outputType, std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier,
                         mlir::ValueRange waitBarrier) {
        const auto alignment = Byte(16);

        // Create Weights Const.DeclareOp, segment it if necessary, create CMX buffers and DMAs bringing data from DDR
        handleWeights(std::move(weightsContent), offset, updateBarrier, waitBarrier);
        offset += opBuffers_.front().weights.getType().cast<NDTypeInterface>().getTotalAllocSize().count();
        offset = vpux::alignValUp(offset, static_cast<std::size_t>(alignment.count()));

        // Create WeightsTable Const.DeclareOp, segment it if necessary, create CMX buffers and DMAs bringing data from
        // DDR
        handleWeightsTable(arch, inputType, outputType, offset, updateBarrier);
        offset += opBuffers_.front().weightsTable.getType().cast<NDTypeInterface>().getTotalAllocSize().count();
        offset = vpux::alignValUp(offset, static_cast<std::size_t>(alignment.count()));
    }

    void createProfilingOutputBuffers(ArrayRef<int64_t> profilingOutputShape, mlir::Type profOutputType,
                                      std::size_t& cmxOffset) {
        const auto numClusters = clusters_.size();
        cmxOffset = vpux::alignValUp(cmxOffset, static_cast<std::size_t>(Byte(32).count()));

        size_t profilingOffset = 0;
        for (size_t cluster = 0; cluster < numClusters; cluster++) {
            auto profoutputcmx_type = getMemRefType(VPURT::BufferSection::CMX_NN, cluster, profilingOutputShape,
                                                    profOutputType, DimsOrder::C);
            auto profoutputcmx = createDeclareTensorOp(builder_, profoutputcmx_type, VPURT::BufferSection::CMX_NN,
                                                       cluster, cmxOffset);
            opBuffers_[cluster].profilingOutputCMX = profoutputcmx;

            auto profoutputddr =
                    createDeclareTensorOp(builder_,
                                          getMemRefType(VPURT::BufferSection::ProfilingOutput, profilingOutputShape,
                                                        profOutputType, DimsOrder::C),
                                          VPURT::BufferSection::ProfilingOutput, 0, profilingOffset);
            profilingOffset += profoutputddr.getType().cast<NDTypeInterface>().getTotalAllocSize().count();
            opBuffers_[cluster].profilingOutputDDR = profoutputddr;
        }
    }

    virtual void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape,
                                    ArrayRef<int64_t> weightsShape, mlir::ArrayAttr strides, VPU::PaddingAttr padding,
                                    std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier,
                                    const llvm::SmallVector<int64_t> /*clustersPerDim*/) = 0;
    virtual void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) = 0;

protected:
    void dmaDuplicatedWeightsBuffers(Const::ContentAttr&& weightsContent, const std::size_t& offset,
                                     VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();
        auto loc = builder_.getUnknownLoc();

        auto weightsTypeIf = weightsContent.getType();
        const auto weightsShape = weightsTypeIf.getShape();
        const auto weightsElementType = weightsTypeIf.getElementType();

        // Create CMX buffer for weights with multiple section indexes
        auto weightsBuffer =
                createBuffer(ctx, builder_, weightsElementType, weightsShape.raw(), DimsOrder::OYXI, clusters_, offset);

        // Create DDR buffer for weights
        const auto weightsDDRType = getMemRefType(VPURT::BufferSection::Constant, weightsShape.raw(),
                                                  weightsElementType, weightsTypeIf.getDimsOrder());
        auto weightsDDRBuffer = builder_.create<vpux::Const::DeclareOp>(loc, weightsDDRType, std::move(weightsContent));

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, waitBarrier, mlir::ValueRange(updateBarrier.getBarrier()), loc,
                                              weightsDDRBuffer, weightsBuffer, 0);

        // Each cluster will get the same broadcasted CMX buffer
        for (size_t cluster = 0; cluster < numClusters; cluster++) {
            opBuffers_[cluster].weights = weightsBuffer;
        }
    }

    void dmaDuplicatedWeightsTableBuffers(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType,
                                          const std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier) {
        const auto numClusters = clusters_.size();
        auto ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();
        auto int32 = builder_.getIntegerType(32, true);

        const auto sparsityPtrStep = 0;
        auto weightsBuffType = opBuffers_.front().weights.getBuffer().getType().cast<NDTypeInterface>();
        auto weightsOutputChannelsStrideInBits = weightsBuffType.getStrides()[vpux::Dims4D::Filter::OC];

        const auto alignmentRequirement = 16;
        const auto alignment =
                (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
        if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
            weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
        }

        auto weightsElemType = weightsBuffType.getElementType();

        const SmallVector<int64_t> wtableShape = {weightsBuffType.getShape()[vpux::Dims4D::Filter::OC], 1, 1, 4};

        // Create distributed, duplicated buffer in CMX; will be used in DMA task to move same content to each
        // tile
        auto wtableCMXBuffer = createBuffer(ctx, builder_, int32, wtableShape, DimsOrder::NHWC, clusters_, offset);

        const auto wtableDDRType = getMemRefType(VPURT::BufferSection::Constant, wtableShape, int32, DimsOrder::NHWC);

        // Create weights table content
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
        const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, 0,
                static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.to<Byte>().count()),
                VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
                wtableShape[vpux::Dims4D::Filter::OC.ind()], weightsElemType);
        auto wtableTensorType = mlir::RankedTensorType::get(wtableShape, int32);
        const auto weightsTableValues =
                mlir::DenseElementsAttr::get(wtableTensorType, llvm::ArrayRef<std::int32_t>(weightsTable));

        auto weightsDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                loc, wtableDDRType,
                vpux::Const::ContentAttr::get(weightsTableValues,
                                              Const::ContentSetup(wtableTensorType).reorder(vpux::DimsOrder::NHWC)));

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                              mlir::ValueRange(updateBarrier.getBarrier()), loc, weightsDDRBuffer,
                                              wtableCMXBuffer, 0);

        // Buffers to use by NCEClusterTask
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            const auto wtableCMXMemRefType =
                    getMemRefType(VPURT::BufferSection::CMX_NN, clusters_[idx], wtableShape, int32, DimsOrder::NHWC);
            opBuffers_[idx].weightsTable = createDeclareTensorOp(builder_, wtableCMXMemRefType,
                                                                 VPURT::BufferSection::CMX_NN, clusters_[idx], offset);
        }
    }

    // Input in SOH/SOW mode is split along H/W axis, therefore we must DMA the each slice to the CMX of the
    // corresponding tile
    void sliceInputAlongOneAxis(const Dim& axis, mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape,
                                ArrayRef<int64_t> weightsShape, mlir::ArrayAttr strides, VPU::PaddingAttr padding,
                                std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier) {
        auto ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();
        const auto origInputTypeIf = ddrInput.getType().cast<NDTypeInterface>();
        const auto origInputShape = origInputTypeIf.getShape();
        auto outputPerClusterShape = Shape(fullOutputShape);

        const auto outStep = divUp(outputPerClusterShape[axis], static_cast<int64_t>(clusters_.size()));
        Shape outputOffsets{0, 0, 0, 0};
        Shape divAxis{1, 1, 1, 1};
        divAxis[axis] = clusters_.size();

        const auto inputStrides = origInputTypeIf.getStrides();
        const auto paddingInfo = PadInfo(padding.getLeft().getInt(), padding.getRight().getInt(),
                                         padding.getTop().getInt(), padding.getBottom().getInt());

        int64_t largestSliceSize = 0;
        for (std::size_t idx = 0; idx < clusters_.size(); idx++) {
            // The entire DDR input resides in one buffer in NetworkInput Section
            // To be able to DMA slices of it, we must create sub-buffers in NetworkInput section, index 0, each
            // sub-buffer having an offset that points to the beginning of the slice.

            // Each cluster will compute an equal slice of output, except the last cluster which may have less
            VPUX_THROW_UNLESS(std::size_t(axis.ind()) < fullOutputShape.size(),
                              "buildHaloMultiClusteringTest: Strategy's axis goes beyond the fullOutputShape");
            outputPerClusterShape[axis] =
                    (idx != clusters_.size() - 1) ? outStep : fullOutputShape[axis.ind()] - idx * outStep;
            outputOffsets[axis] = idx * outStep;

            // Use the shape of the output tile to back-infer the necessary slice of input to compute it
            // That is the slice of input that needs to be DMA'd to CMX, since inter-tile reads are not possible
            const TileInfo outputTile(outputPerClusterShape, outputOffsets, divAxis);
            const auto tilingSolution = vpux::backInferConvTile(outputTile, origInputShape, Shape(weightsShape),
                                                                Shape(), strides, paddingInfo);
            const auto inputTile = tilingSolution.tiles.front();

            const Byte inSliceOffset =
                    inputTile.offsets[Dims4D::Act::H] * static_cast<Byte>(inputStrides[Dims4D::Act::H]) +
                    inputTile.offsets[Dims4D::Act::W] * static_cast<Byte>(inputStrides[Dims4D::Act::W]);
            auto networkInputBuffer = createDeclareTensorOp(builder_, VPURT::BufferSection::NetworkInput,
                                                            inputTile.shape.raw(), origInputTypeIf.getElementType(),
                                                            origInputTypeIf.getDimsOrder(), inputStrides,
                                                            /*locale=*/0, inSliceOffset.count());

            opBuffers_[idx].input = createBuffer(ctx, builder_, origInputTypeIf.getElementType(), inputTile.shape.raw(),
                                                 origInputTypeIf.getDimsOrder(), {clusters_[idx]}, offset);

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, networkInputBuffer,
                                                  opBuffers_[idx].input, 0);

            const auto inputSliceSize =
                    opBuffers_[idx].input.getBuffer().getType().cast<NDTypeInterface>().getCompactAllocSize().count();
            if (largestSliceSize < inputSliceSize) {
                largestSliceSize = inputSliceSize;
            }
        }

        offset += largestSliceSize;
    }

    virtual void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                               VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) = 0;
    virtual void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType,
                                    std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier) = 0;

    mlir::OpBuilder& builder_;
    ArrayRef<int64_t> clusters_;
    SmallVector<Buffers> opBuffers_;
};

class SoHorSoWStrategy final : public Strategy {
public:
    SoHorSoWStrategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters, const Dim& axis, const int64_t haloSz)
            : Strategy(builder, clusters), axis_(axis), haloSz_(haloSz) {
    }

    void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(numClusters);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(numClusters);

        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes.front().cast<NDTypeInterface>().getShape());
        VPUX_THROW_UNLESS(std::size_t(axis_.ind()) < haloShape.size(),
                          "buildHaloMultiClusteringTest: SoHorSoWStrategy's axis goes beyond the halo's shape");
        haloShape[axis_.ind()] = haloSz_;
        const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);

        // Create outward halos for all clusters and add them to the neighbouring clusters' inward halos
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();
            const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[idx]);

            // offset in producer cluster
            SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
            VPUX_THROW_UNLESS(std::size_t(axis_.ind()) < perDimOffset.size(),
                              "buildHaloMultiClusteringTest: SoHorSoWStrategy's axis goes beyond the perDimOffset's "
                              "array length");

            // All the clusters except the first one will produce a halo from the top/left of the workload
            if (idx != 0) {
                perDimOffset[axis_.ind()] = haloSz_;
                const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

                const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[idx - 1]);
                // offset in the halo's target cluster
                SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
                VPUX_THROW_UNLESS(std::size_t(axis_.ind()) < neighbourOffset.size(),
                                  "buildHaloMultiClusteringTest: SoHorSoWStrategy's axis goes beyond the "
                                  "neighbourOffset's array length");
                neighbourOffset[axis_.ind()] = outputTypes[idx - 1].cast<NDTypeInterface>().getShape()[axis_] - haloSz_;
                const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);
                auto neighbourInwardHalo =
                        VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

                const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
                auto outwardHalo =
                        VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

                inwardHalosPerCluster[idx - 1].push_back(neighbourInwardHalo);
                outwardHalosPerCluster[idx].push_back(outwardHalo);
            }

            // All the clusters except the last one will produce a halo from the bottom/right of the workload
            if (idx != numClusters - 1) {
                perDimOffset[axis_.ind()] = ddrOutputType.getShape()[axis_] - 2 * haloSz_;

                const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

                const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[idx + 1]);
                const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, SmallVector<int64_t>{0, 0, 0, 0});
                auto neighbourInwardHalo =
                        VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

                const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
                auto outwardHalo =
                        VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

                inwardHalosPerCluster[idx + 1].push_back(neighbourInwardHalo);
                outwardHalosPerCluster[idx].push_back(outwardHalo);
            }
        }

        int64_t largestSliceSize = 0;
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();

            opBuffers_[idx].output =
                    createITIBuffer(ctx, builder_, ddrOutputType.getElementType(), ddrOutputType.getShape().raw(),
                                    ddrOutputType.getDimsOrder(), clusters_[idx], inwardHalosPerCluster[idx],
                                    outwardHalosPerCluster[idx], offset);

            if (idx != 0) {
                opBuffers_[idx - 1].outputIti.push_back(opBuffers_[idx].output);
            }

            if (idx != numClusters - 1) {
                opBuffers_[idx + 1].outputIti.push_back(opBuffers_[idx].output);
            }

            const auto outputSliceSize =
                    opBuffers_[idx].output.getBuffer().getType().cast<NDTypeInterface>().getCompactAllocSize().count();
            if (largestSliceSize < outputSliceSize) {
                largestSliceSize = outputSliceSize;
            }
        }

        offset += largestSliceSize;
    }

    void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape, ArrayRef<int64_t> weightsShape,
                            mlir::ArrayAttr strides, VPU::PaddingAttr padding, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier,
                            const llvm::SmallVector<int64_t> /*clustersPerDim*/) override {
        sliceInputAlongOneAxis(axis_, ddrInput, fullOutputShape, weightsShape, strides, padding, offset, updateBarrier);
    }

private:
    void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                       VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) override {
        dmaDuplicatedWeightsBuffers(std::move(weightsContent), offset, updateBarrier, waitBarrier);
    }

    void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier) override {
        dmaDuplicatedWeightsTableBuffers(arch, inputType, outputType, offset, updateBarrier);
    }

    Dim axis_;
    int64_t haloSz_;
};

class SoKStrategy final : public Strategy {
public:
    SoKStrategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters): Strategy(builder, clusters) {
    }

    void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();
        auto fullOutputChannelsNum = outputTypes.front().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(numClusters);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(numClusters);

        // All clusters have an equal chunck of channels, except the last one which might have less
        const auto channelSlice = divUp(fullOutputChannelsNum, static_cast<int64_t>(numClusters));
        SmallVector<int64_t> outChannelsPerCluster(numClusters, channelSlice);
        outChannelsPerCluster.back() = fullOutputChannelsNum - (numClusters - 1) * channelSlice;

        // Create outward halos for all clusters and add them to all other clusters' inward halos
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[idx]);
            const auto crtDdrInput = outputTypes[idx].cast<NDTypeInterface>();

            SmallVector<int64_t> haloShape = llvm::to_vector(crtDdrInput.getShape());
            haloShape[Dims4D::Act::C.ind()] = outChannelsPerCluster[idx];
            const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);

            // offset in producer cluster & in halo's target clusters
            // In SOK mode, the entire tensor is a halo for all tensors is other clusters, therefore
            // the channels offset is the offset of the current chunck in the full output.
            // To get it, we add the channel size for all the outputs from the clusters before this one.
            const auto outChannelOffset =
                    std::accumulate(outChannelsPerCluster.begin(), outChannelsPerCluster.begin() + idx,
                                    static_cast<int64_t>(0), [](const int64_t chOffset, const int64_t chSize) {
                                        return chOffset + chSize;
                                    });

            const SmallVector<int64_t> dimOffests = {0, outChannelOffset, 0, 0};
            const auto offsetAttr = getIntArrayAttr(builder_, dimOffests);

            auto inwardHalosVec = SmallVector<mlir::Attribute>();

            for (std::size_t targetIdx = 0; targetIdx < numClusters; targetIdx++) {
                if (targetIdx == idx) {
                    continue;
                }

                const auto targetCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
                auto neighbourInwardHalo = VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, targetCluster);

                inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
                inwardHalosVec.push_back(neighbourInwardHalo);
            }

            const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
            auto outwardHalo =
                    VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

            outwardHalosPerCluster[idx].push_back(outwardHalo);
        }

        for (std::size_t idx = 0; idx < numClusters; idx++) {
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();

            opBuffers_[idx].output =
                    createITIBuffer(ctx, builder_, ddrOutputType.getElementType(), ddrOutputType.getShape().raw(),
                                    ddrOutputType.getDimsOrder(), clusters_[idx], inwardHalosPerCluster[idx],
                                    outwardHalosPerCluster[idx], offset);

            for (std::size_t targetIdx = 0; targetIdx < numClusters; targetIdx++) {
                if (targetIdx == idx) {
                    continue;
                }

                opBuffers_[targetIdx].outputIti.push_back(opBuffers_[idx].output);
            }
        }

        offset += opBuffers_.front().output.getBuffer().getType().cast<NDTypeInterface>().getTotalAllocSize().count();
    }

    void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> /*fullOutputShape*/,
                            ArrayRef<int64_t> /*weightsShape*/, mlir::ArrayAttr /*strides*/,
                            VPU::PaddingAttr /*padding*/, std::size_t& offset, VPURT::ConfigureBarrierOp updateBarrier,
                            const llvm::SmallVector<int64_t> /*clustersPerDim*/) override {
        auto loc = builder_.getUnknownLoc();
        auto ctx = builder_.getContext();
        const auto origInputTypeIf = ddrInput.getType().cast<NDTypeInterface>();

        // The entire DDR input resides in one buffer in NetworkInput Section
        auto networkInputBuffer = createDeclareTensorOp(
                builder_, VPURT::BufferSection::NetworkInput, origInputTypeIf.getShape().raw(),
                origInputTypeIf.getElementType(), origInputTypeIf.getDimsOrder(), origInputTypeIf.getStrides(),
                /*locale=*/0, /*offset=*/0);

        auto cmxInputBuffer =
                createBuffer(ctx, builder_, origInputTypeIf.getElementType(), origInputTypeIf.getShape().raw(),
                             origInputTypeIf.getDimsOrder(), clusters_, offset);

        // Same input should be broadcasted to all clusters
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                              mlir::ValueRange(updateBarrier.getBarrier()), loc, networkInputBuffer,
                                              cmxInputBuffer, 0);

        // Create separate buffers in CMX for NCEClusterTask inputs
        for (std::size_t idx = 0; idx < clusters_.size(); idx++) {
            opBuffers_[idx].input =
                    createBuffer(ctx, builder_, origInputTypeIf.getElementType(), origInputTypeIf.getShape().raw(),
                                 origInputTypeIf.getDimsOrder(), {clusters_[idx]}, offset);
        }

        offset += cmxInputBuffer.getBuffer().getType().cast<NDTypeInterface>().getCompactAllocSize().count();
    }

private:
    // Weights are split over Output Channels (K) dim, with a slice in each tile
    void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                       VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();
        auto loc = builder_.getUnknownLoc();

        auto weightsTypeIf = weightsContent.getType();
        const auto weightsShape = weightsTypeIf.getShape();
        const auto weightsElementType = weightsTypeIf.getElementType();

        // Divide the full OC by the number of clusters; round up if K is not a multiple of numClusters
        Shape weightsOffset{0, 0, 0, 0};
        const auto fullK = weightsShape[vpux::Dims4D::Filter::OC];
        const auto kStep = divUp(fullK, static_cast<int64_t>(numClusters));
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            const Shape weightsOffset{static_cast<int64_t>(idx * kStep), 0, 0, 0};

            // Ensure the last cluster gets the reminder of output channels
            const auto outputChannels = (idx != numClusters - 1) ? kStep : fullK - (numClusters - 1) * kStep;
            const auto perClusterShape =
                    Shape{static_cast<int64_t>(outputChannels), weightsShape[vpux::Dims4D::Filter::IC],
                          weightsShape[vpux::Dims4D::Filter::KY], weightsShape[vpux::Dims4D::Filter::KX]};

            // Create CMX buffer for weights
            auto weightsBuffer = createBuffer(ctx, builder_, weightsElementType, perClusterShape.raw(), DimsOrder::OYXI,
                                              clusters_[idx], offset);

            // Create a DDR buffer for each slice of the weights
            const auto weightsDDRMemRefType = getMemRefType(VPURT::BufferSection::Constant, perClusterShape.raw(),
                                                            weightsElementType, weightsTypeIf.getDimsOrder());

            // Create weights slice by using subview on the full weights content
            auto weightsDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                    loc, weightsDDRMemRefType,
                    weightsContent.transform().subview(weightsOffset, perClusterShape).get());

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, waitBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                                  loc, weightsDDRBuffer, weightsBuffer, 0);

            opBuffers_[idx].weights = weightsBuffer;
        }
    }

    void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier) override {
        const auto numClusters = clusters_.size();
        auto loc = builder_.getUnknownLoc();
        auto ctx = builder_.getContext();
        auto int32 = builder_.getIntegerType(32, true);

        const auto sparsityPtrStep = 0;
        auto weightsBuffType = opBuffers_.front().weights.getBuffer().getType().cast<NDTypeInterface>();
        auto weightsOutputChannelsStrideInBits = weightsBuffType.getStrides()[vpux::Dims4D::Filter::OC];
        auto weightsElemType = weightsBuffType.getElementType();

        const auto alignmentRequirement = 16;
        const auto alignment =
                (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
        if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
            weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
        }

        for (std::size_t idx = 0; idx < numClusters; idx++) {
            // Create weights table DDR buffer for each tile
            const auto outChannels = opBuffers_[idx]
                                             .weights.getBuffer()
                                             .getType()
                                             .cast<NDTypeInterface>()
                                             .getShape()[Dims4D::Filter::OC];
            const auto wtableShape = SmallVector<int64_t>({outChannels, 1, 1, 4});
            const auto wtableDDRType =
                    getMemRefType(VPURT::BufferSection::Constant, wtableShape, int32, DimsOrder::NHWC);

            // Create weights table content for each weights table chunck
            const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
            const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
            const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                    inputType, outputType, 0,
                    static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                    VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
                    outChannels, weightsElemType);
            auto wtableTensorType = mlir::RankedTensorType::get(wtableShape, int32);
            const auto weightsTableValues =
                    mlir::DenseElementsAttr::get(wtableTensorType, llvm::ArrayRef<std::int32_t>(weightsTable));

            auto wtableDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                    loc, wtableDDRType,
                    vpux::Const::ContentAttr::get(
                            weightsTableValues, Const::ContentSetup(wtableTensorType).reorder(vpux::DimsOrder::NHWC)));

            auto wtableCMXBuffer =
                    createBuffer(ctx, builder_, int32, wtableShape, DimsOrder::NHWC, clusters_[idx], offset);

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, wtableDDRBuffer,
                                                  wtableCMXBuffer, 0);

            const auto wtableCMXMemRefType =
                    getMemRefType(VPURT::BufferSection::CMX_NN, clusters_[idx], wtableShape, int32, DimsOrder::NHWC);
            opBuffers_[idx].weightsTable = createDeclareTensorOp(builder_, wtableCMXMemRefType,
                                                                 VPURT::BufferSection::CMX_NN, clusters_[idx], offset);
        }
    }
};

class SoHW3Strategy final : public Strategy {
public:
    SoHW3Strategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters, int64_t splitNum, const int64_t heightHaloSz,
                  const int64_t widthHaloSz)
            : Strategy(builder, clusters), splitNum_(splitNum), heightHaloSz_(heightHaloSz), widthHaloSz_(widthHaloSz) {
    }

    // This Enum is to track for each cluster where we already inserted a halo for each axis
    enum class HaloPosition : size_t { AFTER, BEFORE };

    bool isHaloDim(const llvm::ArrayRef<int64_t> fullShape, llvm::SmallVector<int64_t> haloShape, size_t dim) override {
        return haloShape[dim] != fullShape[dim] &&
               ((haloShape[dim] == heightHaloSz_ && static_cast<std::int32_t>(dim) == Dims4D::Act::H.ind()) ||
                (haloShape[dim] == widthHaloSz_ && static_cast<std::int32_t>(dim) == Dims4D::Act::W.ind()));
    }

    void insertHaloBefore(SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                          SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                          llvm::ArrayRef<mlir::Type> outputTypes, const size_t currentIdx, const size_t beforeIdx,
                          const Dim& axis, const int64_t haloSz, const int64_t haloCornerSz = 0,
                          const int64_t haloNumPerCluster = 1) {
        SmallVector<int64_t> haloShape = llvm::to_vector(
                outputTypes[axis == Dims4D::Act::W ? beforeIdx : currentIdx].cast<NDTypeInterface>().getShape());

        haloShape[axis.ind()] = haloSz;
        Dim axisSOHW = axis;

        if (axis == Dims4D::Act::H) {
            axisSOHW = Dims4D::Act::W;
            haloShape[axisSOHW.ind()] -= haloCornerSz * haloNumPerCluster;
        } else {
            axisSOHW = Dims4D::Act::H;
        }

        const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);
        const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[currentIdx]);

        // offset in producer cluster
        SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
        auto* ctx = builder_.getContext();
        perDimOffset[axis.ind()] = haloSz;
        if ((axis == Dims4D::Act::W) && (beforeIdx == 1)) {
            perDimOffset[Dims4D::Act::H.ind()] =
                    outputTypes[beforeIdx].cast<NDTypeInterface>().getShape()[Dims4D::Act::H] - 2 * haloCornerSz;
        }

        const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[beforeIdx]);
        // offset in the halo's target cluster
        SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
        neighbourOffset[axis.ind()] = outputTypes[beforeIdx].cast<NDTypeInterface>().getShape()[axis] - haloSz;

        const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

        const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);
        auto neighbourInwardHalo =
                VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

        const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
        auto outwardHalo =
                VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

        inwardHalosPerCluster[beforeIdx].push_back(neighbourInwardHalo);
        outwardHalosPerCluster[currentIdx].push_back(outwardHalo);
    }

    void insertHaloAfter(SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                         SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                         llvm::ArrayRef<mlir::Type> outputTypes, const size_t currentIdx, const size_t afterIdx,
                         const Dim& axis, const int64_t haloSz, const int64_t haloCornerSz = 0,
                         const int64_t haloNumPerCluster = 1, const bool haloBefore = false) {
        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes[currentIdx].cast<NDTypeInterface>().getShape());

        haloShape[axis.ind()] = haloSz;
        const Dim axisSOHW = axis == Dims4D::Act::H ? Dims4D::Act::W : Dims4D::Act::H;

        // In case of SOHW we need to subtract the other axis halo size.
        // Also it is posible to have multiple Halos for the other axis, and we need to take in consideration this.
        haloShape[axisSOHW.ind()] -= haloCornerSz * haloNumPerCluster;

        const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);
        auto ddrOutputType = outputTypes[currentIdx].cast<NDTypeInterface>();
        const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[currentIdx]);

        // offset in producer cluster
        SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
        auto* ctx = builder_.getContext();

        perDimOffset[axis.ind()] = ddrOutputType.getShape()[axis] - 2 * haloSz;

        const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[afterIdx]);
        // offset in the halo's target cluster
        SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
        if (haloBefore) {
            // If we inserted a halo before we must start the offset from that position.
            perDimOffset[axisSOHW.ind()] = haloCornerSz;
            neighbourOffset[axisSOHW.ind()] = ddrOutputType.getShape()[axis] - haloSz;
        }

        const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);
        const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);
        auto neighbourInwardHalo =
                VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

        const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
        auto outwardHalo =
                VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

        inwardHalosPerCluster[afterIdx].push_back(neighbourInwardHalo);
        outwardHalosPerCluster[currentIdx].push_back(outwardHalo);
    }

    void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();
        VPUX_THROW_UNLESS(numClusters == 3, "SOHW3 PSS tests supports only 3 clusters.");

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(numClusters);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(numClusters);

        // Conditions to insert halo BEFORE or AFTER over height or width for each cluster
        auto insertAfterWidth = [&](int64_t idx) -> bool {
            return idx < splitNum_;
        };
        auto insertBeforeWidth = [&](int64_t idx) -> bool {
            return idx == splitNum_;
        };
        auto insertAfterHeight = [&](int64_t idx) -> bool {
            return idx == 0;
        };
        auto insertBeforeHeight = [&](int64_t idx) -> bool {
            return idx == 1;
        };

        // Create outward halos for all clusters and add them to the neighbouring clusters' inward halos
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            // These 2 variables are needed to calculate the Halo height in case of multiple width Halos, also Halo
            // width in case of multiple height Halos
            SmallVector<HaloPosition> heightHaloInserted;
            SmallVector<HaloPosition> widthHaloInserted;
            // haloBeforeFlag it is needed to calculate the offsets for the current cluster
            auto haloBeforeFlagW = false;
            auto haloBeforeFlagH = false;

            if (insertAfterWidth(static_cast<std::int64_t>(idx))) {
                widthHaloInserted.push_back(HaloPosition::AFTER);
            }
            if (insertBeforeWidth(static_cast<std::int64_t>(idx))) {
                widthHaloInserted.push_back(HaloPosition::BEFORE);
                haloBeforeFlagW = true;
            }
            if (insertAfterHeight(static_cast<std::int64_t>(idx))) {
                heightHaloInserted.push_back(HaloPosition::AFTER);
            }
            if (insertBeforeHeight(static_cast<std::int64_t>(idx))) {
                heightHaloInserted.push_back(HaloPosition::BEFORE);
                haloBeforeFlagH = true;
            }

            // Insert width halo after
            if (insertAfterWidth(static_cast<std::int64_t>(idx))) {
                insertHaloAfter(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, splitNum_,
                                Dims4D::Act::W, widthHaloSz_, heightHaloSz_, heightHaloInserted.size(),
                                haloBeforeFlagH);
            }

            // Insert width halo before
            // Tile2(up) - Tile0
            // Tile2(down) - Tile1
            if (insertBeforeWidth(static_cast<std::int64_t>(idx))) {
                for (auto idxBeforeW = 0; idxBeforeW < splitNum_; idxBeforeW++) {
                    insertHaloBefore(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, idxBeforeW,
                                     Dims4D::Act::W, widthHaloSz_, heightHaloSz_, heightHaloInserted.size());
                }
            }

            // Insert height halo after
            if (insertAfterHeight(static_cast<std::int64_t>(idx))) {
                insertHaloAfter(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, idx + 1,
                                Dims4D::Act::H, heightHaloSz_, widthHaloSz_, widthHaloInserted.size(), haloBeforeFlagW);
            }

            // Insert height halo before
            if (insertBeforeHeight(static_cast<std::int64_t>(idx))) {
                insertHaloBefore(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, idx - 1,
                                 Dims4D::Act::H, heightHaloSz_, widthHaloSz_, widthHaloInserted.size());
            }
        }

        int64_t largestSliceSize = 0;
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            SmallVector<HaloPosition> heightHaloInserted;
            SmallVector<HaloPosition> widthHaloInserted;
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();

            opBuffers_[idx].output =
                    createITIBuffer(ctx, builder_, ddrOutputType.getElementType(), ddrOutputType.getShape().raw(),
                                    ddrOutputType.getDimsOrder(), clusters_[idx], inwardHalosPerCluster[idx],
                                    outwardHalosPerCluster[idx], offset);

            if (insertAfterWidth(static_cast<std::int64_t>(idx))) {
                opBuffers_[splitNum_].outputIti.push_back(opBuffers_[idx].output);
            }

            if (insertBeforeWidth(static_cast<std::int64_t>(idx))) {
                // Tile2(up) - Tile0
                // Tile2(down) - Tile1
                for (auto idxBeforeW = 0; idxBeforeW < splitNum_; idxBeforeW++) {
                    opBuffers_[idxBeforeW].outputIti.push_back(opBuffers_[idx].output);
                }
            }

            if (insertAfterHeight(static_cast<std::int64_t>(idx))) {
                opBuffers_[idx + 1].outputIti.push_back(opBuffers_[idx].output);
            }

            if (insertBeforeHeight(static_cast<std::int64_t>(idx))) {
                opBuffers_[idx - 1].outputIti.push_back(opBuffers_[idx].output);
            }

            const auto outputSliceSize =
                    opBuffers_[idx].output.getBuffer().getType().cast<NDTypeInterface>().getCompactAllocSize().count();
            if (largestSliceSize < outputSliceSize) {
                largestSliceSize = outputSliceSize;
            }
        }

        offset += largestSliceSize;
    }

    // Tile0 Tile2(up)
    // Tile1 Tile2(down)
    void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape, ArrayRef<int64_t> weightsShape,
                            mlir::ArrayAttr strides, VPU::PaddingAttr padding, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier,
                            const llvm::SmallVector<int64_t> /*clustersPerDim*/) override {
        auto ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();
        const auto origInputTypeIf = ddrInput.getType().cast<NDTypeInterface>();
        const auto origInputShape = origInputTypeIf.getShape();
        auto outputPerClusterShape = Shape(fullOutputShape);

        const auto outStepHeight = divUp(outputPerClusterShape[Dims4D::Act::H], static_cast<int64_t>(splitNum_));
        const auto outStepWidth = divUp(outputPerClusterShape[Dims4D::Act::W], static_cast<int64_t>(splitNum_));
        Shape outputOffsets{0, 0, 0, 0};
        Shape divAxis{1, 1, splitNum_, splitNum_};

        const auto inputStrides = origInputTypeIf.getStrides();
        const auto paddingInfo = PadInfo(padding.getLeft().getInt(), padding.getRight().getInt(),
                                         padding.getTop().getInt(), padding.getBottom().getInt());

        for (auto clusterIdx = 0; clusterIdx < static_cast<int64_t>(clusters_.size()); clusterIdx++) {
            switch (clusterIdx) {
            case 0:
                outputPerClusterShape[Dims4D::Act::H] = outStepHeight;
                outputPerClusterShape[Dims4D::Act::W] = outStepWidth;
                outputOffsets[Dims4D::Act::W] = 0;
                outputOffsets[Dims4D::Act::H] = 0;
                break;
            case 1:
                outputPerClusterShape[Dims4D::Act::H] = fullOutputShape[Dims4D::Act::H.ind()] - outStepHeight;
                outputPerClusterShape[Dims4D::Act::W] = outStepWidth;
                outputOffsets[Dims4D::Act::W] = 0;
                outputOffsets[Dims4D::Act::H] = outStepHeight;
                break;
            case 2:
                outputPerClusterShape[Dims4D::Act::H] = fullOutputShape[Dims4D::Act::H.ind()];
                outputPerClusterShape[Dims4D::Act::W] = fullOutputShape[Dims4D::Act::W.ind()] - outStepWidth;
                outputOffsets[Dims4D::Act::W] = outStepWidth;
                outputOffsets[Dims4D::Act::H] = 0;
                break;
            }

            // Use the shape of the output tile to back-infer the necessary slice of input to compute it
            // That is the slice of input that needs to be DMA'd to CMX, since inter-tile reads are not possible
            const TileInfo outputTile(outputPerClusterShape, outputOffsets, divAxis);
            const auto tilingSolution = vpux::backInferConvTile(outputTile, origInputShape, Shape(weightsShape),
                                                                Shape(), strides, paddingInfo);
            const auto inputTile = tilingSolution.tiles.front();

            const Byte inSliceOffset =
                    inputTile.offsets[Dims4D::Act::H] * static_cast<Byte>(inputStrides[Dims4D::Act::H]) +
                    inputTile.offsets[Dims4D::Act::W] * static_cast<Byte>(inputStrides[Dims4D::Act::W]);
            auto networkInputBuffer = createDeclareTensorOp(builder_, VPURT::BufferSection::NetworkInput,
                                                            inputTile.shape.raw(), origInputTypeIf.getElementType(),
                                                            origInputTypeIf.getDimsOrder(), inputStrides,
                                                            /*locale=*/0, inSliceOffset.count());

            opBuffers_[clusterIdx].input =
                    createBuffer(ctx, builder_, origInputTypeIf.getElementType(), inputTile.shape.raw(),
                                 origInputTypeIf.getDimsOrder(), {clusters_[clusterIdx]}, offset);

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, networkInputBuffer,
                                                  opBuffers_[clusterIdx].input, 0);
        }

        // Tile2 will have all the height and implicitly largest slice size
        auto lastClusterIdx = clusters_.size() - 1;
        offset += opBuffers_[lastClusterIdx]
                          .input.getBuffer()
                          .getType()
                          .cast<NDTypeInterface>()
                          .getCompactAllocSize()
                          .count();
    }

private:
    void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                       VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) override {
        dmaDuplicatedWeightsBuffers(std::move(weightsContent), offset, updateBarrier, waitBarrier);
    }

    void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier) override {
        dmaDuplicatedWeightsTableBuffers(arch, inputType, outputType, offset, updateBarrier);
    }

    int64_t splitNum_;
    int64_t heightHaloSz_;
    int64_t widthHaloSz_;
};

class SoHWStrategy final : public Strategy {
public:
    SoHWStrategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters, ArrayRef<int64_t> clustersPerDim,
                 const int64_t heightHaloSz, const int64_t widthHaloSz)
            : Strategy(builder, clusters),
              clustersPerDim_(clustersPerDim),
              heightHaloSz_(heightHaloSz),
              widthHaloSz_(widthHaloSz) {
    }

    // This Enum is to track for each cluster where we already inserted a halo for each axis
    enum class HaloPosition : size_t { AFTER, BEFORE };

    bool isHaloDim(const llvm::ArrayRef<int64_t> fullShape, llvm::SmallVector<int64_t> haloShape, size_t dim) override {
        return haloShape[dim] != fullShape[dim] &&
               ((haloShape[dim] == heightHaloSz_ && static_cast<std::int32_t>(dim) == Dims4D::Act::H.ind()) ||
                (haloShape[dim] == widthHaloSz_ && static_cast<std::int32_t>(dim) == Dims4D::Act::W.ind()));
    }

    void insertHaloBefore(SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                          SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                          llvm::ArrayRef<mlir::Type> outputTypes, const size_t currentIdx, const size_t beforeIdx,
                          const Dim& axis, const int64_t haloSz, const int64_t haloCornerSz = 0,
                          const int64_t haloNumPerCluster = 1, const bool haloBefore = false) {
        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes[currentIdx].cast<NDTypeInterface>().getShape());
        haloShape[axis.ind()] = haloSz;
        Dim axisSOHW = axis;

        if (axis == Dims4D::Act::H) {
            axisSOHW = Dims4D::Act::W;
        } else {
            axisSOHW = Dims4D::Act::H;
        }
        // In case of SOHW we need to subtract the other axis halo size.
        // Also it is posible to have multiple Halos for the other axis, and we need to take in consideration this.
        haloShape[axisSOHW.ind()] -= haloCornerSz * haloNumPerCluster;

        const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);
        const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[currentIdx]);

        // offset in producer cluster
        SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
        auto* ctx = builder_.getContext();
        perDimOffset[axis.ind()] = haloSz;

        const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[beforeIdx]);
        // offset in the halo's target cluster
        SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
        neighbourOffset[axis.ind()] = outputTypes[beforeIdx].cast<NDTypeInterface>().getShape()[axis] - haloSz;
        if (haloBefore) {
            // If we inserted a halo before we must start the offset from that position.
            perDimOffset[axisSOHW.ind()] = haloCornerSz;
            neighbourOffset[axisSOHW.ind()] = haloCornerSz;
        }

        const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

        const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);
        auto neighbourInwardHalo =
                VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

        const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
        auto outwardHalo =
                VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

        inwardHalosPerCluster[beforeIdx].push_back(neighbourInwardHalo);
        outwardHalosPerCluster[currentIdx].push_back(outwardHalo);
    }

    void insertHaloAfter(SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                         SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                         llvm::ArrayRef<mlir::Type> outputTypes, const size_t currentIdx, const size_t afterIdx,
                         const Dim& axis, const int64_t haloSz, const int64_t haloCornerSz = 0,
                         const int64_t haloNumPerCluster = 1, const bool haloBefore = false) {
        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes[currentIdx].cast<NDTypeInterface>().getShape());

        haloShape[axis.ind()] = haloSz;
        Dim axisSOHW = axis;

        if (axis == Dims4D::Act::H) {
            axisSOHW = Dims4D::Act::W;
        } else {
            axisSOHW = Dims4D::Act::H;
        }
        // In case of SOHW we need to subtract the other axis halo size.
        // Also it is posible to have multiple Halos for the other axis, and we need to take in consideration this.
        haloShape[axisSOHW.ind()] -= haloCornerSz * haloNumPerCluster;

        const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);
        auto ddrOutputType = outputTypes[currentIdx].cast<NDTypeInterface>();
        const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[currentIdx]);

        // offset in producer cluster
        SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
        auto* ctx = builder_.getContext();

        perDimOffset[axis.ind()] = ddrOutputType.getShape()[axis] - 2 * haloSz;

        const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[afterIdx]);
        // offset in the halo's target cluster
        SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
        if (haloBefore) {
            // If we inserted a halo before we must start the offset from that position.
            perDimOffset[axisSOHW.ind()] = haloCornerSz;
            neighbourOffset[axisSOHW.ind()] = haloCornerSz;
        }

        const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);
        const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);
        auto neighbourInwardHalo =
                VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

        const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
        auto outwardHalo =
                VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

        inwardHalosPerCluster[afterIdx].push_back(neighbourInwardHalo);
        outwardHalosPerCluster[currentIdx].push_back(outwardHalo);
    }

    void insertHaloCorner(SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                          SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                          llvm::ArrayRef<mlir::Type> outputTypes, const size_t currentIdx, const size_t secondIdx,
                          const int64_t heightHaloSz, const int64_t widthHaloSz, const HaloPosition positionHeight,
                          const HaloPosition positionWidth) {
        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes[currentIdx].cast<NDTypeInterface>().getShape());

        haloShape[Dims4D::Act::H.ind()] = heightHaloSz;
        haloShape[Dims4D::Act::W.ind()] = widthHaloSz;
        const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);
        auto ddrOutputType = outputTypes[currentIdx].cast<NDTypeInterface>();
        const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[currentIdx]);

        // offset in producer cluster
        SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
        // offset in the halo's target cluster
        SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
        auto* ctx = builder_.getContext();

        if (positionWidth == HaloPosition::BEFORE) {
            perDimOffset[Dims4D::Act::W.ind()] = widthHaloSz;
            neighbourOffset[Dims4D::Act::W.ind()] =
                    outputTypes[secondIdx].cast<NDTypeInterface>().getShape()[Dims4D::Act::W] - widthHaloSz;
        } else {
            perDimOffset[Dims4D::Act::W.ind()] = ddrOutputType.getShape()[Dims4D::Act::W] - 2 * widthHaloSz;
        }
        if (positionHeight == HaloPosition::BEFORE) {
            perDimOffset[Dims4D::Act::H.ind()] = heightHaloSz;
            neighbourOffset[Dims4D::Act::H.ind()] =
                    outputTypes[secondIdx].cast<NDTypeInterface>().getShape()[Dims4D::Act::H] - heightHaloSz;
        } else {
            perDimOffset[Dims4D::Act::H.ind()] = ddrOutputType.getShape()[Dims4D::Act::H] - 2 * heightHaloSz;
        }

        const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

        const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[secondIdx]);
        const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);
        auto neighbourInwardHalo =
                VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

        const auto inwardHaloAttr = builder_.getArrayAttr({neighbourInwardHalo});
        auto outwardHalo =
                VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

        inwardHalosPerCluster[secondIdx].push_back(neighbourInwardHalo);
        outwardHalosPerCluster[currentIdx].push_back(outwardHalo);
    }

    void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();
        VPUX_THROW_UNLESS(numClusters == 4 || numClusters == 6, "SOHW PSS tests supports only 4/6 clusters.");

        const auto widthNumClusters = clustersPerDim_[1];

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(numClusters);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(numClusters);

        // Conditions to insert halo BEFORE or AFTER over height or witdh for each cluster
        auto insertAfterWidth = [&](int64_t idx) -> bool {
            return idx % widthNumClusters < widthNumClusters - 1;
        };
        auto insertBeforeWidth = [&](int64_t idx) -> bool {
            return idx % widthNumClusters > 0;
        };
        auto insertAfterHeight = [&](int64_t idx) -> bool {
            return idx + widthNumClusters < static_cast<std::int64_t>(numClusters);
        };
        auto insertBeforeHeight = [&](int64_t idx) -> bool {
            return idx - widthNumClusters >= 0;
        };
        // Calculate corner cluster index
        auto cornerClusterIdx = [&](size_t idx, HaloPosition itHeight, HaloPosition itWidth) -> size_t {
            size_t clusterIdx = itWidth == HaloPosition::AFTER ? idx + 1 : idx - 1;
            clusterIdx =
                    itHeight == HaloPosition::AFTER ? clusterIdx + widthNumClusters : clusterIdx - widthNumClusters;
            return clusterIdx;
        };

        // Create outward halos for all clusters and add them to the neighbouring clusters' inward halos
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            // We will use heightHaloInserted and widthHaloInserted to calculate where we need Halo corner case
            // Also these 2 variables are needed to calculate the Halo height in case of multiple width Halos, also Halo
            // width in case of multiple height Halos
            SmallVector<HaloPosition> heightHaloInserted;
            SmallVector<HaloPosition> widthHaloInserted;
            // haloBeforeFlag it is needed to calculate the offsets for the current cluster
            auto haloBeforeFlagW = false;
            auto haloBeforeFlagH = false;

            // We check for each cluster how many halo will be inserted to know exact how to calculate the halo width
            // and height For example SOHW with 2 clusters per height and 3 per width:
            //    0 | 1 | 2
            //    -- --- --
            //    3 | 4 | 5
            //    the second cluster per width (number '1') will have 2 halos left and right (BEFORE&AFTER for '0' and
            //    '2') and because of this the halo inserted under (over height '4') will be smaller because we need to
            //    subtract the size of 2 halos:
            //         haloShape[H] = haloShape[H] - halowidthSz * haloWidthNumPerCluster;
            //    because for corner clusters we have only one halo for each dimension.
            if (insertAfterWidth(static_cast<std::int64_t>(idx))) {
                widthHaloInserted.push_back(HaloPosition::AFTER);
            }
            if (insertBeforeWidth(static_cast<std::int64_t>(idx))) {
                widthHaloInserted.push_back(HaloPosition::BEFORE);
                haloBeforeFlagW = true;
            }
            if (insertAfterHeight(static_cast<std::int64_t>(idx))) {
                heightHaloInserted.push_back(HaloPosition::AFTER);
            }
            if (insertBeforeHeight(static_cast<std::int64_t>(idx))) {
                heightHaloInserted.push_back(HaloPosition::BEFORE);
                haloBeforeFlagH = true;
            }

            // Insert width halo after in case is not the last element for each row of cluster
            if (insertAfterWidth(static_cast<std::int64_t>(idx))) {
                insertHaloAfter(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, idx + 1,
                                Dims4D::Act::W, widthHaloSz_, heightHaloSz_, heightHaloInserted.size(),
                                haloBeforeFlagH);
            }

            // Insert width halo before in case is not the first element for each row of cluster
            if (insertBeforeWidth(static_cast<std::int64_t>(idx))) {
                insertHaloBefore(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, idx - 1,
                                 Dims4D::Act::W, widthHaloSz_, heightHaloSz_, heightHaloInserted.size(),
                                 haloBeforeFlagH);
            }

            // Insert height halo after in case is not the last element for each column of cluster
            if (insertAfterHeight(static_cast<std::int64_t>(idx))) {
                insertHaloAfter(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, idx + widthNumClusters,
                                Dims4D::Act::H, heightHaloSz_, widthHaloSz_, widthHaloInserted.size(), haloBeforeFlagW);
            }

            // Insert height halo before in case is not the first element for each column of cluster
            if (insertBeforeHeight(static_cast<std::int64_t>(idx))) {
                insertHaloBefore(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx,
                                 idx - widthNumClusters, Dims4D::Act::H, heightHaloSz_, widthHaloSz_,
                                 widthHaloInserted.size(), haloBeforeFlagW);
            }

            // Insert corner halo depending of heightHaloInserted and widthHaloInserted.
            //    0 | 1 | 2
            //    -- --- --
            //    3 | 4 | 5
            //    if we have halo inserted after over width and after over height (this is the cluster '0') we know the
            //    corner will:
            //      clusterIdx = 0 + 1(we increase with 1 to go the the next cluster '1') + widthNumClusters (+3 to go
            //      under to cluster '4')
            //    For cluster 4 we have 2 halos over width(BEFORE & AFTER) and 1 over height(BEFORE):
            //      so the first corner will be BEFORE and BEFORE (4 -1 -widthNumClusters(3)) = cluster '0'
            //      and the second corner will be AFTER and BEFORE (4 + 1 - 3) = cluster '2'
            for (const auto& itWidth : widthHaloInserted) {
                for (const auto& itHeight : heightHaloInserted) {
                    const auto clusterIdx = cornerClusterIdx(idx, itHeight, itWidth);

                    insertHaloCorner(inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, idx, clusterIdx,
                                     heightHaloSz_, widthHaloSz_, itHeight, itWidth);
                }
            }
        }

        int64_t largestSliceSize = 0;
        for (std::size_t idx = 0; idx < numClusters; idx++) {
            SmallVector<HaloPosition> heightHaloInserted;
            SmallVector<HaloPosition> widthHaloInserted;
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();

            opBuffers_[idx].output =
                    createITIBuffer(ctx, builder_, ddrOutputType.getElementType(), ddrOutputType.getShape().raw(),
                                    ddrOutputType.getDimsOrder(), clusters_[idx], inwardHalosPerCluster[idx],
                                    outwardHalosPerCluster[idx], offset);

            if (insertAfterWidth(static_cast<std::int64_t>(idx))) {
                opBuffers_[idx + 1].outputIti.push_back(opBuffers_[idx].output);
                widthHaloInserted.push_back(HaloPosition::AFTER);
            }

            if (insertBeforeWidth(static_cast<std::int64_t>(idx))) {
                opBuffers_[idx - 1].outputIti.push_back(opBuffers_[idx].output);
                widthHaloInserted.push_back(HaloPosition::BEFORE);
            }

            if (insertAfterHeight(static_cast<std::int64_t>(idx))) {
                opBuffers_[idx + widthNumClusters].outputIti.push_back(opBuffers_[idx].output);
                heightHaloInserted.push_back(HaloPosition::AFTER);
            }

            if (insertBeforeHeight(static_cast<std::int64_t>(idx))) {
                opBuffers_[idx - widthNumClusters].outputIti.push_back(opBuffers_[idx].output);
                heightHaloInserted.push_back(HaloPosition::BEFORE);
            }

            for (const auto& itWidth : widthHaloInserted) {
                for (const auto& itHeight : heightHaloInserted) {
                    const auto clusterIdx = cornerClusterIdx(idx, itHeight, itWidth);

                    opBuffers_[clusterIdx].outputIti.push_back(opBuffers_[idx].output);
                }
            }

            const auto outputSliceSize =
                    opBuffers_[idx].output.getBuffer().getType().cast<NDTypeInterface>().getCompactAllocSize().count();
            if (largestSliceSize < outputSliceSize) {
                largestSliceSize = outputSliceSize;
            }
        }

        offset += largestSliceSize;
    }

    void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape, ArrayRef<int64_t> weightsShape,
                            mlir::ArrayAttr strides, VPU::PaddingAttr padding, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier,
                            const llvm::SmallVector<int64_t> clustersPerDim) override {
        auto ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();
        const auto origInputTypeIf = ddrInput.getType().cast<NDTypeInterface>();
        const auto origInputShape = origInputTypeIf.getShape();
        auto outputPerClusterShape = Shape(fullOutputShape);
        const auto heightNumClusters = clustersPerDim[0];
        const auto widthNumClusters = clustersPerDim[1];

        const auto outStepHeight =
                divUp(outputPerClusterShape[Dims4D::Act::H], static_cast<int64_t>(heightNumClusters));
        const auto outStepWidth = divUp(outputPerClusterShape[Dims4D::Act::W], static_cast<int64_t>(widthNumClusters));
        Shape outputOffsets{0, 0, 0, 0};
        Shape divAxis{1, 1, 1, 1};
        divAxis[Dims4D::Act::H] = heightNumClusters;
        divAxis[Dims4D::Act::W] = widthNumClusters;

        const auto inputStrides = origInputTypeIf.getStrides();
        const auto paddingInfo = PadInfo(padding.getLeft().getInt(), padding.getRight().getInt(),
                                         padding.getTop().getInt(), padding.getBottom().getInt());

        int64_t largestSliceSize = 0;
        for (int64_t idxH = 0; idxH < heightNumClusters; idxH++) {
            for (int64_t idxW = 0; idxW < widthNumClusters; idxW++) {
                const auto clusterIdx = idxH * widthNumClusters + idxW;
                outputPerClusterShape[Dims4D::Act::W] =
                        (idxW != widthNumClusters - 1) ? outStepWidth
                                                       : fullOutputShape[Dims4D::Act::W.ind()] - idxW * outStepWidth;
                outputPerClusterShape[Dims4D::Act::H] =
                        (idxH != heightNumClusters - 1) ? outStepHeight
                                                        : fullOutputShape[Dims4D::Act::H.ind()] - idxH * outStepHeight;

                outputOffsets[Dims4D::Act::W] = idxW * outStepWidth;
                outputOffsets[Dims4D::Act::H] = idxH * outStepHeight;

                // Use the shape of the output tile to back-infer the necessary slice of input to compute it
                // That is the slice of input that needs to be DMA'd to CMX, since inter-tile reads are not possible
                const TileInfo outputTile(outputPerClusterShape, outputOffsets, divAxis);
                const auto tilingSolution = vpux::backInferConvTile(outputTile, origInputShape, Shape(weightsShape),
                                                                    Shape(), strides, paddingInfo);
                const auto inputTile = tilingSolution.tiles.front();

                const Byte inSliceOffset =
                        inputTile.offsets[Dims4D::Act::H] * static_cast<Byte>(inputStrides[Dims4D::Act::H]) +
                        inputTile.offsets[Dims4D::Act::W] * static_cast<Byte>(inputStrides[Dims4D::Act::W]);
                auto networkInputBuffer = createDeclareTensorOp(builder_, VPURT::BufferSection::NetworkInput,
                                                                inputTile.shape.raw(), origInputTypeIf.getElementType(),
                                                                origInputTypeIf.getDimsOrder(), inputStrides,
                                                                /*locale=*/0, inSliceOffset.count());

                opBuffers_[clusterIdx].input =
                        createBuffer(ctx, builder_, origInputTypeIf.getElementType(), inputTile.shape.raw(),
                                     origInputTypeIf.getDimsOrder(), {clusters_[clusterIdx]}, offset);

                VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                      mlir::ValueRange(updateBarrier.getBarrier()), loc,
                                                      networkInputBuffer, opBuffers_[clusterIdx].input, 0);

                const auto inputSliceSize = opBuffers_[clusterIdx]
                                                    .input.getBuffer()
                                                    .getType()
                                                    .cast<NDTypeInterface>()
                                                    .getCompactAllocSize()
                                                    .count();
                if (largestSliceSize < inputSliceSize) {
                    largestSliceSize = inputSliceSize;
                }
            }
        }

        offset += largestSliceSize;
    }

private:
    void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                       VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) override {
        dmaDuplicatedWeightsBuffers(std::move(weightsContent), offset, updateBarrier, waitBarrier);
    }

    void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier) override {
        dmaDuplicatedWeightsTableBuffers(arch, inputType, outputType, offset, updateBarrier);
    }

    ArrayRef<int64_t> clustersPerDim_;
    int64_t heightHaloSz_;
    int64_t widthHaloSz_;
};

class SoHKStrategy final : public Strategy {
public:
    SoHKStrategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters, ArrayRef<int64_t> clustersPerDim,
                 const int64_t heightHaloSz)
            : Strategy(builder, clusters), clustersPerDim_(clustersPerDim), heightHaloSz_(heightHaloSz) {
    }

    bool isHaloDim(const llvm::ArrayRef<int64_t> /*fullShape*/, llvm::SmallVector<int64_t> haloShape,
                   size_t dim) override {
        return (static_cast<std::int32_t>(dim) == Dims4D::Act::H.ind() && haloShape[dim] == heightHaloSz_) ||
               (static_cast<std::int32_t>(dim) == Dims4D::Act::C.ind() &&
                haloShape[Dims4D::Act::H.ind()] != heightHaloSz_);
    }

    void insertHaloOverK(mlir::MLIRContext* ctx, SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                         SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                         llvm::ArrayRef<mlir::Type> outputTypes, const int64_t channelsNumClusters,
                         SmallVector<std::pair<size_t, uint64_t>>& inwardHaloProducerTarget) {
        const auto fullOutputChannelsNum = outputTypes.front().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto channelSlice = divUp(fullOutputChannelsNum, static_cast<int64_t>(channelsNumClusters));
        SmallVector<int64_t> outChannelsPerCluster(channelsNumClusters, channelSlice);
        outChannelsPerCluster.back() = fullOutputChannelsNum - (channelsNumClusters - 1) * channelSlice;
        const auto numClusters = clusters_.size();

        for (size_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto idxH = clusterIdx / channelsNumClusters;
            const auto idxK = clusterIdx % channelsNumClusters;
            const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[clusterIdx]);
            const auto crtDdrInput = outputTypes[clusterIdx].cast<NDTypeInterface>();

            SmallVector<int64_t> haloShape = llvm::to_vector(crtDdrInput.getShape());
            haloShape[Dims4D::Act::C.ind()] = outChannelsPerCluster[idxK];
            haloShape[Dims4D::Act::H.ind()] -= heightHaloSz_;

            const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);

            // Offset in producer cluster & in halo's target clusters
            const auto outChannelOffset =
                    std::accumulate(outChannelsPerCluster.begin(), outChannelsPerCluster.begin() + idxK,
                                    static_cast<int64_t>(0), [](const int64_t chOffset, const int64_t chSize) {
                                        return chOffset + chSize;
                                    });

            SmallVector<int64_t> dimOffests = {0, outChannelOffset, 0, 0};
            // All the clusters except the ones processing the upper part of the tensor will have upper inward height
            // halo. This means that only for the clusters processing the upper part the offset in
            // producer cluster & in halo's target clusters will start from the top
            if (idxH == 0) {
                dimOffests[Dims4D::Act::H.ind()] = 0;
            } else {
                dimOffests[Dims4D::Act::H.ind()] = heightHaloSz_;
            }
            const auto offsetAttr = getIntArrayAttr(builder_, dimOffests);

            auto inwardHalosVec = SmallVector<mlir::Attribute>();

            // Broadcast the output over channels
            for (auto idx = 0; idx < channelsNumClusters; idx++) {
                auto targetIdx = idxH * channelsNumClusters + idx;
                if (targetIdx == clusterIdx) {
                    continue;
                }

                const auto targetCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
                auto neighbourInwardHalo = VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, targetCluster);

                inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
                inwardHalosVec.push_back(neighbourInwardHalo);
                inwardHaloProducerTarget.push_back(std::make_pair(clusterIdx, targetIdx));
            }

            const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
            auto outwardHalo =
                    VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

            outwardHalosPerCluster[clusterIdx].push_back(outwardHalo);
        }
    }

    void insertHaloOverH(mlir::MLIRContext* ctx, SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                         SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                         llvm::ArrayRef<mlir::Type> outputTypes, const int64_t heightNumClusters,
                         const int64_t channelsNumClusters,
                         SmallVector<std::pair<size_t, uint64_t>>& inwardHaloProducerTarget) {
        const auto fullOutputChannelsNum = outputTypes.front().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto channelSlice = divUp(fullOutputChannelsNum, static_cast<int64_t>(channelsNumClusters));
        SmallVector<int64_t> outChannelsPerCluster(channelsNumClusters, channelSlice);
        outChannelsPerCluster.back() = fullOutputChannelsNum - (channelsNumClusters - 1) * channelSlice;
        const auto numClusters = clusters_.size();

        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes.front().cast<NDTypeInterface>().getShape());
        VPUX_THROW_UNLESS(std::size_t(Dims4D::Act::H.ind()) < haloShape.size(),
                          "insertHaloOverH: SoHKStrategy's H axis goes beyond the halo's shape");

        for (size_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            const auto idxH = clusterIdx / channelsNumClusters;
            const auto idxK = clusterIdx % channelsNumClusters;
            const auto ddrOutputType = outputTypes[clusterIdx].cast<NDTypeInterface>();
            const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[clusterIdx]);

            haloShape[Dims4D::Act::H.ind()] = heightHaloSz_;
            haloShape[Dims4D::Act::C.ind()] = outChannelsPerCluster[idxK];
            const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);

            // Offset in producer cluster
            SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
            VPUX_THROW_UNLESS(std::size_t(Dims4D::Act::H.ind()) < perDimOffset.size(),
                              "insertHaloOverH: SoHKStrategy's H axis goes beyond the perDimOffset's "
                              "array length");

            const auto outChannelOffset =
                    std::accumulate(outChannelsPerCluster.begin(), outChannelsPerCluster.begin() + idxK,
                                    static_cast<int64_t>(0), [](const int64_t chOffset, const int64_t chSize) {
                                        return chOffset + chSize;
                                    });

            auto inwardHalosVec = SmallVector<mlir::Attribute>();
            // All the clusters except the ones processing the upper part of the tensor will have a inward height halo
            // at the top of the workload. This means that the offset in producer cluster will start after
            // the heightHaloSz
            if (idxH != 0) {
                // Offset in producer cluster
                perDimOffset[Dims4D::Act::H.ind()] = heightHaloSz_;
                perDimOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

                // Offset in halo's target clusters will be at the end of the workload
                SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
                neighbourOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                neighbourOffset[Dims4D::Act::H.ind()] =
                        outputTypes[(idxH - 1) * heightNumClusters].cast<NDTypeInterface>().getShape()[Dims4D::Act::H] -
                        heightHaloSz_;
                VPUX_THROW_UNLESS(std::size_t(Dims4D::Act::H.ind()) < neighbourOffset.size(),
                                  "buildHaloMultiClusteringTest: SoHKStrategy's axis goes beyond the "
                                  "neighbourOffset's array length");

                const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);

                // Write the halo to the K number of clusters from idxH - 1
                // e.g. SOHK over 4 tiles (heightNumClusters = 2 and channelsNumClusters = 2)
                // Tile2: will write to Tile0 & Tile1 first K
                // Tile3: will write to Tile0 & Tile1 second K
                for (auto idx = 0; idx < channelsNumClusters; idx++) {
                    auto targetIdx = (idxH - 1) * channelsNumClusters + idx;

                    const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
                    auto neighbourInwardHalo =
                            VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

                    inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
                    inwardHalosVec.push_back(neighbourInwardHalo);
                    inwardHaloProducerTarget.push_back(std::make_pair(clusterIdx, targetIdx));
                }

                const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
                auto outwardHalo =
                        VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

                outwardHalosPerCluster[clusterIdx].push_back(outwardHalo);
            }

            inwardHalosVec = SmallVector<mlir::Attribute>();
            // All the clusters except the ones processing the bottom part of the tensor will have a inward height halo
            // at the bottom of the workload. This means that the offset in producer cluster will be 2 * heightHaloSz
            // above the bottom of the workload
            if (static_cast<std::int64_t>(idxH) != heightNumClusters - 1) {
                // Offset in producer cluster
                perDimOffset[Dims4D::Act::H.ind()] = ddrOutputType.getShape()[Dims4D::Act::H] - 2 * heightHaloSz_;
                perDimOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

                // Offset in halo's target clusters will be at the top of the workload
                SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
                neighbourOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);

                // Write the halo to the K number of clusters from idxH + 1
                // e.g. SOHK over 4 tiles (heightNumClusters = 2 and channelsNumClusters = 2)
                // Tile0: will write to Tile2 & Tile3 first K
                // Tile1: will write to Tile2 & Tile3 second K
                for (auto idx = 0; idx < channelsNumClusters; idx++) {
                    auto targetIdx = (idxH + 1) * channelsNumClusters + idx;

                    const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
                    auto neighbourInwardHalo =
                            VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);
                    inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
                    inwardHalosVec.push_back(neighbourInwardHalo);
                    inwardHaloProducerTarget.push_back(std::make_pair(clusterIdx, targetIdx));
                }

                const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
                auto outwardHalo =
                        VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);
                outwardHalosPerCluster[clusterIdx].push_back(outwardHalo);
            }
        }
    }

    void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();

        const auto heightNumClusters = clustersPerDim_[0];
        const auto channelsNumClusters = clustersPerDim_[1];

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(numClusters);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(numClusters);

        SmallVector<std::pair<size_t, uint64_t>> inwardHaloProducerTarget;

        // Create outward halos over channels and add them to the neighbouring clusters' inward halos
        insertHaloOverK(ctx, inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, channelsNumClusters,
                        inwardHaloProducerTarget);

        // Create outward halos over height and add them to the neighbouring clusters' inward halos
        insertHaloOverH(ctx, inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, heightNumClusters,
                        channelsNumClusters, inwardHaloProducerTarget);

        for (size_t idx = 0; idx < numClusters; idx++) {
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();

            opBuffers_[idx].output =
                    createITIBuffer(ctx, builder_, ddrOutputType.getElementType(), ddrOutputType.getShape().raw(),
                                    ddrOutputType.getDimsOrder(), clusters_[idx], inwardHalosPerCluster[idx],
                                    outwardHalosPerCluster[idx], offset);

            for (auto targetIdx : inwardHaloProducerTarget) {
                if (targetIdx.second == idx) {
                    opBuffers_[targetIdx.first].outputIti.push_back(opBuffers_[idx].output);
                }
            }
        }

        offset += opBuffers_.front().output.getBuffer().getType().cast<NDTypeInterface>().getTotalAllocSize().count();
    }

    // Input activation partially replicated: each tile will have at input the full channels and will be splitted over
    // height e.g. SOHK over 4 tiles (heightNumClusters = 2 and channelsNumClusters = 2)
    // Tile0 & Tile1 input: upper half of the tensor
    // Tile2 & Tile3 input: bottom half of the tensor
    void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape, ArrayRef<int64_t> weightsShape,
                            mlir::ArrayAttr strides, VPU::PaddingAttr padding, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier,
                            const llvm::SmallVector<int64_t> clustersPerDim) override {
        auto ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();
        const auto origInputTypeIf = ddrInput.getType().cast<NDTypeInterface>();
        const auto origInputShape = origInputTypeIf.getShape();
        auto outputPerClusterShape = Shape(fullOutputShape);
        const auto heightNumClusters = clustersPerDim[0];
        const auto channelsNumClusters = clustersPerDim[1];
        const auto outStepHeight =
                divUp(outputPerClusterShape[Dims4D::Act::H], static_cast<int64_t>(heightNumClusters));

        Shape outputOffsets{0, 0, 0, 0};
        Shape divAxis{1, 1, 1, 1};
        divAxis[Dims4D::Act::H] = heightNumClusters;
        divAxis[Dims4D::Act::C] = channelsNumClusters;

        const auto inputStrides = origInputTypeIf.getStrides();
        const auto paddingInfo = PadInfo(padding.getLeft().getInt(), padding.getRight().getInt(),
                                         padding.getTop().getInt(), padding.getBottom().getInt());

        int64_t largestSliceSize = 0;
        for (auto idxH = 0; idxH < heightNumClusters; idxH++) {
            // splitting over height for each tile
            outputPerClusterShape[Dims4D::Act::H] =
                    (idxH != heightNumClusters - 1) ? outStepHeight
                                                    : fullOutputShape[Dims4D::Act::H.ind()] - idxH * outStepHeight;
            outputOffsets[Dims4D::Act::H] = idxH * outStepHeight;

            for (auto idxK = 0; idxK < channelsNumClusters; idxK++) {
                const auto clusterIdx = idxH * channelsNumClusters + idxK;
                // Use the shape of the output tile to back-infer the necessary slice of input to compute it
                // That is the slice of input that needs to be DMA'd to CMX, since inter-tile reads are not possible
                const TileInfo outputTile(outputPerClusterShape, outputOffsets, divAxis);
                const auto tilingSolution = vpux::backInferConvTile(outputTile, origInputShape, Shape(weightsShape),
                                                                    Shape(), strides, paddingInfo);
                const auto inputTile = tilingSolution.tiles.front();
                const Byte inSliceOffset =
                        inputTile.offsets[Dims4D::Act::H] * static_cast<Byte>(inputStrides[Dims4D::Act::H]);

                auto networkInputBuffer = createDeclareTensorOp(builder_, VPURT::BufferSection::NetworkInput,
                                                                inputTile.shape.raw(), origInputTypeIf.getElementType(),
                                                                origInputTypeIf.getDimsOrder(), inputStrides,
                                                                /*locale=*/0, inSliceOffset.count());

                opBuffers_[clusterIdx].input =
                        createBuffer(ctx, builder_, origInputTypeIf.getElementType(), inputTile.shape.raw(),
                                     origInputTypeIf.getDimsOrder(), {clusters_[clusterIdx]}, offset);

                VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                      mlir::ValueRange(updateBarrier.getBarrier()), loc,
                                                      networkInputBuffer, opBuffers_[clusterIdx].input, 0);

                const auto inputSliceSize = opBuffers_[clusterIdx]
                                                    .input.getBuffer()
                                                    .getType()
                                                    .cast<NDTypeInterface>()
                                                    .getCompactAllocSize()
                                                    .count();

                if (largestSliceSize < inputSliceSize) {
                    largestSliceSize = inputSliceSize;
                }
            }
        }

        offset += largestSliceSize;
    }

private:
    // Weights are partially splitted: splitted over channels number of clusters and replicated over height number of
    // clusters e.g. SOHK over 4 tiles (heightNumClusters = 2 and channelsNumClusters = 2)
    // Weights splitted over: Tile0 & Tile1
    // Weights replicated: Tile0 - Tile2 and Tile1 - Tile3
    void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                       VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) override {
        auto* ctx = builder_.getContext();
        const auto heightNumClusters = clustersPerDim_[0];
        const auto channelsNumClusters = clustersPerDim_[1];
        auto loc = builder_.getUnknownLoc();

        const auto weightsTypeIf = weightsContent.getType();
        const auto weightsShape = weightsTypeIf.getShape();
        const auto weightsElementType = weightsTypeIf.getElementType();

        // Divide the full OC by the number of clusters; round up if K is not a multiple of numClusters
        const auto fullK = weightsShape[vpux::Dims4D::Filter::OC];
        const auto kStep = divUp(fullK, static_cast<int64_t>(channelsNumClusters));

        for (auto idxK = 0; idxK < channelsNumClusters; idxK++) {
            SmallVector<int64_t> targetClusters;
            const Shape weightsOffset{static_cast<int64_t>(idxK) * kStep, 0, 0, 0};

            // Ensure the last cluster gets the reminder of output channels
            const auto outputChannels =
                    (idxK != channelsNumClusters - 1) ? kStep : fullK - (channelsNumClusters - 1) * kStep;
            const auto perClusterShape =
                    Shape{static_cast<int64_t>(outputChannels), weightsShape[vpux::Dims4D::Filter::IC],
                          weightsShape[vpux::Dims4D::Filter::KY], weightsShape[vpux::Dims4D::Filter::KX]};

            targetClusters.push_back(idxK);
            for (auto idxH = 0; idxH < heightNumClusters - 1; idxH++) {
                auto targetIdx = (idxH + 1) * channelsNumClusters + idxK;
                targetClusters.push_back(targetIdx);
            }

            auto weightsBuffer = createBuffer(ctx, builder_, weightsElementType, perClusterShape.raw(), DimsOrder::OYXI,
                                              targetClusters, offset);

            // Create a DDR buffer for each slice of the weights
            const auto weightsDDRMemRefType = getMemRefType(VPURT::BufferSection::Constant, perClusterShape.raw(),
                                                            weightsElementType, weightsTypeIf.getDimsOrder());

            // Create weights slice by using subview on the full weights content
            auto weightsDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                    loc, weightsDDRMemRefType,
                    weightsContent.transform().subview(weightsOffset, perClusterShape).get());

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, waitBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                                  loc, weightsDDRBuffer, weightsBuffer, 0);

            opBuffers_[idxK].weights = weightsBuffer;
            // Replicated weights over height number of clusters
            for (auto idxH = 0; idxH < heightNumClusters - 1; idxH++) {
                auto targetIdx = (idxH + 1) * channelsNumClusters + idxK;
                opBuffers_[targetIdx].weights = opBuffers_[idxK].weights;
            }
        }
    }

    void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier) override {
        const auto heightNumClusters = clustersPerDim_[0];
        const auto channelsNumClusters = clustersPerDim_[1];
        auto loc = builder_.getUnknownLoc();
        auto ctx = builder_.getContext();
        auto int32 = builder_.getIntegerType(32, true);

        const auto sparsityPtrStep = 0;
        const auto weightsBuffType = opBuffers_.front().weights.getBuffer().getType().cast<NDTypeInterface>();
        auto weightsOutputChannelsStrideInBits = weightsBuffType.getStrides()[vpux::Dims4D::Filter::OC];
        const auto weightsElemType = weightsBuffType.getElementType();

        const auto alignmentRequirement = 16;
        const auto alignment =
                (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
        if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
            weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
        }

        for (auto idxK = 0; idxK < channelsNumClusters; idxK++) {
            // Create weights table DDR buffer for each tile
            const auto outChannels = opBuffers_[idxK]
                                             .weights.getBuffer()
                                             .getType()
                                             .cast<NDTypeInterface>()
                                             .getShape()[Dims4D::Filter::OC];
            const auto wtableShape = SmallVector<int64_t>({outChannels, 1, 1, 4});
            const auto wtableDDRType =
                    getMemRefType(VPURT::BufferSection::Constant, wtableShape, int32, DimsOrder::NHWC);

            // Create weights table content for each weights table chunck
            const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
            const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
            const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                    inputType, outputType, 0,
                    static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                    VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
                    outChannels, weightsElemType);
            auto wtableTensorType = mlir::RankedTensorType::get(wtableShape, int32);
            const auto weightsTableValues =
                    mlir::DenseElementsAttr::get(wtableTensorType, llvm::ArrayRef<std::int32_t>(weightsTable));

            auto wtableDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                    loc, wtableDDRType,
                    vpux::Const::ContentAttr::get(
                            weightsTableValues, Const::ContentSetup(wtableTensorType).reorder(vpux::DimsOrder::NHWC)));

            SmallVector<int64_t> targetClusters;
            targetClusters.push_back(idxK);
            for (auto idxH = 0; idxH < heightNumClusters - 1; idxH++) {
                auto targetIdx = (idxH + 1) * channelsNumClusters + idxK;
                targetClusters.push_back(targetIdx);
            }

            auto wtableCMXBuffer =
                    createBuffer(ctx, builder_, int32, wtableShape, DimsOrder::NHWC, targetClusters, offset);

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, wtableDDRBuffer,
                                                  wtableCMXBuffer, 0);

            auto wtableCMXMemRefType =
                    getMemRefType(VPURT::BufferSection::CMX_NN, clusters_[idxK], wtableShape, int32, DimsOrder::NHWC);
            opBuffers_[idxK].weightsTable = createDeclareTensorOp(
                    builder_, wtableCMXMemRefType, VPURT::BufferSection::CMX_NN, clusters_[idxK], offset);

            for (auto idxH = 0; idxH < heightNumClusters - 1; idxH++) {
                auto targetIdx = (idxH + 1) * channelsNumClusters + idxK;

                wtableCMXMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, clusters_[targetIdx], wtableShape,
                                                    int32, DimsOrder::NHWC);

                opBuffers_[targetIdx].weightsTable = createDeclareTensorOp(
                        builder_, wtableCMXMemRefType, VPURT::BufferSection::CMX_NN, clusters_[targetIdx], offset);
            }
        }
    }

    ArrayRef<int64_t> clustersPerDim_;
    int64_t heightHaloSz_;
};

class SoHK3Strategy final : public Strategy {
public:
    SoHK3Strategy(mlir::OpBuilder& builder, ArrayRef<int64_t> clusters, int64_t splitNum, const int64_t heightHaloSz)
            : Strategy(builder, clusters), splitNum_(splitNum), heightHaloSz_(heightHaloSz) {
    }

    bool isHaloDim(const llvm::ArrayRef<int64_t> /*fullShape*/, llvm::SmallVector<int64_t> haloShape,
                   size_t dim) override {
        return (static_cast<std::int32_t>(dim) == Dims4D::Act::H.ind() && haloShape[dim] == heightHaloSz_) ||
               (static_cast<std::int32_t>(dim) == Dims4D::Act::C.ind() &&
                haloShape[Dims4D::Act::H.ind()] != heightHaloSz_);
    }

    // Tile0 Tile2(up)
    // Tile1 Tile2(down)
    void insertHaloOverK(mlir::MLIRContext* ctx, SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                         SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                         llvm::ArrayRef<mlir::Type> outputTypes,
                         SmallVector<std::pair<size_t, uint64_t>>& inwardHaloProducerTarget) {
        const auto fullOutputChannelsNum = outputTypes.front().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto channelSlice = divUp(fullOutputChannelsNum, static_cast<int64_t>(splitNum_));
        SmallVector<int64_t> outChannelsPerCluster(splitNum_, channelSlice);
        outChannelsPerCluster.back() = fullOutputChannelsNum - (splitNum_ - 1) * channelSlice;
        const auto numClusters = clusters_.size();

        // 4 sections: Section0 - Section1 - Section3 - Section4
        // Tile0 - Section0
        // Tile1 - Section1
        // Tile2(up) - Section3
        // Tile2(down) - Section4
        const auto sections = splitNum_ * splitNum_;
        const auto lastSectionIdx = sections - 1;
        const auto lastClusterIdx = numClusters - 1;

        for (auto sectionIdx = 0; sectionIdx < sections; sectionIdx++) {
            auto clusterIdx = sectionIdx;
            if (sectionIdx == lastSectionIdx)
                clusterIdx = lastClusterIdx;

            const auto idxK = sectionIdx / splitNum_;
            const auto idxH = sectionIdx % splitNum_;
            const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[clusterIdx]);
            // tile2 (tile2(up) & tile2(down)) has all tensor at output - full height
            // tile0 & tile1 height is only half
            // Means that:
            // Halo over K received by tile0 from tile2 should have height of tile0 + the heightHaloSz
            // Halo over K received by tile1 from tile2 should have height of tile1 + the heightHaloSz
            const auto crtDdrInput = outputTypes[sectionIdx % splitNum_].cast<NDTypeInterface>();

            SmallVector<int64_t> haloShape = llvm::to_vector(crtDdrInput.getShape());
            haloShape[Dims4D::Act::C.ind()] = outChannelsPerCluster[idxK];
            // Halo over K received by tile2 size won't include heightHaloSz
            if (clusterIdx != static_cast<int8_t>(lastClusterIdx)) {
                haloShape[Dims4D::Act::H.ind()] -= heightHaloSz_;
            }

            const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);

            // channel offset in producer cluster & in halo's target clusters
            const auto outChannelOffset =
                    std::accumulate(outChannelsPerCluster.begin(), outChannelsPerCluster.begin() + idxK,
                                    static_cast<int64_t>(0), [](const int64_t chOffset, const int64_t chSize) {
                                        return chOffset + chSize;
                                    });

            SmallVector<int64_t> dimOffests = {0, outChannelOffset, 0, 0};
            // For the upper part processing
            // the offset in producer cluster will start from the top
            if (idxH == 0) {
                dimOffests[Dims4D::Act::H.ind()] = 0;
            } else {
                if (idxK == 0) {
                    // Offset in producer cluster after heightHaloSz
                    dimOffests[Dims4D::Act::H.ind()] = heightHaloSz_;
                } else {
                    // Since the outward halo over K for cluster 2 size will include also the heightHaloSz
                    // the offset in producer cluster will be heightHaloSz above the size of tile0 height
                    const auto prevDdrInput = outputTypes[0].cast<NDTypeInterface>();

                    // prevShape is height + heightHaloSz
                    SmallVector<int64_t> prevShape = llvm::to_vector(prevDdrInput.getShape());
                    prevShape[Dims4D::Act::H.ind()] -= 2 * heightHaloSz_;
                    dimOffests[Dims4D::Act::H.ind()] = prevShape[Dims4D::Act::H.ind()];
                }
            }

            // Offset in producer cluster - used for outwardHalo
            const auto offsetAttr = getIntArrayAttr(builder_, dimOffests);

            // Offset in halo's target clusters - used for neighbourInwardHalo
            SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
            neighbourOffset[Dims4D::Act::C.ind()] = dimOffests[Dims4D::Act::C.ind()];
            // For the upper part processing
            // the offset in target cluster will start from the top
            if (idxH == 0) {
                neighbourOffset[Dims4D::Act::H.ind()] = 0;
            } else {
                if (idxK == 0) {
                    // After the Height of the tile0
                    const auto prevDdrInput = outputTypes[0].cast<NDTypeInterface>();

                    SmallVector<int64_t> prevShape = llvm::to_vector(prevDdrInput.getShape());
                    prevShape[Dims4D::Act::H.ind()] -= heightHaloSz_;
                    neighbourOffset[Dims4D::Act::H.ind()] = prevShape[Dims4D::Act::H.ind()];
                } else {
                    // Since halo over K received by tile1 should have height of tile1 + the heightHaloSz
                    // means that offset will be at 0
                    neighbourOffset[Dims4D::Act::H.ind()] = 0;
                }
            }

            const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);

            auto inwardHalosVec = SmallVector<mlir::Attribute>();

            auto targetIdx = (sectionIdx + splitNum_) % sections;
            if (targetIdx == lastSectionIdx) {
                targetIdx = lastClusterIdx;
            }

            const auto targetCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
            auto neighbourInwardHalo =
                    VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, targetCluster);

            inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
            inwardHalosVec.push_back(neighbourInwardHalo);
            inwardHaloProducerTarget.push_back(std::make_pair(clusterIdx, targetIdx));

            const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
            auto outwardHalo =
                    VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

            outwardHalosPerCluster[clusterIdx].push_back(outwardHalo);
        }
    }

    // Tile0 Tile2(up)
    // Tile1 Tile2(down)
    // only Tile0 & Tile1 have Halo over height
    void insertHaloOverH(mlir::MLIRContext* ctx, SmallVector<SmallVector<VPUIP::HaloRegionAttr>>& inwardHalosPerCluster,
                         SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>>& outwardHalosPerCluster,
                         llvm::ArrayRef<mlir::Type> outputTypes,
                         SmallVector<std::pair<size_t, uint64_t>>& inwardHaloProducerTarget) {
        const auto fullOutputChannelsNum = outputTypes.front().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto channelSlice = divUp(fullOutputChannelsNum, static_cast<int64_t>(splitNum_));
        SmallVector<int64_t> outChannelsPerCluster(splitNum_, channelSlice);
        outChannelsPerCluster.back() = fullOutputChannelsNum - (splitNum_ - 1) * channelSlice;

        SmallVector<int64_t> haloShape = llvm::to_vector(outputTypes.front().cast<NDTypeInterface>().getShape());
        VPUX_THROW_UNLESS(std::size_t(Dims4D::Act::H.ind()) < haloShape.size(),
                          "insertHaloOverH: SoHKStrategy's H axis goes beyond the halo's shape");

        for (auto clusterIdx = 0; clusterIdx < splitNum_; clusterIdx++) {
            const auto idxH = clusterIdx;
            // only idxK 0 has halo over height
            const auto idxK = 0;
            const auto ddrOutputType = outputTypes[clusterIdx].cast<NDTypeInterface>();
            const auto clusterAttr = builder_.getI64IntegerAttr(clusters_[clusterIdx]);

            haloShape[Dims4D::Act::H.ind()] = heightHaloSz_;
            haloShape[Dims4D::Act::C.ind()] = outChannelsPerCluster[idxK];
            const auto haloShapeAttr = getIntArrayAttr(builder_, haloShape);

            // Offset in producer cluster
            SmallVector<int64_t> perDimOffset = {0, 0, 0, 0};
            VPUX_THROW_UNLESS(std::size_t(Dims4D::Act::H.ind()) < perDimOffset.size(),
                              "insertHaloOverH: SoHKStrategy's H axis goes beyond the perDimOffset's "
                              "array length");

            const auto outChannelOffset =
                    std::accumulate(outChannelsPerCluster.begin(), outChannelsPerCluster.begin() + idxK,
                                    static_cast<int64_t>(0), [](const int64_t chOffset, const int64_t chSize) {
                                        return chOffset + chSize;
                                    });

            auto inwardHalosVec = SmallVector<mlir::Attribute>();
            // All the clusters except the ones processing the upper part of the tensor will have a inward height
            // halo at the top of the workload. This means that the offset in producer cluster will start after the
            // heightHaloSz
            if (idxH != 0) {
                // Offset in producer cluster
                perDimOffset[Dims4D::Act::H.ind()] = heightHaloSz_;
                perDimOffset[Dims4D::Act::C.ind()] = outChannelOffset;

                const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

                // Offset in halo's target clusters will be at the end of the workload
                SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
                neighbourOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                neighbourOffset[Dims4D::Act::H.ind()] =
                        outputTypes[(idxH - 1) * splitNum_].cast<NDTypeInterface>().getShape()[Dims4D::Act::H] -
                        heightHaloSz_;
                VPUX_THROW_UNLESS(std::size_t(Dims4D::Act::H.ind()) < neighbourOffset.size(),
                                  "buildHaloMultiClusteringTest: SoHKStrategy's axis goes beyond the "
                                  "neighbourOffset's array length");

                const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);

                // Tile1: will write to Tile0 first K
                auto targetIdx = clusterIdx - 1;

                const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
                auto neighbourInwardHalo =
                        VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);

                inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
                inwardHalosVec.push_back(neighbourInwardHalo);
                inwardHaloProducerTarget.push_back(std::make_pair(clusterIdx, targetIdx));

                const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
                auto outwardHalo =
                        VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

                outwardHalosPerCluster[clusterIdx].push_back(outwardHalo);
            }

            inwardHalosVec = SmallVector<mlir::Attribute>();
            // All the clusters except the ones processing the bottom part of the tensor will have a inward height
            // halo at the bottom of the workload. This means that the offset in producer cluster will be 2 *
            // heightHaloSz above the bottom of the workload
            if (static_cast<std::int64_t>(idxH) != splitNum_ - 1) {
                // Offset in producer cluster
                perDimOffset[Dims4D::Act::H.ind()] = ddrOutputType.getShape()[Dims4D::Act::H] - 2 * heightHaloSz_;
                perDimOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                const auto offsetAttr = getIntArrayAttr(builder_, perDimOffset);

                // Offset in halo's target clusters will be at the top of the workload
                SmallVector<int64_t> neighbourOffset = {0, 0, 0, 0};
                neighbourOffset[Dims4D::Act::C.ind()] = outChannelOffset;
                const auto neigbourHaloOffsetAttr = getIntArrayAttr(builder_, neighbourOffset);

                // Tile0: will write to Tile1 first K
                auto targetIdx = clusterIdx + 1;

                const auto neighbourCluster = builder_.getI64IntegerAttr(clusters_[targetIdx]);
                auto neighbourInwardHalo =
                        VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);
                inwardHalosPerCluster[targetIdx].push_back(neighbourInwardHalo);
                inwardHalosVec.push_back(neighbourInwardHalo);
                inwardHaloProducerTarget.push_back(std::make_pair(clusterIdx, targetIdx));

                const auto inwardHaloAttr = builder_.getArrayAttr(inwardHalosVec);
                auto outwardHalo =
                        VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);
                outwardHalosPerCluster[clusterIdx].push_back(outwardHalo);
            }
        }
    }

    void createOutputItiBuffers(ArrayRef<mlir::Type> outputTypes, std::size_t& offset) override {
        auto* ctx = builder_.getContext();
        const auto numClusters = clusters_.size();

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(numClusters);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(numClusters);

        SmallVector<std::pair<size_t, uint64_t>> inwardHaloProducerTarget;

        // Create outward halos over channels and add them to the neighbouring clusters' inward halos
        insertHaloOverK(ctx, inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, inwardHaloProducerTarget);

        // Create outward halos over height and add them to the neighbouring clusters' inward halos
        insertHaloOverH(ctx, inwardHalosPerCluster, outwardHalosPerCluster, outputTypes, inwardHaloProducerTarget);

        for (size_t idx = 0; idx < numClusters; idx++) {
            auto ddrOutputType = outputTypes[idx].cast<NDTypeInterface>();

            opBuffers_[idx].output =
                    createITIBuffer(ctx, builder_, ddrOutputType.getElementType(), ddrOutputType.getShape().raw(),
                                    ddrOutputType.getDimsOrder(), clusters_[idx], inwardHalosPerCluster[idx],
                                    outwardHalosPerCluster[idx], offset);

            for (auto targetIdx : inwardHaloProducerTarget) {
                if (targetIdx.second == idx) {
                    opBuffers_[targetIdx.first].outputIti.push_back(opBuffers_[idx].output);
                }
            }
        }

        // last cluster will contain all tensor at output
        offset += opBuffers_.back().output.getBuffer().getType().cast<NDTypeInterface>().getTotalAllocSize().count();
    }

    // Input activation:
    // e.g. SOHK over 3 tiles
    // Tile0 Tile2(up)
    // Tile1 Tile2(down)
    // Tile0 input: upper half of the tensor
    // Tile1 input: bottom half of the tensor
    // Tile2 input: all tensor
    void createInputBuffers(mlir::Value ddrInput, ArrayRef<int64_t> fullOutputShape, ArrayRef<int64_t> weightsShape,
                            mlir::ArrayAttr strides, VPU::PaddingAttr padding, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier,
                            const llvm::SmallVector<int64_t> /*clustersPerDim*/) override {
        auto ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();
        const auto origInputTypeIf = ddrInput.getType().cast<NDTypeInterface>();
        const auto origInputShape = origInputTypeIf.getShape();
        auto outputPerClusterShape = Shape(fullOutputShape);
        const auto outStepHeight = divUp(outputPerClusterShape[Dims4D::Act::H], static_cast<int64_t>(splitNum_));

        Shape outputOffsets{0, 0, 0, 0};
        Shape divAxis{1, 1, 1, 1};
        divAxis[Dims4D::Act::H] = splitNum_;
        divAxis[Dims4D::Act::C] = splitNum_;

        const auto inputStrides = origInputTypeIf.getStrides();
        const auto paddingInfo = PadInfo(padding.getLeft().getInt(), padding.getRight().getInt(),
                                         padding.getTop().getInt(), padding.getBottom().getInt());

        // Tile0 input: upper half of the tensor
        // Tile1 input: bottom half of the tensor
        for (auto idxH = 0; idxH < splitNum_; idxH++) {
            // splitting over height for the first 2 tiles
            outputPerClusterShape[Dims4D::Act::H] =
                    (idxH != splitNum_ - 1) ? outStepHeight
                                            : fullOutputShape[Dims4D::Act::H.ind()] - idxH * outStepHeight;
            outputOffsets[Dims4D::Act::H] = idxH * outStepHeight;

            const auto clusterIdx = idxH;
            // Use the shape of the output tile to back-infer the necessary slice of input to compute it
            // That is the slice of input that needs to be DMA'd to CMX, since inter-tile reads are not possible
            const TileInfo outputTile(outputPerClusterShape, outputOffsets, divAxis);
            const auto tilingSolution = vpux::backInferConvTile(outputTile, origInputShape, Shape(weightsShape),
                                                                Shape(), strides, paddingInfo);
            const auto inputTile = tilingSolution.tiles.front();
            const Byte inSliceOffset =
                    inputTile.offsets[Dims4D::Act::H] * static_cast<Byte>(inputStrides[Dims4D::Act::H]);

            auto networkInputBuffer = createDeclareTensorOp(builder_, VPURT::BufferSection::NetworkInput,
                                                            inputTile.shape.raw(), origInputTypeIf.getElementType(),
                                                            origInputTypeIf.getDimsOrder(), inputStrides,
                                                            /*locale=*/0, inSliceOffset.count());

            opBuffers_[clusterIdx].input =
                    createBuffer(ctx, builder_, origInputTypeIf.getElementType(), inputTile.shape.raw(),
                                 origInputTypeIf.getDimsOrder(), {clusters_[clusterIdx]}, offset);

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, networkInputBuffer,
                                                  opBuffers_[clusterIdx].input, 0);
        }

        const auto lastClusterIdx = clusters_.size() - 1;
        // Tile2 input: all tensor
        // The entire DDR input resides in one buffer in NetworkInput Section
        auto networkInputBuffer = createDeclareTensorOp(
                builder_, VPURT::BufferSection::NetworkInput, origInputTypeIf.getShape().raw(),
                origInputTypeIf.getElementType(), origInputTypeIf.getDimsOrder(), origInputTypeIf.getStrides(),
                /*locale=*/0, /*offset=*/0);

        opBuffers_[lastClusterIdx].input =
                createBuffer(ctx, builder_, origInputTypeIf.getElementType(), origInputTypeIf.getShape().raw(),
                             origInputTypeIf.getDimsOrder(), {clusters_[lastClusterIdx]}, offset);

        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                              mlir::ValueRange(updateBarrier.getBarrier()), loc, networkInputBuffer,
                                              opBuffers_[lastClusterIdx].input, 0);

        // largestSliceSize will be the lastClusterIdx size (the one with the whole tensor)
        offset += opBuffers_[lastClusterIdx]
                          .input.getBuffer()
                          .getType()
                          .cast<NDTypeInterface>()
                          .getCompactAllocSize()
                          .count();
    }

private:
    // Weights are partially splitted: splitted over channels number of clusters and replicated over height number
    // of clusters e.g. SOHK3 over 3 tiles
    // Tile0 Tile2(up)
    // Tile1 Tile2(down)
    // Weights splitted over: Tile0 - Tile2
    // Weights replicated: Tile0 - Tile1
    void handleWeights(Const::ContentAttr&& weightsContent, std::size_t& offset,
                       VPURT::ConfigureBarrierOp updateBarrier, mlir::ValueRange waitBarrier) override {
        auto* ctx = builder_.getContext();
        auto loc = builder_.getUnknownLoc();

        const auto weightsTypeIf = weightsContent.getType();
        const auto weightsShape = weightsTypeIf.getShape();
        const auto weightsElementType = weightsTypeIf.getElementType();
        const auto numClusters = clusters_.size();

        // Divide the full OC by the number of clusters; round up if K is not a multiple of numClusters
        const auto fullK = weightsShape[vpux::Dims4D::Filter::OC];
        const auto kStep = divUp(fullK, static_cast<int64_t>(splitNum_));

        for (auto clusterIdx = 0, indexK = 0; clusterIdx < static_cast<int>(numClusters); clusterIdx += 2, indexK++) {
            SmallVector<int64_t> targetClusters;
            const Shape weightsOffset{static_cast<int64_t>(indexK) * kStep, 0, 0, 0};

            // Ensure the last cluster gets the reminder of output channels
            const auto outputChannels =
                    (indexK != static_cast<int>(splitNum_) - 1) ? kStep : fullK - (splitNum_ - 1) * kStep;
            const auto perClusterShape =
                    Shape{static_cast<int64_t>(outputChannels), weightsShape[vpux::Dims4D::Filter::IC],
                          weightsShape[vpux::Dims4D::Filter::KY], weightsShape[vpux::Dims4D::Filter::KX]};

            targetClusters.push_back(clusterIdx);
            auto targetIdx = clusterIdx + 1;
            if (targetIdx < splitNum_) {
                targetClusters.push_back(targetIdx);
            }

            auto weightsBuffer = createBuffer(ctx, builder_, weightsElementType, perClusterShape.raw(), DimsOrder::OYXI,
                                              targetClusters, offset);

            // Create a DDR buffer for each slice of the weights
            const auto weightsDDRMemRefType = getMemRefType(VPURT::BufferSection::Constant, perClusterShape.raw(),
                                                            weightsElementType, weightsTypeIf.getDimsOrder());

            // Create weights slice by using subview on the full weights content
            auto weightsDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                    loc, weightsDDRMemRefType,
                    weightsContent.transform().subview(weightsOffset, perClusterShape).get());

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, waitBarrier, mlir::ValueRange(updateBarrier.getBarrier()),
                                                  loc, weightsDDRBuffer, weightsBuffer, 0);

            opBuffers_[clusterIdx].weights = weightsBuffer;
            // Replicated weights over height number of clusters
            if (targetIdx < splitNum_) {
                opBuffers_[targetIdx].weights = opBuffers_[clusterIdx].weights;
            }
        }
    }

    void handleWeightsTable(VPU::ArchKind arch, mlir::Type inputType, mlir::Type outputType, std::size_t& offset,
                            VPURT::ConfigureBarrierOp updateBarrier) override {
        auto loc = builder_.getUnknownLoc();
        auto ctx = builder_.getContext();
        auto int32 = builder_.getIntegerType(32, true);

        const auto sparsityPtrStep = 0;
        const auto weightsBuffType = opBuffers_.front().weights.getBuffer().getType().cast<NDTypeInterface>();
        auto weightsOutputChannelsStrideInBits = weightsBuffType.getStrides()[vpux::Dims4D::Filter::OC];
        const auto weightsElemType = weightsBuffType.getElementType();

        const auto alignmentRequirement = 16;
        const auto alignment =
                (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
        if (weightsOutputChannelsStrideInBits.count() / CHAR_BIT < alignment) {
            weightsOutputChannelsStrideInBits = vpux::Bit(alignment * CHAR_BIT);
        }

        for (auto clusterIdx = 0; clusterIdx < static_cast<int>(clusters_.size()); clusterIdx += 2) {
            // Create weights table DDR buffer for each tile
            const auto outChannels = opBuffers_[clusterIdx]
                                             .weights.getBuffer()
                                             .getType()
                                             .cast<NDTypeInterface>()
                                             .getShape()[Dims4D::Filter::OC];
            const auto wtableShape = SmallVector<int64_t>({outChannels, 1, 1, 4});
            const auto wtableDDRType =
                    getMemRefType(VPURT::BufferSection::Constant, wtableShape, int32, DimsOrder::NHWC);

            // Create weights table content for each weights table chunck
            const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
            const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
            const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                    inputType, outputType, 0,
                    static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
                    VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, ppeConverter, biasConverter,
                    outChannels, weightsElemType);
            auto wtableTensorType = mlir::RankedTensorType::get(wtableShape, int32);
            const auto weightsTableValues =
                    mlir::DenseElementsAttr::get(wtableTensorType, llvm::ArrayRef<std::int32_t>(weightsTable));

            auto wtableDDRBuffer = builder_.create<vpux::Const::DeclareOp>(
                    loc, wtableDDRType,
                    vpux::Const::ContentAttr::get(
                            weightsTableValues, Const::ContentSetup(wtableTensorType).reorder(vpux::DimsOrder::NHWC)));

            SmallVector<int64_t> targetClusters;
            targetClusters.push_back(clusterIdx);
            auto targetIdx = clusterIdx + 1;
            if (targetIdx < splitNum_) {
                targetClusters.push_back(targetIdx);
            }

            auto wtableCMXBuffer =
                    createBuffer(ctx, builder_, int32, wtableShape, DimsOrder::NHWC, targetClusters, offset);

            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder_, mlir::ValueRange(),
                                                  mlir::ValueRange(updateBarrier.getBarrier()), loc, wtableDDRBuffer,
                                                  wtableCMXBuffer, 0);

            auto wtableCMXMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, clusters_[clusterIdx], wtableShape,
                                                     int32, DimsOrder::NHWC);
            opBuffers_[clusterIdx].weightsTable = createDeclareTensorOp(
                    builder_, wtableCMXMemRefType, VPURT::BufferSection::CMX_NN, clusters_[clusterIdx], offset);

            if (targetIdx < splitNum_) {
                wtableCMXMemRefType = getMemRefType(VPURT::BufferSection::CMX_NN, clusters_[targetIdx], wtableShape,
                                                    int32, DimsOrder::NHWC);

                opBuffers_[targetIdx].weightsTable = createDeclareTensorOp(
                        builder_, wtableCMXMemRefType, VPURT::BufferSection::CMX_NN, clusters_[targetIdx], offset);
            }
        }
    }

    int64_t splitNum_;
    int64_t heightHaloSz_;
};

// Generate output workload start/end for each cluster from ITIBuffer info
// ITIbuffer contains total shape (what's produced in the cluster plus what is brought from others) and
// inbound halo shape and offset information. The output workload is obtained by subtracting all the
// halo shapes from the full shape.
// Function returns a map with the current cluster's output ITIBuffer as the key and a pair of
// outputWkldStart, outputWkldEnd for same cluster as the value
DenseMap<VPUIP::ITIBufferType, std::pair<mlir::ArrayAttr, mlir::ArrayAttr>> generateOutputStartEnd(
        mlir::OpBuilder& builder, SmallVector<Buffers>& itiBuffersVec, std::unique_ptr<Strategy>& strategy) {
    auto* ctx = builder.getContext();
    DenseMap<VPUIP::ITIBufferType, std::pair<mlir::ArrayAttr, mlir::ArrayAttr>> workloadsMap;

    for (auto& itiBuffers : itiBuffersVec) {
        auto outputType = itiBuffers.output.getBuffer().getType().cast<VPUIP::ITIBufferType>();
        const auto inwardHalos = outputType.getInwardHaloRegions();
        const auto fullShape = outputType.getShape().raw();

        // For each dimension we will be recording the range of the halos: (halo_start_offset, halo_end_offset)
        SmallVector<SmallVector<std::pair<int64_t, int64_t>>> initialDimRanges(fullShape.size() - 1);

        for (const auto& halo : inwardHalos) {
            auto haloShape = parseIntArrayAttr<int64_t>(halo.getShape());
            auto offset = parseIntArrayAttr<int64_t>(halo.getOffset());

            // Dim 0 is always batch, skip it.
            // fullShape has dims in order NCHW, while oduStartEnd uses WHC format;
            // use rangeIdx to convert between the two formats
            for (size_t dim = fullShape.size() - 1, rangeIdx = 0; dim > 0; dim--, rangeIdx++) {
                // Found the dim(s) the tensor is split across
                if (strategy->isHaloDim(fullShape, haloShape, dim)) {
                    initialDimRanges[rangeIdx].push_back(std::make_pair(offset[dim], offset[dim] + haloShape[dim]));
                }
            }
        }

        // Sort the ranges based on the start offset
        for (size_t dim = 0; dim < initialDimRanges.size(); dim++) {
            llvm::stable_sort(initialDimRanges[dim], [](const std::pair<int64_t, int64_t>& leftRange,
                                                        const std::pair<int64_t, int64_t>& rightRange) {
                return leftRange.first <= rightRange.first;
            });
        }

        SmallVector<std::pair<int64_t, int64_t>> oduStartEnd = {{0, fullShape[Dims4D::Act::W.ind()] - 1},
                                                                {0, fullShape[Dims4D::Act::H.ind()] - 1},
                                                                {0, fullShape[Dims4D::Act::C.ind()] - 1}};
        for (size_t dim = 0; dim < initialDimRanges.size(); dim++) {
            if (initialDimRanges[dim].size() == 0) {
                // Tensor is not segmented across dim; workload will have full size
                continue;
            }

            // After sorting we will have the following dim ranges cases:
            // 1. [0, n(0)] [n(0), n(1)] [n(1), n(2)] ... [n(m), n(m + 1)]
            //       => halos total range is [0, n(m + 1)]
            //       => actual workload is the last slice of the tensor along dim
            //       => subtracting this from full tensor we get the actual output workload: [n(m+1),
            //       fullShape[dim]]
            // 2. [0, n(0)] [n(0), n(1)] ... [n(m), n(m + 1)] [n(p), n(p + 1)] [n(p + 1), n(p + 2)] ... [n(r),
            // fullShape[dim]]
            //       => merging the halos we get 2 ranges: [0, n(p + 2)] and [n(p), fullShape[dim]]
            //       => actual workload is a middle slice of the tensor along dim
            //       => subtracting this from full tensor we get the actual output workload: [n(m+1), n(p)]
            // 3. [n(0), n(1)] [n(1), n(2)] ... [n(m), fullShape[dim]]
            //       => halos total range is [n(0), fullShape[dim]]
            //       => actual workload is the first slice of the tensor along dim
            //       => subtracting this from full tensor we get the actual output workload: [0, n(0)]
            SmallVector<std::pair<int64_t, int64_t>> mergedHalos;
            for (size_t idx = 0; idx < initialDimRanges[dim].size(); idx++) {
                auto crtHaloRange = initialDimRanges[dim][idx];

                // Last range could not be merged with the previous ones, push it separately
                if (idx == initialDimRanges[dim].size() - 1) {
                    mergedHalos.push_back(crtHaloRange);
                    break;
                }

                auto nextHaloRange = initialDimRanges[dim][idx + 1];
                auto mergedHaloRange = crtHaloRange;
                // If end offset of current range matched the start offset of the next one or
                // they are identical, merge them together
                // SOHW strategy is an example of when we can have the same range twice across a dim
                while (crtHaloRange.second == nextHaloRange.first || crtHaloRange == nextHaloRange) {
                    mergedHaloRange.second = nextHaloRange.second;
                    idx++;

                    if (idx > initialDimRanges[dim].size() - 2) {
                        break;
                    }

                    crtHaloRange = initialDimRanges[dim][idx];
                    nextHaloRange = initialDimRanges[dim][idx + 1];
                }

                mergedHalos.push_back(mergedHaloRange);
            }

            // We can either have the current workload at the beginning of the tensor,
            // the end or somewhere in the middle More then 2 halos means the workload
            // is made of 2 discontinous sections, which is impossible
            VPUX_THROW_UNLESS(mergedHalos.size() == 1 || mergedHalos.size() == 2,
                              "buildHaloMultiClusteringTest: Cannot compute valid "
                              "output begin/end for variant");

            if (mergedHalos.size() == 1) {
                // mergedHalos[0].first == 0 => start of the halo is at the beginning of the tensor chunck,
                // therefore the current workload starts at the end of the halo range and ends at the end of the
                // full chunck
                oduStartEnd[dim].first = mergedHalos[0].first == 0 ? mergedHalos[0].second : 0;
                oduStartEnd[dim].second =
                        mergedHalos[0].first == 0 ? oduStartEnd[dim].second : mergedHalos[0].first - 1;
            } else if (mergedHalos.size() == 2) {
                oduStartEnd[dim].first =
                        mergedHalos[0].second > mergedHalos[1].second ? mergedHalos[1].second : mergedHalos[0].second;
                oduStartEnd[dim].second = mergedHalos[0].first > mergedHalos[1].first ? mergedHalos[0].first - 1
                                                                                      : mergedHalos[1].first - 1;
            }
        }

        auto startAttr = getIntArrayAttr(
                ctx, SmallVector<int64_t>{oduStartEnd[0].first, oduStartEnd[1].first, oduStartEnd[2].first});
        auto endAttr = getIntArrayAttr(
                ctx, SmallVector<std::int64_t>{oduStartEnd[0].second, oduStartEnd[1].second, oduStartEnd[2].second});

        workloadsMap.insert(std::make_pair(outputType, std::make_pair(startAttr, endAttr)));
    }

    return workloadsMap;
}

// Add Variant to NCEClusterTaskOp; also compute the halo regions based on the output_iti_buffs
VPUIP::DPUTaskOp addDPUTask(
        mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp& nceTask, Buffers& itiBuffers,
        DenseMap<VPUIP::ITIBufferType, std::pair<mlir::ArrayAttr, mlir::ArrayAttr>>& outputWorkloadsMap,
        VPU::PaddingAttr padding, const VPU::MPEMode mpeMode, mlir::IntegerAttr cluster) {
    auto* ctx = builder.getContext();
    auto outputType = itiBuffers.output.getBuffer().getType().cast<VPUIP::ITIBufferType>();
    SmallVector<mlir::Attribute> haloRegions;
    const auto outwardHalos = outputType.getOutwardHaloRegions();
    for (const auto& outwardHalo : outwardHalos) {
        const auto firstInwardHalo = outwardHalo.getInwardHaloRegions().begin()->cast<VPUIP::HaloRegionAttr>();
        auto outputIti = llvm::find_if(itiBuffers.outputIti, [&firstInwardHalo](VPURT::DeclareBufferOp bufferOp) {
            auto itiType = bufferOp.getBuffer().getType().cast<VPUIP::ITIBufferType>();
            auto inwardHaloRegions = itiType.getInwardHaloRegions();
            return llvm::find(inwardHaloRegions, firstInwardHalo) != inwardHaloRegions.end();
        });
        VPUX_THROW_UNLESS(outputIti != itiBuffers.outputIti.end(),
                          "buildHaloMultiClusteringTest: outward halo is not associated with any output iti buffer");

        const auto outwardHaloShape = parseIntArrayAttr<int64_t>(outwardHalo.getShape());
        const auto outwardHaloOffset = parseIntArrayAttr<int64_t>(outwardHalo.getOffset());
        auto itiType = outputIti->getBuffer().getType().cast<VPUIP::ITIBufferType>();

        const auto xStart = outwardHaloOffset[Dims4D::Act::W.ind()];
        const auto xEnd = xStart + outwardHaloShape[Dims4D::Act::W.ind()] - 1;
        const auto yStart = outwardHaloOffset[Dims4D::Act::H.ind()];
        const auto yEnd = yStart + outwardHaloShape[Dims4D::Act::H.ind()] - 1;
        const auto zStart = outwardHaloOffset[Dims4D::Act::C.ind()];
        const auto zEnd = zStart + outwardHaloShape[Dims4D::Act::C.ind()] - 1;

        int64_t targetOffset = outputIti->getByteOffset() - itiBuffers.output.getByteOffset();
        int64_t sparsityOffset = 0;

        const auto numBitsInByte = Byte(1).to<Bit>().count();

        // Need to apply offsets only for height/width
        const auto dstItiOffset = parseIntArrayAttr<int64_t>(firstInwardHalo.getOffset());
        const auto srcHaloHeightStart = outwardHaloOffset[Dims4D::Act::H.ind()];
        const auto dstHaloHeightStart = dstItiOffset[Dims4D::Act::H.ind()];
        const auto srcHaloWidthStart = outwardHaloOffset[Dims4D::Act::W.ind()];
        const auto dstHaloWidthStart = dstItiOffset[Dims4D::Act::W.ind()];
        const int64_t offset = dstHaloHeightStart * itiType.getStrides()[Dims4D::Act::H].count() +
                               dstHaloWidthStart * itiType.getStrides()[Dims4D::Act::W].count() -
                               srcHaloHeightStart * itiType.getStrides()[Dims4D::Act::H].count() -
                               srcHaloWidthStart * itiType.getStrides()[Dims4D::Act::W].count();

        targetOffset += offset / numBitsInByte;
        sparsityOffset += offset / outputType.getElemTypeSize().count() / numBitsInByte;

        SmallVector<int64_t> targetClustersVec;
        for (auto& inHaloAttr : outwardHalo.getInwardHaloRegions()) {
            const auto inwardHalo = inHaloAttr.cast<VPUIP::HaloRegionAttr>();
            targetClustersVec.push_back(inwardHalo.getClusterId().getInt());
        }

        const int64_t targetWidth = outputIti->getBuffer().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::W];

        const auto sparsityOffsetAttr =
                nceTask.getOutputSparsityMapBuff() != nullptr ? builder.getI64IntegerAttr(sparsityOffset) : nullptr;
        const auto dpuHaloRegion = VPUIP::DPUHaloRegionAttr::get(
                ctx, builder.getI64IntegerAttr(xStart), builder.getI64IntegerAttr(xEnd),
                builder.getI64IntegerAttr(yStart), builder.getI64IntegerAttr(yEnd), builder.getI64IntegerAttr(zStart),
                builder.getI64IntegerAttr(zEnd), builder.getI64IntegerAttr(targetOffset),
                getIntArrayAttr(ctx, targetClustersVec), sparsityOffsetAttr, builder.getI64IntegerAttr(targetWidth));
        haloRegions.push_back(dpuHaloRegion);
    }

    auto inputShape = itiBuffers.input.getBuffer().getType().cast<NDTypeInterface>().getShape();
    const auto inStartAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, 0});
    const auto inEndAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{inputShape[Dims4D::Act::W] - 1, inputShape[Dims4D::Act::H] - 1,
                                                      inputShape[Dims4D::Act::C] - 1});

    const auto outStartAttr = outputWorkloadsMap[outputType].first;
    const auto outEndAttr = outputWorkloadsMap[outputType].second;
    auto haloRegionsAttr = builder.getArrayAttr(haloRegions);
    return nceTask.addDPUTask(builder, outStartAttr, outEndAttr, inStartAttr, inEndAttr, padding, mpeMode, cluster,
                              haloRegionsAttr);
}

}  // namespace

void buildHaloMultiClusteringTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                  mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                  mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto outputs = testDesc.getOutputLayers();
    const auto haloParams = testDesc.getHaloParams();
    const auto segmentationType = haloParams.segmentation;
    const SmallVector<std::int64_t> taskClusters{haloParams.taskClusters.begin(), haloParams.taskClusters.end()};
    const SmallVector<std::int64_t> clustersPerDim{haloParams.clustersPerDim.begin(), haloParams.clustersPerDim.end()};
    const auto numClusters = taskClusters.size();
    const auto profilingParams = testDesc.getProfilingParams();
    auto profOutputType = getUInt64Type(ctx);

    const SmallVector<int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const auto kernelStrides = getIntArrayAttr(ctx, conv.stride);
    const std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    const SmallVector<std::int64_t> kernel = {weightsShape[Dims4D::Filter::KY.ind()],
                                              weightsShape[Dims4D::Filter::KX.ind()]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    const SmallVector<int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<int64_t> weightsTableShape{weightsShape[Dims4D::Filter::OC.ind()], 1, 1, 4};
    const SmallVector<int64_t> outputShape{
            inputShape[Dims4D::Act::N.ind()], weightsShape[Dims4D::Filter::OC.ind()],
            getOutputSpatialDim(inputShape[Dims4D::Act::H.ind()], kernel[0], kernelPaddings.getTop().getInt(),
                                kernelPaddings.getBottom().getInt(), conv.stride[0]),
            getOutputSpatialDim(inputShape[Dims4D::Act::W.ind()], kernel[1], kernelPaddings.getLeft().getInt(),
                                kernelPaddings.getRight().getInt(), conv.stride[1])};
    SmallVector<int64_t> profShape{HW_DPU_PROFILING_SIZE_BYTES / sizeof(int64_t)};
    SmallVector<int64_t> profOutputShape{
            static_cast<int64_t>(numClusters * (HW_DPU_PROFILING_SIZE_BYTES / sizeof(int64_t)))};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildHaloMultiClusteringTest: Got empty parentInputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildHaloMultiClusteringTest: Got empty weightsShape");
    VPUX_THROW_UNLESS(!weightsTableShape.empty(), "buildHaloMultiClusteringTest: Got empty weightsTableShape");

    // set runtime resources
    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = numClusters;
    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    const char* weightsFileName = "weights.dat";

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto profOutputParamType =
            getMemRefType(VPURT::BufferSection::ProfilingOutput, profOutputShape, profOutputType, DimsOrder::C);

    auto getReturnTypesVec = [profilingParams, profOutputParamType](
                                     ArrayRef<nb::OutputLayer> outputs, mlir::Type outputType,
                                     const std::size_t numClusters) -> SmallVector<mlir::Type> {
        SmallVector<mlir::Type> returnTypes;
        returnTypes.reserve(numClusters);
        for (const auto& output : outputs) {
            const auto outputParamType = getMemRefType(
                    vpux::VPURT::BufferSection::NetworkOutput,
                    SmallVector<std::int64_t>(output.shape.begin(), output.shape.end()), outputType, DimsOrder::NHWC);
            returnTypes.push_back(outputParamType);
        }

        if (profilingParams.profilingEnabled()) {
            returnTypes.push_back(profOutputParamType);
        }

        return returnTypes;
    };

    const auto returnTypesVec = getReturnTypesVec(outputs, outputType, numClusters);
    auto argTypesVec = SmallVector<mlir::Type>({inputParamType});
    argTypesVec.append(returnTypesVec.begin(), returnTypesVec.end());
    const auto funcType = builder.getFunctionType(argTypesVec, returnTypesVec);

    auto function = builder.create<mlir::func::FuncOp>(
            loc, printToString("halo_multiclustering_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());
    auto functionInput = function.getArgument(0);

    const auto weightsValues = generateWeights(builder, weightsShape, weightsType, ctx, weightsFileName);
    Const::ContentSetup weightsAttributeSetup(weightsValues.getType());
    weightsAttributeSetup = weightsAttributeSetup.reorder(DimsOrder::OYXI);

    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        auto contentType =
                Const::inferFinalTypeAndSplat(weightsValues, weightsAttributeSetup.getTransformations()).first;
        const auto quantizedType = vpux::changeStorageType(qty, contentType.getElementType());
        weightsAttributeSetup = weightsAttributeSetup.castElemType(quantizedType);
        if (qty.getStorageType().isInteger(4)) {
            weightsAttributeSetup = weightsAttributeSetup.bitPack(4);
        }
    }

    std::unique_ptr<Strategy> multiClusterStrategy;
    switch (segmentationType) {
    case nb::SegmentationType::SOH: {
        multiClusterStrategy.reset(
                new SoHorSoWStrategy(functionBuilder, taskClusters, Dims4D::Act::H, haloParams.heightHaloSize));
        break;
    }
    case nb::SegmentationType::SOW: {
        multiClusterStrategy.reset(
                new SoHorSoWStrategy(functionBuilder, taskClusters, Dims4D::Act::W, haloParams.widthHaloSize));
        break;
    }
    case nb::SegmentationType::SOHW: {
        multiClusterStrategy.reset(new SoHWStrategy(functionBuilder, taskClusters, clustersPerDim,
                                                    haloParams.heightHaloSize, haloParams.widthHaloSize));
        break;
    }
    case nb::SegmentationType::SOK: {
        multiClusterStrategy.reset(new SoKStrategy(functionBuilder, taskClusters));
        break;
    }
    case nb::SegmentationType::SOHK: {
        multiClusterStrategy.reset(
                new SoHKStrategy(functionBuilder, taskClusters, clustersPerDim, haloParams.heightHaloSize));
        break;
    }
    case nb::SegmentationType::SOHK3: {
        multiClusterStrategy.reset(
                new SoHK3Strategy(functionBuilder, taskClusters, clustersPerDim[0], haloParams.heightHaloSize));
        break;
    }
    case nb::SegmentationType::SOHW3: {
        multiClusterStrategy.reset(new SoHW3Strategy(functionBuilder, taskClusters, clustersPerDim[0],
                                                     haloParams.heightHaloSize, haloParams.widthHaloSize));
        break;
    }
    default: {
        VPUX_THROW("Segmentation type unsupported {0}", nb::to_string(segmentationType));
    }
    };

    std::size_t offsetCMX = 0;
    const auto alignment = Byte(16);

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(functionBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);

    // Weights & Weights table segmentation & DMAs
    multiClusterStrategy->handleConstants(Const::ContentAttr::get(weightsValues, std::move(weightsAttributeSetup)),
                                          testDesc.getArchitecture(), inputType, outputType, offsetCMX, updateBarrier,
                                          waitWLMBarrier);

    // Create output and output_iti buffs for each Conv
    multiClusterStrategy->createOutputItiBuffers(returnTypesVec, offsetCMX);
    offsetCMX = vpux::alignValUp(offsetCMX, static_cast<std::size_t>(alignment.count()));

    // Create input buffs for each Conv
    multiClusterStrategy->createInputBuffers(functionInput, outputShape, weightsShape, kernelStrides, kernelPaddings,
                                             offsetCMX, updateBarrier, clustersPerDim);

    if (profilingParams.dpuProfilingEnabled) {
        // Create profilingOutput buffer
        offsetCMX = vpux::alignValUp(offsetCMX, static_cast<std::size_t>(Byte(32).count()));
        multiClusterStrategy->createProfilingOutputBuffers(profShape, profOutputType, offsetCMX);
    }
    auto waitBarrier = updateBarrier;

    // Create NCEClusterTaskOp
    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, freeBarrierId++);

    constexpr int64_t WORKLOAD_CHANNEL_IDX = 2;
    auto itiBuffs = multiClusterStrategy->getBuffers();
    auto outputWorkloadsMap = generateOutputStartEnd(builder, itiBuffs, multiClusterStrategy);
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        auto& itiBuff = itiBuffs[idx];
        auto outputItiBuffs = SmallVector<mlir::Value>();
        for (auto& itiOp : itiBuff.outputIti) {
            outputItiBuffs.push_back(itiOp.getBuffer());
        }

        auto outType = itiBuff.output.getBuffer().getType();
        auto profOutType =
                profilingParams.dpuProfilingEnabled ? itiBuff.profilingOutputCMX.getBuffer().getType() : nullptr;
        const auto workloadStart =
                parseIntArrayAttr<int64_t>(outputWorkloadsMap[outType.cast<VPUIP::ITIBufferType>()].first);
        const auto outChannelOffset = functionBuilder.getI64IntegerAttr(workloadStart[WORKLOAD_CHANNEL_IDX]);

        const auto taskName = printToString("haloConv?t_Conv/cluster_{0}", idx);
        auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                mlir::ValueRange(updateBarrier.getBarrier()), mlir::NameLoc::get(mlir::StringAttr::get(ctx, taskName)),
                outType,
                /*output_sparsity_map=*/nullptr, profilingParams.dpuProfilingEnabled ? profOutType : nullptr,
                itiBuff.input.getBuffer(),
                /*input_sparsity_map=*/nullptr, /*input_storage_element_table=*/nullptr, itiBuff.weights.getBuffer(),
                /*weights_sparsity_map=*/nullptr, itiBuff.weightsTable.getBuffer(),
                /*spr_lookup_table*/ nullptr, itiBuff.input.getBuffer(), /*parent_input_sparsity_map=*/nullptr,
                /*parent_input_storage_element_table=*/nullptr, itiBuff.output.getBuffer(),
                /*parent_output_sparsity_map=*/nullptr, mlir::ValueRange(outputItiBuffs), itiBuff.output.getBuffer(),
                /*output_sparsity_map_buff=*/nullptr,
                profilingParams.dpuProfilingEnabled ? itiBuff.profilingOutputCMX.getBuffer() : nullptr,
                /*max_per_xy=*/nullptr,
                /*min_per_xy=*/nullptr,
                /*min_max_per_tensor=*/mlir::ValueRange(), vpux::VPUIP::NCETaskType::CONV, kernelSize, kernelStrides,
                kernelPaddings,
                /*is_continued=*/nullptr,
                /*cm_sp_pattern=*/nullptr, /*is_segmented=*/nullptr, outChannelOffset,
                /*input_channels_compression=*/nullptr);

        if (profilingParams.dpuProfilingEnabled) {
            auto profAttr = VPUIP::DpuProfilingMetadataAttr::get(
                    ctx, /*bufferId*/ getIntAttr(ctx, 0),
                    /*taskId*/ getIntAttr(ctx, 1), /*maxVariants*/ getIntAttr(ctx, 1),
                    /*numVariants*/ getIntAttr(ctx, 1), /*clusterId*/ getIntAttr(ctx, idx));
            nceTask.setProfilingMetadataAttr(profAttr);
        }
        const auto workloadPadding =
                getMulticlusteringPaddings(ctx, idx, numClusters, segmentationType, kernelPaddings, clustersPerDim);

        auto dpuTask = addDPUTask(functionBuilder, nceTask, itiBuff, outputWorkloadsMap, workloadPadding,
                                  conv.cube_mode, functionBuilder.getI64IntegerAttr(taskClusters[idx]));
        if (profilingParams.dpuProfilingEnabled) {
            dpuTask.setWorkloadIdAttr(vpux::getIntAttr(ctx, offsetCMX / HW_DPU_PROFILING_SIZE_BYTES));
        }
    }

    waitBarrier = updateBarrier;

    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            loc, freeBarrierId++, testDesc.getWLMParams().isWLMPartialEnabled);
    // Create CMX2DDR DMAs to move outputs from each cluster to DDR
    SmallVector<mlir::Value> functionOutputs;
    for (unsigned int idx = 0; idx < static_cast<unsigned int>(numClusters); idx++) {
        auto functionOutput = function.getArgument(1 + idx);
        functionOutputs.push_back(functionOutput);
        auto outputItiBufferOp = itiBuffs[idx].output;
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                              outputItiBufferOp.getBuffer(), functionOutput, 0);

        // move DPU profiling results for cluster [idx]
        if (profilingParams.dpuProfilingEnabled) {
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.getBarrier()),
                                                  mlir::ValueRange(finalBarrier.getBarrier()), loc,
                                                  itiBuffs[idx].profilingOutputCMX.getBuffer(),
                                                  itiBuffs[idx].profilingOutputDDR.getBuffer(), 0);
        }
    }

    if (profilingParams.profilingEnabled()) {
        auto funcProfOutput = function.getArgument(static_cast<unsigned int>(1 + numClusters));
        functionOutputs.push_back(funcProfOutput);
    }

    functionBuilder.create<mlir::func::ReturnOp>(loc, functionOutputs);

    mlir::PassManager pmBuilderEnd(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    if (conv.compress) {
        pmBuilderEnd.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }
    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderEnd.run(module)), "Compilation failed");

    SmallVector<mlir::Type> outputTensorTypesVec;
    for (std::size_t idx = 0; idx < numClusters; idx++) {
        const auto outShape = returnTypesVec[idx].cast<NDTypeInterface>().getShape();
        auto outputTensorType = getTensorType(outShape, outputType, vpux::DimsOrder::NHWC, nullptr);
        outputTensorTypesVec.push_back(outputTensorType);
    }

    mlir::SmallVector<ProfilingDataSection> profilingDataSections;
    if (profilingParams.profilingEnabled()) {
        size_t offset = 0;
        if (profilingParams.dpuProfilingEnabled) {
            const auto dpuProfOutputType =
                    getTensorType(ShapeRef(profOutputShape), profOutputType, DimsOrder::C, nullptr);
            const auto sectionType = dpuProfOutputType.cast<vpux::NDTypeInterface>();
            const auto sectionSize = sectionType.getTotalAllocSize().count();

            profilingDataSections.push_back({HWP_DPU_SECTION_EXEC_TYPE, offset, sectionSize});
            offset += sectionSize;
        }
    }
    buildCNNOp(builder, function.getName(), {getTensorType(ShapeRef(inputShape), inputType, DimsOrder::NHWC, nullptr)},
               outputTensorTypesVec, profilingDataSections);
}  // namespace hwtest

}  // namespace hwtest
}  // namespace vpux
