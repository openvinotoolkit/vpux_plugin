//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/kernel_params_utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/utils/profiling/metadata.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/Support/FileSystem.h>

#include <iostream>
#include <vector>

using namespace vpux;

constexpr auto ACT_RT_CODE_BUFFER_SIZE = (1_MB).to<vpux::Byte>().count();

namespace {

using KernelTextAndEntry = std::pair<VPUMI40XX::DeclareKernelTextOp, VPUMI40XX::DeclareKernelEntryOp>;
using FindKernelTextEntryFuncType = llvm::function_ref<KernelTextAndEntry(mlir::OpBuilder, VPUIP::SwKernelOp,
                                                                          VPURegMapped::IndexType, mlir::StringAttr)>;

//
// ConvertVPUIP2VPUMI40XXPass
//

class ConvertVPUIP2VPUMI40XXPass final : public ConvertVPUIP2VPUMI40XXBase<ConvertVPUIP2VPUMI40XXPass> {
public:
    explicit ConvertVPUIP2VPUMI40XXPass(Logger log, bool enableMemorySideCache)
            : _enableMemorySideCacheOption(enableMemorySideCache) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    bool _enableMemorySideCacheOption;
    enum class DmaNnSrcType { DDR, CMX_NN, Count };

    const std::map<VPU::MemoryKind, DmaNnSrcType> memKind2DmaSrc = {{VPU::MemoryKind::DDR, DmaNnSrcType::DDR},
                                                                    {VPU::MemoryKind::CMX_NN, DmaNnSrcType::CMX_NN}};

    void safeRunOnModule() final;

    mlir::Value convertITIBuffer(mlir::OpBuilder builder, mlir::Value buffer) {
        auto itiBufferType = buffer.getType().dyn_cast<VPUIP::ITIBufferType>();

        if (!itiBufferType) {
            return {buffer};
        }

        auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(buffer.getDefiningOp());

        auto byteOffset = definingOp.getByteOffset();
        auto swizzlingKey = definingOp.getSwizzlingKey();
        auto buffSec = definingOp.getMemorySpace();

        VPURT::DeclareBufferOp res;

        auto memSpace = itiBufferType.getMemSpace();
        auto tileIdx = memSpace.getIndex().value_or(0);

        auto memType = mlir::MemRefType::get(itiBufferType.getShape().raw(), itiBufferType.getElementType(),
                                             itiBufferType.getLayout(), memSpace);
        if (swizzlingKey.has_value()) {
            res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIdx, byteOffset,
                                                         swizzlingKey.value());
        } else {
            res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIdx, byteOffset);
        }

        return res.getResult();
    }

    llvm::SmallVector<mlir::Value> unrollDistributedBuff(mlir::OpBuilder builder, mlir::Value output) {
        if (!output) {
            return {};
        }

        if (output.getType().isa<VPUIP::ITIBufferType>()) {
            return {convertITIBuffer(builder, output)};
        }

        auto distributedOutput = output.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (!distributedOutput) {
            return {output};
        }

        llvm::SmallVector<mlir::Value> results;
        auto distribution = distributedOutput.getDistribution();
        auto outputMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(distribution.getMode().getValue());
        auto duplicatedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::DUPLICATED);
        auto multicastedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::MULTICASTED);
        if ((outputMode & duplicatedMode) || (outputMode & multicastedMode)) {
            auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());

            auto compactType = distributedOutput.getCompactType();

            auto totalClusters = static_cast<size_t>(distribution.getNumClusters().getInt());

            auto byteOffset = definingOp.getByteOffset();
            auto swizzlingKey = definingOp.getSwizzlingKey();
            auto buffSec = definingOp.getMemorySpace();

            VPUX_THROW_WHEN(!definingOp.getSectionIndex().has_value(), "Distributed buffer without section index: {0}",
                            definingOp);

            auto clusters = parseIntArrayAttr<int64_t>(definingOp.getSectionIndex().value());

            VPUX_THROW_WHEN(
                    clusters.size() != totalClusters,
                    "Size of distributed buffer section index ({0}) different than distribution num_clusters ({1})",
                    clusters.size(), totalClusters);

            for (size_t clusterIdx = 0; clusterIdx < totalClusters; ++clusterIdx) {
                VPURT::DeclareBufferOp res;

                auto currMemLocation = compactType.getMemorySpace().cast<IndexedSymbolAttr>().getLeafNameAttr();
                auto newMemSpace =
                        vpux::IndexedSymbolAttr::get(currMemLocation, static_cast<size_t>(clusters[clusterIdx]));
                auto memType = mlir::MemRefType::get(compactType.getShape(), compactType.getElementType(),
                                                     compactType.getLayout(), newMemSpace);
                if (swizzlingKey.has_value()) {
                    res = builder.create<VPURT::DeclareBufferOp>(
                            output.getLoc(), memType, buffSec, clusters[clusterIdx], byteOffset, swizzlingKey.value());
                } else {
                    res = builder.create<VPURT::DeclareBufferOp>(output.getLoc(), memType, buffSec,
                                                                 clusters[clusterIdx], byteOffset);
                }

                results.push_back(res.getResult());
            }
        } else {
            VPUX_THROW("Only distributed buffer with DUPLICATE is accepted as direct output of OP");
        }

        return results;
    }

    mlir::Value extractFromDistributedBuff(mlir::OpBuilder builder, mlir::Value buffer, uint32_t tileIdx) {
        if (!buffer) {
            return nullptr;
        }

        if (buffer.getType().isa<VPUIP::ITIBufferType>()) {
            return {convertITIBuffer(builder, buffer)};
        }

        auto distributedOutput = buffer.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (!distributedOutput) {
            return {buffer};
        }

        mlir::Value value;
        auto distribution = distributedOutput.getDistribution();
        auto outputMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(distribution.getMode().getValue());
        auto duplicatedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::DUPLICATED);
        auto multicastedMode =
                static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::MULTICASTED);
        if ((outputMode & duplicatedMode) || (outputMode & multicastedMode)) {
            auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(buffer.getDefiningOp());

            auto compactType = distributedOutput.getCompactType();

            auto totalClusters = static_cast<size_t>(distribution.getNumClusters().getInt());

            auto byteOffset = definingOp.getByteOffset();
            auto swizzlingKey = definingOp.getSwizzlingKey();
            auto buffSec = definingOp.getMemorySpace();

            VPUX_THROW_WHEN(!definingOp.getSectionIndex().has_value(), "Distributed buffer without section index: {0}",
                            definingOp);

            auto clusters = parseIntArrayAttr<int64_t>(definingOp.getSectionIndex().value());

            VPUX_THROW_WHEN(
                    clusters.size() != totalClusters,
                    "Size of distributed buffer section index ({0}) different than distribution num_clusters ({1})",
                    clusters.size(), totalClusters);

            VPURT::DeclareBufferOp res;

            auto clusters_tileIdx = std::find(clusters.begin(), clusters.end(), tileIdx);

            VPUX_THROW_WHEN(clusters_tileIdx == clusters.end(),
                            "Tile index '{0}' not found in distributed buffer section index array", tileIdx);

            auto currMemLocation = compactType.getMemorySpace().cast<IndexedSymbolAttr>().getLeafNameAttr();
            auto newMemSpace = vpux::IndexedSymbolAttr::get(currMemLocation, static_cast<size_t>(tileIdx));
            auto memType = mlir::MemRefType::get(compactType.getShape(), compactType.getElementType(),
                                                 compactType.getLayout(), newMemSpace);
            if (swizzlingKey.has_value()) {
                res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIdx, byteOffset,
                                                             swizzlingKey.value());
            } else {
                res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIdx, byteOffset);
            }

            value = res.getResult();
        } else {
            VPUX_THROW("Only distributed buffer with DUPLICATE is accepted as direct output of OP");
        }

        return value;
    }

    template <typename DMAType, typename CreatorFunc>
    void lowerDMA(CreatorFunc&& creator, VPURT::TaskOp taskOp,
                  mlir::SmallVector<mlir::SmallVector<mlir::Value>>& previousDMA,
                  mlir::SmallVector<mlir::SmallVector<uint32_t>>& dmaCount, bool& found) {
        for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<DMAType>())) {
            found = true;
            mlir::OpBuilder builderBlk(taskOp);

            const auto port = op.getPort();
            VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
            const auto tileIdx = port.value();

            const auto dmaNnSrcTypeCount = static_cast<size_t>(DmaNnSrcType::Count);

            auto listIdx = static_cast<uint32_t>(0);
            if (dmaNnSrcTypeCount > 1) {
                auto type = op.getInput().getType().template dyn_cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(type, "Value '{0}' has non vpux::NDTypeInterface '{1}'", op.getInput(),
                                  op.getInput().getType());
                auto memKind = type.getMemoryKind();
                VPUX_THROW_UNLESS(memKind2DmaSrc.find(memKind) != memKind2DmaSrc.end(),
                                  "Tensor arg '{0}' should be of DDR or CMX_NN memkind, but '{1}'", op.getInput(),
                                  memKind);
                listIdx = static_cast<uint32_t>(memKind2DmaSrc.at(memKind));
            }

            auto indexType = VPURegMapped::IndexType::get(taskOp.getContext(), tileIdx, listIdx,
                                                          /*value*/ dmaCount[tileIdx][listIdx]);

            auto waitBarriers = taskOp.getWaitBarriers();
            auto updateBarriers = taskOp.getUpdateBarriers();

            auto trivialIndexType = VPURegMapped::IndexType::get(taskOp.getContext(), 0);

            for (auto val : waitBarriers) {
                val.setType(trivialIndexType);
            }

            for (auto val : updateBarriers) {
                val.setType(trivialIndexType);
            }

            previousDMA[tileIdx][listIdx] =
                    creator(op, previousDMA[tileIdx][listIdx], indexType, waitBarriers, updateBarriers)->getResult(0);

            ++dmaCount[tileIdx][listIdx];
        }
    }

    template <typename DMAType>
    bool enableMemorySideCache(DMAType op) {
        auto inputType = op.getInput().getType().template dyn_cast<vpux::NDTypeInterface>();
        auto outputType = op.getOutput().getType().template dyn_cast<vpux::NDTypeInterface>();
        VPUX_THROW_UNLESS(inputType, "Value '{0}' has non vpux::NDTypeInterface input '{1}'", op.getInput(),
                          op.getInput().getType());
        VPUX_THROW_UNLESS(outputType, "Value '{0}' has non vpux::NDTypeInterface output '{1}'", op.getOutput(),
                          op.getOutput().getType());

        auto inputMemKind = inputType.getMemoryKind();
        auto outputMemKind = outputType.getMemoryKind();
        auto isDDR2CMX = inputMemKind == VPU::MemoryKind::CMX_NN && outputMemKind == VPU::MemoryKind::DDR;
        return _enableMemorySideCacheOption && isDDR2CMX;
    }

    void replaceVPURTTaskOpWithNNDMAOp(mlir::MLIRContext*, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                       Logger& _log) {
        _log.info("VPUIP_VPUMI40XX pass: replaceVPURTTaskOpWithNNDMAOp()");

        const auto tileCount = static_cast<size_t>(IE::getTileExecutor(moduleOp).getCount());
        const auto dmaNnSrcTypeCount = static_cast<size_t>(DmaNnSrcType::Count);

        mlir::SmallVector<mlir::SmallVector<mlir::Value>> previousDMA(
                tileCount, mlir::SmallVector<mlir::Value>(dmaNnSrcTypeCount));
        mlir::SmallVector<mlir::SmallVector<uint32_t>> dmaCount(tileCount,
                                                                mlir::SmallVector<uint32_t>(dmaNnSrcTypeCount, 0));

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getBody().getOps<VPURT::TaskOp>())) {
            bool found = false;
            mlir::OpBuilder builderBlk(taskOp);

            lowerDMA<VPUIP::NNDMAOp>(
                    [&builderBlk, this](VPUIP::NNDMAOp dmaOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, dmaOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr,
                                convertITIBuffer(builderBlk, dmaOp.getInput()), dmaResults, previousDMA,
                                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                dmaOp.getIsOutOfOrder(), dmaOp.getIsCritical(), enableMemorySideCache(dmaOp),
                                dmaOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, nullptr, nullptr,
                                dmaOp.getDmaHwpIdAttr(), dmaOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ false, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::PermuteDMAOp>(
                    [&builderBlk, this](VPUIP::PermuteDMAOp permuteDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto dataShape = getShape(permuteDMAOp.getInput());
                        VPUX_THROW_UNLESS(dataShape.size() == 2 || dataShape.size() == 3,
                                          "DMA op shape size should be 2 or 3. but got shape {0}", dataShape);

                        const auto dmaDescriptor = permuteDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          permuteDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, permuteDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, permuteDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                permuteDMAOp.getIsOutOfOrder(), permuteDMAOp.getIsCritical(),
                                enableMemorySideCache(permuteDMAOp), permuteDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, dmaDescriptorValue, permuteDMAOp.getDmaHwpIdAttr(),
                                permuteDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::ExpandDMAOp>(
                    [&builderBlk, this](VPUIP::ExpandDMAOp expandDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, expandDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, expandDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                expandDMAOp.getIsOutOfOrder(), expandDMAOp.getIsCritical(),
                                enableMemorySideCache(expandDMAOp), expandDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, expandDMAOp.getDmaDescriptor().value(),
                                expandDMAOp.getDmaHwpIdAttr(), expandDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::ConvertDMAOp>(
                    [&builderBlk, this](VPUIP::ConvertDMAOp convertDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, convertDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, convertDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                convertDMAOp.getIsOutOfOrder(), convertDMAOp.getIsCritical(),
                                enableMemorySideCache(convertDMAOp), convertDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, nullptr, convertDMAOp.getDmaHwpIdAttr(),
                                convertDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ false, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::SpaceToDepthDMAOp>(
                    [&builderBlk, this](VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, spaceToDepthDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, spaceToDepthDMAOp.getInput(),
                                dmaResults, previousDMA, mlir::ValueRange(waitBarriers),
                                mlir::ValueRange(updateBarriers), 0, 0, spaceToDepthDMAOp.getIsOutOfOrder(),
                                spaceToDepthDMAOp.getIsCritical(), enableMemorySideCache(spaceToDepthDMAOp),
                                spaceToDepthDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, nullptr,
                                spaceToDepthDMAOp.getDmaDescriptor().value(), spaceToDepthDMAOp.getDmaHwpIdAttr(),
                                spaceToDepthDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ false, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::DepthToSpaceDMAOp>(
                    [&builderBlk, this](VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto inOrder = DimsOrder::fromValue(depthToSpaceDMAOp.getInput());
                        const auto outOrder = DimsOrder::fromValue(depthToSpaceDMAOp.getOutputBuff());
                        auto isLegalType = (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC);
                        VPUX_THROW_UNLESS(isLegalType, "DepthToSpaceDMAOp just support NHWC (NCHW TODO), but got {0}.",
                                          inOrder);

                        const auto dmaDescriptor = depthToSpaceDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          depthToSpaceDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, depthToSpaceDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, depthToSpaceDMAOp.getInput(),
                                dmaResults, previousDMA, mlir::ValueRange(waitBarriers),
                                mlir::ValueRange(updateBarriers), 0, 0, depthToSpaceDMAOp.getIsOutOfOrder(),
                                depthToSpaceDMAOp.getIsCritical(), enableMemorySideCache(depthToSpaceDMAOp),
                                depthToSpaceDMAOp.getPort().value(), VPUIP::DMAAccMode::DISABLE, nullptr,
                                dmaDescriptorValue, depthToSpaceDMAOp.getDmaHwpIdAttr(),
                                depthToSpaceDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ false, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::UpsamplingDMAOp>(
                    [&builderBlk, this](VPUIP::UpsamplingDMAOp upsamplingDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto dmaDescriptor = upsamplingDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          upsamplingDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, upsamplingDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, upsamplingDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                upsamplingDMAOp.getIsOutOfOrder(), upsamplingDMAOp.getIsCritical(),
                                enableMemorySideCache(upsamplingDMAOp), upsamplingDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, dmaDescriptorValue,
                                upsamplingDMAOp.getDmaHwpIdAttr(), upsamplingDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::PerAxisTileDMAOp>(
                    [&builderBlk, this](VPUIP::PerAxisTileDMAOp perAxisTileDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        const auto dmaDescriptor = perAxisTileDMAOp.getDmaDescriptor();
                        VPUX_THROW_UNLESS(dmaDescriptor.has_value(), "DMA descriptor attr not found at '{0}'",
                                          perAxisTileDMAOp->getLoc());
                        const auto dmaDescriptorValue = dmaDescriptor.value();

                        const auto numPlanes = checked_cast<uint32_t>(dmaDescriptorValue.getNumPlanes().getInt());
                        VPUX_THROW_UNLESS(numPlanes <= VPUIP::DMA_MAX_NUMBER_PLANES,
                                          "NUM PLANES should be less than or equal to {0}, but got {1}.",
                                          VPUIP::DMA_MAX_NUMBER_PLANES, numPlanes);

                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, perAxisTileDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, perAxisTileDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                perAxisTileDMAOp.getIsOutOfOrder(), perAxisTileDMAOp.getIsCritical(),
                                enableMemorySideCache(perAxisTileDMAOp), perAxisTileDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, dmaDescriptorValue,
                                perAxisTileDMAOp.getDmaHwpIdAttr(), perAxisTileDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::DecompressDMAOp>(
                    [&builderBlk, this](VPUIP::DecompressDMAOp decompressDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, decompressDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, decompressDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                decompressDMAOp.getIsOutOfOrder(), decompressDMAOp.getIsCritical(),
                                enableMemorySideCache(decompressDMAOp), decompressDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DECOMPRESSION, decompressDMAOp.getActCompressionSizeEntry(), nullptr,
                                decompressDMAOp.getDmaHwpIdAttr(), decompressDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::CompressDMAOp>(
                    [&builderBlk, this](VPUIP::CompressDMAOp compressDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, compressDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, compressDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                compressDMAOp.getIsOutOfOrder(), compressDMAOp.getIsCritical(),
                                enableMemorySideCache(compressDMAOp), compressDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::COMPRESSION, compressDMAOp.getActCompressionSizeEntry(), nullptr,
                                compressDMAOp.getDmaHwpIdAttr(), compressDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::GatherDMAOp>(
                    [&builderBlk, this](VPUIP::GatherDMAOp gatherDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, gatherDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, gatherDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                gatherDMAOp.getIsOutOfOrder(), gatherDMAOp.getIsCritical(),
                                enableMemorySideCache(gatherDMAOp), gatherDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, nullptr, gatherDMAOp.getDmaHwpIdAttr(),
                                gatherDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ true, gatherDMAOp.getIndices());
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
                continue;
            }

            lowerDMA<VPUIP::SyncDMAOp>(
                    [&builderBlk, this](VPUIP::SyncDMAOp syncDMAOp, mlir::Value previousDMA,
                                        VPURegMapped::IndexType indexType, mlir::ValueRange waitBarriers,
                                        mlir::ValueRange updateBarriers) {
                        // create dma descriptor
                        auto zeroAttr = vpux::getIntAttr(&getContext(), 0);
                        auto dmaDescriptorAttr =
                                VPUIP::DMADescriptorAttr::get(&getContext(), /*numPlane*/ zeroAttr, /*len*/ zeroAttr,
                                                              /*srcWidth*/ zeroAttr, /*srcStride*/ zeroAttr,
                                                              /*srcPlaneStride*/ zeroAttr, /*dstWidth*/ zeroAttr,
                                                              /*dstStride*/ zeroAttr, /*dstPlaneStride*/
                                                              zeroAttr);

                        llvm::SmallVector<mlir::Value> dmaResults =
                                unrollDistributedBuff(builderBlk, syncDMAOp.getOutputBuff());
                        return builderBlk.create<VPUMI40XX::NNDMAOp>(
                                builderBlk.getUnknownLoc(), indexType, nullptr, syncDMAOp.getInput(), dmaResults,
                                previousDMA, mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0,
                                syncDMAOp.getIsOutOfOrder(), syncDMAOp.getIsCritical(),
                                enableMemorySideCache(syncDMAOp), syncDMAOp.getPort().value(),
                                VPUIP::DMAAccMode::DISABLE, nullptr, dmaDescriptorAttr, syncDMAOp.getDmaHwpIdAttr(),
                                syncDMAOp.getProfilingMetadataAttr(),
                                /*allow_different_in_out_shapes*/ false, nullptr);
                    },
                    taskOp, previousDMA, dmaCount, found);

            if (found) {
                taskOp->erase();
            }
        }

        _log.info("VPUIP_VPUMI40XX pass: replaceVPURTTaskOpWithNNDMAOp() -- end");
    }

    std::pair<mlir::Value, mlir::Value> createComputeOpSwKernel(mlir::MLIRContext* ctx, VPUIP::SwKernelOp op,
                                                                mlir::OpBuilder builderBlk,
                                                                mlir::func::FuncOp kernel_info_funcOp,
                                                                mlir::Operation::operand_range wait_bars,
                                                                mlir::Operation::operand_range update_bars,
                                                                VPURegMapped::IndexType indexType,
                                                                FindKernelTextEntryFuncType getKernelTextEntryMapFunc,
                                                                mlir::Value previousInvo, mlir::Value previousRanges) {
        auto kernel_elf =
                std::string(kernel_info_funcOp->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry").getValue());

        auto paramsVector = vpux::VPUMI40XX::KernelParamsSerializer::createKernelParams(op);

        auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        auto paramsSize = static_cast<long int>(paramsVector.size());

        auto [kernelTextOp, kernelEntryOp] =
                getKernelTextEntryMapFunc(builderBlk, op, indexType, mlir::StringAttr::get(ctx, kernel_elf));

        auto kernelArgsOp = builderBlk.create<VPUMI40XX::DeclareKernelArgsOp>(builderBlk.getUnknownLoc(), indexType,
                                                                              mlir::StringAttr::get(ctx, kernel_elf));

        auto kernelRangeOp = builderBlk.create<VPUMI40XX::ActKernelRangeOp>(
                builderBlk.getUnknownLoc(), indexType, nullptr, previousRanges, kernelTextOp, kernelArgsOp,
                kernelEntryOp,
                mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE)));

        auto tileIndex = op.getTileIndex().value_or(0);

        auto kernelParamsOp = builderBlk.create<VPUMI40XX::KernelParamsOp>(
                op->getLoc(), indexType, op.getInputs(), op.getOutputBuffs(), mlir::StringAttr::get(ctx, kernel_elf),
                mlir::DenseIntElementsAttr::get(mlir::VectorType::get({paramsSize}, uint8Type), paramsVector));

        auto kernelInvocationOp = builderBlk.create<VPUMI40XX::ActKernelInvocationOp>(
                op->getLoc(), indexType, nullptr, previousInvo, mlir::ValueRange(wait_bars),
                mlir::ValueRange(update_bars), kernelRangeOp.getResult(), kernelParamsOp.getResult(),
                op.getProfilingData(),
                /* tile= */ tileIndex,
                /* start_after= */ 0, /* clean_after= */ 0, op.getProfilingMetadataAttr());

        return std::pair<mlir::Value, mlir::Value>(kernelRangeOp.getResult(), kernelInvocationOp.getResult());
    }

    std::pair<mlir::Value, mlir::Value> createCacheOpSwKernel(
            mlir::MLIRContext* ctx, VPUIP::SwKernelOp op, mlir::OpBuilder builderBlk,
            mlir::SymbolRefAttr kernelTaskType, mlir::Operation::operand_range wait_bars,
            mlir::Operation::operand_range update_bars, VPURegMapped::IndexType indexType, mlir::Value previousInvo,
            mlir::Value previousRanges, FindKernelTextEntryFuncType getKernelTextEntryMapFunc) {
        auto taskTypeVal = VPU::symbolizeActShaveTaskType(kernelTaskType.getLeafReference().strref());
        VPUX_THROW_UNLESS(taskTypeVal.has_value(), "Operation '{0}' has invalid VPU.task_type attribute '{1}'",
                          op.getKernelFunction(), kernelTaskType.getLeafReference());

        auto kernel_type = std::string("cache_op");
        switch (taskTypeVal.value()) {
        case VPU::ActShaveTaskType::CACHE_FLUSH:
            kernel_type.append("_flush");
            break;
        case VPU::ActShaveTaskType::CACHE_INVALIDATE:
            kernel_type.append("_invalidate");
            break;
        case VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE:
            kernel_type.append("_flush_invalidate");
            break;
        case VPU::ActShaveTaskType::CACHE_PREFETCH:
            kernel_type.append("_prefetch");
            break;
        default:
            VPUX_THROW("Unrecognized Kernel Task Type '{0}'", kernelTaskType.getLeafReference());
        }

        mlir::Value maybeKernelTextOpRet = nullptr;
        mlir::Value maybeKernelEntryOpRet = nullptr;
        mlir::Value maybeKernelArgsOpRet = nullptr;
        if (taskTypeVal.value() == VPU::ActShaveTaskType::CACHE_PREFETCH) {
            const auto kernelElfName = op->getAttr("kernelElfName").cast<mlir::StringAttr>();
            const auto [maybeKernelTextOp, maybeKernelEntryOp] =
                    getKernelTextEntryMapFunc(builderBlk, op, indexType, kernelElfName);
            maybeKernelTextOpRet = maybeKernelTextOp->getResult(0);
            maybeKernelEntryOpRet = maybeKernelEntryOp->getResult(0);
            maybeKernelArgsOpRet =
                    builderBlk.create<VPUMI40XX::DeclareKernelArgsOp>(op->getLoc(), indexType, kernelElfName)
                            ->getResult(0);
        }

        auto kernelRangeOp = builderBlk.create<VPUMI40XX::ActKernelRangeOp>(
                op->getLoc(), indexType, nullptr, previousRanges, maybeKernelTextOpRet, maybeKernelArgsOpRet,
                maybeKernelEntryOpRet,
                mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(taskTypeVal.value())));

        auto tileIndex = op.getTileIndex().value_or(0);

        auto uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
        SmallVector<uint8_t> paramsVectorDummy = {0xFF};
        auto paramsSize = static_cast<long int>(paramsVectorDummy.size());

        auto kernelParamsOp = builderBlk.create<VPUMI40XX::KernelParamsOp>(
                op->getLoc(), indexType, mlir::ValueRange(), mlir::ValueRange(),
                mlir::StringAttr::get(ctx, kernel_type),
                mlir::DenseIntElementsAttr::get(mlir::VectorType::get({paramsSize}, uint8Type), paramsVectorDummy));

        auto kernelInvocationOp = builderBlk.create<VPUMI40XX::ActKernelInvocationOp>(
                op->getLoc(), indexType, /*taskLocation*/ nullptr, previousInvo, mlir::ValueRange(wait_bars),
                mlir::ValueRange(update_bars), kernelRangeOp.getResult(), kernelParamsOp.getResult(),
                /* profiling_data */ nullptr,
                /* tile= */ tileIndex,
                /* start_after= */ 0, /* clean_after= */ 0, nullptr);

        return std::pair<mlir::Value, mlir::Value>(kernelRangeOp.getResult(), kernelInvocationOp.getResult());
    }

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    void replaceVPURTTaskOpWithKernelOps(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                         Logger& _log) {
        _log.info("VPUIP_VPUMI40XX pass: replaceVPURTTaskOpWithKernelOps()");

        const auto tileCount = static_cast<size_t>(IE::getTileExecutor(moduleOp).getCount());
        mlir::SmallVector<uint32_t> shaveTaskCount(tileCount, 0);

        llvm::DenseMap<mlir::StringAttr, KernelTextAndEntry> kernelTextEntryMap;

        const auto getKernelTextEntryMapFunc = [&kernelTextEntryMap](mlir::OpBuilder builderBlk,
                                                                     vpux::VPUIP::SwKernelOp op,
                                                                     VPURegMapped::IndexType indexType,
                                                                     mlir::StringAttr kernelElf) -> KernelTextAndEntry {
            if (kernelTextEntryMap.find(kernelElf) == kernelTextEntryMap.end()) {
                auto kernelTextOp =
                        builderBlk.create<VPUMI40XX::DeclareKernelTextOp>(op->getLoc(), indexType, kernelElf);
                auto kernelEntryOp =
                        builderBlk.create<VPUMI40XX::DeclareKernelEntryOp>(op->getLoc(), indexType, kernelElf);
                kernelTextEntryMap.try_emplace(kernelElf, std::make_pair(kernelTextOp, kernelEntryOp));
            }

            return kernelTextEntryMap[kernelElf];
        };

        llvm::SmallDenseMap<size_t, std::pair<mlir::Value, mlir::Value>> previousTasks;
        // Forever loop that runs until there are no more changes performed by
        //   the inner loop (so the algorithm has converged).

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getBody().getOps<VPURT::TaskOp>())) {
            bool found = false;

            for (auto op : llvm::make_early_inc_range(taskOp.getBodyRegion().getOps<VPUIP::SwKernelOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto tileIndex = op.getTileIndex().value_or(0);

                auto indexType = VPURegMapped::IndexType::get(ctx, tileIndex, 0, shaveTaskCount[tileIndex]);

                auto wait_bars = taskOp.getWaitBarriers();
                auto update_bars = taskOp.getUpdateBarriers();

                auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_bars) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_bars) {
                    val.setType(trivialIndexType);
                }

                auto sw_kernel_symbol = op.getKernelFunction();

                auto kernel_info_funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(sw_kernel_symbol);

                auto& [previousInvo, previousRanges] = previousTasks[tileIndex];

                std::pair<mlir::Value, mlir::Value> previousVals;

                const auto kernelTaskType = kernel_info_funcOp->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");
                bool isCacheOp = VPUIP::isCacheOpTaskType(kernelTaskType);
                if (!isCacheOp) {
                    previousVals =
                            createComputeOpSwKernel(ctx, op, builderBlk, kernel_info_funcOp, wait_bars, update_bars,
                                                    indexType, getKernelTextEntryMapFunc, previousInvo, previousRanges);
                } else {
                    previousVals =
                            createCacheOpSwKernel(ctx, op, builderBlk, kernelTaskType, wait_bars, update_bars,
                                                  indexType, previousInvo, previousRanges, getKernelTextEntryMapFunc);
                }

                previousRanges = previousVals.first;
                previousInvo = previousVals.second;
                ++shaveTaskCount[tileIndex];
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void replaceVPURTTaskOpWithM2IOps(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                      Logger& _log) {
        _log.info("VPUIP_VPUMI40XX pass: replaceVPURTTaskOpWithM2IOps()");

        auto tileCount = static_cast<size_t>(IE::getTileExecutor(moduleOp).getCount());
        mlir::SmallVector<uint32_t> m2iTaskCount(tileCount, 0);
        mlir::Value previousM2I;

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getBody().getOps<VPURT::TaskOp>())) {
            bool found = false;
            const auto taskIndex = 0;

            for (auto op : llvm::make_early_inc_range(taskOp.getBodyRegion().getOps<VPUIP::M2ITaskOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto indexType = VPURegMapped::IndexType::get(ctx, 0, 0, m2iTaskCount[taskIndex]);

                auto wait_bars = taskOp.getWaitBarriers();
                auto update_bars = taskOp.getUpdateBarriers();

                auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_bars) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_bars) {
                    val.setType(trivialIndexType);
                }

                auto startAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);
                auto cleanAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);

                auto doCscAttr = op.getDoCscAttr().getValue() ? mlir::UnitAttr::get(ctx) : nullptr;
                auto doNormAttr = op.getDoNormAttr().getValue() ? mlir::UnitAttr::get(ctx) : nullptr;

                auto M2IOp = builderBlk.create<VPUMI40XX::M2IOp>(
                        builderBlk.getUnknownLoc(), indexType, /*taskLocation*/ nullptr, previousM2I, op.getInput(),
                        op.getOutputBuff(), op.getProfilingData(), doCscAttr, doNormAttr, op.getInFmtAttr(),
                        op.getOutFmtAttr(), op.getChromaInReverseChannelsAttr(), op.getChromaOutReverseChannelsAttr(),
                        op.getLumaInReverseChannelsAttr(), op.getLumaOutReverseChannelsAttr(), op.getScaleFactorXAttr(),
                        op.getScaleFactorYAttr(), op.getNormAttr(), op.getTileOffsetXAttr(), op.getTileOffsetYAttr(),
                        op.getProfilingMetadataAttr(), op.getInterpAttr(), mlir::ValueRange(wait_bars),
                        mlir::ValueRange(update_bars), startAfterAttr, cleanAfterAttr);
                previousM2I = M2IOp.getResult();

                ++m2iTaskCount[taskIndex];
            }

            if (found) {
                taskOp->erase();
            }
        }
    }

    void replaceNCEClusterTaskOpWithDPUOps(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                           Logger& _log) {
        const auto tileCount = static_cast<size_t>(IE::getTileExecutor(moduleOp).getCount());
        mlir::SmallVector<uint32_t> variantTaskCount(tileCount, 0), invariantTaskCount(tileCount, 0);
        llvm::SmallDenseMap<size_t, std::pair<mlir::Value, mlir::Value>> previousTasks;

        for (auto taskOp : llvm::make_early_inc_range(funcOp.getOps<VPURT::TaskOp>())) {
            bool found = false;

            _log.trace("replaceNCEClusterTaskOpWithDPUOps(): taskOp = {0}", taskOp);

            for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<VPUIP::NCEClusterTaskOp>())) {
                found = true;
                mlir::OpBuilder builderBlk(taskOp);

                auto wait_barriers = taskOp.getWaitBarriers();
                auto update_barriers = taskOp.getUpdateBarriers();

                auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

                for (auto val : wait_barriers) {
                    val.setType(trivialIndexType);
                }

                for (auto val : update_barriers) {
                    val.setType(trivialIndexType);
                }

                const auto& dpuTasks = op.getVariants().getOps<VPUIP::DPUTaskOp>();
                VPUX_THROW_UNLESS(!dpuTasks.empty(), "Encountered op {} with empty dpu list", op);
                const auto& differentMPEModes = std::adjacent_find(dpuTasks.begin(), dpuTasks.end(),
                                                                   [](VPUIP::DPUTaskOp lhs, VPUIP::DPUTaskOp rhs) {
                                                                       return lhs.getMpeMode() != rhs.getMpeMode();
                                                                   });
                if (differentMPEModes != dpuTasks.end()) {
                    VPUIP::DPUTaskOp lhs = *differentMPEModes;
                    VPUIP::DPUTaskOp rhs = *std::next(differentMPEModes);
                    VPUX_THROW("Found dpu tasks {} and {} inside of {} which has different MPE modes {} and {} "
                               "accordingly, but only uniform MPE mode is supported by ELF",
                               lhs, rhs, op, lhs.getMpeMode(), rhs.getMpeMode());
                }

                const auto& differentClusterIds = std::adjacent_find(
                        dpuTasks.begin(), dpuTasks.end(), [](VPUIP::DPUTaskOp lhs, VPUIP::DPUTaskOp rhs) {
                            return lhs.getClusterId().value_or(0) != rhs.getClusterId().value_or(0);
                        });
                if (differentClusterIds != dpuTasks.end()) {
                    VPUIP::DPUTaskOp lhs = *differentClusterIds;
                    VPUIP::DPUTaskOp rhs = *std::next(differentClusterIds);
                    VPUX_THROW("Found dpu tasks {} and {} inside of {} which has different cluster IDs {} and {} "
                               "accordingly, but only uniform cluster IDs in all DPUTaskOp under a NCEClusterTaskOp "
                               "are supported in VPUX40XX",
                               lhs, rhs, op, lhs.getClusterId().value_or(0), rhs.getClusterId().value_or(0));
                }

                VPUIP::DPUTaskOp first = *(dpuTasks.begin());
                auto mpe_freq_mode = VPU::MPEModeAttr::get(ctx, first.getMpeMode());
                uint8_t tileIndex = 0;
                if (first.getClusterId().has_value()) {
                    tileIndex = first.getClusterId().value();
                } else {
                    auto bufferOp = mlir::cast<VPURT::DeclareBufferOp>(op.getInput().getDefiningOp());
                    if (bufferOp.getSection() == VPURT::BufferSection::CMX_NN) {
                        if (bufferOp.getSectionIndex().has_value() && !bufferOp.getSectionIndex().value().empty()) {
                            auto tiles = parseIntArrayAttr<uint8_t>(bufferOp.getSectionIndex().value());
                            tileIndex = *std::min_element(tiles.begin(), tiles.end());
                        }
                    }
                }

                auto& [previousInvariant, previousVariant] = previousTasks[tileIndex];

                auto invariantIndex = VPURegMapped::IndexType::get(ctx, tileIndex, 0, invariantTaskCount[tileIndex]);
                auto startAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);
                auto cleanAfterAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false), 0);

                auto input = extractFromDistributedBuff(builderBlk, op.getInput(), tileIndex);
                auto inputSparsityMap = extractFromDistributedBuff(builderBlk, op.getInputSparsityMap(), tileIndex);
                auto inputSETable = extractFromDistributedBuff(builderBlk, op.getInputStorageElementTable(), tileIndex);
                auto weights = extractFromDistributedBuff(builderBlk, op.getWeights(), tileIndex);
                auto weightsSparsityMap = extractFromDistributedBuff(builderBlk, op.getWeightsSparsityMap(), tileIndex);
                auto weightsTable = extractFromDistributedBuff(builderBlk, op.getWeightTable(), tileIndex);
                auto sprLookupTable = extractFromDistributedBuff(builderBlk, op.getSprLookupTable(), tileIndex);
                auto outputs = unrollDistributedBuff(builderBlk, op.getOutputBuff());
                auto outputSparsityMaps = unrollDistributedBuff(builderBlk, op.getOutputSparsityMapBuff());
                auto invariant = builderBlk.create<VPUMI40XX::DPUInvariantOp>(
                        op->getLoc(), invariantIndex, /*taskLocation*/ nullptr, previousInvariant, input,
                        inputSparsityMap, inputSETable, weights, weightsSparsityMap, weightsTable, sprLookupTable,
                        outputs, outputSparsityMaps, op.getProfilingData(), op.getTaskTypeAttr(), mpe_freq_mode,
                        op.getKernelSizeAttr(), op.getKernelStridesAttr(), op.getKernelPaddingAttr(),
                        op.getActivationWindowChannelLengthAttr(), op.getIsContinuedAttr(), op.getCmSpPatternAttr(),
                        op.getInputChannelsCompressionAttr(), op.getOutChannelOffsetAttr(), op.getIsSuperdenseAttr(),
                        op.getIsInplaceAttr(), op.getInputSeSizeAttr(), op.getOutputSeSizeAttr(),
                        op.getIsPermuteQuantizeAttr(), op.getIsSmallKernelOptimizedAttr(),
                        op.getProfilingMetadataAttr(), wait_barriers, update_barriers, startAfterAttr, cleanAfterAttr);

                previousInvariant = invariant.getResult();
                ++invariantTaskCount[tileIndex];

                auto readLut = sprLookupTable ? mlir::UnitAttr::get(&getContext()) : nullptr;

                auto tempPreviousVariant = previousVariant;
                for (auto dpuTaskOp : op.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                    auto variantIndex = VPURegMapped::IndexType::get(ctx, tileIndex, 0, variantTaskCount[tileIndex]);
                    auto clusterIdAttr = builderBlk.getIntegerAttr(builderBlk.getIntegerType(64, false),
                                                                   dpuTaskOp.getClusterId().value_or(0));
                    VPUX_THROW_WHEN((dpuTaskOp.getInStartAttr() == nullptr) || (dpuTaskOp.getInEndAttr() == nullptr),
                                    "Missing inStart/inEnd in DPUTask");

                    auto variant = builderBlk.create<VPUMI40XX::DPUVariantOp>(
                            builderBlk.getUnknownLoc(), variantIndex, nullptr, tempPreviousVariant,
                            invariant.getResult(), weights, op.getWeightTable(), op.getTaskTypeAttr(),
                            dpuTaskOp.getInStartAttr(), dpuTaskOp.getInEndAttr(), dpuTaskOp.getOutStartAttr(),
                            dpuTaskOp.getOutEndAttr(), dpuTaskOp.getPadAttr(), dpuTaskOp.getMpeModeAttr(),
                            clusterIdAttr, dpuTaskOp.getHaloRegionsAttr(), dpuTaskOp.getWorkloadIdAttr(), readLut);

                    // lut read can be optimised to be read only once per invariant / enabled for its first variant,
                    // as the non-linear function woudn't update until the next invariant
                    //
                    // invar0                         invar1               invar2
                    // |var0[1] var1[0] var2[0]]|  -> |var0[1] var1[0]|  ->|var0[1] var1[0] var2[0]|
                    //       ^                                      ^
                    //       ^lut_read enabled                      ^lut_read disabled

                    readLut = nullptr;
                    tempPreviousVariant = variant.getResult();
                    ++variantTaskCount[tileIndex];
                }
                previousVariant = tempPreviousVariant;

                {
                    auto& ppeRegion = invariant.getPpe();
                    ppeRegion.emplaceBlock();

                    mlir::OpBuilder::InsertionGuard guard(builderBlk);
                    builderBlk.setInsertionPointToEnd(&ppeRegion.front());

                    for (auto ppe : op.getPpe().getOps<VPUIP::PPETaskOp>()) {
                        builderBlk.create<VPUMI40XX::PPETaskOp>(builderBlk.getUnknownLoc(), ppe->getResultTypes(),
                                                                ppe->getOperands(),
                                                                ppe->getAttrDictionary().getValue());
                    }
                }
            }

            if (found) {
                taskOp->erase();
            }
        }
    }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    void setBarrierIndexValues(mlir::MLIRContext* ctx, mlir::func::FuncOp& funcOp, Logger _log) {
        auto barrierCount = 0;

        VPUX_UNUSED(_log);

        for (auto op : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
            auto indexType = VPURegMapped::IndexType::get(ctx, barrierCount);

            op.getOperation()->getResult(0).setType(indexType);

            ++barrierCount;
        }
    }

    template <typename TaskType>
    static bool noCond(TaskType i) {
        VPUX_UNUSED(i);
        return true;
    }

    template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
    size_t countTasksIf(mlir::func::FuncOp& funcOp, Condition&& condition = noCond) {
        auto tasks = funcOp.template getOps<TaskType>();
        return std::count_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
    }

    template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
    mlir::Value findTaskIf(mlir::func::FuncOp& funcOp, Condition&& condition = noCond) {
        auto tasks = funcOp.template getOps<TaskType>();
        auto target = std::find_if(tasks.begin(), tasks.end(), std::forward<Condition>(condition));
        return target != tasks.end() ? (*target).getResult() : mlir::Value();
    }

    template <typename TaskType, typename Condition = decltype(noCond<TaskType>)>
    int64_t gatherTasks(mlir::SmallVector<mlir::Value>& taskValues, mlir::func::FuncOp& funcOp, uint32_t tileIdx,
                        uint32_t listIdx) {
        auto indexCond = [tileIdx, listIdx](auto op) {
            auto type = op.getIndex().getType().template dyn_cast<vpux::VPURegMapped::IndexType>();
            return (type.getTileIdx() == tileIdx) && (type.getListIdx() == listIdx);
        };

        auto head = findTaskIf<TaskType>(funcOp, indexCond);
        if (head) {
            taskValues.push_back(head);
        }
        return countTasksIf<TaskType>(funcOp, indexCond);
    }

    std::pair<mlir::Value, mlir::ValueRange> setupActKernelRt(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp,
                                                              mlir::OpBuilder& builderFunc, bool createStacks = false) {
        // check for actShaveRt info
        mlir::Value actShvRt;
        auto vpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");
        VPUX_THROW_UNLESS(vpuSwModuleOp != nullptr, "setupActKernelConfig: @VPU.SW module missing.");
        auto runtimeKernelFunction = vpuSwModuleOp.lookupSymbol<mlir::func::FuncOp>("runtime");

        // check for actShave stacks info
        auto swRtOpRange = moduleOp.getOps<VPURT::SWRunTimeOp>();
        llvm::SmallVector<mlir::Value> shaveStacks;
        if (!swRtOpRange.empty() && createStacks) {
            VPUX_THROW_WHEN(std::distance(swRtOpRange.begin(), swRtOpRange.end()) > 1,
                            "More than 1 instance of VPURT.SW.Runtime");
            auto swRtOp = *(swRtOpRange.begin());

            for (auto stackSize : mlir::extractFromIntegerArrayAttr<int64_t>(swRtOp.getStacks())) {
                if (stackSize == 0) {
                    continue;
                }

                auto actShaveStackMemrefType =
                        vpux::getLinearMemrefType(ctx, stackSize, vpux::getInt8Type(ctx), VPU::MemoryKind::DDR);

                auto declareBufferOp =
                        builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                   actShaveStackMemrefType,    // Type
                                                                   VPURT::BufferSection::DDR,  // Buffer Type
                                                                   0                           // byteOffset
                        );
                shaveStacks.push_back(declareBufferOp.getResult());
            }
        }

        if (runtimeKernelFunction) {
            const auto kernelElf =
                    std::string(runtimeKernelFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_code").getValue());

            auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

            auto actShvRtOp = builderFunc.create<VPUMI40XX::ActShaveRtOp>(builderFunc.getUnknownLoc(), trivialIndexType,
                                                                          mlir::StringAttr::get(ctx, kernelElf));

            actShvRt = actShvRtOp.getResult();
        } else {
            auto actRtCodeBufferMemrefType = vpux::getLinearMemrefType(ctx, ACT_RT_CODE_BUFFER_SIZE,
                                                                       vpux::getInt8Type(ctx), VPU::MemoryKind::DDR);

            auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                              actRtCodeBufferMemrefType,  // Type
                                                                              VPURT::BufferSection::DDR,  // Buffer Type
                                                                              0                           // byteOffset
            );

            actShvRt = declareBufferOp.getResult();
        }
        return std::make_pair(actShvRt, mlir::ValueRange(shaveStacks));
    }

    void createMappedInferenceOp(mlir::MLIRContext* ctx, mlir::ModuleOp& moduleOp, mlir::func::FuncOp& funcOp,
                                 Logger _log) {
        _log.info("VPUIP_VPUMI40XX pass: createMappedInferenceOp()");

        const auto tileCount = static_cast<size_t>(IE::getTileExecutor(moduleOp).getCount());
        const auto dmaTileCount =
                static_cast<size_t>(IE::getAvailableExecutor(moduleOp, VPU::ExecutorKind::DMA_NN).getCount());

        const auto dmaNnSrcTypeCount = static_cast<size_t>(DmaNnSrcType::Count);

        mlir::SmallVector<mlir::SmallVector<mlir::Value>> dmaTasks(dmaTileCount);
        mlir::SmallVector<mlir::ValueRange> dmaTasksArg(dmaTileCount);
        size_t dmaTasksArgLength = 0;
        mlir::SmallVector<mlir::Value> invariantTasks, variantTasks, actKernelRanges, actKernelInvocations;
        mlir::Value barrierTasks;
        mlir::Value mediaTasks;
        mlir::Value actShvRt;
        mlir::ValueRange actShaveStacks;

        mlir::SmallVector<mlir::SmallVector<int64_t>> dmaCount(dmaTileCount,
                                                               mlir::SmallVector<int64_t>(dmaNnSrcTypeCount, 0));
        mlir::SmallVector<int64_t> invariantCount(tileCount, 0), variantCount(tileCount, 0), rangeCount(tileCount, 0),
                invoCount(tileCount, 0);
        int64_t barrierCount = 0;
        int64_t mediaCount = 0;
        bool hasInvocations = false;

        for (size_t tileIdx = 0; tileIdx < dmaTileCount; ++tileIdx) {
            // dmaTasks
            for (size_t srcType = 0; srcType < dmaNnSrcTypeCount; ++srcType) {
                dmaCount[tileIdx][srcType] =
                        gatherTasks<VPUMI40XX::NNDMAOp>(dmaTasks[tileIdx], funcOp, tileIdx, srcType);
            }
            if (!dmaTasks[tileIdx].empty()) {
                dmaTasksArg[tileIdx] = mlir::ValueRange(dmaTasks[tileIdx]);
                dmaTasksArgLength = tileIdx + 1;
            }
        }

        for (size_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            // invariantTasks
            invariantCount[tileIdx] = gatherTasks<VPUMI40XX::DPUInvariantOp>(invariantTasks, funcOp, tileIdx, 0);

            // variantTasks
            variantCount[tileIdx] = gatherTasks<VPUMI40XX::DPUVariantOp>(variantTasks, funcOp, tileIdx, 0);

            // actKernelRanges
            rangeCount[tileIdx] = gatherTasks<VPUMI40XX::ActKernelRangeOp>(actKernelRanges, funcOp, tileIdx, 0);

            // actKernelInvocations
            invoCount[tileIdx] =
                    gatherTasks<VPUMI40XX::ActKernelInvocationOp>(actKernelInvocations, funcOp, tileIdx, 0);

            if (invoCount[tileIdx] != 0)
                hasInvocations = true;
        }

        // barrierTasks
        barrierTasks = findTaskIf<VPUMI40XX::ConfigureBarrierOp>(funcOp);
        barrierCount = countTasksIf<VPUMI40XX::ConfigureBarrierOp>(funcOp);

        // mediaTasks
        mediaTasks = findTaskIf<VPUMI40XX::M2IOp>(funcOp);
        mediaCount = countTasksIf<VPUMI40XX::M2IOp>(funcOp);

        // create MappedInferenceOp
        mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

        // create ActShaveRtOp
        if (hasInvocations) {
            std::tie(actShvRt, actShaveStacks) = setupActKernelRt(ctx, moduleOp, builderFunc);
        }

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);
        builderFunc.create<VPUMI40XX::MappedInferenceOp>(
                mlir::UnknownLoc::get(ctx), trivialIndexType,
                ArrayRef(dmaTasksArg.data(), dmaTasksArgLength),        // llvm::ArrayRef<::mlir::ValueRange> dmaTasks
                invariantTasks,                                         // mlir::ValueRange invariantTasks
                variantTasks,                                           // mlir::ValueRange variantTasks
                actKernelRanges,                                        // mlir::ValueRange actKernelRanges
                actKernelInvocations,                                   // mlir::ValueRange actKernelInvocations
                mediaTasks,                                             // mlir::Value mediaTasks
                barrierTasks,                                           // mlir::Value barrierTasks
                nullptr,                                                // mlir::Value workItemTasks
                nullptr,                                                // mlir::Value bootstrapTasks
                actShvRt,                                               // mlir::Value actShaveRt
                actShaveStacks,                                         // mlir::ValueRange actShaveStacks
                nullptr,                                                // mlir::Value dmaHwpBase
                nullptr,                                                // mlir::Value hwpWorkpointCfg
                getIntArrayOfArray(ctx, dmaCount),                      // mlir::ArrayAttr dmaCount
                builderFunc.getI64ArrayAttr(ArrayRef(invariantCount)),  // mlir::ArrayAttr invariantCount
                builderFunc.getI64ArrayAttr(ArrayRef(variantCount)),    // mlir::ArrayAttr variantCount
                builderFunc.getI64ArrayAttr(ArrayRef(rangeCount)),      // mlir::ArrayAttr actKernelRangesCount
                builderFunc.getI64ArrayAttr(ArrayRef(invoCount)),       // mlir::ArrayAttr actKernelInvocationsCount
                mediaCount,                                             // mlir::IntegerAttr mediaCount
                barrierCount,                                           // mlir::IntegerAttr barrierCount
                nullptr,                                                // mlir::IntegerAttr workItemCount
                nullptr,                                                // mlir::IntegerAttr bootstrapTasksCount
                nullptr,                                                // mlir::IntegerAttr bootstrapWorkItemTasksCount
                nullptr);                                               // mlir::IntegerAttr finalBarrierId
    }

    void createProfilingMetadataOp(mlir::MLIRContext* ctx, IE::CNNNetworkOp netOp, mlir::func::FuncOp& funcOp,
                                   Logger _log) {
        if (netOp.getProfilingOutputsInfo().empty()) {
            return;
        }
        _log.trace("VPUIP_VPUMI40XX pass: createProfilingMetadataOp()");

        mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

        auto buffer = vpux::buildProfilingMetadataBuffer(netOp, funcOp, _log);
        llvm::ArrayRef<char> rawMetadata{reinterpret_cast<const char*>(buffer.data()), buffer.size()};
        long int bufferSize = buffer.size();

        auto vectorType = mlir::VectorType::get({bufferSize}, getUInt8Type(ctx));
        const auto elemAttr = mlir::DenseElementsAttr::getFromRawBuffer(vectorType, rawMetadata);
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);
        builderFunc.create<VPUMI40XX::ProfilingMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType, elemAttr);
    }

};  // namespace

class ConvertVPURTConfigureBarrierOp final : public mlir::OpRewritePattern<VPURT::ConfigureBarrierOp> {
public:
    ConvertVPURTConfigureBarrierOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::ConfigureBarrierOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPURT::ConfigureBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const override {
        auto ctx = ConvertVPURTConfigureBarrierOp::getContext();

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        mlir::Value origOpResult = origOp.getBarrier();  // E#105629: barrier variable may have incorrect type, so only
                                                         // access it through mlir::Value

        size_t producer_count = 0;
        size_t consumer_count = 0;

        // should use VPUMI40XX TaskOp interface
        for (auto user : origOpResult.getUsers()) {
            if (auto taskOp = mlir::dyn_cast<vpux::VPUMI40XX::ExecutableTaskOpInterface>(user)) {
                for (auto waitBar : taskOp.waitBarriers()) {
                    if (origOpResult == waitBar) {
                        consumer_count += taskOp.getBarrierHitsCount();
                    }
                }

                for (auto updateBar : taskOp.updateBarriers()) {
                    if (origOpResult == updateBar) {
                        producer_count += taskOp.getBarrierHitsCount();
                    }
                }
            }
        }

        mlir::IntegerType uint8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);

        rewriter.replaceOpWithNewOp<VPUMI40XX::ConfigureBarrierOp>(
                origOp,
                trivialIndexType,                                   // setup all barriers with the trivial index (0)
                checked_cast<uint8_t>(origOp.getId()),              // real_id
                -1,                                                 // int64_t next_same_id()
                mlir::IntegerAttr::get(uint8Type, producer_count),  // origOp.producer_countAttr(),
                mlir::IntegerAttr::get(uint8Type, consumer_count),  // origOp.consumer_countAttr(),
                origOp.getIsFinalBarrier());
        barrierCount++;
        return mlir::success();
    }

private:
    Logger _log;
    mutable int barrierCount = 0;
};

class ReturnOpRewritePattern final : public mlir::OpRewritePattern<mlir::func::ReturnOp> {
public:
    ReturnOpRewritePattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::func::ReturnOp>(ctx), _log(std::move(log)) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp origOp,
                                        [[maybe_unused]] mlir::PatternRewriter& rewriter) const override {
        auto function = origOp->getParentOfType<mlir::func::FuncOp>();
        assert(function);
        auto context = function->getContext();

        auto taskOps = function.getOps<VPURegMapped::TaskOpInterface>();

        using Range = std::tuple<VPURegMapped::TaskType, uint32_t, uint32_t>;
        const auto getRange = [](auto taskOp) {
            const auto index = taskOp.getIndexType();
            return std::make_tuple(taskOp.getTaskType(), index.getTileIdx(), index.getListIdx());
        };

        mlir::DenseMap<Range, size_t> rangesSizes;
        const auto getRangeSize = [&](auto taskOp) {
            auto& [_, rangeSize] = rangesSizes.FindAndConstruct(getRange(taskOp));
            if (rangeSize != 0) {
                return rangeSize;
            }

            const auto isOpFromTheSameRange = [taskOp, getRange](auto op) {
                return getRange(taskOp) == getRange(op);
            };

            return rangeSize = std::count_if(std::begin(taskOps), std::end(taskOps), isOpFromTheSameRange);
        };

        const auto isFirst = [](auto taskOp) {
            return taskOp.getIndexType().getValue() == 0;
        };

        const auto isLast = [getRangeSize](auto taskOp) {
            return taskOp.getIndexType().getValue() == getRangeSize(taskOp) - 1;
        };

        mlir::DenseMap<Range, size_t> rangesIndexes;
        mlir::SmallVector<mlir::Attribute> rangesTaskTypesAttrs;
        mlir::SmallVector<mlir::Value> rangesBegins;
        mlir::SmallVector<mlir::Value> rangesEnds;

        for (auto taskOp : taskOps) {
            const auto range = getRange(taskOp);
            const auto result = taskOp.getResult();

            if (isFirst(taskOp)) {
                rangesTaskTypesAttrs.push_back(VPURegMapped::TaskTypeAttr::get(context, taskOp.getTaskType()));
                rangesBegins.push_back(result);
                rangesEnds.push_back({});
                rangesIndexes[range] = rangesBegins.size() - 1;
            }

            if (isLast(taskOp)) {
                rangesEnds[rangesIndexes[range]] = result;
            }
        }

        if (rangesSizes.size() != rangesIndexes.size() || rangesIndexes.size() != rangesTaskTypesAttrs.size() ||
            rangesTaskTypesAttrs.size() != rangesBegins.size() || rangesBegins.size() != rangesEnds.size()) {
            return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<VPUMI40XX::OpRanges>(origOp, mlir::ArrayRef(rangesBegins),
                                                         mlir::ArrayRef(rangesEnds),
                                                         mlir::ArrayAttr::get(context, rangesTaskTypesAttrs));

        return mlir::success();
    }

private:
    Logger _log;
};

void ConvertVPUIP2VPUMI40XXPass::safeRunOnModule() {
    auto ctx = &(getContext());
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp funcOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, funcOp);

    _log.trace("funcOp = {0}", funcOp);

    createProfilingMetadataOp(ctx, netOp, funcOp, _log);

    _log.trace("funcOp after creating ProfilingMetadataOp = {0}", funcOp);

    replaceVPURTTaskOpWithNNDMAOp(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing NNDMA Ops = {0}", funcOp);

    replaceVPURTTaskOpWithKernelOps(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing ActKernel Ops = {0}", funcOp);

    replaceVPURTTaskOpWithM2IOps(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing M2I Ops = {0}", funcOp);

    replaceNCEClusterTaskOpWithDPUOps(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after replacing DPU Ops = {0}", funcOp);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<VPUMI40XX::VPUMI40XXDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<VPURT::DeclareBufferOp>();
    target.addLegalOp<VPUIP::GroupSparseBufferOp>();

    mlir::RewritePatternSet patterns(ctx);

    patterns.add<ConvertVPURTConfigureBarrierOp>(ctx, _log);
    patterns.add<ReturnOpRewritePattern>(ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
    }

    _log.trace("funcOp after replacing Barrier Ops = {0}", funcOp);

    setBarrierIndexValues(ctx, funcOp, _log);

    _log.trace("funcOp after setting Barrier indexes = {0}", funcOp);

    createMappedInferenceOp(ctx, moduleOp, funcOp, _log);

    _log.trace("funcOp after generating MappedInferenceOp = {0}", funcOp);
}

}  // namespace

//
// createConvertVPUIP2VPUMI40XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUIP2VPUMI40XXPass(Logger log, bool enableMemorySideCache) {
    return std::make_unique<ConvertVPUIP2VPUMI40XXPass>(log, enableMemorySideCache);
}
