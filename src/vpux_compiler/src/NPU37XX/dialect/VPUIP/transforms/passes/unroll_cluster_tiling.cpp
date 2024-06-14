//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/profiling.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {
void updateSwProfilingMetadata(VPUIP::SwKernelOp newTask, VPUIP::SwProfilingMetadataAttr attr, size_t clusterId) {
    if (attr == nullptr) {
        return;
    }
    const size_t bufferId = attr.getBufferId().getInt();
    const size_t bufferOffset = attr.getBufferOffset().getInt();
    const size_t clusterSize = attr.getClusterSize().getInt();
    const size_t dataIndex = attr.getDataIndex().getInt();
    const size_t tileId = attr.getTileId().getInt();
    auto profMeta = vpux::getSwProfilingMetaAttr(attr.getContext(), bufferId, bufferOffset, clusterSize, dataIndex,
                                                 tileId, clusterId);
    newTask.setProfilingMetadataAttr(profMeta);
}
};  // namespace

//
// ClusterSWRewriter
//

void VPUIP::arch37xx::ClusterSWRewriter::matchAndRewrite(VPUIP::SwKernelOp swTask, mlir::OpBuilder& builder) const {
    _log.trace("Process SW op: '{0}'", swTask);

    auto vpurtTask = swTask->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    builder.setInsertionPointAfter(vpurtTask);

    if (swTask.getInputs().empty() || swTask.getOutputs().empty()) {
        return;
    }

    auto input = *swTask.getInputs().begin();
    auto output = *swTask.getOutputs().begin();

    auto inputType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto outputType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();

    if (inputType == nullptr && outputType == nullptr) {
        _log.trace("Input and output types are not distributed, nothing to unroll");
        auto oldLoc = swTask->getLoc();
        VPUX_THROW_WHEN(stringifyPrimaryLocation(oldLoc).find("/cluster_") != std::string::npos,
                        "/cluster_ suffix should not be present yet but was found in {0}", oldLoc);
        swTask->setLoc(appendLoc(oldLoc, "cluster_0"));
        return;
    }

    auto inDistribution = inputType.getDistribution();
    auto outDistribution = outputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.getNumClusters() == outDistribution.getNumClusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.getNumClusters(),
                      outDistribution.getNumClusters());

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();
    VPUX_THROW_WHEN(outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for SW op {0}; output in OVERLAPPED mode.", swTask->getLoc());
    VPUX_THROW_WHEN(inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When SW op has input in OVERLAPPED mode then output must be segmented. op = {0}, out mode = '{1}'",
                    swTask->getLoc(), VPU::stringifyDistributionMode(outDistributionMode));

    auto numClusters = inDistribution.getNumClusters().getInt();
    auto loc = swTask->getLoc();

    auto parentInputBuffs = swTask.getInputs();
    auto parentOutputBuffs = swTask.getOutputBuffs();

    // store inputs/outputs per cluster
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> inputBuffs;
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> outputBuffs;
    SmallVector<TileInfo> outputTiles;
    SmallVector<TilingInfo> inputTiles;

    auto allowDiscontinuousBuffers = VPUIP::isStridedDataAccessSupported(swTask);
    for (const auto& input : parentInputBuffs) {
        auto currBuffs = VPUIP::getPerClusterSWMemoryBuffers(_ctx, loc, "input", swTask, input, numClusters, builder,
                                                             _log, allowDiscontinuousBuffers);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            inputBuffs[clusterId].push_back(currBuffs[clusterId]);
        }
    }

    for (const auto& output : parentOutputBuffs) {
        auto currBuffs = VPUIP::getPerClusterSWComputeBuffers(_ctx, loc, "outputBuff", swTask, output, numClusters,
                                                              builder, _log, true);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            outputBuffs[clusterId].push_back(currBuffs[clusterId]);
        }
    }

    auto getPerClusterTileInfo = [&numClusters](ShapeRef shape, ShapeRef offset, int64_t tileDim) {
        Shape axis(shape.size(), 1);
        axis[Dim(tileDim)] = numClusters;
        return TileInfo(shape, offset, axis);
    };

    // For overlapped input, the Swkernel's attr need to be updated according to its input/output tiles
    auto needUpdateAttrs = inDistributionMode == VPU::DistributionMode::OVERLAPPED;
    if (needUpdateAttrs) {
        auto outTileIndex = VPUIP::getTilingDimIndex(outputType);
        VPUX_THROW_UNLESS(outTileIndex.has_value(), "Can not get tiling dim for {0}", outputType);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            SmallVector<TileInfo> tiles;
            for (const auto& operand : parentInputBuffs) {
                auto distributedType = operand.getType().dyn_cast<VPUIP::DistributedBufferType>();
                auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
                VPUX_THROW_UNLESS(tileIndex.has_value(), "Can not get tiling dim for {0}", distributedType);
                auto tileInfo = getPerClusterTileInfo(distributedType.getPerClusterMemoryShapes()[clusterId],
                                                      distributedType.getPerClusterMemoryShapeOffsets()[clusterId],
                                                      tileIndex.value());
                tiles.push_back(tileInfo);
            }
            auto inTiles = TilingInfo(tiles);
            auto outTile = getPerClusterTileInfo(outputType.getPerClusterComputeShapes()[clusterId],
                                                 outputType.getPerClusterComputeShapeOffsets()[clusterId],
                                                 outTileIndex.value());
            inputTiles.push_back(inTiles);
            outputTiles.push_back(outTile);
        }
    }

    auto profilingBuffs = VPUIP::getPerClusterSWMemoryBuffers(_ctx, loc, "profilingBuff", swTask,
                                                              swTask.getProfilingData(), numClusters, builder, _log);

    auto taskArgs = kernelArgsRange(swTask);

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, "cluster_{0}", clusterId);
        mlir::Value profilingData = nullptr;
        mlir::Type profilingOutputType = nullptr;

        if (swTask.getProfilingData()) {
            profilingOutputType = profilingBuffs[clusterId].getType();
            profilingData = profilingBuffs[clusterId];
            VPUX_THROW_WHEN(swTask.getProfilingMetadataAttr() == nullptr, "Missing profiling metadata for '{0}'",
                            swTask);
        }

        SmallVector<mlir::Type> inputTypes;
        for (auto& temp : inputBuffs[clusterId]) {
            inputTypes.push_back(temp.getType());
        }
        for (auto& temp : outputBuffs[clusterId]) {
            inputTypes.push_back(temp.getType());
        }

        auto newArgs = needUpdateAttrs ? VPUIP::getSwkernelNewAttrsAfterTiling(swTask, taskArgs, inputTiles[clusterId],
                                                                               outputTiles[clusterId], _log.nest())
                                       : taskArgs;
        for (auto& arg : newArgs) {
            const auto typedAttr = arg.dyn_cast_or_null<mlir::TypedAttr>();
            const auto type = typedAttr != nullptr ? typedAttr.getType() : mlir::NoneType::get(_ctx);
            inputTypes.push_back(type);
        }

        VPUIP::createRuntimeKernelDefinition(_module, _log.nest(), VPU::getArch(swTask.getOperation()));

        auto module = swTask->getParentOfType<mlir::ModuleOp>();
        auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swTask.getKernelFunctionAttr());
        VPUX_THROW_UNLESS(kernelFunc, "Invalid function call : '{0}', undefined kernel name",
                          swTask.getKernelFunctionAttr());

        const auto kernelCode = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_code");
        const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
        auto newOperands = kernelFunc.getName();

        auto builtInFunction =
                VPUIP::createBuiltInFunction(_module, newOperands, inputTypes, kernelEntryPoint, kernelCode, _log);

        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                builder, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffs[clusterId],
                outputBuffs[clusterId], profilingData, builtInFunction, getIntAttr(builder, clusterId));
        updateSwProfilingMetadata(newTask, swTask.getProfilingMetadataAttr(), clusterId);

        initSwKernel(newTask, inputBuffs[clusterId], outputBuffs[clusterId], newArgs, _log.nest());

        _log.trace("Task created: {0}", newTask);
    }

    vpurtTask->dropAllReferences();
    vpurtTask->remove();
}

namespace {

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::arch37xx::UnrollClusterTilingBase<UnrollClusterTilingPass> {
public:
    explicit UnrollClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollClusterTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    const VPUIP::ClusterDMARewriter dmaRewriter(&ctx, dmaPortCount, _log);
    const VPUIP::arch37xx::ClusterSWRewriter swRewriter(&ctx, module, _log);
    const VPUIP::arch30xx::ClusterNCERewriter nceRewriter(&ctx, _log);

    mlir::SmallVector<mlir::Operation*> toRemove;

    func.walk([&](mlir::Operation* op) {
        mlir::OpBuilder builder(op);
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            dmaRewriter.matchAndRewrite(nndmaOp, builder);
        } else if (auto taskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            nceRewriter.matchAndRewrite(taskOp, builder);
        } else if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            swRewriter.matchAndRewrite(swOp, builder);
        }
    });
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch37xx::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
