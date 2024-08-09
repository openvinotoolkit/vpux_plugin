//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ClusterNCERewriter
//

class ClusterNCERewriter final : public VPUIP::ClusterNCEBaseRewriter {
public:
    ClusterNCERewriter(mlir::MLIRContext* ctx, Logger log): ClusterNCEBaseRewriter(ctx, log) {
    }

private:
    void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                          SmallVector<mlir::Value>& parentOutputSparsityMap,
                          SmallVector<mlir::Value>& outputSparsityMapBuffs,
                          SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                          VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                          mlir::OpBuilder& builder) const override;

    void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                         SmallVector<mlir::Value>& parentInputSparsityMap,
                         SmallVector<mlir::Value>& inputSparsityMapBuffs, SmallVector<mlir::Value>& parentInputSETable,
                         SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                         VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                         mlir::OpBuilder& builder) const override;

    mlir::UnitAttr isSegmentedNCETask(VPUIP::DistributedBufferType /*inputType*/) const override {
        return nullptr;
    };
};

void ClusterNCERewriter::getInputBuffers(
        SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
        SmallVector<mlir::Value>& parentInputSparsityMap, SmallVector<mlir::Value>& inputSparsityMapBuffs,
        SmallVector<mlir::Value>& parentInputSETable, SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
        VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters, mlir::OpBuilder& builder) const {
    inputBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "input", nceTask.getInput(), numClusters, builder);
    parentInputBuffs = inputBuffs;
    inputSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSparsityMap",
                                                              nceTask.getInputSparsityMap(), numClusters, builder);
    inputSETableBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSETable",
                                                          nceTask.getInputStorageElementTable(), numClusters, builder);
    parentInputSparsityMap = inputSparsityMapBuffs;
    parentInputSETable = inputSETableBuffs;
}

void ClusterNCERewriter::getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs,
                                          SmallVector<mlir::Value>& outputBuffs,
                                          SmallVector<mlir::Value>& parentOutputSparsityMap,
                                          SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                          SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                                          VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                          mlir::OpBuilder& builder) const {
    const auto hasHalo = [&]() -> bool {
        auto operandType = nceTask.getOutputBuff().getType();
        auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
        const auto distribution = distributedType.getDistribution();
        const auto distributionMode = distribution.getMode().getValue();

        return (distributionMode == VPU::DistributionMode::OVERLAPPED) ||
               (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) ||
               (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED));
    };

    if (hasHalo()) {
        std::tie(outputBuffs, outputItiBuffs) =
                VPUIP::getPerClusterOutputHaloBuffers(_ctx, loc, "outputBuff", nceTask.getOutputBuff(), numClusters);

        outputSparsityMapBuffs = SmallVector<mlir::Value>(numClusters, nullptr);
        if (auto sparsityClusterOperand = nceTask.getOutputSparsityMapBuff()) {
            std::tie(outputSparsityMapBuffs, std::ignore) = VPUIP::getPerClusterOutputHaloBuffers(
                    _ctx, loc, "outputSparsityMapBuff", sparsityClusterOperand, numClusters);
        }

        parentOutputBuffs = outputBuffs;
        parentOutputSparsityMap = outputSparsityMapBuffs;

        return;
    }

    outputBuffs = VPUIP::getPerClusterComputeBuffers(_ctx, loc, "outputBuff", nceTask.getOutputBuff(), numClusters,
                                                     builder, true);
    outputSparsityMapBuffs = VPUIP::getPerClusterComputeBuffers(
            _ctx, loc, "outputSparsityMapBuff", nceTask.getOutputSparsityMapBuff(), numClusters, builder, true);

    parentOutputBuffs = outputBuffs;
    parentOutputSparsityMap = outputSparsityMapBuffs;
}

//
// ClusterConvertDMARewriter
//

class ClusterConvertDMARewriter final : public VPUIP::ClusterPerElementDMABaseRewriter {
public:
    ClusterConvertDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : ClusterPerElementDMABaseRewriter(ctx, dmaPortCount, log) {
    }

private:
    bool isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const override;
    virtual VPUIP::DMATypeOpInterface wrapIntoTaskOp(VPUIP::DMATypeOpInterface dmaOp, VPURT::TaskOp vpurtTask,
                                                     mlir::Location loc, mlir::Value input, mlir::Value output_buff,
                                                     int64_t port, mlir::OpBuilder& builder) const override;
    UnrollingType getUnrollingType(VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const override;
};

bool ClusterConvertDMARewriter::isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const {
    return mlir::isa<VPUIP::ConvertDMAOp>(dmaOp.getOperation());
}

VPUIP::DMATypeOpInterface ClusterConvertDMARewriter::wrapIntoTaskOp(VPUIP::DMATypeOpInterface, VPURT::TaskOp vpurtTask,
                                                                    mlir::Location loc, mlir::Value input,
                                                                    mlir::Value output_buff, int64_t port,
                                                                    mlir::OpBuilder& builder) const {
    return VPURT::wrapIntoTaskOp<VPUIP::ConvertDMAOp>(builder, vpurtTask.getWaitBarriers(),
                                                      vpurtTask.getUpdateBarriers(), loc, input, output_buff, port);
}

ClusterConvertDMARewriter::UnrollingType ClusterConvertDMARewriter::getUnrollingType(
        VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const {
    // Normally we don't support both distributed input and output NNCMX->NNCMX DMAs
    // but this is an exception since ConvertDMA gets translated from SW Convert layer
    // which has its input and output in NNCMX and is already tiled to fit in NNCMX
    VPUX_THROW_WHEN(inputMode == VPU::DistributionMode::NONE && outputMode == VPU::DistributionMode::NONE,
                    "One of input/output must be distributed type for cluster ConvertDMAOp");
    const auto isSegmentedOrOverlapped = [](VPU::DistributionMode mode) {
        return mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED;
    };
    const auto isDuplicated = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED);
    };
    if ((inputMode == VPU::DistributionMode::NONE && isSegmentedOrOverlapped(outputMode)) ||
        (outputMode == VPU::DistributionMode::NONE && isSegmentedOrOverlapped(inputMode)) ||
        (inputMode == outputMode && isSegmentedOrOverlapped(inputMode))) {
        return UnrollingType::SEGMENTED;
    }
    if ((inputMode == VPU::DistributionMode::NONE && isDuplicated(outputMode)) ||
        (outputMode == VPU::DistributionMode::NONE && isDuplicated(inputMode)) ||
        (isDuplicated(outputMode) && isDuplicated(inputMode))) {
        return UnrollingType::DUPLICATED;
    }
    return UnrollingType::FAILED;
}

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::arch40xx::UnrollClusterTilingBase<UnrollClusterTilingPass> {
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
    const ClusterNCERewriter nceRewriter(&ctx, _log);
    const ClusterConvertDMARewriter convertDMARewriter(&ctx, dmaPortCount, _log);

    func.walk<mlir::WalkOrder::PostOrder>([&](VPURT::TaskOp vpurtTask) {
        auto op = vpurtTask.getInnerTaskOp();
        if (op == nullptr) {
            return;
        }

        mlir::OpBuilder builder(op);
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            dmaRewriter.matchAndRewrite(nndmaOp, builder, /*isDataOverlapped*/ true);
        } else if (auto taskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            nceRewriter.matchAndRewrite(taskOp, builder);
        } else if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            swRewriter.matchAndRewrite(swOp, builder);
        } else if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
            convertDMARewriter.matchAndRewrite(dmaOp, builder);
        }
    });
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
