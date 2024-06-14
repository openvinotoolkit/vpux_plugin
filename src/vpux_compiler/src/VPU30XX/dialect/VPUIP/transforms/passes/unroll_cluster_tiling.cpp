//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// ClusterNCERewriter
//

void VPUIP::arch30xx::ClusterNCERewriter::getInputBuffers(
        SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
        SmallVector<mlir::Value>& parentInputSparsityMap, SmallVector<mlir::Value>& inputSparsityMapBuffs,
        SmallVector<mlir::Value>& parentInputSETable, SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
        VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters, mlir::OpBuilder& builder) const {
    inputBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "input", nceTask.getInput(), numClusters, builder);
    auto parentInput = *nceTask.getInputs().begin();
    auto parentInputType = parentInput.getType().cast<VPUIP::DistributedBufferType>();

    mlir::UnitAttr isSegmented = isSegmentedNCETask(parentInputType);

    parentInputBuffs = VPU::isSegmentedOverC(parentInputType.getDistribution())
                               ? inputBuffs
                               : SmallVector<mlir::Value>(numClusters, parentInput);

    inputSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSparsityMap",
                                                              nceTask.getInputSparsityMap(), numClusters, builder);
    inputSETableBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSETable",
                                                          nceTask.getInputStorageElementTable(), numClusters, builder);

    auto arch = VPU::getArch(nceTask);
    bool isDWOpAndNeedsAlign = VPU::isDWOpAndNeedsAlign(arch, nceTask.getTaskType());
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        // For 37XX and 30XX arch, ensure we have H_per_cluster x W as a multiple of 4 (or 8 for sparse inputs).
        // If the storage element table is present, its segment size has to fit this restriction
        if (isSegmented && clusterId != (numClusters - 1) &&
            (nceTask.getTaskType() == VPUIP::NCETaskType::CONV || isDWOpAndNeedsAlign)) {
            auto inShape = inputBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            if (nceTask.getInputStorageElementTable() != nullptr) {
                inShape = inputSETableBuffs[clusterId].getType().cast<NDTypeInterface>().getShape();
            }
            const auto isInputSparse =
                    nceTask.getInputSparsityMap() != nullptr || nceTask.getInputStorageElementTable() != nullptr;
            const auto hAlignment = VPU::getSOHPerClusterHeightAlignment(inShape[Dims4D::Act::W], isInputSparse);
            VPUX_THROW_UNLESS((inShape[Dims4D::Act::H] % hAlignment) == 0,
                              "For segmented cluster we must have alignment to {0}, type: {1}", hAlignment,
                              inputBuffs[clusterId].getType());
        }
    }

    parentInputSparsityMap = SmallVector<mlir::Value>(numClusters, nceTask.getInputSparsityMap());
    parentInputSETable = SmallVector<mlir::Value>(numClusters, nceTask.getInputStorageElementTable());
}

void VPUIP::arch30xx::ClusterNCERewriter::getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs,
                                                           SmallVector<mlir::Value>& outputBuffs,
                                                           SmallVector<mlir::Value>& parentOutputSparsityMap,
                                                           SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                                           SmallVector<SmallVector<mlir::Value>>& /*outputItiBuffs*/,
                                                           mlir::Location loc, VPUIP::NCEClusterTaskOp nceTask,
                                                           const int64_t numClusters, mlir::OpBuilder& builder) const {
    auto parentInputType = (*nceTask.getInputs().begin()).getType().cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = (*nceTask.getOutputs().begin()).getType().cast<VPUIP::DistributedBufferType>();

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();
    // Elementwise operations may support overlapping for trailing convolution.
    // In that case both input and output modes are OVERLAPPED.
    const auto isEltwise = (nceTask.getTaskType() == VPUIP::NCETaskType::ELTWISE);
    VPUX_THROW_WHEN(!isEltwise && outDistributionMode == VPU::DistributionMode::OVERLAPPED,
                    "No support for NCE output in OVERLAPPED mode.");
    VPUX_THROW_WHEN(!isEltwise && inDistributionMode == VPU::DistributionMode::OVERLAPPED &&
                            outDistributionMode != VPU::DistributionMode::SEGMENTED,
                    "When NCE has input in OVERLAPPED mode then output must be segmented. out mode = '{0}'",
                    VPU::stringifyDistributionMode(outDistributionMode));

    parentOutputSparsityMap = SmallVector<mlir::Value>(numClusters, nceTask.getOutputSparsityMapBuff());

    outputBuffs = VPUIP::getPerClusterComputeBuffers(_ctx, loc, "outputBuff", nceTask.getOutputBuff(), parentOutputType,
                                                     numClusters, builder, true);
    outputSparsityMapBuffs = VPUIP::getPerClusterComputeBuffers(
            _ctx, loc, "outputSparsityMapBuff", nceTask.getOutputSparsityMapBuff(), numClusters, builder, true);

    parentOutputBuffs = SmallVector<mlir::Value>(numClusters, *nceTask.getOutputs().begin());
    if (VPU::isSegmentedOverC(outDistribution)) {
        // for SEG SOK parent output buffers = output buffers
        parentOutputBuffs = outputBuffs;
    }
}

mlir::UnitAttr VPUIP::arch30xx::ClusterNCERewriter::isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const {
    // Only for explicit SEGMENTED mode, not in combination with
    // DUPLICATED or MULTICASTED
    if (inputType.getDistribution().getMode().getValue() != VPU::DistributionMode::SEGMENTED) {
        return nullptr;
    }

    // Segmentation not present on H axis
    const auto numTiles = parseIntArrayAttr<int64_t>(inputType.getDistribution().getNumTiles());
    if (numTiles[Dims4D::Act::H.ind()] <= 1) {
        return nullptr;
    }

    // Segmentation not supported with non NHWC input such as CM Conv
    if (inputType.getDimsOrder() != DimsOrder::NHWC) {
        return nullptr;
    }

    return mlir::UnitAttr::get(_ctx);
}

namespace {

//
// UnrollClusterTilingPass
//

class UnrollClusterTilingPass final : public VPUIP::arch30xx::UnrollClusterTilingBase<UnrollClusterTilingPass> {
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
    const VPUIP::arch30xx::ClusterNCERewriter nceRewriter(&ctx, _log);

    func.walk([&](mlir::Operation* op) {
        mlir::OpBuilder builder(op);

        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            dmaRewriter.matchAndRewrite(nndmaOp, builder);
        } else if (auto taskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            nceRewriter.matchAndRewrite(taskOp, builder);
        }
    });
}

}  // namespace

//
// createUnrollClusterTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch30xx::createUnrollClusterTilingPass(Logger log) {
    return std::make_unique<UnrollClusterTilingPass>(log);
}
