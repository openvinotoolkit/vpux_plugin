//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

using namespace vpux;

namespace {
class AddFetchOpsPass : public VPUMI40XX::AddFetchOpsBase<AddFetchOpsPass> {
public:
    explicit AddFetchOpsPass(Logger log): _log(log) {
    }

private:
    Logger _log;
    void safeRunOnFunc() final;
};

void reindexList(VPUMI40XX::MappedInferenceOp mpi, VPURegMapped::FetchTaskOp firstFetch, const size_t fetchTaskTileIdx,
                 const size_t fetchTaskListIdx) {
    auto ctx = mpi.getContext();
    auto oldHead = mpi.getListHead(VPURegMapped::TaskType::DMA, fetchTaskTileIdx, fetchTaskListIdx);
    oldHead.replaceUsesWithIf(firstFetch, [](mlir::OpOperand& opOperand) {
        return mlir::isa<VPUMI40XX::OpRanges>(opOperand.getOwner());
    });
    mpi.getListHeadMutable(VPURegMapped::TaskType::DMA, fetchTaskTileIdx, fetchTaskListIdx).assign(firstFetch);
    auto newCount = VPUMI40XX::reindexList(mlir::cast<VPURegMapped::TaskOpInterface>(
            mpi.getListHead(VPURegMapped::TaskType::DMA, fetchTaskTileIdx, fetchTaskListIdx).getDefiningOp()));

    auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(mpi.getDmaCount());
    dmaCount[fetchTaskTileIdx][fetchTaskListIdx] = newCount;
    mpi.setDmaCountAttr(getIntArrayOfArray(ctx, dmaCount));
}

bool dmaComp(mlir::Operation* lhs, mlir::Operation* rhs) {
    auto lhsDma = mlir::cast<VPUMI40XX::NNDMAOp>(lhs);
    auto rhsDma = mlir::cast<VPUMI40XX::NNDMAOp>(rhs);

    return lhsDma.getType().getValue() < rhsDma.getType().getValue();
}

VPUMI40XX::NNDMAOp findLastDma(llvm::SmallSetVector<VPUMI40XX::ConfigureBarrierOp, 16>& barrs, uint32_t listIdx,
                               uint32_t tileIdx) {
    llvm::SmallVector<VPUMI40XX::NNDMAOp> directDmas;

    for (auto waitBarr : barrs) {
        auto validDMA = [&listIdx, &tileIdx, &waitBarr](mlir::Operation* op) -> bool {
            auto dma = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(op);
            return dma && (dma.getType().getListIdx() == listIdx) && (dma.getType().getTileIdx() == tileIdx) &&
                   llvm::count_if(dma.getUpdateBarriers(), [&waitBarr](mlir::Value updateBarr) {
                       return updateBarr == waitBarr;
                   });
        };

        auto filteredRange = waitBarr.getResult().getUsers() | vpux::filtered(std::move(validDMA));
        auto filteredVector = to_small_vector(filteredRange);
        auto maxDma = vpux::max_element(filteredVector, dmaComp);

        if (!filteredVector.empty()) {
            directDmas.push_back(mlir::cast<VPUMI40XX::NNDMAOp>(*maxDma));
        }
    }

    if (directDmas.size()) {
        return *std::max_element(directDmas.begin(), directDmas.end(), dmaComp);
    } else {
        return nullptr;
    }
}

VPUMI40XX::NNDMAOp findFirstDma(llvm::SmallSetVector<VPUMI40XX::ConfigureBarrierOp, 16>& barrs, uint32_t listIdx,
                                uint32_t tileIdx) {
    llvm::SmallVector<VPUMI40XX::NNDMAOp> directDmas;

    for (auto updateBarr : barrs) {
        auto validDMA = [&listIdx, &tileIdx, &updateBarr](mlir::Operation* op) {
            auto dma = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(op);

            return dma && (dma.getType().getListIdx() == listIdx) && (dma.getType().getTileIdx() == tileIdx) &&
                   (llvm::count_if(dma.getWaitBarriers(), [&updateBarr](mlir::Value waitBarr) {
                        return waitBarr == updateBarr;
                    }) > 0);
        };

        auto filteredRange = updateBarr.getResult().getUsers() | vpux::filtered(std::move(validDMA));
        auto filteredVector = to_small_vector(filteredRange);
        auto minDma = vpux::min_element(filteredVector, dmaComp);

        if (!filteredVector.empty()) {
            directDmas.push_back(mlir::cast<VPUMI40XX::NNDMAOp>(*minDma));
        }
    }

    if (directDmas.size())
        return *std::min_element(directDmas.begin(), directDmas.end(), dmaComp);
    else {
        return nullptr;
    }
}

VPUMI40XX::NNDMAOp findFirstDma(VPURegMapped::ExecutionGroupOp op, uint32_t listIdx, uint32_t tileIdx) {
    llvm::SmallSetVector<VPUMI40XX::ConfigureBarrierOp, 16> barriers;

    for (auto barr : op.getUpdateBarriers()) {
        barriers.insert(mlir::cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp()));
    }

    do {
        auto dma = findFirstDma(barriers, listIdx, tileIdx);
        if (dma) {
            return dma;
        }

        llvm::SmallSetVector<VPUMI40XX::ConfigureBarrierOp, 16> newLevel;
        for (auto barr : barriers) {
            for (auto user : barr.getResult().getUsers()) {
                auto dependentBarr = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(user);
                if (dependentBarr && (llvm::count_if(dependentBarr.getDependencies(), [&barr](mlir::Value dependency) {
                                          auto dependencyBarr = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(
                                                  dependency.getDefiningOp());
                                          return dependencyBarr == barr;
                                      }) > 0)) {
                    newLevel.insert(dependentBarr);
                }
            }
        }

        barriers = newLevel;

    } while (barriers.size());

    return nullptr;
}

VPUMI40XX::NNDMAOp findLastDma(VPURegMapped::ExecutionGroupOp op, uint32_t listIdx, uint32_t tileIdx) {
    llvm::SmallSetVector<VPUMI40XX::ConfigureBarrierOp, 16> barriers;

    for (auto barr : op.getWaitBarriers()) {
        barriers.insert(mlir::cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp()));
    }

    do {
        auto dma = findLastDma(barriers, listIdx, tileIdx);
        if (dma) {
            return dma;
        }
        llvm::SmallSetVector<VPUMI40XX::ConfigureBarrierOp, 16> newLevel;
        for (auto barr : barriers) {
            for (auto dependency : barr.getDependencies()) {
                newLevel.insert(mlir::cast<VPUMI40XX::ConfigureBarrierOp>(dependency.getDefiningOp()));
            }
        }

        barriers = newLevel;

    } while (barriers.size());

    return nullptr;
}

// initial PRIMITIVE implementation that will not try to smartly insert anything, but try to achieve WLM the simplest
// way possible AKA: find each DPU, and BEFORE EACH DPU task, will insert one ENQUEUE OP, and connecting it's barrier
// lots of assumptions on this pass, will try to summarize them
// - assume every invariant has a consumer barrier     ----         both of them conditions checked in find first\last
// DMA
// - assume that said consumer barrier has a DMA producer  ----
// - assume that the above is true for each tile
// - assume all DMA-s are in the IR AFTER all DPU tasks - guarantee due to current reordering passes

mlir::LogicalResult addFetchTasks(VPUMI40XX::MappedInferenceOp mpi, const int64_t tilesCount,
                                  const size_t fetchTaskTileIdx, const size_t fetchTaskListIdx,
                                  const VPURegMapped::TaskType taskType, Logger log) {
    auto ctx = mpi.getContext();
    auto builder = mlir::OpBuilder(mpi);
    auto dummyIndexType = VPURegMapped::IndexType::get(ctx, fetchTaskTileIdx, fetchTaskListIdx, 0);

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        auto startingInvValue = mpi.getListHead(taskType, tileIdx);
        // theoretically there can be cases where we run for 6 tiles, but only 4 tiles have Variants associated
        if (!startingInvValue)
            continue;

        auto firstGroup = mlir::dyn_cast_or_null<VPURegMapped::ExecutionGroupOp>(startingInvValue.getDefiningOp());
        if (!firstGroup)
            continue;

        // the first task has the special condition that it's wlm will be added as first DMA with a guaranteed
        // barrier consumption event by a dummy DMA

        auto firstDma = mpi.getListHead(VPURegMapped::TaskType::DMA, fetchTaskTileIdx, fetchTaskListIdx);
        auto firstDmaTaskOp = mlir::cast<VPURegMapped::TaskOpInterface>(firstDma.getDefiningOp());

        builder.setInsertionPoint(firstDma.getDefiningOp());

        auto firstFetch = builder.create<VPURegMapped::FetchTaskOp>(
                firstGroup.getLoc(), dummyIndexType,
                nullptr,  // no previous
                firstGroup.getStartIndexes()[0], firstGroup.getEndIndexes()[0], firstGroup.getStartIndexes()[1],
                firstGroup.getEndIndexes()[1]);
        firstDmaTaskOp.setPreviousTask(firstFetch);

        VPURegMapped::ExecutionGroupOp parentGroup = firstGroup;
        VPURegMapped::ExecutionGroupOp grandParentGroup;

        // iterate over remaining
        auto travelingGroup = VPUMI40XX::getNextGroup(firstGroup);
        while (travelingGroup) {
            // In case grandParentGroup is nullptr that means the parentGroup is only second Execution group and we can
            // allow it to use firstDMA as insertion point for Fetch This is because the first execution group goes to
            // ping section in metadata
            auto lastDma = findLastDma(parentGroup, 0, 0);
            auto parentGroupLastDma = lastDma != nullptr ? lastDma : firstDmaTaskOp;

            // Use getOperation() to compare the underlying operations safely
            if (parentGroupLastDma.getOperation() == firstDmaTaskOp.getOperation() && grandParentGroup) {
                log.warning("Could not find a maximum DMA producer for parentGroup {0}", parentGroup);
                return mlir::failure();
            }

            auto insertionDma = grandParentGroup ? findFirstDma(grandParentGroup, 0, 0) : parentGroupLastDma;
            if (grandParentGroup && insertionDma == nullptr) {
                log.warning("Could not find a minimum DMA consumer for grandParentGroup {0}", grandParentGroup);
                return mlir::failure();
            }

            lastDma = findLastDma(travelingGroup, 0, 0);
            if (lastDma == nullptr) {
                log.warning("Could not find a maximum DMA producer for travelingGroup {0}", travelingGroup);
                return mlir::failure();
            }

            // set the insertion point after the finalDMA
            builder.setInsertionPointAfter(insertionDma.getOperation());
            auto fetchTaskOp = builder.create<VPURegMapped::FetchTaskOp>(
                    travelingGroup.getLoc(), dummyIndexType, insertionDma.getResult(),
                    travelingGroup.getStartIndexes()[0], travelingGroup.getEndIndexes()[0],
                    travelingGroup.getStartIndexes()[1], travelingGroup.getEndIndexes()[1]);

            // set the previousIdx to the fetchOp
            insertionDma.getResult().replaceAllUsesExcept(fetchTaskOp.getIndex(), fetchTaskOp.getOperation());

            // move to next set
            grandParentGroup = parentGroup;
            parentGroup = travelingGroup;

            travelingGroup = VPUMI40XX::getNextGroup(travelingGroup);
        }
        reindexList(mpi, firstFetch, fetchTaskTileIdx, fetchTaskListIdx);
    }

    return mlir::success();
}

void AddFetchOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    auto mpi = VPUMI40XX::getMPI(netFunc);

    const size_t DMA_DDR2CMX_LISTIDX = 0;
    const size_t DMA_WLM_TILEIDX = 0;  // all WLM dma's should be on tile0 for now;

    if (mlir::failed(addFetchTasks(mpi, tilesCount, DMA_WLM_TILEIDX, DMA_DDR2CMX_LISTIDX,
                                   VPURegMapped::TaskType::DPUInvariant, _log))) {
        vpux::VPUIP::setWlmStatus(parentModule, vpux::VPUIP::WlmStatus::FAILED);
        signalPassFailure();
        return;
    }
    if (mlir::failed(addFetchTasks(mpi, tilesCount, DMA_WLM_TILEIDX, DMA_DDR2CMX_LISTIDX,
                                   VPURegMapped::TaskType::ActKernelRange, _log))) {
        vpux::VPUIP::setWlmStatus(parentModule, vpux::VPUIP::WlmStatus::FAILED);
        signalPassFailure();
        return;
    }

    return;
}

}  // namespace

//
// createAddFetchOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddFetchOpsPass(Logger log) {
    return std::make_unique<AddFetchOpsPass>(log);
}
