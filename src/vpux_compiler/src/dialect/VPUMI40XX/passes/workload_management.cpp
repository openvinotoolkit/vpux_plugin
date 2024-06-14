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
class WorkloadManagementPass : public VPUMI40XX::WorkloadManagementBase<WorkloadManagementPass> {
public:
    explicit WorkloadManagementPass(Logger log): _log(log) {
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

VPUMI40XX::NNDMAOp createWLMSyncDMA(mlir::OpBuilder& builder, mlir::ValueRange waitBarriers,
                                    mlir::ValueRange updateBarriers, VPUMI40XX::NNDMAOp previousDMA) {
    auto ctx = builder.getContext();
    auto dummyDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_sync_dma"));
    auto zeroAttr = vpux::getIntAttr(ctx, 0);
    auto indexType = previousDMA.getIndexType();
    auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx, /*numPlane*/ zeroAttr, /*len*/ zeroAttr,
                                                           /*srcWidth*/ zeroAttr, /*srcStride*/ zeroAttr,
                                                           /*srcPlaneStride*/ zeroAttr, /*dstWidth*/ zeroAttr,
                                                           /*dstStride*/ zeroAttr, /*dstPlaneStride*/
                                                           zeroAttr);
    auto inputBuff = VPUIP::createDummyBuffer(builder);
    auto outputBuff = VPUIP::createDummyBuffer(builder);
    llvm::SmallVector<mlir::Value> dmaResults = {outputBuff};
    auto newDma = builder.create<VPUMI40XX::NNDMAOp>(
            dummyDmaLoc, indexType, nullptr, inputBuff, dmaResults, previousDMA.getResult(),
            mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers), 0, 0, false, false, false, 0,
            VPUIP::DMAAccMode::DISABLE, nullptr, dmaDescriptorAttr, nullptr, nullptr, false, nullptr);

    // Adjust the producer count for update barriers
    std::for_each(updateBarriers.begin(), updateBarriers.end(), [](auto barr) {
        auto barrier = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp());
        barrier.setProducerCount(barrier.getProducerCount().value() + 1);
    });
    // Adjust the consumer count for update barriers
    std::for_each(waitBarriers.begin(), waitBarriers.end(), [](auto barr) {
        auto barrier = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp());
        barrier.setConsumerCount(barrier.getConsumerCount().value() + 1);
    });
    return newDma;
}

VPUMI40XX::NNDMAOp getNextDma(VPURegMapped::FetchTaskOp fetch) {
    auto nextDma = [](VPURegMapped::TaskOpInterface taskOp) -> VPURegMapped::TaskOpInterface {
        auto dmaIt = llvm::find_if(taskOp.getResult().getUsers(), [&taskOp](mlir::Operation* op) {
            auto dma = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
            return dma && dma.getPreviousTask() && dma.getPreviousTask().getResult() == taskOp.getResult();
        });
        auto res = dmaIt != taskOp.getResult().getUsers().end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*dmaIt)
                                                                : nullptr;
        return res;
    };

    VPURegMapped::TaskOpInterface res = mlir::cast<VPURegMapped::TaskOpInterface>(fetch.getOperation());
    do {
        res = nextDma(res);
    } while (res && mlir::isa<VPURegMapped::FetchTaskOp>(res));

    return mlir::cast<VPUMI40XX::NNDMAOp>(res.getOperation());
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

    VPUX_THROW("Could not find a minimum DMA consumer for {0}", op);
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

    VPUX_THROW("Could not find a maximum DMA producer for {0}", op);
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

void addFetchTasks(VPUMI40XX::MappedInferenceOp mpi, const int64_t tilesCount, const size_t fetchTaskTileIdx,
                   const size_t fetchTaskListIdx, const VPURegMapped::TaskType taskType, Logger log) {
    auto ctx = mpi.getContext();
    auto dmaComp = [](VPUMI40XX::NNDMAOp lhs, VPUMI40XX::NNDMAOp rhs) {
        return lhs.getType().getValue() < rhs.getType().getValue();
    };
    auto builder = mlir::OpBuilder(mpi);

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
        auto firstDmaOp = mlir::cast<VPURegMapped::TaskOpInterface>(firstDma.getDefiningOp());

        builder.setInsertionPoint(firstDma.getDefiningOp());

        auto firstFetch = builder.create<VPURegMapped::FetchTaskOp>(
                firstGroup.getLoc(), VPURegMapped::IndexType::get(ctx, fetchTaskTileIdx, fetchTaskListIdx, 0),
                nullptr,  // no previous
                firstGroup.getStartIndexes()[0], firstGroup.getEndIndexes()[0], firstGroup.getStartIndexes()[1],
                firstGroup.getEndIndexes()[1]);
        firstDmaOp.setPreviousTask(firstFetch);

        VPURegMapped::FetchTaskOp parentFetch = firstFetch;

        VPURegMapped::ExecutionGroupOp parentGroup = firstGroup;
        VPURegMapped::ExecutionGroupOp grandParentGroup;

        // iterate over remaining
        auto travelingGroup = VPUMI40XX::getNextGroup(firstGroup);
        while (travelingGroup) {
            auto parentFetchDma = getNextDma(parentFetch);
            auto firstGrandParentDma = grandParentGroup ? findFirstDma(grandParentGroup, 0, 0) : nullptr;
            auto insertionDma = grandParentGroup ? std::max(firstGrandParentDma, parentFetchDma, dmaComp)
                                                 : findLastDma(parentGroup, 0, 0);

            auto lastDma = findLastDma(travelingGroup, 0, 0);
            if (lastDma.getType().getValue() <= insertionDma.getType().getValue() && grandParentGroup) {
                builder.setInsertionPointAfter(insertionDma.getOperation());
                auto syncDma = createWLMSyncDMA(builder, {}, lastDma.getUpdateBarriers(), insertionDma);
                insertionDma.getResult().replaceAllUsesExcept(syncDma.getIndex(), syncDma.getOperation());

                // Reindex the list incase we added a new DMA else we would never see new position of lastDma in the
                // list
                reindexList(mpi, firstFetch, fetchTaskTileIdx, fetchTaskListIdx);
                lastDma = findLastDma(travelingGroup, 0, 0);
                log.trace("Inserted additional sync DMAs '{0}' after '{1}'", syncDma, insertionDma);
            }

            // In case we still end up with this case, throw exception
            VPUX_THROW_WHEN(lastDma.getType().getValue() <= insertionDma.getType().getValue(),
                            "Could not find a suitable DMA location to fetch group {0}", travelingGroup);

            // set the insertion point after the finalDMa
            builder.setInsertionPointAfter(insertionDma.getOperation());
            auto fetchTaskOp = builder.create<VPURegMapped::FetchTaskOp>(
                    travelingGroup.getLoc(), insertionDma.getIndexType(), insertionDma.getIndex(),
                    travelingGroup.getStartIndexes()[0], travelingGroup.getEndIndexes()[0],
                    travelingGroup.getStartIndexes()[1], travelingGroup.getEndIndexes()[1]);

            // set the previousIdx to the fetchOp
            insertionDma.getResult().replaceAllUsesExcept(fetchTaskOp.getIndex(), fetchTaskOp.getOperation());

            // move to next set
            parentFetch = fetchTaskOp;
            grandParentGroup = parentGroup;
            parentGroup = travelingGroup;

            travelingGroup = VPUMI40XX::getNextGroup(travelingGroup);
        }
        reindexList(mpi, firstFetch, fetchTaskTileIdx, fetchTaskListIdx);
    }

    return;
}

void WorkloadManagementPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();

    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    auto mpi = VPUMI40XX::getMPI(netFunc);

    const size_t DMA_DDR2CMX_LISTIDX = 0;
    const size_t DMA_WLM_TILEIDX = 0;  // all WLM dma's should be on tile0 for now;

    addFetchTasks(mpi, tilesCount, DMA_WLM_TILEIDX, DMA_DDR2CMX_LISTIDX, VPURegMapped::TaskType::DPUInvariant, _log);
    addFetchTasks(mpi, tilesCount, DMA_WLM_TILEIDX, DMA_DDR2CMX_LISTIDX, VPURegMapped::TaskType::ActKernelRange, _log);

    return;
}

}  // namespace

//
// createWorkloadManagementPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createWorkloadManagementPass(Logger log) {
    return std::make_unique<WorkloadManagementPass>(log);
}
