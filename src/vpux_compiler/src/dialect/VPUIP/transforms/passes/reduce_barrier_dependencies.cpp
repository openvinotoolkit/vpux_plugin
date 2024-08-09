//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

using namespace vpux;
namespace {

//
//  ReduceBarrierDependenciesPass
//

// This pass works in 3 steps:
//    1. Add new barrier if consecutive tasks on same FIFO do not have any barrier between them
//    2. leverage optimizeBarriers()
//    3. Remove temporary barriers introduced in step 1
// TODO E#126891: Make this approach part of optimizeBarriers, such that FIFO dependencies are tracked
class ReduceBarrierDependenciesPass final : public VPUIP::ReduceBarrierDependenciesBase<ReduceBarrierDependenciesPass> {
public:
    explicit ReduceBarrierDependenciesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ReduceBarrierDependenciesPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    SmallVector<std::pair<size_t, size_t>> blockRange;

    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    auto getNewBarrier = [&barrierInfo](size_t prevIndex, size_t currentIndex) {
        auto prevTask = barrierInfo.getTaskOpAtIndex(prevIndex);
        mlir::OpBuilder builder(prevTask);
        // postProcessBarrierOps will move the barrier to right place in IR and remove unused barriers
        builder.setInsertionPoint(prevTask);

        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(prevTask.getLoc());
        barrierInfo.addNewBarrier(newBarrier);

        const auto newBarrierIdn = barrierInfo.getIndex(newBarrier);
        barrierInfo.addProducer(newBarrierIdn, prevIndex);
        barrierInfo.addConsumer(newBarrierIdn, currentIndex);

        return newBarrier;
    };

    // Check if two tasks are in the same blockRange
    auto inSameTaskBlock = [&blockRange](size_t task1, size_t task2) {
        return any_of(blockRange.begin(), blockRange.end(), [&](const std::pair<size_t, size_t>& range) {
            return (task1 >= range.first && task1 <= range.second) && (task2 >= range.first && task2 <= range.second);
        });
    };

    std::vector<std::tuple<size_t, size_t, size_t>> addedDeps;
    const auto allQueues = VPURT::getTaskOpQueues(funcOp, barrierInfo);

    // Step 1: Go over the FIFOs, if the two tasks are in the same blockRange and does not have barrier dependency
    // add a temporary dependency between them
    for (const auto& entry : allQueues) {
        if (entry.first.type != VPU::ExecutorKind::DMA_NN) {
            // TODO: E#126579
            continue;
        }
        const auto& fifoTasks = entry.second;
        size_t prevIndex = 0;
        size_t currIndex = prevIndex + 1;
        size_t commonBarrier = -1;

        while (currIndex < fifoTasks.size()) {
            const auto taskOne = fifoTasks[prevIndex];
            const auto taskTwo = fifoTasks[currIndex];
            // Check if the two tasks do not have barrier dependency already and that they're in the same block
            if (!barrierInfo.hasBarrierDependency(taskOne, taskTwo, commonBarrier) &&
                inSameTaskBlock(taskOne, taskTwo)) {
                // add dependency from prevIndex to currIndex
                auto newBarrier = getNewBarrier(taskOne, taskTwo);
                addedDeps.emplace_back(barrierInfo.getIndex(newBarrier), taskOne, taskTwo);
            }
            prevIndex = currIndex;
            currIndex++;
        }
    }

    // Step 2: Leverage optimizeBarriers to optimize different dependency types e.g. optimizeBarriersWithSameProducers,
    // optimizeBarriersProducers, optimizeBarriersConsumers
    barrierInfo.optimizeBarriers(false);

    // Step 3: Remove the temporarily added barrier dependency
    for (const auto& dep : addedDeps) {
        size_t commonBarrier = -1;
        const auto concernedBarrier = std::get<0>(dep);
        const auto prevIdx = std::get<1>(dep);
        const auto currIdx = std::get<2>(dep);

        // If the common barrier between the two tasks after optimization is still the same barrier we added
        // Then remove it as it's not needed because of FIFO dependency
        if (barrierInfo.hasBarrierDependency(prevIdx, currIdx, commonBarrier) && commonBarrier == concernedBarrier) {
            barrierInfo.removeProducer(concernedBarrier, prevIdx);
            barrierInfo.removeConsumer(concernedBarrier, currIdx);
        }
        // MLIR erases the barrier if 0 producer and consumer
    }

    barrierInfo.updateIR();
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(funcOp);
}

}  // namespace

//
// createReduceBarrierDependenciesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createReduceBarrierDependenciesPass(Logger log) {
    return std::make_unique<ReduceBarrierDependenciesPass>(log);
}
