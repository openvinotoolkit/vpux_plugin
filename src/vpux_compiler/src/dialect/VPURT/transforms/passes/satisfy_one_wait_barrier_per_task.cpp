//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

namespace {

class SatisfyOneWaitBarrierPerTaskPass final :
        public VPURT::SatisfyOneWaitBarrierPerTaskBase<SatisfyOneWaitBarrierPerTaskPass> {
public:
    explicit SatisfyOneWaitBarrierPerTaskPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

BarrierInfo::TaskSet getBarriersProducerTasks(const BarrierInfo::TaskSet& waitBarriers, BarrierInfo& barrierInfo) {
    BarrierInfo::TaskSet newBarrierProducers;
    for (auto& waitBarrierInd : waitBarriers) {
        // merge all producers
        const auto& barrierProducers = barrierInfo.getBarrierProducers(waitBarrierInd);
        llvm::set_union(newBarrierProducers, barrierProducers);
    }

    return newBarrierProducers;
}

BarrierInfo::TaskSet getBarriersConsumerTasks(const BarrierInfo::TaskSet& waitBarriers, size_t availableSlots,
                                              BarrierInfo& barrierInfo) {
    // find all parallel consumers
    BarrierInfo::TaskSet parallelConsumers;
    for (auto& waitBarrier : waitBarriers) {
        llvm::set_union(parallelConsumers, barrierInfo.getBarrierConsumers(waitBarrier));
    }

    // find consumers with same wait barriers
    BarrierInfo::TaskSet parallelConsumersSameWaitBarriers;
    size_t parallelConsumerSlotCount = 0;
    for (auto& consumer : parallelConsumers) {
        if (waitBarriers != barrierInfo.getWaitBarriers(consumer)) {
            continue;
        }

        parallelConsumerSlotCount += barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(consumer));
        if (parallelConsumerSlotCount > availableSlots) {
            break;
        }

        parallelConsumersSameWaitBarriers.insert(consumer);
    }

    return parallelConsumersSameWaitBarriers;
}

bool canMergeBarriersForTasks(const BarrierInfo::TaskSet& producers, size_t availableSlots, BarrierInfo& barrierInfo) {
    size_t producerSlotCount = 0;
    for (auto& producerInd : producers) {
        producerSlotCount += barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(producerInd));

        if (producerSlotCount > availableSlots) {
            // exceeding producer slot count
            return false;
        }
    }

    return true;
}

void unlinkTaskFromParallelBarriers(const BarrierInfo::TaskSet& tasks, const BarrierInfo::TaskSet& waitBarriers,
                                    BarrierInfo& barrierInfo) {
    // remove consumer task from parallel barriers
    for (auto& waitBarrierInd : waitBarriers) {
        const auto barrierConsumers = barrierInfo.getBarrierConsumers(waitBarrierInd);
        const auto waitBarrier = barrierInfo.getBarrierOpAtIndex(waitBarrierInd);

        // remove link from parallel barriers
        if (barrierConsumers.size() == tasks.size()) {
            // if only consumer, barrier can be reset
            barrierInfo.resetBarrier(waitBarrier);
        } else {
            for (auto& taskInd : tasks) {
                barrierInfo.removeConsumer(waitBarrier, taskInd);
            }
        }
    }
}

// merge wait barriers if task wait barrier count > 1 by
// creating a new barrier replacing parallel wait barriers
/*
        x..xn    y..ym             x..xn    y..ym
          |        |              /    \   /    \
        Bar0     Bar1     -->   Bar0    Bar1    Bar2
       /    \   /    \           |       |       |
    u..ui   o..oj   a..ak      u..ui   o..oj   a..ak
*/
void mergeLegalParallelProducers(VPURT::TaskOp taskOp, const BarrierInfo::TaskSet& parallelProducers,
                                 const BarrierInfo::TaskSet& parallelConsumers, BarrierInfo& barrierInfo) {
    // create a new barrier for all parallel producers
    mlir::OpBuilder builder(taskOp);
    auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp.getLoc());
    barrierInfo.addNewBarrier(newBarrier);

    // add all legal producers to new barrier
    for (auto& barrierProducer : parallelProducers) {
        barrierInfo.addProducer(newBarrier, barrierProducer);
    }

    // add all parallel consumers with same wait barriers
    for (auto& consumer : parallelConsumers) {
        barrierInfo.addConsumer(newBarrier, consumer);
    }
}

// try to create batches of producers using existing barriers if producers satisfy order constraint
SmallVector<BarrierInfo::TaskSet> createProducerBatches(const BarrierInfo::TaskSet& waitBarriers, size_t availableSlots,
                                                        BarrierInfo& barrierInfo) {
    SmallVector<BarrierInfo::TaskSet> legalBatches;

    auto prevLastUserInd = std::numeric_limits<size_t>::max();
    for (auto& waitBarrierInd : waitBarriers) {
        const auto& barrierProducers = barrierInfo.getBarrierProducers(waitBarrierInd);
        if (legalBatches.empty()) {
            prevLastUserInd = VPURT::getMaxEntry(barrierProducers);
            legalBatches = canMergeBarriersForTasks(barrierProducers, availableSlots, barrierInfo)
                                   ? SmallVector<BarrierInfo::TaskSet>{barrierProducers}
                                   : barrierInfo.createLegalVariantBatches(barrierProducers, availableSlots);
            continue;
        }

        const auto firstUserInd = VPURT::getMinEntry(barrierProducers);
        if (prevLastUserInd >= firstUserInd) {
            // producers do not satisfy order constraint
            return {};
        }

        prevLastUserInd = VPURT::getMaxEntry(barrierProducers);
        auto currentBatchPlusBarrierProducers = legalBatches.back();
        llvm::set_union(currentBatchPlusBarrierProducers, barrierProducers);

        if (canMergeBarriersForTasks(currentBatchPlusBarrierProducers, availableSlots, barrierInfo)) {
            // can add to the same batch
            legalBatches.back() = std::move(currentBatchPlusBarrierProducers);
        } else {
            // need to create a new batch
            if (canMergeBarriersForTasks(barrierProducers, availableSlots, barrierInfo)) {
                legalBatches.push_back(barrierProducers);
            } else {
                legalBatches.append(barrierInfo.createLegalVariantBatches(barrierProducers, availableSlots));
            }
        }
    }

    return legalBatches;
}

// linearize wait barriers if task wait barrier count > 1 by
// linearizing legal batches of parallel wait barriers
/*
        x..xn    y..ym             x..xn
          |        |              /     \
        Bar0     Bar1     -->   Bar0     Bar1
       /    \   /    \           |        |
    u..ui   o..oj   a..ak      u..ui    y..ym
                                       /     \
                                     Bar2   Bar3
                                      |      |
                                    o..oj   a..ak
*/
void linearizeLegalParallelProducers(VPURT::TaskOp taskOp, const BarrierInfo::TaskSet& waitBarriers,
                                     const BarrierInfo::TaskSet& parallelProducers,
                                     const BarrierInfo::TaskSet& parallelConsumers, size_t availableSlots,
                                     BarrierInfo& barrierInfo) {
    // create legal batches of barrier producers
    auto legalBatches = createProducerBatches(waitBarriers, availableSlots, barrierInfo);
    if (legalBatches.empty()) {
        // create new batches of producers that satisfy order constraint
        legalBatches = barrierInfo.createLegalVariantBatches(parallelProducers, availableSlots);
    }

    // last batch is the current task
    BarrierInfo::TaskSet nextBatch = parallelConsumers;
    mlir::OpBuilder builder(taskOp);

    // create a new barrier between batches
    for (const auto& batch : legalBatches | reversed) {
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp.getLoc());
        barrierInfo.addNewBarrier(newBarrier);

        for (auto& barrierProducer : batch) {
            barrierInfo.addProducer(newBarrier, barrierProducer);
        }

        for (auto& barrierConsumer : nextBatch) {
            barrierInfo.addConsumer(newBarrier, barrierConsumer);
        }

        nextBatch = batch;
    }
}

// modify barriers for task such that it is driven by a single wait barrier by merging parallel wait barriers
// or linearizing the producers of parallel wait barriers
bool ensureTaskDrivenBySingleBarrier(size_t taskInd, VPURT::TaskOp taskOp, const BarrierInfo::TaskSet& waitBarriers,
                                     size_t availableSlots, BarrierInfo& barrierInfo, Logger log) {
    log.trace("Got '{0}' parallel wait barriers for '{1}'", waitBarriers.size(), taskInd);

    auto parallelProducers = getBarriersProducerTasks(waitBarriers, barrierInfo);
    auto parallelConsumers = getBarriersConsumerTasks(waitBarriers, availableSlots, barrierInfo);
    if (canMergeBarriersForTasks(parallelProducers, availableSlots, barrierInfo)) {
        log.nest().trace("Can merge '{0}' parallel producers for '{1}'", parallelProducers.size(), taskInd);
        mergeLegalParallelProducers(taskOp, parallelProducers, parallelConsumers, barrierInfo);
    } else {
        log.nest().trace("Have to linearize '{0}' parallel producers for '{1}'", parallelProducers.size(), taskInd);
        linearizeLegalParallelProducers(taskOp, waitBarriers, parallelProducers, parallelConsumers, availableSlots,
                                        barrierInfo);
    }

    for (auto& id : parallelConsumers) {
        log.trace("Unlink parallel barriers from '{0}'", id);
    }
    unlinkTaskFromParallelBarriers(parallelConsumers, waitBarriers, barrierInfo);
    log.trace("Task '{0}', now has one parallel wait barrier", taskInd);
    return true;
}

void SatisfyOneWaitBarrierPerTaskPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();

    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(func);
    _log.trace("There are {0} slots for each barrier",
               maxSlotsSum < maxAvailableSlots ? maxSlotsSum : maxAvailableSlots);

    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    auto availableSlots = std::min(maxAvailableSlots, maxSlotsSum) / 2;
    bool modifiedIR = false;

    // merge parallel wait barriers
    func->walk([&](VPURT::TaskOp taskOp) {
        const auto taskInd = barrierInfo.getIndex(taskOp);
        const auto waitBarriers = barrierInfo.getWaitBarriers(taskInd);
        if (waitBarriers.size() < 2) {
            // valid configuration
            return;
        }

        modifiedIR |= ensureTaskDrivenBySingleBarrier(taskInd, taskOp, waitBarriers, availableSlots, barrierInfo,
                                                      _log.nest());
    });

    if (!modifiedIR) {
        // IR was not modified
        barrierInfo.clearAttributes();
        return;
    }

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
    VPURT::verifyBarrierSlots(func, _log);
}

}  // namespace

//
// createSatisfyOneWaitBarrierPerTaskPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSatisfyOneWaitBarrierPerTaskPass(Logger log) {
    return std::make_unique<SatisfyOneWaitBarrierPerTaskPass>(log);
}
