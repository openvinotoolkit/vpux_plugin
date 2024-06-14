//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

using namespace vpux;

namespace {

class SplitExceedingVariantCountBarriersPass final :
        public VPURT::SplitExceedingVariantCountBarriersBase<SplitExceedingVariantCountBarriersPass> {
public:
    explicit SplitExceedingVariantCountBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// split barriers if producer count > AVAILABLE SLOTS
// will produce ceil(NUM PRODUCERS / AVAILABLE SLOTS) barriers
// Note: if maxSlotsSumLimitEnabled is true, the maxAvailableSlots passed will be half of MAX_VARIANT_SUM not
// MAX_VARIANT_COUNT
/*
    x1  x2  ... xn             x1  ... x256   x257 ... xn
     \  \   /   /               \   |   /        \  |  /
         Bar0         -->          Bar0            Bar1
          |                            \          /
          u0                                u0
*/
void splitBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots, BarrierInfo& barrierInfo,
                           Logger log) {
    // producers will be divided into batches
    const auto& barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
    auto barrierProducersCount = barrierProducers.size();
    log.trace("Got barrier: '{0}' at '{1}' with barrier producer count '{2}'", barrierOp->getName(),
              barrierOp->getLoc(), barrierProducersCount);

    // consumers remain the same for all produced batched barriers
    const auto barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
    // create batches for producers
    auto producerBatches = barrierInfo.createLegalVariantBatches(barrierProducers, availableSlots);

    mlir::OpBuilder builder(barrierOp);

    // create barriers based on batches
    for (const auto& producerBatch : producerBatches) {
        builder.setInsertionPoint(barrierOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(barrierOp->getLoc());
        log.trace("Created new barrier '{0}'", newBarrier);
        barrierInfo.addNewBarrier(newBarrier);

        for (const auto& producer : producerBatch) {
            log.nest().trace("Add producer '{0}' to new barrier, whose slots are {1}", producer,
                             barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(producer)));
            barrierInfo.addProducer(newBarrier, producer);
        }

        for (const auto& consumer : barrierConsumers) {
            log.nest().trace("Add consumer '{0}' to new barrier, whose slots are {1}", consumer,
                             barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(consumer)));
            barrierInfo.addConsumer(newBarrier, consumer);
        }
    }

    barrierInfo.resetBarrier(barrierOp);
    log.trace("Barrier successfully replaced with batch of barriers");
}

// split barriers if consumer count > AVAILABLE SLOTS
// will produce ceil(NUM PRODUCERS / AVAILABLE SLOTS) barriers
// Note: if maxSlotsSumLimitEnabled is true, the maxAvailableSlots passed will be half of MAX_VARIANT_SUM not
// MAX_VARIANT_COUNT
/*
          u0                                u0
          |                            /          \
         Bar0         -->          Bar0            Bar1
     /  /   \   \               /   |   \        /  |  \
    x1  x2  ... xn             x1  ... x256   x257 ... xn
*/
void splitBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots, BarrierInfo& barrierInfo,
                           Logger log) {
    // consumers will be divided into batches
    const auto& barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
    auto barrierConsumersCount = barrierConsumers.size();
    log.trace("Got barrier: '{0}' at '{1}' with barrier consumer count '{2}'", barrierOp->getName(),
              barrierOp->getLoc(), barrierConsumersCount);

    // producers remain the same for all produced batched barriers
    const auto barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
    // crete batches for consumers
    auto consumerBatches = barrierInfo.createLegalVariantBatches(barrierConsumers, availableSlots);

    mlir::OpBuilder builder(barrierOp);

    // create barriers based on batches
    for (const auto& consumerBatch : consumerBatches) {
        builder.setInsertionPoint(barrierOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(barrierOp->getLoc());
        log.trace("Created new barrier '{0}'", newBarrier);
        barrierInfo.addNewBarrier(newBarrier);

        for (const auto& producer : barrierProducers) {
            log.nest().trace("Add producer '{0}' to new barrier, whose slots are {1}", producer,
                             barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(producer)));
            barrierInfo.addProducer(newBarrier, producer);
        }

        for (const auto& consumer : consumerBatch) {
            log.nest().trace("Add consumer '{0}' to new barrier, whose slots are {1}", consumer,
                             barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(consumer)));
            barrierInfo.addConsumer(newBarrier, consumer);
        }
    }

    barrierInfo.resetBarrier(barrierOp);
    log.trace("Barrier successfully replaced with batch of barriers");
}

void SplitExceedingVariantCountBarriersPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();

    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);

    // A hw limit from VPUX40XX - variants sum of one barrier cann't exceed maxVariantSum
    const auto maxSlotsSum = maxVariantSum.hasValue() ? checked_cast<size_t>(maxVariantSum.getValue())
                                                      : VPUIP::getBarrierMaxVariantSum(func);
    bool maxSlotsSumLimitEnabled = false;
    // TODO: we may need more clear way to set maxSlotsSumLimitEnabled after more Arch need this
    if (maxSlotsSum < maxAvailableSlots) {
        maxSlotsSumLimitEnabled = true;
    }
    _log.trace("There are {0} slots for each barrier, means max available variants for each barrier (producers and "
               "consumers)",
               maxSlotsSumLimitEnabled ? maxSlotsSum : maxAvailableSlots);

    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    // TODO: E#107973: allow a unequal/uneven barrier slots assignment
    const auto availableSlots = maxSlotsSumLimitEnabled ? maxSlotsSum / 2 : maxAvailableSlots / 2;

    // verify each task individually satisfies variant count
    func->walk([&](VPURT::TaskOp taskOp) {
        VPUX_THROW_UNLESS(!mlir::isa<VPUIP::NCEClusterTilingOp>(taskOp.getInnerTaskOp()),
                          "Inner task op wrapped with NCEClusterTilingOp '{0}'", taskOp);
        VPUX_THROW_UNLESS(barrierInfo.getNumOfSlotsUsed(taskOp) <= availableSlots,
                          "Task '{0}' uses too many barrier slots '{1}', available slots are '{2}' for producers "
                          "or consumers",
                          taskOp->getLoc(), barrierInfo.getNumOfSlotsUsed(taskOp), availableSlots);
    });

    // Note: profiling parser logic assumes that
    // invariants of the same layer should use the same barriers

    // split barriers if needed
    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto producerNum = barrierInfo.getProducerSlotCount(barrierOp);
        auto consumerNum = barrierInfo.getConsumerSlotCount(barrierOp);
        if (maxSlotsSumLimitEnabled && (producerNum + consumerNum <= maxSlotsSum)) {
            return;
        }

        if (producerNum <= availableSlots) {
            // valid producer configuration - barrier is legal
            return;
        }

        splitBarrierProducers(barrierOp, availableSlots, barrierInfo, _log);
    });

    // split barriers if needed
    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto producerNum = barrierInfo.getProducerSlotCount(barrierOp);
        auto consumerNum = barrierInfo.getConsumerSlotCount(barrierOp);
        if (maxSlotsSumLimitEnabled && (producerNum + consumerNum <= maxSlotsSum)) {
            return;
        }

        if (consumerNum <= availableSlots) {
            // valid consumer configuration - barrier is legal
            return;
        }

        splitBarrierConsumers(barrierOp, availableSlots, barrierInfo, _log);
    });

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
    VPURT::verifyBarrierSlots(func, _log);
}

}  // namespace

//
// createSplitExceedingVariantCountBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSplitExceedingVariantCountBarriersPass(Logger log) {
    return std::make_unique<SplitExceedingVariantCountBarriersPass>(log);
}
