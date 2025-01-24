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

    const auto availableSlots = vpux::VPUIP::getAvailableSlots(maxSlotsSum, maxAvailableSlots);
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

    barrierInfo.splitBarriersWithExceedingVariantCount(availableSlots, maxSlotsSum, maxAvailableSlots);

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
    VPUX_THROW_UNLESS(VPURT::verifyBarrierSlots(func, _log), "Barrier slot count check failed");
}

}  // namespace

//
// createSplitExceedingVariantCountBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSplitExceedingVariantCountBarriersPass(Logger log) {
    return std::make_unique<SplitExceedingVariantCountBarriersPass>(log);
}
