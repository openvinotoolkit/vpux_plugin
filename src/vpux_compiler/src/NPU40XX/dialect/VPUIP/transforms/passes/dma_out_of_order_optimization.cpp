//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {

//
//  DMAOutOfOrderOptimizationPass
//

class DMAOutOfOrderOptimizationPass final :
        public VPUIP::arch40xx::DMAOutOfOrderOptimizationBase<DMAOutOfOrderOptimizationPass> {
public:
    explicit DMAOutOfOrderOptimizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool isMemoryOverlap(VPURT::DeclareBufferOp bufOp1, VPURT::DeclareBufferOp bufOp2);
};

bool DMAOutOfOrderOptimizationPass::isMemoryOverlap(VPURT::DeclareBufferOp bufOp1, VPURT::DeclareBufferOp bufOp2) {
    auto type1 = bufOp1.getBuffer().getType().cast<vpux::NDTypeInterface>();
    auto type2 = bufOp2.getBuffer().getType().cast<vpux::NDTypeInterface>();

    // When buffers are located on CMX and DDR respectively, the two buffers are definitely non-overlapping
    if (type1.getMemoryKind() != type2.getMemoryKind()) {
        return false;
    }

    // When buffers are located on different CMX tiles, the two buffers are definitely non-overlapping
    if (type1.getMemSpace().getIndex().has_value() && type2.getMemSpace().getIndex().has_value() &&
        type1.getMemSpace().getIndex().value() != type2.getMemSpace().getIndex().value()) {
        return false;
    }

    // Need to check the offsets of the buffers to determine whether there is an overlap for other possible cases:
    // 1. when both buffers are on DDR
    // 2. when two buffers are both on CMX but one is DUPLICATED on all CMX tiles
    auto start1 = bufOp1.getByteOffset();
    auto end1 = start1 + type1.getTotalAllocSize().count();
    auto start2 = bufOp2.getByteOffset();
    auto end2 = start2 + type2.getTotalAllocSize().count();

    if (end1 <= start2 || end2 <= start1) {
        return false;
    }

    return true;
}

void DMAOutOfOrderOptimizationPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    // Algorithm will check each independent DMA queues identified by port and channel
    // and will store vector of previous DMA tasks which are to be checked for memory overlap
    // against the next task. If no overlap then ORD bit can be cleared.
    // If task on queue waits for a new barrier or ORD bit cannot be set then vector of tasks will
    // be cleared Vector of tasks per queue will always contain 1 task with some dependency (either
    // barrier or ORD bit set) and if available all subsequent tasks with ORD bit cleared
    vpux::DenseMap<int64_t, SmallVector<VPUIP::DMATypeOpInterface>> dmaPreviousTasksQueueMap;
    vpux::DenseMap<int64_t, mlir::DenseSet<mlir::Value>> dmaPreviousWaitBarrierMap;

    const auto previousDMAsOnQueueWaitsFor = [&](int64_t queueId, mlir::ValueRange waitBarriers) {
        bool dmaAlreadyWaiting = true;
        for (const auto& barrier : waitBarriers) {
            if (dmaPreviousWaitBarrierMap[queueId].find(barrier) != dmaPreviousWaitBarrierMap[queueId].end()) {
                continue;
            }
            dmaPreviousWaitBarrierMap[queueId].insert(barrier);
            dmaAlreadyWaiting = false;
        }

        return dmaAlreadyWaiting;
    };

    // Traverse IR and check all dmaOps
    funcOp->walk([&](VPUIP::DMATypeOpInterface dmaOp) {
        VPUX_THROW_WHEN(dmaOp.getPortAttribute() == nullptr, "DMA op has no port attribute, op - '{0}'",
                        dmaOp->getLoc());

        const auto port = dmaOp.getPortVal();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
        const auto portValue = port.value();

        auto channelType = dmaOp.getChannelType();

        auto dmaQueueTaskId = getDMAQueueIdEncoding(portValue, channelType);

        _log.trace("Identified DMA operation on port - {0} and channel - {1}:'{2}', ", portValue, channelType,
                   dmaOp->getLoc());

        auto taskOp = dmaOp->getParentOfType<VPURT::TaskOp>();
        VPUX_THROW_WHEN(taskOp == nullptr, "Parent must be VPURT::TaskOp");
        const auto waitBarriers = taskOp.getWaitBarriers();
        if (dmaPreviousTasksQueueMap.find(dmaQueueTaskId) == dmaPreviousTasksQueueMap.end()) {
            // First task on this queue. Just store information about it
            dmaPreviousTasksQueueMap[dmaQueueTaskId].push_back(dmaOp);
            dmaPreviousWaitBarrierMap[dmaQueueTaskId].insert(waitBarriers.begin(), waitBarriers.end());
            return;
        }

        // if task has new wait barrier just clean the queue and leave this op as last one
        if (!previousDMAsOnQueueWaitsFor(dmaQueueTaskId, waitBarriers)) {
            dmaPreviousTasksQueueMap[dmaQueueTaskId].clear();
            dmaPreviousTasksQueueMap[dmaQueueTaskId].push_back(dmaOp);
            return;
        }

        auto dmaInputDeclBuff = dmaOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(dmaInputDeclBuff || dmaOp.getInput().getDefiningOp<Const::DeclareOp>(),
                          "DMA op does no supported input input - '{0}'", dmaOp->getLoc());

        bool memOverlap = false;
        if (dmaInputDeclBuff != nullptr) {
            for (auto& prevDmaOp : dmaPreviousTasksQueueMap[dmaQueueTaskId]) {
                auto prevDmaOutputDeclBuff = prevDmaOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

                VPUX_THROW_UNLESS(prevDmaOutputDeclBuff, "DMA op does not have DeclareBufferOp on its output - '{0}'",
                                  prevDmaOp->getLoc());

                // Perform a check on dma inputs and outputs to understand if there is
                // no overlap
                if ((memOverlap = isMemoryOverlap(prevDmaOutputDeclBuff, dmaInputDeclBuff))) {
                    break;
                }
            }
        }

        if (memOverlap) {
            _log.nest().trace("No optimization possible due to overlap in memory ranges");
            dmaPreviousTasksQueueMap[dmaQueueTaskId].clear();
        } else {
            // No overlap. OOO bit can be set
            _log.nest().trace("Out of order execution possible, disable ORD bit");
            dmaOp.setOutOfOrder();
        }
        dmaPreviousTasksQueueMap[dmaQueueTaskId].push_back(dmaOp);
    });

    VPURT::verifyBarrierSlots(funcOp, _log);
}

}  // namespace

//
// createDMAOutOfOrderOptimizationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createDMAOutOfOrderOptimizationPass(Logger log) {
    return std::make_unique<DMAOutOfOrderOptimizationPass>(log);
}
