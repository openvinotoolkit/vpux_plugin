//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"

using namespace vpux;

size_t VPURT::getMinEntry(const BarrierInfo::TaskSet& entries) {
    if (entries.empty()) {
        return std::numeric_limits<size_t>::min();
    }
    return *std::min_element(entries.begin(), entries.end());
}

size_t VPURT::getMaxEntry(const BarrierInfo::TaskSet& entries) {
    if (entries.empty()) {
        return std::numeric_limits<size_t>::max();
    }
    return *std::max_element(entries.begin(), entries.end());
}

// generate FIFOs of Task Ops using index from BarrierInfo
/*
    FIFO-0 | 0, 2, 4
    FIFO-1 | 1, 3, 5, 6
*/
std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> VPURT::getTaskOpQueues(
        mlir::func::FuncOp funcOp, BarrierInfo& barrierInfo, std::optional<VPU::ExecutorKind> targetExecutorKind) {
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> taskOpQueues;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
        if (targetExecutorKind.has_value() && targetExecutorKind.value() != taskQueueType.type) {
            return;
        }
        const auto taskInd = barrierInfo.getIndex(taskOp);
        taskOpQueues[taskQueueType].push_back(taskInd);
    });

    return taskOpQueues;
}

void VPURT::postProcessBarrierOps(mlir::func::FuncOp func) {
    // move barriers to top and erase unused
    auto barrierOps = to_small_vector(func.getOps<VPURT::DeclareVirtualBarrierOp>());
    auto& block = func.getBody().front();

    VPURT::DeclareVirtualBarrierOp prevBarrier = nullptr;
    for (auto& barrierOp : barrierOps) {
        // remove barriers with no use
        if (barrierOp.getBarrier().use_empty()) {
            barrierOp->erase();
            continue;
        }

        // move barriers to top of block
        if (prevBarrier != nullptr) {
            barrierOp->moveAfter(prevBarrier);
        } else {
            barrierOp->moveBefore(&block, block.begin());
        }

        prevBarrier = barrierOp;
    }
}

// It should be called at ending of each pass which may change barriers after SplitExceedingVariantCountBarriersPass
bool VPURT::verifyBarrierSlots(mlir::func::FuncOp func, Logger log) {
    auto barrierSim = VPURT::BarrierSimulator{func};
    if (mlir::failed(barrierSim.checkProducerAndConsumerCount(log))) {
        log.error("verifyBarrierSlots failed");
        return false;
    }
    return true;
}

bool VPURT::verifyOneWaitBarrierPerTask(mlir::func::FuncOp funcOp, Logger log) {
    bool hasOneWaitBarrierPerTask = true;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getWaitBarriers().size() > 1) {
            log.error("Task '{0}' has more then one wait barrier", taskOp);
            hasOneWaitBarrierPerTask = false;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });

    return hasOneWaitBarrierPerTask;
}

// simulate execution of tasks an barriers to generate an order for tasks an barriers which will represent execution
// order tasks and barriers in IR to match that order - required for virtual to physical barrier mapping
// orderByConsumption = false, barriers are ordered based on first producer, orderByConsumption = true: barriers are
// ordered based on consumer
void VPURT::orderExecutionTasksAndBarriers(mlir::func::FuncOp funcOp, BarrierInfo& barrierInfo,
                                           bool orderByConsumption) {
    barrierInfo.updateIR();

    auto taskOpQueues = VPURT::getTaskOpQueues(funcOp, barrierInfo);
    SmallVector<size_t> newTaskOpOrder;
    SmallVector<size_t> newBarrierOrder;
    std::set<size_t> newBarrierOrderSet;

    // initialize front task from each FIFO
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>::iterator> frontTasks;
    for (auto& taskOpQueue : taskOpQueues) {
        frontTasks[taskOpQueue.first] = taskOpQueue.second.begin();
    }

    // check that all FIFOs/Queues reached end
    auto allQueuesReachedEnd = [&]() {
        for (auto& entry : frontTasks) {
            if (entry.second != taskOpQueues[entry.first].end()) {
                return false;
            }
        }
        return true;
    };

    // check that all FIFOs/Queues reached wait for a barrier
    auto allQueuesWaiting = [&]() {
        for (auto& entry : frontTasks) {
            if (entry.second != taskOpQueues[entry.first].end() && barrierInfo.getWaitBarriers(*entry.second).empty()) {
                return false;
            }
        }
        return true;
    };

    // retrieve ready op(s) (that don't have wait barriers) from each FIFO/Queue
    // favour cross FIFO/Queue order e.g. DMA-FIFO-0, DMA-FIFO-1, DMA-FIFO-0, DMA-FIFO-1
    const auto findReadyOps = [&]() {
        SmallVector<size_t> readyOps;
        while (!allQueuesWaiting()) {
            for (auto& entry : frontTasks) {
                if (entry.second != taskOpQueues[entry.first].end() &&
                    barrierInfo.getWaitBarriers(*entry.second).empty()) {
                    readyOps.push_back(*entry.second);
                    ++entry.second;
                }
            }
        }
        return readyOps;
    };

    // reduce producer count for barrier of ready op
    // reset barrier if producer count reaches 0
    const auto removeBarrierProducer = [&](size_t readyOp) {
        const auto updateBarriers = barrierInfo.getUpdateBarriers(readyOp);
        for (const auto& updateBarrier : updateBarriers) {
            auto barrierOp = barrierInfo.getBarrierOpAtIndex(updateBarrier);
            barrierInfo.removeProducer(barrierOp, readyOp);

            // Order the barrier by producer to WA hang issue in runtime
            // TODO: E#109198 find the root cause of runtime hang
            // TODO: E#104375 add the catch such case in compiler simulation
            if (!orderByConsumption && !newBarrierOrderSet.count(updateBarrier)) {
                newBarrierOrder.push_back(updateBarrier);
                newBarrierOrderSet.insert(updateBarrier);
            }

            if (barrierInfo.getBarrierProducers(barrierOp).empty()) {
                if (orderByConsumption) {
                    // barriers will be ordered by order of consumption, each barrier is consumed only once
                    newBarrierOrder.push_back(updateBarrier);
                }
                barrierInfo.resetBarrier(barrierOp);
            }
        }
    };

    // simulate per FIFO execution - all FIFOs must reach end
    while (!allQueuesReachedEnd()) {
        const auto readyOps = findReadyOps();
        // at each step there must be some ready ops
        VPUX_THROW_WHEN(readyOps.empty(), "Failed to simulate execution");

        for (auto& readyOp : readyOps) {
            // tasks will be ordered by order of becoming ready
            newTaskOpOrder.push_back(readyOp);
            removeBarrierProducer(readyOp);
        }
    }

    // ensure number of tasks remains the same
    VPUX_THROW_UNLESS(newTaskOpOrder.size() == barrierInfo.getNumOfTasks(),
                      "Failed to order all tasks, there are {0} tasks, got {1}", barrierInfo.getNumOfTasks(),
                      newTaskOpOrder.size());

    size_t barriersWithNoUse = 0;
    funcOp->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        if (!barrierOp.getBarrier().use_empty()) {
            return;
        }
        ++barriersWithNoUse;
    });

    // ensure number of barriers remains the same
    VPUX_THROW_UNLESS(newBarrierOrder.size() == barrierInfo.getNumOfVirtualBarriers() - barriersWithNoUse,
                      "Failed to order all barriers, there are {0} used barriers, got {1}",
                      barrierInfo.getNumOfVirtualBarriers() - barriersWithNoUse, newBarrierOrder.size());

    // reorder tasks in the IR based on new order
    mlir::Operation* prevTaskOp = nullptr;
    for (auto& opIdx : newTaskOpOrder) {
        mlir::Operation* taskOp = barrierInfo.getTaskOpAtIndex(opIdx);
        if (prevTaskOp != nullptr) {
            taskOp->moveAfter(prevTaskOp);
        } else {
            auto declareBufferOps = to_small_vector(funcOp.getOps<VPURT::DeclareBufferOp>());
            if (declareBufferOps.empty()) {
                auto barrierOps = to_small_vector(funcOp.getOps<VPURT::DeclareVirtualBarrierOp>());
                if (!barrierOps.empty()) {
                    taskOp->moveAfter(barrierOps.back());
                } else {
                    auto& block = funcOp.getBody().front();
                    taskOp->moveBefore(&block, block.begin());
                }
            } else {
                taskOp->moveAfter(declareBufferOps.back());
            }
        }
        prevTaskOp = taskOp;
    }

    // reorder barriers in the IR based on new order
    auto& block = funcOp.getBody().front();
    mlir::Operation* prevBarrier = nullptr;
    for (auto& barrierOpIdx : newBarrierOrder) {
        mlir::Operation* barrierOp = barrierInfo.getBarrierOpAtIndex(barrierOpIdx);
        // move barriers to top of block
        if (prevBarrier != nullptr) {
            barrierOp->moveAfter(prevBarrier);
        } else {
            barrierOp->moveBefore(&block, block.begin());
        }
        prevBarrier = barrierOp;
    }

    // regenerate barrier info based on new order
    barrierInfo = vpux::BarrierInfo{funcOp};
}
