//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

//
// Constructor
//

vpux::BarrierInfo::BarrierInfo(mlir::func::FuncOp func)
        : _log(Logger::global().nest("barrier-info", 0)),
          _func(func),
          _taskIndexAttrName(mlir::StringAttr::get(func->getContext(), "task-index")),
          _barrierIndexAttrName(mlir::StringAttr::get(func->getContext(), "barrier-index")),
          _syncTaskAttrName(mlir::StringAttr::get(func->getContext(), "sync-task")) {
    buildBarrierMaps(func);
}

vpux::BarrierInfo::BarrierInfo(): _log(Logger::global().nest("barrier-info", 0)), _func(nullptr) {
}

//
// clearAttributes
//

void vpux::BarrierInfo::clearAttributes() {
    auto removeAttributeFromRange = [](mlir::StringAttr attrName, auto range) {
        for (auto op : range) {
            VPUX_THROW_UNLESS(op->hasAttr(attrName), "Remove: attribute '{0}' was not set for '{1}' operation at '{2}'",
                              attrName, op->getName(), op->getLoc());
            op->removeAttr(attrName);
        }
    };

    removeAttributeFromRange(_taskIndexAttrName, _allTaskOps);
    removeAttributeFromRange(_barrierIndexAttrName, _allBarrierOps);
}

//
// getIndex (TaskOp)
//

uint32_t vpux::BarrierInfo::getIndex(VPURT::TaskOp taskOp) const {
    const auto attr = taskOp->getAttrOfType<mlir::IntegerAttr>(_taskIndexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                      _taskIndexAttrName, taskOp->getName(), taskOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

//
// getIndex (DeclareVirtualBarrierOp)
//

uint32_t vpux::BarrierInfo::getIndex(VPURT::DeclareVirtualBarrierOp barrierOp) const {
    const auto attr = barrierOp->getAttrOfType<mlir::IntegerAttr>(_barrierIndexAttrName);
    VPUX_THROW_UNLESS(attr != nullptr, "Get: attribute '{0}' was not set for '{1}' operation at '{2}'",
                      _barrierIndexAttrName, barrierOp->getName(), barrierOp->getLoc());

    return checked_cast<uint32_t>(attr.getValue().getZExtValue());
}

//
// getTaskOpAtIndex
//

VPURT::TaskOp vpux::BarrierInfo::getTaskOpAtIndex(size_t opIdx) const {
    VPUX_THROW_UNLESS(_allTaskOps.size() > opIdx, "Task: Invalid index '{0}' for _allTaskOps", opIdx);
    VPUX_THROW_UNLESS(getIndex(_allTaskOps[opIdx]) == opIdx, "Task: Index not matching '{0}'", opIdx);
    return _allTaskOps[opIdx];
}

//
// getBarrierOpAtIndex
//

VPURT::DeclareVirtualBarrierOp vpux::BarrierInfo::getBarrierOpAtIndex(size_t opIdx) const {
    VPUX_THROW_UNLESS(_allBarrierOps.size() > opIdx, "Barrier: Invalid index '{0}' for _allBarrierOps", opIdx);
    VPUX_THROW_UNLESS(getIndex(_allBarrierOps[opIdx]) == opIdx, "Barrier: Index not matching '{0}' '{1}'", opIdx,
                      _allBarrierOps[opIdx]);
    return _allBarrierOps[opIdx];
}

//
// getWaitBarriers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getWaitBarriers(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd <= _taskWaitBarriers.size(), "Task not found in _taskWaitBarriers, '{0}'", taskInd);
    return _taskWaitBarriers[taskInd];
}

//
// getUpdateBarriers (by Idn)
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getUpdateBarriers(size_t taskInd) {
    VPUX_THROW_UNLESS(taskInd < _taskUpdateBarriers.size(), "Task not found in _taskUpdateBarriers, '{0}'", taskInd);
    return _taskUpdateBarriers[taskInd];
}

//
// getBarrierProducers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierProducers(size_t barrierInd) {
    VPUX_THROW_UNLESS(barrierInd <= _barrierProducerMap.size(), "Barrier not found in _barrierProducerMap, '{0}'",
                      barrierInd);
    return _barrierProducerMap[barrierInd];
}

//
// getBarrierConsumers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierConsumers(size_t barrierInd) {
    VPUX_THROW_UNLESS(barrierInd <= _barrierConsumerMap.size(), "Barrier not found in _barrierConsumerMap, '{0}'",
                      barrierInd);
    return _barrierConsumerMap[barrierInd];
}

//
// getBarrierProducers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp) {
    auto barrierInd = getIndex(barrierOp);
    return getBarrierProducers(barrierInd);
}

//
// getBarrierConsumers
//

BarrierInfo::TaskSet& vpux::BarrierInfo::getBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp) {
    auto barrierInd = getIndex(barrierOp);
    return getBarrierConsumers(barrierInd);
}

//
// getNumOfSlotsUsed
//

size_t vpux::BarrierInfo::getNumOfSlotsUsed(VPURT::TaskOp op) {
    // This function returns the number of variants used by a task for a barrier.
    // On VPU H/W, a NCE task is executed (across multiple DPUs) via workloads descriptors (known as variants).
    // Each variant must update the barrier to signal that it is complete.
    // An NCE task may have multiple workloads descriptors (which are generated in the NCE DPU workloads pass).
    // Therefore, the number of variants must be verified as they will all update a barrier and
    // contribute to the architecture specific MAX VARIANT COUNT that a barrier has.
    // A DMA/UPA does not have variants, therefore they always just requires 1 producer slot to a barrier.

    if (op.getExecutorKind() == VPU::ExecutorKind::DPU) {
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op.getInnerTaskOp());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op.getInnerTaskOp())) {
            auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
            return std::distance(swKernelRun.begin(), swKernelRun.end());
        }
        return 1;
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::DMA_NN || op.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA ||
        op.getExecutorKind() == VPU::ExecutorKind::M2I) {
        return 1;
    }

    // TODO: Analyze and define executor type for funcOp - E#117624
    if (op.getExecutorKind() == VPU::ExecutorKind::UNKNOWN) {
        return 1;
    }

    VPUX_THROW("Unsupported executor: {0}", op.getExecutorKind());
}

size_t vpux::BarrierInfo::getNumOfSlotsUsedByTask(VPURT::TaskOp op) const {
    return getNumOfSlotsUsed(op);
}

//
// getProducerSlotCount
//

size_t vpux::BarrierInfo::getProducerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp) {
    size_t producerSlotCount = 0;
    for (const auto& producer : getBarrierProducers(barrierOp)) {
        producerSlotCount += getNumOfSlotsUsed(getTaskOpAtIndex(producer));
    }
    return producerSlotCount;
}

//
// getConsumerSlotCount
//

size_t vpux::BarrierInfo::getConsumerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp) {
    size_t consumerSlotCount = 0;
    for (const auto& consumer : getBarrierConsumers(barrierOp)) {
        consumerSlotCount += getNumOfSlotsUsed(getTaskOpAtIndex(consumer));
    }
    return consumerSlotCount;
}

//
// getNumOfVirtualBarriers
//

size_t vpux::BarrierInfo::getNumOfVirtualBarriers() const {
    return _allBarrierOps.size();
}

size_t vpux::BarrierInfo::getBarrierMaxVariantSum() const {
    return VPUIP::getBarrierMaxVariantSum(_func);
}

//
// getNumOfTasks
//

size_t vpux::BarrierInfo::getNumOfTasks() const {
    return _allTaskOps.size();
}

//
// resizeBitMap
//

void vpux::BarrierInfo::resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits) {
    bitMap.resize(length);
    for (auto& bit : bitMap) {
        bit.resize(bits);
    }
}

void vpux::BarrierInfo::resetBitMap(SmallVector<llvm::BitVector>& bitMap) {
    for (auto& bit : bitMap) {
        bit.reset();
    }
}

SmallVector<size_t> vpux::BarrierInfo::getBarriersForTaskBlock(size_t blockInd, bool blockStartSyncPoint,
                                                               bool blockEndSyncPoint, bool updateBarriers) const {
    VPUX_THROW_WHEN(blockInd >= getControlGraphBlockCount(), "Invalid task block index ({0})", blockInd);
    auto [blockStartInd, blockEndInd] = getControlGraphBlockTaskRange(blockInd, blockStartSyncPoint, blockEndSyncPoint);

    std::set<size_t> barrierInd;
    if (updateBarriers) {
        for (unsigned taskInd = blockStartInd; taskInd <= blockEndInd; ++taskInd) {
            for (auto barIdx : _taskUpdateBarriers[taskInd]) {
                barrierInd.insert(barIdx);
            }
        }
    } else {
        for (unsigned taskInd = blockStartInd; taskInd <= blockEndInd; ++taskInd) {
            for (auto barIdx : _taskWaitBarriers[taskInd]) {
                barrierInd.insert(barIdx);
            }
        }
    }
    SmallVector<size_t> barrierIndVec(std::make_move_iterator(barrierInd.begin()),
                                      std::make_move_iterator(barrierInd.end()));
    return barrierIndVec;
}

size_t vpux::BarrierInfo::getControlGraphBlockIndex(size_t taskInd) const {
    // assumes uniform distribution of sync-points across control graph.
    return _controlGraphBlockSize == 0 ? 0 : taskInd / _controlGraphBlockSize;
}

//
//  producersControlsAllConsumers
//

bool vpux::BarrierInfo::producersControlsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                                      const TaskSet& origConsumers,
                                                      ArrayRef<TaskSet> origWaitBarriersMap) {
    // Get new consumers not in original consumers

    auto consumersWithoutDirectControl = llvm::set_difference(newConsumers, origConsumers);
    if (consumersWithoutDirectControl.empty()) {
        return true;
    }
    if (origProducers.empty()) {
        return false;
    }

    auto consumerHasImplicitTaskQueueType = inImplicitQueueTypeDependencyList(consumersWithoutDirectControl);
    if (!consumerHasImplicitTaskQueueType) {
        return false;
    }
    auto producerHasImplicitTaskQueueType = inImplicitQueueTypeDependencyList(origProducers);
    if (!producerHasImplicitTaskQueueType) {
        return false;
    }

    if (*std::max_element(origProducers.begin(), origProducers.end()) >=
        *std::min_element(consumersWithoutDirectControl.begin(), consumersWithoutDirectControl.end())) {
        return false;
    }

    auto anyProducerCanRunParallelWithConsumer = llvm::any_of(origProducers, [&](const auto& producer) {
        return llvm::any_of(consumersWithoutDirectControl, [&](const auto& consumer) {
            return origWaitBarriersMap[producer] == origWaitBarriersMap[consumer];
        });
    });
    return !anyProducerCanRunParallelWithConsumer;
}

//
// inImplicitQueueTypeDependencyList
//

bool vpux::BarrierInfo::inImplicitQueueTypeDependencyList(const TaskSet& taskList) {
    // ensure that _taskQueueTypeMap is build at given time with buildTaskControlMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");
    auto allTasksAreInImplicitQueueTypeDependencyList = llvm::all_of(taskList, [&](const auto& taskInd) {
        for (const auto& item : _taskQueueTypeMap) {
            if (item.second.test(taskInd)) {
                return true;
            }
        }
        return false;
    });
    return allTasksAreInImplicitQueueTypeDependencyList;
}

//
// buildBarrierMaps
//

void vpux::BarrierInfo::buildBarrierMaps(mlir::func::FuncOp func) {
    _log.trace("Collect initial producer maps");

    _allTaskOps = to_small_vector(func.getOps<VPURT::TaskOp>());
    _allBarrierOps = to_small_vector(func.getOps<VPURT::DeclareVirtualBarrierOp>());

    _log.nest().trace("There are '{0}' VPURT::TaskOp", _allTaskOps.size());
    _log.nest().trace("There are '{0}' VPURT::DeclareVirtualBarrierOp", _allBarrierOps.size());

    // resize bit maps
    _taskWaitBarriers = SmallVector<TaskSet>(_allTaskOps.size(), {});
    _taskUpdateBarriers = SmallVector<TaskSet>(_allTaskOps.size(), {});
    _barrierConsumerMap = SmallVector<TaskSet>(_allBarrierOps.size(), {});
    _barrierProducerMap = SmallVector<TaskSet>(_allBarrierOps.size(), {});

    // set index to task ops and barrier ops
    for (const auto& p : _allTaskOps | indexed) {
        p.value()->setAttr(_taskIndexAttrName, getIntAttr(p.value().getContext(), p.index()));
        if (p.value()->hasAttr(_syncTaskAttrName)) {
            _syncTasksIds.push_back(p.index());
        }
    }

    if (!_syncTasksIds.empty()) {
        // For now for simplicity limit support to syncTaskIds with equal difference
        // between consecutive ones
        _controlGraphBlockSize = _syncTasksIds[0] + 1;
        for (size_t i = 1; i < _syncTasksIds.size(); i++) {
            auto syncTaskIdDiff = _syncTasksIds[i] - _syncTasksIds[i - 1];
            VPUX_THROW_WHEN(syncTaskIdDiff != _controlGraphBlockSize,
                            "Expected sync task difference is '{0}', but got '{1}'", _controlGraphBlockSize,
                            syncTaskIdDiff);
        }
    }

    for (const auto& p : _allBarrierOps | indexed) {
        p.value()->setAttr(_barrierIndexAttrName, getIntAttr(p.value().getContext(), p.index()));
    }

    for (auto& op : func.getOps()) {
        if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op)) {
            addTaskOp(taskOp);
        } else if (auto barrierOp = mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op)) {
            _log.nest().trace("Found 'VPURT::DeclareVirtualBarrierOp' Operation at '{0}'", op.getLoc());
            VPUX_THROW_WHEN(barrierOp.getBarrier() == nullptr, "DeclareVirtualBarrierOp '{0}' has no barrier.",
                            barrierOp);

            // ensure all barrier users are TaskOps
            for (auto* user : barrierOp.getBarrier().getUsers()) {
                VPUX_THROW_WHEN(mlir::dyn_cast<VPURT::TaskOp>(user) == nullptr, "Got non-TaskOp Operation at '{0}'",
                                op.getLoc());
            }
        }
    }
}

//
// addConsumer
//

void vpux::BarrierInfo::addConsumer(size_t barrierInd, size_t taskInd) {
    _log.trace("Add consumer '{0}' for barrier '{1}'", taskInd, barrierInd);
    _barrierConsumerMap[barrierInd].insert(taskInd);
    _taskWaitBarriers[taskInd].insert(barrierInd);
}

void vpux::BarrierInfo::addConsumer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd) {
    addConsumer(getIndex(barrierOp), taskInd);
}

//
// addConsumers
//

void vpux::BarrierInfo::addConsumers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierConsumerMap[barrierInd].insert(taskInd);
        _taskWaitBarriers[taskInd].insert(barrierInd);
    }
}

//
// addProducer
//

void vpux::BarrierInfo::addProducer(size_t barrierInd, size_t taskInd) {
    _log.trace("Add producer '{0}' for barrier '{1}'", taskInd, barrierInd);
    _barrierProducerMap[barrierInd].insert(taskInd);
    _taskUpdateBarriers[taskInd].insert(barrierInd);
}

void vpux::BarrierInfo::addProducer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd) {
    addProducer(getIndex(barrierOp), taskInd);
}

//
// addProducers
//

void vpux::BarrierInfo::addProducers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierProducerMap[barrierInd].insert(taskInd);
        _taskUpdateBarriers[taskInd].insert(barrierInd);
    }
}

//
// removeProducers
//
void vpux::BarrierInfo::removeProducers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierProducerMap[barrierInd].erase(taskInd);
        _taskUpdateBarriers[taskInd].erase(barrierInd);
    }
}

void vpux::BarrierInfo::removeProducers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds) {
    const auto barrierInd = getIndex(barrierOp);
    removeProducers(barrierInd, taskInds);
}

//
// removeConsumers
//
void vpux::BarrierInfo::removeConsumers(size_t barrierInd, const TaskSet& taskInds) {
    for (const auto& taskInd : taskInds) {
        _barrierConsumerMap[barrierInd].erase(taskInd);
        _taskWaitBarriers[taskInd].erase(barrierInd);
    }
}

void vpux::BarrierInfo::removeConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds) {
    const auto barrierInd = getIndex(barrierOp);
    removeConsumers(barrierInd, taskInds);
}

//
// addTaskOp
//

void vpux::BarrierInfo::addTaskOp(VPURT::TaskOp taskOp) {
    const auto taskInd = getIndex(taskOp);
    _log.trace("Found 'TaskOp' Operation '{0}'", taskInd);

    for (const auto& bar : taskOp.getWaitBarriers()) {
        // Note: can also be VPURT::ConfigureBarrierOp
        auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
        VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid wait barrier op type {0}", bar);
        addConsumer(barrierOp, taskInd);
    }

    for (const auto& bar : taskOp.getUpdateBarriers()) {
        // Note: can also be VPURT::ConfigureBarrierOp
        auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
        VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid wait barrier op type {0}", bar);
        addProducer(barrierOp, taskInd);
    }
}

//
// addNewBarrier
//

size_t vpux::BarrierInfo::addNewBarrier(VPURT::DeclareVirtualBarrierOp barrierOp) {
    size_t barrierInd = _allBarrierOps.size();
    barrierOp->setAttr(_barrierIndexAttrName, getIntAttr(barrierOp.getContext(), barrierInd));

    _log.trace("Add new barrier '{0}', new barrier size '{1}'", barrierInd, _allBarrierOps.size());
    _allBarrierOps.push_back(barrierOp);

    _barrierConsumerMap.push_back({});
    _barrierProducerMap.push_back({});
    return barrierInd;
}

//
// setWaitBarriers
//

void vpux::BarrierInfo::setWaitBarriers(size_t taskInd, const TaskSet& barriers) {
    // remove previous wait barriers
    for (auto barrierInd : _taskWaitBarriers[taskInd]) {
        _barrierConsumerMap[static_cast<size_t>(barrierInd)].erase(taskInd);
    }

    for (auto barrierInd : barriers) {
        _barrierConsumerMap[barrierInd].insert(taskInd);
    }
    _taskWaitBarriers[taskInd] = barriers;
}

//
// setUpdateBarriers
//

void vpux::BarrierInfo::setUpdateBarriers(size_t taskInd, const TaskSet& barriers) {
    // remove previous update barriers
    for (auto barrierInd : _taskUpdateBarriers[taskInd]) {
        _barrierProducerMap[static_cast<size_t>(barrierInd)].erase(taskInd);
    }

    for (auto barrierInd : barriers) {
        _barrierProducerMap[barrierInd].insert(taskInd);
    }
    _taskUpdateBarriers[taskInd] = barriers;
}

//
// removeProducer
//

void vpux::BarrierInfo::removeProducer(size_t barrierInd, size_t taskInd) {
    _barrierProducerMap[barrierInd].erase(taskInd);
    _taskUpdateBarriers[taskInd].erase(barrierInd);
}

void vpux::BarrierInfo::removeProducer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd) {
    removeProducer(getIndex(barrierOp), taskInd);
}

//
// removeConsumer
//
void vpux::BarrierInfo::removeConsumer(size_t barrierInd, size_t taskInd) {
    _barrierConsumerMap[barrierInd].erase(taskInd);
    _taskWaitBarriers[taskInd].erase(barrierInd);
}

void vpux::BarrierInfo::removeConsumer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd) {
    removeConsumer(getIndex(barrierOp), taskInd);
}

//
// resetBarrier
//

void vpux::BarrierInfo::resetBarrier(VPURT::DeclareVirtualBarrierOp barrierOp) {
    const auto barrierInd = getIndex(barrierOp);
    resetBarrier(checked_cast<size_t>(barrierInd));
}

//
// resetBarrier
//

void vpux::BarrierInfo::resetBarrier(size_t barrierInd) {
    _log.trace("Reset barrier '{0}'", barrierInd);

    for (auto taskInd : _barrierProducerMap[barrierInd]) {
        _taskUpdateBarriers[static_cast<size_t>(taskInd)].erase(barrierInd);
    }
    _barrierProducerMap[barrierInd].clear();

    for (auto taskInd : _barrierConsumerMap[barrierInd]) {
        _taskWaitBarriers[static_cast<size_t>(taskInd)].erase(barrierInd);
    }
    _barrierConsumerMap[barrierInd].clear();
}

// Function that will perform split of a control graph with sync points
// present on boundaries of such split
// Example:
//  number of all task: 1200
//  requested split size: 500
//
// Graph will be split into 3 blocks with tasks indexes
// being placed in blocks in following way
//  block 0:    0 -  499
//  block 1:  500 -  999
//  block 2: 1000 - 1199
// Tasks with indexes 499 and 999 are sync points
// meaning that if there was any dependency previously present between blocks
// it will be modified to go through this sync point, example:
//  old dependency:   200 -> Bar -> 600
//  new dependencies: 200 -> newBar1 -> 499  and  499 -> newBar1 -> 600
//
// Goal of this split it to enable faster and less memory hungry optimizations
// on control graph as they can be done within single block between sync tasks
// For this example optimization can be done in 3 ranges: 0 - 499, 499 - 999, 999 - 1199
void vpux::BarrierInfo::splitControlGraphToBlocks(size_t blockSize) {
    VPUX_THROW_WHEN(blockSize < 2, "Minimal block size is 2, requested size - {1}", blockSize);
    VPUX_THROW_WHEN(_controlGraphBlockSize > 0,
                    "Partitioning on control graph was already performed, size - {0}, requested size - {1}",
                    _controlGraphBlockSize, blockSize);

    auto tasksSize = _allTaskOps.size();
    if (blockSize >= tasksSize) {
        _log.trace("Not enough tasks to perform split, requested split size - '{0}', num of tasks - '{1}'", blockSize,
                   tasksSize);
        return;
    }

    _controlGraphBlockSize = blockSize;

    size_t numOfBlocks = tasksSize / blockSize;
    if (tasksSize % blockSize > 0) {
        numOfBlocks++;
    }

    _log.trace("Split control graph: num of task - '{0}', blocks size - '{1}', number of blocks - '{2}'", tasksSize,
               blockSize, numOfBlocks);

    auto barriersCount = _allBarrierOps.size();
    std::map<size_t, std::pair<size_t, size_t>> syncTaskWaitAndUpdateBarrierIndMap;

    // STEP 1:
    // Add explicit sync tasks wait and update barriers
    // which will be used to connect tasks with dependencies
    // going outside of block scope
    // If those new barriers are redundant, could have been combined with other existing barrier
    // or need a split because of large number of producer/consumers they will be optimized/processed
    // by follow-up passes
    for (size_t blockInd = 1; blockInd < numOfBlocks; blockInd++) {
        size_t syncTaskInd = blockInd * blockSize - 1;

        auto syncTaskOp = getTaskOpAtIndex(syncTaskInd);
        auto insertionOp = getTaskOpAtIndex((blockInd - 1) * blockSize);

        mlir::OpBuilder builder(insertionOp);
        //      |<-----syncWaitBar
        //  syncTask
        //      |----->syncUpdBar
        auto waitBar = builder.create<VPURT::DeclareVirtualBarrierOp>(syncTaskOp->getLoc());
        auto waitBarInd = addNewBarrier(waitBar);
        addConsumer(waitBarInd, syncTaskInd);

        builder.setInsertionPoint(syncTaskOp);
        auto updateBar = builder.create<VPURT::DeclareVirtualBarrierOp>(syncTaskOp->getLoc());
        auto updateBarInd = addNewBarrier(updateBar);

        addProducer(updateBarInd, syncTaskInd);

        _log.trace("Sync task at index '{0}', wait bar - '{1}', update bar - '{2}'", syncTaskInd, waitBarInd,
                   updateBarInd);

        syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd] = std::make_pair(waitBarInd, updateBarInd);
    }

    std::map<VPURT::TaskQueueType, size_t> firstTaskWithNoWaitBarForQueueInBlockMap;
    std::map<VPURT::TaskQueueType, size_t> lastTaskWithNoUpdateBarForQueueInBlockMap;

    // STEP 2
    // Identify operations in each block with no wait or update barriers that are first or last
    // for given queue:
    // - last op with no update barrier on each queue - link it to wait barrier of next sync task
    // - first op with no wait barrier on each queue - link it to update barrier of previous sync task
    //
    // If they will not have barriers assigned their position might be shifted
    // with respect to sync task during barrier legalization steps in follow-up passes
    for (size_t taskInd = 0; taskInd < tasksSize; taskInd++) {
        size_t taskBlockInd = taskInd / blockSize;
        auto taskUpdateBarriers = _taskUpdateBarriers[taskInd];
        size_t syncTaskInd = taskBlockInd * blockSize + blockSize - 1;

        // If task does not update any barrier store this information for each queue
        // and when moving to next block add a link to sync task. When encountering sync
        // task add connection from those ops to this sync task
        if (taskBlockInd < numOfBlocks - 1) {
            if (taskInd == syncTaskInd) {
                auto syncTaskWaitBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].first;
                for (const auto& queueListTaskPair : lastTaskWithNoUpdateBarForQueueInBlockMap) {
                    // Create new connection
                    // queueLastTaskInd -> syncWaitBarInd -> syncTaskInd
                    //
                    //  0(last op on queue)       0--------->|
                    //                                       |
                    //        syncWaitBar    =>         syncWaitBar
                    //             |                         |
                    //  1(sync) <--|              1(sync) <--|
                    _log.trace("New dep for task with no update barrier: task '{0}' -> sync task '{1}'",
                               queueListTaskPair.second, syncTaskInd);
                    addProducer(syncTaskWaitBarInd, queueListTaskPair.second);
                }
                lastTaskWithNoUpdateBarForQueueInBlockMap.clear();
            } else if (taskUpdateBarriers.empty()) {
                auto taskOp = getTaskOpAtIndex(taskInd);
                const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
                lastTaskWithNoUpdateBarForQueueInBlockMap[taskQueueType] = taskInd;
            }
        }

        // If task does not wait for any barrier and is first on a given queue
        // for this block (e.g. DMA task) then add a dependency from previous syncTask
        // Otherwise after call to orderExecutionTasksAndBarriers might move it to earlier block
        // and destroy assumption about how graph got split
        if (taskBlockInd > 0) {
            if (taskInd == syncTaskInd || taskInd == tasksSize - 1) {
                size_t prevSyncTaskInd = taskBlockInd * blockSize - 1;
                auto prevSyncTaskUpdateBarInd = syncTaskWaitAndUpdateBarrierIndMap[prevSyncTaskInd].second;
                for (const auto& queueListTaskPair : firstTaskWithNoWaitBarForQueueInBlockMap) {
                    // Create new connection
                    // prevSyncTaskInd -> syncUpdBarInd -> queueFirstTaskInd
                    //
                    //  1(sync)---|               1(sync) ---|
                    //            |                          |
                    //       syncUpdBar       =>        syncUpdBar
                    //                                       |
                    //  2(first op on queue)      2<---------|
                    _log.trace("New dep for task with no wait barrier: sync task '{0}' -> task '{1}'", prevSyncTaskInd,
                               queueListTaskPair.second);
                    addConsumer(prevSyncTaskUpdateBarInd, queueListTaskPair.second);
                }
                firstTaskWithNoWaitBarForQueueInBlockMap.clear();
            } else if (_taskWaitBarriers[taskInd].empty()) {
                auto taskOp = getTaskOpAtIndex(taskInd);
                const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
                if (firstTaskWithNoWaitBarForQueueInBlockMap.find(taskQueueType) ==
                    firstTaskWithNoWaitBarForQueueInBlockMap.end()) {
                    firstTaskWithNoWaitBarForQueueInBlockMap[taskQueueType] = taskInd;
                }
            }
        }
    }

    auto getSetWithNoSyncTasks = [&](const TaskSet& taskSet) {
        TaskSet taskSetWithNoSync;
        for (const auto& task : taskSet) {
            if (!isSyncPoint(task)) {
                taskSetWithNoSync.insert(task);
            }
        }

        return taskSetWithNoSync;
    };

    // STEP 3
    // Perform main processing to identify all ops with long dependency -
    // dependencies that go beyond the scope of existing block
    // Example:
    //  block 0:    0 -  499
    //  block 1:  500 -  999
    //  old dependency:   200 -> Bar -> 600
    //  new dependencies: 200 -> syncWaitBar -> 499(syncTask)  and  499(SyncTask) -> syncUpdBar -> 600
    // For this task scan all barriers and check such that within its producers and consumers have tasks from different
    // blocks. If yes then split those tasks and barrier so that tasks from different groups do not share barrier,
    // besides sync tasks
    for (size_t barInd = 0; barInd < barriersCount; ++barInd) {
        // Check barrier consumers and producers and check if they are from different blocks
        llvm::SmallSet<size_t, 16> blocksSet;

        VPUX_THROW_WHEN(_barrierProducerMap[barInd].empty(), "Barrier has no producers");
        VPUX_THROW_WHEN(_barrierConsumerMap[barInd].empty(), "Barrier has no consumers");

        // Gather information about barrier producers and consumers
        // and store them based on blockId
        std::map<size_t, TaskSet> barProdBlockTasksMap;
        for (const auto& taskInd : _barrierProducerMap[barInd]) {
            size_t taskBlockInd = taskInd / blockSize;
            blocksSet.insert(taskBlockInd);
            barProdBlockTasksMap[taskBlockInd].insert(taskInd);
        }

        std::map<size_t, TaskSet> barConsBlockTasksMap;
        for (const auto& taskInd : _barrierConsumerMap[barInd]) {
            size_t taskBlockInd = taskInd / blockSize;
            blocksSet.insert(taskBlockInd);
            barConsBlockTasksMap[taskBlockInd].insert(taskInd);
        }

        if (blocksSet.size() < 2) {
            // Only tasks from single block included
            // no need to split
            continue;
        }

        // Get barrier block assignment, which would correspond to earliest producer block id
        size_t barBlockId = std::min(barProdBlockTasksMap.begin()->first, barConsBlockTasksMap.begin()->first);

        // If this is last block then early exist. No need to analyze any subsequent barriers
        // as barriers are ordered and from that point no any subsequent barrier would need a split
        if (barBlockId == numOfBlocks + 1) {
            break;
        }

        _log.trace("Bar {0} needs a split, bar block id - {0}", barInd, barBlockId);

        // Remove tasks from this barrier which do not belong to this block
        for (const auto& barProdBlockTasksPair : barProdBlockTasksMap) {
            const auto blockId = barProdBlockTasksPair.first;
            if (blockId == barBlockId) {
                continue;
            }
            // Example control flow that will be shown throughout STEP 3
            // how it is modified
            // bar0 - barrier that is rossing boundary of blocks
            // 0p, 2p, 4p - barrier producer (p) tasks
            // 5c - barrier consumer (c) task
            // 1, 3 - sync tasks
            //
            //    0p -------->|         0p --------->|
            //                |                     bar0
            //    1(sync)     |         1(sync)      |
            //                |                      |
            //    2p -------->|         2p           |
            //                |    =>                |
            //    3(sync)     |         3(sync)      |
            //                |                      |
            //    4p -------->|         4p           |
            //               bar0                    |
            //    5c <--------|         5c <---------|
            _log.nest().trace("Remove producer for bar {0} from block {1}", barInd, blockId);
            removeProducers(barInd, barProdBlockTasksPair.second);

            if (blockId != barProdBlockTasksMap.rbegin()->first ||
                barConsBlockTasksMap.find(blockId) == barConsBlockTasksMap.end()) {
                // If this is not the last block produced by this barrier or there are no consumers in this block
                // add producers of this bar to now produce sync wait barrier
                //
                //    0p -------->|         0p ------------------->|
                //               bar0                             bar0
                //    1(sync)     |         1(sync)                |
                //                |                                |
                //    2p          |         2p -------->|          |
                //                |     =>          syncWaitBar    |
                //    3(sync)     |         3(sync) <---|          |
                //                |                                |
                //    4p          |         4p                     |
                //                |                                |
                //    5c <--------|         5c <-------------------|
                //
                size_t nextSyncTaskInd = blockId * blockSize - 1 + blockSize;
                if (nextSyncTaskInd < tasksSize) {
                    size_t syncTaskWaitBarInd = syncTaskWaitAndUpdateBarrierIndMap[nextSyncTaskInd].first;

                    _log.nest(2).trace("Add producers of sync bar {0} for tasks from block {1}", syncTaskWaitBarInd,
                                       blockId);
                    addProducers(syncTaskWaitBarInd, getSetWithNoSyncTasks(barProdBlockTasksPair.second));
                }
            } else {
                // If this is last block produced by this barrier and there are consumers in this block
                // create a new barrier to handle remaining deps that were produced by currently analyzed barInd
                //
                //    0p ------------------>|       0p ------------------->|
                //                         bar0                          bar0
                //    1(sync)               |       1(sync)                |
                //                          |                              |
                //    2p --------->|        |       2p -------->|          |
                //            syncWaitBar   |  =>          syncWaitBar     |
                //    3(sync) <----|        |       3(sync) <---|          |
                //                          |                              |
                //    4p                    |       4p -------->|          |
                //                          |                newBar        |
                //    5c <------------------|       5c <--------|<---------|
                //
                auto taskOp = getTaskOpAtIndex(*barProdBlockTasksPair.second.begin());
                mlir::OpBuilder builder(taskOp);
                auto newBar = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp->getLoc());
                auto newBarInd = addNewBarrier(newBar);

                _log.nest(2).trace("Add producers and consumers for new bar {0} for tasks from block {1}", newBarInd,
                                   blockId);
                addProducers(newBarInd, barProdBlockTasksPair.second);
                addConsumers(newBarInd, barConsBlockTasksMap[blockId]);
            }

            size_t syncTaskInd = blockId * blockSize - 1;
            size_t syncTaskUpdBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].second;

            //    0p -------------->|         0p --------------->|
            //                     bar0                         bar0
            //    1(sync)           |         1(sync)            |
            //                      |             |----->|       |
            //                      |                syncUpdBar  |
            //                      |         |<---------|       |
            //                      |         |                  |
            //    2p -------->|     |         2p ------->|       |
            //          syncWaitBar |    =>        syncWaitBar   |
            //    3(sync) <---|     |         3(sync) <--|       |
            //                      |             |              |
            //                      |             |----->|       |
            //                      |                syncUpdBar  |
            //                      |         |<---------|       |
            //                      |         |                  |
            //    4p --->|          |         4p --->|           |
            //         newBar       |              newBar        |
            //    5c <---|----------|         5c <---|-----------|
            //
            _log.nest(2).trace("Add consumers of sync bar {0} for tasks from block {1}", syncTaskUpdBarInd, blockId);
            addConsumers(syncTaskUpdBarInd, getSetWithNoSyncTasks(barProdBlockTasksPair.second));
        }
        // Remove tasks from this barrier which do not belong to this block
        for (const auto& barConsBlockTasksPair : barConsBlockTasksMap) {
            if (barConsBlockTasksPair.first == barBlockId) {
                continue;
            }
            //    0p ------------------->|        0p ----------------->|
            //                          bar0                          bar0
            //    1(sync)                |        1(sync)
            //                           |
            //    2p -------->|          |        2p ------->|
            //             syncWaitBar   |   =>          syncWaitBar
            //    3(sync) <---|          |        3(sync) <--|
            //       |                   |           |
            //       |------->|          |           |------>|
            //             syncUpdBar    |               syncUpdBar
            //    |<----------|          |        |<---------|
            //    |                      |        |
            //    4p -------->|          |        4p -------->|
            //             newBar        |                 newBar
            //    5c <--------|<---------|        5c <--------|
            //
            _log.nest().trace("Remove consumers of bar {0} from block {1}", barInd, barConsBlockTasksPair.first);
            removeConsumers(barInd, barConsBlockTasksPair.second);
        }

        size_t barSyncTaskInd = barBlockId * blockSize + blockSize - 1;

        // Sync task should be a consumer of this barrier, but only when
        // it was not the earliest producer
        if (*barProdBlockTasksMap[barBlockId].begin() != barSyncTaskInd) {
            // Remove it from producers just in case it was there
            removeProducer(barInd, barSyncTaskInd);
            //    0p ------------------>|     0p ------------------>|
            //                        bar0                        bar0
            //    1(sync)                     1(sync) <-------------|
            //
            //    2p -------->|               2p ------->|
            //            syncWaitBar                 syncWaitBar
            //    3(sync) <---|               3(sync) <--|
            //      |                     =>    |
            //      |-------->|                 |------->|
            //            syncUpdBar                  syncUpdBar
            //    |<----------|               |<---------|
            //    |                           |
            //    4p -------->|               4p ------->|
            //             newBar                     newBar
            //    5c <--------|               5c <-------|
            //
            addConsumer(barInd, barSyncTaskInd);
        }

        // Link tasks from other blocks from this barrier
        // as consumers of sync task barrier
        for (const auto& barConsBlockTasksPair : barConsBlockTasksMap) {
            if (barConsBlockTasksPair.first == barBlockId) {
                continue;
            }
            size_t syncTaskInd = barConsBlockTasksPair.first * blockSize - 1;
            size_t syncTaskUpdBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].second;

            //    0p ------------------->|     0p ------------------>|
            //                         bar0                        bar0
            //    1(sync) <--------------|     1(sync) <-------------|
            //      |                            |
            //      |-->syncUpdBar               |-->syncUpdBar
            //    |<---------|                 |<---------|
            //    |                            |
            //    2p -------->|                2p ------->|
            //           syncWaitBar                 syncWaitBar
            //    3(sync) <---|                3(sync) <--|
            //      |                     =>   |
            //      |-------->|                |-------------->|
            //           syncUpdBar                        syncUpdBar
            //    |<----------|                |<--------------|
            //    |                            |               |
            //    4p -------->|                4p ------->|    |
            //             newBar                      newBar  |
            //    5c <--------|                5c <-------|<---| (in this partuclar example this dep is redundant
            //                                                    but it will be optimized by later passes)
            //
            _log.nest().trace("Add consumers of sync bar {0} from block {1}", syncTaskUpdBarInd,
                              barConsBlockTasksPair.first);
            addConsumers(syncTaskUpdBarInd, getSetWithNoSyncTasks(barConsBlockTasksPair.second));
        }
    }

    // STEP 4
    // Add explicit dependency between sync tasks
    // If it is not needed because it was already represented implicitly by other
    // dependencies it will be removed by follow-up passes
    for (size_t blockInd = 2; blockInd < numOfBlocks; blockInd++) {
        size_t syncTaskInd = (blockInd - 1) * blockSize - 1;
        size_t nextSyncTaskInd = blockInd * blockSize - 1;

        // Create new connection
        // syncTaskInd -> syncTaskUpdateBarInd -> nextSyncTaskInd
        auto syncTaskUpdateBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].second;

        _log.trace("New dep: sync task '{0}' -> next sync task '{1}' using barrier '{2}'", syncTaskInd, nextSyncTaskInd,
                   syncTaskUpdateBarInd);
        addConsumer(syncTaskUpdateBarInd, nextSyncTaskInd);
    }

    // STEP 5
    // Scan all barriers and perform check and sanitization needed after
    // modifications done in STEP 3
    for (size_t barInd = 0; barInd < barriersCount; ++barInd) {
        // If after modifying control graph there is a barrier with no consumers then it
        // is redundant and related producers should be linked to next sync task
        if (_barrierConsumerMap[barInd].empty() && !_barrierProducerMap[barInd].empty()) {
            _log.trace("Empty consumers for bar - {0}", barInd);
            for (auto taskInd : _barrierProducerMap[barInd]) {
                size_t taskBlockInd = taskInd / blockSize;
                size_t syncTaskInd = taskBlockInd * blockSize + blockSize - 1;
                if (syncTaskInd >= tasksSize) {
                    continue;
                }

                if (taskInd != syncTaskInd) {
                    // Create new connection
                    // taskInd -> syncTaskWaitBarInd -> syncTaskInd
                    auto syncTaskWaitBarInd = syncTaskWaitAndUpdateBarrierIndMap[syncTaskInd].first;
                    addProducer(syncTaskWaitBarInd, taskInd);
                }
            }

            resetBarrier(barInd);
        }

        // Check if after modifying control graph there are no barriers
        // with empty producers and non empty consumers
        if (_barrierProducerMap[barInd].empty()) {
            VPUX_THROW_UNLESS(_barrierConsumerMap[barInd].empty(), "Barrier {0} has no producers but has consumers",
                              barInd);
        }
    }

    // STEP 6
    // Remove not used barriers created for sync tasks
    for (const auto& syncTaskWitAndUpdateBarrierInd : syncTaskWaitAndUpdateBarrierIndMap) {
        const auto waitBarInd = syncTaskWitAndUpdateBarrierInd.second.first;
        const auto updateBarInd = syncTaskWitAndUpdateBarrierInd.second.first;
        if (_barrierProducerMap[waitBarInd].empty()) {
            resetBarrier(waitBarInd);
        }
        if (_barrierConsumerMap[updateBarInd].empty()) {
            resetBarrier(updateBarInd);
        }
    }

    // STEP 7
    // Put dedicated attribute on sync tasks for follow-up passes so that
    // they are aware of the split and can take advantage of it (e.g. perform optimizeBarriers in parts)
    for (size_t blockInd = 1; blockInd < numOfBlocks; blockInd++) {
        size_t syncTaskInd = blockInd * blockSize - 1;
        auto syncTaskOp = getTaskOpAtIndex(syncTaskInd);
        syncTaskOp->setAttr(_syncTaskAttrName, mlir::UnitAttr::get(syncTaskOp->getContext()));
    }
}

bool vpux::BarrierInfo::verifyControlGraphSplit() {
    if (_syncTasksIds.empty()) {
        return true;
    }

    for (size_t i = 1; i < _syncTasksIds.size(); i++) {
        auto syncTaskIdDiff = _syncTasksIds[i] - _syncTasksIds[i - 1];
        VPUX_THROW_WHEN(syncTaskIdDiff != _controlGraphBlockSize,
                        "Expected sync task difference is '{0}', but got '{1}'", _controlGraphBlockSize,
                        syncTaskIdDiff);
    }

    auto getBlockInd = [&](size_t taskInd) {
        return taskInd / _controlGraphBlockSize;
    };

    auto getSyncTaskInd = [&](size_t blockInd) {
        return (blockInd + 1) * _controlGraphBlockSize - 1;
    };

    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); taskInd++) {
        size_t taskBlockInd = getBlockInd(taskInd);
        size_t syncTaskInd = getSyncTaskInd(taskBlockInd);

        for (const auto& updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            auto barrierConsumers = _barrierConsumerMap[updateBarrierInd];

            for (const auto& childTaskInd : barrierConsumers) {
                size_t childBlockInd = getBlockInd(childTaskInd);

                if (taskInd == syncTaskInd && taskBlockInd + 1 != childBlockInd) {
                    _log.error("Out of block dep for sync task: {0}(block {1}) -> {2}(block{3}) ", taskInd,
                               taskBlockInd, childTaskInd, childBlockInd);
                    return false;
                } else if (taskInd != syncTaskInd && taskBlockInd != childBlockInd) {
                    _log.error("Out of block dep for task: {0}(block {1}) -> {2}(block{3}) ", taskInd, taskBlockInd,
                               childTaskInd, childBlockInd);
                    return false;
                }
            }
        }
    }
    return true;
}

void vpux::BarrierInfo::removeSyncTaskAttributes() {
    if (_syncTasksIds.empty()) {
        return;
    }

    for (const auto& syncTaskInd : _syncTasksIds) {
        auto syncTaskOp = getTaskOpAtIndex(syncTaskInd);
        VPUX_THROW_UNLESS(syncTaskOp->hasAttr(_syncTaskAttrName), "Expected sync task attribute at task index '{0}'",
                          syncTaskInd);
        syncTaskOp->removeAttr(_syncTaskAttrName);
    }
}

bool vpux::BarrierInfo::isSyncPoint(size_t taskIdx) {
    // Currently we assume that sync points are uniformly distributed every _controlGraphBlockSize tasks
    if (_controlGraphBlockSize == 0 || taskIdx == _allTaskOps.size() - 1) {
        return false;
    }

    return (taskIdx + 1) % _controlGraphBlockSize == 0;
}

size_t vpux::BarrierInfo::getControlGraphBlockCount() const {
    return _syncTasksIds.size() + 1;
}

std::pair<size_t, size_t> vpux::BarrierInfo::getControlGraphBlockTaskRange(size_t blockInd, bool blockStartSyncPoint,
                                                                           bool blockEndSyncPoint) const {
    VPUX_THROW_WHEN(blockInd >= getControlGraphBlockCount(), "Invalid task block index ({0})", blockInd);
    size_t blockStartInd, blockEndInd;
    if (blockInd == 0) {
        blockStartInd = 0;  // for first block the start index of the block is the index if the first task
    } else {
        blockStartInd = blockStartSyncPoint ? _syncTasksIds[blockInd - 1] : _syncTasksIds[blockInd - 1] + 1;
    }

    if (blockInd == getControlGraphBlockCount() - 1) {
        blockEndInd = getNumOfTasks() - 1;  // for last block the end index of the block is the index if the last task
    } else {
        blockEndInd = blockEndSyncPoint ? _syncTasksIds[blockInd] : _syncTasksIds[blockInd] - 1;
    }

    return std::make_pair(blockStartInd, blockEndInd);
}

//
// optimizeBarriers
//
void vpux::BarrierInfo::optimizeBarriers() {
    // A -> B -> C

    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    // Note: It also will merge barriers which have the same producers but different consumers

    // Barrier are optimized based on order of task ops

    _log.trace("Optimize barriers");
    _log = _log.nest();
    _log.trace("Total tasks count: {0}, total barrier count: {1}", _allTaskOps.size(), _allBarrierOps.size());

    // Perform optimization in tasks blocks matching the distribution of synchronization points.
    // For update barriers and wait barriers we need to include sync points on both ends of the block
    // because tasks from within a given range can have updateBarriers whose consumer
    // is the upper bound sync point, and they can have waitBarriers whose producer is the lower bound sync point.
    for (size_t taskBlockIndex = 0; taskBlockIndex < getControlGraphBlockCount(); ++taskBlockIndex) {
        auto [blockStartInd, blockEndInd] = getControlGraphBlockTaskRange(taskBlockIndex);
        _log.trace("Block {0}, task range [{1}, {2}] ({3} tasks)", taskBlockIndex, blockStartInd, blockEndInd,
                   blockEndInd - blockStartInd + 1);

        // optimize producers
        _log.trace("Optimize producers / update barriers");
        optimizeBarrierProducers(taskBlockIndex);

        // optimize barriers which have the same producers but different consumers
        _log.trace("Optimize barriers / same producers, different consumers");
        optimizeBarriersWithSameProducers(taskBlockIndex);

        // optimize consumers
        _log.trace("Optimize consumers / wait barriers");
        optimizeBarrierConsumers(taskBlockIndex);
    }

    _log.trace("Total tasks count: {0}, total barrier count: {1}", _allTaskOps.size(), _allBarrierOps.size());
    _log = _log.unnest();
}

void vpux::BarrierInfo::optimizeBarrierProducers(size_t blockIdx) {
    // TODO: E#79318 optimize loops
    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
    const auto barriersRangeVec = getBarriersForTaskBlock(blockIdx, /* blockStartSyncPoint */ true,
                                                          /* blockEndSyncPoint */ true, /* updateBarriers */ true);
    if (barriersRangeVec.empty()) {
        return;
    }
    size_t barrierOffset = barriersRangeVec[0];

    blockEndInd = std::min(blockEndInd, _allTaskOps.size() - 1);
    VPUX_THROW_WHEN(blockStartInd > blockEndInd, "Lower bound of task optimization range is greater than upper bound");

    unsigned blockSize = blockEndInd - blockStartInd + 1;
    unsigned barrierSize = barriersRangeVec.back() - barriersRangeVec.front() + 1;

    SmallVector<llvm::BitVector> updateBarriers;
    resizeBitMap(updateBarriers, blockSize, checked_cast<uint32_t>(barrierSize));
    resetBitMap(updateBarriers);

    _log.nest().trace("Collect redundant dependencies");
    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        setBarrierMask(updateBarriers[taskInd - blockStartInd], _taskUpdateBarriers[taskInd], barrierOffset);
        if (taskInd == blockEndInd && isSyncPoint(taskInd)) {
            // ignore updating updateBarriers for upper bound sync point because it only has barriers whose consumers
            // are tasks from outside of the current range.
            continue;
        }
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, childTaskInd)) {
                    updateBarriers[taskInd - blockStartInd] |=
                            updateBarriers[static_cast<size_t>(childTaskInd - blockStartInd)];
                } else {
                    VPUX_THROW("Task {0} has update barriers with a consumer ({1}) from outside of the current "
                               "range [{2}, {3}].",
                               taskInd, childTaskInd, blockStartInd, blockEndInd);
                }
            }
        }
    }

    _log.nest().trace("Remove redundant dependencies");
    for (size_t taskInd = static_cast<unsigned>(blockStartInd); taskInd <= static_cast<unsigned>(blockEndInd);
         ++taskInd) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, childTaskInd)) {
                    updateBarriers[taskInd - blockStartInd].reset(
                            updateBarriers[static_cast<size_t>(childTaskInd - blockStartInd)]);
                }
            }
        }
        TaskSet targetUpdateBarriers;
        for (auto bar : updateBarriers[taskInd - blockStartInd].set_bits()) {
            targetUpdateBarriers.insert(bar + barrierOffset);
        }
        setUpdateBarriers(taskInd, targetUpdateBarriers);
    }
}

void vpux::BarrierInfo::optimizeBarrierConsumers(size_t blockIdx) {
    // TODO: E#79318 optimize loops
    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
    const auto barriersRangeVec = getBarriersForTaskBlock(blockIdx, /* blockStartSyncPoint  */ true,
                                                          /* blockEndSyncPoint */ true, /* updateBarriers */ false);
    if (barriersRangeVec.empty()) {
        return;
    }
    size_t barrierOffset = barriersRangeVec[0];

    blockEndInd = std::min(blockEndInd, _allTaskOps.size() - 1);
    VPUX_THROW_WHEN(blockStartInd > blockEndInd, "Lower bound of task optimization range is greater than upper bound");

    unsigned blockSize = blockEndInd - blockStartInd + 1;
    unsigned barrierSize = barriersRangeVec.back() - barriersRangeVec.front() + 1;
    // optimize consumers

    SmallVector<llvm::BitVector> waitBarriers;
    resizeBitMap(waitBarriers, blockSize, checked_cast<uint32_t>(barrierSize));
    resetBitMap(waitBarriers);

    _log.nest().trace("Collect redundant dependencies");
    for (auto taskInd = blockStartInd; taskInd <= blockEndInd; ++taskInd) {
        setBarrierMask(waitBarriers[taskInd - blockStartInd], _taskWaitBarriers[taskInd], barrierOffset);
        if (taskInd == blockStartInd && isSyncPoint(taskInd)) {
            // ignore updating waitBarriers for lower bound sync point because it only has barriers whose producers
            // are tasks from outside of the current range.
            continue;
        }
        for (auto waitBarrierInd : _taskWaitBarriers[taskInd]) {
            for (auto parentTaskInd : _barrierProducerMap[waitBarrierInd]) {
                if (inRange(blockStartInd, blockEndInd, parentTaskInd)) {
                    waitBarriers[taskInd - blockStartInd] |=
                            waitBarriers[static_cast<size_t>(parentTaskInd - blockStartInd)];
                } else {
                    VPUX_THROW("Task {0} has wait barriers with a producer ({1}) from outside of the current range "
                               "[{2}, {3}].",
                               taskInd, parentTaskInd, blockStartInd, blockEndInd);
                }
            }
        }
    }

    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        for (auto waitBarrierInd : _taskWaitBarriers[taskInd]) {
            for (auto parentTaskInd : _barrierProducerMap[waitBarrierInd]) {
                if (inRange(blockStartInd, blockEndInd, parentTaskInd)) {
                    waitBarriers[taskInd - blockStartInd].reset(
                            waitBarriers[static_cast<size_t>(parentTaskInd - blockStartInd)]);
                }
            }
        }

        TaskSet targetWaitBarriers;
        for (auto bar : waitBarriers[taskInd - blockStartInd].set_bits()) {
            targetWaitBarriers.insert(bar + barrierOffset);
        }
        setWaitBarriers(taskInd, targetWaitBarriers);
    }
}

void vpux::BarrierInfo::optimizeBarriersWithSameProducers(size_t blockIdx) {
    // Collect a vector of barrier indexes for current task range.
    // If a new barrier is created between subsequent calls to optimizeBarrier (eg. during looped linearization),
    // iterating over barriersRangeVec will not imply iterating over largely overlapping barrier index ranges
    // when moving from one task block to another.
    const auto barriersRangeVec =
            getBarriersForTaskBlock(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ false);

    if (barriersRangeVec.empty()) {
        return;
    }
    _log.nest().trace("Barrier range [{0}, {1}]", barriersRangeVec.front(), barriersRangeVec.back());
    VPUX_THROW_WHEN(barriersRangeVec.front() > _allBarrierOps.size() - 1,
                    "Invalid barrier value for lower limit of barrier range optimization");
    VPUX_THROW_WHEN(barriersRangeVec.back() > _allBarrierOps.size() - 1,
                    "Invalid barrier value for upper limit of barrier range optimization");

    const auto maxVariantCount = getBarrierMaxVariantSum();

    auto addFunc = [&](size_t sum, size_t taskInd) {
        return sum + getNumOfSlotsUsedByTask(getTaskOpAtIndex(taskInd));
    };

    // Check the slot count is valid or not if the two barriers are merged
    auto legalMergeSlotCount = [&](const size_t& bar1, const size_t& bar2) {
        size_t slotCount = std::accumulate(_barrierProducerMap[bar1].begin(), _barrierProducerMap[bar1].end(),
                                           static_cast<size_t>(0), addFunc);
        BarrierInfo::TaskSet consumers = _barrierConsumerMap[bar1];
        // Use set operation to remove duplicated consumers
        llvm::set_union(consumers, _barrierConsumerMap[bar2]);
        slotCount = std::accumulate(consumers.begin(), consumers.end(), slotCount, addFunc);
        return slotCount <= maxVariantCount;
    };

    for (size_t ind1 = 0; ind1 < barriersRangeVec.size(); ++ind1) {
        size_t barInd = barriersRangeVec[ind1];
        for (size_t ind2 = ind1 + 1; ind2 < barriersRangeVec.size(); ++ind2) {
            size_t childBarInd = barriersRangeVec[ind2];
            if (_barrierProducerMap[barInd] == _barrierProducerMap[childBarInd]) {
                _log.nest().trace("Same producers for barId '{0}' '{1}'", barInd, childBarInd);
                if (legalMergeSlotCount(barInd, childBarInd)) {
                    for (auto consumerInd : _barrierConsumerMap[childBarInd]) {
                        // move all consumers to one barrier
                        addConsumer(barInd, static_cast<size_t>(consumerInd));
                    }
                    if (_func != nullptr) {  // test scenarios do not initialize FuncOp
                        _log.trace("New consumers number - {0}", getConsumerSlotCount(getBarrierOpAtIndex(barInd)));
                    }
                    resetBarrier(childBarInd);
                }
            }
        }
    }
}

bool vpux::BarrierInfo::inRange(const unsigned low, const unsigned high, const unsigned val) const {
    return val >= low && val <= high;
}

void vpux::BarrierInfo::setBarrierMask(llvm::BitVector& mask, const BarrierInfo::TaskSet& barriers, size_t offset) {
    for (auto bar : barriers) {
        mask.set(bar - offset);
    }
}

//
// buildTaskQueueTypeMap
//

void vpux::BarrierInfo::buildTaskQueueTypeMap(bool considerTaskFifoDependency) {
    if (_taskQueueTypeMap.empty()) {
        // resize implicit dependency map
        const auto module = _func->getParentOfType<mlir::ModuleOp>();
        const auto dmaPortNum = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();

        auto dmaChannels = getDMAChannelsWithIndependentLinkAgents(VPU::getArch(module));
        for (auto dmaPortIdx : irange(dmaPortNum)) {
            for (auto dmaChannel : dmaChannels) {
                VPURT::TaskQueueType taskQueueType;
                taskQueueType.type = VPU::ExecutorKind::DMA_NN;
                taskQueueType.id = getDMAQueueIdEncoding(dmaPortIdx, dmaChannel);
                llvm::BitVector taskList(checked_cast<uint32_t>(_allTaskOps.size()));
                _taskQueueTypeMap.insert(std::make_pair(taskQueueType, taskList));
            }
        }
    }

    if (considerTaskFifoDependency) {
        for (const auto& taskOp : _allTaskOps | reversed) {
            auto taskInd = getIndex(taskOp);
            auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
            if (_taskQueueTypeMap.find(taskQueueType) != _taskQueueTypeMap.end()) {
                _taskQueueTypeMap[taskQueueType].set(taskInd);
            }
        }
    }
}

//
// buildTaskControlMap
//

std::pair<SmallVector<llvm::BitVector>, size_t> vpux::BarrierInfo::buildTaskControlMap(
        size_t blockIdx, bool considerTaskFifoDependency) {
    SmallVector<llvm::BitVector> taskControlMap;
    VPUX_THROW_WHEN(blockIdx >= getControlGraphBlockCount(), "Invalid task block index ({0})", blockIdx);

    auto [blockStartInd, blockEndInd] =
            getControlGraphBlockTaskRange(blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);

    VPUX_THROW_UNLESS(blockStartInd <= blockEndInd, "Invalid range of tasks for building task control map [{0}, {1}]",
                      blockStartInd, blockEndInd);

    _log.trace("Build task control map for task range [{0}, {1}]", blockStartInd, blockEndInd);
    auto newTaskControlMapSize = blockEndInd - blockStartInd + 1;
    if (taskControlMap.size() != newTaskControlMapSize) {
        resizeBitMap(taskControlMap, newTaskControlMapSize, checked_cast<uint32_t>(newTaskControlMapSize));
    }
    resetBitMap(taskControlMap);

    if (considerTaskFifoDependency) {
        auto copyBitsRange = [](const llvm::BitVector& srcVec, int fromIdx, int toIdx) {
            llvm::SmallVector<int> dstVec;  // implementation on SmallVector is faster than on a slice of BitVector
            for (int idx = srcVec.find_first_in(fromIdx, toIdx, true); idx <= toIdx && idx != -1;
                 idx = srcVec.find_next(idx)) {
                dstVec.push_back(idx - fromIdx);
            }
            return dstVec;
        };

        for (const auto& item : _taskQueueTypeMap) {
            const auto tasksInSameFIFO = copyBitsRange(item.second, blockStartInd, blockEndInd);
            for (size_t i = 0; i < tasksInSameFIFO.size(); i++) {
                auto taskInd = tasksInSameFIFO[i];
                for (size_t j = i + 1; j < tasksInSameFIFO.size(); j++) {
                    auto nextTaskInd = tasksInSameFIFO[j];
                    taskControlMap[taskInd].set(nextTaskInd);
                }
            }
        }
    }

    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto consumerIdx : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, consumerIdx)) {
                    taskControlMap[taskInd - blockStartInd].set(consumerIdx - blockStartInd);
                }
            }
        }
    }

    for (auto taskInd = blockEndInd + 1; taskInd-- > blockStartInd;) {
        if (taskInd == blockEndInd && isSyncPoint(taskInd)) {
            continue;
        }
        for (auto updateBarrierInd : _taskUpdateBarriers[taskInd]) {
            for (auto childTaskInd : _barrierConsumerMap[static_cast<size_t>(updateBarrierInd)]) {
                if (inRange(blockStartInd, blockEndInd, childTaskInd)) {
                    taskControlMap[taskInd - blockStartInd] |=
                            taskControlMap[static_cast<size_t>(childTaskInd - blockStartInd)];
                } else {
                    VPUX_THROW("Task {0} has update barriers with a consumer ({1}) from outside of the current "
                               "range [{2}, {3}].",
                               taskInd, childTaskInd, blockStartInd, blockEndInd);
                }
            }
        }
    }

    return std::make_pair(taskControlMap, blockStartInd);
}

//
// controlPathExistsBetweenTasksInSameBlock
//

bool vpux::BarrierInfo::controlPathExistsBetweenTasksInSameBlock(const SmallVector<llvm::BitVector>& taskControlMap,
                                                                 size_t taskAInd, size_t taskBInd,
                                                                 bool biDirection) const {
    // ensure that taskControlMap is build at given time with buildTaskControlMap() for the correct task block
    VPUX_THROW_WHEN(taskControlMap.empty(), "Task control map not initialized");

    VPUX_THROW_WHEN(taskAInd >= taskControlMap.size(), "taskAInd out of range {0}", taskAInd);
    VPUX_THROW_WHEN(taskBInd >= taskControlMap.size(), "taskBInd out of range {0}", taskBInd);

    if (biDirection) {
        return taskControlMap[taskAInd][taskBInd] || taskControlMap[taskBInd][taskAInd];
    }
    return taskControlMap[taskAInd][taskBInd];
}

//
// updateIR
//

void vpux::BarrierInfo::updateIR() {
    // update IR using stored dependencies
    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); ++taskInd) {
        auto taskOp = getTaskOpAtIndex(taskInd);

        taskOp.getWaitBarriersMutable().clear();
        for (auto barrierInd : _taskWaitBarriers[taskInd]) {
            auto barrierOp = getBarrierOpAtIndex(static_cast<size_t>(barrierInd));
            taskOp.getWaitBarriersMutable().append(barrierOp.getBarrier());
        }

        taskOp.getUpdateBarriersMutable().clear();
        for (auto barrierInd : _taskUpdateBarriers[taskInd]) {
            auto barrierOp = getBarrierOpAtIndex(static_cast<size_t>(barrierInd));
            taskOp.getUpdateBarriersMutable().append(barrierOp.getBarrier());
        }
    }
}

//
// logBarrierInfo
//

void vpux::BarrierInfo::logBarrierInfo() {
    // useful for logging and debugging barrier dependencies
    _log.setName("barrier-info-log");
    _log.trace("Logging BarrierInfo");

    for (size_t taskInd = 0; taskInd < _allTaskOps.size(); ++taskInd) {
        _log.nest().trace("Task '{0}'", taskInd);
        for (const auto& barrierOp : getWaitBarriers(taskInd)) {
            _log.nest(2).trace("Task '{0}' waits for '{1}'", taskInd, barrierOp);
        }
        for (const auto& barrierOp : getUpdateBarriers(taskInd)) {
            _log.nest(2).trace("Task '{0}' updates '{1}'", taskInd, barrierOp);
        }
    }
}

//
// createLegalVariantBatches
//

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfo::createLegalVariantBatches(const TaskSet& tasks,
                                                                               size_t availableSlots) {
    // store batches of tasks
    SmallVector<TaskSet> legalBatches(1);

    // store total slot count used by batch
    size_t totalSlotCount = 0;

    const auto isLegalVariantCountWith = [&](size_t numSlotsUsedByTask) -> bool {
        return (totalSlotCount + numSlotsUsedByTask) <= availableSlots;
    };

    // create batches for new barriers
    auto orderedTasks = std::set<size_t>(tasks.begin(), tasks.end());
    for (const auto& taskInd : orderedTasks) {
        // find number of slots consumed by this task
        auto numSlotsUsedByTask = getNumOfSlotsUsed(getTaskOpAtIndex(taskInd));

        // check if new batch needs to be created
        if (!isLegalVariantCountWith(numSlotsUsedByTask)) {
            legalBatches.push_back({});
            totalSlotCount = 0;
        }

        legalBatches.rbegin()->insert(taskInd);
        totalSlotCount += numSlotsUsedByTask;
    }

    return legalBatches;
}

//
// haveSameImplicitDependencyTaskQueueType
//

std::optional<VPURT::TaskQueueType> vpux::BarrierInfo::haveSameImplicitDependencyTaskQueueType(
        const TaskSet& taskInds) {
    if (taskInds.empty()) {
        return std::nullopt;
    }
    auto firstTaskInd = *taskInds.begin();
    // ensure that _taskQueueTypeMap is build at given time with buildTaskControlMap()
    VPUX_THROW_WHEN(_taskQueueTypeMap.empty(), "Task queue map not initialized");
    for (const auto& item : _taskQueueTypeMap) {
        // get the task queue type for the first task, then check all the other tasks have same task queue type
        if (!item.second.test(firstTaskInd)) {
            continue;
        }

        for (const auto& taskInd : taskInds) {
            if (!item.second.test(taskInd) || isSyncPoint(taskInd)) {
                return std::nullopt;
            }
        }
        return item.first;
    }
    return std::nullopt;
}

//
// canBarriersBeMerged
//

bool vpux::BarrierInfo::canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                                            const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                                            ArrayRef<TaskSet> origWaitBarriersMap) {
    // two barriers A and B can be merged if
    // 1. any producer of barrier A controls any consumer of barrier B
    if (!producersControlsAllConsumers(barrierProducersA, barrierConsumersB, barrierConsumersA, origWaitBarriersMap)) {
        return false;
    }

    // 2. any producer of barrier B controls any consumer of barrier A
    if (!producersControlsAllConsumers(barrierProducersB, barrierConsumersA, barrierConsumersB, origWaitBarriersMap)) {
        return false;
    }

    return true;
}

//
// getWaitBarriersMap
//

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfo::getWaitBarriersMap() {
    return _taskWaitBarriers;
}

BarrierInfoTest::BarrierInfoTest(BarrierInfoTest::BarrierMaps& barrierMaps) {
    initializeBarrierMaps(barrierMaps);
}

void BarrierInfoTest::initializeBarrierMaps(BarrierInfoTest::BarrierMaps& barrierMaps) {
    BarrierInfo::_barrierProducerMap = toTaskSet(barrierMaps.barrierProducerMap);
    BarrierInfo::_barrierConsumerMap = toTaskSet(barrierMaps.barrierConsumerMap);
    BarrierInfo::_taskUpdateBarriers = toTaskSet(barrierMaps.taskUpdateBarriers);
    BarrierInfo::_taskWaitBarriers = toTaskSet(barrierMaps.taskWaitBarriers);
    BarrierInfo::_controlGraphBlockSize = barrierMaps.controlGraphBlockSize;
    BarrierInfo::_allBarrierOps.resize(barrierMaps.Nbarriers);
    BarrierInfo::_allTaskOps.resize(barrierMaps.Ntasks);
    BarrierInfo::_syncTasksIds = barrierMaps.syncTasksIds;
}

void vpux::BarrierInfoTest::setMaxVariantCountPerBarrier(size_t variantCount) {
    _maxVariantCountPerBarrier = variantCount;
}

size_t vpux::BarrierInfoTest::getBarrierMaxVariantSum() const {
    return _maxVariantCountPerBarrier;
}

size_t vpux::BarrierInfoTest::getNumOfSlotsUsedByTask([[maybe_unused]] VPURT::TaskOp op) const {
    return 1;  // for tests, assume single slot per task (as for eg. VPU::ExecutorKind::DMA_NN)
}

VPURT::TaskOp vpux::BarrierInfoTest::getTaskOpAtIndex([[maybe_unused]] size_t opIdx) const {
    return nullptr;
}

BarrierInfoTest::BarrierMaps vpux::BarrierInfoTest::getOptimizedMaps() {
    BarrierInfoTest::BarrierMaps optimizedMaps;
    optimizedMaps.taskWaitBarriers = toTaskVec(BarrierInfo::_taskWaitBarriers);
    optimizedMaps.taskUpdateBarriers = toTaskVec(BarrierInfo::_taskUpdateBarriers);
    optimizedMaps.barrierConsumerMap = toTaskVec(BarrierInfo::_barrierConsumerMap);
    optimizedMaps.barrierProducerMap = toTaskVec(BarrierInfo::_barrierProducerMap);
    return optimizedMaps;
}

BarrierInfoTest::BarrierMaps vpux::BarrierInfoTest::optimizeBarrierProducers(size_t blockIdx) {
    BarrierInfo::optimizeBarrierProducers(blockIdx);
    return getOptimizedMaps();
}

BarrierInfoTest::BarrierMaps vpux::BarrierInfoTest::optimizeBarrierConsumers(size_t blockIdx) {
    BarrierInfo::optimizeBarrierConsumers(blockIdx);
    return getOptimizedMaps();
}

BarrierInfoTest::BarrierMaps vpux::BarrierInfoTest::optimizeBarriersWithSameProducers(size_t blockIdx) {
    BarrierInfo::optimizeBarriersWithSameProducers(blockIdx);
    return getOptimizedMaps();
}

BarrierInfoTest::BarrierMaps vpux::BarrierInfoTest::optimizeBarriers() {
    BarrierInfo::optimizeBarriers();
    return getOptimizedMaps();
}

SmallVector<BarrierInfo::TaskSet> vpux::BarrierInfoTest::toTaskSet(SmallVector<SmallVector<size_t>>& map) {
    SmallVector<BarrierInfo::TaskSet> convertedMap(map.size());
    for (auto deps : map | indexed) {
        for (auto dep : deps.value()) {
            convertedMap[deps.index()].insert(dep);
        }
    }
    return convertedMap;
}

BarrierMap vpux::BarrierInfoTest::toTaskVec(SmallVector<BarrierInfo::TaskSet>& map) {
    BarrierMap convertedMap(map.size());
    for (auto deps : map | indexed) {
        if (deps.value().size() == 0) {
            continue;
        }
        convertedMap[deps.index()].reserve(deps.value().size());
        for (auto dep : deps.value()) {
            convertedMap[deps.index()].push_back(dep);
        }
        llvm::sort(convertedMap[deps.index()]);
    }
    return convertedMap;
}
