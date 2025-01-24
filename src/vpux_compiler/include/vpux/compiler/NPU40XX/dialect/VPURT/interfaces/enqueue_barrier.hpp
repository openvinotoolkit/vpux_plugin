//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux {
namespace VPURT {

//
// EnqueueBarrierHandler
// Module to find barrier
//
class EnqueueBarrierHandler {
public:
    EnqueueBarrierHandler() = delete;
    EnqueueBarrierHandler(mlir::func::FuncOp func, BarrierInfo& barrierInfo, Logger log = Logger::global());
    // Below constructor is meant to be used only for testing purpose
    EnqueueBarrierHandler(BarrierInfoTest& barrierInfoTest,
                          std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueTypeMap,
                          SmallVector<size_t>& barrierToPidVec, size_t _barrierFifoDepth = 1, size_t dmaFifoDepth = 64,
                          bool optimizeAndMergeEnqFlag = true, Logger log = Logger::global());

    mlir::LogicalResult calculateEnqueueBarriers();
    std::optional<size_t> getEnqueueBarrier(size_t taskInd);
    mlir::Value getEnqueueBarrier(VPURT::TaskOp taskOp);

private:
    void initPrevPhysBarrierData(mlir::func::FuncOp func);
    void initPrevPhysBarrierData(SmallVector<size_t>& barrierToPidVec, size_t nPhysBars);
    std::optional<size_t> getStartBarrierIndex(mlir::func::FuncOp func);
    std::optional<size_t> getInitialEnqueueBarrier(size_t taskInd);

    bool isBarrierAConsumedBeforeBarrierB(size_t barrierA, size_t barrierB);
    bool areTasksABeforeTasksB(const BarrierInfo::TaskSet& tasksA, const BarrierInfo::TaskSet& tasksB);
    bool isBarrierConsumedBeforeTask(size_t bar, size_t taskInd);
    bool isBarrierConsumptionDependantOnTask(size_t bar, size_t taskInd);
    void delayEnqIfNeededBasedOnPrevEnq(std::optional<size_t>& enqBarOpt, std::optional<size_t> previousEnqBarOpt);

    void optimizeEnqueueIfPossible(std::optional<size_t>& enqBarOpt, BarrierInfo::TaskSet& waitBarriers,
                                   BarrierInfo::TaskSet& updateBarriers, std::optional<size_t> previousTaskIndOpt,
                                   std::optional<size_t> previousEnqBarOpt);

    mlir::LogicalResult delayEnqIfNeededBasedOnFifoState(
            std::optional<size_t>& enqBarOpt, std::vector<std::optional<size_t>>& outstandingEnqueuesTaskIndexVec,
            std::vector<std::optional<size_t>>& outstandingEnqueuesTaskWaitBarIndexVec,
            size_t outstandingEnquOpsCounter);

    std::optional<size_t> getNthPrevBarInstance(size_t vid, size_t n);

    mlir::LogicalResult findInitialEnqWithLcaForGivenBarriers(std::optional<size_t>& enqBarOpt,
                                                              BarrierInfo::TaskSet& waitBarriers,
                                                              BarrierInfo::TaskSet& updateBarriers);

    BarrierInfo _barrierInfo;
    std::optional<size_t> _startBarrierIndex;

    // For each queue store a vector of task indexes on this queue
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> _taskQueueTypeMap;

    // Default size of BARRIER FIFO depth, which corresponds to
    // VPU-FW guarantee on the level the barrier FIFO gets refilled in case
    // of initialization or barrier interrupt callback.
    // Use 1 in case barrier FIFO mode is not enabled
    static constexpr int64_t BARRIER_FIFO_SIZE = 4;

    const size_t _barrierFifoDepth;

    // For each barrier index store index of barrier with same PID
    // If there is no value set it means barrier is the first one to use given PID
    SmallVector<std::optional<size_t>> _barrierPidPrevUsageVec;

    // Default size of DMA FIFO handling outstanding independent DMA enqueues
    static constexpr int64_t DMA_OUTSTANDING_TRANSACTIONS = 64;
    // Configured DMA FIFO size. For testing purpose it might be lower
    const size_t _dmaFifoDepth;

    // Flag indicating if optimization for merging conecutive enqueues should be performed
    const bool _optimizeAndMergeEnqFlag;

    Logger _log;

    // For each task index store index of barrier at which it should be enqueued
    // If not barrier is set it means task can be enqueued at bootstrap
    SmallVector<std::optional<size_t>> _tasksEnqBar;

    // Indicate if during enqueuement algorithm should also verify if enqueues
    // would be possible to be ordered to guarantee task HW FIFO order and WorkItem
    // placement in adjacent way for the same barrier
    // TODO: To be turned off/removed once E#144867 is merged
    constexpr static bool performEnqueueOrderingCheck = true;

    // Internal class to manage access and rebuild of BarrierInfo task control map
    // It will store up to cache size number of instance of task control map
    // to reduce an overhead of rebuilding this structure if code that checks dependencies
    // needs to analyze different blocks of schedule
    static constexpr int64_t TASK_CONTROL_MAP_CACHE_SIZE = 2;
    class TaskControlMapCache {
    public:
        TaskControlMapCache(): _cacheSize(TASK_CONTROL_MAP_CACHE_SIZE) {
            _taskControlMapAndOffsetVec.resize(_cacheSize);
            _blockIdxOfTaskControlMap.resize(_cacheSize);
        }

        std::pair<SmallVector<llvm::BitVector>, size_t>& getTaskControlMapAndOffset(BarrierInfo& barrierInfo,
                                                                                    size_t blockIdx) {
            for (size_t i = 0; i < _cacheSize; i++) {
                if (_blockIdxOfTaskControlMap[i].has_value() && _blockIdxOfTaskControlMap[i].value() == blockIdx) {
                    return _taskControlMapAndOffsetVec[i];
                }
            }

            // Find slot index to store task control map. Pick the one which is either empty
            // or has the largest difference compared to what is needed (passed as argument)
            // Assumption is that in most cases neighboring blocks can be needed in parallel
            // and if difference is larger then they can be freed
            size_t indexToStore = 0;
            int maxBlockIndexDiff = 0;
            for (size_t i = 0; i < _cacheSize; i++) {
                if (!_blockIdxOfTaskControlMap[i].has_value()) {
                    indexToStore = i;
                    break;
                }
                int blockIndexDiff =
                        std::abs((static_cast<int>(blockIdx) - static_cast<int>(_blockIdxOfTaskControlMap[i].value())));
                if (blockIndexDiff > maxBlockIndexDiff) {
                    maxBlockIndexDiff = blockIndexDiff;
                    indexToStore = i;
                }
            }

            _blockIdxOfTaskControlMap[indexToStore] = blockIdx;
            _taskControlMapAndOffsetVec[indexToStore] = barrierInfo.buildTaskControlMap(blockIdx);
            return _taskControlMapAndOffsetVec[indexToStore];
        }

    private:
        SmallVector<std::pair<SmallVector<llvm::BitVector>, size_t>> _taskControlMapAndOffsetVec;
        SmallVector<std::optional<size_t>> _blockIdxOfTaskControlMap;
        size_t _cacheSize;
    };

    TaskControlMapCache _taskControlMapCache;
};

}  // namespace VPURT
}  // namespace vpux
