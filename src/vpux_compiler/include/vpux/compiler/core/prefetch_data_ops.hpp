//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>

namespace vpux {

class PrefetchDataOps final {
    using scheduledOps = vpux::FeasibleMemoryScheduler::ScheduledOpInfoVec;

public:
    explicit PrefetchDataOps(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo);

public:
    bool enableDataOpPrefetching();

private:
    struct CycleInfo {
        size_t opIdx_;
        size_t cycleBegin_;
        size_t cycleCost_;
        VPU::ExecutorKind executorKind_;

        CycleInfo() = default;
        CycleInfo(size_t opIdx, size_t cycleBegin, size_t cycleCost,
                  VPU::ExecutorKind executorKind = VPU::ExecutorKind::DMA_NN)
                : opIdx_(opIdx), cycleBegin_(cycleBegin), cycleCost_(cycleCost), executorKind_(executorKind){};

        size_t getOpIdx() const {
            return opIdx_;
        }

        size_t getCycleBegin() const {
            return cycleBegin_;
        }

        size_t getCycleCost() const {
            return cycleCost_;
        }

        size_t getCycleEnd() const {
            return cycleBegin_ + cycleCost_;
        }

        VPU::ExecutorKind getExecutorKind() const {
            return executorKind_;
        }
    };

private:
    void init();
    // operation info
    bool isScheduled(size_t opIdx);
    bool hasDependencies(size_t opIdx);
    bool dependenciesScheduled(size_t opIdx);
    size_t getOperationCycleCost(size_t opIdx);
    size_t getDependencyCycleEnd(size_t opIdx);

    // pipeline utilities
    bool prefetchPipelineStallsExist();
    void reducePrefetchPipelineStalls(size_t targetRollback);
    size_t getNextFreePipelineCycle(VPU::ExecutorKind executorKind, bool prefetchFIFO = false);
    void updatePrefetchPipeline(size_t fromCycle, size_t toCycle, size_t adjustCycles);

    // 1) prefetch mode DMA scheduling
    // actually schedule op
    CycleInfo scheduleOp(size_t opIdx, size_t cycleBegin, size_t cycleCost, VPU::ExecutorKind executorKind,
                         bool prefetchFIFO = false);
    // schedule prefetch DMAs during optimal cycles in prefetch pipeline
    void schedulePrefetchDMA(size_t opIdx, size_t optimalCycleEnd);

    // 2) definedDMAOrder mode DMA scheduling
    // schedule DMA in next available cycle in DMA pipeline
    void scheduleDMA(size_t opIdx, size_t optimalCycleEnd = std::numeric_limits<size_t>::min());
    // schedule DMAs early in parallel to compute
    void prefetchDataOps(size_t dmaTargetEndCycle);

    // schedule dependencies, free DMAs in prefetch pipeline, DMAs with dependencies in DMA pipeline
    size_t scheduleDependenciesForCompute(size_t opIdx, VPU::ExecutorKind executorKind);
    // schedule dependencies for compute and the compute during earliest cycle
    void scheduleComputeOperation(size_t opIdx, VPU::ExecutorKind executorKind);
    // perform scheduling compute ops, also schedule dependencies
    // two modes: 1) prefetch and 2) definedDMAOrder
    void performCycleScheduling();

    // create a pipeline for all data ops
    void createDataOpPipeline();
    // sort vector
    void sortOps(SmallVector<CycleInfo>& toBeSorted);
    // create a new order for IR
    SmallVector<CycleInfo> getNewOrder();
    // reorder IR such that prefetch DMAs are before compute
    void reorderToPrefetch(ArrayRef<CycleInfo> sortedOpCycles);

private:
    Logger _log;

    // incoming objects
    scheduledOps _scheduledOps;
    AsyncDepsInfo& _depsInfo;

    // info about ops
    mlir::DenseSet<size_t> _dataOpIdx;
    mlir::DenseSet<size_t> _computeExecutorKindOpIdx;
    mlir::DenseMap<size_t, size_t> _operationCycleCost;

    // used for prefetching
    bool _prefetchOpsDefined = false;
    // for compute op prefetch DMAs until next X compute executor kind op cycle begin
    const size_t _advanceComputeExecutorKindOpsForPrefetch = 2;
    // ratio for number of dynamic spills to all DMAs to enable prefetching
    const double _dynamicSpillRatio = 0.3;
    const size_t _dynamicSpillMinCount = 1200;
    size_t _dmaCount = 0;
    size_t _dynamicSpillCount = 0;
    // used to represent a stall
    const size_t _cycleInfoStallDummyOp = std::numeric_limits<size_t>::max();
    // FIFO or pipeline storage
    SmallVector<CycleInfo> _prefetchPipeline;
    SmallVector<CycleInfo> _prefetchPipelineStalls;
    mlir::DenseMap<size_t, CycleInfo> _operationCycles;
    mlir::DenseMap<VPU::ExecutorKind, SmallVector<CycleInfo>> _executorPipelineCycles;
};

}  // namespace vpux
