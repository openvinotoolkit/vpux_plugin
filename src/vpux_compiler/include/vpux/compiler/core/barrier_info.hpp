//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallSet.h>

namespace vpux {

class BarrierInfo {
public:
    // TaskSet is used to store barrier's producer/consumer task op index as well as task op's
    // wait/update barrier index, which is supposed to have better performance than BitVector when the data size is
    // small.
    using TaskSet = llvm::SmallSet<size_t, 16>;
    explicit BarrierInfo();
    explicit BarrierInfo(mlir::func::FuncOp func);
    friend class BarrierInfoTest;
    virtual ~BarrierInfo() = default;

public:
    void updateIR();
    void clearAttributes();
    TaskSet& getWaitBarriers(size_t taskInd);
    TaskSet& getUpdateBarriers(size_t taskInd);
    uint32_t getIndex(VPURT::TaskOp taskOp) const;
    uint32_t getIndex(VPURT::BarrierOpInterface barrierOp) const;
    virtual VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const;
    VPURT::BarrierOpInterface getBarrierOpAtIndex(size_t opIdx) const;
    void enableUnevenVariantSplit();

private:
    void addTaskOp(VPURT::TaskOp taskOp);
    void buildBarrierMaps(mlir::func::FuncOp func);
    /**
     * @brief Creates LUT storing control graph block index for every task
     */
    void buildTaskBlockMap();
    void setWaitBarriers(size_t taskIdn, const TaskSet& barriers);
    void setUpdateBarriers(size_t taskIdn, const TaskSet& barriers);
    void resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits);
    void resetBitMap(SmallVector<llvm::BitVector>& bitMap);
    bool producersControlsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                       const TaskSet& origConsumers, ArrayRef<TaskSet> origWaitBarriersMap);
    bool inImplicitQueueTypeDependencyList(const TaskSet& taskList);

    void optimizeBarrierProducers(size_t blockIdx);
    void optimizeBarrierConsumers(size_t blockIdx);
    void optimizeBarriersWithSameProducers(size_t blockIdx, bool checkValidSlotCount = true);

    bool inRange(const size_t low, const size_t high, const size_t val) const;
    void setBarrierMask(llvm::BitVector& mask, const BarrierInfo::TaskSet& barriers, size_t offset = 0);
    void splitBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots);
    void splitBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots);
    SmallVector<BarrierInfo::TaskSet> createProducerBatches(const BarrierInfo::TaskSet& waitBarriers,
                                                            size_t availableSlots, bool considerTaskExecutorType);
    void linearizeLegalParallelProducers(size_t taskInd, const BarrierInfo::TaskSet& parallelProducers,
                                         const BarrierInfo::TaskSet& parallelConsumers, size_t availableSlots,
                                         bool considerTaskExecutorType);

    /**
     * @brief check if total slot count in provided set of tasks is smaller or equal to availableSlots
     *
     * @param producers - set of tasks which slots should be checked
     * @param availableSlots - number of slots available for provided tasks (producers)
     * @return true when total slots count in producers <= availableSlots
     * @return false otherwise
     *
     */
    bool canMergeBarriersForTasks(const BarrierInfo::TaskSet& producers, size_t availableSlots);
    bool eliminateParallelWaitBarriers(size_t taskInd, size_t availableSlots, bool considerTaskExecutorType);
    void mergeLegalParallelProducers(size_t taskInd, const BarrierInfo::TaskSet& parallelProducers,
                                     const BarrierInfo::TaskSet& parallelConsumers);
    void linkTasksToBarriers(const TaskSet& tasksToAdd, const TaskSet& newBarriers, bool waitBarriers,
                             size_t availableSlots);

public:
    void logBarrierInfo();
    void optimizeBarriers(bool checkValidSlotCount = true, bool considerTaskFifoDependency = false);

    /**
     * @brief Eliminate tasks not controlled by barriers
     *
     * Eliminate tasks not controlled by barriers, by sharing wait / update barriers of parent / child DMA to create a
     * schedule fully managed by barriers  which simplifies runtime handling 1) update barriers: find task(s) without
     * update barrier, find next task (on the same FIFO) with update barrier(s) link update barrier(s) to all tasks that
     * don't have update barrier 2) wait barriers: find task(s) without wait barrier, find previous task (on the same
     * FIFO) with wait barrier(s) link wait barrier(s) to all tasks that don't have wait barrier
     *
     *    Bar0
     *    |             Bar0
     *    DMA-0            |
     *    |     =>  DMA-0 DMA-1
     *    DMA-1            |
     *    |             Bar1
     *    Bar1
     *
     */
    void shareWaitAndUpdateBarriers(size_t availableSlots);

    void buildTaskQueueTypeMap();
    std::pair<SmallVector<llvm::BitVector>, size_t> buildTaskControlMap(size_t blockIdx,
                                                                        bool considerTaskFifoDependency = true);
    virtual size_t getNumOfTasks() const;
    size_t getNumOfBarrierOps() const;
    virtual size_t getBarrierMaxVariantSum() const;
    static size_t getNumOfSlotsUsed(VPURT::TaskOp op);
    virtual size_t getNumOfSlotsUsedByTask(VPURT::TaskOp op) const;
    void resetBarrier(VPURT::BarrierOpInterface barrierOp);
    void resetBarrier(size_t barrierInd);
    size_t addNewBarrier(VPURT::BarrierOpInterface barrierOp);
    size_t addNewTaskOp(VPURT::TaskOp taskOp);
    bool controlPathExistsBetweenTasksInSameBlock(const SmallVector<llvm::BitVector>& taskControlMap, size_t taskAInd,
                                                  size_t taskBInd, bool biDirection = true) const;
    size_t getProducerSlotCount(VPURT::BarrierOpInterface barrierOp);
    size_t getConsumerSlotCount(VPURT::BarrierOpInterface barrierOp);
    void addProducer(size_t barrierInd, size_t taskInd);
    void addProducer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void addProducers(size_t barrierInd, const TaskSet& taskInds);
    void addConsumer(size_t barrierInd, size_t taskInd);
    void addConsumer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void addConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducer(size_t barrierInd, size_t taskInd);
    void removeConsumer(size_t barrierInd, size_t taskInd);

    void removeProducer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void removeConsumer(VPURT::BarrierOpInterface barrierOp, size_t taskInd);
    void removeProducers(size_t barrierInd, const TaskSet& taskInds);
    void removeConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducers(VPURT::BarrierOpInterface barrierOp, const TaskSet& taskInds);
    void removeConsumers(VPURT::BarrierOpInterface barrierOp, const TaskSet& taskInds);
    TaskSet& getBarrierProducers(VPURT::BarrierOpInterface barrierOp);
    TaskSet& getBarrierConsumers(VPURT::BarrierOpInterface barrierOp);
    TaskSet& getBarrierProducers(size_t barrierIdn);
    TaskSet& getBarrierConsumers(size_t barrierIdn);
    SmallVector<TaskSet> createLegalVariantBatches(const TaskSet& tasks, size_t availableSlots,
                                                   bool considerTaskExecutorType = false);
    std::optional<VPURT::TaskQueueType> haveSameImplicitDependencyTaskQueueType(const TaskSet& taskInds);
    bool canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                             const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                             ArrayRef<TaskSet> origWaitBarriersMap);
    SmallVector<TaskSet> getWaitBarriersMap();
    void splitControlGraphToBlocks(const size_t blockSize);
    bool verifyControlGraphSplit();
    void splitBarriersWithExceedingVariantCount(size_t availableSlots, size_t maxSlotsSum, size_t maxAvailableSlots);
    void splitBarrierProducers(size_t availableSlots, size_t maxSlotsSum, bool maxSlotsSumLimitEnabled);
    void splitBarrierConsumers(size_t availableSlots, size_t maxSlotsSum, bool maxSlotsSumLimitEnabled);
    bool ensureTasksDrivenBySingleBarrier(size_t availableSlots, bool mergeWaitBarriersIteratively = false,
                                          bool considerTaskExecutorType = false);
    void removeSyncTaskAttributes();
    bool hasBarrierDependency(size_t taskOneIdx, size_t taskTwoIdx, size_t& commonBarrier);
    bool isSyncPoint(size_t taskIdx);

    /**
     * @brief Get control graph block count
     * The number of control graph blocks is by 1 larger than the number of sync points.
     * Eg. if control graph split is not done (no sync points present) it retuns 1.
     *
     * @return Number of control graph blocks
     */
    size_t getControlGraphBlockCount() const;

    /**
     * @brief Get task indexes range for given control graph tasks block.
     *
     * If present, the returned tasks range include sync-points on both ends. For first and last block
     * the range includes first and last task indexes for lower and upper bound respectively.
     *
     * @param blockInd control graph block index
     * @param blockStartSyncPoint if false the lower bound sync-point is not included.
     * @param blockEndSyncPoint if false the upper bound sync-point is not included.
     * @return Lower and upper bound range for given task block
     */
    std::pair<size_t, size_t> getControlGraphBlockTaskRange(size_t blockInd, bool blockStartSyncPoint = true,
                                                            bool blockEndSyncPoint = true) const;
    /**
     * @brief Get vector of barriers associated with given control graph tasks block.
     *
     * If present, the returned tasks range include sync-points on both ends. For first and last block
     * the range includes first and last task indexes for lower and upper bound respectively.
     *
     * @param blockInd control graph block index
     * @param blockStartSyncPoint if false the lower bound sync-point is not included.
     * @param blockEndSyncPoint if false the upper bound sync-point is not included.
     * @param updateBarriers - if true return update barriers, otherwise return wait barriers for given blockInd
     * @return Barriers associated with given task block
     */
    SmallVector<size_t> getBarriersForTaskBlock(size_t blockInd, bool blockStartSyncPoint = false,
                                                bool blockEndSyncPoint = false, bool updateBarriers = true) const;
    /**
     * @brief Get control graph block index for given task index
     *
     * @param taskInd - task index
     * @return block index
     */
    size_t getControlGraphBlockIndex(size_t taskInd) const;

    /**
     * @brief Create barrier representation of dependencies implied FIFOs execution order.
     * The newly created dependencies are stored internally and can be removed by calling
     * @param blockIdx - task block index for which the dependencies should be generated
     * @param executorKind - set of FIFO executors that should be taken into account. By default all FIFOs are
     * considered.
     *
     * @see removeBarrierDependenciesImpliedByFIFO()
     *
     * @return Number of newly created connections between tasks (producers and consumers) and barriers
     */
    unsigned createBarrierDependenciesImpliedByFIFO(
            size_t blockIdx, std::optional<mlir::DenseSet<vpux::VPU::ExecutorKind>> executorKind = std::nullopt);

    /**
     * @brief Remove barrier representation of dependencies implied FIFOs execution order created by
     * @see createBarrierDependenciesImpliedByFIFO(size_t blockIdx)
     *
     * @return Number of removed connections between tasks (producers and consumers) and barriers
     */
    unsigned removeBarrierDependenciesImpliedByFIFO();

private:
    Logger _log;
    mlir::func::FuncOp _func;

    mlir::StringAttr _taskIndexAttrName;
    mlir::StringAttr _barrierIndexAttrName;
    mlir::StringAttr _syncTaskAttrName;

    SmallVector<VPURT::TaskOp> _allTaskOps;
    SmallVector<VPURT::BarrierOpInterface> _allBarrierOps;

    bool _enableUnevenVariantSplit{false};

    // Control graph split variables
    // After graph split is performed in splitControlGraphToBlocks method
    // the variables store information about the resulting graph blocks
    //
    // Example:  number of all task: 1200, split size: 500
    // Graph is split into 3 blocks with tasks indexes
    // being placed in blocks in following way
    //  block 0:    0 -  499
    //  block 1:  500 -  999
    //  block 2: 1000 - 1199
    // If control graph is split some methods can use it to reduce overhead of processing barriers (smaller memory)
    // Ids of tasks that are synchronization points between blocks. From example above that would be 499 and 999
    // If control graph split is not done, this vector should be empty.
    SmallVector<size_t> _syncTasksIds;
    // _taskToBlockMap stores indexes of tasks blocks for every task in the graph. This is initialized based on
    // _syncTasksIds. The map is built in order to improve performance.
    SmallVector<size_t> _taskToBlockMap;
    // Note:
    //  - task produces its update barriers
    //  - task consumes its wait barriers

    // indexOf(VPURT::BarrierOpInterface) 'is produced by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierProducerMap;
    // indexOf(VPURT::BarrierOpInterface) 'is consumed by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierConsumerMap;

    // indexOf(VPURT::TaskOp) 'waits for' [ indexOf(VPURT::BarrierOpInterface)... ].
    SmallVector<TaskSet> _taskWaitBarriers;
    // indexOf(VPURT::TaskOp) 'updates' [ indexOf(VPURT::BarrierOpInterface)... ].
    SmallVector<TaskSet> _taskUpdateBarriers;

    // If optimization is to be done taking into account FIFO dependencies, these dependencies are temporarily stored.
    // The tuple contains barrier index and parent and child task indexes.
    SmallVector<std::tuple<size_t, size_t, size_t>> _fifoDependencies;

    // Initialize below structure with buildTaskQueueTypeMap()
    // indexOf(VPURT::TaskQueueType) 'contains' [ indexOf(VPURT::TaskOp)... ].
    std::map<VPURT::TaskQueueType, llvm::BitVector> _taskQueueTypeMap;
};

using BarrierMap = SmallVector<SmallVector<size_t>>;
class BarrierInfoTest : public BarrierInfo {
public:
    struct BarrierMaps {
        BarrierMap barrierProducerMap = {};
        BarrierMap barrierConsumerMap = {};
        BarrierMap taskUpdateBarriers = {};
        BarrierMap taskWaitBarriers = {};
        size_t nTasks = 0;
        size_t nBarriers = 0;
        SmallVector<size_t> syncTasksIds = {};
        std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> taskQueueTypeMap = {};
    };

    explicit BarrierInfoTest(mlir::func::FuncOp func);
    explicit BarrierInfoTest(BarrierInfoTest::BarrierMaps& barrierMaps);
    void initializeBarrierMaps(BarrierInfoTest::BarrierMaps& barrierMaps);
    void setTaskQueueTypeMap(const std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps);
    void setMaxVariantCountPerBarrier(size_t variantCount);
    size_t getNumOfTasks() const override;
    size_t getBarrierMaxVariantSum() const override;
    size_t getNumOfSlotsUsedByTask(VPURT::TaskOp op) const override;
    VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const override;
    BarrierInfoTest::BarrierMaps optimizeBarrierProducers(size_t blockIdx);
    BarrierInfoTest::BarrierMaps optimizeBarriersWithSameProducers(size_t blockIdx, bool checkValidSlotCount = true);
    BarrierInfoTest::BarrierMaps optimizeBarrierConsumers(size_t blockIdx);
    BarrierInfoTest::BarrierMaps getBarrierMaps();
    BarrierInfoTest::BarrierMaps optimizeBarriers(bool checkValidSlotCount = true,
                                                  bool considerTaskFifoDependency = false);
    SmallVector<BarrierInfo::TaskSet> toTaskSet(SmallVector<SmallVector<size_t>>& map);
    SmallVector<SmallVector<size_t>> toTaskVec(SmallVector<BarrierInfo::TaskSet>& map);

private:
    size_t _maxVariantCountPerBarrier = 0;
};

//
// Helper routines to work with BarrierInfoTest
//
void fillProducersAndConsumers(BarrierInfoTest::BarrierMaps& barrierMaps);

}  // namespace vpux
