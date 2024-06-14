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

public:
    void updateIR();
    void clearAttributes();
    TaskSet& getWaitBarriers(size_t taskInd);
    TaskSet& getUpdateBarriers(size_t taskInd);
    uint32_t getIndex(VPURT::TaskOp taskOp) const;
    uint32_t getIndex(VPURT::DeclareVirtualBarrierOp barrierOp) const;
    virtual VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const;
    VPURT::DeclareVirtualBarrierOp getBarrierOpAtIndex(size_t opIdx) const;

private:
    void addTaskOp(VPURT::TaskOp taskOp);
    void buildBarrierMaps(mlir::func::FuncOp func);
    void setWaitBarriers(size_t taskIdn, const TaskSet& barriers);
    void setUpdateBarriers(size_t taskIdn, const TaskSet& barriers);
    void resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits);
    void resetBitMap(SmallVector<llvm::BitVector>& bitMap);
    bool producersControlsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                       const TaskSet& origConsumers, ArrayRef<TaskSet> origWaitBarriersMap);
    bool inImplicitQueueTypeDependencyList(const TaskSet& taskList);

    void optimizeBarrierProducers(size_t blockIdx);
    void optimizeBarrierConsumers(size_t blockIdx);
    void optimizeBarriersWithSameProducers(size_t blockIdx);

    bool inRange(const unsigned low, const unsigned high, const unsigned val) const;
    void setBarrierMask(llvm::BitVector& mask, const BarrierInfo::TaskSet& barriers, size_t offset = 0);

public:
    void logBarrierInfo();
    void optimizeBarriers();

    void buildTaskQueueTypeMap(bool considerTaskFifoDependency = true);
    std::pair<SmallVector<llvm::BitVector>, size_t> buildTaskControlMap(size_t blockIdx,
                                                                        bool considerTaskFifoDependency = true);
    size_t getNumOfTasks() const;
    size_t getNumOfVirtualBarriers() const;
    virtual size_t getBarrierMaxVariantSum() const;
    static size_t getNumOfSlotsUsed(VPURT::TaskOp op);
    virtual size_t getNumOfSlotsUsedByTask(VPURT::TaskOp op) const;
    void resetBarrier(VPURT::DeclareVirtualBarrierOp barrierOp);
    void resetBarrier(size_t barrierInd);
    size_t addNewBarrier(VPURT::DeclareVirtualBarrierOp barrierOp);
    bool controlPathExistsBetweenTasksInSameBlock(const SmallVector<llvm::BitVector>& taskControlMap, size_t taskAInd,
                                                  size_t taskBInd, bool biDirection = true) const;
    size_t getProducerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp);
    size_t getConsumerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp);
    void addProducer(size_t barrierInd, size_t taskInd);
    void addProducer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd);
    void addProducers(size_t barrierInd, const TaskSet& taskInds);
    void addConsumer(size_t barrierInd, size_t taskInd);
    void addConsumer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd);
    void addConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducer(size_t barrierInd, size_t taskInd);
    void removeConsumer(size_t barrierInd, size_t taskInd);
    void removeProducer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd);
    void removeConsumer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd);
    void removeProducers(size_t barrierInd, const TaskSet& taskInds);
    void removeConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds);
    void removeConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds);
    TaskSet& getBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp);
    TaskSet& getBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp);
    TaskSet& getBarrierProducers(size_t barrierIdn);
    TaskSet& getBarrierConsumers(size_t barrierIdn);
    SmallVector<TaskSet> createLegalVariantBatches(const TaskSet& tasks, size_t availableSlots);
    std::optional<VPURT::TaskQueueType> haveSameImplicitDependencyTaskQueueType(const TaskSet& taskInds);
    bool canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                             const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                             ArrayRef<TaskSet> origWaitBarriersMap);
    SmallVector<TaskSet> getWaitBarriersMap();
    void splitControlGraphToBlocks(size_t blockSize);
    bool verifyControlGraphSplit();
    void removeSyncTaskAttributes();
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

private:
    Logger _log;
    mlir::func::FuncOp _func;

    mlir::StringAttr _taskIndexAttrName;
    mlir::StringAttr _barrierIndexAttrName;
    mlir::StringAttr _syncTaskAttrName;

    SmallVector<VPURT::TaskOp> _allTaskOps;
    SmallVector<VPURT::DeclareVirtualBarrierOp> _allBarrierOps;

    // Store maximal number of tasks that are present in single control
    // graph block after split performed in splitControlGraphToBlocks method
    // Example:  number of all task: 1200, split size: 500
    // Graph is split into 3 blocks with tasks indexes
    // being placed in blocks in following way
    //  block 0:    0 -  499
    //  block 1:  500 -  999
    //  block 2: 1000 - 1199
    // Value of 0 means no split performed
    // If control graph is split some methods can use it to reduce overhead of processing barriers (smaller memory)
    size_t _controlGraphBlockSize = 0;
    // Ids of tasks that are synchronization points between blocks. From example above that would be 499 and 999
    SmallVector<size_t> _syncTasksIds;

    // Note:
    //  - task produces its update barriers
    //  - task consumes its wait barriers

    // indexOf(VPURT::DeclareVirtualBarrierOp) 'is produced by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierProducerMap;
    // indexOf(VPURT::DeclareVirtualBarrierOp) 'is consumed by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierConsumerMap;

    // indexOf(VPURT::TaskOp) 'waits for' [ indexOf(VPURT::DeclareVirtualBarrierOp)... ].
    SmallVector<TaskSet> _taskWaitBarriers;
    // indexOf(VPURT::TaskOp) 'updates' [ indexOf(VPURT::DeclareVirtualBarrierOp)... ].
    SmallVector<TaskSet> _taskUpdateBarriers;

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
        size_t Ntasks = 0;
        size_t Nbarriers = 0;
        size_t controlGraphBlockSize = 0;
        SmallVector<size_t> syncTasksIds = {};
    };

    explicit BarrierInfoTest(BarrierInfoTest::BarrierMaps& barrierMaps);
    void initializeBarrierMaps(BarrierInfoTest::BarrierMaps& barrierMaps);
    void setMaxVariantCountPerBarrier(size_t variantCount);
    size_t getBarrierMaxVariantSum() const override;
    size_t getNumOfSlotsUsedByTask(VPURT::TaskOp op) const override;
    VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const override;
    BarrierInfoTest::BarrierMaps optimizeBarrierProducers(size_t blockIdx);
    BarrierInfoTest::BarrierMaps optimizeBarriersWithSameProducers(size_t blockIdx);
    BarrierInfoTest::BarrierMaps optimizeBarrierConsumers(size_t blockIdx);
    BarrierInfoTest::BarrierMaps getOptimizedMaps();
    BarrierInfoTest::BarrierMaps optimizeBarriers();
    SmallVector<BarrierInfo::TaskSet> toTaskSet(SmallVector<SmallVector<size_t>>& map);
    SmallVector<SmallVector<size_t>> toTaskVec(SmallVector<BarrierInfo::TaskSet>& map);

private:
    size_t _maxVariantCountPerBarrier = 0;
};

}  // namespace vpux
