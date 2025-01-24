//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/dma.hpp"

using namespace vpux;
namespace {

using ExecutionGroup = llvm::SmallVector<size_t>;
using ExecutionGroupList = llvm::SmallVector<ExecutionGroup>;
enum class MinMaxOption { Min, Max };

//
//  LegalizeScheduleForWlmFetchDmasPass
//
class LegalizeScheduleForWlmFetchDmasPass final :
        public VPUIP::arch40xx::LegalizeScheduleForWlmFetchDmasBase<LegalizeScheduleForWlmFetchDmasPass> {
public:
    explicit LegalizeScheduleForWlmFetchDmasPass(const int virtualBarrierThreshold, Logger log)
            : _virtualBarrierThreshold(virtualBarrierThreshold) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    int _virtualBarrierThreshold;
    void safeRunOnFunc() final;

    bool isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx);
    VPURT::TaskOp findLastDmaBeforeExecGroup(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findFirstDmaAfterExecGroup(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findDMAsThroughBarriersBFS(size_t startBarrier, BarrierInfo& barrierInfo, MinMaxOption option,
                                             bool bfsDirUp);
    void createSWTaskExecutionGroups(BarrierInfo& barrierInfo,
                                     std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& swQueue, size_t tilesCount);
    void createDPUTaskExecutionGroups(BarrierInfo& barrierInfo,
                                      std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& dpuQueue,
                                      size_t tilesCount);

    VPURT::TaskOp createDummyDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                                 BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas);

    void insertDMAForFetchTasks(DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups,
                                VPU::ExecutorKind executorKind, mlir::Operation* bufferInsertionPoint,
                                mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                SmallVector<VPURT::TaskOp>& dummyDmas, size_t tilesCount,
                                SmallVector<std::pair<size_t, size_t>>& blockRange);

    SmallVector<size_t> getDmasUpdatingBarriers(llvm::DenseSet<size_t>& barriers, BarrierInfo& barrierInfo);

private:
    // Will be initialized in safeRunOnFunc(), this is done to suppress the UNINIT_CTOR warning
    size_t _maxKernelInvoCount = 0;
    size_t _maxKernelRangeCount = 0;
    size_t _maxInvarCount = 0;
    size_t _maxVarCount = 0;
    size_t _numAllTaskOps = 0;

    VPURT::TaskOp _firstDMATaskOp;
    VPURT::TaskOp _lastDMATaskOp;
    DenseMap<VPURT::TaskQueueType, ExecutionGroupList> _listOfSWExecutionGroups;
    DenseMap<VPURT::TaskQueueType, ExecutionGroupList> _listOfDPUExecutionGroups;
};

void updateBarriersForDma(SmallVector<size_t>& consumes, SmallVector<size_t>& producesIn, VPURT::TaskOp dmaOp,
                          BarrierInfo& barrierInfo) {
    auto dmaIdx = barrierInfo.getIndex(dmaOp);
    for (auto pIn : producesIn) {
        barrierInfo.addProducer(pIn, dmaIdx);
    }
    for (auto consume : consumes) {
        barrierInfo.addConsumer(consume, dmaIdx);
    }
}

void updateBarriersForDma(SmallVector<mlir::Value>& consumes, SmallVector<mlir::Value>& producesIn, VPURT::TaskOp dmaOp,
                          BarrierInfo& barrierInfo) {
    auto dmaIdx = barrierInfo.getIndex(dmaOp);
    for (auto pIn : producesIn) {
        auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(pIn.getDefiningOp());
        barrierInfo.addProducer(barrOp, dmaIdx);
    }
    for (auto consume : consumes) {
        auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(consume.getDefiningOp());
        barrierInfo.addConsumer(barrOp, dmaIdx);
    }
}

// Fetch tasks are only attached to DMAs on port 0 and list 0 in later dialect
// In this context supportedDMA is a DMA which has channel DDR and port 0
bool isDMAOnSupportedPortAndChannel(VPURT::TaskOp dmaTaskOp) {
    if (auto dma = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(dmaTaskOp.getInnerTaskOp())) {
        // Check if this is DMA on Port 0 Channel DDR
        if (vpux::getDMAQueueIdEncoding(0, VPUIP::DmaChannelType::DDR) ==
            vpux::getDMAQueueIdEncoding(dma.getPortVal().value_or(0), dma.getChannelType())) {
            return true;
        }
    }
    return false;
}

mlir::Value createDummyBuffer(mlir::OpBuilder& builder, mlir::Operation* insertionPoint) {
    auto ctx = builder.getContext();
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (insertionPoint != nullptr) {
        builder.setInsertionPoint(insertionPoint);
    }

    const auto nameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    const auto ddrSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, nameAttr);
    const auto layout = DimsOrder::NCHW.toAffineMap(ctx);

    auto zeroBufferMemref = mlir::MemRefType::get({0, 0, 0, 0}, builder.getI32Type(), layout, ddrSymbolAttr);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), zeroBufferMemref, VPURT::BufferSection::DDR,
                                                  0);
}

/*
Function returns the sibling task on the last tile by default
When a value of tile is provided it returns the sibling task on questioned tile
If the task is not running on the asked tile e.g. SHV running on single cluster then it returns SIZE_MAX

In following case when passed the index of Task 0 it will return Task 3
    Task 0 (CMX, 0) .. Task 1 (CMX, 1) .. Task 2 (CMX, 2) .. Task 3 (CMX, 3)
*/
size_t getSiblingTaskOpOnTile(size_t inputTaskOpIdx, BarrierInfo& barrierInfo, size_t tile = SIZE_MAX) {
    if (tile == 0) {
        return inputTaskOpIdx;
    }
    auto inputTaskOp = barrierInfo.getTaskOpAtIndex(inputTaskOpIdx);

    auto getTileIndex = [&](mlir::Operation* op) -> size_t {
        auto taskOp = llvm::cast<VPURT::TaskOp>(op);
        // For the usage pattern of tile index in this function all DMAs should have 0 tile index
        if (taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
            return 0;
        }
        return VPURT::getTaskQueueType(taskOp, false).id;
    };

    mlir::Operation* prevOp = nullptr;
    mlir::Operation* currentOp = inputTaskOp->getNextNode();

    while (currentOp != nullptr) {
        size_t tileIndex = getTileIndex(currentOp);

        // If the requested tile is not SIZE_MAX, check for a match
        if (tile != SIZE_MAX && tileIndex == tile) {
            auto siblingOp = mlir::cast<VPURT::TaskOp>(currentOp);
            return barrierInfo.getIndex(siblingOp);
        }

        // If tile index is 0, return the previous operation
        // If prev operation is null then we have case when the inputTaskOpIdx is only running on 1 tile
        if (tileIndex == 0) {
            if (prevOp != nullptr) {
                auto prev = mlir::cast<VPURT::TaskOp>(prevOp);
                return barrierInfo.getIndex(prev);
            } else {
                return inputTaskOpIdx;
            }
        }

        // Move to the next node
        prevOp = currentOp;
        currentOp = currentOp->getNextNode();
    }

    // If the requested tile was provided but not found, return SIZE_MAX
    if (tile != SIZE_MAX) {
        return SIZE_MAX;
    }
    return 0;
}

template <typename T>
bool compareVPURTOpPosition(const T& lhs, const T& rhs, const BarrierInfo& barrierInfo, bool useIROrder = false) {
    static_assert(std::is_same_v<T, mlir::Value> || std::is_same_v<T, VPURT::TaskOp> ||
                          std::is_same_v<T, VPURT::DeclareVirtualBarrierOp> || std::is_same_v<T, size_t>,
                  "Unsupported type for comparison");

    if constexpr (std::is_same_v<T, mlir::Value>) {
        auto lfsOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(lhs.getDefiningOp());
        auto rhsOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(rhs.getDefiningOp());
        return barrierInfo.getIndex(lfsOp) < barrierInfo.getIndex(rhsOp);
    } else if constexpr (std::is_same_v<T, VPURT::DeclareVirtualBarrierOp>) {
        return barrierInfo.getIndex(lhs) < barrierInfo.getIndex(rhs);
    } else if constexpr (std::is_same_v<T, size_t>) {
        return lhs < rhs;
    } else if constexpr (std::is_same_v<T, VPURT::TaskOp>) {
        if (useIROrder) {
            // Use IR order for comparison
            return lhs->isBeforeInBlock(rhs);
        } else {
            return barrierInfo.getIndex(lhs) < barrierInfo.getIndex(rhs);
        }
    }
}

// Function to find min or max position in a vector of TaskOps
VPURT::TaskOp findMinMaxPosition(const SmallVector<size_t>& dmas, BarrierInfo& barrierInfo, MinMaxOption option) {
    if (dmas.empty()) {
        return nullptr;
    }

    auto comparePositions = [](size_t lhs, size_t rhs) {
        return lhs < rhs;
    };

    if (option == MinMaxOption::Min) {
        auto minPosIt = std::min_element(dmas.begin(), dmas.end(), comparePositions);
        return barrierInfo.getTaskOpAtIndex(*minPosIt);
    } else {
        auto maxPosIt = std::max_element(dmas.begin(), dmas.end(), comparePositions);
        return barrierInfo.getTaskOpAtIndex(*maxPosIt);
    }
}

// Check if two list of barriers have any barrier in common
std::optional<mlir::Value> findCommonBarrier(const SmallVector<mlir::Value>& barrierListOne,
                                             const SmallVector<mlir::Value>& barrierListTwo, BarrierInfo& barrierInfo) {
    mlir::DenseSet<mlir::Value> elements(barrierListOne.begin(), barrierListOne.end());
    SmallVector<mlir::Value> commonBarriers;

    for (mlir::Value barr : barrierListTwo) {
        if (elements.find(barr) != elements.end()) {
            commonBarriers.push_back(barr);
        }
    }

    if (commonBarriers.empty()) {
        return std::nullopt;
    }

    llvm::sort(commonBarriers, [&](const auto& lhs, const auto& rhs) {
        return compareVPURTOpPosition(lhs, rhs, barrierInfo, true);
    });

    return commonBarriers[commonBarriers.size() - 1];
}

// Return last task that updates series of barriers
// As we check for the last DMA, sort the vector and use the last one
mlir::Operation* findLastTaskToUpdate(mlir::ValueRange barriers, BarrierInfo& barrierInfo) {
    auto barrierVector = to_small_vector(barriers);

    // Collect all tasks that update any barrier in barrierVector
    SmallVector<VPURT::TaskOp> allUpdatingTasks;
    for (auto barrierOp : barrierVector) {
        // Lambda to check if a task updates the current barrierOp
        auto validUser = [&barrierOp, &barrierInfo](VPURT::TaskOp op) -> bool {
            auto updateBarrierList = to_small_vector(op.getUpdateBarriers());
            return findCommonBarrier(updateBarrierList, {barrierOp}, barrierInfo).has_value();
        };

        // Get all users of the current barrierOp and filter based on the validUser condition
        auto bOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrierOp.getDefiningOp());
        for (auto usr : bOp.getResult().getUsers()) {
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(usr);
            if (taskOp && validUser(taskOp)) {
                allUpdatingTasks.push_back(taskOp);
            }
        }
    }

    // Sort all updating tasks collected based on position to find the last one
    if (!allUpdatingTasks.empty()) {
        llvm::sort(allUpdatingTasks, [&](const auto& lhs, const auto& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo, true);
        });
        return allUpdatingTasks[allUpdatingTasks.size() - 1];
    }

    return nullptr;
}

// Create new barrier and add producer and consumer
VPURT::DeclareVirtualBarrierOp createNewBarrier(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                mlir::Operation* insertionPoint, VPURT::TaskOp producer,
                                                VPURT::TaskOp consumer) {
    builder.setInsertionPointAfter(insertionPoint);
    auto newBarrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(insertionPoint->getLoc());
    barrierInfo.addNewBarrier(newBarrierOp);

    if (producer != nullptr) {
        barrierInfo.addProducer(newBarrierOp, barrierInfo.getIndex(producer));
    }

    if (consumer != nullptr) {
        barrierInfo.addConsumer(newBarrierOp, barrierInfo.getIndex(consumer));
    }

    return newBarrierOp;
}

// Returns a DMA which copies 0 len data from DDR to DDR
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::createDummyDma(mlir::OpBuilder& builder, mlir::Value inputBuf,
                                                                  mlir::Value outputBuf, BarrierInfo& barrierInfo,
                                                                  SmallVector<VPURT::TaskOp>& dummyDmas) {
    auto ctx = builder.getContext();
    auto dummyDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_dummy_dma"));

    auto newDMAOp =
            VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, mlir::ValueRange({}), mlir::ValueRange({}), dummyDmaLoc,
                                                  inputBuf, outputBuf, 0, false, false, nullptr, nullptr);
    auto newDMA = newDMAOp->getParentOfType<VPURT::TaskOp>();
    barrierInfo.addNewTaskOp(newDMA);
    dummyDmas.push_back(newDMA);
    return newDMA;
}

bool LegalizeScheduleForWlmFetchDmasPass::isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx) {
    auto taskOp = barrierInfo.getTaskOpAtIndex(dmaIdx);
    return taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN && isDMAOnSupportedPortAndChannel(taskOp) &&
           dmaIdx < _numAllTaskOps;
}

void LegalizeScheduleForWlmFetchDmasPass::createDPUTaskExecutionGroups(
        BarrierInfo& barrierInfo, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& dpuQueue, size_t tilesCount) {
    VPURT::TaskQueueType queueType;
    queueType.type = VPU::ExecutorKind::DPU;

    for (size_t tile = 0; tile < tilesCount; ++tile) {
        queueType.id = tile;
        auto tileDpuQueue = dpuQueue[queueType];

        ExecutionGroup execGroup;
        uint32_t execGroupVariantCount = 0;
        uint32_t execGroupInVariantCount = 0;

        auto& tempVector = _listOfDPUExecutionGroups[queueType];

        for (auto taskIdx : tileDpuQueue) {
            auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
            uint32_t dpuSize = 0;

            for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<VPUIP::NCEClusterTaskOp>())) {
                const auto& dpuTasks = to_small_vector(op.getVariants().getOps<VPUIP::DPUTaskOp>());
                dpuSize = dpuTasks.size();
                execGroupVariantCount += dpuTasks.size();
                execGroupInVariantCount++;
            }

            // Check if the current task exceeds either variant or invariant limits
            if (execGroupVariantCount > _maxVarCount || execGroupInVariantCount > _maxInvarCount) {
                // Push current group to the list
                tempVector.push_back(execGroup);

                // Start a new group
                execGroup.clear();
                execGroup.push_back(taskIdx);

                // Reset counts for the new group
                execGroupVariantCount = dpuSize;
                execGroupInVariantCount = 1;
            } else {
                // Otherwise, continue adding to the current group
                execGroup.push_back(taskIdx);
            }
        }

        // Push any remaining group
        if (!execGroup.empty()) {
            tempVector.push_back(execGroup);
        }
    }
}

// Create SW execution groups of tasks for each cluster
void LegalizeScheduleForWlmFetchDmasPass::createSWTaskExecutionGroups(
        BarrierInfo& barrierInfo, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& swQueue, size_t tilesCount) {
    VPURT::TaskQueueType queueType;
    queueType.type = VPU::ExecutorKind::SHAVE_ACT;

    for (size_t tile = 0; tile < tilesCount; ++tile) {
        queueType.id = tile;
        auto tileSWQueue = swQueue[queueType];

        ExecutionGroup execGroup;
        uint32_t execGroupInvoCount = 0;
        uint32_t execGroupRangeCount = 0;

        auto& tempVector = _listOfSWExecutionGroups[queueType];

        for (auto taskIdx : tileSWQueue) {
            auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
            auto ops = taskOp.getBody().getOps<VPUIP::SwKernelOp>();
            auto count = std::distance(ops.begin(), ops.end());

            // Update counts with the current task
            execGroupInvoCount += count;
            execGroupRangeCount += count;

            // Check if current group exceeds kernel invocation or range limits
            if (execGroupInvoCount > _maxKernelInvoCount || execGroupRangeCount > _maxKernelRangeCount) {
                // Push the current group to the vector
                tempVector.push_back(execGroup);

                // Start a new group
                execGroup.clear();
                execGroup.push_back(taskIdx);

                // Reset counts for the new group
                execGroupInvoCount = count;
                execGroupRangeCount = count;
            } else {
                // Otherwise, continue adding to the current group
                execGroup.push_back(taskIdx);
            }
        }

        // Push any remaining group
        if (!execGroup.empty()) {
            tempVector.push_back(execGroup);
        }
    }
}

VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findDMAsThroughBarriersBFS(size_t startBarrier,
                                                                              BarrierInfo& barrierInfo,
                                                                              MinMaxOption option, bool bfsDirUp) {
    std::queue<size_t> barriersToExplore;
    barriersToExplore.push(startBarrier);
    std::unordered_set<size_t> visitedBarriers;
    SmallVector<size_t> possibleDMAs;

    while (!barriersToExplore.empty()) {
        auto currentBarrier = barriersToExplore.front();
        barriersToExplore.pop();

        if (visitedBarriers.count(currentBarrier)) {
            continue;  // Skip already visited barriers to avoid cycles
        }
        visitedBarriers.insert(currentBarrier);

        SmallVector<size_t> currentOps = bfsDirUp ? to_small_vector(barrierInfo.getBarrierProducers(currentBarrier))
                                                  : to_small_vector(barrierInfo.getBarrierConsumers(currentBarrier));

        llvm::sort(currentOps, [&](const size_t& lhs, const size_t& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo);
        });

        for (auto op : currentOps) {
            auto nextBarriers = bfsDirUp ? barrierInfo.getWaitBarriers(op) : barrierInfo.getUpdateBarriers(op);

            auto filteredRange = currentOps | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                                     return isValidDMA(barrierInfo, dmaIdx);
                                 });
            auto filteredVector = to_small_vector(filteredRange);
            possibleDMAs.insert(possibleDMAs.end(), filteredVector.begin(), filteredVector.end());

            if (!possibleDMAs.empty()) {
                return findMinMaxPosition(possibleDMAs, barrierInfo, option);
            }

            for (auto barrier : nextBarriers) {
                barriersToExplore.push(barrier);
            }
        }
    }

    return (option == MinMaxOption::Max) ? _firstDMATaskOp : _lastDMATaskOp;
}

/*
  452              650
   |                |
   v                v
  +------------------+
  | Execution   Group|
  +------------------+
9184               9277

Find First DMA for a group tries to find a DMA which is the first DMA that can be used as a marker to tell the
Execution Group has finished execution e.g. in above case 9277 is the last barrier in execution group. The function
looks for a DMA which is on tile 0 list 0 and waits for 9277

If there is no DMA which waits for 9277 using BFS check if the user's barrier (barrier->task->barrier) has a DMA user
and use it as first DMA
*/
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findFirstDmaAfterExecGroup(BarrierInfo& barrierInfo,
                                                                              ExecutionGroup& executionGroup) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> updateBarriers;
    for (const auto& taskIndex : executionGroup) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIndex);
        auto upBarriers = taskOp.getUpdateBarriers();
        for (auto updateBarrier : upBarriers) {
            auto updateBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(updateBarrier.getDefiningOp());
            updateBarriers.push_back(updateBarrierOp);
        }
    }

    llvm::sort(updateBarriers, [&](const auto& lhs, const auto& rhs) {
        return compareVPURTOpPosition(lhs, rhs, barrierInfo);
    });

    auto lastBarrier = barrierInfo.getIndex(updateBarriers[updateBarriers.size() - 1]);
    SmallVector<size_t> possibleWaitingDMAs;

    auto consumers = to_small_vector(barrierInfo.getBarrierConsumers(lastBarrier));
    auto filteredRange = consumers | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                             return isValidDMA(barrierInfo, dmaIdx);
                         });

    auto filteredVector = to_small_vector(filteredRange);
    possibleWaitingDMAs.insert(possibleWaitingDMAs.end(), filteredVector.begin(), filteredVector.end());

    if (!possibleWaitingDMAs.empty()) {
        return findMinMaxPosition(possibleWaitingDMAs, barrierInfo, MinMaxOption::Min);
    }

    return findDMAsThroughBarriersBFS(lastBarrier, barrierInfo, MinMaxOption::Min, /*bfsDirUp=*/false);
}

/*
  452                650
   |                |
   v                v
  +------------------+
  | Execution Group  |
  +------------------+
9184               9277

Find Last DMA for a group tries to find a DMA which is the last DMA that can be used as a marker to tell the Execution
Group should start execution e.g. in above case 9184 is the first barrier in execution group. The function looks for a
DMA which is on tile 0 list 0 and updates 9184

If there is no DMA which updates 9184 using BFS check if the user's barrier (barrier<-task<-barrier) has a DMA user and
use it as last DMA
*/
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findLastDmaBeforeExecGroup(BarrierInfo& barrierInfo,
                                                                              ExecutionGroup& executionGroup) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> waitBarriers;
    SmallVector<size_t> possibleUpdatingDMAs;

    for (const auto& taskIdx : executionGroup) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
        auto wBarriers = taskOp.getWaitBarriers();
        for (auto waitBarrier : wBarriers) {
            auto waitBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(waitBarrier.getDefiningOp());
            waitBarriers.push_back(waitBarrierOp);
        }
    }

    // Collect possible updating DMAs from all wait barriers
    for (const auto& waitBarrier : waitBarriers) {
        auto barrierIdx = barrierInfo.getIndex(waitBarrier);
        auto updatingDMAs = to_small_vector(barrierInfo.getBarrierProducers(barrierIdx));

        // Filter producers to retain only valid DMAs
        auto filteredRange = updatingDMAs | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                                 return isValidDMA(barrierInfo, dmaIdx);
                             });

        possibleUpdatingDMAs.insert(possibleUpdatingDMAs.end(), filteredRange.begin(), filteredRange.end());
    }

    // If valid DMAs were found, return the one with the latest position
    if (!possibleUpdatingDMAs.empty()) {
        return findMinMaxPosition(possibleUpdatingDMAs, barrierInfo, MinMaxOption::Max);
    }

    // Initialize variable to track the latest DMA found across all barriers
    VPURT::TaskOp maxDmaOp;
    bool foundAnyDMA = false;

    // Search through all barriers using BFS to find the DMA with the latest position
    for (const auto& waitBarrier : waitBarriers) {
        auto barrierIdx = barrierInfo.getIndex(waitBarrier);
        auto dmaOp = findDMAsThroughBarriersBFS(barrierIdx, barrierInfo, MinMaxOption::Max, /*bfsDirUp=*/true);

        // Check if a DMA was found, and track the latest position DMA across all barriers
        if (dmaOp && (!foundAnyDMA || compareVPURTOpPosition(maxDmaOp, dmaOp, barrierInfo))) {
            maxDmaOp = dmaOp;
            foundAnyDMA = true;
        }
    }

    return maxDmaOp;
}

SmallVector<size_t> LegalizeScheduleForWlmFetchDmasPass::getDmasUpdatingBarriers(llvm::DenseSet<size_t>& barriers,
                                                                                 BarrierInfo& barrierInfo) {
    llvm::DenseSet<size_t> allTaskUpdatingBarriers;
    for (auto barrIdx : barriers) {
        auto allProducers = barrierInfo.getBarrierProducers(barrIdx);
        for (auto p : allProducers) {
            allTaskUpdatingBarriers.insert(p);
        }
    }
    auto updatingTasks = to_small_vector(allTaskUpdatingBarriers);
    auto filteredRange = updatingTasks | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                             return isValidDMA(barrierInfo, dmaIdx);
                         });
    return to_small_vector(filteredRange);
}

// Function returns barriers used by parentGroup excluding the barriers used by travelingGroup
llvm::DenseSet<size_t> getExclusiveBarriersUsedByGroup(ExecutionGroup& parentGroup, ExecutionGroup& travelingGroup,
                                                       BarrierInfo& barrierInfo) {
    llvm::DenseSet<size_t> parentBarriersUsed;
    llvm::DenseSet<size_t> travelingBarriersUsed;
    llvm::DenseSet<size_t> exclusiveBarriers;

    auto getBarriersUsedByGroup = [&barrierInfo](ExecutionGroup& execGroup, llvm::DenseSet<size_t>& barriersUsed) {
        for (auto i : execGroup) {
            auto wBarr = barrierInfo.getWaitBarriers(i);
            auto uBarr = barrierInfo.getUpdateBarriers(i);

            std::for_each(wBarr.begin(), wBarr.end(), [&barriersUsed](size_t barrier) {
                barriersUsed.insert(barrier);
            });

            std::for_each(uBarr.begin(), uBarr.end(), [&barriersUsed](size_t barrier) {
                barriersUsed.insert(barrier);
            });
        }
    };

    getBarriersUsedByGroup(parentGroup, parentBarriersUsed);
    getBarriersUsedByGroup(travelingGroup, travelingBarriersUsed);

    for (auto barrier : parentBarriersUsed) {
        if (travelingBarriersUsed.find(barrier) == travelingBarriersUsed.end()) {
            exclusiveBarriers.insert(barrier);
        }
    }
    return exclusiveBarriers;
}

/*
Form legalization perspective of traveling group we care about the last tasks from grand-parent group and parent group.
Each group can have at max 32 Invariant/Kernel Invocation

    +----------------------------+   +--------------------+   +---------------------+
    | GrandParentGroup    |lastOp|   | ParentGroup |lastOp|   |      TravelingGroup |
    +----------------------------+   +--------------------+   +---------------------+

Case 1: Last task of grand parent and last task of parent have common wait and update barriers
B:9184 and B:9277 represent the barriers associated with last task of parent group and also last task of grand parent
group

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184               B:9277   B:9184             B:9277

                        | |
                         v

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              BNew1      BNew2                B:9277
                    |            ^
                    |            |
                    |            |
                    v            |
                +-----+       +-----+
                | D1  |.......| D2  |
                +-----+       +-----+


Case 2: Last task of grand parent and last task of parent group share barriers
B:9277 and B:9353 represent the barriers associated with last task of parent group
B:9184 and B:9277 represent the barriers associated with last task of grand parent group

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              B:9277     B:9277              B:9353

                            | |
                             v

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              B:New      B:9277             B:9353
                    |            ^
                    |            |
                    |            |
                    v            |
                +-----+        +-----+
                | D1  |........| D2  |
                +-----+        +-----+


The barriers are common we cannot use them as wait and update and need to insert a
new barriers to delay traveling group

Case 3: Last task of grand parent and last task of parent have exclusive barriers
B:9729 and B:9353 represent the barriers associated with last task of parent group
B:9184 and B:9277 is the barriers associated with last task of grand parent group

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184               B:9277     B:9279              B:9353

                        | |
                         v

  +--------------------+         +------------------+
  |LT GrandParent Group|         | LT Parent Group  |
  +--------------------+         +------------------+
B:9184              B:9277     B:9279               B:9353
                    |            ^
                    |            |
                    |            |
                    v            |
                +-----+          +-----+
                | D1  |..........| D2  |
                +-----+          +-----+

*/
void LegalizeScheduleForWlmFetchDmasPass::insertDMAForFetchTasks(
        DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups, VPU::ExecutorKind executorKind,
        mlir::Operation* bufferInsertionPoint, mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
        SmallVector<VPURT::TaskOp>& dummyDmas, size_t tilesCount, SmallVector<std::pair<size_t, size_t>>& blockRange) {
    auto inSameTaskBlock = [&blockRange](size_t task1, size_t task2) {
        return any_of(blockRange.begin(), blockRange.end(), [&](const std::pair<size_t, size_t>& range) {
            return (task1 >= range.first && task1 <= range.second) && (task2 >= range.first && task2 <= range.second);
        });
    };

    VPURT::TaskQueueType queueType;
    queueType.type = executorKind;
    queueType.id = 0;

    auto executionGroupListForTile = listOfExecutionGroups[queueType];
    if (executionGroupListForTile.size() < 3) {
        return;
    }

    auto parentGroup = executionGroupListForTile[0];
    ExecutionGroup grandParentGroup;

    size_t groupIdx = 1;
    auto travelingGroup = executionGroupListForTile[groupIdx];

    // Reuse the same Decl Buffer for all Dummy DMAs
    mlir::Value inBuffer = nullptr;
    mlir::Value outBuffer = nullptr;

    while (groupIdx < executionGroupListForTile.size()) {
        auto hasGrandParent = !grandParentGroup.empty();
        auto firstGrandParentDma = hasGrandParent ? findFirstDmaAfterExecGroup(barrierInfo, grandParentGroup) : nullptr;
        auto insertionDma = hasGrandParent ? firstGrandParentDma : findLastDmaBeforeExecGroup(barrierInfo, parentGroup);
        auto insertionIndex = barrierInfo.getIndex(insertionDma);

        bool legalizationRequired = true;
        if (hasGrandParent) {
            // Check if the DMA is updating any barriers used by parent group
            // Function returns exclusive barriers used by parent group
            auto barriersByParentGroup = getExclusiveBarriersUsedByGroup(parentGroup, travelingGroup, barrierInfo);
            auto dmasUpdatingBarriers = getDmasUpdatingBarriers(barriersByParentGroup, barrierInfo);
            if (!dmasUpdatingBarriers.empty()) {
                auto lastDmaToUpdateBarrierInParentGroup =
                        barrierInfo.getIndex(findMinMaxPosition(dmasUpdatingBarriers, barrierInfo, MinMaxOption::Max));
                if (lastDmaToUpdateBarrierInParentGroup > insertionIndex) {
                    legalizationRequired = false;
                }
            }
        }

        if (legalizationRequired && hasGrandParent) {
            size_t parentTaskIdx = parentGroup.size() - 1;
            auto lastParentTaskOpIdx = parentGroup[parentTaskIdx];
            auto lastGrandParentTaskIdx = grandParentGroup[grandParentGroup.size() - 1];

            auto lastGrandParentTaskOpIdx = getSiblingTaskOpOnTile(lastGrandParentTaskIdx, barrierInfo);
            auto lastGrandParentTaskOp = barrierInfo.getTaskOpAtIndex(lastGrandParentTaskOpIdx);
            auto lastGrandParentTaskUpdateBarriers = to_small_vector(lastGrandParentTaskOp.getUpdateBarriers());
            auto lastGrandParentTaskWaitBarriers = to_small_vector(lastGrandParentTaskOp.getWaitBarriers());

            auto lastParentTaskOp = barrierInfo.getTaskOpAtIndex(lastParentTaskOpIdx);
            auto lastParentTaskWaitBarriers = to_small_vector(lastParentTaskOp.getWaitBarriers());
            auto lastParentTaskUpdateBarriers = to_small_vector(lastParentTaskOp.getUpdateBarriers());

            if (!inSameTaskBlock(lastParentTaskOpIdx, lastGrandParentTaskIdx)) {
                grandParentGroup = parentGroup;
                parentGroup = travelingGroup;

                ++groupIdx;
                if (groupIdx < executionGroupListForTile.size()) {
                    travelingGroup = executionGroupListForTile[groupIdx];
                }
                continue;
            }

            inBuffer = inBuffer != nullptr ? inBuffer : createDummyBuffer(builder, bufferInsertionPoint);
            outBuffer = outBuffer != nullptr ? outBuffer : createDummyBuffer(builder, bufferInsertionPoint);

            auto commonWaitBarrierOpt =
                    findCommonBarrier(lastGrandParentTaskWaitBarriers, lastParentTaskWaitBarriers, barrierInfo);
            auto commonUpdateBarrierOpt =
                    findCommonBarrier(lastGrandParentTaskUpdateBarriers, lastParentTaskUpdateBarriers, barrierInfo);

            if (commonWaitBarrierOpt && commonUpdateBarrierOpt) {
                auto insertionBarrier =
                        mlir::cast<VPURT::DeclareVirtualBarrierOp>(commonWaitBarrierOpt.value().getDefiningOp());
                auto newBarrierOneOp = createNewBarrier(builder, barrierInfo, insertionBarrier, nullptr, nullptr);
                auto newBarrierTwoOp = createNewBarrier(builder, barrierInfo, insertionBarrier, nullptr, nullptr);

                for (auto barr : lastParentTaskWaitBarriers) {
                    auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barr.getDefiningOp());
                    for (size_t tile = 0; tile < tilesCount; ++tile) {
                        // Along with updating dependencies for lastParentTaskOpIdx, we must also update the
                        // dependencies for sibling on other tiles
                        /*
                            Example: If the original consumers for BarrierX, for legalization also needs to consume
                            BarrierY then all the siblings on other tiles must also consume BarrierY This helps to
                            eliminate the need to perform legalization per tile
                                        |---->DPU0_0                            |---->DPU0_0
                                        |---->DPU0_1                            |---->DPU0_1
                            BarrierX                        =>        BarrierY
                                        |---->DPU0_2                            |---->DPU0_2
                                        |---->DPU0_3                            |---->DPU0_3

                            This is done inorder to also legalize the siblings as FetchTasks are for tasks per tile
                        */
                        auto siblingIdxOnTile = getSiblingTaskOpOnTile(lastParentTaskOpIdx, barrierInfo, tile);
                        if (siblingIdxOnTile != SIZE_MAX) {
                            barrierInfo.removeConsumer(barrOp, siblingIdxOnTile);
                            barrierInfo.addConsumer(newBarrierTwoOp, siblingIdxOnTile);
                        }
                    }
                }

                builder.setInsertionPointAfter(lastGrandParentTaskOp);
                auto dummyDmaOne = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);

                SmallVector<mlir::Value> produceIn = {newBarrierTwoOp};
                SmallVector<mlir::Value> consumes = {newBarrierOneOp};
                updateBarriersForDma(consumes, produceIn, dummyDmaOne, barrierInfo);

                for (auto barr : lastGrandParentTaskUpdateBarriers) {
                    auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barr.getDefiningOp());
                    for (size_t tile = 0; tile < tilesCount; ++tile) {
                        auto siblingIdxOnTile = getSiblingTaskOpOnTile(lastGrandParentTaskIdx, barrierInfo, tile);
                        if (siblingIdxOnTile != SIZE_MAX) {
                            barrierInfo.removeProducer(barrOp, siblingIdxOnTile);
                            barrierInfo.addProducer(newBarrierOneOp, siblingIdxOnTile);
                        }
                    }
                }

                builder.setInsertionPointAfter(dummyDmaOne);
                auto dummyDmaTwo = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(consumes, produceIn, dummyDmaTwo, barrierInfo);

                /*
                    Since DMAX position in FIFO is before DMA1 and since the barriers are same for GP and TG we would
                    end up enqueuing them at same barrier this leads to inference hang as TG wouldn't be fetched

                    To overcome this we must also update all DPU/SW task that waits on the same barrier as GP (except
                    the tasks in GP)

                    Bar0[--DMAX--]73
                                 73[----GP----]X
                                               X-DMA1-Y
                                               X-DMA2-Y
                                                      Y[----PG----]74
                                                     *Y[----TG----]74

                */
                auto allConsumersOfWaitBarrier = to_small_vector(barrierInfo.getBarrierConsumers(insertionBarrier));
                auto validUser = [&](size_t taskIdx) -> bool {
                    auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
                    // Only need to change the barrier deps for the tasks after last task of grand parent
                    if (taskOp->isBeforeInBlock(lastGrandParentTaskOp) ||
                        taskIdx == barrierInfo.getIndex(lastGrandParentTaskOp)) {
                        return false;
                    }

                    // Don't modify DMA and tasks which doesn't not have same type as tasks in GP as they will be
                    // legalized with insertFetchTask for DPU/SW
                    if (taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN ||
                        taskOp.getExecutorKind() != executorKind) {
                        return false;
                    }
                    return true;
                };

                auto filteredRange = allConsumersOfWaitBarrier | vpux::filtered(std::move(validUser));
                auto filteredVector = to_small_vector(filteredRange);
                for (auto consumer : filteredVector) {
                    barrierInfo.removeConsumer(insertionBarrier, consumer);
                    barrierInfo.addConsumer(newBarrierTwoOp, consumer);
                }

            } else if (auto commonBarrOpt = findCommonBarrier(lastGrandParentTaskUpdateBarriers,
                                                              lastParentTaskWaitBarriers, barrierInfo)) {
                auto commonBarrierOp =
                        mlir::cast<VPURT::DeclareVirtualBarrierOp>(commonBarrOpt.value().getDefiningOp());
                auto commonBarrierIndex = barrierInfo.getIndex(commonBarrierOp);

                builder.setInsertionPointAfter(lastGrandParentTaskOp);
                auto dummyDmaOne = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                auto newBarrierOneOp = createNewBarrier(builder, barrierInfo, commonBarrierOp, nullptr, nullptr);

                for (size_t tile = 0; tile < tilesCount; ++tile) {
                    auto siblingIdxOnTile = getSiblingTaskOpOnTile(lastGrandParentTaskIdx, barrierInfo, tile);
                    if (siblingIdxOnTile != SIZE_MAX) {
                        barrierInfo.removeProducer(commonBarrierIndex, siblingIdxOnTile);
                        barrierInfo.addProducer(newBarrierOneOp, siblingIdxOnTile);
                    }
                }
                SmallVector<mlir::Value> produceIn = {commonBarrierOp};
                SmallVector<mlir::Value> consumes = {newBarrierOneOp};
                updateBarriersForDma(consumes, produceIn, dummyDmaOne, barrierInfo);

                builder.setInsertionPointAfter(dummyDmaOne);
                auto dummyDmaTwo = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(consumes, produceIn, dummyDmaTwo, barrierInfo);

            } else {
                /*
                    Special Case:
                    Example:
                    We have a TaskOp that waits on Barrier `C`, and `C` is also a barrier that the last task
                    of the parent group waits on. In this case, dummy DMAs cannot produce in `C`.

                    Solution:
                    Depending on the availability of barriers, we can resolve this in one of two ways:

                    Case 1: If all consumers of all wait barrier for last task of parent are before grand parent
                            we need to create a barrier

                    Case 2: If we have atleast 1 barrier from wait barrier of last parent task which has all users after
                    grand parent Then this barrier can be used for legalization but other can't e.g. we can use barrier
                   D as all users are after last grand parent task

                    Examples:

                    Case 1:
                    C[--TaskOp1--]X
                    D[--TaskOp2--]Y
                    A[--Last of GP--]B
                                            B[--DummyDMA1--]NewBarrier
                                            B[--DummyDMA2--]NewBarrier
                    NewBarrier,C,D[--Last of PG--]E

                    Case 2:
                    C[--TaskOp1--]X
                    A[--Last of GP--]B
                                            B[--DummyDMA1--]D
                                            B[--DummyDMA2--]D
                    D[--TaskOp2--]Y
                    C,D[--Last of PG--]E

                */
                auto lastParentWaitBarriersIdx = to_small_vector(barrierInfo.getWaitBarriers(lastParentTaskOpIdx));
                auto lastGrandParentUpdateBarriersIdx =
                        to_small_vector(barrierInfo.getUpdateBarriers(lastGrandParentTaskIdx));

                // Collect all barriers which can be used for legalizing
                SmallVector<size_t> barrierIndexesToUpdateByDummyDma;
                for (auto barrierIdx : lastParentWaitBarriersIdx) {
                    bool isUsedBeforeGrandParent = false;
                    for (auto barrierConsumer : barrierInfo.getBarrierConsumers(barrierIdx)) {
                        if (barrierConsumer < lastGrandParentTaskIdx) {
                            isUsedBeforeGrandParent = true;
                            break;
                        }
                    }
                    if (!isUsedBeforeGrandParent) {
                        barrierIndexesToUpdateByDummyDma.push_back(barrierIdx);
                    }
                }

                // If not barriers were available for use, create a new barrier
                if (barrierIndexesToUpdateByDummyDma.empty()) {
                    // Need to create new barrier for dummyDma -> lastParentTask dependency
                    auto insertionPointBarrierOp = barrierInfo.getBarrierOpAtIndex(lastParentWaitBarriersIdx[0]);
                    auto newBarrierOp =
                            createNewBarrier(builder, barrierInfo, insertionPointBarrierOp, nullptr, nullptr);
                    barrierInfo.addConsumer(newBarrierOp, lastParentTaskOpIdx);
                    barrierIndexesToUpdateByDummyDma.push_back(barrierInfo.getIndex(newBarrierOp));
                }

                builder.setInsertionPointAfter(lastGrandParentTaskOp);
                auto dummyDmaOne = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(/*consumes*/ lastGrandParentUpdateBarriersIdx,
                                     /*producesIn*/ barrierIndexesToUpdateByDummyDma, dummyDmaOne, barrierInfo);

                builder.setInsertionPointAfter(dummyDmaOne);
                auto dummyDmaTwo = createDummyDma(builder, inBuffer, outBuffer, barrierInfo, dummyDmas);
                updateBarriersForDma(/*consumes*/ lastGrandParentUpdateBarriersIdx,
                                     /*producesIn*/ barrierIndexesToUpdateByDummyDma, dummyDmaTwo, barrierInfo);

                /*
                    DMA Ordering:
                    `firstDMA`    <- DMA which waits for the update barrier of the grandparent group
                    `lastDMA`     <- DMA which updates the wait barrier of the traveling group (TG),
                                     either directly or through a FIFO dependency.

                    During workload-management pass, if `lastDMA` is positioned before or equal to `firstDMA`,
                    then FetchTask cannot be inserted.

                    Post Legalization:
                    - DMA1 waits for the barrier (e.g., barrier 30) which is the update barrier for the grandparent
                        group (GP).
                    - DMA2 updates the wait barrier (e.g., barrier 15) for the traveling group (TG).

                    To address this, we ensure DMA1 and DMA2 also update barrier 15, making DMA2 the last DMA
                    and DMA1 the first DMA.

                    Final Layout:
                            [---DMAX---]15
                    20[---GP---]30
                            30[--DMA1--]35, 15
                            30[--DMA2--]35, 15
                    35[---PG---]40
                    15[---TG---]

                */
                auto lastDma = findLastDmaBeforeExecGroup(barrierInfo, travelingGroup);
                if (lastDma->isBeforeInBlock(dummyDmaOne)) {
                    auto travelingGroupWaitBarriers = barrierInfo.getWaitBarriers(travelingGroup[0]);
                    for (auto waitBarrier : travelingGroupWaitBarriers) {
                        barrierInfo.addProducer(waitBarrier, barrierInfo.getIndex(dummyDmaOne));
                        barrierInfo.addProducer(waitBarrier, barrierInfo.getIndex(dummyDmaTwo));
                    }
                }
            }
        }

        grandParentGroup = parentGroup;
        parentGroup = travelingGroup;

        ++groupIdx;
        if (groupIdx < executionGroupListForTile.size()) {
            travelingGroup = executionGroupListForTile[groupIdx];
        }
    }
}

void LegalizeScheduleForWlmFetchDmasPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto module = netFunc->getParentOfType<mlir::ModuleOp>();

    auto barriersOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    auto numVirtualBarriers = static_cast<int64_t>(std::distance(barriersOps.begin(), barriersOps.end()));
    if (numVirtualBarriers > _virtualBarrierThreshold) {
        _log.info("Skip schedule legalization due to high number of barriers: {0}", numVirtualBarriers);
        vpux::VPUIP::setWlmStatus(module, vpux::VPUIP::WlmStatus::FAILED);
        return;
    }

    // Ease LIT tests by reducing the group sizes
    auto archKind = VPU::getArch(netFunc);
    _maxVarCount = maxVarCountPerGroup.hasValue()
                           ? checked_cast<size_t>(maxVarCountPerGroup.getValue())
                           : VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::DPUVariant, archKind) / 2;

    _maxInvarCount =
            maxInvarCountPerGroup.hasValue()
                    ? checked_cast<size_t>(maxInvarCountPerGroup.getValue())
                    : VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::DPUInvariant, archKind) / 2;

    _maxKernelInvoCount =
            maxKernelInvoCountPerGroup.hasValue()
                    ? checked_cast<size_t>(maxKernelInvoCountPerGroup.getValue())
                    : VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::ActKernelInvocation, archKind) / 2;

    _maxKernelRangeCount =
            maxKernelRangeCountPerGroup.hasValue()
                    ? checked_cast<size_t>(maxKernelRangeCountPerGroup.getValue())
                    : VPURegMapped::getDefaultTaskListCount(VPURegMapped::TaskType::ActKernelRange, archKind) / 2;

    mlir::OpBuilder builder(netFunc);
    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = static_cast<size_t>(IE::getTileExecutor(parentModule).getCount());

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    SmallVector<std::pair<size_t, size_t>> blockRange;
    for (size_t blockIdx = 0; blockIdx < barrierInfo.getControlGraphBlockCount(); ++blockIdx) {
        auto [blockStartInd, blockEndInd] = barrierInfo.getControlGraphBlockTaskRange(
                blockIdx, /* blockStartSyncPoint */ true, /* blockEndSyncPoint */ true);
        blockRange.push_back({blockStartInd, blockEndInd});
    }

    _numAllTaskOps = barrierInfo.getNumOfTasks();
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, true);
    barrierInfo.buildTaskQueueTypeMap();

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    // Will have a map for each cluster along with task index of the task
    auto taskQueues = VPURT::getTaskOpQueues(netFunc, barrierInfo);

    VPURT::TaskQueueType queueType;
    queueType.type = VPU::ExecutorKind::DMA_NN;
    queueType.id = getDMAQueueIdEncoding(/*port*/ 0, VPUIP::DmaChannelType::DDR);

    auto dQueue = taskQueues[queueType];
    _firstDMATaskOp = barrierInfo.getTaskOpAtIndex(dQueue[0]);
    _lastDMATaskOp = barrierInfo.getTaskOpAtIndex(dQueue[dQueue.size() - 1]);

    createDPUTaskExecutionGroups(barrierInfo, taskQueues, tilesCount);
    createSWTaskExecutionGroups(barrierInfo, taskQueues, tilesCount);

    SmallVector<VPURT::TaskOp> dummyDmas;
    insertDMAForFetchTasks(_listOfDPUExecutionGroups, VPU::ExecutorKind::DPU, bufferInsertionPoint, builder,
                           barrierInfo, dummyDmas, tilesCount, blockRange);

    insertDMAForFetchTasks(_listOfSWExecutionGroups, VPU::ExecutorKind::SHAVE_ACT, bufferInsertionPoint, builder,
                           barrierInfo, dummyDmas, tilesCount, blockRange);

    // Apply the changes now as we need to make sure the new DMAs don't break the schedule
    barrierInfo.updateIR();

    // Correct the position of new dmas after realizing the changes from barrierInfo
    for (auto dummyDma : dummyDmas) {
        auto waitBarriers = to_small_vector(dummyDma.getWaitBarriers());
        if (waitBarriers.empty()) {
            continue;
        }

        auto lastTaskToUpdate = findLastTaskToUpdate(waitBarriers, barrierInfo);
        if (dummyDma->isBeforeInBlock(lastTaskToUpdate)) {
            dummyDma->moveAfter(lastTaskToUpdate);
        }
    }

    // Reorder barriers in production order, this will also verify the schedule
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo);
    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(netFunc);
}

}  // namespace

//
// createLegalizeScheduleForWlmFetchDmasPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createLegalizeScheduleForWlmFetchDmasPass(
        const int virtualBarrierThreshold, Logger log) {
    return std::make_unique<LegalizeScheduleForWlmFetchDmasPass>(virtualBarrierThreshold, log);
}
