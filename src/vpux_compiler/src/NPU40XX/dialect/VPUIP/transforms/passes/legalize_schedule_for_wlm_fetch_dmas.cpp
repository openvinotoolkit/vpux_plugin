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
#include "vpux/compiler/utils/dma.hpp"

#include <npu_40xx_nnrt.hpp>
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

    VPURT::TaskOp getNextDma(BarrierInfo& barrierInfo, VPURT::TaskOp fetchDMA);

    bool isValidDMA(BarrierInfo& barrierInfo, size_t dmaIdx);
    VPURT::TaskOp findLastDma(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findFirstDma(BarrierInfo& barrierInfo, ExecutionGroup& executionGroup);
    VPURT::TaskOp findDMAsThroughBarriersBFS(size_t startBarrier, BarrierInfo& barrierInfo, MinMaxOption option,
                                             bool bfsDirUp);
    void createSWTaskExecutionGroups(BarrierInfo& barrierInfo,
                                     std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& swQueue, size_t tilesCount);
    void createDPUTaskExecutionGroups(BarrierInfo& barrierInfo,
                                      std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& dpuQueue,
                                      size_t tilesCount);

    VPURT::TaskOp createDummyDma(mlir::OpBuilder& builder, mlir::Value inputBuf, mlir::Value outputBuf,
                                 mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                                 BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas);

    void insertDMAForFetchTasks(DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups,
                                VPU::ExecutorKind executorKind, mlir::Operation* bufferInsertionPoint,
                                mlir::Operation* barrierInsertionPoint, mlir::OpBuilder& builder,
                                BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas);

private:
    // Will be initialized in safeRunOnFunc(), this is done to suppress the UNINIT_CTOR warning
    size_t _numAllTaskOps = 0;
    VPURT::TaskOp _firstDMATaskOp;
    VPURT::TaskOp _lastDMATaskOp;
    DenseMap<VPURT::TaskQueueType, ExecutionGroupList> _listOfSWExecutionGroups;
    DenseMap<VPURT::TaskQueueType, ExecutionGroupList> _listOfDPUExecutionGroups;
};

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

mlir::Value createDummyBuffer(mlir::OpBuilder& builder, mlir::Operation* insertionPoint, long int c = 0) {
    auto ctx = builder.getContext();
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (insertionPoint != nullptr) {
        builder.setInsertionPoint(insertionPoint);
    }

    const auto nameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    const auto ddrSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, nameAttr);
    const auto layout = DimsOrder::NCHW.toAffineMap(ctx);

    auto zeroBufferMemref = mlir::MemRefType::get({0, 0, 0, c}, builder.getI32Type(), layout, ddrSymbolAttr);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), zeroBufferMemref, VPURT::BufferSection::DDR,
                                                  0);
}

size_t getLastTaskOp(ExecutionGroup& execGroup, BarrierInfo& barrierInfo) {
    auto lastTaskTravelingGroup = execGroup[execGroup.size() - 1];
    auto lastTaskTravelingGroupOp = barrierInfo.getTaskOpAtIndex(lastTaskTravelingGroup);

    auto getTileIndex = [&](mlir::Operation* op) -> size_t {
        auto taskOp = llvm::cast<VPURT::TaskOp>(op);
        if (auto nceOp = llvm::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp())) {
            const auto& dpuTasks = nceOp.getVariants().getOps<VPUIP::DPUTaskOp>();
            VPUIP::DPUTaskOp first = *(dpuTasks.begin());

            uint8_t tileIndex = 0;
            if (first.getClusterId().has_value()) {
                tileIndex = first.getClusterId().value();
            } else {
                auto bufferOp = mlir::cast<VPURT::DeclareBufferOp>(nceOp.getInput().getDefiningOp());
                if (bufferOp.getSection() == VPURT::BufferSection::CMX_NN) {
                    if (bufferOp.getSectionIndex().has_value() && !bufferOp.getSectionIndex().value().empty()) {
                        auto tiles = parseIntArrayAttr<uint8_t>(bufferOp.getSectionIndex().value());
                        tileIndex = *std::min_element(tiles.begin(), tiles.end());
                    }
                }
            }
            return tileIndex;
        }
        if (auto swOp = llvm::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp())) {
            return swOp.getTileIndex().value_or(0);
        }
        return 0;
    };

    mlir::Operation* prevOp = nullptr;
    mlir::Operation* currentOp = lastTaskTravelingGroupOp->getNextNode();

    while (currentOp != nullptr) {
        size_t tileIndex = getTileIndex(currentOp);
        // If tile index is 0, return the previous operation
        // If prev operation is null then we have case when the lastTaskTravelingGroup is only running on 1 tile
        if (tileIndex == 0) {
            if (prevOp != nullptr) {
                auto prev = mlir::cast<VPURT::TaskOp>(prevOp);
                return barrierInfo.getIndex(prev);
            } else {
                return lastTaskTravelingGroup;
            }
        }

        // Move to the next node
        prevOp = currentOp;
        currentOp = currentOp->getNextNode();
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
bool hasCommonBarrier(const SmallVector<mlir::Value>& barrierListOne, const SmallVector<mlir::Value>& barrierListTwo) {
    mlir::DenseSet<mlir::Value> elements(barrierListOne.begin(), barrierListOne.end());
    for (mlir::Value barr : barrierListTwo) {
        if (elements.find(barr) != elements.end()) {
            return true;
        }
    }
    return false;
}

// Return first task that waits on series of barriers
// As we check for the first DMA, sort the vector and use the first one
mlir::Operation* findFirstTaskToWaitOn(mlir::ValueRange barriers, BarrierInfo& barrierInfo) {
    auto barrierVector = to_small_vector(barriers);
    llvm::sort(barrierVector, [&](const auto& lhs, const auto& rhs) {
        return compareVPURTOpPosition(lhs, rhs, barrierInfo);
    });
    auto barrierOp = barrierVector[0];

    auto validUser = [&barrierOp](mlir::Operation* op) -> bool {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
        if (taskOp == nullptr) {
            return false;
        }
        auto waitBarrierList = to_small_vector(taskOp.getWaitBarriers());
        return hasCommonBarrier(waitBarrierList, {barrierOp});
    };

    auto bOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrierOp.getDefiningOp());
    SmallVector<VPURT::TaskOp> users;
    for (auto usr : bOp.getResult().getUsers()) {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(usr);
        if (taskOp != nullptr)
            users.push_back(taskOp);
    }

    auto filteredRange = users | vpux::filtered(std::move(validUser));
    auto filteredVector = to_small_vector(filteredRange);
    if (!filteredVector.empty()) {
        llvm::sort(filteredVector, [&](const auto& lhs, const auto& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo);
        });
        return filteredVector[0];
    }
    return nullptr;
}

// Return last task that updates series of barriers
// As we check for the last DMA, sort the vector and use the last one
mlir::Operation* findLastTaskToUpdate(mlir::ValueRange barriers, BarrierInfo& barrierInfo, bool useIROrder = false) {
    auto barrierVector = to_small_vector(barriers);
    llvm::sort(barrierVector, [&](const auto& lhs, const auto& rhs) {
        return compareVPURTOpPosition(lhs, rhs, barrierInfo);
    });
    auto barrierOp = barrierVector[barrierVector.size() - 1];
    auto validUser = [&barrierOp](VPURT::TaskOp op) -> bool {
        auto updateBarrierList = to_small_vector(op.getUpdateBarriers());
        return hasCommonBarrier(updateBarrierList, {barrierOp});
    };

    auto bOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrierOp.getDefiningOp());
    SmallVector<VPURT::TaskOp> users;
    for (auto usr : bOp.getResult().getUsers()) {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(usr);
        if (taskOp != nullptr)
            users.push_back(taskOp);
    }

    auto filteredRange = users | vpux::filtered(std::move(validUser));
    auto filteredVector = to_small_vector(filteredRange);
    if (!filteredVector.empty()) {
        llvm::sort(filteredVector, [&](const auto& lhs, const auto& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo, useIROrder);
        });
        return filteredVector[filteredVector.size() - 1];
    }

    return nullptr;
}

// Add wait barrier for given task as producer for dummy DMA
void addBarriersAsProducersForDummyDma(mlir::ValueRange barriers, BarrierInfo& barrierInfo, VPURT::TaskOp dummyDma) {
    size_t dummyDmaIdx = barrierInfo.getIndex(dummyDma);
    for (auto barrier : barriers) {
        auto barrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrier.getDefiningOp());
        barrierInfo.addProducer(barrierOp, dummyDmaIdx);
    }
}

// Add update barrier for given task as consumer for dummy DMA
void addBarriersAsConsumersForDummyDma(mlir::ValueRange barriers, BarrierInfo& barrierInfo, VPURT::TaskOp dummyDma) {
    size_t dummyDmaIdx = barrierInfo.getIndex(dummyDma);
    for (auto barrier : barriers) {
        auto barrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrier.getDefiningOp());
        barrierInfo.addConsumer(barrierOp, dummyDmaIdx);
    }
}

// Create new barrier and add producer and consumer
VPURT::DeclareVirtualBarrierOp createNewBarrier(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                mlir::Operation* insertionPoint, VPURT::TaskOp producer,
                                                VPURT::TaskOp consumer) {
    builder.setInsertionPoint(insertionPoint);
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
                                                                  mlir::Value outputBuf, mlir::ValueRange waitBarriers,
                                                                  mlir::ValueRange updateBarriers,
                                                                  BarrierInfo& barrierInfo,
                                                                  SmallVector<VPURT::TaskOp>& dummyDmas) {
    auto ctx = builder.getContext();
    auto dummyDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_dummy_dma"));

    auto newDMAOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, waitBarriers, updateBarriers, dummyDmaLoc, inputBuf,
                                                          outputBuf, 0, false, false, nullptr, nullptr);
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

// Given a DMA find the next DMA on same FIFO
// TODO: Simplify this after E-119383
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::getNextDma(BarrierInfo& barrierInfo, VPURT::TaskOp fetchDMA) {
    auto validDMA = [](VPURT::TaskOp op) {
        return op.getExecutorKind() == VPU::ExecutorKind::DMA_NN && isDMAOnSupportedPortAndChannel(op);
    };

    auto totalTasks = barrierInfo.getNumOfTasks();
    auto dmaPosition = barrierInfo.getIndex(fetchDMA);
    auto nextDma = barrierInfo.getTaskOpAtIndex(dmaPosition + 1);
    while (!validDMA(nextDma)) {
        dmaPosition += 1;
        if (totalTasks <= dmaPosition) {
            _log.warning("Choosing last DMA in IR as next DMA for '{0}'", fetchDMA);
            return fetchDMA;
        }
        nextDma = barrierInfo.getTaskOpAtIndex(dmaPosition);
    }
    return nextDma;
}

// Create DPU execution groups of tasks for each cluster
void LegalizeScheduleForWlmFetchDmasPass::createDPUTaskExecutionGroups(
        BarrierInfo& barrierInfo, std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& dpuQueue, size_t tilesCount) {
    VPURT::TaskQueueType queueType;
    queueType.type = VPU::ExecutorKind::DPU;
    // Because CMX Metadata Space is double buffered we divide by 2 here
    auto invariantCount = npu40xx::nn_public::VPU_INVARIANT_COUNT / 2;
    auto variantCount = npu40xx::nn_public::VPU_VARIANT_COUNT / 2;

    for (size_t tile = 0; tile < tilesCount; ++tile) {
        queueType.id = tile;
        auto tileDpuQueue = dpuQueue[queueType];

        ExecutionGroup execGroup;
        uint32_t execGroupVariantCount = 0;
        uint32_t execGroupInVariantCount = 0;

        auto& tempVector = _listOfDPUExecutionGroups[queueType];
        for (auto taskIdx : tileDpuQueue) {
            auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
            auto dpuSize = 0;
            for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<VPUIP::NCEClusterTaskOp>())) {
                const auto& dpuTasks = to_small_vector(op.getVariants().getOps<VPUIP::DPUTaskOp>());
                dpuSize = dpuTasks.size();
                execGroupVariantCount += dpuTasks.size();
                execGroupInVariantCount++;
            }
            if (execGroupVariantCount < variantCount && execGroupInVariantCount < invariantCount) {
                execGroup.push_back(taskIdx);
            } else {
                // Push the whole group to bigger list
                tempVector.push_back(execGroup);
                execGroup.clear();

                // Prepare next group
                execGroup.push_back(taskIdx);
                execGroupVariantCount = dpuSize;
                execGroupInVariantCount = 1;
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
    // Because CMX Metadata Space is double buffered we divide by 2 here
    auto kernelInvoCount = npu40xx::nn_public::VPU_KERNEL_INVO_COUNT / 2;
    auto kernelRangeCount = npu40xx::nn_public::VPU_KERNEL_RANGE_COUNT / 2;

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

            execGroupInvoCount += count;
            execGroupRangeCount += count;

            if (execGroupInvoCount < kernelInvoCount && execGroupRangeCount < kernelRangeCount) {
                execGroup.push_back(taskIdx);
            } else {
                // Push the whole group to bigger list
                tempVector.push_back(execGroup);
                execGroup.clear();

                // Prepare next group
                execGroup.push_back(taskIdx);
                execGroupInvoCount = 1;
                execGroupRangeCount = 1;
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
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findFirstDma(BarrierInfo& barrierInfo,
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
VPURT::TaskOp LegalizeScheduleForWlmFetchDmasPass::findLastDma(BarrierInfo& barrierInfo,
                                                               ExecutionGroup& executionGroup) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> waitBarriers;
    for (const auto& taskIdx : executionGroup) {
        auto taskOp = barrierInfo.getTaskOpAtIndex(taskIdx);
        auto wBarriers = taskOp.getWaitBarriers();
        for (auto waitBarrier : wBarriers) {
            auto waitBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(waitBarrier.getDefiningOp());
            waitBarriers.push_back(waitBarrierOp);
        }
    }

    llvm::sort(waitBarriers, [&](const auto& lhs, const auto& rhs) {
        return compareVPURTOpPosition(lhs, rhs, barrierInfo);
    });

    auto firstBarrier = barrierInfo.getIndex(waitBarriers[0]);
    SmallVector<size_t> possibleUpdatingDMAs;

    auto updatingDMAs = to_small_vector(barrierInfo.getBarrierProducers(firstBarrier));
    auto filteredRange = updatingDMAs | vpux::filtered([this, &barrierInfo](size_t dmaIdx) {
                             return isValidDMA(barrierInfo, dmaIdx);
                         });
    auto filteredVector = to_small_vector(filteredRange);
    possibleUpdatingDMAs.insert(possibleUpdatingDMAs.end(), filteredVector.begin(), filteredVector.end());

    if (!possibleUpdatingDMAs.empty()) {
        return findMinMaxPosition(possibleUpdatingDMAs, barrierInfo, MinMaxOption::Max);
    }

    return findDMAsThroughBarriersBFS(firstBarrier, barrierInfo, MinMaxOption::Max, /*bfsDirUp=*/true);
}

/*
TravelingGroup - We iterate/travel over all groups and check if there is a place to insert FetchTask for them, in this
context a TravelingGroup is the current group we're on during iteration
Parent Group - In the same context as above, the group before Traveling Group is Parent Group
Grand Parent Group - In the same context as above, the group before Parent Group is Grand Parent Group

The right time to fetch the descriptors of TravelingGroup in CMX Metadata, in a double buffered system is when
GrandParentGroup has finished execution (Ping Section is safe to be rewritten). Based on this the Insertion DMA is a DMA
that marks finish of GrandParent Group and Last DMA refers to a DMA which marks the start of traveling group

For detailed definition of First and Last DMA check findLastDma and findFirstDma functions
The Fetch task for all the cases would be inserted immediately after InsertionDMA

Case 0: Last DMA > Insertion DMA - no change required in the schedule

Case 1: Insertion DMA(650) == Last DMA (650)

                       DMA:650
DMA:452             /          \              DMA:661
   |                |           |                   |
   v                v           v                   v
  +------------------+         +------------------+
  | GrandParent Group|         | Traveling Group  |
  +------------------+         +------------------+
B:9184               B:9277    B:9279                  B:9353

                        | |
                         v

                       DMA:650
DMA:452             /          \              DMA:661
   |                |           |                   |
   v                v           v                   v
  +------------------+         +------------------+
  | GrandParent Group|         | Traveling Group  |
  +------------------+         +------------------+
B:9184             B:9277    B:9279                  B:9353
                    |           |
                    |           |
                    |  +-----+  |
                    +--| D1  |--+
                       +-----+

This case we already have a DMA 650 which waits for 9277 and updates 9279
This case we're just okay to insert 1 DMA after 650 which waits for 9277 and updates 9279 such that after the new
DMA(D1) is created, DMA:650 is the first DMA which marks finish of Grand Parent and last DMA which marks start of
Traveling Group is D1. Post insertion of dummy DMA the order of DMAs would be DMA:650->D1


Case 2: Insertion DMA (650) > Last DMA (500)

DMA:452          DMA:650     DMA:500             DMA:661
   |                |           |                   |
   v                v           v                   v
  +------------------+         +------------------+
  | GrandParent Group|         | Traveling Group  |
  +------------------+         +------------------+
B:9184               B:9277    B:9279                  B:9353

                        | |
                         v

DMA:452          DMA:650     DMA:500             DMA:661
   |                |           |                   |
   v                v           v                   v
  +------------------+         +------------------+
  | GrandParent Group|         | Traveling Group  |
  +------------------+         +------------------+
B:9184              B:9277    B:9279                  B:9353
                    |           |
                    |           |
                    |           |
            +----+-------------------------+
            |                              |
            v                              v
            +-----+                      +-----+
            | D1  |                      | D2  |
            +-----+                      +-----+


500 is the DMA which marks the start of Traveling Group, by this time we should have the Descriptors for Traveling
Group in the Metadata. However because 650 the DMA which marks the finish of GrandParentGroup is after 500 we can't
do it safely. For the solution we insert a DMA D1 which waits on 9277 and D2 which waits on 9277 such that D2 is
after D1. Second step we ask D1 and D2 to update 9279

Now after the new DMAs the first DMA which marks finish of Grand Parent is D1 and last DMA which marks start of
Traveling Group is D2.

Insertion Point here matters, D1 is inserted right after Grand Parent and D2 is inserted right before Traveling
Group


Case 3: Insertion DMA (560) > Last DMA (500) (Common Barriers)

DMA:452          DMA:650     DMA:500             DMA:661
   |                |           |                   |
   v                v           v                   v
  +------------------+         +------------------+
  | GrandParent Group|         | Traveling Group  |
  +------------------+         +------------------+
B:9184              B:9277    B:9277                  B:9353

                        | |
                         v

DMA:452          DMA:650     DMA:500             DMA:661
   |                |           |                   |
   v                v           v                   v
  +------------------+         +------------------+
  | GrandParent Group|         | Traveling Group  |
  +------------------+         +------------------+
B:9184              B:9277    Barrier Two         B:9353
                    |                  \
                    |                   \
                    |                    \
    +----+-----------                     \
    |                                      \
    v                                       \
    +-----+                        +-----+   \
    | D1  | ----->Barrier One--->  | D2  | --->Barrier Two
    +-----+                        +-----+


Same as case 3 however because the barriers are common we cannot use them as wait and update and need to insert a
new barriers to delay traveling group

*/
void LegalizeScheduleForWlmFetchDmasPass::insertDMAForFetchTasks(
        DenseMap<VPURT::TaskQueueType, ExecutionGroupList>& listOfExecutionGroups, VPU::ExecutorKind executorKind,
        mlir::Operation* bufferInsertionPoint, mlir::Operation* barrierInsertionPoint, mlir::OpBuilder& builder,
        BarrierInfo& barrierInfo, SmallVector<VPURT::TaskOp>& dummyDmas) {
    VPURT::TaskQueueType queueType;
    queueType.type = executorKind;
    queueType.id = 0;

    auto executionGroupListForTile = listOfExecutionGroups[queueType];
    if (executionGroupListForTile.size() < 3) {
        return;
    }

    auto parentGroup = executionGroupListForTile[0];
    auto parentFetch = _firstDMATaskOp;
    ExecutionGroup grandParentGroup;

    size_t groupIdx = 1;
    auto travelingGroup = executionGroupListForTile[groupIdx];

    // Reuse the same Decl Buffer for all Dummy DMAs
    mlir::Value inBuffer = nullptr;
    mlir::Value outBuffer = nullptr;

    while (groupIdx < executionGroupListForTile.size()) {
        size_t tIdx = 0;
        auto parentFetchDma = getNextDma(barrierInfo, parentFetch);
        auto hasGrandParent = !grandParentGroup.empty();

        auto firstGrandParentDma = hasGrandParent ? findFirstDma(barrierInfo, grandParentGroup) : nullptr;
        auto insertionDma = hasGrandParent ? std::max(firstGrandParentDma, parentFetchDma,
                                                      [&barrierInfo](const auto& lhs, const auto& rhs) {
                                                          return compareVPURTOpPosition(lhs, rhs, barrierInfo);
                                                      })
                                           : findLastDma(barrierInfo, parentGroup);

        auto insertionGroup = hasGrandParent ? grandParentGroup : parentGroup;
        auto lastDma = findLastDma(barrierInfo, travelingGroup);
        if (insertionDma == firstGrandParentDma) {
            tIdx = groupIdx - 2;
            insertionGroup = grandParentGroup;
        } else {
            tIdx = groupIdx - 1;
            insertionGroup = parentGroup;
        }

        auto lastDmaIdx = barrierInfo.getIndex(lastDma);
        auto insertionDmaIdx = barrierInfo.getIndex(insertionDma);

        if (lastDmaIdx == insertionDmaIdx) {
            inBuffer = inBuffer != nullptr ? inBuffer : createDummyBuffer(builder, bufferInsertionPoint);
            outBuffer = outBuffer != nullptr ? outBuffer : createDummyBuffer(builder, bufferInsertionPoint);

            builder.setInsertionPointAfter(lastDma);

            auto firstDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);
            addBarriersAsProducersForDummyDma(lastDma.getUpdateBarriers(), barrierInfo, firstDummyDma);
            addBarriersAsConsumersForDummyDma(lastDma.getWaitBarriers(), barrierInfo, firstDummyDma);
        } else if (lastDmaIdx < insertionDmaIdx) {
            inBuffer = inBuffer != nullptr ? inBuffer : createDummyBuffer(builder, bufferInsertionPoint);
            outBuffer = outBuffer != nullptr ? outBuffer : createDummyBuffer(builder, bufferInsertionPoint);

            auto lastTaskOpIdx = getLastTaskOp(executionGroupListForTile[tIdx], barrierInfo);

            auto lastTaskOp = barrierInfo.getTaskOpAtIndex(lastTaskOpIdx);
            auto lastTaskWaitBarriers = to_small_vector(lastTaskOp.getWaitBarriers());
            auto lastTaskUpdateBarriers = to_small_vector(lastTaskOp.getUpdateBarriers());

            auto firstTaskOpIdx = travelingGroup[0];
            auto firstTaskOp = barrierInfo.getTaskOpAtIndex(firstTaskOpIdx);
            auto firstTaskWaitBarriers = to_small_vector(firstTaskOp.getWaitBarriers());
            auto firstTaskUpdateBarriers = to_small_vector(firstTaskOp.getUpdateBarriers());

            auto lastTaskToUpdate = findLastTaskToUpdate(lastTaskOp.getUpdateBarriers(), barrierInfo);
            auto lastTaskInsertionPoint = lastTaskToUpdate != nullptr ? lastTaskToUpdate : lastTaskOp;

            auto firstTaskToWait = findFirstTaskToWaitOn(firstTaskOp.getWaitBarriers(), barrierInfo);
            auto firstTaskInsertionPoint = firstTaskToWait != nullptr ? firstTaskToWait : firstTaskOp;

            if (hasCommonBarrier(lastTaskUpdateBarriers, firstTaskWaitBarriers)) {
                builder.setInsertionPointAfter(lastTaskInsertionPoint);
                auto firstDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);

                addBarriersAsConsumersForDummyDma(lastTaskOp.getUpdateBarriers(), barrierInfo, firstDummyDma);

                auto newBarrierOneOp =
                        createNewBarrier(builder, barrierInfo, barrierInsertionPoint, firstDummyDma, nullptr);

                builder.setInsertionPoint(firstTaskInsertionPoint);
                auto secondDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);
                barrierInfo.addConsumer(newBarrierOneOp, barrierInfo.getIndex(secondDummyDma));

                auto newBarrierTwoOp =
                        createNewBarrier(builder, barrierInfo, barrierInsertionPoint, secondDummyDma, nullptr);

                for (auto waitBarrier : firstTaskWaitBarriers) {
                    auto waitBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(waitBarrier.getDefiningOp());

                    auto validUser = [&waitBarrier, &executorKind](mlir::Operation* op) -> bool {
                        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
                        if (taskOp == nullptr) {
                            return false;
                        }
                        SmallVector<mlir::Value> waitBarrierVector = {waitBarrier};
                        auto userWaitBarrierVector = to_small_vector(taskOp.getWaitBarriers());
                        const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);

                        return hasCommonBarrier(userWaitBarrierVector, waitBarrierVector) &&
                               executorKind == taskQueueType.type;
                    };

                    auto filteredRange = waitBarrierOp.getResult().getUsers() | vpux::filtered(std::move(validUser));
                    for (auto filteredUser : filteredRange) {
                        auto filteredUserOp = mlir::dyn_cast<VPURT::TaskOp>(filteredUser);
                        if (filteredUserOp != nullptr) {
                            auto filteredUserIdx = barrierInfo.getIndex(filteredUserOp);
                            barrierInfo.removeConsumer(waitBarrierOp, filteredUserIdx);
                            barrierInfo.addConsumer(newBarrierTwoOp, filteredUserIdx);
                        }
                    }
                }
            } else {
                // Case when we have consecutive tasks e.g. last task of parent and first task of current group
                // In this case because both the wait and update barriers are same it needs special handling
                if (hasCommonBarrier(lastTaskUpdateBarriers, firstTaskUpdateBarriers) &&
                    hasCommonBarrier(lastTaskWaitBarriers, firstTaskWaitBarriers)) {
                    for (auto barrier : lastTaskUpdateBarriers) {
                        auto barrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrier.getDefiningOp());
                        barrierInfo.removeProducer(barrierOp, lastTaskOpIdx);
                    }
                    auto newBarrierOneOp =
                            createNewBarrier(builder, barrierInfo, barrierInsertionPoint, lastTaskOp, nullptr);

                    for (auto barrier : firstTaskWaitBarriers) {
                        auto barrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(barrier.getDefiningOp());
                        barrierInfo.removeConsumer(barrierOp, firstTaskOpIdx);
                    }

                    auto newBarrierTwoOp =
                            createNewBarrier(builder, barrierInfo, barrierInsertionPoint, nullptr, firstTaskOp);

                    builder.setInsertionPoint(firstTaskInsertionPoint);
                    auto firstDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);
                    barrierInfo.addConsumer(newBarrierOneOp, barrierInfo.getIndex(firstDummyDma));

                    builder.setInsertionPoint(firstTaskInsertionPoint);
                    auto secondDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);
                    barrierInfo.addProducer(newBarrierTwoOp, barrierInfo.getIndex(secondDummyDma));
                    barrierInfo.addConsumer(newBarrierOneOp, barrierInfo.getIndex(secondDummyDma));
                    barrierInfo.addProducer(newBarrierTwoOp, barrierInfo.getIndex(firstDummyDma));

                } else {
                    builder.setInsertionPointAfter(lastTaskInsertionPoint);
                    auto firstDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);
                    addBarriersAsConsumersForDummyDma(lastTaskOp.getUpdateBarriers(), barrierInfo, firstDummyDma);

                    builder.setInsertionPoint(firstTaskInsertionPoint);
                    auto secondDummyDma = createDummyDma(builder, inBuffer, outBuffer, {}, {}, barrierInfo, dummyDmas);

                    addBarriersAsProducersForDummyDma(firstTaskOp.getWaitBarriers(), barrierInfo, secondDummyDma);
                    addBarriersAsConsumersForDummyDma(lastTaskOp.getUpdateBarriers(), barrierInfo, secondDummyDma);
                    addBarriersAsProducersForDummyDma(firstTaskOp.getWaitBarriers(), barrierInfo, firstDummyDma);
                }
            }
        }

        parentFetch = insertionDma;
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

    auto barriersOps = netFunc.getOps<VPURT::DeclareVirtualBarrierOp>();
    auto numVirtualBarriers = static_cast<int64_t>(std::distance(barriersOps.begin(), barriersOps.end()));
    if (numVirtualBarriers > _virtualBarrierThreshold) {
        _log.info("Skip schedule legalization due to high number of barriers: {0}", numVirtualBarriers);
        return;
    }

    mlir::OpBuilder builder(netFunc);
    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = static_cast<size_t>(IE::getTileExecutor(parentModule).getCount());

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    // TODO: Once E-135323 is resolved we can remove this check
    if (barrierInfo.getControlGraphBlockCount() > 1) {
        _log.warning("Couldn't legalize schedule for WLM as there are more than 1 control blocks in the IR");
        return;
    }
    _numAllTaskOps = barrierInfo.getNumOfTasks();
    VPURT::orderExecutionTasksAndBarriers(netFunc, barrierInfo, true);
    barrierInfo.buildTaskQueueTypeMap();

    // Identify existing position of DeclareBufferOp, will be used as insertion point
    // for new tasks that will be inserted in IR
    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    mlir::Operation* barrierInsertionPoint = &netFunc.getBody().front().front();
    if (!barriersOps.empty()) {
        auto barrierInsertionPointIt = barriersOps.begin();
        std::advance(barrierInsertionPointIt, std::distance(barriersOps.begin(), barriersOps.end()) - 1);
        barrierInsertionPoint = *barrierInsertionPointIt;
    }

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
    insertDMAForFetchTasks(_listOfDPUExecutionGroups, VPU::ExecutorKind::DPU, bufferInsertionPoint,
                           barrierInsertionPoint, builder, barrierInfo, dummyDmas);

    insertDMAForFetchTasks(_listOfSWExecutionGroups, VPU::ExecutorKind::SHAVE_ACT, bufferInsertionPoint,
                           barrierInsertionPoint, builder, barrierInfo, dummyDmas);

    // Apply the changes now as we need to make sure the new DMAs don't break the schedule
    barrierInfo.updateIR();

    // Correct the position of new dmas after realizing the changes from barrierInfo
    if (!dummyDmas.empty()) {
        for (auto dummyDma : dummyDmas) {
            SmallVector<VPURT::DeclareVirtualBarrierOp> barriers;
            auto waitBarriers = to_small_vector(dummyDma.getWaitBarriers());
            if (waitBarriers.empty()) {
                continue;
            }

            for (auto waitBarrier : waitBarriers) {
                auto waitBarrierOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(waitBarrier.getDefiningOp());
                barriers.push_back(waitBarrierOp);
            }

            llvm::sort(barriers, [&](const auto& lhs, const auto& rhs) {
                return compareVPURTOpPosition(lhs, rhs, barrierInfo);
            });
            auto lastWaitBarrier = barriers[barriers.size() - 1];
            auto lastTaskToUpdate = findLastTaskToUpdate({lastWaitBarrier}, barrierInfo, true);
            if (dummyDma->isBeforeInBlock(lastTaskToUpdate)) {
                dummyDma->moveAfter(lastTaskToUpdate);
            }
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
