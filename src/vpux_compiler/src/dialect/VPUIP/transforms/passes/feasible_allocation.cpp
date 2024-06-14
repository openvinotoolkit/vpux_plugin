//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/core/prefetch_data_ops.hpp"
#include "vpux/compiler/core/schedule_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include "vpux/compiler/core/developer_build_utils.hpp"

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

using namespace vpux;

namespace {

//
// MemRefAllocRewrite
//

class MemRefAllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    MemRefAllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult MemRefAllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.getMemref();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto valType = val.getType().cast<vpux::NDTypeInterface>();
    auto section = VPURT::getBufferSection(valType.getMemoryKind());
    auto sectionIndex = valType.getMemSpace().getIndex();

    if (sectionIndex.has_value()) {
        auto sectionIndexArrayAttr = getIntArrayAttr(getContext(), ArrayRef(sectionIndex.value()));
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, sectionIndexArrayAttr,
                                                            offset, nullptr);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, offset);
    }

    return mlir::success();
}

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<VPURT::Alloc> {
public:
    AllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::Alloc>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::Alloc origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(VPURT::Alloc origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.getBuffer();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto valType = val.getType().cast<vpux::NDTypeInterface>();
    auto section = VPURT::getBufferSection(valType.getMemoryKind());
    auto sectionIndex = valType.getMemSpace().getIndex();

    auto swizzlingKey = origOp.getSwizzlingKeyAttr();

    if (sectionIndex.has_value()) {
        auto sectionIndexArrayAttr = getIntArrayAttr(getContext(), ArrayRef(sectionIndex.value()));
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, sectionIndexArrayAttr,
                                                            offset, swizzlingKey);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, nullptr, offset,
                                                            swizzlingKey);
    }

    return mlir::success();
}

//
// AllocDistributedRewrite
//

class AllocDistributedRewrite final : public mlir::OpRewritePattern<VPURT::AllocDistributed> {
public:
    AllocDistributedRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::AllocDistributed>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::AllocDistributed origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocDistributedRewrite::matchAndRewrite(VPURT::AllocDistributed origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.getBuffer();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto section = VPURT::getBufferSection(val.getType().cast<vpux::NDTypeInterface>().getMemoryKind());

    auto swizzlingKey = origOp.getSwizzlingKeyAttr();

    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, nullptr, offset, swizzlingKey);

    return mlir::success();
}

//
// FeasibleAllocationPass
//

class FeasibleAllocationPass final : public VPUIP::FeasibleAllocationBase<FeasibleAllocationPass> {
public:
    FeasibleAllocationPass(VPUIP::MemKindCreateFunc memKindCb, VPUIP::MemKindCreateFunc secondLevelmemKindCb,
                           bool linearizeSchedule, bool enablePipelining, bool enablePrefetching,
                           bool optimizeFragmentation, bool optimizeDynamicSpilling, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    void updateAsyncExecuteOpPosition(mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
                                      llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);
    void updateAsyncExecuteOpPositionOfSpillOps(AsyncDepsInfo& depsInfo,
                                                FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void assignCyclesToExecOps(AsyncDepsInfo& depsInfo, FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void linearizeComputeOps(bool linearizeSchedule, bool enablePipelining, mlir::func::FuncOp& netFunc,
                             AsyncDepsInfo& depsInfo);

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPUIP::MemKindCreateFunc _secondLvlMemKindCb;
    VPU::MemoryKind _memKind{vpux::VPU::MemoryKind::CMX_NN};
    VPU::MemoryKind _secondLvlMemKind{vpux::VPU::MemoryKind::DDR};
    mlir::SymbolRefAttr _memKindAttr;
    bool _linearizeSchedule{false};
    bool _enablePipelining{true};
    bool _enablePrefetching{true};
    bool _enableScheduleStatistics{false};
    bool _optimizeFragmentation{true};
    bool _optimizeDynamicSpilling{true};
};

FeasibleAllocationPass::FeasibleAllocationPass(VPUIP::MemKindCreateFunc memKindCb,
                                               VPUIP::MemKindCreateFunc secondLvlmemKindCb, bool linearizeSchedule,
                                               bool enablePipelining, bool enablePrefetching,
                                               bool optimizeFragmentation, bool optimizeDynamicSpilling, Logger log)
        : _memKindCb(std::move(memKindCb)),
          _secondLvlMemKindCb(std::move(secondLvlmemKindCb)),
          _linearizeSchedule(linearizeSchedule),
          _enablePipelining(enablePipelining),
          _enablePrefetching(enablePrefetching),
          _optimizeFragmentation(optimizeFragmentation),
          _optimizeDynamicSpilling(optimizeDynamicSpilling) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult FeasibleAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.has_value()) {
        return mlir::failure();
    }

    _memKind = maybeMemKind.value();
    _memKindAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(_memKind));

    auto maybeSecondLvlMemKind = _secondLvlMemKindCb(secondLvlMemSpaceName.getValue());
    if (!maybeSecondLvlMemKind.has_value()) {
        return mlir::failure();
    }

    _secondLvlMemKind = maybeSecondLvlMemKind.value();

    if (enablePipelining.hasValue()) {
        _enablePipelining = enablePipelining.getValue();
    }

    if (enablePrefetching.hasValue()) {
        _enablePrefetching = enablePrefetching.getValue();
    }

    if (optimizeDynamicSpilling.hasValue()) {
        _optimizeDynamicSpilling = optimizeDynamicSpilling.getValue();
    }

    if (linearizeSchedule.hasValue()) {
        _linearizeSchedule = linearizeSchedule.getValue();
    }

    return mlir::success();
}

// This method will update all AsyncExecOp position in the block so that their
// order is aligned with order generated by list-scheduler. All operations will
// appear in non-descending order of start time. Such reordering is needed as
// execution order has more constraints than topological order that IR is
// aligned with. Without such sorting insertion of token dependency might hit
// an error.
void FeasibleAllocationPass::updateAsyncExecuteOpPosition(
        mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
        llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps) {
    // Update placement of AsyncExecuteOps
    mlir::Operation* prevAsyncOp = nullptr;
    for (auto& schedOp : scheduledOps) {
        if (!schedOp.isOriginalOp()) {
            continue;
        }
        mlir::Operation* asyncOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        VPUX_THROW_UNLESS(asyncOp != nullptr, "AsyncOp not located based on index");
        if (prevAsyncOp != nullptr) {
            asyncOp->moveAfter(prevAsyncOp);
        } else {
            // For the first element place it before current first async exec op
            auto firstAsyncExecOp = *(netFunc.getOps<mlir::async::ExecuteOp>().begin());
            asyncOp->moveBefore(firstAsyncExecOp);
        }
        prevAsyncOp = asyncOp;
    }
}

// This method will update all AsyncExecOp position in the block related to
// spill ops. The goal is to move them close to their dependency which can
// be some data or control flow source or an operation that executes on same queue type
// This is done to get rid of potential long dependencies across IR as scheduler
// initially places spill ops as late as they are needed but in fact based on their
// dependencies and executor type they will be executed earlier
void FeasibleAllocationPass::updateAsyncExecuteOpPositionOfSpillOps(
        AsyncDepsInfo& depsInfo, FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    std::map<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, mlir::async::ExecuteOp>
            lastExecOpOnGivenQueueInstance;
    SmallVector<std::pair<size_t, size_t>> pairOfSpillOpAndItsNewPlaceInSchedVec;

    for (auto& schedOp : scheduledOps) {
        if (schedOp.queueType.execKind != VPU::ExecutorKind::DMA_NN) {
            continue;
        }

        mlir::async::ExecuteOp execOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        VPUX_THROW_UNLESS(execOp != nullptr, "Async ExecuteOp not located based on index");

        if (!schedOp.isOriginalSpillWriteOp() && !schedOp.isOriginalSpillReadOp()) {
            for (auto instIndex : schedOp.executorInstanceMask.set_bits()) {
                auto queueKey = std::make_pair(schedOp.queueType, instIndex);
                lastExecOpOnGivenQueueInstance[queueKey] = execOp;
            }

            continue;
        }

        SmallVector<mlir::async::ExecuteOp> depExecOps;
        for (const auto& dep : execOp.getDependencies()) {
            depExecOps.push_back(dep.getDefiningOp<mlir::async::ExecuteOp>());
        }

        for (auto instIndex : schedOp.executorInstanceMask.set_bits()) {
            auto queueKey = std::make_pair(schedOp.queueType, instIndex);
            if (lastExecOpOnGivenQueueInstance.find(queueKey) != lastExecOpOnGivenQueueInstance.end()) {
                depExecOps.push_back(lastExecOpOnGivenQueueInstance[queueKey]);
            }
        }

        std::sort(depExecOps.begin(), depExecOps.end(),
                  [](mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
                      return execOp1->isBeforeInBlock(execOp2);
                  });

        if (!depExecOps.empty()) {
            auto lastDepExecOp = depExecOps.back();
            if (lastDepExecOp->getNextNode() != execOp.getOperation()) {
                execOp->moveAfter(lastDepExecOp);

                // Store information where given spill op is to be placed
                pairOfSpillOpAndItsNewPlaceInSchedVec.push_back(
                        std::make_pair(schedOp.op_, depsInfo.getIndex(lastDepExecOp)));
            }
        }

        for (auto instIndex : schedOp.executorInstanceMask.set_bits()) {
            auto queueKey = std::make_pair(schedOp.queueType, instIndex);
            lastExecOpOnGivenQueueInstance[queueKey] = execOp;
        }
    }

    // Update operation placement in scheduledOps vector
    for (auto& pairOfSpillOpAndItsNewPlaceInSched : pairOfSpillOpAndItsNewPlaceInSchedVec) {
        auto spillOp = pairOfSpillOpAndItsNewPlaceInSched.first;
        auto placeAfterOp = pairOfSpillOpAndItsNewPlaceInSched.second;

        // Update scheduledOps
        auto indexToInsertItr =
                std::find_if(scheduledOps.begin(), scheduledOps.end(), [&](const ScheduledOpInfo& opInfo) {
                    return opInfo.op_ == placeAfterOp;
                });

        auto indexOfSpillOpItr =
                std::find_if(scheduledOps.begin(), scheduledOps.end(), [&](const ScheduledOpInfo& opInfo) {
                    return opInfo.op_ == spillOp;
                });

        VPUX_THROW_WHEN(indexToInsertItr == scheduledOps.end(),
                        "No iterator located matching opIdx '{0}' in scheduledOps vec", placeAfterOp);
        VPUX_THROW_WHEN(indexOfSpillOpItr == scheduledOps.end(),
                        "No iterator located matching opIdx '{0}' in scheduledOps vec", spillOp);

        auto indexToInsert = std::distance(scheduledOps.begin(), indexToInsertItr) + 1;

        auto indexOfSpillOp = std::distance(scheduledOps.begin(), indexOfSpillOpItr);

        // Insert operation to new location that wil be aligned with changes in IR
        scheduledOps.insert(scheduledOps.begin() + indexToInsert, scheduledOps[indexOfSpillOp]);

        // Remove operation from old location
        indexOfSpillOp++;
        scheduledOps.erase(scheduledOps.begin() + indexOfSpillOp);
    }
}

// This method will check cycles after spilling and remove stall regions which
// might have been introduced by spilling optimizations and assign the cycle start
// and cycle end attribute to the async.execute operation
void FeasibleAllocationPass::assignCyclesToExecOps(AsyncDepsInfo& depsInfo,
                                                   FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    // find stalls on all pipelines
    auto stallsToRemove = getStallsOnAllExecutorPipelines(scheduledOps);

    // remove stalls from operations
    for (auto& schedOp : scheduledOps) {
        // sum stalls to current cycle
        size_t stallSize = 0;
        auto stalls = stallsToRemove.begin();
        while (stalls != stallsToRemove.end() && stalls->first < schedOp.cycleBegin_) {
            stallSize += checked_cast<size_t>(stalls->second - stalls->first);
            ++stalls;
        }
        // update cycles
        schedOp.cycleBegin_ -= stallSize;
        schedOp.cycleEnd_ -= stallSize;
        // store cycles for barrier scheduler
        auto execOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        execOp->setAttr(cycleBegin, getIntAttr(execOp->getContext(), schedOp.cycleBegin_));
        execOp->setAttr(cycleEnd, getIntAttr(execOp->getContext(), schedOp.cycleEnd_));
        if (schedOp.queueType.execKind == VPU::ExecutorKind::DMA_NN) {
            SmallVector<uint64_t> executorInstanceMaskVec;
            for (auto portIdx : schedOp.executorInstanceMask.set_bits()) {
                executorInstanceMaskVec.push_back(portIdx);
            }
            VPUIP::VPUIPDialect::setExecutorInstanceMask(
                    execOp, getIntArrayAttr(execOp->getContext(), executorInstanceMaskVec));
        }
    }
}

// This method will inject dependencies between operations based on linearization option
void FeasibleAllocationPass::linearizeComputeOps(bool linearizeSchedule, bool enablePipelining,
                                                 mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo) {
    // various linearization options
    mlir::async::ExecuteOp prevExecOp = nullptr;
    llvm::DenseMap<VPU::ExecutorKind, mlir::async::ExecuteOp> prevExecOpMap;
    netFunc->walk([&](mlir::async::ExecuteOp execOp) {
        const auto currExecutor = VPUIP::VPUIPDialect::getExecutorKind(execOp);
        if (!linearizeSchedule && !VPUIP::VPUIPDialect::isComputeExecutorKind(currExecutor)) {
            return;
        }

        if (linearizeSchedule || !enablePipelining) {
            if (prevExecOp != nullptr) {
                depsInfo.addDependency(prevExecOp, execOp);
            }
            prevExecOp = execOp;
        } else {
            if (prevExecOpMap.find(currExecutor) != prevExecOpMap.end()) {
                depsInfo.addDependency(prevExecOpMap[currExecutor], execOp);
            }
            prevExecOpMap[currExecutor] = execOp;
        }
    });
}

void FeasibleAllocationPass::safeRunOnFunc() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    parseEnv("IE_NPU_ENABLE_SCHEDULE_STATISTICS", _enableScheduleStatistics);

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    // cluster information
    auto tileOp = IE::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");
    auto tileCount = tileOp.getCount();

    auto dmaPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    auto dmaCount = dmaPorts.getCount();

    // linear scan
    auto available = IE::getAvailableMemory(module, _memKindAttr);
    const auto maxSize = available.size();
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;

    // Check for reserved memory which memory scheduler should take into account
    // so that they not overlap with other buffers
    auto reservedMemVec = IE::getReservedMemOffsetAndSizeVec(module, _memKindAttr);

    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), reservedMemVec, alignment);
    auto& aliasesInfo = getAnalysis<AliasesInfoMemType<VPU::MemoryKind::CMX_NN>>();
    auto& liveRangeInfo = getAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>>();
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();

    linearizeComputeOps(_linearizeSchedule, _enablePipelining, func, depsInfo);

    // VPUNN cost model
    const auto arch = VPU::getArch(module);
    const auto costModel = VPU::createCostModel(arch);

    // If schedule analysis is enabled dynamic spilling stats will be gathered
    vpux::SpillStats dynamicSpillingBeforePrefetching;
    vpux::SpillStats dynamicSpillingAfterPrefetching;
    vpux::SpillStats dynamicSpillingAfterSpillOptimizations;

    // feasible memory scheduler - list scheduler
    FeasibleMemoryScheduler scheduler(_memKind, _secondLvlMemKind, liveRangeInfo, depsInfo, _log, scan, arch, costModel,
                                      tileCount, dmaCount, _enableScheduleStatistics, _optimizeFragmentation);

    // 1. initial schedule
    auto scheduledOps = scheduler.generateSchedule();

    if (_enableScheduleStatistics) {
        dynamicSpillingBeforePrefetching = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 2. prefetching
    if (_enablePrefetching && !_linearizeSchedule) {
        PrefetchDataOps prefetching(scheduledOps, depsInfo);
        prefetching.enableDataOpPrefetching();

        LinearScan<mlir::Value, LinearScanHandler> prefetchScan(maxSize.count(), reservedMemVec, alignment);
        auto prefetchLiveRangeInfo = MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>{func, aliasesInfo};
        // prefetching logic has reordered IR, depsInfo needs to be regenerated since
        // scheduling depends on incrementing value of async-deps-info along IR
        depsInfo = AsyncDepsInfo{func};
        if (!_enablePipelining) {
            linearizeComputeOps(_linearizeSchedule, _enablePipelining, func, depsInfo);
        }

        FeasibleMemoryScheduler schedulerWithPrefetch(_memKind, _secondLvlMemKind, prefetchLiveRangeInfo, depsInfo,
                                                      _log, prefetchScan, arch, costModel, tileCount, dmaCount,
                                                      _enableScheduleStatistics, _optimizeFragmentation);
        scheduledOps = schedulerWithPrefetch.generateSchedule();
        scan = std::move(prefetchScan);
    }

    scheduler.cleanUpAndLogSchedule(scheduledOps);
    // TODO: recurse to strategy with useful info

    if (_enableScheduleStatistics) {
        dynamicSpillingAfterPrefetching = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 3. optimize spills
    FeasibleMemorySchedulerSpilling spilling(_memKind, _secondLvlMemKind, depsInfo, aliasesInfo, _log, scan);
    if (_optimizeDynamicSpilling) {
        spilling.optimizeDataOpsSpills(scheduledOps);
        spilling.removeComputeOpRelocationSpills(scheduledOps);
        spilling.removeRedundantSpillWrites(scheduledOps);
    }

    if (_enableScheduleStatistics) {
        dynamicSpillingAfterSpillOptimizations = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 3. re-order the IR
    updateAsyncExecuteOpPosition(func, depsInfo, scheduledOps);

    // 4. insert spill dmas
    spilling.insertSpillDmaOps(scheduledOps);

    // 5. add cycle info to async.execute
    assignCyclesToExecOps(depsInfo, scheduledOps);

    // 6. update dependencies
    // Recreate aliasesInfo after spill insertion to get updated information about
    // root buffers of affected spill result users.
    aliasesInfo = AliasesInfoMemType<VPU::MemoryKind::CMX_NN>(func);
    FeasibleMemorySchedulerControlEdges controlEdges(_memKind, depsInfo, aliasesInfo, _log, scan);
    // controlEdges.insertDependenciesBasic(scheduledOps); // Old method, maintained only for debug
    controlEdges.insertMemoryControlEdges(scheduledOps);
    // Linearize DMA tasks before unrolling will introduce additional dependency across different DMA engines.
    // But it's fine for single DMA engine. So insert dependency to simplify barrier scheduling.
    if (dmaCount == 1) {
        controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::DMA_NN);
    }
    linearizeComputeOps(_linearizeSchedule, _enablePipelining, func, depsInfo);
    controlEdges.updateDependenciesInIR();

    // After dependencies are determined, location of spill operations should be updated
    // to be close to their dependencies
    updateAsyncExecuteOpPositionOfSpillOps(depsInfo, scheduledOps);

    if (_enableScheduleStatistics) {
        // verify all dependencies preserved for correct analysis
        verifyDependenciesPreservedInCycles(depsInfo, scheduledOps);

        // schedule statistics
        printScheduleStatistics(func, depsInfo, _log, scheduledOps);

        // dynamic spilling statistics
        printSpillingStatistics(_log, dynamicSpillingBeforePrefetching, dynamicSpillingAfterPrefetching,
                                dynamicSpillingAfterSpillOptimizations);

        // create a tracing JSON
        createTracingJSON(func);
    }

    // 7. convert to allocated ops
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.getMemref().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::Alloc>([&](VPURT::Alloc op) {
        const auto type = op.getBuffer().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::AllocDistributed>([&](VPURT::AllocDistributed op) {
        const auto type = op.getBuffer().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemRefAllocRewrite>(scan.handler(), &ctx, _log);
    patterns.add<AllocRewrite>(scan.handler(), &ctx, _log);
    patterns.add<AllocDistributedRewrite>(scan.handler(), &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
        return;
    }

    IE::setUsedMemory(func, _memKindAttr, scan.handler().maxAllocatedSize());
}

}  // namespace

//
// createFeasibleAllocationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFeasibleAllocationPass(
        MemKindCreateFunc memKindCb, MemKindCreateFunc secondLvlmemKindCb, const bool linearizeSchedule,
        const bool enablePrefetching, const bool enablePipelining, const bool optimizeFragmentation,
        const bool optimizeDynamicSpilling, Logger log) {
    return std::make_unique<FeasibleAllocationPass>(std::move(memKindCb), std::move(secondLvlmemKindCb),
                                                    linearizeSchedule, enablePipelining, enablePrefetching,
                                                    optimizeFragmentation, optimizeDynamicSpilling, log);
}
