//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/core/prefetch_edge_generator.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    AllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.memref();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);
    rewriter.replaceOpWithNewOp<IERT::StaticAllocOp>(origOp, val.getType(), offset);

    return mlir::success();
}

//
// FeasibleAllocationPass
//

class FeasibleAllocationPass final : public IERT::FeasibleAllocationBase<FeasibleAllocationPass> {
public:
    FeasibleAllocationPass(IERT::AttrCreateFunc memSpaceCb, IERT::AttrCreateFunc secondLevelMemSpaceCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    void updateAsyncExecuteOpPosition(mlir::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
                                      llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);
    void updateAsyncExecuteOpDependencies(AsyncDepsInfo& depsInfo,
                                          llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);
    // SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> removeRedundantPrefetchSpills(
    //         llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

    // mateusz
    void optimizeDataOpsSpills(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps,
                               AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo);

private:
    IERT::AttrCreateFunc _memSpaceCb;
    IERT::AttrCreateFunc _secondLvlMemSpaceCb;
    mlir::Attribute _memSpace;
    mlir::Attribute _secondLvlMemSpace;
};

FeasibleAllocationPass::FeasibleAllocationPass(IERT::AttrCreateFunc memSpaceCb,
                                               IERT::AttrCreateFunc secondLvlMemSpaceCb, Logger log)
        : _memSpaceCb(std::move(memSpaceCb)), _secondLvlMemSpaceCb(std::move(secondLvlMemSpaceCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult FeasibleAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    _memSpace = _memSpaceCb(ctx, memSpaceName.getValue());

    if (_memSpace == nullptr) {
        return mlir::failure();
    }

    _secondLvlMemSpace =
            (_secondLvlMemSpaceCb != nullptr ? _secondLvlMemSpaceCb(ctx, secondLvlMemSpaceName.getValue()) : nullptr);

    return mlir::success();
}

// This method will update all AsyncExecOp position in the block so that their
// order is aligned with order generated by list-scheduler. All operations will
// appear in non-descending order of start time. Such reordering is needed as
// execution order has more constraints than topological order that IR is
// aligned with. Without such sorting insertion of token dependency might hit
// an error.
void FeasibleAllocationPass::updateAsyncExecuteOpPosition(
        mlir::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
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

// This method will update all AsyncExecOp token dependencies so that resulting
// execution is aligned with order generated by list-scheduler
void FeasibleAllocationPass::updateAsyncExecuteOpDependencies(
        AsyncDepsInfo& depsInfo, llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps) {
    // Go through all the tasks and add token dependencies between
    // all tasks with start time t to all tasks with time t+1
    for (auto opIt = scheduledOps.begin(); opIt != scheduledOps.end(); opIt++) {
        if (!opIt->isOriginalOp()) {
            continue;
        }
        size_t nextTimeDiff = 0;
        for (auto nextTimeOpIt = opIt; nextTimeOpIt != scheduledOps.end(); nextTimeOpIt++) {
            if (!nextTimeOpIt->isOriginalOp()) {
                continue;
            } else if (nextTimeDiff == 0 && nextTimeOpIt->time_ > opIt->time_) {
                nextTimeDiff = nextTimeOpIt->time_ - opIt->time_;
            }

            if (nextTimeDiff != 0) {
                if (nextTimeOpIt->time_ == opIt->time_ + nextTimeDiff) {
                    // Insert dependency between op at time t to op at
                    // time t+1
                    auto srcAsyncOp = depsInfo.getExecuteOpAtIndex(opIt->op_);
                    auto dstAsyncOp = depsInfo.getExecuteOpAtIndex(nextTimeOpIt->op_);
                    VPUX_THROW_UNLESS((srcAsyncOp != nullptr) && (dstAsyncOp != nullptr),
                                      "srcAsyncOp/dstAsyncOp not located based on index");
                    depsInfo.addDependency(srcAsyncOp, dstAsyncOp);
                } else if (nextTimeOpIt->time_ > (opIt->time_ + nextTimeDiff)) {
                    break;
                }
            }
        }
    }
    depsInfo.updateTokenDependencies();
}

// mateusz
void FeasibleAllocationPass::optimizeDataOpsSpills(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps,
                                                   AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo) {
    // Collect information about all data ops that have been spilled
    // For each such dataOp store a sequence of indexes for scheduleOps array
    // where each entry corresponds to related spillWrite/Read operation. First entry
    // (index 0) is the index for original dataOp
    std::unordered_map<FeasibleMemoryScheduler::operationIdxType, SmallVector<size_t>> dataOpSpillTree;
    for (unsigned i = 0; i < scheduledOps.size(); i++) {
        auto& op = scheduledOps[i];
        // Check if this is spillRead/Write of data op
        if ((op.isSpillWrite() || op.isSpillRead()) && op.isDataOp()) {
            // Find if this is spilling of already encountered dataOp
            auto dataOpIt = dataOpSpillTree.find(op.op_);
            if (dataOpIt != dataOpSpillTree.end()) {
                // If dataOp was already identified store index of related spill operation
                dataOpIt->second.push_back(i);
            } else {
                // If this is spilling of new op, find source op and check if this is dataOp
                int j;
                for (j = i - 1; j >= 0; j--) {
                    if (scheduledOps[j].isOriginalOp() && scheduledOps[j].op_ == op.op_) {
                        // As a first element store index to original operation
                        dataOpSpillTree[scheduledOps[j].op_].push_back(j);
                        // Store index to identified spillWrite/Read operation
                        dataOpSpillTree[scheduledOps[j].op_].push_back(i);
                        break;
                    }
                }
                VPUX_THROW_UNLESS(j >= 0,
                                  "Unable to find in scheduled ops original operation for a given spill op '{0}'",
                                  op.op_);
            }
        }
    }

    // Dump data ops spilling information
    std::cout << "Data operation spilling sequence:\n";
    for (auto& dataOp : dataOpSpillTree) {
        std::cout << "  Operation " << dataOp.first << "\n";
        for (auto& i : dataOp.second) {
            auto& op = scheduledOps[i];
            std::cout << "    [" << i << "]: op = " << op.op_ << "\t type = " << op.opTypeName().data()
                      << "\t time = " << op.time_ << "\n";
        }
    }

    SmallVector<size_t> operationIndexesToRemove;
    // Check if between original op / spillRead and spillWrite buffer
    // from dataOp is used by any operation
    for (auto& dataOp : dataOpSpillTree) {
        std::cout << "  Operation " << dataOp.first << "\n";
        auto dataOpSpillIndexes = dataOp.second;
        for (size_t i = 0; i < dataOpSpillIndexes.size() - 1; i++) {
            if (scheduledOps[dataOpSpillIndexes[i]].isSpillWrite()) {
                continue;
            }
            auto& origOrSpillReadOpIndex = dataOpSpillIndexes[i];

            if (!scheduledOps[dataOpSpillIndexes[i + 1]].isSpillWrite()) {
                continue;
            }
            auto nextSpillWriteOpIndex = dataOpSpillIndexes[i + 1];

            VPUX_THROW_UNLESS(origOrSpillReadOpIndex < nextSpillWriteOpIndex,
                              "Incorrect order of indexes of spill read and next spill write ops for scheduledOps");

            bool isBufferUsedAsArgument = false;
            bool isBufferUsedAsResult = false;
            auto buffer = scheduledOps[origOrSpillReadOpIndex].getBuffer(0);
            for (size_t schedOpIdx = origOrSpillReadOpIndex + 1; schedOpIdx < nextSpillWriteOpIndex; schedOpIdx++) {
                // TODO: optimize below code
                auto execOp = depsInfo.getExecuteOpAtIndex(scheduledOps[schedOpIdx].op_);

                for (auto operand : execOp->getOperands()) {
                    if (operand.getType().isa<mlir::async::ValueType>()) {
                        if (aliasInfo.getRoot(operand) == buffer) {
                            isBufferUsedAsArgument = true;
                            break;
                        }
                    }
                }

                for (auto res : execOp.results()) {
                    if (aliasInfo.getRoot(res) == buffer) {
                        isBufferUsedAsResult = true;
                        break;
                    }
                }

                if (isBufferUsedAsArgument || isBufferUsedAsResult) {
                    break;
                }
            }

            auto isBufferUsed = isBufferUsedAsArgument || isBufferUsedAsResult;

            // If buffer was not used by any operation in between then given read-write pair is not needed
            // This can happen if scheduler prefetched dataOp which got immediately spilled
            if (!isBufferUsed) {
                std::cout << "  Buffer not used at all\n";
                // ops can be removed, update next operation
                operationIndexesToRemove.push_back(origOrSpillReadOpIndex);
                operationIndexesToRemove.push_back(nextSpillWriteOpIndex);

                if (scheduledOps[origOrSpillReadOpIndex].isOriginalOp()) {
                    // In such case update next read operation to be original operation
                    for (size_t j = i + 2; j < dataOpSpillIndexes.size(); j++) {
                        auto nextSpillReadIndex = dataOpSpillIndexes[j];
                        if (scheduledOps[nextSpillReadIndex].isSpillRead()) {
                            scheduledOps[nextSpillReadIndex].opType_ = scheduledOps[origOrSpillReadOpIndex].opType_;
                            break;
                        }
                    }
                }
            } else if (isBufferUsedAsArgument && !isBufferUsedAsResult) {
                // If buffer was used just as an argument then next spillWrite can be removed
                // as buffer state has not changed
                operationIndexesToRemove.push_back(nextSpillWriteOpIndex);
            }
        }
    }

    // Sort operation indexes
    std::sort(operationIndexesToRemove.begin(), operationIndexesToRemove.end());

    // Remove in reverse order to have indexes valid after erasing entries in scheduledOp
    for (auto opIt = operationIndexesToRemove.rbegin(); opIt != operationIndexesToRemove.rend(); opIt++) {
        scheduledOps.erase(scheduledOps.begin() + *opIt);
    }
}

// // only optimize prefetch spills for now, extend to optimize all spills
// // TODO: This is temporary code and will be replaced by more generic spilling optimization solution
// SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> FeasibleAllocationPass::removeRedundantPrefetchSpills(
//         llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps) {
//     SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> optimizedSchedule;

//     std::unordered_map<size_t, SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>> potentialRedundantDataSpills;
//     std::set<size_t> prefetchedOps;

//     for (auto& schedOp : scheduledOps) {
//         if (schedOp.isPrefetched()) {
//             prefetchedOps.insert(schedOp.op_);
//         } else if (!schedOp.isOriginalOp() && schedOp.isDataOp()) {
//             potentialRedundantDataSpills[schedOp.op_].push_back(schedOp);
//         }
//     }

//     size_t removedSpillWriteStallTime = 0;
//     for (auto schedOp : scheduledOps) {
//         // remove spill write time
//         schedOp.time_ -= removedSpillWriteStallTime;
//         // check if redundant op can be removed
//         if (prefetchedOps.find(schedOp.op_) != prefetchedOps.end() &&
//             potentialRedundantDataSpills.find(schedOp.op_) != potentialRedundantDataSpills.end()) {
//             // only insert the spill read but as original
//             if (schedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_READ) {
//                 schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_OP;
//                 optimizedSchedule.push_back(schedOp);
//             } else if (schedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_WRITE) {
//                 std::cout << "Removed redundat spill of opIdx: " << schedOp.op_ << std::endl;
//                 ++removedSpillWriteStallTime;
//             }
//         } else {
//             optimizedSchedule.push_back(schedOp);
//         }
//     }

//     return optimizedSchedule;
// }

void FeasibleAllocationPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    // linear scan
    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    auto available = resources.getAvailableMemory(_memSpace);
    const auto maxSize = available.size();
    const uint64_t alignment = 64;

    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), alignment);
    auto& aliasesInfo = getChildAnalysis<AliasesInfo>(netFunc);
    auto& liveRangeInfo = getChildAnalysis<MemLiveRangeInfo>(netFunc);
    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    // Copy classes for iteration with prefetch edges, as for prefetching
    // scheduler will run twice and first iteration is used to gather information
    // about the schedule and second one will perform the final allocation
    auto prefetchScan = scan;
    auto prefetchLiveRangeInfo = liveRangeInfo;

    // feasible memory scheduler - list scheduler
    FeasibleMemoryScheduler scheduler(_memSpace, liveRangeInfo, depsInfo, aliasesInfo, _log, scan);

    // 1. initial schedule
    auto scheduledOps = scheduler.generateSchedule();

    // 2. prefetching
    bool PREFETCHING_ENABLED = true;
    if (PREFETCHING_ENABLED) {
        // 2.1. optimization for inital schedule - generating prefetch edges
        PrefetchEdgeGenerator PrefetchEdgeGenerator(scheduledOps, depsInfo);
        auto prefetchEdges = PrefetchEdgeGenerator.generatePrefetchEdges();

        // 2.2. schedule again with prefetching
        if (!prefetchEdges.empty()) {
            FeasibleMemoryScheduler schedulerWithPrefetch(_memSpace, prefetchLiveRangeInfo, depsInfo, aliasesInfo, _log,
                                                          prefetchScan);
            scheduledOps = schedulerWithPrefetch.generateSchedule(prefetchEdges);

            scan = prefetchScan;
        }
    }

    std::cout << "Schedule before optimize spills\n";
    for (const auto& op : scheduledOps) {
        std::string resourceInfo = "<none>";
        if (op.hasActiveResource()) {
            resourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfResources(); resourceIdx++) {
                if (op.isActiveResource(resourceIdx)) {
                    resourceInfo += "resource = [" + std::to_string(op.beginResource(resourceIdx)) + " " +
                                    std::to_string(op.endResource(resourceIdx)) + "] size = " +
                                    std::to_string((op.endResource(resourceIdx) - op.beginResource(resourceIdx))) +
                                    ", ";
                }
            }
        }
        std::cout << "op = " << op.op_ << "\t type = " << op.opTypeName().data() << "\t time = " << op.time_ << " \t"
                  << resourceInfo << std::endl;
    }

    // mateusz
    optimizeDataOpsSpills(scheduledOps, depsInfo, aliasesInfo);

    // 3. optimize spills
    // scheduledOps = removeRedundantPrefetchSpills(scheduledOps);

    std::cout << "\nSchedule after optimize spills\n";
    for (const auto& op : scheduledOps) {
        std::string resourceInfo = "<none>";
        if (op.hasActiveResource()) {
            resourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfResources(); resourceIdx++) {
                if (op.isActiveResource(resourceIdx)) {
                    resourceInfo += "resource = [" + std::to_string(op.beginResource(resourceIdx)) + " " +
                                    std::to_string(op.endResource(resourceIdx)) + "] size = " +
                                    std::to_string((op.endResource(resourceIdx) - op.beginResource(resourceIdx))) +
                                    ", ";
                }
            }
        }
        std::cout << "op = " << op.op_ << "\t type = " << op.opTypeName().data() << "\t time = " << op.time_ << " \t"
                  << resourceInfo << std::endl;
    }

    FeasibleMemorySchedulerSpilling spilling(netFunc, _memSpace, _secondLvlMemSpace, depsInfo, aliasesInfo, _log, scan);
    spilling.removeRedundantSpillWrites(scheduledOps);

    // for (const auto& op : scheduledOps) {
    //     std::string resourceInfo = "<none>";
    //     if (op.hasActiveResource()) {
    //         resourceInfo = "";
    //         for (size_t resourceIdx = 0; resourceIdx < op.numOfResources(); resourceIdx++) {
    //             if (op.isActiveResource(resourceIdx)) {
    //                 resourceInfo += "resource = [" + std::to_string(op.beginResource(resourceIdx)) + " " +
    //                                 std::to_string(op.endResource(resourceIdx)) + "] size = " +
    //                                 std::to_string((op.endResource(resourceIdx) - op.beginResource(resourceIdx))) +
    //                                 ", ";
    //             }
    //         }
    //     }
    //     _log.trace("op = '{0}'\t type = '{1}'\t time = '{2}'\t '{3}'", op.op_, op.opTypeName(), op.time_,
    //     resourceInfo); std::cout << "op = " << op.op_ << "\t type = " << op.opTypeName().data() << "\t time = " <<
    //     op.time_ << " \t "
    //               << resourceInfo << std::endl;
    // }

    // 4. re-order the IR
    updateAsyncExecuteOpPosition(netFunc, depsInfo, scheduledOps);

    // 5. insert spill dmas
    spilling.insertSpillCopyOps(scheduledOps);

    // 6. update dependencies
    updateAsyncExecuteOpDependencies(depsInfo, scheduledOps);

    // 7. convert to allocated ops
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().dyn_cast<mlir::MemRefType>();
        return type == nullptr || type.getMemorySpace() != _memSpace;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AllocRewrite>(scan.handler(), &ctx, _log);  // mateusz

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
        return;
    }

    resources.setUsedMemory(_memSpace, scan.handler().maxAllocatedSize());  // mateusz
}

}  // namespace

//
// createFeasibleAllocationPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createFeasibleAllocationPass(AttrCreateFunc memSpaceCb,
                                                                     AttrCreateFunc secondLvlMemSpaceCb, Logger log) {
    return std::make_unique<FeasibleAllocationPass>(std::move(memSpaceCb), std::move(secondLvlMemSpaceCb), log);
}
