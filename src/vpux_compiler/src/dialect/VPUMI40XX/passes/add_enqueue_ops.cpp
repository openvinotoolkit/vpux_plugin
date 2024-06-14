//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

using namespace vpux;

namespace {

class AddEnqueueOpsPass : public VPUMI40XX::AddEnqueueOpsBase<AddEnqueueOpsPass> {
public:
    explicit AddEnqueueOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    static constexpr int64_t DMA_OUTSTANDING_TRANSACTIONS = 64;
    void safeRunOnFunc() final;
};

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps) {
    if (enquOps.size() == 0) {
        return;
    }

    auto ctx = enquOps[0].getContext();
    auto index = [&ctx](auto taskIdx) {
        return VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(taskIdx));
    };

    enquOps[0].getResult().setType(index(0));
    enquOps[0].getPreviousTaskIdxMutable().clear();

    for (size_t i = 1; i < enquOps.size(); i++) {
        auto enqu = enquOps[i];
        enqu.getResult().setType(index(i));
        enqu.getPreviousTaskIdxMutable().assign(enquOps[i - 1]);
    }

    return;
}

mlir::ValueRange getClosestProductionBarriers(VPURegMapped::TaskOpInterface taskOp) {
    do {
        auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(taskOp.getOperation());
        if (executableTaskOp && (executableTaskOp.updateBarriers().size() != 0)) {
            return executableTaskOp.updateBarriers();
        }

        auto taskOpUsers = taskOp.getOperation()->getResult(0).getUsers();
        auto nextTaskDown = llvm::find_if(taskOpUsers, [&taskOp](mlir::Operation* user) {
            auto next = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
            return next && (next.getPreviousTask() == taskOp);
        });

        taskOp = nextTaskDown != taskOpUsers.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nextTaskDown) : nullptr;

    } while (taskOp);

    return mlir::ValueRange{};
}

void dfs(mlir::Value val, llvm::SetVector<mlir::Value>& visited) {
    visited.insert(val);
    for (auto user : val.getUsers()) {
        auto barr = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(user);
        if (!barr)
            continue;
        if (!visited.contains(barr.getResult())) {
            dfs(barr.getResult(), visited);
        }
    }
}

llvm::SmallVector<mlir::Value> lca(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == rhs)
        return {lhs};

    llvm::SmallVector<mlir::Value> lcas;

    llvm::SetVector<mlir::Value> visitedLhs, visitedRhs;
    llvm::SetVector<mlir::Value> intersection;

    dfs(lhs, visitedLhs);
    dfs(rhs, visitedRhs);

    // get the intersection of the 2
    for (auto lhsIt : visitedLhs) {
        if (visitedRhs.contains(lhsIt)) {
            intersection.insert(lhsIt);
        }
    }

    // each barr who's deps is not in the intersection is an LCA
    for (auto val : intersection) {
        auto barr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(val.getDefiningOp());
        auto count = llvm::count_if(barr.getDependencies(), [&intersection](mlir::Value val) {
            return intersection.contains(val);
        });
        if (count == 0) {
            lcas.push_back(val);
        }
    }
    return lcas;
}

llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& lhs, mlir::Value rhs) {
    llvm::SmallVector<mlir::Value> lcas;

    for (auto val : lhs) {
        lcas.append(lca(val, rhs));
    }

    return lcas;
}

llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& barrierVals) {
    // sanity... only makes sense in debug modes
    assert(std::all_of(barrierVals.begin(), barrierVals.end(),
                       [](mlir::Value val) {
                           return mlir::isa<VPUMI40XX::ConfigureBarrierOp>(val.getDefiningOp());
                       }) &&
           "LCA requires all of the values to be defined by configureBarrierOps {0}");

    if (barrierVals.size() <= 1) {
        return barrierVals;
    }

    if (barrierVals.size() == 2) {
        return lca(barrierVals[0], barrierVals[1]);
    }

    llvm::SmallVector<mlir::Value> lcas = lca(barrierVals[0], barrierVals[1]);
    for (size_t i = 2; i < barrierVals.size(); ++i) {
        lcas = lca(lcas, barrierVals[i]);
    }

    return lcas;
}

VPURegMapped::TaskOpInterface getNextOp(VPURegMapped::TaskOpInterface op) {
    auto users = op.getResult().getUsers();
    auto nexOpIt = llvm::find_if(users, [&op](mlir::Operation* user) {
        auto nextTask = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
        return nextTask && (nextTask.getTaskType() == op.getTaskType()) && (nextTask.getPreviousTask() == op);
    });

    op = nexOpIt != users.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nexOpIt) : nullptr;
    return op;
}

// TODO: (E#115494) consider explicitly materializing the "previous ID" inside the IR
llvm::SmallVector<mlir::Value> getPreviousUsages(mlir::ValueRange barrs,
                                                 llvm::SmallVector<VPUMI40XX::ConfigureBarrierOp> allBarrs) {
    llvm::SmallVector<mlir::Value> previousUsages;

    // assuming taskIdx of barrier is the same as it's listIdx
    auto getPreviousUsage = [&allBarrs](int64_t startIdx, uint8_t pid) -> mlir::Value {
        if (startIdx == 0) {
            return nullptr;
        }
        for (auto idx = startIdx - 1; idx >= 0; idx--) {
            if (allBarrs[idx].getId() == pid) {
                return allBarrs[idx].getResult();
            }
        }
        return nullptr;
    };

    for (auto barr : barrs) {
        auto barrOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(barr.getDefiningOp());
        auto idx = barrOp.getType().getValue();
        auto pid = barrOp.getId();
        auto previousUsage = getPreviousUsage(idx, pid);
        if (previousUsage)
            previousUsages.push_back(previousUsage);
    }

    return previousUsages;
}

// TODO: ned to figure out a clean way to get barriers purely from taskOpInterface
VPUMI40XX::ExecutableTaskOpInterface getBarrieredOp(VPURegMapped::TaskOpInterface primary,
                                                    VPURegMapped::TaskOpInterface secondary) {
    if (primary.getTaskType() == VPURegMapped::TaskType::DPUInvariant) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(primary.getOperation());
    } else if (primary.getTaskType() == VPURegMapped::TaskType::ActKernelRange) {
        return mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(secondary.getOperation());
    } else {
        VPUX_THROW("Unknown TaskType for pair {0} {1}", primary.getResult(), secondary.getResult());
        return nullptr;
    }

    return nullptr;
}

// Check if barrier that was chosen for enqueuing a task does not depend on a barrier
// that is to be produced by this task itself what will create a deadlock during execution
bool verifyValidEnqueueBarrierForTask(mlir::Value enqueueBar, mlir::ValueRange taskBars, Logger log) {
    // Identify minimal virtual ID of barrier produced by task. Barriers below
    // this ID will not be analyzed as they cannot be dependant of task barriers
    unsigned int minVid = std::numeric_limits<unsigned int>::max();
    for (auto taskBar : taskBars) {
        auto vid = taskBar.getType().cast<VPURegMapped::IndexType>().getValue();
        minVid = std::min(minVid, vid);
    }

    if (enqueueBar.getType().cast<VPURegMapped::IndexType>().getValue() < minVid) {
        return true;
    }

    mlir::DenseSet<mlir::Value> explored;
    std::queue<mlir::Value> queue;

    // Perform BFS starting from enqueu barrier up the dependency chain
    // to see if this barrier does not depend on any barriers produced
    // by task itself
    queue.push(enqueueBar);
    explored.insert(enqueueBar);

    while (!queue.empty()) {
        auto bar = queue.front();
        queue.pop();

        auto taskBarIt = std::find(taskBars.begin(), taskBars.end(), bar);
        if (taskBarIt != taskBars.end()) {
            auto enqueueBarOp = enqueueBar.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
            auto taskBarOp = (*taskBarIt).getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
            log.error("Enqueue barrier '{0}' depends topologically on task to be enqueued itself which updates "
                      "barrier '{1}'",
                      enqueueBarOp, taskBarOp);
            return false;
        }

        auto barOp = bar.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
        for (auto barDep : barOp.getDependencies()) {
            // Ignore barriers which are earlier in schedule then minVid
            if (barDep.getType().cast<VPURegMapped::IndexType>().getValue() < minVid) {
                continue;
            }

            if (explored.find(barDep) == explored.end()) {
                queue.push(barDep);
                explored.insert(barDep);
            }
        }
    }

    return true;
}

void addEnqus(VPUMI40XX::MappedInferenceOp mpi, const VPURegMapped::TaskType primary,
              const VPURegMapped::TaskType secondary, const int64_t tilesCount,
              const llvm::SmallVector<vpux::VPUMI40XX::ConfigureBarrierOp>& barriers, mlir::Value& firstEnqu,
              VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder, int64_t& counter, Logger log) {
    auto ctx = mpi.getContext();

    auto getFetchTask = [](mlir::Value result) {
        auto fetchIt = llvm::find_if(result.getUsers(), [](mlir::Operation* user) {
            return mlir::isa<VPURegMapped::FetchTaskOp>(user);
        });

        auto fetchTask = fetchIt != result.getUsers().end() ? mlir::cast<VPURegMapped::FetchTaskOp>(*fetchIt) : nullptr;
        return fetchTask;
    };

    auto secondaryCompare = [](mlir::Operation* lhs, mlir::Operation* rhs) {
        auto lhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(lhs);
        auto rhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(rhs);
        return lhsTask.getIndexType().getValue() < rhsTask.getIndexType().getValue();
    };

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        auto startVal = mpi.getListHead(primary, tileIdx);
        if (!startVal)
            continue;

        // reset local previousEnqu
        VPURegMapped::EnqueueOp localPreviousEnqu;

        // strongly assume that the FIRST OP always has a fetchTask
        auto previousFetchTask = getFetchTask(startVal);
        VPUX_THROW_UNLESS(previousFetchTask, "Starting OP {0} does not have a fetchTask", startVal);

        VPURegMapped::TaskOpInterface taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(startVal.getDefiningOp());
        do {
            auto filteredRange =
                    to_small_vector(taskOp.getResult().getUsers() | vpux::filtered([&secondary](mlir::Operation* op) {
                                        // filter out the usages that are of the secondaryType
                                        auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(op);
                                        return taskOp && taskOp.getTaskType() == secondary;
                                    }));

            auto firstSecondaryIt = vpux::min_element(filteredRange, secondaryCompare);
            auto lastSecondaryIt = vpux::max_element(filteredRange, secondaryCompare);
            auto firstSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**firstSecondaryIt);
            auto lastSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**lastSecondaryIt);

            auto fetchTask = getFetchTask(taskOp.getResult());
            previousFetchTask = fetchTask ? fetchTask : previousFetchTask;
            auto fetchTaskUpdateBarrs = getClosestProductionBarriers(
                    mlir::cast<VPURegMapped::TaskOpInterface>(previousFetchTask.getOperation()));

            mlir::ValueRange updateBarriers = getBarrieredOp(taskOp, lastSecondary).updateBarriers();
            auto previousUsages = getPreviousUsages(updateBarriers, barriers);
            previousUsages.append(fetchTaskUpdateBarrs.begin(), fetchTaskUpdateBarrs.end());

            auto lcas = lca(previousUsages);

            VPUX_THROW_UNLESS(lcas.size(), "Could not find a lowest commont ancestor");

            auto minBarrierIt = std::min_element(lcas.begin(), lcas.end(), [](mlir::Value lhs, mlir::Value rhs) {
                auto lhsBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp());
                auto rhsBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp());
                return lhsBarr.getType().getValue() < rhsBarr.getType().getValue();
            });
            auto enqueueTarget = *minBarrierIt;

            // we can get into a corner case where a subsequent DPU can be enqueued before it's predecessing enque,
            // based on barrier constraints. we could include the previous barriers enqueue barrier into the LCA, but
            // need to prove that will be satisfactory. For now, assuming that barriers are topologically ordered, we
            // will compare with
            // the previous enqueOp's triggering barriers list idx

            // not needed anymore as long as we are sure to topologically sory enqueue tasks themselves
            // since now Runtime guarantees ordered submission of enqueues
            if (localPreviousEnqu) {
                auto previousEnquBarrier = localPreviousEnqu.getBarrier();

                enqueueTarget = std::max(previousEnquBarrier, enqueueTarget, [](mlir::Value lhs, mlir::Value rhs) {
                    return lhs.getType().cast<VPURegMapped::IndexType>().getValue() <
                           rhs.getType().cast<VPURegMapped::IndexType>().getValue();
                });
            }

            VPUX_THROW_UNLESS(verifyValidEnqueueBarrierForTask(enqueueTarget, updateBarriers, log),
                              "Invalid enqueue barrier found for task '{0}'", taskOp);

            // if the previous enqueue's barrier is the same as the target barrier, we can just add this variant range
            // to the previous enqueue. This is made with the assumption that we topologically iterate over the variants
            // list by their listOrder

            if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                localPreviousEnqu.getEndMutable().assign(lastSecondary->getResult(0));
            } else {
                auto index = VPURegMapped::IndexType::get(ctx, counter);
                mlir::Value previousEnquVal = localPreviousEnqu
                                                      ? localPreviousEnqu.getResult()
                                                      : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                        taskOp->getLoc(), index, previousEnquVal, enqueueTarget, secondary,
                        firstSecondary->getResult(0), lastSecondary->getResult(0));
                counter++;

                if (!firstEnqu)
                    firstEnqu = localPreviousEnqu.getResult();
            }

            taskOp = getNextOp(taskOp);
        } while (taskOp);

        globalPreviousEnqu = localPreviousEnqu;
    }
}

void AddEnqueueOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    auto barriers = to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());

    mlir::Value firstEnqu;
    VPURegMapped::EnqueueOp globalPreviousEnqu;
    int64_t globalEnquCounter = 0;

    addEnqus(mpi, VPURegMapped::TaskType::DPUInvariant, VPURegMapped::TaskType::DPUVariant, tilesCount, barriers,
             firstEnqu, globalPreviousEnqu, builder, globalEnquCounter, _log);
    addEnqus(mpi, VPURegMapped::TaskType::ActKernelRange, VPURegMapped::TaskType::ActKernelInvocation, tilesCount,
             barriers, firstEnqu, globalPreviousEnqu, builder, globalEnquCounter, _log);

    std::array<VPUMI40XX::ExecutableTaskOpInterface, DMA_OUTSTANDING_TRANSACTIONS> outstandingEnqueuedDmas;

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask)
                continue;
            // reset local previousEnqu
            VPURegMapped::EnqueueOp localPreviousEnqu;
            // reset previousBuffer
            outstandingEnqueuedDmas.fill(nullptr);
            int64_t outstandingEnquOpsCounter = 0;

            while (dmaTask) {
                auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(dmaTask.getDefiningOp());
                if (executableTaskOp &&
                    (executableTaskOp.updateBarriers().size() || executableTaskOp.waitBarriers().size())) {
                    auto targetBarrs = executableTaskOp.updateBarriers().size() ? executableTaskOp.updateBarriers()
                                                                                : executableTaskOp.waitBarriers();
                    auto previousUsages = getPreviousUsages(targetBarrs, barriers);

                    auto oldestOutstandingDma = outstandingEnqueuedDmas[outstandingEnquOpsCounter];
                    if (oldestOutstandingDma) {
                        auto outstandingBarrierCondition = oldestOutstandingDma.updateBarriers();
                        previousUsages.append(outstandingBarrierCondition.begin(), outstandingBarrierCondition.end());
                    }

                    auto lcas = lca(previousUsages);

                    if (lcas.size()) {
                        auto minBarrierIt =
                                std::min_element(lcas.begin(), lcas.end(), [](mlir::Value lhs, mlir::Value rhs) {
                                    auto lhsBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp());
                                    auto rhsBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp());
                                    return lhsBarr.getType().getValue() < rhsBarr.getType().getValue();
                                });
                        auto enqueueTarget = *minBarrierIt;

                        if (localPreviousEnqu) {
                            auto previousEnquBarrier = localPreviousEnqu.getBarrier();
                            enqueueTarget =
                                    std::max(previousEnquBarrier, enqueueTarget, [](mlir::Value lhs, mlir::Value rhs) {
                                        return lhs.getType().cast<VPURegMapped::IndexType>().getValue() <
                                               rhs.getType().cast<VPURegMapped::IndexType>().getValue();
                                    });
                        }

                        VPUX_THROW_UNLESS(verifyValidEnqueueBarrierForTask(enqueueTarget, targetBarrs, _log),
                                          "Invalid enqueue barrier found for task '{0}'", executableTaskOp);

                        if (localPreviousEnqu && (localPreviousEnqu.getBarrier() == enqueueTarget)) {
                            localPreviousEnqu.getEndMutable().assign(dmaTask);
                        } else {
                            auto index = VPURegMapped::IndexType::get(netFunc.getContext(),
                                                                      checked_cast<uint32_t>(globalEnquCounter));
                            mlir::Value previousEnquVal =
                                    localPreviousEnqu ? localPreviousEnqu.getResult()
                                                      : (globalPreviousEnqu ? globalPreviousEnqu.getResult() : nullptr);
                            localPreviousEnqu = builder.create<VPURegMapped::EnqueueOp>(
                                    dmaTask.getLoc(), index, previousEnquVal, enqueueTarget,
                                    VPURegMapped::TaskType::DMA, dmaTask, dmaTask);

                            outstandingEnqueuedDmas[outstandingEnquOpsCounter] = executableTaskOp;
                            outstandingEnquOpsCounter = (outstandingEnquOpsCounter + 1) % DMA_OUTSTANDING_TRANSACTIONS;
                            globalEnquCounter++;
                        }

                        if (!firstEnqu)
                            firstEnqu = localPreviousEnqu.getResult();
                    } else {
                        if (localPreviousEnqu) {
                            localPreviousEnqu.getEndMutable().assign(dmaTask);
                        }
                    }
                } else if (localPreviousEnqu) {
                    localPreviousEnqu.getEndMutable().assign(dmaTask);
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaTask.getDefiningOp()));
                dmaTask = nextDma ? nextDma.getResult() : nullptr;
            }
        }
    }

    // for multi tile need to sort enqueuOps to be contigous for the same barrier
    for (auto barrier : barriers) {
        llvm::DenseMap<VPURegMapped::TaskType, llvm::SmallVector<VPURegMapped::EnqueueOp>> buckets;
        for (auto user : barrier.getResult().getUsers()) {
            auto enqu = mlir::dyn_cast<VPURegMapped::EnqueueOp>(user);
            if (!enqu) {
                continue;
            }

            buckets[enqu.getTaskType()].push_back(enqu);
        }

        for (auto& mapIt : buckets) {
            llvm::sort(mapIt.getSecond(), [](VPURegMapped::EnqueueOp lhs, VPURegMapped::EnqueueOp rhs) {
                return lhs.getResult().getType().cast<VPURegMapped::IndexType>().getValue() <
                       rhs.getResult().getType().cast<VPURegMapped::IndexType>().getValue();
            });
            for (auto enqu : mapIt.getSecond()) {
                enqu.getOperation()->moveBefore(mpi.getOperation());
            }
        }
    }

    auto enquOps = to_small_vector(netFunc.getOps<VPURegMapped::EnqueueOp>());
    mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
    mpi.setWorkItemCount(enquOps.size());
    reindexEnqueueOps(enquOps);
}

}  // namespace

//
// createAddEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddEnqueueOpsPass(Logger log) {
    return std::make_unique<AddEnqueueOpsPass>(log);
}
