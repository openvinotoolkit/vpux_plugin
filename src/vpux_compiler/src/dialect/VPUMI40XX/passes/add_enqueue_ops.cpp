//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
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

// Check if barrier that was chosen for enqueuing a task does not depend on a barrier
// that is to be produced by this task itself what will create a deadlock during execution
bool verifyEnqueueBarrierHasNoTopoDepOnBarrs(mlir::Value enqueueBar, mlir::ValueRange taskBars, Logger log) {
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

// Go through all enqueue tasks and process whole schedule with respect to barrier consumption events
// and check if no enqueue task chosen barrier is not yet fully consumed at the moment of enqueuement
// what means that it will be consumed by some future tasks not yet enqueued
void verifyEnqueueBarrierIsNotBlockedByFutureTask(VPUMI40XX::MappedInferenceOp mpi,
                                                  SmallVector<VPURegMapped::EnqueueOp>& enquOps,
                                                  SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers,
                                                  SmallVector<SmallVector<mlir::Operation*>>& lastDmaWithNoEnqueue,
                                                  int64_t tilesCount, Logger log) {
    // Build information on all barrier consumer count
    SmallVector<int64_t> barrierConsumerCounter(barriers.size(), -1);
    auto barrierCount = barrierConsumerCounter.size();
    for (auto& barrier : barriers) {
        auto barrierIdx = barrier.getResult().getType().cast<VPURegMapped::IndexType>().getValue();
        VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                        "Invalid barrier index - {0} out of possible amount of barriers {1}", barrierIdx, barrierCount);
        VPUX_THROW_WHEN(barrierConsumerCounter[barrierIdx] > -1, "Barrier {0} has already been updated", barrierIdx);

        if (barrier.getConsumerCount().has_value()) {
            barrierConsumerCounter[barrierIdx] = barrier.getConsumerCount().value();
        }
    }

    // Before processing enqueue tasks first traverse all initial DMAs on each FIFO
    // as they do not have corresponding enqueue op. Update barriers they consume
    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTask = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTask) {
                continue;
            }

            if (lastDmaWithNoEnqueue[tileIdx][listIdx] == nullptr) {
                continue;
            }

            auto firstDmaTaskIdx = dmaTask.getType().cast<VPURegMapped::IndexType>().getValue();
            auto lastDmaTaskIdx = lastDmaWithNoEnqueue[tileIdx][listIdx]
                                          ->getResult(0)
                                          .getType()
                                          .cast<VPURegMapped::IndexType>()
                                          .getValue();

            log.trace("Process DMAs with no enqueue op - DMA{0}:{1}:{2}-{3}", tileIdx, listIdx, firstDmaTaskIdx,
                      lastDmaTaskIdx);

            mlir::Operation* dmaOp;
            mlir::Operation* nextDmaOp = dmaTask.getDefiningOp();

            do {
                dmaOp = nextDmaOp;

                if (auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(dmaOp)) {
                    for (const auto& waitBar : executableTaskOp.waitBarriers()) {
                        auto barrierIdx = waitBar.getType().cast<VPURegMapped::IndexType>().getValue();
                        VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                                        "Invalid barrier index - {0} out of possible amount of barriers {1}",
                                        barrierIdx, barrierCount);
                        barrierConsumerCounter[barrierIdx]--;
                    }
                }

                auto nextDma = VPUMI40XX::getNextOp(mlir::cast<VPURegMapped::TaskOpInterface>(dmaOp));
                nextDmaOp = nextDma ? nextDma.getOperation() : nullptr;

            } while (nextDmaOp != nullptr && dmaOp != lastDmaWithNoEnqueue[tileIdx][listIdx]);
        }
    }

    for (auto& enquOp : enquOps) {
        auto enqBarIdx = enquOp.getBarrier().getType().cast<VPURegMapped::IndexType>().getValue();

        log.trace("EnqOp '{0}' at barrier idx '{1}'", enquOp.getIndex().getType(), enqBarIdx);
        VPUX_THROW_TYPED_UNLESS(
                WlmRollbackException, barrierConsumerCounter[enqBarIdx] == 0,
                "Barrier '{0}' for enqueue not yet consumed, remaining counters: '{1}'. Execution blocked at "
                "enqueue op '{2}'",
                enqBarIdx, barrierConsumerCounter[enqBarIdx], enquOp);

        auto nextTaskOpToProcess = mlir::cast<VPURegMapped::TaskOpInterface>(enquOp.getStart().getDefiningOp());
        auto lastTaskOp = mlir::cast<VPURegMapped::TaskOpInterface>(enquOp.getEnd().getDefiningOp());
        VPURegMapped::TaskOpInterface taskOp;

        do {
            taskOp = nextTaskOpToProcess;

            VPUMI40XX::ExecutableTaskOpInterface barrieredOp;
            if (enquOp.getTaskType() == VPURegMapped::TaskType::DPUVariant) {
                auto dpuVariantOp = mlir::cast<VPUMI40XX::DPUVariantOp>(taskOp.getOperation());
                barrieredOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(
                        dpuVariantOp.getInvariant().getDefiningOp());
            } else {
                barrieredOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(taskOp.getOperation());
            }

            if (barrieredOp) {
                for (const auto& waitBar : barrieredOp.waitBarriers()) {
                    auto barrierIdx = waitBar.getType().cast<VPURegMapped::IndexType>().getValue();
                    VPUX_THROW_WHEN(barrierIdx >= barrierCount,
                                    "Invalid barrier index - {0} out of possible amount of barriers {1}", barrierIdx,
                                    barrierCount);
                    VPUX_THROW_UNLESS(barrierConsumerCounter[barrierIdx] > 0,
                                      "Barrier {0} consumer count cannot be decremented - {1}", barrierIdx,
                                      barrierConsumerCounter[barrierIdx]);
                    barrierConsumerCounter[barrierIdx]--;
                }
            }
            nextTaskOpToProcess = VPUMI40XX::getNextOp(taskOp);
        } while (taskOp != lastTaskOp);
    }
}

/*
                  Virt#617: Phys#11
                /        |          \
               /         |           \
             DMA1       DMA2         DMA0 --------> Virt#619: Phys#12
               |         |                                    |
               |         |                                Enqueue SHV0
               |         |
               |         |
               |         |
            Virt#620: Phys#15
                    |
                    |
                    .
                    .
                    .
            Virt#650: Phys#11
                    |
                   SHV0


1. If Enqueue of ShaveTasks happens at Virt#619
2. At the point of enqueue the previousUsage virt#617:phys#11 of waitBarrier virt#650:phys#11 has not
been consumed fully consumed (full consumption happens at Virt#620)
3. When this enqueue happens runtime checks the state of phys#11 and since phys#11 has not been reprogrammed yet it
has production counter 0 thus runtime allows the Shave task to execute
4. Because the SHV FIFO is empty, we start executing the SHV immediately while the DMA1 and DMA2 are still writing
to CMX

This sort of behaviour creates unstable runs as sometimes we will see the inference pass while some times the
runtime throws context violation
*/

// This function checks for above described case and delay enqueue if it's possible
mlir::Value delayEnqueueOpForFirstTaskInFifo(VPUMI40XX::ExecutableTaskOpInterface barrieredOp, mlir::Value enqTarget,
                                             const SmallVector<VPUMI40XX::ConfigureBarrierOp>& barriers,
                                             VPUMI40XX::lcaCache& cache) {
    auto enqTargetOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(enqTarget.getDefiningOp());
    auto enqBarrierIdx = enqTargetOp.getType().getValue();
    auto waitBarriers =
            llvm::SmallVector<mlir::Value>(barrieredOp.waitBarriers().begin(), barrieredOp.waitBarriers().end());
    auto waitBarrierUsages = VPUMI40XX::getPreviousUsages(waitBarriers, barriers);
    if (waitBarrierUsages.empty()) {
        return enqTarget;
    }
    auto maxBarrierIt = VPUMI40XX::getMaxBarrier(waitBarrierUsages);

    // maxBarrierOp here represents the last barrier which has the same pid as the wait barrier of
    // the taskOp
    auto maxBarrierOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(maxBarrierIt->getDefiningOp());
    if (maxBarrierOp.getType().getValue() >= enqBarrierIdx) {
        return enqTarget;
    }

    SmallVector<mlir::Value> maxBarrierUpdateBarriers;
    for (auto user : maxBarrierOp.getResult().getUsers()) {
        if (auto executableTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(user)) {
            auto waitBarriers = llvm::SmallVector<mlir::Value>(executableTaskOp.waitBarriers().begin(),
                                                               executableTaskOp.waitBarriers().end());
            if (VPUMI40XX::contains(waitBarriers, maxBarrierOp.getResult())) {
                auto updateBarriers = llvm::SmallVector<mlir::Value>(executableTaskOp.updateBarriers().begin(),
                                                                     executableTaskOp.updateBarriers().end());
                auto minUpdateBarrier = VPUMI40XX::getMinBarrier(updateBarriers);
                maxBarrierUpdateBarriers.push_back(*minUpdateBarrier);
            }
        }
    }

    // Min barrier here represents the first barrier that is updated by the users of maxBarrierOp
    // Min barrier is chosen for the check as we only care about reprogramming and making sure the barrier is
    // consumed rather than checking when all the users have finished and updated their barriers
    auto minBarrierUpdateBarrierIt = VPUMI40XX::getMinBarrier(maxBarrierUpdateBarriers);
    auto maxBarrierUpdateBarrierIt = VPUMI40XX::getMaxBarrier(maxBarrierUpdateBarriers);
    auto minBarrierUpdateBarrierOp =
            mlir::cast<VPUMI40XX::ConfigureBarrierOp>(minBarrierUpdateBarrierIt->getDefiningOp());
    auto maxBarrierUpdateBarrierOp =
            mlir::cast<VPUMI40XX::ConfigureBarrierOp>(maxBarrierUpdateBarrierIt->getDefiningOp());

    // We have case where the enqueue is between the first and the last update barrier of the previous usage of wait
    // barrier. These case can be problematic as we would not know if maxBarrierUpdateBarrierOp has been produced or
    // not hence can't garntee reprogramming of previous usage of wait barrier
    if (maxBarrierUpdateBarrierOp.getType().getValue() > enqBarrierIdx &&
        enqBarrierIdx > minBarrierUpdateBarrierOp.getType().getValue()) {
        auto lcas = VPUMI40XX::lca(maxBarrierUpdateBarriers, cache);
        return *VPUMI40XX::getMinBarrier(lcas);
    } else {
        return enqTarget;
    };
}

void addEnqus(VPUMI40XX::MappedInferenceOp mpi, const VPURegMapped::TaskType primary,
              const VPURegMapped::TaskType secondary, const int64_t tilesCount,
              const llvm::SmallVector<vpux::VPUMI40XX::ConfigureBarrierOp>& barriers, mlir::Value& firstEnqu,
              VPURegMapped::EnqueueOp& globalPreviousEnqu, mlir::OpBuilder& builder, int64_t& counter,
              VPUMI40XX::lcaCache& cache, Logger log) {
    auto ctx = mpi.getContext();

    auto getFetchTask = [](mlir::Value result) {
        auto fetchIt = llvm::find_if(result.getUsers(), [](mlir::Operation* user) {
            return mlir::isa<VPURegMapped::FetchTaskOp>(user);
        });

        auto fetchTask = fetchIt != result.getUsers().end() ? mlir::cast<VPURegMapped::FetchTaskOp>(*fetchIt) : nullptr;
        return fetchTask;
    };

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        bool isFirstEnqueue = true;
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

            auto firstSecondaryIt = vpux::min_element(filteredRange, VPUMI40XX::taskOpComparator);
            auto lastSecondaryIt = vpux::max_element(filteredRange, VPUMI40XX::taskOpComparator);
            auto firstSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**firstSecondaryIt);
            auto lastSecondary = mlir::cast<VPURegMapped::TaskOpInterface>(**lastSecondaryIt);

            auto fetchTask = getFetchTask(taskOp.getResult());
            previousFetchTask = fetchTask ? fetchTask : previousFetchTask;
            /*
                FetchTask
                    |
                    |
                    .   Bar1
                    .   /
                   DMA0_1
                    |   \
                    |   Bar2
                    .
                    .

                Start with the fetchTask, get all the users such that the previous task for user is fetchTask
                For the next iteration this user becomes the task and the loop continues until we find a task which
               is of type ExecutableTaskOpInterface e.g. DMA0_1 and use its updateBarriers to ensure the fetch task
               has been completed
            */
            auto fetchTaskUpdateBarrs = VPUMI40XX::getClosestProductionBarriers(
                    mlir::cast<VPURegMapped::TaskOpInterface>(previousFetchTask.getOperation()));

            auto barrieredOp = VPUMI40XX::getBarrieredOp(taskOp, lastSecondary);
            // Similar approach comparing to DMA L:593, to ensure both wait,update have been reprogrammed before
            // Enqueuing the task have both wait and update barriers in the lca search space
            llvm::SmallVector<mlir::Value> targetBarriers(barrieredOp.updateBarriers().begin(),
                                                          barrieredOp.updateBarriers().end());
            llvm::append_range(targetBarriers, barrieredOp.waitBarriers());

            auto previousUsages = getPreviousUsages(targetBarriers, barriers);

            // If there are multiple barriers updated by fetch task we only need 1 to
            // take into account for LCA algorithm as it is sufficient to identify fetch
            // task completion. Taking multiple barriers only increases complexity for LCA
            if (!fetchTaskUpdateBarrs.empty()) {
                previousUsages.push_back(*fetchTaskUpdateBarrs.begin());
            }

            auto lcas = VPUMI40XX::lca(previousUsages, cache);

            VPUX_THROW_UNLESS(lcas.size(), "Could not find a lowest common ancestor");

            auto enqueueTarget = *VPUMI40XX::getMinBarrier(lcas);

            // We need to check the first task which has a reused wait barrier
            auto previousUsagesWaitBarrier = getPreviousUsages(barrieredOp.waitBarriers(), barriers);
            if (isFirstEnqueue && !previousUsagesWaitBarrier.empty()) {
                enqueueTarget = delayEnqueueOpForFirstTaskInFifo(barrieredOp, enqueueTarget, barriers, cache);
                isFirstEnqueue = false;
            }

            // we can get into a corner case where a subsequent DPU can be enqueued before it's preceding enqueue,
            // based on barrier constraints. we could include the previous barriers enqueue barrier into the
            // LCA, but need to prove that will be satisfactory. For now, assuming that barriers are
            // topologically ordered, we will compare with the previous enqueOp's triggering barriers list idx

            // not needed anymore as long as we are sure to topologically sort enqueue tasks themselves
            // since now Runtime guarantees ordered submission of enqueues
            if (localPreviousEnqu) {
                auto previousEnquBarrier = localPreviousEnqu.getBarrier();

                enqueueTarget = std::max(previousEnquBarrier, enqueueTarget, [](mlir::Value lhs, mlir::Value rhs) {
                    return lhs.getType().cast<VPURegMapped::IndexType>().getValue() <
                           rhs.getType().cast<VPURegMapped::IndexType>().getValue();
                });
            }

            VPUX_THROW_TYPED_UNLESS(WlmRollbackException,
                                    verifyEnqueueBarrierHasNoTopoDepOnBarrs(enqueueTarget, targetBarriers, log),
                                    "Invalid enqueue barrier found for task '{0}'", taskOp);

            // if the previous enqueue's barrier is the same as the target barrier, we can just add this variant
            // range to the previous enqueue. This is made with the assumption that we topologically iterate over
            // the variants list by their listOrder
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

            taskOp = VPUMI40XX::getNextOp(taskOp);
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

    // We often call LCA for same pair of barriers in in that case we having cache is beneficial
    VPUMI40XX::lcaCache cache;

    addEnqus(mpi, VPURegMapped::TaskType::DPUInvariant, VPURegMapped::TaskType::DPUVariant, tilesCount, barriers,
             firstEnqu, globalPreviousEnqu, builder, globalEnquCounter, cache, _log);
    addEnqus(mpi, VPURegMapped::TaskType::ActKernelRange, VPURegMapped::TaskType::ActKernelInvocation, tilesCount,
             barriers, firstEnqu, globalPreviousEnqu, builder, globalEnquCounter, cache, _log);

    std::array<VPUMI40XX::ExecutableTaskOpInterface, DMA_OUTSTANDING_TRANSACTIONS> outstandingEnqueuedDmas;

    // Store information on last DMAs on each FIFO which does not have a corresponding
    // enqueue op added in this pass.
    SmallVector<SmallVector<mlir::Operation*>> lastDmaWithNoEnqueue(tilesCount, SmallVector<mlir::Operation*>(2));

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
                    // Include both updateBarriers and waitBarriers in the search space for lca
                    // Only having one of them doesn't ensure right place for enqueue leading to enqueue early and
                    // causing barrier underflow
                    /*
                        B2Prev (428:14)
                            |
                            .
                            .
                        B1Prev (429:5) <- Enq DMA0_1 (Enqueue happens after reprogramming of B1Prev by runtime)
                            |
                            |
                            .
                            .

                            B1 (460:5)
                            |
                          DMA_01
                            |
                            B2 (461:14)

                       In the case above if we only choose updateBarrier's (B2) previous usage (B2Prev) for lca and
                       B2Prev is chosen as Enqueue for DMA_01. Then the problem is that DMA_01 is in the FIFO and
                       starts listening to production of Phy5 which is also mapped to B1Prev apart from B1 and hence
                       decrements the counter after execution for Phy5 causing underflow
                    */
                    llvm::SmallVector<mlir::Value> targetBarrs(executableTaskOp.waitBarriers().begin(),
                                                               executableTaskOp.waitBarriers().end());
                    llvm::append_range(targetBarrs, executableTaskOp.updateBarriers());
                    auto previousUsages = getPreviousUsages(targetBarrs, barriers);

                    auto oldestOutstandingDma = outstandingEnqueuedDmas[outstandingEnquOpsCounter];
                    if (oldestOutstandingDma) {
                        auto outstandingBarrierCondition = oldestOutstandingDma.updateBarriers();
                        previousUsages.append(outstandingBarrierCondition.begin(), outstandingBarrierCondition.end());
                    }

                    auto lcas = VPUMI40XX::lca(previousUsages, cache);

                    if (lcas.size()) {
                        auto enqueueTarget = *VPUMI40XX::getMinBarrier(lcas);

                        if (localPreviousEnqu) {
                            auto previousEnquBarrier = localPreviousEnqu.getBarrier();
                            enqueueTarget =
                                    std::max(previousEnquBarrier, enqueueTarget, [](mlir::Value lhs, mlir::Value rhs) {
                                        return lhs.getType().cast<VPURegMapped::IndexType>().getValue() <
                                               rhs.getType().cast<VPURegMapped::IndexType>().getValue();
                                    });
                        }

                        VPUX_THROW_TYPED_UNLESS(
                                WlmRollbackException,
                                verifyEnqueueBarrierHasNoTopoDepOnBarrs(enqueueTarget, targetBarrs, _log),
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
                        } else {
                            lastDmaWithNoEnqueue[tileIdx][listIdx] = dmaTask.getDefiningOp();
                        }
                    }
                } else if (localPreviousEnqu) {
                    localPreviousEnqu.getEndMutable().assign(dmaTask);
                } else {
                    lastDmaWithNoEnqueue[tileIdx][listIdx] = dmaTask.getDefiningOp();
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
    if (!enquOps.empty()) {
        mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
    }
    mpi.setWorkItemCount(enquOps.size());
    VPUMI40XX::reindexEnqueueOps(enquOps);

    // Verify enqueue ops can be enqueued at given barriers
    verifyEnqueueBarrierIsNotBlockedByFutureTask(mpi, enquOps, barriers, lastDmaWithNoEnqueue, tilesCount, _log);
}

}  // namespace

//
// createAddEnqueueOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddEnqueueOpsPass(Logger log) {
    return std::make_unique<AddEnqueueOpsPass>(log);
}
