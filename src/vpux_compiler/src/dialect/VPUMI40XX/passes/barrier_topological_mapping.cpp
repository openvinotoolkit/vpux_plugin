//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

using namespace vpux;

namespace {

SmallVector<VPURegMapped::TaskOpInterface> getBarrieredTaskFronts(VPUMI40XX::MappedInferenceOp mpi) {
    auto cond = [](VPURegMapped::TaskOpInterface op) {
        return (op.getTaskType() == VPURegMapped::TaskType::DPUInvariant) ||
               (op.getTaskType() == VPURegMapped::TaskType::DMA) ||
               (op.getTaskType() == VPURegMapped::TaskType::ActKernelInvocation);
    };

    llvm::SmallVector<VPURegMapped::TaskOpInterface> fronts;
    for (auto operand : mpi.getOperands()) {
        auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(operand.getDefiningOp());
        if (taskOp && cond(taskOp)) {
            fronts.push_back(taskOp);
        }
    }

    return fronts;
}

size_t reindexBarrList(mlir::func::FuncOp netFunc) {
    auto ctx = netFunc.getContext();
    auto currIdx = 0;
    for (auto barr : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        barr.getResult().setType(VPURegMapped::IndexType::get(ctx, currIdx));
        currIdx++;
    }

    return currIdx;
}

class BarrierTopologicalMappingPass : public VPUMI40XX::BarrierTopologicalMappingBase<BarrierTopologicalMappingPass> {
public:
    explicit BarrierTopologicalMappingPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    /**
     * Existing implementation require dense graph of dependencies between barriers.
     * Transitive reduction in that case require O(V^3) operations, which is not feasible for large models.
     * For the number of operations above the threshold, compilation times can take unreasoble time and
     * we must switch to non-WLM flow. In case if we skip transitive reduction, we will get stuck in add_enqueues pass.
     * E125659
     */
};

//
// The goal of this pass is to provide a topological mapping for barriers in the IR to match the order of their
// dependencies. As a result, we will get two things:
//    * Topological order of barriers
//    * The list of dependencies in each ConfigureBarrierOp that matches this order

void BarrierTopologicalMappingPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);

    auto barriers = vpux::to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());

    _log.trace("barriers count: {0}, mpi barriers count: {1}", barriers.size(), mpi.getBarrierCount());
    VPUX_THROW_WHEN(barriers.size() != mpi.getBarrierCount(), "Number of barriers is not equal to barrier count");

    // Final barrier is the actual last one inside our list (by definition of final barrier)
    // Final barrier the only barrier that not other barrier has a dependency upon
    auto finalBarrierOp = barriers.back();
    VPUX_THROW_WHEN(!finalBarrierOp.getIsFinalBarrier(), "Last barrier in list not a final barrier");

    auto currentTaskPerFifo = getBarrieredTaskFronts(mpi);

    _log.trace("Find barrier dependencies");

    // Process all FIFOs of tasks using barriers and find bar-to-bar dependencies
    // For Bar0 -> Op0 -> Bar1 task barriers, barrier depepndencies would be
    // Bar0 -> Bar1, so Bar1 depends on Bar0
    for (auto taskOp : currentTaskPerFifo) {
        // Store informaction about wait barriers in case there are tasks with no update barriers
        // and the update barrier is only present on some next task
        // Example:
        // bar0 -> DMA0 -> DMA1 -> bar1
        llvm::SetVector<mlir::Value> waitBarsSet;
        while (taskOp) {
            auto barrieredOp = mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(taskOp.getOperation());

            auto waitBars = barrieredOp.waitBarriers();
            waitBarsSet.insert(waitBars.begin(), waitBars.end());

            auto updateBars = barrieredOp.updateBarriers();
            if (updateBars.empty()) {
                // Update given FIFO to next task
                taskOp = taskOp.getNextTask();
                continue;
            }

            // Create dependencies from wait barriers to update barriers
            for (auto waitBar : waitBarsSet) {
                for (auto updateBar : updateBars) {
                    // Store information about waitBar -> updateBar dependency
                    auto updateBarOp = VPUMI40XX::getBarrierOp(updateBar.getDefiningOp());
                    VPUX_THROW_WHEN(updateBarOp == nullptr, "No a barrier op");
                    auto barDeps = updateBarOp.getDependencies();
                    if (!barDeps.empty() && llvm::is_contained(barDeps, waitBar)) {
                        continue;
                    }

                    // Add new dependency
                    updateBarOp.getDependenciesMutable().append(waitBar);
                }
            }
            waitBarsSet.clear();

            // Update given FIFO to next task
            taskOp = taskOp.getNextTask();
        }
    }

    // Make barrier order in IR to follow barrier consumption order in the schedule
    //
    // Update the order of barrierOp bottom to top
    // starting from final barrier.
    // Use per barrier dependency data to find next barriers
    // to place
    auto insertionPoint = finalBarrierOp.getOperation();

    // For next barriers set from which next barrier to insert is picked
    // use SetVector to guarantee compilations gives same blob as nextBarOpsSet
    // is used to define order of barriers in IR
    llvm::SetVector<VPUMI40XX::ConfigureBarrierOp> nextBarOpsSet;
    llvm::DenseSet<mlir::Value> visited;
    llvm::DenseSet<VPUMI40XX::ConfigureBarrierOp> inserted;

    // start the process from bottom - from final barrier
    for (auto depBar : finalBarrierOp.getDependencies()) {
        auto depBarOp = VPUMI40XX::getBarrierOp(depBar.getDefiningOp());
        nextBarOpsSet.insert(depBarOp);
        visited.insert(depBar);
    }

    inserted.insert(finalBarrierOp);
    visited.insert(finalBarrierOp.getBarrier());

    // For next barriers pick next barrier to insert. Barrier is ready
    // if all of its users - barrier which dependent have already been
    // processed and placed in the IR before
    auto getNextReadyBarOp = [&]() {
        VPUMI40XX::ConfigureBarrierOp readyBarOp = nullptr;

        for (auto barOp : nextBarOpsSet) {
            auto barOpReady = true;
            for (auto user : barOp.getBarrier().getUsers()) {
                auto userBarOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(user);
                if (!userBarOp) {
                    continue;
                }
                if (!inserted.contains(userBarOp)) {
                    barOpReady = false;
                    break;
                }
            }

            if (barOpReady) {
                readyBarOp = barOp;
                nextBarOpsSet.remove(barOp);
                break;
            }
        }

        return readyBarOp;
    };

    _log.trace("Update barrier order in IR");

    // Until there are barriers to process perform
    // barrier insertion in the IR using BFS approach from bottom to top.
    // Get next ready barrier and place it above last inserted barrier. In the first iteration that
    // would be final barrier.
    // For each barrier get barriers that this barrier depends on. They will be
    // processed in next iteration.
    while (!nextBarOpsSet.empty()) {
        auto barOp = getNextReadyBarOp();
        VPUX_THROW_WHEN(barOp == nullptr, "No ready barrier found");

        barOp->moveBefore(insertionPoint);
        insertionPoint = barOp.getOperation();
        inserted.insert(barOp);

        auto depBars = barOp.getDependencies();
        for (auto depBar : depBars) {
            if (visited.contains(depBar)) {
                continue;
            }

            auto depBarOp = VPUMI40XX::getBarrierOp(depBar.getDefiningOp());
            nextBarOpsSet.insert(depBarOp);
            visited.insert(depBar);
        }
    }

    auto newCount = reindexBarrList(netFunc);
    _log.trace("Reindex barrier list done");

    // Set the initial barrier again as we change the barrier order
    if (auto barrierTaskOps = to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());
        !barrierTaskOps.empty()) {
        auto barrierTasks = mpi.getBarrierTasksMutable();
        barrierTasks.clear();
        barrierTasks.append(barrierTaskOps.front().getResult());
    }

    mpi.setBarrierCount(newCount);
}  // namespace

}  // namespace

//
// createBarrierTopologicalMappingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createBarrierTopologicalMappingPass(Logger log) {
    return std::make_unique<BarrierTopologicalMappingPass>(log);
}
