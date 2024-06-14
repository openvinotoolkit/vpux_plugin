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
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

using namespace vpux;

namespace {

constexpr size_t defaultPtrSetSize = 16;

void depthLookup(
        llvm::DenseSet<mlir::Operation*>& visited,
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& dependencyMap,
        llvm::DenseSet<std::pair<mlir::Operation*, mlir::Operation*>>& toRemove, mlir::Operation* parent,
        mlir::Operation* child) {
    if (visited.contains(child)) {
        return;
    }

    auto& grandChildred = dependencyMap[child];
    for (auto grandChild : grandChildred) {
        toRemove.insert(std::make_pair(parent, grandChild));
        depthLookup(visited, dependencyMap, toRemove, parent, grandChild);
    }
    visited.insert(child);
}

// hugely suboptmimal but good for a start
void transitiveReduction(
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& dependencyMap) {
    llvm::DenseSet<std::pair<mlir::Operation*, mlir::Operation*>> toRemove;

    for (auto& [op, dependencies] : dependencyMap) {
        llvm::DenseSet<mlir::Operation*> visited;
        for (auto dependency : dependencies) {
            depthLookup(visited, dependencyMap, toRemove, op, dependency);
        }
    }

    for (auto& pairIt : toRemove) {
        dependencyMap[pairIt.first].erase(pairIt.second);
    }
}

void topologicalSort(
        VPUMI40XX::ConfigureBarrierOp lastOp,
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& barDependencies) {
    llvm::SmallVector<VPUMI40XX::ConfigureBarrierOp> topologicalOrder;
    std::queue<VPUMI40XX::ConfigureBarrierOp> toVisit;

    llvm::DenseMap<VPUMI40XX::ConfigureBarrierOp, int64_t> inDegrees;
    for (auto& [bar, dependencies] : barDependencies) {
        for (auto dependency : dependencies) {
            auto dependencyBarrOp = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(dependency);
            inDegrees[dependencyBarrOp]++;
        }
    }

    toVisit.push(lastOp);
    while (toVisit.size() > 0) {
        auto current = toVisit.front();
        toVisit.pop();

        topologicalOrder.push_back(current);

        for (auto dep : current.getDependencies()) {
            auto depBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(dep.getDefiningOp());
            auto& inDegree = inDegrees[depBarr];
            inDegree--;

            if (inDegree == 0) {
                toVisit.push(depBarr);
            }
        }
    }

    auto last = topologicalOrder.back();
    for (auto barrierIt = topologicalOrder.begin(); (*barrierIt) != last; barrierIt++) {
        barrierIt->getOperation()->moveAfter(last.getOperation());
    }
}

// barriers are not list-chained(yet) TODO:: reindexing needs to have a unified approach
// for now we assume they are all in one list...
size_t reindexBarrList(mlir::func::FuncOp netFunc) {
    auto ctx = netFunc.getContext();
    auto currIdx = 0;
    for (auto barr : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        barr.getResult().setType(VPURegMapped::IndexType::get(ctx, currIdx));
        currIdx++;
    }

    return currIdx;
}

// TODO: E109317
VPURegMapped::TaskOpInterface getNext(VPURegMapped::TaskOpInterface taskOp) {
    auto users = taskOp.getResult().getUsers();
    auto nexOpIt = llvm::find_if(users, [&taskOp](mlir::Operation* user) {
        auto next = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(user);
        return next && (next.getPreviousTask() == taskOp);
    });

    taskOp = nexOpIt != users.end() ? mlir::cast<VPURegMapped::TaskOpInterface>(*nexOpIt) : nullptr;
    return taskOp;
}

llvm::SmallVector<VPURegMapped::TaskOpInterface> getBarrieredTaskTails(VPUMI40XX::MappedInferenceOp mpi) {
    auto getTaskTail = [](VPURegMapped::TaskOpInterface op) {
        auto next = getNext(op);
        while (next) {
            op = next;
            next = getNext(op);
        }
        return op;
    };

    auto cond = [](VPURegMapped::TaskOpInterface op) {
        return (op.getTaskType() == VPURegMapped::TaskType::DPUInvariant) ||
               (op.getTaskType() == VPURegMapped::TaskType::DMA) ||
               (op.getTaskType() == VPURegMapped::TaskType::ActKernelInvocation);
    };

    llvm::SmallVector<VPURegMapped::TaskOpInterface> tails;
    for (auto operand : mpi.getOperands()) {
        auto taskOp = mlir::dyn_cast<VPURegMapped::TaskOpInterface>(operand.getDefiningOp());
        if (taskOp && cond(taskOp)) {
            tails.push_back(getTaskTail(taskOp));
        }
    }

    return tails;
}

class BarrierTopologicalMappingPass : public VPUMI40XX::BarrierTopologicalMappingBase<BarrierTopologicalMappingPass> {
public:
    explicit BarrierTopologicalMappingPass(const int barrierThreshold, Logger log)
            : _barrierThreshold(static_cast<size_t>(barrierThreshold)) {
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
    size_t _barrierThreshold;
};

llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> collectBarriers(VPUMI40XX::ConfigureBarrierOp barrier,
                                                                       bool collectUpdateBarriers) {
    llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> deps;

    auto contains = [&barrier](mlir::ValueRange range) {
        auto found = llvm::find(range, barrier.getResult());
        return found != range.end();
    };

    for (auto user : barrier.getResult().getUsers()) {
        auto dep = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(user);
        if (dep && mlir::isa<VPURegMapped::TaskOpInterface>(dep.getOperation())) {
            mlir::ValueRange barriers;
            if (collectUpdateBarriers) {
                barriers = dep.updateBarriers();
            } else {
                barriers = dep.waitBarriers();
            }

            if (contains(barriers)) {
                deps.insert(user);
            }
        }
    }

    return deps;
}

llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> barrierDependencies(VPUMI40XX::ConfigureBarrierOp barrier) {
    return collectBarriers(barrier, true);
}

llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> barrierDependents(VPUMI40XX::ConfigureBarrierOp barrier) {
    return collectBarriers(barrier, false);
}

void BarrierTopologicalMappingPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto tails = getBarrieredTaskTails(mpi);

    std::vector<llvm::SmallVector<int64_t>> dependencies(mpi.getBarrierCount());
    auto barriers = vpux::to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());

    VPUX_THROW_WHEN(barriers.size() != mpi.getBarrierCount(), "Number of barriers is not equal to barrier count");
    VPUX_THROW_WHEN(barriers.size() > _barrierThreshold,
                    "Number of barriers {0} is above threshold {1} which suitable for WLM optimization",
                    barriers.size(), _barrierThreshold);
    // construct a dense op-to-op dependency based on barrier deps and list adjacency
    llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>> opDependencies;
    llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>> opDependents;

    for (auto tail : tails) {
        auto traveler = tail;
        while (traveler) {
            // TODO: E109083 should be regMapped interface
            auto& dependencies = opDependencies[traveler.getOperation()];
            auto& dependents = opDependents[traveler.getOperation()];

            auto execTaskOp = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(traveler.getOperation());

            if (execTaskOp) {
                for (auto barr : execTaskOp.waitBarriers()) {
                    dependencies.insert(barr.getDefiningOp());
                }
            }

            if (traveler.getPreviousTask()) {
                dependencies.insert(traveler.getPreviousTask().getOperation());
            }

            if (execTaskOp) {
                for (auto barr : execTaskOp.updateBarriers()) {
                    dependents.insert(barr.getDefiningOp());
                }
            }

            auto nextOp = getNext(traveler);
            if (nextOp) {
                dependents.insert(nextOp.getOperation());
            }

            traveler = traveler.getPreviousTask();
        }
    }

    // also add the barriers
    for (auto barrier : barriers) {
        opDependencies[barrier.getOperation()] = barrierDependencies(barrier);
        opDependents[barrier.getOperation()] = barrierDependents(barrier);
    }

    // now remove all deps that are not a barrier
    for (auto& mapIt : llvm::make_early_inc_range(opDependents)) {
        auto op = mapIt.first;
        if (mlir::isa<VPUMI40XX::ConfigureBarrierOp>(op)) {
            continue;
        }

        auto& dependencies = opDependencies[op];
        auto& dependents = opDependents[op];

        for (auto dependent : dependents) {
            auto& dependantDependencies = opDependencies[dependent];
            dependantDependencies.insert(dependencies.begin(), dependencies.end());
            dependantDependencies.erase(op);
        }

        for (auto dependency : dependencies) {
            auto& dependencyDependants = opDependents[dependency];
            dependencyDependants.insert(dependents.begin(), dependents.end());
            dependencyDependants.erase(op);
        }

        // erasing from dependencies only since that's what will iterate upon next. Don't care about dependents, and
        // easier to iterate;
        auto erased = opDependencies.erase(op);
        VPUX_THROW_UNLESS(erased, "op in DependentsMap not present in dependencies map {0}", op);
    }

    transitiveReduction(opDependencies);

    auto valueCompare = [](const mlir::Value& lhs, const mlir::Value& rhs) {
        auto lhsVal = lhs.getType().cast<VPURegMapped::IndexType>();
        auto rhsVal = rhs.getType().cast<VPURegMapped::IndexType>();

        return lhsVal.getValue() > rhsVal.getValue();
    };

    for (auto mapIt : opDependencies) {
        auto barrier = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(mapIt.first);
        VPUX_THROW_UNLESS(barrier, "DependencyMap expected to contain only barrierOps {0}", mapIt.first);

        auto& dependencies = mapIt.second;

        // since later we will have to re-sort the barrier ops IR odrder to keep in line with use-def-chain dominance
        // requirements, we will apply a standard topological sorting on all the barrierOps .
        // the topological sorting will analyze SSA chains, more specifically the "dependencies" variadic operands of
        // each barrier, and will walk the OPs based on this chain.
        // in order to achieve deterministic walking order (and also deterministically sorted ops) we will insert the
        // operands in a deterministic order, sorted by the unique index of the current barriers

        std::set<mlir::Value, decltype(valueCompare)> sortedDependencies(valueCompare);

        for (auto dep : dependencies) {
            auto barrierDep = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(dep);
            VPUX_THROW_UNLESS(barrierDep, "DependencyMap expected to contain only barrierOps {0} -> {1}", mapIt.first,
                              dep);

            sortedDependencies.insert(barrierDep.getResult());
        }
        auto dependenciesVector =
                llvm::SmallVector<mlir::Value, defaultPtrSetSize>(sortedDependencies.begin(), sortedDependencies.end());
        barrier.getDependenciesMutable().assign(std::move(dependenciesVector));
    }

    // Final barrier is the actual last one inside our list (by definition of final barrier)
    // Final barrier the only barrier that not other barrier has a dependency upon
    auto finalBarrier = barriers.back();
    VPUX_THROW_WHEN(!finalBarrier.getIsFinalBarrier(), "Last barrier in list not a final barrier");
    VPUX_THROW_WHEN(barriers.size() != opDependencies.size(), "One of the barriers has been missed");
    topologicalSort(finalBarrier, opDependencies);

    // reindex new order
    auto newCount = reindexBarrList(netFunc);
    mpi.setBarrierCount(newCount);
}  // namespace

}  // namespace

//
// createBarrierTopologicalMappingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createBarrierTopologicalMappingPass(const int barrierThreshold,
                                                                                 Logger log) {
    return std::make_unique<BarrierTopologicalMappingPass>(barrierThreshold, log);
}
