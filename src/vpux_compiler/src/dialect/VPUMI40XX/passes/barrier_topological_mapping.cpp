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

llvm::SmallVector<VPURegMapped::TaskOpInterface> getBarrieredTaskTails(VPUMI40XX::MappedInferenceOp mpi) {
    auto getTaskTail = [](VPURegMapped::TaskOpInterface op) {
        auto next = op.getNextTask();
        while (next) {
            op = next;
            next = op.getNextTask();
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

size_t reindexBarrList(mlir::func::FuncOp netFunc) {
    auto ctx = netFunc.getContext();
    auto currIdx = 0;
    for (auto barr : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        barr.getResult().setType(VPURegMapped::IndexType::get(ctx, currIdx));
        currIdx++;
    }

    return currIdx;
}

llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> gerBarrierUsers(VPUMI40XX::ConfigureBarrierOp barrier,
                                                                       bool updateRelationship) {
    llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> deps;

    auto contains = [&barrier](mlir::ValueRange range) {
        auto found = llvm::find(range, barrier.getResult());
        return found != range.end();
    };

    for (auto user : barrier.getResult().getUsers()) {
        auto dep = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(user);
        if (dep && mlir::isa<VPURegMapped::TaskOpInterface>(dep.getOperation())) {
            mlir::ValueRange barriers;
            if (updateRelationship) {
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

llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> getBarrierProducers(VPUMI40XX::ConfigureBarrierOp barrier) {
    return gerBarrierUsers(barrier, true);
}

llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize> getBarrierConsumers(VPUMI40XX::ConfigureBarrierOp barrier) {
    return gerBarrierUsers(barrier, false);
}

llvm::DenseSet<size_t> getIndicesOfWaitBarriersForOps(
        const llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>& ops) {
    llvm::DenseSet<size_t> waitBarrierIndicies;
    for (const auto user : ops) {
        auto dep = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(user);
        for (const auto b : dep.waitBarriers()) {
            auto bar = b.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
            auto position = mlir::cast<VPURegMapped::IndexType>(bar.getType()).getValue();
            waitBarrierIndicies.insert(position);
        }
    }

    return waitBarrierIndicies;
}

// Build op-to-op and bar-to-op depencies and dependets maps
// tails - represent the tails of lists with barrierrd tasks ops (DPUInvariant, DMA, ActKernelInvocation)
// for each op we add:
//     waitBarriers and previous task into opDependencies
//     updateBarriers and next task into opDependents
void buildOpDepsAdjacentLists(
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependencies,
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependents,
        llvm::SmallVector<VPURegMapped::TaskOpInterface>& tails) {
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

                for (auto barr : execTaskOp.updateBarriers()) {
                    dependents.insert(barr.getDefiningOp());
                }
            }

            if (traveler.getPreviousTask()) {
                dependencies.insert(traveler.getPreviousTask().getOperation());
            }

            auto nextOp = traveler.getNextTask();
            if (nextOp) {
                dependents.insert(nextOp.getOperation());
            }

            traveler = traveler.getPreviousTask();
        }
    }
}

void topologicalSort(VPUMI40XX::ConfigureBarrierOp lastOp,
                     llvm::DenseMap<VPUMI40XX::ConfigureBarrierOp, int64_t>& inDegrees, uint64_t barrierCount) {
    std::vector<VPUMI40XX::ConfigureBarrierOp> topologicalOrder;
    topologicalOrder.reserve(barrierCount);
    std::queue<VPUMI40XX::ConfigureBarrierOp> toVisit;
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

SmallVector<llvm::DenseSet<size_t>> convertFromOpGraphToIndicesGraph(
        const llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, 16>>& dependencyMap) {
    SmallVector<llvm::DenseSet<size_t>> adjList(dependencyMap.size());
    for (auto& [op, dependencies] : dependencyMap) {
        auto barrier = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(op);
        auto pos = mlir::cast<VPURegMapped::IndexType>(barrier.getType()).getValue();
        for (auto dep : dependencies) {
            auto deps = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(dep);
            auto indexDeps = mlir::cast<VPURegMapped::IndexType>(deps.getType()).getValue();
            adjList[pos].insert(indexDeps);
        }
    }
    return adjList;
}

void dfs(std::vector<bool>& visited, SmallVector<llvm::DenseSet<size_t>>& graph,
         llvm::DenseSet<std::pair<size_t, size_t>>& toRemove, size_t parent, size_t child) {
    if (visited[child]) {
        return;
    }
    for (size_t grandChild : graph[child]) {
        toRemove.insert(std::make_pair(parent, grandChild));
        dfs(visited, graph, toRemove, parent, grandChild);
    }
    visited[child] = true;
}

void transitiveReduction(SmallVector<llvm::DenseSet<size_t>>& graph) {
    llvm::DenseSet<std::pair<size_t, size_t>> toRemove;
    for (size_t parent = 0; parent < graph.size(); ++parent) {
        std::vector<bool> visited(graph.size(), false);
        for (auto& child : graph[parent]) {
            dfs(visited, graph, toRemove, parent, child);
        }
    }

    for (auto& pairIt : toRemove) {
        graph[pairIt.first].erase(pairIt.second);
    }
}

// This function removing non-barrier ops from opDependencies map
// We will iterate over the graph, take non-barrier op and replace it to dependencies and dependats
// The goal of transitive Closure - avoid adding unnecessary edges during propagation, we already have a set of
// dependencies from bar-to-bar graph and must reuse it
void removeNonBarrierTasksFromGraphWithTransitiveClosure(
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependencies,
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependents,
        SmallVector<llvm::DenseSet<size_t>>& transitiveClosure) {
    for (auto& [op, dependents] : llvm::make_early_inc_range(opDependents)) {
        if (mlir::isa<VPUMI40XX::ConfigureBarrierOp>(op)) {
            continue;
        }

        auto& dependencies = opDependencies[op];

        for (auto dependent : dependents) {
            auto& dependantDependencies = opDependencies[dependent];
            for (auto dependency : dependencies) {
                if (mlir::isa<VPUMI40XX::ConfigureBarrierOp>(dependent) &&
                    mlir::isa<VPUMI40XX::ConfigureBarrierOp>(dependency)) {
                    auto dependentBarrier = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(dependent);
                    auto posDependentBarrier =
                            mlir::cast<VPURegMapped::IndexType>(dependentBarrier.getType()).getValue();

                    auto dependencyBarrier = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(dependency);
                    auto posDependencyBarrier =
                            mlir::cast<VPURegMapped::IndexType>(dependencyBarrier.getType()).getValue();

                    if (!transitiveClosure[posDependentBarrier].contains(posDependencyBarrier)) {
                        dependantDependencies.insert(dependency);
                    }
                } else {
                    dependantDependencies.insert(dependency);
                }
            }
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
}

llvm::DenseMap<VPUMI40XX::ConfigureBarrierOp, int64_t> updateBarrierDependencies(
        const llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependencies,
        SmallVector<llvm::DenseSet<size_t>>& graphOfIndicies,
        llvm::SmallVector<vpux::VPUMI40XX::ConfigureBarrierOp>& barriers) {
    auto valueCompare = [](const mlir::Value& lhs, const mlir::Value& rhs) {
        auto lhsVal = mlir::cast<VPURegMapped::IndexType>(lhs.getType());
        auto rhsVal = mlir::cast<VPURegMapped::IndexType>(rhs.getType());

        return lhsVal.getValue() > rhsVal.getValue();
    };

    llvm::DenseMap<VPUMI40XX::ConfigureBarrierOp, int64_t> inDegrees;
    for (auto& [bar, dependencies] : opDependencies) {
        auto barrier = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(bar);
        VPUX_THROW_UNLESS(barrier, "DependencyMap expected to contain only barrierOps {0}", bar);
        auto barrierPos = mlir::cast<VPURegMapped::IndexType>(barrier.getType()).getValue();
        // since later we will have to re-sort the barrier ops IR odrder to keep in line with use-def-chain dominance
        // requirements, we will apply a standard topological sorting on all the barrierOps .
        // the topological sorting will analyze SSA chains, more specifically the "dependencies" variadic operands of
        // each barrier, and will walk the OPs based on this chain.
        // in order to achieve deterministic walking order (and also deterministically sorted ops) we will insert the
        // operands in a deterministic order, sorted by the unique index of the current barriers
        auto indexesOfDependencies = graphOfIndicies[barrierPos];

        std::set<mlir::Value, decltype(valueCompare)> sortedDependencies(valueCompare);

        for (auto depIndex : indexesOfDependencies) {
            auto barrierDep = barriers[depIndex];
            inDegrees[barrierDep]++;
            sortedDependencies.insert(barrierDep.getResult());
        }
        auto dependenciesVector =
                llvm::SmallVector<mlir::Value, defaultPtrSetSize>(sortedDependencies.begin(), sortedDependencies.end());
        barrier.getDependenciesMutable().assign(std::move(dependenciesVector));
    }
    return inDegrees;
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

//
// The goal of this pass is to provide a topological mapping for barriers in the IR to match the order of their
// dependencies. As a result, we will get two things:
//    * Topological order of barriers
//    * The list of dependencies in each ConfigureBarrierOp that matches this order

void BarrierTopologicalMappingPass::safeRunOnFunc() {
    _log.info("BarrierTopologicalMapping pass: start()");
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);

    auto barriers = vpux::to_small_vector(netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>());

    VPUX_THROW_WHEN(barriers.size() != mpi.getBarrierCount(), "Number of barriers is not equal to barrier count");
    VPUX_THROW_TYPED_WHEN(WlmRollbackException, barriers.size() > _barrierThreshold,
                          "Number of barriers {0} is above threshold {1} which suitable for WLM optimization",
                          barriers.size(), _barrierThreshold);

    //
    // We should consider two types of dependencies here:
    //  * From barrier to barrier
    //  * From op to op, which affect final barrier dependencies as well
    // If we restore all these dependencies, we will get a dense graph of operations (V - vertices and E ~ V^2 edges)
    //
    // The current implementation of restoring graph dependencies is split into two parts:
    //  * Step 1 - build a closure map based only on barriers; this should cover many dependency edges.
    //  * Step 2 - build op-to-op dependencies based on barrier dependencies, list adjacency, and the transitive closure
    //  of the barrier graph
    // In the worst case, it will still be O(V^3), but in most cases, we have a sparse graph of barrier dependencies.

    // op1->bar1->op2
    // dependecies for bar1 is op1
    // dependents for bar1 is op2

    // adjacency list for op-to-op and bar-to-op dependencies
    llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>> opDependencies;

    // adjacency list for op-to-op and bar-to-op dependents
    llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>> opDependents;

    // adjacency list for bar-to-bar dependencies (keep only barrier index)
    SmallVector<llvm::DenseSet<size_t>> barrierDependenciesInIndexRepresentation(barriers.size());

    for (auto barrier : barriers) {
        auto opsProducers = getBarrierProducers(barrier);
        opDependencies[barrier.getOperation()] = opsProducers;
        auto barrierIndex = mlir::cast<VPURegMapped::IndexType>(barrier.getType()).getValue();
        barrierDependenciesInIndexRepresentation[barrierIndex] = getIndicesOfWaitBarriersForOps(opsProducers);
        opDependents[barrier.getOperation()] = getBarrierConsumers(barrier);
    }

    // build transitive closure for bar-to-bar dependencies
    auto barDepTransClosure = barrierDependenciesInIndexRepresentation;
    for (auto& curDeps : barDepTransClosure) {
        for (auto curDepInd : llvm::DenseSet<size_t>(curDeps)) {
            const auto& depOfDeps = barDepTransClosure[curDepInd];
            curDeps.insert(depOfDeps.begin(), depOfDeps.end());
        }
    }

    _log.info("BarrierTopologicalMapping pass: transitiveClosure was built()");

    auto tails = getBarrieredTaskTails(mpi);
    buildOpDepsAdjacentLists(opDependencies, opDependents, tails);
    _log.info("BarrierTopologicalMapping pass: buildOpDepsAdjacentLists");

    // Remove non-barrier ops and introduce only new edges
    removeNonBarrierTasksFromGraphWithTransitiveClosure(opDependencies, opDependents, barDepTransClosure);

    // in extendedBarrierDependencies we have bar-to-bar dependencies
    // which take into account not only explicit barrier dependencies
    // but also the dependencies that we receive from the FIFO order of execution of operations
    auto extendedBarToBarDependencies = convertFromOpGraphToIndicesGraph(opDependencies);

    // TODO E127869. Explicitly copy barrier-to-barrier dependencies to the final graph. It's needed because after
    // removeNonBarrierTasksFromGraphWithTransitiveClosure() we have only new dependencies which are coming from
    // FIFO op ordering in list. In that case after this step extendedBarToBarDependencies will contain all neccesary
    // barrier dependencies
    for (size_t i = 0; i < barrierDependenciesInIndexRepresentation.size(); ++i) {
        extendedBarToBarDependencies[i].insert(barrierDependenciesInIndexRepresentation[i].begin(),
                                               barrierDependenciesInIndexRepresentation[i].end());
    }

    transitiveReduction(extendedBarToBarDependencies);
    _log.info("BarrierTopologicalMapping pass: transitiveReduction was built()");

    auto inDegrees = updateBarrierDependencies(opDependencies, extendedBarToBarDependencies, barriers);

    // Final barrier is the actual last one inside our list (by definition of final barrier)
    // Final barrier the only barrier that not other barrier has a dependency upon
    auto finalBarrier = barriers.back();
    VPUX_THROW_WHEN(!finalBarrier.getIsFinalBarrier(), "Last barrier in list not a final barrier");
    VPUX_THROW_WHEN(barriers.size() != opDependencies.size(), "One of the barriers has been missed");
    topologicalSort(finalBarrier, inDegrees, barriers.size());

    auto newCount = reindexBarrList(netFunc);

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

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createBarrierTopologicalMappingPass(const int barrierThreshold,
                                                                                 Logger log) {
    return std::make_unique<BarrierTopologicalMappingPass>(barrierThreshold, log);
}
