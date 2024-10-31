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

void setIndicesOfWaitBarriersForOps(const llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>& ops,
                                    llvm::BitVector& waitBarrierIndicies) {
    for (const auto user : ops) {
        auto dep = mlir::dyn_cast<VPUMI40XX::ExecutableTaskOpInterface>(user);
        for (const auto b : dep.waitBarriers()) {
            auto bar = b.getDefiningOp<VPUMI40XX::ConfigureBarrierOp>();
            auto position = mlir::cast<VPURegMapped::IndexType>(bar.getType()).getValue();
            waitBarrierIndicies.set(position);
        }
    }
}

// Build op-to-op and bar-to-op dependencies and dependents maps
// tails - represent the tails of lists with barrier tasks ops (DPUInvariant, DMA, ActKernelInvocation)
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

SmallVector<llvm::BitVector> convertFromOpGraphToIndicesGraph(
        const llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, 16>>& dependencyMap) {
    llvm::BitVector initDeps(dependencyMap.size(), false);
    SmallVector<llvm::BitVector> adjList(dependencyMap.size(), initDeps);

    for (auto& [op, dependencies] : dependencyMap) {
        auto barrier = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(op);
        auto pos = mlir::cast<VPURegMapped::IndexType>(barrier.getType()).getValue();
        for (auto dep : dependencies) {
            auto deps = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(dep);
            auto indexDeps = mlir::cast<VPURegMapped::IndexType>(deps.getType()).getValue();
            adjList[pos].set(indexDeps);
        }
    }
    return adjList;
}

void optimizeDepsMap(SmallVector<llvm::BitVector>& graph) {
    // A -> B -> C
    //
    // If B depends on A and C depends on [A, B] ==> we can remove A from C deps list,
    // since it will be implicit dependency taken from B.
    //
    // Graph transitive closure has memory demand that scales with square of number of graph nodes,
    // but is substantially faster than recursive algorithms. Currently, the pass operates for graphs with number of
    // barriers below _barrierThreshold hence the amount of required memory is modest. If the optimization is enabled
    // for LLM scale models, the graph optimizations should be done within tasks blocks defined by tasks graph split
    // introduced in SplitControlGraph pass (#E134113)
    //
    auto depsMapClosure = graph;

    // For each graph node create transitive closure with the dependent nodes with indexes smaller than the index of the
    // current node.
    for (size_t idx = 0; idx < depsMapClosure.size(); idx++) {
        for (auto curDepInd : graph[idx].set_bits()) {
            depsMapClosure[idx] |= depsMapClosure[curDepInd];
        }
    }

    // The graph nodes can contain dependencies on other nodes with indexes larger than the current index.
    // For example, for a graph
    //
    // node: dependencies
    // 0: 5
    // 1:
    // 2: 1
    // 3: 0
    // 4: 2
    // 5: 4
    // 6: 1, 3
    //
    // nodes from 1 to 6 depend on other nodes with indexes smaller their own index, but node 0 depends on node  with
    // larger index (5) for which graph closure have not been set at the first graph traversal. For such a graph, node 6
    // does not need to depend on 1 because it already depends on node 3 which depends on 0 which depends on 5 which
    // depends on 4 which depends on 2 which depends on 1

    // Update the transitive closure for node dependencies with indexes larger than the current node.
    for (size_t idx = depsMapClosure.size(); idx-- > 0;) {
        const auto curDeps = depsMapClosure[idx];  // use a copy to iterate over original set of bits
        for (auto curDepInd : curDeps.set_bits()) {
            depsMapClosure[idx] |= depsMapClosure[curDepInd];
        }
    }

    // Remove all unnecessary edges.
    for (size_t idx = graph.size(); idx-- > 0;) {
        auto curDeps = graph[idx];  // use a copy to iterate over original set of bits

        // If node does not have any dependency or it has only one dependency then skip
        if (curDeps.count() <= 1) {
            continue;
        }

        for (auto curDepInd : curDeps.set_bits()) {
            const auto& depOfDeps = depsMapClosure[curDepInd];
            graph[idx].reset(depOfDeps);
        }
    }
}

// This function removing non-barrier ops from opDependencies map
// We will iterate over the graph, take non-barrier op and replace it to dependencies and dependats
// The goal of transitive Closure - avoid adding unnecessary edges during propagation, we already have a set of
// dependencies from bar-to-bar graph and must reuse it
void removeNonBarrierTasksFromGraphWithTransitiveClosure(
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependencies,
        llvm::DenseMap<mlir::Operation*, llvm::SmallPtrSet<mlir::Operation*, defaultPtrSetSize>>& opDependents,
        SmallVector<llvm::BitVector>& transitiveClosure) {
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

                    if (!transitiveClosure[posDependentBarrier].test(posDependencyBarrier)) {
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
        SmallVector<llvm::BitVector>& graphOfIndicies,
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

        for (auto depIndex : indexesOfDependencies.set_bits()) {
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
    llvm::BitVector initDeps(barriers.size(), false);
    SmallVector<llvm::BitVector> barrierDependenciesInIndexRepresentation(barriers.size(), initDeps);

    for (auto barrier : barriers) {
        auto opsProducers = getBarrierProducers(barrier);
        opDependencies[barrier.getOperation()] = opsProducers;
        auto barrierIndex = mlir::cast<VPURegMapped::IndexType>(barrier.getType()).getValue();
        VPUX_THROW_UNLESS(barrierIndex < barriers.size(), "Incorrect barrier index ({0})", barrierIndex);
        setIndicesOfWaitBarriersForOps(opsProducers, barrierDependenciesInIndexRepresentation[barrierIndex]);
        opDependents[barrier.getOperation()] = getBarrierConsumers(barrier);
    }

    // build transitive closure for bar-to-bar dependencies
    auto barDepTransClosure = barrierDependenciesInIndexRepresentation;
    for (size_t idx = 0; idx < barDepTransClosure.size(); idx++) {
        for (auto curDepInd : barrierDependenciesInIndexRepresentation[idx].set_bits()) {
            barDepTransClosure[idx] |= barDepTransClosure[curDepInd];
        }
    }
    _log.trace("Transitive closure for bar-to-bar dependencies was built");

    auto tails = getBarrieredTaskTails(mpi);
    buildOpDepsAdjacentLists(opDependencies, opDependents, tails);
    _log.trace("OpDepsAdjacentLists was built");

    // Remove non-barrier ops and introduce only new edges
    removeNonBarrierTasksFromGraphWithTransitiveClosure(opDependencies, opDependents, barDepTransClosure);
    _log.trace("Removed non-barrier tasks from graph");

    // in extendedBarrierDependencies we have bar-to-bar dependencies
    // which take into account not only explicit barrier dependencies
    // but also the dependencies that we receive from the FIFO order of execution of operations
    auto extendedBarToBarDependencies = convertFromOpGraphToIndicesGraph(opDependencies);
    _log.trace("Converted from Op graph to indices graph");

    // TODO E127869. Explicitly copy barrier-to-barrier dependencies to the final graph. It's needed because after
    // removeNonBarrierTasksFromGraphWithTransitiveClosure() we have only new dependencies which are coming from
    // FIFO op ordering in list. In that case after this step extendedBarToBarDependencies will contain all necessary
    // barrier dependencies
    for (size_t i = 0; i < barrierDependenciesInIndexRepresentation.size(); ++i) {
        extendedBarToBarDependencies[i] |= barrierDependenciesInIndexRepresentation[i];
    }
    _log.trace("Extended barrier-to-barrier dependencies created");

    optimizeDepsMap(extendedBarToBarDependencies);
    _log.trace("Optimized barrier map");

    auto inDegrees = updateBarrierDependencies(opDependencies, extendedBarToBarDependencies, barriers);
    _log.trace("Updated barrier dependencies");

    // Final barrier is the actual last one inside our list (by definition of final barrier)
    // Final barrier the only barrier that not other barrier has a dependency upon
    auto finalBarrier = barriers.back();
    VPUX_THROW_WHEN(!finalBarrier.getIsFinalBarrier(), "Last barrier in list not a final barrier");
    VPUX_THROW_WHEN(barriers.size() != opDependencies.size(), "One of the barriers has been missed");
    topologicalSort(finalBarrier, inDegrees, barriers.size());
    _log.trace("Topological sort done");

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
