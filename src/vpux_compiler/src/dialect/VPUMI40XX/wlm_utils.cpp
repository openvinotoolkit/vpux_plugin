//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"

namespace vpux {
namespace VPUMI40XX {

//
// AddEnqueue Utils
//

bool contains(const llvm::SmallVector<mlir::Value>& vec, const mlir::Value& element) {
    return std::find(vec.begin(), vec.end(), element) != vec.end();
};

VPUMI40XX::ConfigureBarrierOp getBarrierOp(mlir::Operation* op) {
    if (op == nullptr) {
        return nullptr;
    }

    auto maybeBarrier = mlir::dyn_cast_or_null<VPUMI40XX::ConfigureBarrierOp>(op);
    return maybeBarrier;
}

size_t getBarrierIndex(mlir::Operation* op) {
    auto barrierOp = getBarrierOp(op);
    VPUX_THROW_WHEN(barrierOp == nullptr, "Expected barrier: got {0}", op);
    return barrierOp.getType().getValue();
};

bool taskOpComparator(mlir::Operation* lhs, mlir::Operation* rhs) {
    auto lhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(lhs);
    auto rhsTask = mlir::cast<VPURegMapped::TaskOpInterface>(rhs);
    return lhsTask.getIndexType().getValue() < rhsTask.getIndexType().getValue();
}

// Function to get the maximum barrier based on their type values(virtual id)
mlir::Value* getMaxBarrier(SmallVector<mlir::Value>& barriers) {
    return std::max_element(barriers.begin(), barriers.end(), [](mlir::Value lhs, mlir::Value rhs) {
        return mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp()).getType().getValue() <
               mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp()).getType().getValue();
    });
}

// Function to get the minimum barrier based on their type values(virtual id)
mlir::Value* getMinBarrier(SmallVector<mlir::Value>& barriers) {
    return std::min_element(barriers.begin(), barriers.end(), [](mlir::Value lhs, mlir::Value rhs) {
        return mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp()).getType().getValue() <
               mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp()).getType().getValue();
    });
}

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

llvm::SmallVector<mlir::Value> lca(mlir::Value lhs, mlir::Value rhs, lcaCache& cache) {
    if (lhs == rhs)
        return {lhs};

    auto lhsBar = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(lhs.getDefiningOp());
    auto rhsBarr = mlir::cast<VPUMI40XX::ConfigureBarrierOp>(rhs.getDefiningOp());
    auto lhsPos = lhsBar.getType().cast<VPURegMapped::IndexType>().getValue();
    auto rhsPos = rhsBarr.getType().cast<VPURegMapped::IndexType>().getValue();

    if (lhsPos > rhsPos) {
        std::swap(lhs, rhs);
    }

    if (cache.contains({lhsPos, rhsPos})) {
        return cache[{lhsPos, rhsPos}];
    }

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
    cache[{lhsPos, rhsPos}] = lcas;
    return lcas;
}

llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& lhs, mlir::Value rhs, lcaCache& cache) {
    llvm::SmallVector<mlir::Value> lcas;

    for (auto val : lhs) {
        lcas.append(lca(val, rhs, cache));
    }

    return lcas;
}

llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& barrierVals, lcaCache& cache) {
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
        return lca(barrierVals[0], barrierVals[1], cache);
    }

    llvm::SmallVector<mlir::Value> lcas = lca(barrierVals[0], barrierVals[1], cache);
    for (size_t i = 2; i < barrierVals.size(); ++i) {
        lcas = lca(lcas, barrierVals[i], cache);
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

// TODO E#132327: ned to figure out a clean way to get barriers purely from taskOpInterface
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

}  // namespace VPUMI40XX
}  // namespace vpux
