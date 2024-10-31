//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"

namespace vpux {
namespace VPUMI40XX {
using lcaCache = llvm::DenseMap<std::pair<uint32_t, uint32_t>, llvm::SmallVector<mlir::Value>>;

//
// AddEnqueue Utils
//

bool contains(const llvm::SmallVector<mlir::Value>& vec, const mlir::Value& element);

VPUMI40XX::ConfigureBarrierOp getBarrierOp(mlir::Operation* op);

size_t getBarrierIndex(mlir::Operation* op);

bool taskOpComparator(mlir::Operation* lhs, mlir::Operation* rhs);

// Function to get the maximum barrier based on their type values(virtual id)
mlir::Value* getMaxBarrier(SmallVector<mlir::Value>& barriers);

// Function to get the minimum barrier based on their type values(virtual id)
mlir::Value* getMinBarrier(SmallVector<mlir::Value>& barriers);

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps);

mlir::ValueRange getClosestProductionBarriers(VPURegMapped::TaskOpInterface taskOp);

void dfs(mlir::Value val, llvm::SetVector<mlir::Value>& visited);

llvm::SmallVector<mlir::Value> lca(mlir::Value lhs, mlir::Value rhs, lcaCache& cache);
llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& lhs, mlir::Value rhs, lcaCache& cache);
llvm::SmallVector<mlir::Value> lca(llvm::SmallVector<mlir::Value>& barrierVals, lcaCache& cache);

VPURegMapped::TaskOpInterface getNextOp(VPURegMapped::TaskOpInterface op);

// TODO: (E#115494) consider explicitly materializing the "previous ID" inside the IR
llvm::SmallVector<mlir::Value> getPreviousUsages(mlir::ValueRange barrs,
                                                 llvm::SmallVector<VPUMI40XX::ConfigureBarrierOp> allBarrs);

// TODO: need to figure out a clean way to get barriers purely from taskOpInterface
VPUMI40XX::ExecutableTaskOpInterface getBarrieredOp(VPURegMapped::TaskOpInterface primary,
                                                    VPURegMapped::TaskOpInterface secondary);

}  // namespace VPUMI40XX
}  // namespace vpux
