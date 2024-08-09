//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux {
namespace VPURT {

size_t getMinEntry(const BarrierInfo::TaskSet& entries);
size_t getMaxEntry(const BarrierInfo::TaskSet& entries);
std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> getTaskOpQueues(
        mlir::func::FuncOp funcOp, BarrierInfo& barrierInfo,
        std::optional<VPU::ExecutorKind> targetExecutorKind = std::nullopt);
void postProcessBarrierOps(mlir::func::FuncOp func);
bool verifyBarrierSlots(mlir::func::FuncOp func, Logger log);
bool verifyOneWaitBarrierPerTask(mlir::func::FuncOp funcOp, Logger log);
void orderExecutionTasksAndBarriers(mlir::func::FuncOp funcOp, BarrierInfo& barrierInfo,
                                    bool orderByConsumption = false);

}  // namespace VPURT
}  // namespace vpux
