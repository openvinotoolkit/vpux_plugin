//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect_interfaces.hpp"

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

using namespace vpux;

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const {
    return true;
}

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const {
    return true;
}

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const {
    return true;
}

void VPUIP::FuncInlinerInterface::handleTerminator(mlir::Operation*, mlir::ValueRange) const {
}

void VPUIP::FuncInlinerInterface::processInlinedCallBlocks(
        mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    auto parentOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(parentOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    DenseMap<VPURT::TaskQueueType, std::pair<VPURT::TaskOp, VPURT::TaskOp>> taskQueuesFirstAndLastOpMap;

    for (mlir::Block& block : inlinedBlocks) {
        for (auto& op : block.getOperations()) {
            if (mlir::isa<mlir::func::ReturnOp, VPURT::DeclareBufferOp, VPURT::DeclareVirtualBarrierOp,
                          Const::DeclareOp>(op)) {
                continue;
            }

            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
            VPUX_THROW_WHEN(taskOp == nullptr, "Unexpected operation type: {0}", op.getName());

            const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);

            if (taskQueuesFirstAndLastOpMap.find(taskQueueType) == taskQueuesFirstAndLastOpMap.end()) {
                // First occurrence of task on this queue
                taskQueuesFirstAndLastOpMap[taskQueueType] = std::make_pair(taskOp, taskOp);
            } else {
                // In case new task spotted, update last task info
                taskQueuesFirstAndLastOpMap[taskQueueType].second = taskOp;
            }
        }
    }

    // Identify first and last task on each execution queue.
    // For first tasks if they do no wait on any barrier connect them with start barrier
    // For end tasks if they do not update any barrier connect then to end barrier
    for (auto& taskQueuesFirstAndLastOp : taskQueuesFirstAndLastOpMap) {
        auto queueFirstOp = taskQueuesFirstAndLastOp.second.first;
        auto queueLastOp = taskQueuesFirstAndLastOp.second.second;
        if (queueFirstOp.getWaitBarriers().empty() && !parentOp.getWaitBarriers().empty()) {
            // Empty "waits" barriers means
            // this operation is one of the first operations from the callable region
            // Add "waits" barriers(if exist) from the parent VPURT::TaskOp
            // to wait operators from the previous callable region
            queueFirstOp.getWaitBarriersMutable().append(parentOp.getWaitBarriers());
        }

        if (queueLastOp.getUpdateBarriers().empty() && !parentOp.getUpdateBarriers().empty()) {
            // Empty "update" barriers means
            // this operation is one of the last operations from the callable region
            // Add "update" barriers(if exist) from the parent VPURT::TaskOp
            // to notify operators from the next callable region
            queueLastOp.getUpdateBarriersMutable().append(parentOp.getUpdateBarriers());
        }
    }
}

std::tuple<mlir::Block*, mlir::Block::iterator> VPUIP::FuncInlinerInterface::getInlineBlockAndPoint(
        mlir::Operation* call) const {
    auto taskOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    return std::make_tuple(taskOp->getBlock(), std::next(taskOp->getIterator()));
}

void VPUIP::FuncInlinerInterface::eraseCall(mlir::Operation* call) const {
    auto taskOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    taskOp->erase();
}
