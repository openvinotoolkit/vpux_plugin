//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

//
// FeasibleMemorySchedulerSpilling class is a support class for FeasibleMemoryScheduler
// to handle spilling and insert correct spill-write and spill-read CopyOps within
// async dialect and perform required connections between new ops to achieve correct
// execution order for models that need spilling
//
class FeasibleMemorySchedulerSpilling final {
public:
    explicit FeasibleMemorySchedulerSpilling(mlir::FuncOp netFunc, mlir::Attribute memSpace,
                                             mlir::Attribute secondLvlMemSpace, AsyncDepsInfo& depsInfo,
                                             AliasesInfo& aliasInfo, Logger log,
                                             LinearScan<mlir::Value, LinearScanHandler>& scan);

    void insertSpillCopyOps(llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps);

private:
    void createSpillWrite(llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps,
                          size_t schedOpIndex);
    void createSpillRead(llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps,
                         size_t schedOpIndex);
    mlir::async::ExecuteOp insertSpillWriteCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                  mlir::async::ExecuteOp insertAfterExecOp, mlir::Value bufferToSpill,
                                                  size_t allocatedAddress);
    mlir::async::ExecuteOp insertSpillReadCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                 mlir::async::ExecuteOp spillWriteExecOp,
                                                 mlir::async::ExecuteOp insertBeforeExecOp, size_t allocatedAddress);
    void updateSpillWriteReadUsers(mlir::async::ExecuteOp opThatWasSpilled, mlir::Value bufferToSpill,
                                   mlir::async::ExecuteOp spillWriteExecOp, mlir::async::ExecuteOp spillReadExecOp);
    mlir::Value getAsyncResultForBuffer(mlir::Value buffer);
    mlir::Value getBufferFromAsyncResult(mlir::Value asyncResult);

private:
    Logger _log;
    // first level mem space
    mlir::Attribute _memSpace;
    // second level mem space which is used for spilling
    mlir::Attribute _secondLvlMemSpace;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // insertion point for allocation related operations
    mlir::Operation* _allocOpInsertionPoint;
    // Vector of pairs of operation ID and inserted spill-write exec-op that doesn't have yet corresponding spill-read
    // op
    llvm::SmallVector<std::pair<mlir::Value, mlir::async::ExecuteOp>> _opIdAndSpillWritePairs;
};

}  // namespace vpux