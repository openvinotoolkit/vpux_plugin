//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// InsertSyncTasks
//

def InsertSyncTasks : PassBase<"insert-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Insert sync tasks";

    let description = [{
        Insert sync task that mark start and end of each FuncOp.
        Example:

               | op -> .. -> op |
        funcOp{| op -> .. -> op |}
               | op -> .. -> op |

                        =>

               |          /-> op -> .. -> op -\           |
        funcOp{| SyncTask --> op -> .. -> op --> SyncTask |}
               |          \-> op -> .. -> op -/           |

    }];

    let constructor = "vpux::VPURT::arch40xx::createInsertSyncTasksPass()";
}

//
// OptimizeSyncTasks
//

def OptimizeSyncTasks : PassBase<"optimize-sync-tasks", "vpux::FunctionPass"> {
    let summary = "Remove unnecessary sync tasks";

    let description = [{
        Remove unnecessary sync task that were injected previously by InsertSyncTasks pass.
        Example:

                 /-> op -> .. -> op -\                       /-> op -> .. -> op -\
        SyncTask --> op -> .. -> op --> SyncTask -> SyncTask --> op -> .. -> op --> SyncTask
                 \-> op -> .. -> op -/                       \-> op -> .. -> op -/

                        =>

        /-> op -> .. -> op -\           /-> op -> .. -> op -\
        --> op -> .. -> op --> SyncTask --> op -> .. -> op -->
        \-> op -> .. -> op -/           \-> op -> .. -> op -/
    }];

    let constructor = "vpux::VPURT::arch40xx::createOptimizeSyncTasksPass()";
}


#endif
