//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// AddUpdateBarrierForSwKernels
//

def AddUpdateBarrierForSwKernels : PassBase<"add-update-barrier-for-sw-kernels", "vpux::FunctionPass"> {
    let summary = "Adds a dummy update barrier for the SwKernels that don't have any";

    let description = [{
        Adds a dummy update barrier for the SW Kernels that don't have any. The update barrier is the
        only mechanism through which the VPU runtime can know if a SW Kernel has completed the execution or not.
    }];

    let constructor = "vpux::VPURT::arch37xx::createAddUpdateBarrierForSwKernelsPass()";
}

//
// AddFinalBarrier
//

def AddFinalBarrier : PassBase<"add-final-barrier", "vpux::FunctionPass"> {
    let summary = "Adds a dummy barrier to state that the task execution flow has been finished";

    let description = [{
        Adds a dummy update barrier state that the task execution flow has been finished.
        When the barrier's producer count is changed to zero, it means all tasks are finished.
    }];

    let constructor = "vpux::VPURT::arch37xx::createAddFinalBarrierPass()";
}

#endif
