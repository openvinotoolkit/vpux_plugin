
//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM_PASSES
#define VPUX_COMPILER_DIALECT_VPUASM_PASSES

include "mlir/Pass/PassBase.td"

//
// HoistInputOutputs
//

def HoistInputOutputs : PassBase<"hoist-input-outputs", "vpux::ModulePass"> {
    let summary = "Hoist Input Outputs to Module declaration";

    let description = [{
        Initially our Input and Outputs are providied as func arguments, that positionally represent the original network
        I/O's as defined by the ABI with the Plugin. This has the consequence that if we want to
        have OPS that reference sub-parts of these I/O's we have to add extra subview (or similar) ops that represent
        the transfer of a particular view of the I/O.
        As these "View" operations are not something we will want to execute, it is desired to remove them at a certain
        stage of compilation (where we would tend to keep only the ops that are of interenst to the RUNTIME)

        For this to happen, we will have to introduce *alias* buffer declarations that declare the exact desired buffer
        that we wish to use.

        This leads to a state of the IR where both the alias buffer declarations and function arguments are permitted
        to represent an I/O.

        This pass will effectively remove any funcArgument, and will hoist up their declarations to a global ModuleOp
        (similarliy with IE.CNNNetworkOp with the critical difference that it will hold buffer/memref semantics)

        Any reference within the function body to the funcArguments will be replaced by the aliased declarations
    }];

    let constructor = "vpux::VPUASM::createHoistInputOutputsPass()";

    let dependentDialects = [
        "vpux::VPUASM::VPUASMDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::IE::IEDialect"
    ];
}

//
// AddProfilingSection
//

def AddProfilingSection : PassBase<"add-profiling-section", "vpux::ModulePass"> {
    let summary = "Adds the profiling metadata section OP";

    let constructor = "vpux::VPUASM::createAddProfilingSectionPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect",
        "vpux::VPUASM::VPUASMDialect"
    ];
}

//
// BarriersToManagedBarriers
//

def BarriersToManagedBarriers : PassBase<"barriers-to-managed-barriers", "vpux::FunctionPass"> {
    let summary = "Rewrites ConfigureBarrierTasks to ManagedBarrierTasks";

    let constructor = "vpux::VPUASM::createBarriersToManagedBarriersPass()";

    let dependentDialects = [
        "vpux::VPUASM::VPUASMDialect"
    ];
}

#endif // VPUX_COMPILER_DIALECT_VPUASM_PASSES
