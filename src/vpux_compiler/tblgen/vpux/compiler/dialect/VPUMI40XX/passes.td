//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUMI40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// SetupProfilingVPUMI40XX
//

def SetupProfilingVPUMI40XX : PassBase<"setup-profiling-VPUMI40XX", "vpux::ModulePass"> {
    let summary = "Apply HW profiling settings in MI";

    let description = "This pass configures DmaHwpBase and HwpWorkpointCfg.";

    let constructor = "vpux::VPUMI40XX::createSetupProfilingVPUMI40XXPass()";

    let options = [
        Option<
            "enableDMAProfiling", "dma-profiling",
            "std::string", [{"false"}],
            "Enable DMA task profiling (true|static|false)"
        >
    ];

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

//
// BarrierCompuation
//

def BarrierComputation : PassBase<"barrier-computation-VPUMI40XX", "vpux::FunctionPass"> {
    let summary = "Compute and setup scheduling related fields";

    let description = [{
        This pass computes and sets up the (barrier) scheduling related fields - next_same_id, start_after, clean_after.
        This logic was clasically part of the parser, but in ELF pipeline, it is included in the lowering.
    }];

    let constructor = "vpux::VPUMI40XX::createBarrierComputationPass()";

}


def ReorderMPIOps : PassBase<"reorder-mapped-inference-ops", "vpux::FunctionPass"> {
    let summary = [{Reorder Mapped Inference dialect ops ot a *canonical* order.}];
    let description = [{
        The mapped inference representation does not follow the *classical* rules of operation order inside the IR.
        In higher level dialects, OP order in the IR mostly followed the topological order of the execution, with
        Async ops and VPURT.TaskOps generally decorating extra information on the parallelism.

        In MappedInference representation execution flow is expressed differently, as it explicitly defines the list
        semantics of how the operations follow each other in execution flow, while parralelism is expressed at "list level".

        This pass will just re-order the OPS inside the IR to folow this semantics. This reorder serves EXCLUSIVELY
        for HUMAN READABILITY of the IR but serves NO FUNCTIONAL purpose, and does not introduce positional semantics to the dialect.
    }];

    let constructor = "vpux::VPUMI40XX::reorderMappedInferenceOpsPass()";
}

def ResolveTaskLocation : PassBase<"resolve-task-location", "vpux::VPURegMapped::ResolveTaskLocationPass"> {
    let summary = [{ 40xx-specific version of resolve task location pass }];

    let constructor = "vpux::VPUMI40XX::createResolveTaskLocationPass()";
}

def BarrierTopologicalMapping : PassBase<"barrier-topological-mapping", "vpux::FunctionPass"> {
    let summary = [{Construct barrier topological dependencies}];

    let description = [{
        Calculates mapped inference specific barrier topological ordering and materializez BarrierTask dependencies.
        The defintion of an inter-barrier task topological depency is defined as follows :
            - A barrier task B1 has a topological dependency over another barrier task B0 IF AND ONLY IF the PRODUCTION of B1 is
            GUARANTEED to happen AFTER the CONSUMPTION of B0.
    }];

    let constructor = "vpux::VPUMI40XX::createBarrierTopologicalMappingPass()";
}

def GroupExecutionOps : PassBase<"group-execution-ops", "vpux::FunctionPass"> {
    let summary = [{Group execution ops into one view}];

    let constructor = "vpux::VPUMI40XX::createGroupExecutionOpsPass()";
}

def UnGroupExecutionOps : PassBase<"ungroup-execution-ops", "vpux::FunctionPass">  {
    let summary = [{UnGroup execution ops from one view}];

    let constructor = "vpux::VPUMI40XX::createUnGroupExecutionOpsPass()";
}

def WorkloadManagement : PassBase<"workload-management", "vpux::FunctionPass"> {
    let summary = [{Workload management pass}];

    let constructor = "vpux::VPUMI40XX::createWorkloadManagementPass()";
}

def ResolveWLMTaskLocation : PassBase<"resolve-wlm-task-location", "vpux::FunctionPass"> {
    let summary = [{ 40xx and WLM specific version of resolve task location pass }];
    let description = [{In case of workload management task locations assignment works by a different logic. In non-WLM
            scenario, the NNR feeders will bring tasks in local memory in a circular buffer manner, so the logic can
            iterate over all tasks and apply the round-robin logic.
            With WLM we bring tasks in groups and each group gets a ping-pong buffer. A group is materialized by an
            ExecutionGroup op, and every task inside an execution group will receive a task location from inside the PP
            buffer pool}];

    let constructor = "vpux::VPUMI40XX::createResolveWLMTaskLocationPass()";
}

def PropagateFinalBarrier : PassBase<"propagate-final-barrier", "vpux::FunctionPass"> {
    let summary = [{ Propagate info about final barrier to mapped inference}];

    let constructor = "vpux::VPUMI40XX::createPropagateFinalBarrierPass()";
}

def AddEnqueueOps : PassBase<"add-enqueue-ops", "vpux::FunctionPass"> {
    let summary = [{ Add Enqueue Ops}];

    let constructor = "vpux::VPUMI40XX::createAddEnqueueOpsPass()";
}

def LinkEnqueueTargets : PassBase<"link-enqueue-targets", "vpux::FunctionPass"> {
    let summary = [{ Hard-Link Enqueue Ops targets}];

    let description = [{
        An enqueue op supports a range of tasks. It's semantics specify that all tasks inside that range can be
        enqueued(in order). As a performance optimization, we wouldn't want all tasks to be enqueued individually.
        So for the task types that support hard-linking, we would link all the tasks that are within an enqueue range,
        and set the enqueueOp to only enqueue the First one.
    }];

    let constructor = "vpux::VPUMI40XX::createLinkEnqueueTargetsPass()";
}

def LinkAllOps : PassBase<"link-all-ops", "vpux::FunctionPass"> {
    let summary = [{ Hard-Link all ops that support HardLink}];

    let description = [{
        For non-managed inferences, runtime will support and expect all tasks within a list to be HW link listed (hard-linked).
        This pass will hardLink all tasks that have support for it, with the exception of the first task.
    }];

    let constructor = "vpux::VPUMI40XX::createLinkAllOpsPass()";
}

def UnrollFetchTaskOps : PassBase<"unroll-fetch-ops", "vpux::FunctionPass"> {
    let summary = [{ Unroll FetchTask Ops}];

    let constructor = "vpux::VPUMI40XX::createUnrollFetchTaskOpsPass()";
}

def SplitEnqueueOps : PassBase<"split-enqueue-ops", "vpux::FunctionPass"> {
    let summary = [{ Split enqueue Ops fot avoid bug when we have last op in task buffer in LL}];

    let description = [{
        There is a situation in which we are preempting at a boundary level and, in specific case of operations which linked,
        we might end up reading incorrect information from CMX during save state stage.
        Every enqueue op has [start, end] range inside. This pass split existing enqueue in two if find the task which contains special attribute
        As a result enqueue[start,end] will be transformed to two enqueues: enqueue[start, task_with_attr] , enqueue[task_with_attr+1, end]
    }];

    let constructor = "vpux::VPUMI40XX::createSplitEnqueueOpsPass()";
}

def UnrollEnqueueOps : PassBase<"unroll-enqueue-ops", "vpux::FunctionPass"> {
    let summary = [{ Unroll enqueue Ops so that each targets only one task}];

    let constructor = "vpux::VPUMI40XX::createUnrollEnqueueOpsPass()";
}

def AddBootstrapOps : PassBase<"add-bootstrap-ops", "vpux::FunctionPass"> {
    let summary = [{ Add Bootstrap Ops}];
    let description = [{Previously we initialize and used barriers on FW in fixed order
                        from 0 to MAX_BARRIER.
                        Bootstrap operation allow us start barriers in any order}];

    let constructor = "vpux::VPUMI40XX::createAddBootstrapOpsPass()";
}

def NextSameIdAssignment : PassBase<"next-same-id-assignment", "vpux::FunctionPass"> {
    let summary = "Compute and setup scheduling related fields";

    let description = [{
        This pass computes and sets up the (barrier) scheduling related fields - next_same_id
    }];

    let constructor = "vpux::VPUMI40XX::createNextSameIdAssignmentPass()";

}

#endif
