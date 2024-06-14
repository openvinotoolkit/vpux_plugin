//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//
// Base classes
//
class VPURegMapped_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPURegMapped_Dialect,
            mnemonic,
            traits
        >;

def VPURegMapped_RegisterFieldWrapperOp :
        VPURegMapped_Op<"RegisterFieldWrapper",
            []
        > {
    let summary = "Register Field Wrapper Op";

    let arguments = (ins
        VPURegMapped_RegisterFieldAttr:$regFieldAttr
    );

    let assemblyFormat = [{
        attr-dict
        `regFieldAttr` `(` $regFieldAttr `)`
    }];
}

def VPURegMapped_RegisterWrapperOp :
        VPURegMapped_Op<"RegisterWrapper",
            []
        > {
    let summary = "Register Wrapper Op";

    let arguments = (ins
        VPURegMapped_RegisterAttr:$regAttr
    );

    let assemblyFormat = [{
        attr-dict
        `regAttr` `(` $regAttr `)`
    }];
}

def VPURegMapped_RegisterMappedWrapperOp :
        VPURegMapped_Op<"RegisterMappedWrapper",
            []
        > {
    let summary = "RegisterMapped Wrapper Op";

    let arguments = (ins
        VPURegMapped_RegisterMappedAttr:$regMapped
    );

    let assemblyFormat = [{
        attr-dict
        `regMapped` `(` $regMapped `)`
    }];
}


//
// FetchTaskOp
//

def VPURegMapped_FetchTaskOp:
        VPURegMapped_Op<"FetchTask", [
            DeclareOpInterfaceMethods<DotInterface, ["getNodeColor", "printAttributes", "getEdgeDirection"]>,
            DeclareOpInterfaceMethods<VPURegMapped_TaskOpInterface, ["setTaskLocation", "getTaskLocation"]>,
            AttrSizedOperandSegments
        ]> {

    let summary = "Op to represent fetching of specific tasks";

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$previousTask,
        VPURegMapped_IndexType:$primary_start,
        VPURegMapped_IndexType:$primary_end,
        Optional<VPURegMapped_IndexType>:$secondary_start,
        Optional<VPURegMapped_IndexType>:$secondary_end
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        (`previousTask` `(` $previousTask^ `:` type($previousTask) `)` )?
        `primary` `(` $primary_start `->` $primary_end `)`
        `secondary` `(` $secondary_start `->` $secondary_end `)`
        `(` type($primary_start) `->` type($primary_end) `:` type($secondary_start) `->` type($secondary_end) `)`
        `->` type($index)
        attr-dict
    }];
}

//
// EnqueueOp
//

def VPURegMapped_EnqueueOp:
        VPURegMapped_Op<"Enqueue", [
            DeclareOpInterfaceMethods<DotInterface, ["getNodeColor", "printAttributes", "getEdgeDirection"]>,
            AttrSizedOperandSegments
        ]> {

        let summary = "Op to represent actual enquement of a range of contigous tasks";

        let arguments = (ins
            Optional<VPURegMapped_IndexType>:$previousTaskIdx,
            Optional<VPURegMapped_IndexType>:$barrier,
            VPURegMapped_TaskTypeAttr:$taskType,
            VPURegMapped_IndexType:$start,
            VPURegMapped_IndexType:$end
        );

        let results = (outs
            VPURegMapped_IndexType:$index
        );

        let assemblyFormat = [{
            (`previousTaskIdx` `(` $previousTaskIdx^ `:` type($previousTaskIdx) `)` )?
            (`at` `(` $barrier^ `:` type($barrier) `)`)?
            `(` $start `->` $end `:` type($start) `->` type($end) `)`
            `->` type(results)

            attr-dict
        }];
    }

//
// ExecutionGroupOp
//

def VPURegMapped_ExecutionGroupOp :
    VPURegMapped_Op<"ExecutionGroup", [
        AttrSizedOperandSegments,
        AttrSizedResultSegments
    ]> {

    let summary = "Op to represent a grouping of executable ops that are fetched at once";

    let arguments = (ins
        Variadic<VPURegMapped_IndexType>:$previousTaskIdx,
        Variadic<VPURegMapped_IndexType>:$waitBarriers,
        Variadic<VPURegMapped_IndexType>:$updateBarriers,
        VPURegMapped_TaskTypeAttr:$task_type
    );

    let results = (outs
        Variadic<VPURegMapped_IndexType>:$startIndexes,
        Variadic<VPURegMapped_IndexType>:$endIndexes
    );

    let regions = (region
        SizedRegion<1>:$tasks
    );

    let hasVerifier =1;
}

//
// GroupYieldOp
//

def VPURegMapped_GroupYieldOp :
    VPURegMapped_Op<"GroupYield", [
        HasParent<"ExecutionGroupOp">,
        Terminator,
        AttrSizedOperandSegments
    ]> {

    let arguments = (ins
        Variadic<VPURegMapped_IndexType>:$listHeads,
        Variadic<VPURegMapped_IndexType>:$listTails
    );

    let hasVerifier = 1;
}

//
// DeclareTaskBufferOp
//

def DeclareTaskBufferOp :
        VPURegMapped_Op<"DeclareTaskBuffer", [
        ]> {
    let summary = "Buffer Op representing a discrete location of a task in runtime metadataBuffer";

    let arguments = (ins
        VPURegMapped_TaskTypeAttr:$task_type
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        $task_type `->` type(results)
    }];
}

//
// ViewTaskRangeOp
//

def ViewTaskRangeOp : VPURegMapped_Op<"ViewTaskRange", [
                        NoMemoryEffect
                    ]> {
    let summary = "Task range op, that represents a range of OP's as a memory (memref)";

    let arguments = (ins
        VPURegMapped_IndexType:$first,
        VPURegMapped_IndexType:$last
    );

    let results = (outs
        MemRefRankOf<[UI8], [2]>:$memview
    );

    let assemblyFormat = [{
        `(` $first `->` $last `:` type($first) `->` type($last) `)` `->` type($memview)
        attr-dict
    }];
}

//
// DeclareTaskAddrBufferOp
//

def DeclareTaskAddrBufferOp : VPURegMapped_Op<"DeclareTaskAddrBuffer", [
                                NoMemoryEffect
                            ]> {
    let summary = "Declare the buffer that would contain actual addresses of metadata elems";

    let arguments = (ins
        VPURegMapped_IndexType:$first,
        VPURegMapped_IndexType:$last,
        OptionalAttr<UI32Attr>:$offset
    );

    let results = (outs
        MemRefRankOf<[UI32], [1]>:$output
    );

    let assemblyFormat = [{
        `(` $first `->` $last `:` type($first) `->` type($last) `)` `->` type($output)
        attr-dict
    }];
}

//
// TaskSinkOp
//

def TaskSinkOp : VPURegMapped_Op<"TaskSink", [
                    NoMemoryEffect
                ]> {
    let summary = "Declare the sink where the task ptr will actually be copied";

    let arguments = (ins
        UI8Attr:$tile,
        VPURegMapped_TaskTypeAttr:$taskType
    );

    let results = (outs
        MemRefRankOf<[UI32], [1]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `tile` `(` $tile `)` `->` type($output)
    }];
}

#endif
