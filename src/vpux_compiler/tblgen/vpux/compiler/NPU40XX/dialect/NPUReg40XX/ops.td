//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops_interfaces.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUASM/attributes.td"

include "mlir/IR/SymbolInterfaces.td"

class NPUReg40XX_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            NPUReg40XX_Dialect,
            mnemonic,
            traits
        >;

class NPUReg40XX_ExecutableTaskOp<string mnemonic, list<Trait> traits = []> :
        NPUReg40XX_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                ]
            )
        > {
}

//
// NNDMAOp
//

def NPUReg40XX_NNDMAOp :
        NPUReg40XX_ExecutableTaskOp<"NNDMA",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol,
                IsolatedFromAbove
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DMADescriptorAttr:$dma_descriptor,
        SymbolRefAttr:$input,
        SymbolRefArrayAttr:$output_buffs,
        OptionalAttr<SymbolRefAttr>:$next_link,
        OptionalAttr<SymbolRefAttr>:$act_compression_size_entry,
        OptionalAttr<SymbolRefAttr>:$indices
    );
}

//
// MappedInferenceOp
//

def NPUReg40XX_MappedInferenceOp :
        NPUReg40XX_ExecutableTaskOp<"MappedInference",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serializeCached", "getBinarySize"]>
            ]
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64ArrayOfArraysAttr:$dmaCount,
        I64ArrayAttr:$dmaDDRCount,
        I64ArrayAttr:$dmaCMXCount,
        I64ArrayAttr:$invariantCount,
        I64ArrayAttr:$variantCount,
        I64ArrayAttr:$actKernelRangesCount,
        I64ArrayAttr:$actKernelInvocationsCount,
        I64Attr:$mediaCount,
        I64Attr:$barrierCount,
        SymbolRefAttr:$mappedInferenceVersion,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        OptionalAttr<SymbolRefAttr>:$dmaHwpBase,
        OptionalAttr<SymbolRefAttr>:$hwpWorkpointCfg,
        OptionalAttr<SymbolRefAttr>:$managedMappedInference,
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefArrayAttr>:$invariantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$variantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelRanges,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$mediaTasks,
        OptionalAttr<SymbolRefAttr>:$barrierTasks
    );
}

//
// ConfigureBarrierOp
//

def NPUReg40XX_ConfigureBarrierOp :
        NPUReg40XX_ExecutableTaskOp<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>
            ]
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPURegMapped_RegisterMappedAttr:$barrier_descriptor
    );
}

//
// ActShaveRtOp
//

def NPUReg40XX_ActShaveRtOp :
        NPUReg40XX_ExecutableTaskOp<"ActShaveRt",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
            ]
        > {
    let summary = "Act Shave Management Kernel";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$kernel_path
    );

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
        uint32_t getVersion();
    }];
}

//
// ActKernelRangeOp
//

def NPUReg40XX_ActKernelRangeOp :
        NPUReg40XX_ExecutableTaskOp<"ActKernelRange",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
            ]> {
    let summary = "Activation Kernel Range";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        ActKernelRangeDescriptorAttr:$act_range_descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$kernel_text,
        OptionalAttr<SymbolRefAttr>:$kernel_entry
    );
}

//
// ActKernelInvocationOp
//

def NPUReg40XX_ActKernelInvocationOp :
        NPUReg40XX_ExecutableTaskOp<"ActKernelInvocation",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
            ]> {
    let summary = "Activation Kernel Invocation";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        ActKernelInvocationDescriptorAttr:$act_invo_descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$kernel_range,
        OptionalAttr<SymbolRefAttr>:$kernel_data,
        SymbolRefAttr:$kernel_params,
        OptionalAttr<SymbolRefAttr>:$profiling_data
    );
}

//
// DPU Invariant Op
//

def NPUReg40XX_DPUInvariantOp :
        NPUReg40XX_ExecutableTaskOp<"DPUInvariant", [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
        ]> {

    let summary = "DPU Invariant Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        DPUInvariantDescriptorAttr:$dpu_invariant_descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        SymbolRefAttr:$input,
        OptionalAttr<SymbolRefAttr>:$input_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$input_storage_element_table,

        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weights_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$weight_table,
        OptionalAttr<SymbolRefAttr>:$spr_lookup_table,

        OptionalAttr<SymbolRefAttr>:$output,
        OptionalAttr<SymbolRefAttr>:$output_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$profiling_data,
        UnitAttr:$is_zero_offset_weights_table,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        UnitAttr:$is_continued
    );

}

//
// DPU Variant Op
//

def NPUReg40XX_DPUVariantOp :
        NPUReg40XX_ExecutableTaskOp<"DPUVariant", [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
        ]> {

    let summary = "DPU Variant Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<SymbolRefAttr>:$next_link,
        VPUASM_IndexTypeAttr:$task_index,

        DPUVariantDescriptorAttr:$dpu_variant_descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        SymbolRefAttr:$invariant_task_location,
        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weight_table,
        VPUIP_NCETaskTypeAttr:$nce_task_type,
        OptionalAttr<IntAttr>:$workload_id
    );

}

//
// M2IOp
//

def NPUReg40XX_M2IOp :
        NPUReg40XX_ExecutableTaskOp<"M2I",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
            ]
        > {

    let summary = "M2I Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefAttr:$input,
        SymbolRefAttr:$output_buff,
        OptionalAttr<SymbolRefAttr>:$profiling_data,
        OptionalAttr<SymbolRefAttr>:$next_link,
        M2IDescriptorAttr:$m2i_descriptor
    );

}

//
// NNRTConfigOp
//

def NPUReg40XX_NNrtConfigOp :
        NPUReg40XX_ExecutableTaskOp<"NNrtConfig",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serializeCached", "getBinarySize"]>
            ]
        > {
    let summary = "Task representing the NNRTConfig structure inside managed mapped inference";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UnitAttr:$isActKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        OptionalAttr<SymbolRefAttr>:$dmaHwpBase,
        OptionalAttr<SymbolRefAttr>:$hwpWorkpointCfg
    );
}

//
// ManagedMappedInferenceOp
//

def NPUReg40XX_ManagedMappedInferenceOp :
        NPUReg40XX_ExecutableTaskOp<"ManagedMappedInference",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serializeCached", "getBinarySize"]>
            ]
        > {
    let summary = "Task representing the ManagedMappedInference structure";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64Attr:$finalBarrier,
        I64ArrayAttr:$dmaDDRCount,
        I64ArrayAttr:$dmaCMXCount,
        I64Attr:$workItemsCount,
        I64Attr:$taskConfigsCount,
        I64Attr:$bootsrapWorkItemsCount,
        I64Attr:$bootstrapTaskCount,
        I64Attr:$barrierConfigurationTasksCount,
        I64Attr:$barriersReprogrammingCount,
        I64Attr:$barrierConfigurationStride,
        UI8Attr:$actshv_used,
        UI8Attr:$dpu_used,
        UI8Attr:$media_used,
        UI8Attr:$dma_from_ddr_used,
        UI8Attr:$dma_from_cmx_used,
        SymbolRefAttr:$nnrtConfig,
        SymbolRefAttr:$mappedInferenceVersion,
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefAttr>:$workItems,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        OptionalAttr<SymbolRefAttr>:$bootstrapTasks,
        OptionalAttr<SymbolRefAttr>:$barrierConfigurationTasks,
        OptionalAttr<SymbolRefAttr>:$numOfBarrierReprogrammings
    );
}

//
// ManagedBarrierOp
//

def NPUReg40XX_ManagedBarrierOp :
        NPUReg40XX_ExecutableTaskOp<"ManagedBarrier",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>
            ]
        > {
    let summary = "A task to configure the setup for a managed barrier";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPURegMapped_RegisterMappedAttr:$barrier_descriptor
    );
}

//
// WorkItemOp
//

def NPUReg40XX_WorkItemOp :
        NPUReg40XX_ExecutableTaskOp<"WorkItem",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>
            ]
        > {
    let summary = "A task to configure the work item";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPURegMapped_RegisterMappedAttr:$work_item_descriptor,
        VPURegMapped_TaskTypeAttr:$task_type,
        SymbolRefAttr:$first_task
    );
}

//
// MappedInferenceVersionOp
//

def MappedInferenceVersionOp :
        NPUReg40XX_Op<"MappedInferenceVersion",
            [
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize", "getBinarySize"]>,
                Symbol
            ]
        > {
    let summary = "Represents a versioning mechanism for Mapped Inference";

    let arguments = (ins
          SymbolNameAttr:$sym_name,
          UI32Attr:$major,
          UI32Attr:$minor,
          UI32Attr:$patch
        );

    let assemblyFormat = [{
        `(` $major `_` $minor `_` $patch `)`
        attr-dict}];
}
