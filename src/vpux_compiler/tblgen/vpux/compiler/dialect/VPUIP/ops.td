//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS
#define VPUX_COMPILER_DIALECT_VPUIP_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/const/attributes.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/types.td"
include "vpux/compiler/dialect/VPURT/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"

//
// Base classes
//

class VPUIP_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIP_Dialect,
            mnemonic,
            traits
        >;

class VPUIP_TaskOp<int numResult, string mnemonic, list<Trait> traits = []> :
        VPUIP_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    InferTypeOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                    VPUIP_AsyncLayerOpInterface,
                    DeclareOpInterfaceMethods<VPUIP_TaskOpInterface>,
                    MemoryEffectsOpInterface
                ]
            )
        > {
    code commonExtraClassDeclaration = VPUIP_InferLayerReturnTypes<numResult>.impl;

    let extraClassDeclaration = commonExtraClassDeclaration;
}

class VPUIP_UPATaskOp<int numResult, string mnemonic, list<Trait> traits = []> :
        VPUIP_TaskOp<
            numResult,
            mnemonic,
            !listconcat(
                traits,
                [
                    VPUIP_UPATask
                ]
            )
        >;

//
// NNDMAOp
//

def VPUIP_NNDMAOp :
        VPUIP_TaskOp<1, "NNDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer, VPUIP_BoundedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$spillId,
        OptionalAttr<UnitAttr>:$compress_candidate,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata,
        UnitAttr:$split_candidate,
        UnitAttr:$profiling_buffer_mgmt
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "int32_t":$dma_hwp_id)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical, "mlir::IntegerAttr":$spillId)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$spillId)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$spillId, "mlir::UnitAttr":$compress_candidate)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$spillId, "mlir::UnitAttr":$compress_candidate, "bool":$split_candidate)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// PermuteDMAOp
//

def VPUIP_PermuteDMAOp :
        VPUIP_TaskOp<1, "PermuteDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "NN DMA task which enables permutation";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<AffineMapAttr>:$mem_perm,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::AffineMapAttr":$mem_perm,
                 "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::AffineMapAttr":$mem_perm,
                  "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// GatherDMAOp
//

def VPUIP_GatherDMAOp :
        VPUIP_TaskOp<1,"GatherDMA",
            [
                InferTypeOpInterface,
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = [{
        Represents a gather operation using gather addressing mode of DMA for NPU40XX.
        See scatter-gather addressing in NPU40XX HAS for more information.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$input,
        MemRefOf<[AnySignlessInteger, Index,I64,UI64]>:$indices,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer]>:$outputBuff,


        IntAttr:$elementSize,
        IntAttr:$padding,

        OptionalAttr<IntAttr>:$port,

        OptionalAttr<VPUIP_DmaChannelType>:$channelType,
        UnitAttr:$isOutOfOrder,
        UnitAttr:$isCritical,

        OptionalAttr<VPUIP_DMADescriptorAttr>:$DMADescriptor,
        OptionalAttr<IntAttr>:$dmaHwpId,

        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$indices, "mlir::Value":$outputBuff,
                 "mlir::IntegerAttr":$elementSize, "mlir::IntegerAttr":$padding,
                 "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$indices, "mlir::Value":$outputBuff,
                 "int64_t":$elementSize, "int64_t":$padding,
                 "int64_t":$port)
        >
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `indices` `(` $indices `:` type($indices) `)`
        `outputs` `(` $outputBuff `:` type($outputBuff) `)`
        `->` type(results)
    }];
}

//
// ConvertDMAOp
//

def VPUIP_ConvertDMAOp :
        VPUIP_TaskOp<1, "ConvertDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "NN DMA task which enables data conversion";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer, VPUIP_BoundedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// DecompressDMAOp
//

def VPUIP_DecompressDMAOp :
        VPUIP_TaskOp<1, "DecompressDMAOp",
            [
                ViewLikeOpInterface,
                DotInterface,
                VPUIP_DMATypeOpInterface,
                AttrSizedOperandSegments
            ]
        > {
    let summary = [{
        NN DMA task for BitCompactor weight compression & Activation Spill Decompression.
        In case of BitCompactor weight compression: use BitCompactor library to compress weights and decompress with the HW feature in the DMA (BTC)
        In case of Activation Spill Decompression: decompress with the HW feature in the DMA (BTC)
        Omits VPUIP_SameShape trait & VPUIP_SameElementType (verifier for input/output same element type)}];

    let arguments = (ins
        AnyMemRef:$input,
        Optional<AnyMemRef>:$act_compression_size_entry,
        Optional<AnyMemRef>:$act_compression_sparsity_map,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$act_compression_sparsity_map, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`act_compression_size_entry` `(` $act_compression_size_entry^  `:` type($act_compression_size_entry) `)`)?
        (`act_compression_sparsity_map` `(` $act_compression_sparsity_map^  `:` type($act_compression_sparsity_map) `)`)?
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// CompressDMAOp
//

def VPUIP_CompressDMAOp :
        VPUIP_TaskOp<1, "CompressDMAOp",
            [
                ViewLikeOpInterface,
                DotInterface,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = [{
        NN DMA task for Activation Spill Compression.
        In case of Activation Spill Compression: compress with the HW feature in the DMA (BTC).
        Omits VPUIP_SameShape trait & VPUIP_SameElementType (verifier for input/output same element type)}];

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$act_compression_size_entry,
        Optional<AnyMemRef>:$act_compression_sparsity_map,
        AnyMemRef:$output_buff,

        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$act_compression_sparsity_map, "mlir::Value":$output_buff, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$actCompressionSizeEntryBuff, "mlir::Value":$output_buff, "mlir::IntegerAttr":$port, "mlir::UnitAttr":$is_out_of_order, "mlir::UnitAttr":$is_critical)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `act_compression_size_entry` `(` $act_compression_size_entry  `:` type($act_compression_size_entry) `)`
        (`act_compression_sparsity_map` `(` $act_compression_sparsity_map^  `:` type($act_compression_sparsity_map) `)`)?
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// SyncDMAOp
//

def VPUIP_SyncDMAOp :
        VPUIP_TaskOp<1, "SyncDMA",
            [
                ViewLikeOpInterface,
                VPUIP_DMATypeOpInterface,
                DotInterface
            ]
        > {
    let summary = "Sync DMA task";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyMemRef:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// ConvertUPAOp
//

def VPUIP_ConvertUPAOp :
        VPUIP_UPATaskOp<1, "ConvertUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape
            ]
        > {
    let summary = "Convert UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        OptionalAttr<F64Attr>:$scale,
        OptionalAttr<F64Attr>:$bias,
        UnitAttr:$fromDetectionOutput,
        UnitAttr:$haveBatch,
        OptionalAttr<IntAttr>:$batchID
    );

    let results = (outs
        AnyMemRef:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output)
        >
    ];

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SoftMaxUPAOp
//

def VPUIP_SoftMaxUPAOp :
        VPUIP_UPATaskOp<1, "SoftMaxUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType
            ]
        > {
    let summary = "SoftMax UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$axisInd
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PoolingUPAOp
//

def VPUIP_PoolingUPAOp :
        VPUIP_UPATaskOp<1, "PoolingUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType
            ]
        > {
    let summary = "MAX and AVG Pooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        VPUIP_PoolLayerTypeAttr:$task_type,
        I64ArrayAttr:$kernel,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        UnitAttr:$excludePad
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AdaptiveAvgUPAOp
//

def VPUIP_AdaptiveAvgPoolUPAOp :
        VPUIP_UPATaskOp<1, "AdaptiveAvgPoolUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Adaptive AVG UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[SI32]>:$pooled_spatial_shape,
        MemRefOf<[F16, F32]>:$output_buff

    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $pooled_spatial_shape `:` type($pooled_spatial_shape) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}
//
// AdaptiveMaxUPAOp
//

def VPUIP_AdaptiveMaxPoolUPAOp :
        VPUIP_UPATaskOp<2, "AdaptiveMaxPoolUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "Adaptive Max UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, F32]>:$input,
        MemRefOf<[SI32]>:$pooled_spatial_shape,
        MemRefOf<[F16, F32]>:$output_buff,
        MemRefOf<[SI32]>:$output_index_buff,
        TypeAttr:$index_element_type
    );

    let results = (outs
        MemRefOf<[F16, F32]>:$output,
        MemRefOf<[SI32]>:$output_index
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $pooled_spatial_shape `:` type($pooled_spatial_shape) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `,` $output_index_buff `:` type($output_index_buff) `)`
        `->` type(results)
    }];
}
//
// ConvolutionUPAOp
//

def VPUIP_ConvolutionUPAOp :
        VPUIP_UPATaskOp<1, "ConvolutionUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType
            ]
        > {
    let summary = "Convolution UPA SHAVE kernel (reference implementation)";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$filter,
        Optional<F16MemRef>:$bias,
        F16MemRef:$output_buff,

        I64ArrayAttr:$strides,
        I64ArrayAttr:$dilations,
        I64ArrayAttr:$padsBegin,
        I64ArrayAttr:$padsEnd,
        IntAttr:$groups
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $filter `:` type($filter) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// ReLUUPAOp
//

def VPUIP_ReLUUPAOp :
        VPUIP_UPATaskOp<1, "ReLUUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "ReLU UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// RollUPAOp
//

def VPUIP_RollUPAOp :
        VPUIP_UPATaskOp<1, "RollUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Roll UPA SHAVE kernel";

    let arguments = (ins
            AnyMemRef:$data,
            MemRefOf<[SI32]>:$shift,
            MemRefOf<[SI32]>:$axes,
            AnyMemRef:$output_buff
    );

    let results = (outs
            AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $shift `:` type($shift) `,` $axes `:` type($axes) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

}

//
// SigmoidOp
//

def VPUIP_SigmoidUPAOp :
        VPUIP_UPATaskOp<1, "SigmoidUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// HSwishUPAOp
//

def VPUIP_HSwishUPAOp :
        VPUIP_UPATaskOp<1, "HSwishUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "HSwish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ErfOp
//

def VPUIP_ErfUPAOp :
        VPUIP_UPATaskOp<1, "ErfUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Erf UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MishUPAOp
//

def VPUIP_MishUPAOp :
        VPUIP_UPATaskOp<1, "MishUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Mish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FloorUPAOp
//

def VPUIP_FloorUPAOp :
        VPUIP_UPATaskOp<1, "FloorUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Floor UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// RoundUPAOp
//

def VPUIP_RoundUPAOp :
        VPUIP_UPATaskOp<1, "RoundUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Round UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IE_RoundModeAttr:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
       }];
}

//
// ClampUPAOp
//

def VPUIP_ClampUPAOp :
        VPUIP_UPATaskOp<1, "ClampUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Clamp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$min,
        F64Attr:$max
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// EluUPAOp
//

def VPUIP_EluUPAOp :
        VPUIP_UPATaskOp<1, "EluUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Elu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$x
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TanhUPAOp
//

def VPUIP_TanhUPAOp :
        VPUIP_UPATaskOp<1, "TanhUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Tanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SinUPAOp
//

def VPUIP_SinUPAOp :
        VPUIP_UPATaskOp<1, "SinUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sin UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CosUPAOp
//

def VPUIP_CosUPAOp :
        VPUIP_UPATaskOp<1, "CosUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Cos UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TanUPAOp
//

def VPUIP_TanUPAOp :
        VPUIP_UPATaskOp<1, "TanUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Tan UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SqrtUPAOp
//

def VPUIP_SqrtUPAOp :
        VPUIP_UPATaskOp<1, "SqrtUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sqrt UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SinhUPAOp
//

def VPUIP_SinhUPAOp :
        VPUIP_UPATaskOp<1, "SinhUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Sinh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CoshUPAOp
//

def VPUIP_CoshUPAOp :
        VPUIP_UPATaskOp<1, "CoshUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Cosh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AsinhUPAOp
//

def VPUIP_AsinhUPAOp :
        VPUIP_UPATaskOp<1, "AsinhUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Asinh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AcoshUPAOp
//

def VPUIP_AcoshUPAOp :
        VPUIP_UPATaskOp<1, "AcoshUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Acosh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AbsUPAOp
//

def VPUIP_AbsUPAOp :
        VPUIP_UPATaskOp<1, "AbsUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Abs UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// HSigmoidOp
//

def VPUIP_HSigmoidUPAOp :
        VPUIP_UPATaskOp<1, "HSigmoidUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "HSigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AtanUPAOp
//

def VPUIP_AtanUPAOp :
        VPUIP_UPATaskOp<1, "AtanUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Atan UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AsinUPAOp
//

def VPUIP_AsinUPAOp :
        VPUIP_UPATaskOp<1, "AsinUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Asin UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AcosUPAOp
//

def VPUIP_AcosUPAOp :
        VPUIP_UPATaskOp<1, "AcosUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Acos UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// AtanhUPAOp
//

def VPUIP_AtanhUPAOp :
        VPUIP_UPATaskOp<1, "AtanhUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Atanh UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogUPAOp
//

def VPUIP_LogUPAOp :
        VPUIP_UPATaskOp<1, "LogUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType
            ]
        > {
    let summary = "Log UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SeluUPAOp
//

def VPUIP_SeluUPAOp :
        VPUIP_UPATaskOp<1, "SeluUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Selu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$data,
        F16MemRef:$output_buff,

        F64Attr:$alphaValue,
        F64Attr:$lambdaValue
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GeluUPAOp
//

def VPUIP_GeluUPAOp :
        VPUIP_UPATaskOp<1, "GeluUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Gelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ExpUPAOp
//

def VPUIP_ExpUPAOp :
        VPUIP_UPATaskOp<1, "ExpUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Exp UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FakeQuantizeUPAOp
//

def VPUIP_FakeQuantizeUPAOp :
        VPUIP_UPATaskOp<1, "FakeQuantizeUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType
            ]
        > {
    let summary = "FakeQuantize UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$levels,
        Const_ContentAttr:$input_low,
        Const_ContentAttr:$input_high,
        Const_ContentAttr:$output_low,
        Const_ContentAttr:$output_high
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// QuantCastUPAOp
//

def VPUIP_QuantCastUPAOp :
        VPUIP_UPATaskOp<1, "QuantCastUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "QuantCast UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, quant_QuantizedType]>:$input,
        MemRefOf<[F16, quant_QuantizedType]>:$output_buff
    );

    let results = (outs
        MemRefOf<[F16, quant_QuantizedType]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PReluUPAOp
//

def VPUIP_PReluUPAOp :
        VPUIP_UPATaskOp<1, "PReluUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "PRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$negative_slope,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $negative_slope `:` type($negative_slope) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LeakyReluUPAOp
//

def VPUIP_LeakyReluUPAOp :
        VPUIP_UPATaskOp<1,  "LeakyReluUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "LeakyRelu UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$negative_slope
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// EltwiseUPAOp
//

def VPUIP_EltwiseUPAOp :
        VPUIP_UPATaskOp<1, "EltwiseUPA",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "Eltwise UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input1,
        F16MemRef:$input2,
        F16MemRef:$output_buff,

        VPU_EltwiseTypeAttr:$task_type
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LogicalNotUPAOp
//

def VPUIP_LogicalNotUPAOp :
        VPUIP_UPATaskOp<1, "LogicalNotUPA",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "Logical Not UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input1,
        F16MemRef:$output_buff,

        VPU_EltwiseTypeAttr:$task_type
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SelectUPAOp
//

def VPUIP_SelectUPAOp :
        VPUIP_UPATaskOp<1, "SelectUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape
            ]
        > {
    let summary = "Select UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[SI32,F16]>:$input1,
        MemRefOf<[SI32,F16]>:$input2,
        MemRefOf<[SI32,F16]>:$input3,
        MemRefOf<[SI32,F16]>:$output_buff
    );

    let results = (outs
        MemRefOf<[SI32,F16]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) `,` $input2 `:` type($input2) `,` $input3 `:` type($input3) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DetectionOutput
//

def VPUIP_DetectionOutputUPAOp :
        VPUIP_UPATaskOp<1, "DetectionOutputUPA",
            [
                AttrSizedOperandSegments,
                ViewLikeOpInterface
            ]
        > {
    let summary = "DetectionOutput UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$in_box_logits,
        F16MemRef:$in_class_preds,
        F16MemRef:$in_proposals,
        Optional<F16MemRef>:$in_additional_preds,
        Optional<F16MemRef>:$in_additional_proposals,

        F16MemRef:$output_buff,

        IE_DetectionOutputAttr:$attr
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $in_box_logits `:` type($in_box_logits) `,` $in_class_preds `:` type($in_class_preds) `,` $in_proposals `:` type($in_proposals) (`,` $in_additional_preds^ `:` type($in_additional_preds))? (`,` $in_additional_proposals^ `:` type($in_additional_proposals))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SwishUPAOp
//

def VPUIP_SwishUPAOp :
        VPUIP_UPATaskOp<1, "SwishUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Swish UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$beta_value
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// HardSigmoidUPAOp
//

def VPUIP_HardSigmoidUPAOp :
        VPUIP_UPATaskOp<1, "HardSigmoidUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "HardSigmoid UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$alpha_value,
        F64Attr:$beta_value
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}


//
// EmbeddingBagOffsetsSumUPAOp
//

def VPUIP_EmbeddingBagOffsetsSumUPAOp :
        VPUIP_UPATaskOp<1, "EmbeddingBagOffsetsSumUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "EmbeddingBagOffsetsSum UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        I64ArrayAttr:$indices_value,
        I64ArrayAttr:$offsets_value,
        IntAttr:$default_index_value,
        F64ArrayAttr:$weights_value,

        AnyMemRef:$output_buff
    );

    let results = (outs
        AnyMemRef:$output
    );
    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// EmbeddingSegmentsSumUPAOp
//

def VPUIP_EmbeddingSegmentsSumUPAOp :
        VPUIP_UPATaskOp<1, "EmbeddingSegmentsSumUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "EmbeddingSegmentsSum UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$emb_table,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$indices_value,
        I64ArrayAttr:$segment_ids_value,
        IntAttr:$num_segments_value,
        IntAttr:$default_index_value,
        F64ArrayAttr:$per_sample_weights_value
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $emb_table `:` type($emb_table) `)`
        `ouputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GRNUPAOp
//

def VPUIP_GRNUPAOp :
        VPUIP_UPATaskOp<1, "GRNUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "GRN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        F64Attr:$bias
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReduceUPAOp
//

def VPUIP_ReduceUPAOp :
        VPUIP_UPATaskOp<1, "ReduceUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Reduce UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[F16, SI32]>:$input,
        MemRefOf<[F16, SI32]>:$output_buff,

        I64ArrayAttr:$axes_value,
        UnitAttr:$keep_dims,
        VPUIP_ReduceLayerTypeAttr:$task_type
    );

    let results = (outs
        MemRefOf<[F16, SI32]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NormUPAOp
//

def VPUIP_NormUPAOp :
        VPUIP_UPATaskOp<1, "NormUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Norm UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,

        F16MemRef:$output_buff,

        F64Attr:$alpha,
        F64Attr:$beta,
        F64Attr:$bias,
        IntAttr:$local_size,
        IE_LRN_IERegionAttr:$region
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// BroadcastUPAOp
//

def VPUIP_BroadcastUPAOp :
        VPUIP_UPATaskOp<1, "BroadcastUPA",
            [
                ViewLikeOpInterface

            ]
        > {
    let summary = "Broadcast UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        MemRefOf<[SI32]>:$target_shape,
        Optional<MemRefOf<[SI32]>>:$axes_mapping,

        F16MemRef:$output_buff,

        OptionalAttr<IE_BroadcastTypeAttr>:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $target_shape `:` type($target_shape) (`,` $axes_mapping^ `:` type($axes_mapping))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// BucketizeUPAOp
//

def VPUIP_BucketizeUPAOp :
        VPUIP_UPATaskOp<1, "BucketizeUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Bucketize UPA SHAVE kernel";

    let arguments = (ins
        MemRefOf<[SI32, SI64, F16, F32]>:$data,
        MemRefOf<[SI32, SI64, F16, F32]>:$buckets,

        MemRefOf<[SI32, SI64]>:$output_buff,

        TypeAttr:$output_type,
        UnitAttr:$with_right_bound
    );

    let results = (outs
        MemRefOf<[SI32, SI64]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $buckets `:` type($buckets) `)`
        `ouputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// ScaleShiftUPAOp
//

def VPUIP_ScaleShiftUPAOp :
        VPUIP_UPATaskOp<1, "ScaleShiftUPA",
            [
                AttrSizedOperandSegments,
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "ScaleShift UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        Optional<F16MemRef>:$weights,
        Optional<F16MemRef>:$biases,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) (`,` $weights^ `:` type($weights))? (`,` $biases^ `:` type($biases))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// VPUIP_UpsamplingUPAOp
//

def VPUIP_UpsamplingUPAOp :
        VPUIP_UPATaskOp<1, "UpsamplingUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType
            ]
        > {
    let summary = "Upsampling UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$upsampling_factor,
        OptionalAttr<IE_UpsamplingPadAttr>:$pad
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PerAxisTileUPAOp
//

def VPUIP_PerAxisTileUPAOp :
        VPUIP_UPATaskOp<1, "PerAxisTileUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Tile for per axis case UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$axis,
        IntAttr:$tiles
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NegativeUPAOp
//

def VPUIP_NegativeUPAOp :
        VPUIP_UPATaskOp<1, "NegativeUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Negative UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SignUPAOp
//

def VPUIP_SignUPAOp :
        VPUIP_UPATaskOp<1, "SignUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Sign UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ProposalUPAOp
//

def VPUIP_ProposalUPAOp :
        VPUIP_UPATaskOp<2, "ProposalUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "Proposal UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$class_probs,
        F16MemRef:$bbox_deltas,
        F16MemRef:$image_shape,
        F16MemRef:$output_buff,
        F16MemRef:$probs_buff,
        IE_ProposalAttr:$proposal_attrs
    );

    let results = (outs
        F16MemRef:$output,
        F16MemRef:$probs
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $class_probs `:` type($class_probs) `,` $bbox_deltas `:` type($bbox_deltas) `,` $image_shape `:` type($image_shape) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `,` $probs_buff `:` type($probs_buff) `)`
        `->` type(results)
    }];
}

//
// ROIPoolingUPAOp
//

def VPUIP_ROIPoolingUPAOp :
        VPUIP_UPATaskOp<1, "ROIPoolingUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ROIPooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        F16MemRef:$output_buff,

        I64ArrayAttr:$output_size,
        F64Attr:$spatial_scale,
        IE_ROIPoolingMethodAttr:$method
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PSROIPoolingUPAOp
//

def VPUIP_PSROIPoolingUPAOp :
        VPUIP_UPATaskOp<1, "PSROIPoolingUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "PSROIPooling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        F16MemRef:$output_buff,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        IntAttr:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<IE_PSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ROIAlignUPAOp
//

def VPUIP_ROIAlignUPAOp :
        VPUIP_UPATaskOp<1, "ROIAlignUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ROIAlign UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$coords,
        MemRefOf<[SI32]>:$roisIdx,
        F16MemRef:$output_buff,

        IntAttr:$pooled_h,
        IntAttr:$pooled_w,
        IntAttr:$sampling_ratio,
        F64Attr:$spatial_scale,
        IE_ROIAlignMethodAttr:$poolingMode
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $coords `:` type($coords) `,` $roisIdx `:` type($roisIdx)`)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PermuteUPAOp
//

def VPUIP_PermuteUPAOp :
        VPUIP_UPATaskOp<1, "PermuteUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Permute UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        AffineMapAttr:$order_value
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
}

//
// DPUTaskOp
//

def VPUIP_DPUTaskOp :
        VPUIP_Op<
            "DPUTask",
            [
                HasParent<"vpux::VPUIP::NCEClusterTaskOp">,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>

            ]
        > {
    let summary = "This object represents workload for a single DPU tile";

    let arguments = (ins
        I64ArrayAttr:$outStart,
        I64ArrayAttr:$outEnd,
        OptionalAttr<I64ArrayAttr>:$inStart,
        OptionalAttr<I64ArrayAttr>:$inEnd,
        VPU_PaddingAttr:$pad,
        VPU_MPEModeAttr:$mpe_mode,
        OptionalAttr<IntAttr>:$cluster_id,
        OptionalAttr<DPUHaloRegionArrayAttr>: $haloRegions,
        OptionalAttr<IntAttr>:$workload_id
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode,
            "mlir::IntegerAttr":$clusterId
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "mlir::ArrayAttr":$inStart,
            "mlir::ArrayAttr":$inEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "mlir::ArrayAttr":$inStart,
            "mlir::ArrayAttr":$inEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode,
            "mlir::IntegerAttr":$clusterId
        )>,
        OpBuilder<(ins
            "mlir::ArrayAttr":$outStart,
            "mlir::ArrayAttr":$outEnd,
            "mlir::ArrayAttr":$inStart,
            "mlir::ArrayAttr":$inEnd,
            "VPU::PaddingAttr":$pad,
            "VPU::MPEMode":$mpeMode,
            "mlir::IntegerAttr":$clusterId,
            "mlir::ArrayAttr":$haloRegions
        )>
    ];

    let assemblyFormat = [{
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// PPETaskOp
//

def VPUIP_PPETaskOp :
        VPU_PPETaskOpBase<
            VPUIP_Dialect,
            "PPETask",
            [
                HasParent<"vpux::VPUIP::NCEClusterTaskOp">
            ]
         > {
    let builders = [
        OpBuilder<(ins
            CArg<"vpux::VPU::PPEMode", "vpux::VPU::PPEMode::NOOP">:$ppe_layer_type,
            CArg<"int64_t", "std::numeric_limits<int32_t>::min()">:$clamp_low,
            CArg<"int64_t", "std::numeric_limits<int32_t>::max()">:$clamp_high,
            CArg<"int64_t", "1">:$lrelu_mult,
            CArg<"int64_t", "0">:$lrelu_shift
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "double":$clamp_low,
            "double":$clamp_high,
            "double":$quant_scale
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "double":$clamp_low,
            "double":$clamp_high,
            "double":$quant_scale,
            "double":$fp_prelu_alpha
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "int64_t":$clamp_low,
            "int64_t":$clamp_high,
            "int64_t":$lrelu_mult,
            "int64_t":$lrelu_shift,
            "int64_t":$quant_mult,
            "int64_t":$quant_shift
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "int64_t":$clamp_low,
            "int64_t":$clamp_high,
            "int64_t":$lrelu_mult,
            "int64_t":$lrelu_shift,
            "int64_t":$quant_mult,
            "int64_t":$quant_shift,
            "int64_t":$quant_post_shift,
            "double":$quant_scale
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "int64_t":$clamp_low,
            "int64_t":$clamp_high,
            "int64_t":$lrelu_mult,
            "int64_t":$lrelu_shift,
            "int64_t":$quant_mult,
            "int64_t":$quant_shift,
            "int64_t":$quant_post_shift,
            "double":$quant_scale,
            "double":$ppe_fp_scale,
            "double":$ppe_fp_bias
        )>,
        OpBuilder<(ins
            "vpux::VPU::PPEMode":$ppe_layer_type,
            "int64_t":$clamp_low,
            "int64_t":$clamp_high,
            "int64_t":$lrelu_mult,
            "int64_t":$lrelu_shift,
            "int64_t":$quant_mult,
            "int64_t":$quant_shift,
            "int64_t":$quant_post_shift,
            "double":$quant_scale,
            "int64_t":$in1_quant_mult,
            "int64_t":$in2_quant_mult,
            "double":$fp_prelu_alpha
        )>
    ];
}

//
// NCEClusterTaskOp
//

def VPUIP_NCEClusterTaskOp :
        VPUIP_TaskOp<1, "NCEClusterTask",
            [
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                MultiViewOpInterface,
                NoRegionArguments,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "NCE Cluster Task Operation";

    let description = [{
        This operation defines NCE cluster task which describes single cluster of 5 DPUs. It is
        comprised of two argument categories:

        * Variants - describes the attributes for an individual DPU within the cluster.
        * Invariants - describes the collective attributes of the cluster.
        * is_superdense - a kind of ODU mode: allow OC = 4 and avoid channel alignment for IC <= 4 case.

        The variants argument takes on a region argument and up to 5 DPUTaskOps. The invariants
        take on a variety of argument types.

        Attribute superdense can be used for DPU permuteQuantize of NCHW FP16 input data.

        The NCEClusterTaskOp also supports fixed PPE functions as well as generic PPE instruction
        lists. The generic PPE instruction list argument needs to be described as a region of PPE
        supported ops. Single fixed PPE functions and generic PPE instruciton list usage is
        mutually exclusive.
    }];

    let arguments = (ins
        AnyTypeOf<[MemRefOf<[F16, BF16, quant_QuantizedType]>, VPUIP_DistributedBuffer, VPUIP_Buffer, VPUIP_ITIBuffer]>:$input,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$input_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[I32]>, VPUIP_DistributedBuffer]>>:$input_storage_element_table,
        Optional<AnyTypeOf<[MemRefOf<[F16, BF16, quant_QuantizedType]>, VPUIP_DistributedBuffer, VPUIP_Buffer]>>:$weights,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$weights_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$weight_table,
        Optional<AnyTypeOf<[MemRefOf<[SI32]>, VPUIP_DistributedBuffer]>>:$instruction_list_table,
        Optional<AnyTypeOf<[MemRefOf<[UI16]>, VPUIP_DistributedBuffer]>>:$spr_lookup_table,
        Optional<AnyTypeOf<[MemRefOf<[UI8]>, VPUIP_DistributedBuffer]>>:$activation_window,

        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$parent_input,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer]>>:$parent_input_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[I32]>, VPUIP_DistributedBuffer]>>:$parent_input_storage_element_table,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$parent_output,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>: $parent_output_sparsity_map,
        Variadic<VPUIP_ITIBuffer>:$output_ITI_buff,

        AnyTypeOf<[MemRefOf<[F16, F32, BF16, quant_QuantizedType]>, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$output_buff,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>:$output_sparsity_map_buff,
        Optional<AnyTypeOf<[MemRefOf<[UI64]>, VPUIP_DistributedBuffer]>>:$profiling_data,

        VPUIP_NCETaskTypeAttr:$task_type,
        OptionalAttr<I64ArrayAttr>:$kernel_size,
        OptionalAttr<I64ArrayAttr>:$kernel_strides,
        OptionalAttr<VPU_PaddingAttr>:$kernel_padding,
        OptionalAttr<IntAttr>:$activation_window_channel_length,
        UnitAttr:$is_continued,
        OptionalAttr<IntAttr>:$cm_sp_pattern,
        UnitAttr:$is_segmented,
        OptionalAttr<IntAttr>:$out_channel_offset,
        UnitAttr:$input_channels_compression,
        UnitAttr:$is_superdense,
        OptionalAttr<BoolAttr>:$is_inplace,
        OptionalAttr<IntAttr>:$input_se_size,
        OptionalAttr<IntAttr>:$output_se_size,
        UnitAttr:$is_permute_quantize,
        UnitAttr:$is_small_kernel_optimized,
        OptionalAttr<VPUIP_DpuProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[MemRefOf<[F16, F32, BF16, quant_QuantizedType]>, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>:$output,
        Optional<AnyTypeOf<[MemRefOf<[I1]>, VPUIP_DistributedBuffer, VPUIP_ITIBuffer]>>:$output_sparsity_map,
        Optional<AnyTypeOf<[MemRefOf<[UI64]>, VPUIP_DistributedBuffer]>>:$profiling_output
    );

    let regions = (region
        SizedRegion<1>:$variants,
        SizedRegion<1>:$ppe
    );

    let skipDefaultBuilders = 0;

    let builders = [
        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$output_sparsity_map, "mlir::Type":$profiling_output,
            "mlir::ValueRange":$operands,
            "llvm::ArrayRef<mlir::NamedAttribute>":$attributes
        )>,

        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>,

        OpBuilder<(ins
            "mlir::Type":$output,
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>,

        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>,

        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$profiling_output,
            "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_output,
            "mlir::Value":$output_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>,

        OpBuilder<(ins
            "mlir::Value":$input, "mlir::Value":$input_sparsity_map, "mlir::Value":$input_storage_element_table,
            "mlir::Value":$weights, "mlir::Value":$weights_sparsity_map,
            "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_input_sparsity_map, "mlir::Value":$parent_input_storage_element_table,
            "mlir::Value":$parent_output, "mlir::Value":$parent_output_sparsity_map,
            "mlir::Value":$output_buff, "mlir::Value":$output_sparsity_map_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>,

        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$output_sparsity_map, "mlir::Type":$profiling_output,
            "mlir::Value":$input, "mlir::Value":$input_sparsity_map, "mlir::Value":$input_storage_element_table,
            "mlir::Value":$weights, "mlir::Value":$weights_sparsity_map,
            "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_input_sparsity_map, "mlir::Value":$parent_input_storage_element_table,
            "mlir::Value":$parent_output, "mlir::Value":$parent_output_sparsity_map,
            "mlir::Value":$output_buff, "mlir::Value":$output_sparsity_map_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>,

        OpBuilder<(ins
            "mlir::Type":$output, "mlir::Type":$output_sparsity_map, "mlir::Type":$profiling_output,
            "mlir::Value":$input, "mlir::Value":$input_sparsity_map, "mlir::Value":$input_storage_element_table,
            "mlir::Value":$weights, "mlir::Value":$weights_sparsity_map,
            "mlir::Value":$weight_table,
            "mlir::Value":$instruction_list_table, "mlir::Value":$spr_lookup_table, "mlir::Value":$activation_window,
            "mlir::Value":$parent_input, "mlir::Value":$parent_input_sparsity_map, "mlir::Value":$parent_input_storage_element_table,
            "mlir::Value":$parent_output, "mlir::Value":$parent_output_sparsity_map, "mlir::ValueRange":$output_ITI_buff,
            "mlir::Value":$output_buff, "mlir::Value":$output_sparsity_map_buff, "mlir::Value":$profiling_data,
            "vpux::VPUIP::NCETaskType":$task_type,
            "mlir::ArrayAttr":$kernel_size, "mlir::ArrayAttr":$kernel_strides, "vpux::VPU::PaddingAttr":$kernel_padding,
            "mlir::IntegerAttr":$activation_window_channel_length,
            CArg<"mlir::UnitAttr", "nullptr">:$is_continued,
            CArg<"mlir::IntegerAttr", "nullptr">:$cm_sp_pattern,
            CArg<"mlir::UnitAttr", "nullptr">:$is_segmented,
            CArg<"mlir::IntegerAttr", "nullptr">:$out_channel_offset,
            CArg<"mlir::UnitAttr", "nullptr">:$input_channels_compression,
            CArg<"mlir::UnitAttr", "nullptr">:$is_superdense,
            CArg<"mlir::BoolAttr", "nullptr">:$is_inplace,
            CArg<"mlir::IntegerAttr", "nullptr">:$input_se_size,
            CArg<"mlir::IntegerAttr", "nullptr">:$output_se_size,
            CArg<"mlir::UnitAttr", "nullptr">:$is_permute_quantize,
            CArg<"mlir::UnitAttr", "nullptr">:$is_small_kernel_optimized
        )>
    ];

    let assemblyFormat = [{
        attr-dict
        `input` `(` $input  `:` type($input) `)`
        (`input_sparsity_map` `(` $input_sparsity_map^  `:` type($input_sparsity_map) `)`)?
        (`input_storage_element_table` `(` $input_storage_element_table^  `:` type($input_storage_element_table) `)`)?
        (`weights` `(` $weights^  `:` type($weights) `)`)?
        (`weights_sparsity_map` `(` $weights_sparsity_map^  `:` type($weights_sparsity_map) `)`)?
        (`weight_table` `(` $weight_table^  `:` type($weight_table) `)`)?
        (`instruction_list_table` `(` $instruction_list_table^  `:` type($instruction_list_table) `)`)?
        (`spr_lookup_table` `(` $spr_lookup_table^  `:` type($spr_lookup_table) `)`)?
        (`activation_window` `(` $activation_window^  `:` type($activation_window) `)`)?
        `parent_input` `(` $parent_input `:` type($parent_input) `)`
        (`parent_input_sparsity_map` `(` $parent_input_sparsity_map^  `:` type($parent_input_sparsity_map) `)`)?
        (`parent_input_storage_element_table` `(` $parent_input_storage_element_table^  `:` type($parent_input_storage_element_table) `)`)?
        `parent_output` `(` $parent_output `:` type($parent_output) `)`
        (`parent_output_sparsity_map` `(` $parent_output_sparsity_map^  `:` type($parent_output_sparsity_map) `)`)?
        (`output_ITI_buff` `(` $output_ITI_buff^ `:` type($output_ITI_buff) `)`)?
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`output_sparsity_map` `(` $output_sparsity_map_buff^  `:` type($output_sparsity_map_buff) `)`)?
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        `->` type($output) `` custom<OptionalResultTypes>(type($output_sparsity_map), type($profiling_output)) ` `
        `variants` `:` $variants
        `PPE` `:` $ppe
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId = nullptr);
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode,
                                          mlir::IntegerAttr clusterId, mlir::ArrayAttr haloRegions);
        vpux::VPUIP::DPUTaskOp addDPUTask(mlir::OpBuilder& builder,
                                          mlir::ArrayAttr outStart, mlir::ArrayAttr outEnd,
                                          vpux::VPU::PaddingAttr pad, vpux::VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId = nullptr);

        template <typename... Args>
        vpux::VPUIP::PPETaskOp addPPETask(mlir::OpBuilder& builder, Args&&... args);

        int64_t getNumVariants();

        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DPU;
        }

        static mlir::StringRef getDefaultDialect() {
            return "VPUIP";
        }

        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                std::optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::OpaqueProperties,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
    }];
}

//
// CTCGreedyDecoderUPAOp
//

def VPUIP_CTCGreedyDecoderUPAOp :
        VPUIP_UPATaskOp<1, "CTCGreedyDecoderUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "CTCGreedyDecoder UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$sequenceLengths,
        F16MemRef:$output_buff,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLengths `:` type($sequenceLengths) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CTCGreedyDecoderSeqLenUPAOp
//

def VPUIP_CTCGreedyDecoderSeqLenUPAOp :
        VPUIP_UPATaskOp<2, "CTCGreedyDecoderSeqLenUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "CTCGreedyDecoderSeqLen UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        MemRefOf<[SI32]>:$sequenceLength,
        Optional<MemRefOf<[SI32]>>:$blankIndex,
        MemRefOf<[SI32]>:$output_buff,
        MemRefOf<[SI32]>:$outputLength_buff,

        UnitAttr:$mergeRepeated
    );

    let results = (outs
        MemRefOf<[SI32]>:$output,
        MemRefOf<[SI32]>:$outputLength
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $sequenceLength `:` type($sequenceLength) `,` ($blankIndex^ `:` type($blankIndex))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `,` $outputLength_buff `:` type($outputLength_buff) `)`
        `->` type(results)
    }];
}

//
// PadOp
//

def VPUIP_PadUPAOp :
        VPUIP_UPATaskOp<1,"Pad",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Pad UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,
        OptionalAttr<F64Attr>:$pad_value,
        IE_PadModeAttr:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// LSTMCellUPAOp
//

def VPUIP_LSTMCellUPAOp :
        VPUIP_UPATaskOp<2, "LSTMCellUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "LSTMCell UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$inputData,
        F16MemRef:$initialHiddenState,
        F16MemRef:$initialCellState,
        F16MemRef:$weights,
        F16MemRef:$biases,
        F16MemRef:$outputHiddenState_buff,
        F16MemRef:$outputCellState_buff
    );

    let results = (outs
        F16MemRef:$outputHiddenState,
        F16MemRef:$outputCellState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenState_buff `:` type($outputHiddenState_buff)
        `,` $outputCellState_buff `:` type($outputCellState_buff) `)`
        `->` type(results)
    }];
}

//
// VPUIP_SwKernelRun
//

def VPUIP_SwKernelRun
    : VPUIP_Op<"SW.Kernel.run"> {

    let arguments  = (ins
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$args,
        OptionalAttr<ArrayAttr>:$attrs
    );

    let assemblyFormat = [{
        attr-dict
        ( `(` $args^ `)` `:` type($args) )?
    }];
}

//
// VPUIP_SwKernelOp
//

def VPUIP_SwKernelOp :
        VPUIP_TaskOp<1, "SW.Kernel",
            [
                MultiViewOpInterface,
                IsolatedFromAbove,
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Software Layer Task";

    let description = [{
        This operation defines Activation shave task.
        There are two different modes or handling inputs with dynamic shapes
        - In the first mode, dynamic inputs are accepted as BoundedBuffers, which combine data and dynamic shape into one type.
        dynamicInputShapes and dynamicInputShapesMap are not used in that case:
                inputs[3]: [BoundedBuffer, MemRef, BoundedBuffer]
                dynamicInputShapes[]: []
                dynamicInputShapesMap[]: nullptr
        - In the second mode all BoundedBuffer's are unrolled by ungroup-bounded-buffers pass: E#111348.
        Separate handling of dynamic data and dynamic shape is needed because feasible allocation doesn't support multiple root buffers per input
                inputs[3]: [MemRef, MemRef, MemRef]
                dynamicInputShapes[2]: [MemRef, MemRef]
                dynamicInputShapesMap[3]: [0, -1, 1]
        For outputs, the same applies.
    }];

    let arguments = (ins
        SymbolRefAttr:$kernelFunction,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$inputs,
        Variadic<MemRefOf<[SI32]>>:$dynamicInputShapes,
        OptionalAttr<DenseI32ArrayAttr>:$dynamicInputShapesMap,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$output_buffs,
        Variadic<MemRefOf<[SI32]>>:$dynamicOutputShapeBuffs,
        OptionalAttr<DenseI32ArrayAttr>:$dynamicOutputShapesMap,
        Optional<AnyTypeOf<[MemRefOf<[UI32]>, VPUIP_DistributedBuffer]>>:$profiling_data,
        OptionalAttr<IntAttr>:$tileIndex,
        OptionalAttr<I64ArrayOfArraysAttr>:$strides,
        OptionalAttr<VPUIP_SwProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>>:$results,
        Variadic<MemRefOf<[SI32]>>:$dynamicOutputShapes,
        Optional<AnyTypeOf<[MemRefOf<[UI32]>, VPUIP_DistributedBuffer]>>:$profiling_output
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let builders = [
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$strides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$strides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$strides
        )>,
        OpBuilder<(ins
            "mlir::ValueRange":$inputs,
            "mlir::ValueRange":$output_buffs,
            "mlir::ValueRange":$dynamicInputShapes,
            "llvm::ArrayRef<int32_t>":$dynamicInputShapesMap,
            "mlir::ValueRange":$dynamicOutputShapeBuffs,
            "llvm::ArrayRef<int32_t>":$dynamicOutputShapesMap,
            "mlir::Value":$profiling_data,
            "mlir::SymbolRefAttr":$kernelFunction,
            "mlir::IntegerAttr":$tileIndex,
            "mlir::ArrayAttr":$strides
        )>
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_ACT;
        }

        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::OpaqueProperties,
                                                    mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes);

        static vpux::VPUIP::KernelInfo getKernelInfo(mlir::Operation* origOp);
        static vpux::VPUIP::KernelInfo getDummyKernelInfo();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];

    let hasVerifier = 1;
}

//
// LSTMSequenceUPAOp
//

def VPUIP_LSTMSequenceUPAOp :
        VPUIP_UPATaskOp<3, "LSTMSequenceUPA",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "LSTMSequence UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$inputData,
        F16MemRef:$initialHiddenState,
        F16MemRef:$initialCellState,
        F16MemRef:$weights,
        F16MemRef:$biases,
        F16MemRef:$outputHiddenValues_buff,
        F16MemRef:$outputCellState_buff,
        F16MemRef:$outputHiddenState_buff,

        IntAttr:$sequenceLength,
        IE_RNNSequenceDirectionAttr:$direction
    );

    let results = (outs
        F16MemRef:$outputHiddenValues,
        F16MemRef:$outputCellState,
        F16MemRef:$outputHiddenState
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputData `:` type($inputData) `,` $initialHiddenState `:` type($initialHiddenState)
        `,` $initialCellState `:` type($initialCellState) `,` $weights `:` type($weights) `,` $biases `:` type($biases) `)`
        `outputs` `(` $outputHiddenValues_buff `:` type($outputHiddenValues_buff)
        `,` $outputCellState_buff `:` type($outputCellState_buff)
        `,` $outputHiddenState_buff `:` type($outputHiddenState_buff) `)`
        `->` type(results)
    }];
}

//
// InterpolateOp
//

def VPUIP_InterpolateUPAOp :
        VPUIP_UPATaskOp<1,"Interpolate",
            [
                ViewLikeOpInterface,
            ]
        > {
    let summary = "Interpolate UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,

        F16MemRef:$output_buff,

        IE_InterpolateModeAttr:$mode,
        IE_InterpolateCoordModeAttr:$coord_mode,
        IE_InterpolateNearestModeAttr:$nearest_mode,
        UnitAttr:$antialias
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}


//
// StridedSliceUPAOp
//

def VPUIP_StridedSliceUPAOp :
     VPUIP_UPATaskOp<1, "StridedSlice",
         [
             ViewLikeOpInterface
         ]
     > {
    let summary = "StridedSlice UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        I64ArrayAttr:$begins,
        I64ArrayAttr:$ends,
        I64ArrayAttr:$strides
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// FullyConnectedUPAOp
//

def VPUIP_FullyConnectedUPAOp :
        VPUIP_UPATaskOp<1, "FullyConnectedUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "FullyConnected UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$weights,
        Optional<F16MemRef>:$bias,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $weights `:` type($weights) (`,` $bias^ `:` type($bias))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// RegionYoloOp
//

def VPUIP_RegionYoloUPAOp :
        VPUIP_UPATaskOp<1, "RegionYoloUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "RegionYolo UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$coords,
        IntAttr:$classes,
        IntAttr:$num_regions,
        OptionalAttr<BoolAttr>:$do_softmax,
        OptionalAttr<I64ArrayAttr>:$mask
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ReorgYoloUPAOp
//

def VPUIP_ReorgYoloUPAOp :
        VPUIP_UPATaskOp<1, "ReorgYoloUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ReorgYolo UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        IntAttr:$stride
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// MVNOp
//

def VPUIP_MVNUPAOp :
        VPUIP_UPATaskOp<1, "MVNUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType
            ]
        > {
    let summary = "MVN UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        OptionalAttr<BoolAttr>:$across_channels,
        OptionalAttr<BoolAttr>:$normalize_variance,
        F64Attr:$eps
    );

    let results = (outs
        AnyMemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherUPAOp
//

def VPUIP_GatherUPAOp :
        VPUIP_UPATaskOp<1, "GatherUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "Gather UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32]>:$indices,
        AnyMemRef:$output_buff,

        IntAttr:$axis,
        IntAttr:$batch_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherNDUPAOp
//

def VPUIP_GatherNDUPAOp :
        VPUIP_UPATaskOp<1, "GatherNDUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "GatherND UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32]>:$indices,
        AnyMemRef:$output_buff,

        IntAttr:$batch_dims
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SpaceToDepthOp
//

def VPUIP_SpaceToDepthUPAOp :
        VPUIP_UPATaskOp<1, "SpaceToDepthUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType
            ]
        > {
    let summary = "SpaceToDepth UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$block_size,
        IE_SpaceToDepthModeAttr:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// SpaceToDepthDMAOp
//

def VPUIP_SpaceToDepthDMAOp :
        VPUIP_TaskOp<1, "SpaceToDepthDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a UPA SpaceToDepth task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        IntAttr:$block_size,
        IE_SpaceToDepthModeAttr:$mode,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::SpaceToDepthModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::SpaceToDepthModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// PerAxisTileDMAOp
//

def VPUIP_PerAxisTileDMAOp :
        VPUIP_TaskOp<1, "PerAxisTileDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a UPA PerAxisTileUPAOp task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        OptionalAttr<IntAttr>:$axis,
        OptionalAttr<IntAttr>:$tiles,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$axis,
            "mlir::IntegerAttr":$tiles, "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "mlir::IntegerAttr":$tiles, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ScatterNDUpdateUPAOp
//

def VPUIP_ScatterNDUpdateUPAOp :
        VPUIP_UPATaskOp<1, "ScatterNDUpdateUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ScatterNDUpdate UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32]>:$indices,
        AnyMemRef:$updates,
        AnyMemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `,` $updates `:` type($updates) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ScatterUpdateUPAOp
//

def VPUIP_ScatterUpdateUPAOp :
        VPUIP_UPATaskOp<1, "ScatterUpdateUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ScatterUpdate UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI32]>:$indices,
        AnyMemRef:$updates,
        AnyMemRef:$output_buff,
        OptionalAttr<IntAttr>:$axis_value
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `,` $updates `:` type($updates) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}
//
// CeilingUPAOp
//

def VPUIP_CeilingUPAOp :
        VPUIP_UPATaskOp<1, "CeilingUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType,
                VPUIP_Legacy4D
            ]
        > {
    let summary = "Ceiling UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// GatherElementsUPAOp
//

def VPUIP_GatherElementsUPAOp :
        VPUIP_UPATaskOp<1, "GatherElementsUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "GatherElements UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$input,
        MemRefOf<[SI64, SI32]>:$indices,
        AnyMemRef:$output_buff,

        IntAttr:$axis
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `,` $indices `:` type($indices) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DepthToSpaceUPAOp
//

def VPUIP_DepthToSpaceUPAOp :
        VPUIP_UPATaskOp<1, "DepthToSpaceUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType
            ]
        > {
    let summary = "DepthToSpace UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels
    );

    let results = (outs
        F16MemRef:$output
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode)
        >
    ];

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// DepthToSpaceDMAOp
//

def VPUIP_DepthToSpaceDMAOp :
        VPUIP_TaskOp<1, "DepthToSpaceDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a UPA DepthToSpace task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,

        OptionalAttr<IntAttr>:$port,
        IntAttr:$block_size,
        IE_DepthToSpaceModeAttr:$mode,
        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IE_ChannelPaddingAttr>:$padded_channels,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor,
            "vpux::IE::ChannelPaddingAttr":$padded_channels)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::IntegerAttr":$block_size,
            "vpux::IE::DepthToSpaceModeAttr":$mode, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port,
            "vpux::IE::ChannelPaddingAttr":$padded_channels)
        >
    ];

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NormalizeIEOp
//

def VPUIP_NormalizeIEUPAOp :
        VPUIP_UPATaskOp<1, "NormalizeIEUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "NormalizeIE UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$data,
        F16MemRef:$weights,
        F16MemRef:$output_buff,

        F64Attr:$eps,
        BoolAttr:$across_spatial,
        BoolAttr:$channel_shared
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $weights `:` type($weights) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// CumSumUPAOp
//

def VPUIP_CumSumUPAOp :
        VPUIP_UPATaskOp<1, "CumSumUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "CumSum UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff,

        OptionalAttr<IntAttr>:$axis_value,
        UnitAttr:$exclusive,
        UnitAttr:$reverse
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// NonMaxSuppression
//

def VPUIP_NonMaxSuppressionUPAOp :
        VPUIP_UPATaskOp<3, "NonMaxSuppressionUPA",
            [
                MultiViewOpInterface,
            ]
        > {
    let summary = "NonMaxSuppression UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$in_box_coords,
        F16MemRef:$in_box_scores,
        MemRefOf<[SI32]>:$out_selected_indices_buff,
        F16MemRef:$out_selected_scores_buff,
        MemRefOf<[SI32]>:$out_valid_outputs_buff,

        IE_BoxEncodingTypeAttr:$box_encoding,
        UnitAttr:$sort_result_descending,

        IntAttr:$max_output_boxes_per_class_value,
        F64Attr:$iou_threshold_value,
        F64Attr:$score_threshold_value,
        F64Attr:$soft_nms_sigma_value
    );

    let results = (outs
        MemRefOf<[SI32]>:$out_selected_indices,
        F16MemRef:$out_selected_scores,
        MemRefOf<[SI32]>:$out_valid_outputs
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $in_box_coords `:` type($in_box_coords) `,`
                    $in_box_scores `:` type($in_box_scores)
        `outputs` `(` $out_selected_indices_buff `:` type($out_selected_indices_buff) `,`
                    $out_selected_scores_buff `:` type($out_selected_scores_buff) `,`
                    $out_valid_outputs_buff `:` type($out_valid_outputs_buff) `)`
        `->` type(results)
    }];
}

//
// ReverseSequenceOp
//

def VPUIP_ReverseSequenceUPAOp :
        VPUIP_UPATaskOp<1, "ReverseSequenceUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ReverseSequence UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$data,
        F16MemRef:$seq_length,
        F16MemRef:$output_buff,

        IntAttr:$seq_axis,
        IntAttr:$batch_axis
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `,` $seq_length `:` type($seq_length) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// TopKOp
//

def VPUIP_TopKUPAOp:
        VPUIP_UPATaskOp<2,"TopK",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "TopK UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        MemRefOf<[SI32]>:$k,
        F16MemRef:$output_values_buff,
        MemRefOf<[SI32]>:$target_shape_buff,

        IntAttr:$axis,
        IE_TopKModeAttr:$mode,
        IE_TopKSortTypeAttr:$sort,
        TypeAttr:$element_type
    );

    let results = (outs
        F16MemRef:$output_values,
        MemRefOf<[SI32]>:$target_shape
    );
}

//
// SoftPlusUPAOp
//

def VPUIP_SoftPlusUPAOp :
        VPUIP_UPATaskOp<1, "SoftPlusUPA",
            [
                ViewLikeOpInterface,
                VPUIP_SameShape,
                VPUIP_SameElementType
            ]
        > {
    let summary = "SoftPlus UPA SHAVE kernel";

    let arguments = (ins
        F16MemRef:$input,
        F16MemRef:$output_buff
    );

    let results = (outs
        F16MemRef:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// YuvToRgbOp
//

def VPUIP_YuvToRgbUPAOp :
        VPUIP_UPATaskOp<1,"YuvToRgbUPA",
            [
                ViewLikeOpInterface,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "Yuv-to-Rgb UPA SHAVE kernel";

    let arguments = (ins
                 MemRefOf<[UI8, F16, F32]> :$input1,
        Optional<MemRefOf<[UI8, F16, F32]>>:$input2,
        Optional<MemRefOf<[UI8, F16, F32]>>:$input3,
                 MemRefOf<[UI8, F16, F32]> :$output_buff,

        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt
    );

    let results = (outs
        MemRefOf<[UI8, F16, F32]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input1 `:` type($input1) (`,` $input2^ `:` type($input2))? (`,` $input3^ `:` type($input3))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

}

//
// M2ITaskOp
//

def VPUIP_M2ITaskOp :
        VPUIP_TaskOp<1,"M2ITask",
            [
                MultiViewOpInterface
            ]
        > {
    let summary = "M2I task op";

    let arguments = (ins
        MemRefOf<[UI8, F16]>:$input,
        MemRefOf<[UI8, F16]>:$output_buff,
        Optional<MemRefOf<[UI8]>>:$profiling_data,

        BoolAttr:$do_csc,
        BoolAttr:$do_norm,
        VPU_M2iColorFmtAttr:$inFmt,
        VPU_M2iColorFmtAttr:$outFmt,
        UnitAttr:$chroma_in_reverse_channels,
        UnitAttr:$chroma_out_reverse_channels,
        UnitAttr:$luma_in_reverse_channels,
        UnitAttr:$luma_out_reverse_channels,
        UI32Attr:$scale_factor_x,
        UI32Attr:$scale_factor_y,
        OptionalAttr<F64ArrayAttr>:$norm,
        OptionalAttr<UI32Attr>:$tile_offset_x,
        OptionalAttr<UI32Attr>:$tile_offset_y,
        OptionalAttr<VPUIP_M2IProfilingMetadataAttr>:$profilingMetadata,
        DefaultValuedAttr<VPU_M2iInterpAttr, "vpux::VPU::M2iInterp::NEAREST">:$interp
    );

    let results = (outs
        MemRefOf<[UI8, F16]>:$output,
        Optional<MemRefOf<[UI8]>>:$profiling_output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        (`profiling_data` `(` $profiling_data^  `:` type($profiling_data) `)`)?
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::M2I;
        }

        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::OpaqueProperties,
                                                    mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes);
    }];
}

//
// NCEClusterTiling
//

def VPUIP_NCEClusterTilingOp :
        VPUIP_Op<
            "NCEClusterTiling",
            [
                NoTerminator,
                IsolatedFromAbove,
                MultiViewOpInterface,
                AttrSizedOperandSegments,
                MemoryEffectsOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                VPUIP_AsyncLayerOpInterface,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>,
            ] # GraphRegionNoTerminator.traits
        > {
    let summary = "Operation that encapsulates details of tiling operation between clusters";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>>:$inputs,
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>>:$output_buffs
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>>:$results
    );

    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

        mlir::Operation* getInnerTaskOp();
        template <typename T>
        T getInnerTaskOpOfType();
        mlir::MutableArrayRef<mlir::BlockArgument> getInnerInputs();
        mlir::MutableArrayRef<mlir::BlockArgument> getInnerOutputs();

        vpux::IndexedSymbolAttr getExecutor();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];
}

//
// ExtractImagePatchesUPAOp
//

def VPUIP_ExtractImagePatchesUPAOp :
        VPUIP_UPATaskOp<1, "ExtractImagePatchesUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "ExtractImagePatches UPA SHAVE kernel";

    let arguments = (ins
        AnyMemRef:$data,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$rates,
        IE_PadTypeAttr:$autoPad
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $data `:` type($data) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

}

//
// DistributedCastOp
//

def VPUIP_DistributedCastOp :
        VPUIP_Op<
            "DistributedCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that casts one DistributedBuffer type to another.";

    let arguments = (ins
        AnyTypeOf<[VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` qualified(type($input)) `)`
        `->` qualified(type(results))
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

//
// NonDistributedCast
//

def VPUIP_NonDistributedCastOp:
        VPUIP_Op<
            "NonDistributedCastOp",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that casts one DistributedBuffer type to non-distributed buffer type.";

    let arguments = (ins
        AnyTypeOf<[VPUIP_DistributedBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` qualified(type($input)) `)`
        `->` qualified(type(results))
    }];

    let hasVerifier = 1;
}


//
// SubViewOp
//

def VPUIP_SubViewOp :
        VPUIP_Op<
            "SubView",
            [
                Pure,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface>,
                DotInterface
            ]
        > {
    let summary = "Extract single subview from buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$source,

        I64ArrayAttr:$static_offsets,
        I64ArrayAttr:$static_sizes,
        OptionalAttr<I64ArrayAttr>:$static_strides,
        OptionalAttr<I64ArrayOfArraysAttr>:$explicit_output_shapes
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$result
    );

    let assemblyFormat = [{
        $source $static_offsets $static_sizes ($static_strides^)?
        attr-dict `:` type($source) `to` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }

        Byte getByteOffset();

        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                vpux::IE::TypeComparisonMode::STRICT_EQUAL,
                /*checkInferredDimsOrder=*/true,
                /*checkInferredMemSpace=*/true
            );
        }
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$static_sizes)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$static_offsets, "mlir::ArrayAttr":$static_sizes, "mlir::ArrayAttr":$static_strides)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$static_offsets, "vpux::ShapeRef":$static_sizes, "vpux::ShapeRef":$static_strides)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$static_offsets, "vpux::ArrayRef<int64_t>":$static_sizes, "vpux::ArrayRef<int64_t>":$static_strides)
        >
    ];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

//
// ConcatViewOp
//

def VPUIP_ConcatViewOp :
        VPUIP_Op<
            "ConcatView",
            [
                Pure,
                InferTypeOpInterface,
                ViewLikeOpInterface
            ]
        > {
    let summary = "VPUIP ConcatView layer. Dummy operation to maintain use-def chains.";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer, VPUIP_BoundedBuffer]>>:$inputs,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer, VPUIP_BoundedBuffer]>:$output_buff
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer, VPUIP_SparseBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        mlir::Value getViewSource() {
            return getOutputBuff();
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}

//
// PermuteCastOp
//

def VPUIP_PermuteCastOp :
        VPUIP_Op<
            "PermuteCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that changes the layout information of a buffer";

    let description = [{
        The op changes layout information in the following way:
            * dst_order: order of output buffer is set to value of this arg
            * mem_perm: describes the permutation applied on input buffer's memory shape
                        to obtain the memory shape of the output buffer.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$source,

        AffineMapAttr:$dst_order,
        AffineMapAttr:$mem_perm
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$result
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let hasFolder = 1;
}

//
// QuantizeCastOp
//

def VPUIP_QuantizeCastOp :
        VPUIP_Op<
            "QuantizeCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that changes the quantization information of a buffer";

    let arguments = (ins
        AnyTypeOf<[MemRefOf<[AnyInteger, quant_QuantizedType]>, VPUIP_DistributedBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[MemRefOf<[AnyInteger, quant_QuantizedType]>, VPUIP_DistributedBuffer]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// GenericReshape
//

def VPUIP_GenericReshapeOp :
        VPUIP_Op<
            "GenericReshape",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure,
                VPUIP_SameOperandsAndResultElementType
            ]
        > {
    let summary = "Generic Reshape layer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_SparseBuffer, VPUIP_DistributedBuffer]>:$output
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `->` type(results)
    }];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

//
// StaticAllocOp
//

def VPUIP_StaticAllocOp :
        VPUIP_Op<
            "StaticAlloc",
            [
                DeclarationOp,
                DotInterface,
                Pure
            ]
        > {
    let summary = "Static buffer allocation";

    let arguments = (ins
        IntAttr:$offset
    );

    let results = (outs
        AnyMemRef:$memory
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];

    let assemblyFormat = [{
        `<` $offset `>` attr-dict `->` type(results)
    }];
}

//
// CopyOp
//

def VPUIP_CopyOp :
        VPUIP_Op<
            "Copy",
            [
                ViewLikeOpInterface,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                MemoryEffectsOpInterface,
                DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>,
                DotInterface
            ]
        > {
    let summary = "Copy layer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_SparseBuffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_SparseBuffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output_buff
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_SparseBuffer, VPUIP_DistributedBuffer, VPUIP_BoundedBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::GREEN;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}

//
// TimestampOp
//

def VPUIP_TimestampOp :
        VPUIP_Op<
            "Timestamp",
            [
                ViewLikeOpInterface,
                InferTypeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                MemoryEffectsOpInterface,
                SameOperandsAndResultElementType,
                DotInterface
            ]
        > {
    let summary = "Get timer timestamp operation";

    let description = [{
        Get timer timestamp operation
    }];

    let arguments = (ins
        MemRefOf<[UI32, UI64]>:$output_buff
    );

    let results = (outs
        MemRefOf<[UI32, UI64]>:$output
    );

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::BLUE;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;

    let assemblyFormat = [{
        attr-dict
        `(` $output_buff `:` type($output_buff) `)`
        `->` type($output)
    }];
}

//
// ShapeCastOp
//

def VPUIP_ShapeCastOp :
        VPUIP_Op<
            "ShapeCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface>
            ]
        > {
    let summary = "Operation that changes the shape information of a buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$source,
        I64ArrayAttr:$shape,
        OptionalAttr<I64ArrayOfArraysAttr>:$explicit_output_shapes,
        OptionalAttr<I64ArrayOfArraysAttr>:$explicit_output_offsets
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$result
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ShapeRef":$shape)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "vpux::ArrayRef<int64_t>":$shape)
        >,
        OpBuilder<
            (ins "mlir::Value":$source, "mlir::ArrayAttr":$shape)
        >,
    ];
}

//
// GroupSparseBufferOp
//

def VPUIP_GroupSparseBufferOp :
        VPUIP_Op<
            "GroupSparseBuffer",
            [
                Pure,
                DotInterface,
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<GroupedViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Group sparsity data and metadata into a value";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$data,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$sparsityMap,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$storageElementTable,

        UnitAttr:$is_weights,
        OptionalAttr<VPUIP_SparsityCompressionAttr>:$sparsity_compression,

        OptionalAttr<VPU_SEAttr>:$seAttr
    );

    let results = (outs
        VPUIP_SparseBuffer:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$data,
                CArg<"bool", "{}">:$is_weights, CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap,
                CArg<"bool", "{}">:$is_weights, CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"bool", "{}">:$is_weights, CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"VPU::SEAttr", "{}">:$seAttr)
        >
    ];

    let assemblyFormat = [{
        `(` $data
            (`,` $sparsityMap^ `` custom<OptionalTypes>(type($sparsityMap)))?
            (`,` $storageElementTable^ `` custom<OptionalTypes>(type($storageElementTable)))?
        `)`
        attr-dict
        `` custom<OptionalTypes>(type($data))
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];

    let hasCanonicalizer = 1;
}

//
// UngroupSparseBufferOp
//

def VPUIP_UngroupSparseBufferOp :
        VPUIP_Op<
            "UngroupSparseBuffer",
            [
                Pure,
                DotInterface,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<MultiViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Ungroups sparse value into data and metadata buffers";

    let arguments = (ins
        VPUIP_SparseBuffer:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$data,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$sparsityMap,
        Optional<AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>>:$storageElementTable
    );

    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "mlir::Type":$data, "mlir::Type":$sparsityMap, "mlir::Type":$storageElementTable,
            "mlir::Value":$input
        )>,

        OpBuilder<(ins
            "mlir::Value":$input
        )>,
    ];

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        `` custom<OptionalTypes>(type($input))
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }];
}

//
// StorageElementTable
//

def VPUIP_StorageElementTableOp :
        VPUIP_Op<
            "StorageElementTable",
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface>,
            ]
        > {
    let summary = "Declares a Storage Element Pointers table";

    let description = [{
        A Storage Element represents a 1x1xN volume that contains sparse data, where N
        represents the number of channels stored. The Storage Element Table is comprised
        of pointers to these Storage Elements, which have the following structure:

        31-29 28                            9 8         0
        -------------------------------------------------
        | xx |           DATA_PTR            | BASE_PTR |
        -------------------------------------------------

        The DATA_PTR represents the offset to a Storage Element in relation to the start of
        the input data. BASE_PTR is used to decide what base address is added to DATA_PTR
        in order to find the location of the Storage Element in memory during inference.

        This operation represents the Storage Element Table in relation to the input data,
        on top of which transformations can be applied. This operation will later get
        converted to a constant, where the pointers are generated based on the information
        contained in this operation.

        The following information is contained:
        - dataShape, dataElemType, dataStrides: information about the input data that
          is associated with this Storage Element Table
        - seSize: the size of a Storage Element
        - seDepth: the number of Storage Elements per depth
        - seAttr: information on how the input data is transformed
        - basePtrs: base pointers associated with each Storage Element pointer
    }];

    let arguments = (ins
        I64ArrayAttr:$dataShape,
        TypeAttr:$dataElemType,
        IntAttr:$seSize,
        IntAttr:$seDepth,
        OptionalAttr<VPU_SEAttr>:$seAttr,
        OptionalAttr<I64ArrayAttr>:$dataStrides,
        OptionalAttr<I32ElementsAttr>:$basePtrs
    );

    let results = (outs
        MemRefOf<[I32]>:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let assemblyFormat = [{
         attr-dict `->` type(results)
    }];

    let builders = [
        OpBuilder<(ins
            CArg<"llvm::ArrayRef<int64_t>">:$dataShape,
            CArg<"mlir::Type">:$dataElemType,
            CArg<"int64_t">:$seSize,
            CArg<"int64_t">:$seDepth,
            CArg<"VPU::SEAttr">:$seAttr
        )>
    ];

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs,
                vpux::IE::TypeComparisonMode::STRICT_EQUAL,
                /*checkInferredDimsOrder=*/true,
                /*checkInferredMemSpace=*/true
            );
        }
    }];
}

//
// StubOp
//

def VPUIP_StubOp :
        VPUIP_Op<
            "Stub",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Substitute operation for stubbing.";

    let arguments = (ins
        Variadic<AnyMemRef>:$inputs
    );

    let results = (outs
        Variadic<AnyMemRef>:$outputs
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $inputs `:` type($inputs) `)`
        `->` type(results)
    }];
}

//
// DeformablePSROIPoolingUPAOp
//

def VPUIP_DeformablePSROIPoolingUPAOp :
        VPUIP_UPATaskOp<1, "DeformablePSROIPoolingUPA",
            [
                ViewLikeOpInterface
            ]
        > {
    let summary = "DeformablePSROIPooling UPA SHAVE kernel";

let arguments = (ins
        F16MemRef:$input_score_maps,
        F16MemRef:$input_rois,
        Optional<MemRefOf<[F16]>>:$input_transformations,
        F16MemRef:$output_buff,

        IntAttr:$output_dim,
        F64Attr:$spatial_scale,
        OptionalAttr<IntAttr>:$group_size,
        OptionalAttr<IntAttr>:$spatial_bins_x,
        OptionalAttr<IntAttr>:$spatial_bins_y,
        OptionalAttr<F64Attr>:$trans_std,
        OptionalAttr<IntAttr>:$part_size,
        OptionalAttr<IE_DeformablePSROIPoolingModeAttr>:$mode
    );

    let results = (outs
        F16MemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input_score_maps `:` type($input_score_maps) `,` $input_rois `:` type($input_rois) (`,` $input_transformations^ `:` type($input_transformations))? `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ViewOp
//

def VPUIP_ViewOp :
        VPUIP_Op<
            "ViewOp",
            [
                Pure,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>
            ]
        > {
    let summary = "Create a new view for buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$source
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$result
    );

    let assemblyFormat = [{
        $source
        attr-dict `:` type($source) `to` type(results)
    }];

}

//
// ExpandOp
//

def VPUIP_ExpandOp :
        VPUIP_Op<
            "Expand",
            [
                ViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPUIP_LayerOpInterface>,
                InferTypeOpInterface
            ]
        > {
    let summary = "Expand layer";

    let arguments = (ins
        AnyMemRef:$input,
        AnyMemRef:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end
    );

    let results = (outs
        AnyMemRef:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        vpux::DotNodeColor getNodeColor() {
            return vpux::DotNodeColor::ORANGE;
        }
    }] # VPUIP_InferLayerReturnTypes<1>.impl;
}

//
// ExpandDMAOp
//

def VPUIP_ExpandDMAOp :
        VPUIP_TaskOp<1, "ExpandDMA",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which expands tensor by copying values from original buffer";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer]>:$output_buff,

        I64ArrayAttr:$pads_begin,
        I64ArrayAttr:$pads_end,

        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_Buffer, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$pads_begin,
            "mlir::ArrayAttr":$pads_end, "VPUIP::DMADescriptorAttr":$dma_descriptor)
        >,
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$pads_begin,
            "mlir::ArrayAttr":$pads_end, "VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::IntegerAttr":$port)
        >
    ];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// UpsamplingDMAOp
//

def VPUIP_UpsamplingDMAOp :
        VPUIP_TaskOp<1, "UpsamplingDMAOp",
            [
                ViewLikeOpInterface,
                VPUIP_SameElementType,
                VPUIP_SameInOutDimsOrder,
                VPUIP_DMATypeOpInterface
            ]
        > {
    let summary = "A NNDMA task which performs a functionality equivalent DMA transfer to a UPA UpsamplingDMA task";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$input,
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output_buff,
        I64ArrayAttr:$upsampling_factor,

        OptionalAttr<VPUIP_DMADescriptorAttr>:$dma_descriptor,
        OptionalAttr<I64ArrayAttr>:$expand,
        OptionalAttr<IntAttr>:$port,
        UnitAttr:$is_out_of_order,
        UnitAttr:$is_critical,
        OptionalAttr<IntAttr>:$dma_hwp_id,
        OptionalAttr<VPUIP_DmaProfilingMetadataAttr>:$profilingMetadata
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$upsampling_factor, "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::ArrayAttr":$expand)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$upsampling_factor, "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::ArrayAttr":$expand, "int64_t":$port)
        >,
        OpBuilder<
            (ins "mlir::Value":$input, "mlir::Value":$output_buff, "mlir::ArrayAttr":$upsampling_factor, "vpux::VPUIP::DMADescriptorAttr":$dma_descriptor, "mlir::ArrayAttr":$expand, "int64_t":$port, "bool":$is_out_of_order, "bool":$is_critical, "mlir::IntegerAttr":$dma_hwp_id, "vpux::VPUIP::DmaProfilingMetadataAttr":$profilingMetadata)
        >,
    ];

    let extraClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::DMA_NN;
        }

        vpux::DotNodeColor getNodeColor() {
            const auto memKind = getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return memKind == vpux::VPU::MemoryKind::DDR ? vpux::DotNodeColor::RED : vpux::DotNodeColor::GREEN;
        }
    }] # commonExtraClassDeclaration;

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` type($input) `)`
        `outputs` `(` $output_buff `:` type($output_buff) `)`
        `->` type(results)
    }];
}

//
// ExtractFlatSlice
//

def VPUIP_ExtractFlatSliceOp :
        VPUIP_Op<
            "ExtractFlatSlice",
            [
                Pure,
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface>
            ]
        > {
    let summary = "Operation that extracts flat slice from DistributedBuffer";

    let description = [{
        This operation provides view to single slice from DistributedBuffer on the tiling dim.
        In contrast to SubView, this operation can take buffer slice from single cluster.
        For example, !VPUIP.DistributedBuffer<1x32x16x128xf16, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]>
        And offset is 20, which effectively means, that view will be memref<1x1x16x128xf16, [@CMX_NN, 2]>, because
        dims are distributed as 8+8+8+8 and 20 is 8+8+4, which means slice is located in 2 cluster
    }];

    let arguments = (ins
        AnyTypeOf<[AnyMemRef, VPUIP_DistributedBuffer]>:$source,
        IntAttr:$offset
    );

    let results = (outs
        AnyMemRef:$result
    );

    let hasVerifier = 1;

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $source `:` type($source) `)`
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        Byte getByteOffset();
    }];

}

//
// WorkloadCastOp
//

def VPUIP_WorkloadCastOp :
        VPUIP_Op<
            "WorkloadCast",
            [
                DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                Pure
            ]
        > {
    let summary = "Operation that casts one DistributedBuffer type to another.";

    let arguments = (ins
        AnyTypeOf<[VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPUIP_DistributedBuffer, VPUIP_SparseBuffer]>:$output
    );

    let assemblyFormat = [{
        attr-dict
        `inputs` `(` $input `:` qualified(type($input)) `)`
        `->` qualified(type(results))
    }];
}


//
// ProfilingSectionInfoOp
//

def VPUIP_ProfilingSectionOp :
        VPUIP_Op<
            "ProfilingSection",
            [
                IsolatedFromAbove,
                HasParent<"vpux::IE::DataInfoOp">
            ]
        > {
    let summary = "Information about memory offset and size of engine specific profiling section";

    let description = [{
        This operation is bound to `IE.DataInfo` operation and holds information about profiling section:
          * Engine name
          * Start offset(bytes)
          * Section size(bytes)
    }];

    let arguments = (ins
        IntAttr:$sectionType,
        IntAttr:$offset,
        IntAttr:$size
    );

    let assemblyFormat = [{
        `type` $sectionType `:` $size `bytes` `from` $offset
        attr-dict
    }];

    let hasVerifier = 0;
}

//
// GroupBoundedBufferOp
//
def VPUIP_GroupBoundedBufferOp :
        VPUIP_Op<"GroupBoundedBuffer",
            [
                Pure,
                DotInterface,
                DeclareOpInterfaceMethods<GroupedViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Creates a BoundedBuffer type by combining upper-bounded data buffer and a buffer with dynamic shape";

    let arguments = (ins
        AnyTypeOf<[AnyMemRef]>:$data,
        AnyTypeOf<[AnyMemRef]>:$dynamicShape
    );

    let results = (outs
        VPUIP_BoundedBuffer:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
    let hasCanonicalizer = 1;
}

//
// UngroupBoundedBufferOp
//

def VPUIP_UngroupBoundedBufferOp :
        VPUIP_Op<
            "UngroupBoundedBuffer",
            [
                Pure,
                DotInterface,
                DeclareOpInterfaceMethods<MultiViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
            ]
        > {
    let summary = "Unpacks BoundedBuffer type into upper-bounded buffer and its dynamic shape buffer";

    let arguments = (ins
        VPUIP_BoundedBuffer:$input
    );

    let results = (outs
        AnyTypeOf<[AnyMemRef]>:$data,
        AnyTypeOf<[AnyMemRef]>:$dynamicShape
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

#endif
