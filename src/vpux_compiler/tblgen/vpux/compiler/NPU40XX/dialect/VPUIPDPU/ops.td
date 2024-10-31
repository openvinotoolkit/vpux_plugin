//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPDPU/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"

//
// Base classes
//

class VPUIPDPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIPDPU_Dialect,
            mnemonic,
            !listconcat(
                traits,
                [
                    DefinedInArch<"vpux::VPU::ArchKind::NPU40XX">
                ]
            )
        >;

//
// ODUCmxPortsOp
//

def VPUIPDPU_ODUCmxPortsOp :
        VPUIPDPU_Op<"ODUCmxPorts",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU CMX ports config.";

    let arguments = (ins
        VPUIPDPU_ODUCmxPorts:$cmx_ports
    );

    let assemblyFormat = [{
        `cmx_ports` `(` $cmx_ports `)`
        attr-dict
    }];
}

//
// ODUWriteCombineBufferOp
//

def VPUIPDPU_ODUWriteCombineBufferOp :
        VPUIPDPU_Op<"ODUWriteCombineBuffer",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU Write Combine Buffer";

    let arguments = (ins
        VPUIPDPU_ODUWcbCombineMode:$activations_mode,
        OptionalAttr<VPUIPDPU_ODUWcbCombineMode>:$sparsity_mode
    );

    let assemblyFormat = [{
        `activations_mode` `(` $activations_mode `)`
        (`sparsity_mode` `(` $sparsity_mode^ `)`)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//
// ODUHaloCfg
//

def VPUIPDPU_ODUHaloCfgOp :
        VPUIPDPU_Op<"ODUHaloCfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariantOp">,
                NoTerminator
            ]
        > {
    let summary = "ODU Halo configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict-with-keyword
        $region
    }];

    let hasVerifier = 1;
}

//
// ODUHaloRegionOp
//

def VPUIPDPU_ODUHaloRegionOp :
        VPUIPDPU_Op<"ODUHaloRegion",
            [
                HasParent<"vpux::VPUIPDPU::ODUHaloCfgOp">
            ]
        > {
    let summary = "ODU define halo region configuration";

    let arguments = (ins
        IntAttr:$begin_coord_x,
        IntAttr:$begin_coord_y,
        IntAttr:$end_coord_x,
        IntAttr:$end_coord_y,
        IntAttr:$activations_offset,
        OptionalAttr<IntAttr>:$sparsity_offset,
        IntAttr:$target_width,
        VPUIPDPU_DPUTiles:$cast_to_tile
    );

    let assemblyFormat = [{
        `begin_coord_x` `(` $begin_coord_x `)`
        `begin_coord_y` `(` $begin_coord_y `)`
        `end_coord_x` `(` $end_coord_x `)`
        `end_coord_y` `(` $end_coord_y `)`
        `activations_offset` `(` $activations_offset `)`
        (`sparsity_offset` `(` $sparsity_offset^ `)`)?
        `target_width` `(` $target_width `)`
        `cast_to_tile` `(` $cast_to_tile `)`
        attr-dict
    }];
}

#endif
