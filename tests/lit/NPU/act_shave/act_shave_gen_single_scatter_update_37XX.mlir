//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX37XX
//
// This file generates a blob with scatterupdate activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<10x9x10x9xf16>
        IE.DataInfo "updates" : tensor<10x9x4x2x9xf16>
    }
    outputsInfo : {
        IE.DataInfo "scatter_update" : tensor<10x9x10x9xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096,  // Size in bytes for the actSHAVE0 in the first tile.
        4096,  // Size in bytes for the actSHAVE1 in the first tile.
        4096,  // Size in bytes for the actSHAVE2 in the second tile.
        4096   // Size in bytes for the actSHAVE3 in the second tile.
    ]

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_ScatterUpdate(%inputs : memref<*xf16>, %indices : memref<*xsi32>, %updates: memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "scatter_update.cpp",
            VPU.kernel_entry = "scatter_update"
        }
    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%0: memref<10x9x10x9xf16>, %1: memref<10x9x4x2x9xf16>, %2: memref<10x9x10x9xf16>) -> memref<10x9x10x9xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<10x9x10x9xf16, [@CMX_NN, 0]>
    %in_tile1_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <12992> -> memref<10x9x4x2x9xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <16256> -> memref<10x9x10x9xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<10x9x10x9xf16>) outputs(%in_tile0_cmx : memref<10x9x10x9xf16, [@CMX_NN, 0]>) -> memref<10x9x10x9xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<10x9x4x2x9xf16>) outputs(%in_tile1_cmx : memref<10x9x4x2x9xf16, [@CMX_NN, 0]>) -> memref<10x9x4x2x9xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b1  : !VPURT.Barrier) updates(%b2  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_ScatterUpdate            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<10x9x10x9xf16, [@CMX_NN, 0]>, %in_tile1_cmx as %arg1: memref<10x9x4x2x9xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg2: memref<10x9x10x9xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<10x9x10x9xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [3]}(%arg0, %arg1, %arg2)
                    : memref<10x9x10x9xf16, [@CMX_NN, 0]>
                    , memref<10x9x4x2x9xf16, [@CMX_NN, 0]>
                    , memref<10x9x10x9xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<10x9x10x9xf16, [@CMX_NN, 0]>) outputs(%2 : memref<10x9x10x9xf16>) -> memref<10x9x10x9xf16>
    }
    return %2: memref<10x9x10x9xf16>

}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           10,
// CHECK:           9,
// CHECK:           10,
// CHECK:           9
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           12960,
// CHECK:           1440,
// CHECK:           144,
// CHECK:           16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "updates",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         9,
// CHECK:         4,
// CHECK:         2,
// CHECK:         9
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           10368,
// CHECK:           1152,
// CHECK:           288,
// CHECK:           144,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "scatter_update",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         9,
// CHECK:         10,
// CHECK:         9
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         12960,
// CHECK:         1440,
// CHECK:         144,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 7,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         9,
// CHECK:         10,
// CHECK:         9
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           12960,
// CHECK:           1440,
// CHECK:           144,
// CHECK:           16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "updates",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         9,
// CHECK:         4,
// CHECK:         2,
// CHECK:         9
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           10368,
// CHECK:           1152,
// CHECK:           288,
// CHECK:           144,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "scatter_update",
// CHECK:       dimensions: [
// CHECK:         10,
// CHECK:         9,
// CHECK:         10,
// CHECK:         9
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           12960,
// CHECK:           1440,
// CHECK:           144,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
