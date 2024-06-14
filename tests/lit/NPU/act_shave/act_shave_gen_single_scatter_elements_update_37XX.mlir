//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %s
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX37XX
//
// This file generates a blob with scatterupdate activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "Parameter_199" : tensor<2x3x4xf16>
        IE.DataInfo "Parameter_200" : tensor<1x3x1xf16>
    }
    outputsInfo : {
        IE.DataInfo "ScatterElementsUpdate_203" : tensor<2x3x4xf16>
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
    func.func private @builtin_ScatterElementsUpdate(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64)
        attributes {
            VPU.kernel_code = "scatter_elements_update.cpp",
            VPU.kernel_entry = "scatter_elements_update"
        }
    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%arg0: memref<2x3x4xf16>, %arg1: memref<1x3x1xf16>, %arg2: memref<2x3x4xf16>) -> memref<2x3x4xf16> {

    %cst = const.Declare memref<1x3x1xsi32> = dense<[[[1], [0], [1]]]> : tensor<1x3x1xsi32>
    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x3x4xf16, [@CMX_NN, 0]>
    %in_tile1_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x3x1xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<2x3x4xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<2x3x4xf16>) outputs(%in_tile0_cmx : memref<2x3x4xf16, [@CMX_NN, 0]>) -> memref<2x3x4xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x3x1xf16>) outputs(%in_tile1_cmx : memref<1x3x1xf16, [@CMX_NN, 0]>) -> memref<1x3x1xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_ScatterElementsUpdate            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %0: memref<2x3x4xf16, [@CMX_NN, 0]>, %in_tile1_cmx as %1: memref<1x3x1xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %2: memref<2x3x4xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<2x3x4xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [1]}(%0, %1, %2)
                    : memref<2x3x4xf16, [@CMX_NN, 0]>
                    , memref<1x3x1xf16, [@CMX_NN, 0]>
                    , memref<2x3x4xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<2x3x4xf16, [@CMX_NN, 0]>) outputs(%arg2 : memref<2x3x4xf16>) -> memref<2x3x4xf16>
    }
    return %arg2: memref<2x3x4xf16>

}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_199",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           3,
// CHECK:           4
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           192,
// CHECK:           64,
// CHECK:           16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Parameter_200",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           3,
// CHECK:           1
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           48,
// CHECK:           16,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "ScatterElementsUpdate_203",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           3,
// CHECK:           4
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           192,
// CHECK:           64,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 6,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_199",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           3,
// CHECK:           4
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           192,
// CHECK:           64,
// CHECK:           16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Parameter_200",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           3,
// CHECK:           1
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           48,
// CHECK:           16,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "ScatterElementsUpdate_203",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           3,
// CHECK:           4
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           192,
// CHECK:           64,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
