//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-NPU37XX
//
// This file generates a blob with reverse_sequence activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<3x10xf32>
        IE.DataInfo "seq_lengths" : tensor<3xsi32>
    }
    outputsInfo : {
        IE.DataInfo "reverse_sequence" : tensor<3x10xf32>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_reversesequence(%input : memref<*xf32>, %input : memref<*xsi32>, %output : memref<*xf32>, %batch_axis : i64, %seq_axis : i64)
        attributes {
            VPU.kernel_code = "reverse_sequence.cpp",
            VPU.kernel_entry = "reverse_sequence"
        }
}

func.func @main(%0: memref<3x10xf32>, %1: memref<3xsi32>, %2: memref<3x10xf32>) -> memref<3x10xf32> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3x10xf32, [@CMX_NN, 0]>
    %in_tile1_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<3xsi32, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<3x10xf32, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<3x10xf32>) outputs(%in_tile0_cmx : memref<3x10xf32, [@CMX_NN, 0]>) -> memref<3x10xf32, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<3xsi32>) outputs(%in_tile1_cmx : memref<3xsi32, [@CMX_NN, 0]>) -> memref<3xsi32, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b1  : !VPURT.Barrier) updates(%b2  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_reversesequence            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<3x10xf32, [@CMX_NN, 0]>, %in_tile1_cmx as %arg1: memref<3xsi32, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg2: memref<3x10xf32, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<3x10xf32, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0, 1]}(%arg0, %arg1, %arg2)
                    : memref<3x10xf32, [@CMX_NN, 0]>
                    , memref<3xsi32, [@CMX_NN, 0]>
                    , memref<3x10xf32, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<3x10xf32, [@CMX_NN, 0]>) outputs(%2 : memref<3x10xf32>) -> memref<3x10xf32>
    }
    return %2: memref<3x10xf32>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           3,
// CHECK:           10
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:           32,
// CHECK:           320,
// CHECK:           32
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "seq_lengths",
// CHECK:       dimensions: [
// CHECK:           3
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I32",
// CHECK:       bit_strides: [
// CHECK:           32,
// CHECK:           32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "reverse_sequence",
// CHECK:       dimensions: [
// CHECK:           3,
// CHECK:           10
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:           32,
// CHECK:           320,
// CHECK:           32
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
// CHECK:         3,
// CHECK:         10
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         320,
// CHECK:         32
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "seq_lengths",
// CHECK:       dimensions: [
// CHECK:         3
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "reverse_sequence",
// CHECK:       dimensions: [
// CHECK:         3,
// CHECK:         10
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         320,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
