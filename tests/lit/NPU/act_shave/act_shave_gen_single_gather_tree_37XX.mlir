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
// This file generates a blob with gather tree shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "step_ids" : tensor<10x1x10xf32>
        IE.DataInfo "parent_ids" : tensor<10x1x10xf32>
        IE.DataInfo "max_seq_len" : tensor<1xf32>
        IE.DataInfo "end_token" : tensor<1xf32>
    }
    outputsInfo : {
        IE.DataInfo "final_ids" : tensor<10x1x10xf32>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_gather_tree(%input : memref<*xf32>, %input : memref<*xf32>, %input : memref<*xf32>, %input : memref<*xf32>, %output : memref<*xf32>)
        attributes {
            VPU.kernel_code = "gather_tree.cpp",
            VPU.kernel_entry = "gather_tree"
        }
}

func.func @main(%0: memref<10x1x10xf32>, %1: memref<10x1x10xf32>, %2: memref<1xf32>, %3: memref<1xf32>, %4: memref<10x1x10xf32>) -> memref<10x1x10xf32> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<10x1x10xf32, [@CMX_NN, 0]>
    %in_tile1_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <400> -> memref<10x1x10xf32, [@CMX_NN, 0]>
    %in_tile2_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <800> -> memref<1xf32, [@CMX_NN, 0]>
    %in_tile3_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <1200> -> memref<1xf32, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<10x1x10xf32, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    %b4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<10x1x10xf32>) outputs(%in_tile0_cmx : memref<10x1x10xf32, [@CMX_NN, 0]>) -> memref<10x1x10xf32, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<10x1x10xf32>) outputs(%in_tile1_cmx : memref<10x1x10xf32, [@CMX_NN, 0]>) -> memref<10x1x10xf32, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1xf32>) outputs(%in_tile2_cmx : memref<1xf32, [@CMX_NN, 0]>) -> memref<1xf32, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b3 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1xf32>) outputs(%in_tile3_cmx : memref<1xf32, [@CMX_NN, 0]>) -> memref<1xf32, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b4  : !VPURT.Barrier) updates(%b3  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_gather_tree            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<10x1x10xf32, [@CMX_NN, 0]>, %in_tile1_cmx as %arg1: memref<10x1x10xf32, [@CMX_NN, 0]>, %in_tile2_cmx as %arg2: memref<1xf32, [@CMX_NN, 0]>, %in_tile3_cmx as %arg3: memref<1xf32, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg4: memref<10x1x10xf32, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<10x1x10xf32, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run (%arg0, %arg1, %arg2, %arg3, %arg4)
                    : memref<10x1x10xf32, [@CMX_NN, 0]>
                    , memref<10x1x10xf32, [@CMX_NN, 0]>
                    , memref<1xf32, [@CMX_NN, 0]>
                    , memref<1xf32, [@CMX_NN, 0]>
                    , memref<10x1x10xf32, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b4 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<10x1x10xf32, [@CMX_NN, 0]>) outputs(%4 : memref<10x1x10xf32>) -> memref<10x1x10xf32>
    }
    return %4: memref<10x1x10xf32>

}


}

// CHECK:    identifier: "Test",
// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "step_ids",
// CHECK:        dimensions: [
// CHECK:          10,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          320,
// CHECK:          320,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "parent_ids",
// CHECK:        dimensions: [
// CHECK:          10,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          320,
// CHECK:          320,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "max_seq_len",
// CHECK:        dimensions: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          2
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 1,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "end_token",
// CHECK:        dimensions: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          3
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 1,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "final_ids",
// CHECK:        dimensions: [
// CHECK:          10,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          320,
// CHECK:          320,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
// CHECK:    task_count: 11,
// CHECK:    options: [

// CHECK:    ],

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "step_ids",
// CHECK:        dimensions: [
// CHECK:          10,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          320,
// CHECK:          320,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "parent_ids",
// CHECK:        dimensions: [
// CHECK:          10,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          320,
// CHECK:          320,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "max_seq_len",
// CHECK:        dimensions: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          2
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 1,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "end_token",
// CHECK:        dimensions: [
// CHECK:          1
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        locale_index: [
// CHECK:          3
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 1,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "final_ids",
// CHECK:        dimensions: [
// CHECK:          10,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        locale_index: [
// CHECK:          0
// CHECK:        ],
// CHECK:        data_dtype: "FP32",
// CHECK:        quant_zero: [
// CHECK:          0
// CHECK:        ],
// CHECK:        quant_mult: [
// CHECK:          1
// CHECK:        ],
// CHECK:        quant_shift: [
// CHECK:          0
// CHECK:        ],
// CHECK:        order: 291,
// CHECK:        base_ptrs: [

// CHECK:        ],
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          320,
// CHECK:          320,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
