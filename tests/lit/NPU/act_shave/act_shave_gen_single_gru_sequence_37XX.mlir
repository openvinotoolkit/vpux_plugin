//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-NPU37XX
//
// This file generates a blob with GRUCell activation shave
// demonstrate that the runtime cannot handle this. It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
            DataInfo "Parameter_187" : tensor<2x1x10xf16>
            DataInfo "Parameter_188" : tensor<2x1x4xf16>
        }
        outputsInfo : {
            DataInfo "GRUSequence_193.0" : tensor<2x1x1x4xf16>
            DataInfo "GRUSequence_193.1" : tensor<2x1x4xf16>
        }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_GRUSequence(memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, memref<*xf16>, i64, i64, i64, i64, f64)
        attributes {
            VPU.kernel_code = "gru_sequence.cpp",
            VPU.kernel_entry = "gru_sequence"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%arg0: memref<2x1x10xf16, @DDR>, %arg1: memref<2x1x4xf16, @DDR>, %arg2: memref<2x1x1x4xf16, @DDR>, %arg3: memref<2x1x4xf16, @DDR>) -> (memref<2x1x1x4xf16, @DDR>, memref<2x1x4xf16, @DDR>) {

    %cst = const.Declare memref<1x12x10xf16> = dense<1.0> : tensor<1x12x10xf16>
    %cst_0 = const.Declare memref<1x12x4xf16> = dense<1.0> : tensor<1x12x4xf16>
    %cst_1 = const.Declare memref<1x16xf16> = dense<1.0> : tensor<1x16xf16>

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<2x1x10xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<2x1x4xf16, @DDR>
    %4 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<2x1x1x4xf16, @DDR>
    %5 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<2x1x4xf16, @DDR>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <384> -> memref<2x1x10xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<2x1x4xf16, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x10xf16, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x12x4xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <448> -> memref<1x16xf16, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x1x1x4xf16, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<2x1x4xf16, [@CMX_NN, 0]>

    VPURT.Task attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<2x1x10xf16, @DDR>) outputs(%6 : memref<2x1x10xf16, [@CMX_NN, 0]>) -> memref<2x1x10xf16, [@CMX_NN, 0]>
    }
    VPURT.Task attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst : memref<1x12x10xf16>) outputs(%8 : memref<1x12x10xf16, [@CMX_NN, 0]>) -> memref<1x12x10xf16, [@CMX_NN, 0]>
    }
    VPURT.Task attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_1 : memref<1x16xf16>) outputs(%10 : memref<1x16xf16, [@CMX_NN, 0]>) -> memref<1x16xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 1 : i64} inputs(%3 : memref<2x1x4xf16, @DDR>) outputs(%7 : memref<2x1x4xf16, [@CMX_NN, 0]>) -> memref<2x1x4xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x12x4xf16>) outputs(%9 : memref<1x12x4xf16, [@CMX_NN, 0]>) -> memref<1x12x4xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
                @VPU.SW::@builtin_GRUSequence
                inputs(%6 as %arg4: memref<2x1x10xf16, [@CMX_NN, 0]>, %7 as %arg5: memref<2x1x4xf16, [@CMX_NN, 0]>, %8 as %arg6: memref<1x12x10xf16, [@CMX_NN, 0]>, %9 as %arg7: memref<1x12x4xf16, [@CMX_NN, 0]>, %10 as %arg8: memref<1x16xf16, [@CMX_NN, 0]>)
                outputs(%11 as %arg9: memref<2x1x1x4xf16, [@CMX_NN, 0]>, %12 as %arg10: memref<2x1x4xf16, [@CMX_NN, 0]>)
                on tile 0

        -> (memref<2x1x1x4xf16, [@CMX_NN, 0]>, memref<2x1x4xf16, [@CMX_NN, 0]>){
                VPUIP.SW.Kernel.run {attrs = [4, 0, 1, 1, 0.000000e+00]}(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10)
                : memref<2x1x10xf16, [@CMX_NN, 0]>
                , memref<2x1x4xf16, [@CMX_NN, 0]>
                , memref<1x12x10xf16, [@CMX_NN, 0]>
                , memref<1x12x4xf16, [@CMX_NN, 0]>
                , memref<1x16xf16, [@CMX_NN, 0]>
                , memref<2x1x1x4xf16, [@CMX_NN, 0]>
                , memref<2x1x4xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 0 : i64} inputs(%12 : memref<2x1x4xf16, [@CMX_NN, 0]>) outputs(%5 : memref<2x1x4xf16, @DDR>) -> memref<2x1x4xf16, @DDR>
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %13 = VPUIP.NNDMA {port = 1 : i64} inputs(%11 : memref<2x1x1x4xf16, [@CMX_NN, 0]>) outputs(%4 : memref<2x1x1x4xf16, @DDR>) -> memref<2x1x1x4xf16, @DDR>
    }
    return %arg2, %arg3 : memref<2x1x1x4xf16, @DDR>, memref<2x1x4xf16, @DDR>
}

}

// CHECK:   identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "Parameter_187",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          160,
// CHECK:          160,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "Parameter_188",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          64,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "GRUSequence_193.0",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          1,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          64,
// CHECK:          64,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "GRUSequence_193.1",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          64,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:   task_count: 10,

// CHECK:   options: [
// CHECK:   ],

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "Parameter_187",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          160,
// CHECK:          160,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "Parameter_188",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          64,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "GRUSequence_193.0",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          1,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          64,
// CHECK:          64,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "GRUSequence_193.1",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          1,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          64,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
