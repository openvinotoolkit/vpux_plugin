//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json

//
// This file generates a blob with erf activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<5x6x4xf16>
        DataInfo "input1" : tensor<5xsi32>
        DataInfo "input2" : tensor<5xsi32>
        DataInfo "input3" : tensor<5xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<7x6x4xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_EmbeddingSegmentsSum(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>,memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
        attributes {
            VPU.kernel_code = "embedding_segments_sum.cpp",
            VPU.kernel_entry = "embedding_segments_sum"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%1: memref<5x6x4xf16>, %2: memref<5xsi32>, %3: memref<5xsi32>, %4: memref<5xf16>, %5: memref<7x6x4xf16>) -> memref<7x6x4xf16> {

    %cst = const.Declare memref<5xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 1.000000e+01]> : tensor<5xf16>
    %cst_0 = const.Declare memref<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    %cst_1 = const.Declare memref<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    %in0= VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<5x6x4xf16, [@CMX_NN, 0]>
    %in1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<5xsi32, [@CMX_NN, 0]>
    %in2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<5xsi32, [@CMX_NN, 0]>
    %in3 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<5xf16, [@CMX_NN, 0]>
    %out0 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<7x6x4xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<5x6x4xf16>) outputs(%in0 : memref<5x6x4xf16, [@CMX_NN, 0]>) -> memref<5x6x4xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<5xsi32>) outputs(%in1 : memref<5xsi32, [@CMX_NN, 0]>) -> memref<5xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%b0 : !VPURT.Barrier)  {
        %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<5xsi32>) outputs(%in2 : memref<5xsi32, [@CMX_NN, 0]>) -> memref<5xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%b0 : !VPURT.Barrier)  {
        %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<5xf16>) outputs(%in3 : memref<5xf16, [@CMX_NN, 0]>) -> memref<5xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_EmbeddingSegmentsSum           // The reference to the Kernel function.
                    inputs(%in0 as %arg0: memref<5x6x4xf16, [@CMX_NN, 0]>, %in1 as %arg1: memref<5xsi32, [@CMX_NN, 0]>, %in2 as %arg2: memref<5xsi32, [@CMX_NN, 0]>, %in3 as %arg3: memref<5xf16, [@CMX_NN, 0]>)
                    outputs(%out0 as %arg4: memref<7x6x4xf16, [@CMX_NN, 0]>)   //
                    on tile 0                           // The tile index to execute on.
        -> memref<7x6x4xf16, [@CMX_NN, 0]> {
                VPUIP.SW.Kernel.run {attrs = [7 : i32, 0 : i32]}(%arg0, %arg1, %arg2,%arg3,%arg4)
                    : memref<5x6x4xf16, [@CMX_NN, 0]>
                    , memref<5xsi32, [@CMX_NN, 0]>
                    , memref<5xsi32, [@CMX_NN, 0]>
                    , memref<5xf16, [@CMX_NN, 0]>
                    , memref<7x6x4xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier)  {
        %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%out0 : memref<7x6x4xf16, [@CMX_NN, 0]>) outputs(%5 : memref<7x6x4xf16>) -> memref<7x6x4xf16>
    }
    return %5 : memref<7x6x4xf16>
}

}

// CHECK:    identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "input0",
// CHECK:        dimensions: [
// CHECK:          5,
// CHECK:          6,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          384,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input1",
// CHECK:        dimensions: [
// CHECK:          5
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "I32",
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input2",
// CHECK:        dimensions: [
// CHECK:          5
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "I32",
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input3",
// CHECK:        dimensions: [
// CHECK:          5
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "output",
// CHECK:        dimensions: [
// CHECK:          7,
// CHECK:          6,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    task_count: 8,

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "input0",
// CHECK:        dimensions: [
// CHECK:          5,
// CHECK:          6,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          384,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input1",
// CHECK:        dimensions: [
// CHECK:          5
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "I32",
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input2",
// CHECK:        dimensions: [
// CHECK:          5
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "I32",
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input3",
// CHECK:        dimensions: [
// CHECK:          5
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "output",
// CHECK:        dimensions: [
// CHECK:          7,
// CHECK:          6,
// CHECK:          4
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          384,
// CHECK:          64,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ]
