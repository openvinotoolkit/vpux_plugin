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
// This file generates a blob with erf activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "Parameter_187" : tensor<5x10xf16>
    }
    outputsInfo : {
        DataInfo "EmbeddingBagPackedSum_190" : tensor<3x10xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_EmbeddingBagPackedSum(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
        attributes {
            VPU.kernel_code = "embedding_bag_packed_sum.cpp",
            VPU.kernel_entry = "embedding_bag_packed_sum"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%arg0: memref<5x10xf16, @DDR>, %arg1: memref<3x10xf16, @DDR>) -> memref<3x10xf16, @DDR> {

    %cst = const.Declare memref<3x2xsi32> = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    %cst_0 = const.Declare memref<3x2xf16> = dense<1.000000e+00> : tensor<3x2xf16>

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<5x10xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<3x10xf16, @DDR>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<5x10xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<3x2xsi32, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<3x2xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<3x10xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<5x10xf16, @DDR>) outputs(%4 : memref<5x10xf16, [@CMX_NN, 0]>) -> memref<5x10xf16, [@CMX_NN, 0]>
    }
    VPURT.Task attributes {isTrailingSWLayer = false} {
        %8 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_0 : memref<3x2xf16>) outputs(%6 : memref<3x2xf16, [@CMX_NN, 0]>) -> memref<3x2xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %8 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst : memref<3x2xsi32>) outputs(%5 : memref<3x2xsi32, [@CMX_NN, 0]>) -> memref<3x2xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_EmbeddingBagPackedSum
                inputs(%4 as %arg2: memref<5x10xf16, [@CMX_NN, 0]>, %5 as %arg3: memref<3x2xsi32, [@CMX_NN, 0]>, %6 as %arg4: memref<3x2xf16, [@CMX_NN, 0]>)
                outputs(%7 as %arg5: memref<3x10xf16, [@CMX_NN, 0]>)
                on tile 0

        -> memref<3x10xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4, %arg5)
            : memref<5x10xf16, [@CMX_NN, 0]>
            , memref<3x2xsi32, [@CMX_NN, 0]>
            , memref<3x2xf16
            , [@CMX_NN, 0]>
            , memref<3x10xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<3x10xf16, [@CMX_NN, 0]>) outputs(%3 : memref<3x10xf16, @DDR>) -> memref<3x10xf16, @DDR>
    }
    return %arg1 : memref<3x10xf16, @DDR>
}

}

// CHECK:    identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "Parameter_187",
// CHECK:        dimensions: [
// CHECK:          5,
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
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "EmbeddingBagPackedSum_190",
// CHECK:        dimensions: [
// CHECK:          3,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          160,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    task_count: 7,

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "Parameter_187",
// CHECK:        dimensions: [
// CHECK:          5,
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
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "EmbeddingBagPackedSum_190",
// CHECK:        dimensions: [
// CHECK:          3,
// CHECK:          10
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          160,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
