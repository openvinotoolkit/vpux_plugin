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
            DataInfo "Parameter_199" : tensor<1x157x512xf16>
            DataInfo "Parameter_200" : tensor<1x1x384xf16>
        }
        outputsInfo : {
            DataInfo "GRUSequence_205.0" : tensor<1x1x157x384xf16>
            DataInfo "GRUSequence_205.1" : tensor<1x1x384xf16>
        }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_GRUSequenceLastPart(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, f64)
        attributes {
            VPU.kernel_code = "gru_sequence_last_part.cpp",
            VPU.kernel_entry = "gru_sequence_last_part"
        }
    func.func private @builtin_GRUSequenceFirstPart(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, f64)
        attributes {
            VPU.kernel_code = "gru_sequence_first_part.cpp",
            VPU.kernel_entry = "gru_sequence_first_part"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%arg0: memref<1x157x512xf16, @DDR>, %arg1: memref<1x1x384xf16, @DDR>, %arg2: memref<1x1x157x384xf16, @DDR>, %arg3: memref<1x1x384xf16, @DDR>) -> (memref<1x1x157x384xf16, @DDR>, memref<1x1x384xf16, @DDR>) {
    %cst = const.Declare memref<1x1152x512xf16> = dense<1.0> : tensor<1x1152x512xf16>
    %cst_0 = const.Declare memref<1x1152x384xf16> = dense<1.0> : tensor<1x1152x384xf16>
    %cst_1 = const.Declare memref<1x1536xf16> = dense<1.0> : tensor<1x1536xf16>
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <1179648> -> memref<1x157x512xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1152x512xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <1340416> -> memref<1x1x157x1152xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1008384> -> memref<1x1x384xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1152x384xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <1005312> -> memref<1x1536xf16, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <884736> -> memref<1x1x157x384xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <1009152> -> memref<1x1x384xf16, [@CMX_NN, 0]>
    %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x157x512xf16, @DDR>) outputs(%0 : memref<1x157x512xf16, [@CMX_NN, 0]>) -> memref<1x157x512xf16, [@CMX_NN, 0]>
    }
    %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst : memref<1x1152x512xf16>) outputs(%1 : memref<1x1152x512xf16, [@CMX_NN, 0]>) -> memref<1x1152x512xf16, [@CMX_NN, 0]>
    }
    %10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%8, %9 : !VPURT.Barrier, !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                @VPU.SW::@builtin_GRUSequenceFirstPart
                inputs(%0 as %arg4: memref<1x157x512xf16, [@CMX_NN, 0]>, %1 as %arg5: memref<1x1152x512xf16, [@CMX_NN, 0]>)
                outputs(%2 as %arg6: memref<1x1x157x1152xf16, [@CMX_NN, 0]>)
                on tile 0
        -> memref<1x1x157x1152xf16, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run {attrs = [384, 157, 0.000000e+00]}(%arg4, %arg5, %arg6)
                : memref<1x157x512xf16, [@CMX_NN, 0]>
                , memref<1x1152x512xf16, [@CMX_NN, 0]>
                , memref<1x1x157x1152xf16, [@CMX_NN, 0]>
      }
    }
    %11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg1 : memref<1x1x384xf16, @DDR>) outputs(%3 : memref<1x1x384xf16, [@CMX_NN, 0]>) -> memref<1x1x384xf16, [@CMX_NN, 0]>
    }
    %12 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1152x384xf16>) outputs(%4 : memref<1x1152x384xf16, [@CMX_NN, 0]>) -> memref<1x1152x384xf16, [@CMX_NN, 0]>
    }
    %13 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_1 : memref<1x1536xf16>) outputs(%5 : memref<1x1536xf16, [@CMX_NN, 0]>) -> memref<1x1536xf16, [@CMX_NN, 0]>
    }
    %14 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%11, %12, %13 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
                @VPU.SW::@builtin_GRUSequenceLastPart
                inputs(%2 as %arg4: memref<1x1x157x1152xf16, [@CMX_NN, 0]>, %3 as %arg5: memref<1x1x384xf16, [@CMX_NN, 0]>, %4 as %arg6: memref<1x1152x384xf16, [@CMX_NN, 0]>, %5 as %arg7: memref<1x1536xf16, [@CMX_NN, 0]>)
                outputs(%6 as %arg8: memref<1x1x157x384xf16, [@CMX_NN, 0]>, %7 as %arg9: memref<1x1x384xf16, [@CMX_NN, 0]>)
                on tile 0
        -> (memref<1x1x157x384xf16, [@CMX_NN, 0]>, memref<1x1x384xf16, [@CMX_NN, 0]>){
                VPUIP.SW.Kernel.run {attrs = [384, 0, 157, 1, 0.000000e+00]}(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9)
                : memref<1x1x157x1152xf16, [@CMX_NN, 0]>
                , memref<1x1x384xf16, [@CMX_NN, 0]>
                , memref<1x1152x384xf16, [@CMX_NN, 0]>
                , memref<1x1536xf16, [@CMX_NN, 0]>
                , memref<1x1x157x384xf16, [@CMX_NN, 0]>
                , memref<1x1x384xf16, [@CMX_NN, 0]>
      }
    }
    VPURT.Task waits(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<1x1x384xf16, [@CMX_NN, 0]>) outputs(%arg3 : memref<1x1x384xf16, @DDR>) -> memref<1x1x384xf16, @DDR>
    }
    VPURT.Task waits(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %15 = VPUIP.NNDMA {port = 1 : i64} inputs(%6 : memref<1x1x157x384xf16, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x1x157x384xf16, @DDR>) -> memref<1x1x157x384xf16, @DDR>
    }
    return %arg2, %arg3 : memref<1x1x157x384xf16, @DDR>, memref<1x1x384xf16, @DDR>
}

}

// CHECK:   identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "Parameter_199",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          157,
// CHECK:          512
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          1286144,
// CHECK:          8192,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "Parameter_200",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          384
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          6144,
// CHECK:          6144,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "GRUSequence_205.0",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          157,
// CHECK:          384
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          964608,
// CHECK:          964608,
// CHECK:          6144,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "GRUSequence_205.1",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          384
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          6144,
// CHECK:          6144,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:   task_count: 9,

// CHECK:   options: [
// CHECK:   ],

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "Parameter_199",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          157,
// CHECK:          512
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          1286144,
// CHECK:          8192,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "Parameter_200",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          384
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          6144,
// CHECK:          6144,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "GRUSequence_205.0",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          157,
// CHECK:          384
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          964608,
// CHECK:          964608,
// CHECK:          6144,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "GRUSequence_205.1",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          384
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          6144,
// CHECK:          6144,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],
