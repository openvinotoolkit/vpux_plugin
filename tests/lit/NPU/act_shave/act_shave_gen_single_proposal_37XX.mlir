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
// This file generates a blob with proposal activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "a_scores" : tensor<1x12x8x8xf16>
        DataInfo "b_boxes" : tensor<1x24x8x8xf16>
    }
    outputsInfo : {
        DataInfo "Proposal_202.0" : tensor<300x5xf16>
        DataInfo "Proposal_202.1" : tensor<300xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_Proposal(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, f64, i64, i64, none, none, i64, i64, i64, f64, f64, i64)
        attributes {
            VPU.kernel_code = "proposal.cpp",
            VPU.kernel_entry = "proposal"
        }
}

func.func @main(%arg0: memref<1x12x8x8xf16, @DDR>, %arg1: memref<1x24x8x8xf16, @DDR>, %arg2: memref<300x5xf16, @DDR>, %arg3: memref<300xf16, @DDR>) -> (memref<300x5xf16, @DDR>, memref<300xf16, @DDR>) {

    %cst = const.Declare memref<3xf16> = dense<[2.250000e+02, 2.250000e+02, 1.000000e+00]> : tensor<3xf32>, [#const.ConvertElemType<f16>]
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x12x8x8xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x24x8x8xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <10752> -> memref<3xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<300x5xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <7616> -> memref<300xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x12x8x8xf16, @DDR>) outputs(%0 : memref<1x12x8x8xf16, [@CMX_NN, 0]>) -> memref<1x12x8x8xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x24x8x8xf16, @DDR>) outputs(%1 : memref<1x24x8x8xf16, [@CMX_NN, 0]>) -> memref<1x24x8x8xf16, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<3xf16>) outputs(%2 : memref<3xf16, [@CMX_NN, 0]>) -> memref<3xf16, [@CMX_NN, 0]>
    }
    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
                    @VPU.SW::@builtin_Proposal            // The reference to the Kernel function.
                    inputs(%0 as %arg4: memref<1x12x8x8xf16, [@CMX_NN, 0]>, %1 as %arg5: memref<1x24x8x8xf16, [@CMX_NN, 0]>, %2 as %arg6: memref<3xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%3 as %arg7: memref<300x5xf16, [@CMX_NN, 0]>, %4 as %arg8: memref<300xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> (memref<300x5xf16, [@CMX_NN, 0]>, memref<300xf16, [@CMX_NN, 0]>) {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [8, 6000, 300, 0.69999998807907104, 1, 8, [5.000000e-01, 1.2000000476837158, 2.000000e+00], [8.000000e+00, 1.600000e+01, -1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00], 1, 0, 1, 2.000000e+00, 2.000000e+00, 0]}(%arg4, %arg5, %arg6, %arg7, %arg8)
                    : memref<1x12x8x8xf16, [@CMX_NN, 0]>
                    , memref<1x24x8x8xf16, [@CMX_NN, 0]>
                    , memref<3xf16, [@CMX_NN, 0]>
                    , memref<300x5xf16, [@CMX_NN, 0]>
                    , memref<300xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<300x5xf16, [@CMX_NN, 0]>) outputs(%arg2 : memref<300x5xf16, @DDR>) -> memref<300x5xf16, @DDR>
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<300xf16, [@CMX_NN, 0]>) outputs(%arg3 : memref<300xf16, @DDR>) -> memref<300xf16, @DDR>
    }
    return %arg2, %arg3 : memref<300x5xf16, @DDR>, memref<300xf16, @DDR>

}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "a_scores",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         12,
// CHECK:         8,
// CHECK:         8
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         12288,
// CHECK:         1024,
// CHECK:         128,
// CHECK:         16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "b_boxes",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         24,
// CHECK:         8,
// CHECK:         8
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         24576,
// CHECK:         1024,
// CHECK:         128,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "Proposal_202.0",
// CHECK:       dimensions: [
// CHECK:         300,
// CHECK:         5
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         80,
// CHECK:         16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Proposal_202.1",
// CHECK:       dimensions: [
// CHECK:         300
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 8,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "a_scores",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         12,
// CHECK:         8,
// CHECK:         8
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         12288,
// CHECK:         1024,
// CHECK:         128,
// CHECK:         16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "b_boxes",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         24,
// CHECK:         8,
// CHECK:         8
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         24576,
// CHECK:         1024,
// CHECK:         128,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Proposal_202.0",
// CHECK:       dimensions: [
// CHECK:         300,
// CHECK:         5
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         80,
// CHECK:         16
// CHECK:       ]
// CHECK:     },
// CHECK:     {
// CHECK:       name: "Proposal_202.1",
// CHECK:       dimensions: [
// CHECK:         300
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
