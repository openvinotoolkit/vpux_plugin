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
// This file generates a blob with space_to_depth activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {
  module @VPU.SW {
    func.func private @builtin_SpaceToBatch(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none, none, none) attributes {VPU.kernel_code = "space_to_batch.cpp", VPU.kernel_entry = "space_to_batch", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_224" : tensor<2x8x8x3xf16>
  } outputsInfo : {
    DataInfo "SpaceToBatch_228" : tensor<48x2x2x3xf16>
  }
  VPURT.SW.Runtime
      entryPoint: @VPU.SW::@runtime
      stack_configuration: [
          4096,  // Size in bytes for the actSHAVE0 in the first tile.
          4096,  // Size in bytes for the actSHAVE1 in the first tile.
          4096,  // Size in bytes for the actSHAVE2 in the second tile.
          4096   // Size in bytes for the actSHAVE3 in the second tile.
      ]
  func.func @main(%arg1: memref<2x8x8x3xf16, @DDR>, %arg2: memref<48x2x2x3xf16, @DDR>) -> memref<48x2x2x3xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x8x8x3xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> memref<48x2x2x3xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<48x2x2x3xf16, @DDR>
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%3 : !VPURT.Barrier) {
      %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<2x8x8x3xf16, @DDR>) outputs(%0 : memref<2x8x8x3xf16, [@CMX_NN, 0]>) -> memref<2x8x8x3xf16, [@CMX_NN, 0]>
    }
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SpaceToBatch inputs(%0 as %arg3: memref<2x8x8x3xf16, [@CMX_NN, 0]>) outputs(%1 as %arg4: memref<48x2x2x3xf16, [@CMX_NN, 0]>) on tile 0 -> memref<48x2x2x3xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [[1, 6, 4, 1], [0, 1, 0, 0], [0, 3, 0, 0]]}(%arg3, %arg4) : memref<2x8x8x3xf16, [@CMX_NN, 0]>, memref<48x2x2x3xf16, [@CMX_NN, 0]>
      }
    }
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<48x2x2x3xf16, [@CMX_NN, 0]>) outputs(%2 : memref<48x2x2x3xf16, @DDR>) -> memref<48x2x2x3xf16, @DDR>
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) {
      %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<48x2x2x3xf16, @DDR>) outputs(%arg2 : memref<48x2x2x3xf16, @DDR>) -> memref<48x2x2x3xf16, @DDR>
    }
    return %arg2 : memref<48x2x2x3xf16, @DDR>
  }
}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_224",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           8,
// CHECK:           8,
// CHECK:           3
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           3072,
// CHECK:           384,
// CHECK:           48,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "SpaceToBatch_228",
// CHECK:       dimensions: [
// CHECK:         48,
// CHECK:         2,
// CHECK:         2,
// CHECK:         3
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         192,
// CHECK:         96,
// CHECK:         48,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 4,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_224",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           8,
// CHECK:           8,
// CHECK:           3
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           3072,
// CHECK:           384,
// CHECK:           48,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "SpaceToBatch_228",
// CHECK:       dimensions: [
// CHECK:         48,
// CHECK:         2,
// CHECK:         2,
// CHECK:         3
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         192,
// CHECK:         96,
// CHECK:         48,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
