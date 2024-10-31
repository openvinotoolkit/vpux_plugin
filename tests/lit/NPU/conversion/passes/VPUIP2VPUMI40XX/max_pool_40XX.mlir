//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x9x8xf16>
  }

  func.func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x9x8xf16, #NHWC, @DDR>) -> memref<1x64x9x8xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %5 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }
    %cst = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<0> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <41984> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    }
    %cst_0 = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<0> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <42000> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
      %8 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%3 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [7, 8, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
      } PPE : {
        PPETask {opaque_ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>) -> memref<1x64x9x8xf16, #NHWC, @DDR>
    }
    return %arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>
  }
}

// CHECK: func.func private @maxpool_f16_f16
// CHECK: VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>

// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-NEXT: const.Declare
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-NEXT: const.Declare
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.DPUInvariant
// CHECK: VPUMI40XX.DPUVariant
// CHECK-NOT: VPURT.Task
// CHECK: VPUMI40XX.NNDMA

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @maxpool_f16_f16_two_outputs inputsInfo : {
    DataInfo "input_0" : tensor<1x64x38x112xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x18x56xf16>
    DataInfo "output_1" : tensor<1x64x18x56xi1>
  }

  func.func private @maxpool_f16_f16_two_outputs(%arg0: memref<1x64x38x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>,
                                     %arg1: memref<1x64x18x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>,
                                     %arg2: memref<1x64x18x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) ->
      (memref<1x64x18x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>,
      memref<1x64x18x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) {
    %bar_0 = VPURT.ConfigureBarrier<15> -> !VPURT.Barrier
    %bar_1 = VPURT.ConfigureBarrier<14> -> !VPURT.Barrier
    %bar_2 = VPURT.ConfigureBarrier<12> -> !VPURT.Barrier
    %bar_3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %bar_4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier
    %bar_5 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %bar_6 = VPURT.ConfigureBarrier<13> -> !VPURT.Barrier
    %bar_7 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %bar_8 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    %bar_9 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier
    %bar_10 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier
    %bar_11 = VPURT.ConfigureBarrier<9> -> !VPURT.Barrier
    %bar_12 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier
    %bar_13 = VPURT.ConfigureBarrier<8> -> !VPURT.Barrier
    %bar_14 = VPURT.ConfigureBarrier<11> -> !VPURT.Barrier
    %bar_15 = VPURT.ConfigureBarrier<10> -> !VPURT.Barrier

    %cst_0 = const.Declare memref<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %cst_1 = const.Declare memref<256x64x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<256x64x1x1xf16, {order = #NHWC}>
    %cst_2 = const.Declare memref<1x1x1x1040xui8> = dense<1> : tensor<1x1x1x1040xui8>
    %cst_3 = const.Declare memref<1x1x1x1040xui8> = dense<1> : tensor<1x1x1x1040xui8>

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <654912> -> memref<1x64x38x112xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <136704> -> memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <289920> -> memref<1x64x39x112xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <145216> -> memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <281408> -> memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <849024> -> memref<1x1x1x1040xui8, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <850112> -> memref<1x64x37x112xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x64x18x56xf16, #NHWC, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <129536> -> memref<1x64x18x56xi1, #NHWC, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <1380544> -> memref<1x1x1x1040xui8, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <289920> -> memref<256x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <137600> -> memref<256x1x1x4xsi32, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <653824> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <654848> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <DDR> <1605632> -> memref<1x64x19x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    %16 = VPURT.DeclareBuffer <DDR> <2007040> -> memref<1x64x19x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    %17 = VPURT.DeclareBuffer <DDR> <530432> -> memref<1x64x39x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %18 = VPURT.DeclareBuffer <DDR> <1075200> -> memref<1x64x37x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0] <849024> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <850048> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <DDR> <1741824> -> memref<1x64x19x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <1380544> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <1381568> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %24 = VPURT.DeclareBuffer <DDR> <2015552> -> memref<1x64x19x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>

    VPURT.Task waits(%bar_0 : !VPURT.Barrier) updates(%bar_1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x38x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) outputs(%0 : memref<1x64x38x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x38x112xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_1 : !VPURT.Barrier) updates(%bar_2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [2, 2],
        minimumHardwareExecutionCost = 4294967400 : i64,
        output_se_size = 64 : i64,
        task_type = #VPUIP.nce_task_type<MAXPOOL>
      }
      input(%0 : memref<1x64x38x112xf16, #NHWC, [@CMX_NN, 0]>)
      weight_table(%13 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%0 : memref<1x64x38x112xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output(%1 : memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output_sparsity_map(%2 : memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>)
      outputs(%1 : memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>)
      output_sparsity_map(%2 : memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>
      variants : {
        DPUTask {inEnd = [111, 37, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 18, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
      }
      PPE : {
        PPETask {opaque_ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar_2 : !VPURT.Barrier) updates(%bar_3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>) outputs(%15 : memref<1x64x19x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) -> memref<1x64x19x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    }
    VPURT.Task waits(%bar_3 : !VPURT.Barrier) updates(%bar_4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>) outputs(%16 : memref<1x64x19x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) -> memref<1x64x19x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    }
    VPURT.Task waits(%bar_4 : !VPURT.Barrier) updates(%bar_5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_3 : memref<1x1x1x1040xui8>) outputs(%6 : memref<1x1x1x1040xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1040xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_5 : !VPURT.Barrier) updates(%bar_6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1x64x39x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) outputs(%3 : memref<1x64x39x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x39x112xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_6 : !VPURT.Barrier) updates(%bar_7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<1x64x37x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) outputs(%7 : memref<1x64x37x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x37x112xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_6 : !VPURT.Barrier) updates(%bar_8, %bar_9 : !VPURT.Barrier, !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [2, 2],
        minimumHardwareExecutionCost = 4294967400 : i64,
        output_se_size = 64 : i64,
        task_type = #VPUIP.nce_task_type<MAXPOOL>
      }
      input(%3 : memref<1x64x39x112xf16, #NHWC, [@CMX_NN, 0]>)
      weight_table(%19 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%3 : memref<1x64x39x112xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output(%4 : memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output_sparsity_map(%5 : memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>)
      outputs(%4 : memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>)
      output_sparsity_map(%5 : memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>
      variants : {
        DPUTask {inEnd = [111, 38, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 18, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {opaque_ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar_7 : !VPURT.Barrier) updates(%bar_8, %bar_9 : !VPURT.Barrier, !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_2 : memref<1x1x1x1040xui8>) outputs(%10 : memref<1x1x1x1040xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1040xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_8 : !VPURT.Barrier) updates(%bar_10 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<1x64x19x56xf16, #NHWC, [@CMX_NN, 0]>) outputs(%21 : memref<1x64x19x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) -> memref<1x64x19x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    }
    VPURT.Task waits(%bar_9 : !VPURT.Barrier) updates(%bar_11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [3, 3],
        kernel_strides = [2, 2],
        minimumHardwareExecutionCost = 4294967400 : i64,
        output_se_size = 64 : i64,
        task_type = #VPUIP.nce_task_type<MAXPOOL>
      }
      input(%7 : memref<1x64x37x112xf16, #NHWC, [@CMX_NN, 0]>)
      weight_table(%22 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%7 : memref<1x64x37x112xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output(%8 : memref<1x64x18x56xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output_sparsity_map(%9 : memref<1x64x18x56xi1, #NHWC, [@CMX_NN, 0]>)
      outputs(%8 : memref<1x64x18x56xf16, #NHWC, [@CMX_NN, 0]>)
      output_sparsity_map(%9 : memref<1x64x18x56xi1, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x18x56xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x18x56xi1, #NHWC, [@CMX_NN, 0]>
      variants : {
        DPUTask {inEnd = [111, 36, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 17, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {opaque_ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar_10 : !VPURT.Barrier) updates(%bar_12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%5 : memref<1x64x19x56xi1, #NHWC, [@CMX_NN, 0]>) outputs(%24 : memref<1x64x19x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) -> memref<1x64x19x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    }
    VPURT.Task waits(%bar_12 : !VPURT.Barrier) updates(%bar_13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_1 : memref<256x64x1x1xf16, #NHWC>) outputs(%11 : memref<256x64x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<256x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_13 : !VPURT.Barrier) updates(%bar_11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<256x1x1x4xsi32>) outputs(%12 : memref<256x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x4xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar_11 : !VPURT.Barrier) updates(%bar_14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%8 : memref<1x64x18x56xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x18x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) -> memref<1x64x18x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    }
    VPURT.Task waits(%bar_14 : !VPURT.Barrier) updates(%bar_15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %25 = VPUIP.NNDMA {port = 0 : i64} inputs(%9 : memref<1x64x18x56xi1, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x64x18x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>) -> memref<1x64x18x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
    }
    return %arg1, %arg2 : memref<1x64x18x56xf16, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>, memref<1x64x18x56xi1, {order = #NHWC, strides = [200704, 1, 3584, 64]}, @DDR>
  }
}

// CHECK: func.func private @maxpool_f16_f16_two_outputs
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 0 : ui8} <15, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <14, -1> -> !VPURegMapped.Index<0:0:1>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <12, -1> -> !VPURegMapped.Index<0:0:2>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:3>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:4>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <2, -1> -> !VPURegMapped.Index<0:0:5>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8} <13, -1> -> !VPURegMapped.Index<0:0:6>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:7>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <3, -1> -> !VPURegMapped.Index<0:0:8>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <5, -1> -> !VPURegMapped.Index<0:0:9>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <7, -1> -> !VPURegMapped.Index<0:0:10>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <9, -1> -> !VPURegMapped.Index<0:0:11>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <6, -1> -> !VPURegMapped.Index<0:0:12>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <8, -1> -> !VPURegMapped.Index<0:0:13>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <11, -1> -> !VPURegMapped.Index<0:0:14>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, producer_count = 1 : ui8} <10, -1> -> !VPURegMapped.Index<0:0:15>
// CHECK-NEXT: const.Declare
// CHECK-NEXT: const.Declare
// CHECK-NEXT: const.Declare
// CHECK-NEXT: const.Declare
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <DDR>
// CHECK-NEXT: VPURT.DeclareBuffer <DDR>
// CHECK-NEXT: VPURT.DeclareBuffer <DDR>
// CHECK-NEXT: VPURT.DeclareBuffer <DDR>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <DDR>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <CMX_NN>
// CHECK-NEXT: VPURT.DeclareBuffer <DDR>
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.DPUInvariant
// CHECK: VPUMI40XX.DPUVariant
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.DPUInvariant
// CHECK: VPUMI40XX.DPUVariant
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.DPUInvariant
// CHECK: VPUMI40XX.DPUVariant
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK-NOT: VPURT.Task
// CHECK-NEXT: VPUMI40XX.NNDMA
// CHECK-SAME: VPUMI40XX.NNDMATransaction
// CHECK: VPUMI40XX.MappedInference
