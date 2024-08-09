//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule {

  IE.CNNNetwork entryPoint : @singleEltwise inputsInfo : {
    DataInfo "input_0" : tensor<1x32x56x56xui8>
    DataInfo "input_1" : tensor<1x32x56x56xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x56x56xui8>
  }

func.func @singleEltwise(%arg0: memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg1: memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>, %arg2: memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
  %0 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %5 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
  %6 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  VPURT.Task updates(%5 : !VPURT.Barrier) {
    %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%5 : !VPURT.Barrier) {
    %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  }
  VPURT.Task waits(%6 : !VPURT.Barrier) {
    %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
  }
  VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
    %7 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i32, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%0 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weights(%1 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_input(%3 : memref<1x32x56x56x!quant.uniform<u8<0:3>:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_output(%4 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) -> memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]> variants : {
      DPUTask {inStart = [0, 0, 0], inEnd = [15, 15, 15], outEnd = [55, 55, 31], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
      PPETask <ADD> {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_post_shift = 0 : i64, quant_scale = [1.000000e+00], quant_shift = [14]}
    }
  }
  return %arg2 : memref<1x32x56x56x!quant.uniform<u8:f32, 1.000000e+00>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
}
}

// CHECK: func.func @singleEltwise
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
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
