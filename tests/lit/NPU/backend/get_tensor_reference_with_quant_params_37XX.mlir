//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.4850000044878791:3>

module @TestGetTensorReference attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  module @UsedMemory {
    IE.MemoryResource 0 bytes of @DDR
  }
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @UsedMemory {
      IE.MemoryResource 393216 bytes of @CMX_NN
    }
    IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
      DataInfo "input0" : tensor<1x64x32x32xui8, {order = #NHWC}>
    }
    outputsInfo : {
      DataInfo "output0" : tensor<1x64x32x32xf16, {order = #NHWC}>
    }
  func.func @main(%arg0: memref<1x64x32x32xui8, #NHWC, @DDR>, %arg1: memref<1x64x32x32xf16, #NHWC, @DDR>) -> memref<1x64x32x32xf16, #NHWC, @DDR> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {DescriptorHandle = 1234 : ui64} -> memref<1x64x32x32x!qElemType, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {DescriptorHandle = 5678 : ui64} -> memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {cycleBegin = 2 : i64, cycleEnd = 4 : i64} {
      %9 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 4780 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%3 : memref<1x64x32x32x!qElemType, #NHWC, [@CMX_NN, 0]>) weights(%3 : memref<1x64x32x32x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_input(%3 : memref<1x64x32x32x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_output(%4 : memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x32x32xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 31, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24330], in2_quant_mult = [24330], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_post_shift = 0 : i64, quant_shift = [28]}
      }
    }
    return %arg1 : memref<1x64x32x32xf16, #NHWC, @DDR>
  }
}

// CHECK:  output_data: {
// CHECK-NOT:  parent_output_tensor: {
// CHECK-NOT:  parent_input_tensor: {
// CHECK-NOT:  input_data: {
// CHECK-NOT:  output_data: {
// CHECK-NOT:  weights_data: {
// CHECK:  descriptor: 5678,
// CHECK-NOT:  parent_output_tensor: {
// CHECK-NOT:  parent_input_tensor: {
// CHECK-NOT:  input_data: {
// CHECK-NOT:  output_data: {
// CHECK:  weights_data: {
