//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --resolve-wlm-task-location %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  module @UsedMemory {
    IE.MemoryResource 0 bytes of @DDR
  }
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @UsedMemory {
      IE.MemoryResource 21760 bytes of @CMX_NN
    }
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<1x16x14x14xf16, @DDR>) -> memref<1x16x14x14xf16, @DDR> {
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<1x1x1x4864xui8, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%10 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}(%11 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%12 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%13 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
    %startIndexes:2, %endIndexes:2 = "VPURegMapped.ExecutionGroup"(%11, %13) ({
      %23 = VPUMI40XX.DPUInvariant {activation_window_channel_length = 0 : i64, clean_after = 0 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
        VPUMI40XX.PPETask <ADD> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01]}
      }
      %24 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
        VPUMI40XX.PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
      %25 = VPUMI40XX.DPUVariant calls(%23 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
      %26 = VPUMI40XX.DPUVariant previousTask(%25 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
      "VPURegMapped.GroupYield"(%23, %25, %24, %26) {operandSegmentSizes = array<i32: 2, 2>} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> ()
    }) {operandSegmentSizes = array<i32: 0, 1, 1>, resultSegmentSizes = array<i32: 2, 2>, task_type = #VPURegMapped.task_type<DPUInvariant>} : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:3>) -> (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
    %15 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %16 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %17 = VPURegMapped.FetchTask primary(%startIndexes#0 -> %endIndexes#0) secondary(%startIndexes#1 -> %endIndexes#1) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0>
    %18 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64} inputs(%15 : memref<1x1x1x1xi32, @DDR>) outputs(%16 : memref<1x1x1x1xi32, @DDR>) previousDMA(%17 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%18 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %20 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%19 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %21 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %22 = VPUMI40XX.MappedInference dmas((%17, %21) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%startIndexes#0 : !VPURegMapped.Index<0:0:0>) variants(%startIndexes#1 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:2>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:3>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:4>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:5>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:6>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:7>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:8>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:9>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:10>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:11>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:12>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:13>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:14>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:15>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:16>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:17>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:18>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:19>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:20>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:21>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:22>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:23>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:24>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:25>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:26>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:27>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:28>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:29>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:30>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:31>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:32>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:33>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:34>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:35>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:36>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:37>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:38>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:39>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:40>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:41>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:42>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:43>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:44>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:45>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:46>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:47>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:48>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:49>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:50>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:51>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:52>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:53>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:54>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:55>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:56>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:57>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:58>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:59>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:60>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:61>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:62>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:63>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:2>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:3>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:4>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:5>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:6>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:7>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:8>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:9>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:10>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:11>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:12>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:13>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:14>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:15>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:16>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:17>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:18>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:19>
// CHECK:VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:20>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:21>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:22>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:23>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:24>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:25>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:26>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:27>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:28>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:29>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:30>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:31>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:32>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:33>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:34>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:35>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:36>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:37>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:38>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:39>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:40>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:41>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:42>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:43>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:44>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:45>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:46>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:47>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:48>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:49>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:50>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:51>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:52>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:53>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:54>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:55>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:56>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:57>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:58>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:59>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:60>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:61>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:62>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:63>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:64>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:65>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:66>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:67>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:68>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:69>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:70>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:71>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:72>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:73>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:74>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:75>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:76>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:77>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:78>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:79>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:80>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:81>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:82>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:83>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:84>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:85>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:86>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:87>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:88>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:89>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:90>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:91>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:92>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:93>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:94>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:95>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:96>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:97>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:98>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:99>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:100>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:101>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:102>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:103>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:104>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:105>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:106>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:107>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:108>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:109>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:110>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:111>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:112>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:113>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:114>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:115>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:116>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:117>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:118>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:119>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:120>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:121>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:122>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:123>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:124>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:125>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:126>
// CHECK: VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:127>
