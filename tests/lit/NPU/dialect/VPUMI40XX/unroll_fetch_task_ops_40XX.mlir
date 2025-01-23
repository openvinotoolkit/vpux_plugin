//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --unroll-fetch-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
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
    %15 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %16 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %17 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %18 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%15 : !VPURegMapped.Index<0:0:0>) input(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%11 : !VPURegMapped.Index<0:0:1>) updates(%12 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %20 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%16 : !VPURegMapped.Index<0:0:1>) previousTask(%19 : !VPURegMapped.Index<0:0:0>) input(%7 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%12 : !VPURegMapped.Index<0:0:2>) updates(%13 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %21 = VPUMI40XX.DPUVariant taskLocation(%17 : !VPURegMapped.Index<0:0:0>) calls(%19 : <0:0:0>) weights(%6 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %22 = VPUMI40XX.DPUVariant taskLocation(%18 : !VPURegMapped.Index<0:0:1>) previousTask(%21 : !VPURegMapped.Index<0:0:0>) calls(%20 : <0:0:1>) weights(%9 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %23 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %24 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %25 = VPURegMapped.FetchTask primary(%19 -> %20) secondary(%21 -> %22) (<0:0:0> -> <0:0:1> : !VPURegMapped.Index<0:0:0> -> !VPURegMapped.Index<0:0:1>) -> <0:0:0>
    %26 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 1 : i64} inputs(%23 : memref<1x1x1x1xi32, @DDR>) outputs(%24 : memref<1x1x1x1xi32, @DDR>) previousDMA(%25 : !VPURegMapped.Index<0:0:0>) updates(%10 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %27 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, @DDR>) outputs(%2 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%26 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:0>) updates(%11 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %28 = VPUMI40XX.NNDMA {is_out_of_order, port = 1 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%5 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%27 : !VPURegMapped.Index<0:0:2>) updates(%12 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %29 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x14x14xf16, @DDR>) waits(%13 : !VPURegMapped.Index<0:0:3>) updates(%14 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %30 = VPURegMapped.Enqueue at(%10 : !VPURegMapped.Index<0:0:0>) (%21 -> %21 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %31 = VPURegMapped.Enqueue previousTaskIdx(%30 : !VPURegMapped.Index<0:0:0>) at(%10 : !VPURegMapped.Index<0:0:0>) (%22 -> %22 : <0:0:1> -> <0:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %32 = VPUMI40XX.MappedInference dmas((%25, %29) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%19 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%10 : !VPURegMapped.Index<0:0:0>) workItemTasks(%30 : !VPURegMapped.Index<0:0:0>) dmaCount([[4, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(5) workItemCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x16x14x14xf16, @DDR>
  }
}

// CHECK-NOT: VPURegMapped.FetchTask
//CHECK: [[VAL15:%.*]] = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL16:%.*]] = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL17:%.*]] = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL18:%.*]] = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[VAL19:%.*]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL20:%.*]] = VPUMI40XX.DPUInvariant
//CHECK: [[VAL21:%.*]] = VPUMI40XX.DPUVariant taskLocation([[VAL17]] : !VPURegMapped.Index<0:0:0>) calls([[VAL19]] : <0:0:0>)
//CHECK: [[VAL22:%.*]] = VPUMI40XX.DPUVariant taskLocation([[VAL18]] : !VPURegMapped.Index<0:0:1>) previousTask([[VAL21]] : !VPURegMapped.Index<0:0:0>) calls([[VAL20]] : <0:0:1>)
//CHECK: VPURegMapped.ViewTaskRange([[VAL19]] -> [[VAL20]] : <0:0:0> -> <0:0:1>) -> memref<2x{{.*}}xui8>
//CHECK: VPURegMapped.ViewTaskRange([[VAL15]] -> [[VAL16]] : <0:0:0> -> <0:0:1>) -> memref<2x{{.*}}xui8, [@CMX_NN, 0]>
//CHECK: VPURegMapped.ViewTaskRange([[VAL21]] -> [[VAL22]] : <0:0:0> -> <0:0:1>) -> memref<2x224xui8>
//CHECK: VPURegMapped.ViewTaskRange([[VAL17]] -> [[VAL18]] : <0:0:0> -> <0:0:1>) -> memref<2x224xui8, [@CMX_NN, 0]>
