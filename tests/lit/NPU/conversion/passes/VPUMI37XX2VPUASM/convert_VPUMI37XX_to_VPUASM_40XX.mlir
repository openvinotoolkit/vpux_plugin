//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%"  --convert-VPUMI37XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.CNNNetwork entryPoint : @twoDma inputsInfo : {
  DataInfo "input_0" : tensor<1x16x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x16x16x16xf16>
  DataInfo "output_1" : tensor<1x16x16x16xf16>
}

func.func @twoDma() {
  %0 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:0>
  %1 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:1>
  %2 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:1:2>
  %3 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
  %4 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
  %5 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:2>

  %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %8 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

  %11 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
  %12 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

  %18 = VPUMI37XX.NNDMA {port = 1 : i64} taskLocation(%2 : !VPURegMapped.Index<0:1:2>) inputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%8 : memref<1x16x16x16xf16, #NHWC, @DDR>) waits(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
  %15 = VPUMI37XX.NNDMA {port = 0 : i64} taskLocation(%5 : !VPURegMapped.Index<0:0:2>) inputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x16x16xf16, #NHWC, @DDR>) waits(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %17 = VPUMI37XX.NNDMA {port = 1 : i64} taskLocation(%1 : !VPURegMapped.Index<0:1:1>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) nextDMAIdx(%18 : !VPURegMapped.Index<0:1:2>) waits(%11 : !VPURegMapped.Index<0:0:0>) updates(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
  %14 = VPUMI37XX.NNDMA {port = 0 : i64} taskLocation(%4 : !VPURegMapped.Index<0:0:1>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) nextDMAIdx(%15 : !VPURegMapped.Index<0:0:2>) waits(%11 : !VPURegMapped.Index<0:0:0>) updates(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %16 = VPUMI37XX.NNDMA {port = 1 : i64} taskLocation(%0 : !VPURegMapped.Index<0:1:0>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) nextDMAIdx(%17 : !VPURegMapped.Index<0:1:1>) updates(%11 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
  %13 = VPUMI37XX.NNDMA {port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:0:0>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) nextDMAIdx(%14 : !VPURegMapped.Index<0:0:1>) updates(%11 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  %19 = VPUMI37XX.MappedInference dmas(%13, %16 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>) barriers(%11 : !VPURegMapped.Index<0:0:0>) dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
  return
}

// CHECK: func.func @twoDma()

//CHECK: VPUASM.DeclareTaskBuffer @[[TB10:.*]] idx(!VPURegMapped.Index<0:1:0>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB11:.*]] idx(!VPURegMapped.Index<0:1:1>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB12:.*]] idx(!VPURegMapped.Index<0:1:2>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB00:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB01:.*]] idx(!VPURegMapped.Index<0:0:1>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB02:.*]] idx(!VPURegMapped.Index<0:0:2>) <DMA>

//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF1:.*]] !VPUASM.Buffer< "NetworkOutput"[0] <0>
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF2:.*]] !VPUASM.Buffer< "NetworkOutput"[1] <0>

//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF4:.*]] !VPUASM.Buffer< "CMX_NN"[1] <0>

//CHECK-NEXT: VPUASM.ConfigureBarrier @[[SYMBARRIER0:.*]] idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(2 : 2)
//CHECK-NEXT: VPUASM.ConfigureBarrier @[[SYMBARRIER1:.*]] idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(2 : 2)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA12:.*]] idx(!VPURegMapped.Index<0:1:2>)
    //CHECK-SAME: taskLocation(@[[TB12]]) input(@[[SYMBUF4]]) outputs([@[[SYMBUF2]]]) waits([1 : ui8]) updates([]) start_after(0)
    //CHECK-SAME: descriptor(<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA02:.*]] idx(!VPURegMapped.Index<0:0:2>)
    //CHECK-SAME: taskLocation(@[[TB02]]) input(@[[SYMBUF3]]) outputs([@[[SYMBUF1]]]) waits([1 : ui8]) updates([]) start_after(0)
    //CHECK-SAME: descriptor(<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA11:.*]] idx(!VPURegMapped.Index<0:1:1>)
    //CHECK-SAME: taskLocation(@[[TB11]]) links(@[[SYMDMA12]]) input(@[[SYMBUF0]]) outputs([@[[SYMBUF4]]]) waits([0 : ui8]) updates([1 : ui8]) start_after(0)
    //CHECK-SAME: descriptor(<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA01:.*]] idx(!VPURegMapped.Index<0:0:1>)
    //CHECK-SAME: taskLocation(@[[TB01]]) links(@[[SYMDMA02]]) input(@[[SYMBUF0]]) outputs([@[[SYMBUF3]]]) waits([0 : ui8]) updates([1 : ui8]) start_after(0)
    //CHECK-SAME: descriptor(<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA10:.*]] idx(!VPURegMapped.Index<0:1:0>)
    //CHECK-SAME: taskLocation(@[[TB10]]) links(@[[SYMDMA11]]) input(@[[SYMBUF0]]) outputs([@[[SYMBUF4]]]) waits([]) updates([0 : ui8]) start_after(0)
    //CHECK-SAME: descriptor(<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA00:.*]] idx(!VPURegMapped.Index<0:0:0>)
    //CHECK-SAME: taskLocation(@[[TB00]]) links(@[[SYMDMA01]]) input(@[[SYMBUF0]]) outputs([@[[SYMBUF3]]]) waits([]) updates([0 : ui8]) start_after(0)
    //CHECK-SAME: descriptor(<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)

//CHECK: VPUASM.MappedInference_37XX @MappedInference : dmas([@[[SYMDMA00]], @[[SYMDMA10]]]) barriers(@[[SYMBARRIER0]])
//CHECK-SAME: dmaCount([3, 3]) invariantCount(0) variantCount(0) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.CNNNetwork entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x32x32x32xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x16x32xf16>
  DataInfo "output_1" : tensor<1x64x16x32xf16>
}
func.func @multiple_clusters_dpu_soh_f16_f16_f16() {
  %0 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
  %1 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
  %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
  %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>

  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

  %14 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) input(%10 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%4 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%12 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%9 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%6 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%7 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
  }
  %15 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) input(%11 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) weights(%5 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) parent_input(%9 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%6 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%8 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) -> <0:0:1> PPE : {
  }

  %16 = "VPUMI37XX.DPUVariant"(%0, %14) {end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  %17 = "VPUMI37XX.DPUVariant"(%1, %15) {end = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:1>

  %18 = VPUMI37XX.MappedInference invariants(%14 : !VPURegMapped.Index<0:0:0>) variants(%16 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: func.func @multiple_clusters_dpu_soh_f16_f16_f16

//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) <DPUInvariant>

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer< "CMX_NN"[1] <0>

//CHECK: %[[DISTRIB_BUFF_VAL0:.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
//CHECK: VPUASM.SymbolizeValueOp @[[SYMBUFF2:.*]](%[[DISTRIB_BUFF_VAL0]] : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <4096>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF4:.*]] !VPUASM.Buffer< "CMX_NN"[1] <4096>
//CHECK: %[[DISTRIB_BUFF_VAL1:.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
//CHECK: VPUASM.SymbolizeValueOp @[[SYMBUFF5:.*]](%[[DISTRIB_BUFF_VAL1]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>)
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF6:.*]] !VPUASM.Buffer< "CMX_NN"[0] <69632>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF7:.*]] !VPUASM.Buffer< "CMX_NN"[1] <69632>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF8:.*]] !VPUASM.Buffer< "CMX_NN"[0] <102400>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF9:.*]] !VPUASM.Buffer< "CMX_NN"[1] <102400>

//CHECK: VPUASM.DPUInvariant_37XX @[[INVARIANTSYM0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBIVAR0]])
    // CHECK-SAME: input(@[[SYMBUFF6]]) weights(@[[SYMBUFF0]]) weight_table(@[[SYMBUFF8]]) parent_input(@[[SYMBUFF5]]) parent_output(@[[SYMBUFF2]]) outputs([@[[SYMBUFF3]]])
//CHECK: VPUASM.DPUInvariant_37XX @[[INVARIANTSYM1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[TBIVAR1]])
    // CHECK-SAME: input(@[[SYMBUFF7]]) weights(@[[SYMBUFF1]]) weight_table(@[[SYMBUFF9]]) parent_input(@[[SYMBUFF5]]) parent_output(@[[SYMBUFF2]]) outputs([@[[SYMBUFF4]]])

//CHECK: VPUASM.DPUVariant_37XX @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBVAR0]]) calls @[[INVARIANTSYM0]]
//CHECK: VPUASM.DPUVariant_37XX @[[SYMVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[TBVAR1]]) calls @[[INVARIANTSYM1]]

//CHECK: VPUASM.MappedInference_37XX @MappedInference : invariants(@[[INVARIANTSYM0]]) variants(@[[SYMVAR0]]) dmaCount([0, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.CNNNetwork entryPoint : @multiple_clusters_dpu_sok_f16_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x32x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x16x16xf16>
  DataInfo "output_1" : tensor<1x64x16x16xf16>
}

func.func @multiple_clusters_dpu_sok_f16_f16_f16() {
  %0 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
  %1 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
  %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
  %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
  %4 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>

  %5 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x32x16x16xf16, #NHWC, @DDR>
  %6 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x64x16x16xf16, #NHWC, @DDR>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> !VPUIP.DistributedBuffer<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 1]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <51200> -> memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [1] <51200> -> memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 1]>
  %17 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %18 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 1]>
  %19 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>
  %20 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>

  %22 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) input(%11 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%7 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) parent_output(%9 : !VPUIP.DistributedBuffer<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%19, %20 : memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>, memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>) -> <0:0:0> PPE : {
  }
  %23 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) input(%12 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 1]>) weights(%8 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%14 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x16x16xf16, {order = #NHWC, strides = [8192, 1, 512, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) parent_output(%9 : !VPUIP.DistributedBuffer<1x64x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%19, %20 : memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>, memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>) -> <0:0:1> PPE : {
  }

  %24 = "VPUMI37XX.DPUVariant"(%0, %22) {end = [15, 15, 31], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  %25 = "VPUMI37XX.DPUVariant"(%1, %23) {end = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 32]} : (!VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:1>

  %26 = VPUMI37XX.MappedInference invariants(%22 : !VPURegMapped.Index<0:0:0>) variants(%24 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: func.func @multiple_clusters_dpu_sok_f16_f16_f16

//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBDMA0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer< "NetworkOutput"[1] <0>

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[1] <0>

//CHECK-DAG: %[[DISTBUFF_VAL0:.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2048> -> !VPUIP.DistributedBuffer<[[DISTBUFF_TYPE0:.*]]>
//CHECK-DAG: %[[DISTBUFF_VAL1:.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<[[DISTBUFF_TYPE1:.*]]>

//CHECK-DAG: VPUASM.SymbolizeValueOp @[[SYMBUFF4:.*]](%[[DISTBUFF_VAL0]] : !VPUIP.DistributedBuffer<[[DISTBUFF_TYPE0]]>)
//CHECK-DAG: VPUASM.SymbolizeValueOp @[[SYMBUFF5:.*]](%[[DISTBUFF_VAL1]] : !VPUIP.DistributedBuffer<[[DISTBUFF_TYPE1]]>)

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF6:.*]] !VPUASM.Buffer< "CMX_NN"[0] <34816>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF7:.*]] !VPUASM.Buffer< "CMX_NN"[1] <34816>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF8:.*]] !VPUASM.Buffer< "CMX_NN"[0] <51200>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF9:.*]] !VPUASM.Buffer< "CMX_NN"[1] <51200>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF10:.*]] !VPUASM.Buffer< "CMX_NN"[0] <2048>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF11:.*]] !VPUASM.Buffer< "CMX_NN"[1] <2048>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF12:.*]] !VPUASM.Buffer< "CMX_NN"[0] <34816>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF13:.*]] !VPUASM.Buffer< "CMX_NN"[1] <34816>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF14:.*]] !VPUASM.Buffer< "CMX_NN"[0] <2048>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF15:.*]] !VPUASM.Buffer< "CMX_NN"[1] <2048>

//CHECK: VPUASM.DPUInvariant_37XX @[[SYMINV0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBIVAR0]])
    //CHECK-SAME: input(@[[SYMBUFF6]]) weights(@[[SYMBUFF2]]) weight_table(@[[SYMBUFF8]])
    //CHECK-SAME: parent_input(@[[SYMBUFF5]]) parent_output(@[[SYMBUFF4]]) outputs([@[[SYMBUFF14]], @[[SYMBUFF15]]])

//CHECK: VPUASM.DPUInvariant_37XX @[[SYMINV1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[TBIVAR1]])
    //CHECK-SAME: input(@[[SYMBUFF7]]) weights(@[[SYMBUFF3]]) weight_table(@[[SYMBUFF9]])
    //CHECK-SAME: parent_input(@[[SYMBUFF5]]) parent_output(@[[SYMBUFF4]]) outputs([@[[SYMBUFF14]], @[[SYMBUFF15]]])

//CHECK: VPUASM.DPUVariant_37XX @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBVAR0]]) calls @[[SYMINV0]]
//CHECK: VPUASM.DPUVariant_37XX @[[SYMVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[TBVAR1]]) calls @[[SYMINV1]]

//CHECK: VPUASM.MappedInference_37XX @MappedInference : invariants(@[[SYMINV0]]) variants(@[[SYMVAR0]]) dmaCount([0, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0)
