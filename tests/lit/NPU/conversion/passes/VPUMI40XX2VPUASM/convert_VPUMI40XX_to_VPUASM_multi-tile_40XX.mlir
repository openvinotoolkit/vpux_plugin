//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.CNNNetwork entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x32x32x32xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x16x32xf16>
  DataInfo "output_1" : tensor<1x64x16x32xf16>
}
func.func @multiple_clusters_dpu_soh_f16_f16_f16() {
  %0 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
  %1 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:0>
  %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
  %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:0>

  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

  %14 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) input(%10 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%4 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%12 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
  }
  %15 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%3 : !VPURegMapped.Index<1:0:0>) input(%11 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) weights(%5 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) outputs(%8 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) -> <1:0:0> PPE : {
  }

  %16 = VPUMI40XX.DPUVariant taskLocation(%0 : !VPURegMapped.Index<0:0:0>) calls(%14 : !VPURegMapped.Index<0:0:0>) weight_table(%12 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:0>
  %17 = VPUMI40XX.DPUVariant taskLocation(%1 : !VPURegMapped.Index<1:0:0>) calls(%15 : !VPURegMapped.Index<1:0:0>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<1:0:0>

  %18 = VPUMI40XX.MappedInference invariants(%14, %15 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) variants(%16, %17 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 1, 0, 0, 0, 0]) variantCount([1, 1, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  VPUMI40XX.OpRanges
}

//CHECK: func.func @multiple_clusters_dpu_soh_f16_f16_f16

//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR1:.*]] idx(!VPURegMapped.Index<1:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR1:.*]] idx(!VPURegMapped.Index<1:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer< "CMX_NN"[1] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <4096>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF4:.*]] !VPUASM.Buffer< "CMX_NN"[1] <4096>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF6:.*]] !VPUASM.Buffer< "CMX_NN"[0] <69632>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF7:.*]] !VPUASM.Buffer< "CMX_NN"[1] <69632>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF8:.*]] !VPUASM.Buffer< "CMX_NN"[0] <102400>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF9:.*]] !VPUASM.Buffer< "CMX_NN"[1] <102400>

//CHECK: VPUASM.DPUInvariant @[[INVARIANTSYM0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBIVAR0]])
    // CHECK-SAME: input(@[[SYMBUFF6]]) weights(@[[SYMBUFF0]]) weight_table(@[[SYMBUFF8]]) output(@[[SYMBUFF3]])
//CHECK: VPUASM.DPUInvariant @[[INVARIANTSYM1:.*]] idx(!VPURegMapped.Index<1:0:0>) taskLocation(@[[TBIVAR1]])
    // CHECK-SAME: input(@[[SYMBUFF7]]) weights(@[[SYMBUFF1]]) weight_table(@[[SYMBUFF9]]) output(@[[SYMBUFF4]])

//CHECK: VPUASM.DPUVariant @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBVAR0]]) invariant @[[INVARIANTSYM0]] calls @[[TBIVAR0]]
//CHECK: VPUASM.DPUVariant @[[SYMVAR1:.*]] idx(!VPURegMapped.Index<1:0:0>) taskLocation(@[[TBVAR1]]) invariant @[[INVARIANTSYM1]] calls @[[TBIVAR1]]

// CHECK: VPUASM.MappedInference @MappedInference : invariants([@[[INVARIANTSYM0]], @[[INVARIANTSYM1]]]) variants([@[[SYMVAR0]], @[[SYMVAR1]]])
// CHECK-SAME{LITERAL}: dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
// CHECK-SAME: invariantCount([1, 1, 0, 0, 0, 0]) variantCount([1, 1, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0)

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
  %1 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:0>
  %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
  %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:0>
  %4 = VPURegMapped.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>

  %5 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x32x16x16xf16, #NHWC, @DDR>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %8 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
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

  %21 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%4 : !VPURegMapped.Index<0:0:0>) inputs(%5 : memref<1x32x16x16xf16, #NHWC, @DDR>) outputs(%17, %18 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  %22 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) input(%11 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%7 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%19, %20 : memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>, memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>) -> <0:0:0> PPE : {
  }
  %23 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} taskLocation(%3 : !VPURegMapped.Index<1:0:0>) input(%12 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 1]>) weights(%8 : memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%14 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) outputs(%19, %20 : memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 0]>, memref<1x32x16x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, [@CMX_NN, 1]>) -> <1:0:0> PPE : {
  }

  %24 = VPUMI40XX.DPUVariant taskLocation(%0 : !VPURegMapped.Index<0:0:0>) calls(%22 : !VPURegMapped.Index<0:0:0>) weight_table(%13 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [15, 15, 31], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:0>
  %25 = VPUMI40XX.DPUVariant taskLocation(%1 : !VPURegMapped.Index<1:0:0>) calls(%23 : !VPURegMapped.Index<1:0:0>) weight_table(%13 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [15, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 32], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<1:0:0>

  %26 = VPUMI40XX.MappedInference dmas((%21) : (!VPURegMapped.Index<0:0:0>)) invariants(%22, %23 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) variants(%24, %25 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) dmaCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 1, 0, 0, 0, 0]) variantCount([1, 1, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  VPUMI40XX.OpRanges
}

//CHECK: func.func @multiple_clusters_dpu_sok_f16_f16_f16

//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR1:.*]] idx(!VPURegMapped.Index<1:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR1:.*]] idx(!VPURegMapped.Index<1:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBDMA0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>

//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[1] <0>

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

//CHECK: VPUASM.NNDMA @[[SYMDMA0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBDMA0]])
    // CHECK-SAME: input(@[[SYMBUFF0]]) outputs([@[[SYMBUFF12]], @[[SYMBUFF13]]])

//CHECK: VPUASM.DPUInvariant @[[SYMINV0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBIVAR0]])
    //CHECK-SAME: input(@[[SYMBUFF6]]) weights(@[[SYMBUFF2]]) weight_table(@[[SYMBUFF8]])
    //CHECK-SAME: output(@[[SYMBUFF14]])

//CHECK: VPUASM.DPUInvariant @[[SYMINV1:.*]] idx(!VPURegMapped.Index<1:0:0>) taskLocation(@[[TBIVAR1]])
    //CHECK-SAME: input(@[[SYMBUFF7]]) weights(@[[SYMBUFF3]]) weight_table(@[[SYMBUFF9]])
    //CHECK-SAME: output(@[[SYMBUFF15]])

//CHECK: VPUASM.DPUVariant @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[TBVAR0]]) invariant @[[SYMINV0]] calls @[[TBIVAR0]]
//CHECK: VPUASM.DPUVariant @[[SYMVAR1:.*]] idx(!VPURegMapped.Index<1:0:0>) taskLocation(@[[TBVAR1]]) invariant @[[SYMINV1]] calls @[[TBIVAR1]]

// CHECK{LITERAL}: VPUASM.MappedInference @MappedInference : dmas([[
// CHECK-SAME: @[[SYMDMA0]]]]) invariants([@[[SYMINV0]], @[[SYMINV1]]]) variants([@[[SYMVAR0]], @[[SYMVAR1]]])
// CHECK-SAME{LITERAL}: dmaCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
// CHECK-SAME: invariantCount([1, 1, 0, 0, 0, 0]) variantCount([1, 1, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0)
