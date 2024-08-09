//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @conv_input_se_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x32xf16>
    DataInfo "output_1" : tensor<1x64x16x32xf16>
  }
  func.func private @conv_input_se_soh_f16_f16_f16(%arg0: memref<1x32x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x64x16x32xf16, #NHWC, @DDR>, %arg2: memref<1x64x16x32xf16, #NHWC, @DDR>) -> (memref<1x64x16x32xf16, #NHWC, @DDR>, memref<1x64x16x32xf16, #NHWC, @DDR>) {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [0] <103424> -> memref<1x32x16x32xi1, #NHWC, [@CMX_NN, 0]>
    %34 = VPURT.DeclareBuffer <CMX_NN> [1] <103424> -> memref<1x32x16x32xi1, #NHWC, [@CMX_NN, 1]>
    %35 = VPURT.DeclareBuffer <CMX_NN> [0] <105472> -> memref<1x1x16x32xi32, #NHWC, [@CMX_NN, 0]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [1] <105472> -> memref<1x1x16x32xi32, #NHWC, [@CMX_NN, 1]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <103424> -> !VPUIP.DistributedBuffer<1x32x32x32xi1, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %38 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <105472> -> !VPUIP.DistributedBuffer<1x1x32x32xi32, {order = #NHWC, strides = [1024, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %24 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, input_se_size = 32 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} input(%11 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) input_sparsity_map(%33 : memref<1x32x16x32xi1, #NHWC, [@CMX_NN, 0]>) input_storage_element_table(%35 : memref<1x1x16x32xi32, #NHWC, [@CMX_NN, 0]>) weights(%5 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_input_sparsity_map(%37 : !VPUIP.DistributedBuffer<1x32x32x32xi1, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_input_storage_element_table(%38 : !VPUIP.DistributedBuffer<1x1x32x32xi32, {order = #NHWC, strides = [1024, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%8 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
    }
    %25 = "VPUMI37XX.DPUVariant"(%24) {end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %26 = VPUMI37XX.DPUInvariant {clean_after = 0 : ui64, input_se_size = 32 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) input_sparsity_map(%34 : memref<1x32x16x32xi1, #NHWC, [@CMX_NN, 1]>) input_storage_element_table(%36 : memref<1x1x16x32xi32, #NHWC, [@CMX_NN, 1]>) weights(%6 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%14 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_input_sparsity_map(%37 : !VPUIP.DistributedBuffer<1x32x32x32xi1, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_input_storage_element_table(%38 : !VPUIP.DistributedBuffer<1x1x32x32xi32, {order = #NHWC, strides = [1024, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%9 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) -> <0:0:1> PPE : {
    }
    %27 = "VPUMI37XX.DPUVariant"(%26) {end = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} : (!VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:1>
    %32 = VPUMI37XX.MappedInference invariants(%24 : !VPURegMapped.Index<0:0:0>) variants(%25 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2 : memref<1x64x16x32xf16, #NHWC, @DDR>, memref<1x64x16x32xf16, #NHWC, @DDR>
  }
}

//CHECK-LABEL: @conv_input_se_soh_f16_f16_f16
//CHECK: %[[VAL24:.*]] = VPUMI37XX.DPUInvariant
//CHECK: %[[VAL25:.*]] = "VPUMI37XX.DPUVariant"
//CHECK: %[[VAL26:.*]] = VPUMI37XX.DPUInvariant
//CHECK: %[[VAL27:.*]] = "VPUMI37XX.DPUVariant"

//CHECK-DAG: %[[INVSEC:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.DPUInvariants"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL24]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL26]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[VARSEC:.*]] = ELFNPU37XX.CreateSection {{.*}} secName = ".text.DPUVariants"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL25]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection %[[VAL27]] : !VPURegMapped.Index<0:0:1>

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32_MULTICAST_BASE>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32_MULTICAST_BASE>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL24]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32_MULTICAST_BASE>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32_MULTICAST_BASE>
//CHECK-DAG ELFNPU37XX.RelocImmOffset baseOp(%[[VAL28]] : !VPURegMapped.Index<0:0:1>) {{.*}} <R_VPU_32>
