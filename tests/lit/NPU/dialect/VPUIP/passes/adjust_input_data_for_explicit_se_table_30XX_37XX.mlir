//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-input-data-for-explicit-se-table %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseConvSETable(%arg0: memref<1x48x10x10xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x48x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x48x3x3xf16, #NHWC, [@CMX_NN, 0]> {
  %cst_weights = const.Declare memref<48x48x1x1xf16, #NHWC> = dense<1.0> : tensor<48x48x1x1xf16, {order = #NHWC}>
  %cst_weights_table = const.Declare memref<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
  %cst_input_sm = const.Declare memref<1x48x12x21xi1, #NHWC> = dense<1> : tensor<1x48x12x21xi1, {order = #NHWC}>
  %cst_input_se = const.Declare memref<1x3x12x21xi32, #NHWC> = dense<1> : tensor<1x3x12x21xi32, {order = #NHWC}>

  %barrier = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x48x6x10xf16, #NHWC, [@CMX_NN, 0]>
  %input_sm = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x48x12x21xi1, #NHWC, [@CMX_NN, 0]>
  %input_se = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x12x21xi32, #NHWC, [@CMX_NN, 0]>
  %output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x48x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %parent_input = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x48x10x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %parent_input_sm = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x48x21x21xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %parent_input_se = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x3x21x21xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
  %parent_output = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x48x20x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
  %weights = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<48x48x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %weights_table = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<48x1x1x4xsi32, [@CMX_NN, 0]>

  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_weights : memref<48x48x1x1xf16, #NHWC>) outputs(%weights : memref<48x48x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<48x48x1x1xf16, #NHWC, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_weights_table : memref<48x1x1x4xsi32>) outputs(%weights_table : memref<48x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<48x1x1x4xsi32, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_input_sm : memref<1x48x12x21xi1, #NHWC>) outputs(%input_sm : memref<1x48x12x21xi1, #NHWC, [@CMX_NN, 0]>) -> memref<1x48x12x21xi1, #NHWC, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NNDMA inputs(%cst_input_se : memref<1x3x12x21xi32, #NHWC>) outputs(%input_se : memref<1x3x12x21xi32, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x12x21xi32, #NHWC, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%barrier : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %out = VPUIP.NCEClusterTask {input_se_size = 16 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
            input(%input : memref<1x48x6x10xf16, #NHWC, [@CMX_NN, 0]>)
            input_sparsity_map(%input_sm : memref<1x48x12x21xi1, #NHWC, [@CMX_NN, 0]>)
            input_storage_element_table(%input_se : memref<1x3x12x21xi32, #NHWC, [@CMX_NN, 0]>)
            weights(%weights : memref<48x48x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%weights_table : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%parent_input : !VPUIP.DistributedBuffer<1x48x10x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            parent_input_sparsity_map(%parent_input_sm : !VPUIP.DistributedBuffer<1x48x21x21xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            parent_input_storage_element_table(%parent_input_se : !VPUIP.DistributedBuffer<1x3x21x21xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>)
            parent_output(%parent_output : !VPUIP.DistributedBuffer<1x48x20x20xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            outputs(%output : memref<1x48x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    -> memref<1x48x3x3xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
  }

  return %arg1 : memref<1x48x3x3xf16, #NHWC, [@CMX_NN, 0]>

  // CHECK:       VPUIP.NCEClusterTask
  // CHECK-NOT:      input({{%.+}} : memref<1x48x6x10xf16, #NHWC, [@CMX_NN, 0]>)
  // CHECK:          input({{%.+}} : memref<1x48x12x21xf16, #NHWC, [@CMX_NN, 0]>)
  // CHECK-NOT:      parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x48x10x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
  // CHECK:          parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x48x21x21xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInput = !VPUIP.DistributedBuffer<
    1x16x80x144xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 40, 144], [1, 16, 40, 144]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]],
    memory_shapes = [[1, 16, 44, 144], [1, 16, 36, 144]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]]
}>

!ParentInputSM = !VPUIP.DistributedBuffer<
    1x16x161x289xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 8, 1],
    compute_shapes = [[1, 16, 88, 289], [1, 16, 73, 289]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]],
    memory_shapes = [[1, 16, 88, 289], [1, 16, 73, 289]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]]
}>

!ParentInputSET = !VPUIP.DistributedBuffer<
    1x1x161x289xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 1, 8, 1],
    compute_shapes = [[1, 1, 88, 289], [1, 1, 73, 289]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]],
    memory_shapes = [[1, 1, 88, 289], [1, 1, 73, 289]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 88, 0]]
}>

!ParentOutput = !VPUIP.DistributedBuffer<
    1x16x160x288xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 80, 288], [1, 16, 80, 288]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 80, 0]],
    memory_shapes = [[1, 16, 80, 288], [1, 16, 80, 288]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 80, 0]]
}>

func.func @SparseConvSETableWithExplictDistributed() -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]> {
    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>
    %input_sm = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x73x289xi1, #NHWC, [@CMX_NN, 0]>
    %input_se = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x73x289xi32, #NHWC, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x16x2x2xf16, #NHWC, [@CMX_NN, 0]>
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>
    %parent_input = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInput
    %parent_input_sm = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputSM
    %parent_input_se = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputSET
    %parent_output = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentOutput

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %out = VPUIP.NCEClusterTask {input_se_size = 16 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
              input(%input : memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>)
              input_sparsity_map(%input_sm : memref<1x16x73x289xi1, #NHWC, [@CMX_NN, 0]>)
              input_storage_element_table(%input_se : memref<1x1x73x289xi32, #NHWC, [@CMX_NN, 0]>)
              weights(%weights : memref<16x16x2x2xf16, #NHWC, [@CMX_NN, 0]>)
              weight_table(%weights_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%parent_input : !ParentInput)
              parent_input_sparsity_map(%parent_input_sm : !ParentInputSM)
              parent_input_storage_element_table(%parent_input_se : !ParentInputSET)
              parent_output(%parent_output : !ParentOutput)
              outputs(%output : memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>)
      -> memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [287, 159, 15], outStart = [0, 80, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }

    return %output : memref<1x16x80x288xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-NOT:      input({{%.+}} : memref<1x16x36x144xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:          input({{%.+}} : memref<1x16x73x289xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-NOT:      parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x16x80x144xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 16, 40, 144], [1, 16, 40, 144]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 40, 0]], memory_shapes = {{\[\[}}1, 16, 44, 144], [1, 16, 36, 144]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 44, 0]]}>)
    // CHECK:          parent_input({{%.+}} : !VPUIP.DistributedBuffer<1x16x161x289xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = {{\[\[}}1, 16, 88, 289], [1, 16, 73, 289]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 88, 0]], memory_shapes = {{\[\[}}1, 16, 88, 289], [1, 16, 73, 289]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 88, 0]]}>)
}
