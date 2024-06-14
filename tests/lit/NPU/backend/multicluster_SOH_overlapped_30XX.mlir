//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x30x30xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

module @TestMultiClusterSOHOverlapped {

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
}

IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @UsedMemory {
        IE.MemoryResource 1048576 bytes of @CMX_NN
    }
}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x32x32xf16>
    }
    outputsInfo : {
        DataInfo "conv" : tensor<1x16x30x30xf16>
    }

  func.func @main(%arg0: memref<1x16x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x16x30x30xf16, #NHWC, @DDR>) -> memref<1x16x30x30xf16, #NHWC, @DDR> {
    %cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare memref<16x16x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %4 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<1x16x17x32xf16, #NHWC, @DDR>) outputs(%5 : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    }
    %6 = VPURT.DeclareBuffer <DDR> <15360> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    %7 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x16x17x32xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>
    }
    %8 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <36416> -> !WeightsDistributed
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<16x16x3x3xf16, #NHWC>) outputs(%8 : !WeightsDistributed) -> !WeightsDistributed
    }
    %9 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !WeightsTableDistributed
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<16x1x1x4xsi32>) outputs(%9 : !WeightsTableDistributed) -> !WeightsTableDistributed
    }
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <36416> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [1] <36416> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [1] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%10 : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%12 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%14 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%2 : !InputDistributed) parent_output(%3 : !OutputDistributed) outputs(%16 : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {cluster_id = 0 : i64, outEnd = [29, 14, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
      } PPE :  {
      }
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%11 : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>) weights(%13 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%15 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%2 : !InputDistributed) parent_output(%3 : !OutputDistributed) outputs(%17 : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]> variants :  {
        DPUTask {cluster_id = 1 : i64, outEnd = [29, 29, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 15, 0]}
      } PPE :  {
      }
    }
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>
    %19 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x15x30xf16, #NHWC, @DDR>
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 0]>) outputs(%19 : memref<1x16x15x30xf16, #NHWC, @DDR>) -> memref<1x16x15x30xf16, #NHWC, @DDR>
    }
    %20 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>
    %21 = VPURT.DeclareBuffer <DDR> <14400> -> memref<1x16x15x30xf16, #NHWC, @DDR>
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<1x16x15x30xf16, #NHWC, [@CMX_NN, 1]>) outputs(%21 : memref<1x16x15x30xf16, #NHWC, @DDR>) -> memref<1x16x15x30xf16, #NHWC, @DDR>
    }
    return %arg1 : memref<1x16x30x30xf16, #NHWC, @DDR>
  }
}

// CHECK:   identifier: "TestMultiClusterSOHOverlapped",

// Load part of input to Cluster 0
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               17,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               17408.0,
// CHECK:               2.0,
// CHECK:               1024.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 0
// CHECK:             locale: "VPU_DDR_Heap",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           },
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               17,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               17408.0,
// CHECK:               2.0,
// CHECK:               1024.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",

// Load part of input to Cluster 1
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               17,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               17408.0,
// CHECK:               2.0,
// CHECK:               1024.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 15360
// CHECK:             locale: "VPU_DDR_Heap",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           },
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               17,
// CHECK:               32
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               17408.0,
// CHECK:               2.0,
// CHECK:               1024.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 0
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",

// Load Weights to Cluster 0&1
// CHECK:         task_type: "NNDMATask",
// CHECK:         task: {
// CHECK:           src: {
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               16,
// CHECK:               3,
// CHECK:               3
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               288.0,
// CHECK:               2.0,
// CHECK:               96.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 0
// CHECK:             locale: "GraphFile",
// CHECK:             locale_index: [
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           },
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               16,
// CHECK:               3,
// CHECK:               3
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               288.0,
// CHECK:               2.0,
// CHECK:               96.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:             data: {
// CHECK:               data_index: 36416
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0,
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",

// Load WeightsTable to Cluster 0&1
// CHECK:         task_type: "NNDMATask",
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               1,
// CHECK:               1,
// CHECK:               4
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               4.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               4.0
// CHECK:             ],
// CHECK:               data_index: 0
// CHECK:             locale: "GraphFile",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "I32",
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               16,
// CHECK:               1,
// CHECK:               1,
// CHECK:               4
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               4.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               16.0,
// CHECK:               4.0
// CHECK:             ],
// CHECK:               data_index: 34112
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0,
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "I32",

// Copy out from Cluster 0
// CHECK:         task_type: "NNDMATask",
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               15,
// CHECK:               30
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               14400.0,
// CHECK:               2.0,
// CHECK:               960.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 17408
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               15,
// CHECK:               30
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               14400.0,
// CHECK:               2.0,
// CHECK:               960.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 0
// CHECK:             locale: "VPU_DDR_Heap",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",

// Copy out from Cluster 1
// CHECK:         task_type: "NNDMATask",
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               15,
// CHECK:               30
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               14400.0,
// CHECK:               2.0,
// CHECK:               960.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 17408
// CHECK:             },
// CHECK:             locale: "VPU_CMX_NN",
// CHECK:             locale_index: [
// CHECK:               1
// CHECK:             ],
// CHECK:             data_dtype: "FP16",
// CHECK:           dst: {
// CHECK:             dimensions: [
// CHECK:               1,
// CHECK:               16,
// CHECK:               15,
// CHECK:               30
// CHECK:             ],
// CHECK:             strides: [
// CHECK:               2.0,
// CHECK:               14400.0,
// CHECK:               2.0,
// CHECK:               960.0,
// CHECK:               32.0
// CHECK:             ],
// CHECK:               data_index: 14400
// CHECK:             locale: "VPU_DDR_Heap",
// CHECK:             locale_index: [
// CHECK:               0
// CHECK:             ],
// CHECK:             data_dtype: "FP16",

// NCE Task on Cluster 0
// CHECK:         task_type: "NCE2Task",
// CHECK:         task: {
// CHECK:           invariant: {
// CHECK:              kernelH: 3,
// CHECK:              kernelW: 3,
// CHECK:              kernel_strideH: 1,
// CHECK:              kernel_strideW: 1,
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                data: {
// CHECK:                  data_index: 0
// CHECK:                },
// CHECK:                locale: "VPU_CMX_NN",
// CHECK:                locale_index: [
// CHECK:                  0
// CHECK:                ],
// CHECK:                data_dtype: "FP16",
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 30,
// CHECK:                 30
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 28800.0,
// CHECK:                 2.0,
// CHECK:                 960.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 17408
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 17,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 17408.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 0
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 15,
// CHECK:                 30
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 14400.0,
// CHECK:                 2.0,
// CHECK:                 960.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 17408
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 3,
// CHECK:                 3
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 288.0,
// CHECK:                 2.0,
// CHECK:                 96.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 36416
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 4.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 4.0
// CHECK:               ],
// CHECK:                 data_index: 34112
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_end_X: 29,
// CHECK:               workload_end_Y: 14,
// CHECK:               workload_end_Z: 15

// NCE Task on Cluster 1
// CHECK:         task_type: "NCE2Task",
// CHECK:         task: {
// CHECK:           invariant: {
// CHECK:              kernelH: 3,
// CHECK:              kernelW: 3,
// CHECK:              kernel_strideH: 1,
// CHECK:              kernel_strideW: 1,
// CHECK:             parent_input_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 32,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 32768.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                data: {
// CHECK:                  data_index: 0
// CHECK:                },
// CHECK:                locale: "VPU_CMX_NN",
// CHECK:                locale_index: [
// CHECK:                  0
// CHECK:                ],
// CHECK:                data_dtype: "FP16",
// CHECK:             parent_output_tensor: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 30,
// CHECK:                 30
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 28800.0,
// CHECK:                 2.0,
// CHECK:                 960.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 17408
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 0
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             input_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 17,
// CHECK:                 32
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 17408.0,
// CHECK:                 2.0,
// CHECK:                 1024.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 0
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             output_data: {
// CHECK:               dimensions: [
// CHECK:                 1,
// CHECK:                 16,
// CHECK:                 15,
// CHECK:                 30
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 14400.0,
// CHECK:                 2.0,
// CHECK:                 960.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 17408
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:             weights_data: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 16,
// CHECK:                 3,
// CHECK:                 3
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 2.0,
// CHECK:                 288.0,
// CHECK:                 2.0,
// CHECK:                 96.0,
// CHECK:                 32.0
// CHECK:               ],
// CHECK:                 data_index: 36416
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "FP16",
// CHECK:             weights_table: {
// CHECK:               dimensions: [
// CHECK:                 16,
// CHECK:                 1,
// CHECK:                 1,
// CHECK:                 4
// CHECK:               ],
// CHECK:               strides: [
// CHECK:                 4.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 16.0,
// CHECK:                 4.0
// CHECK:               ],
// CHECK:                 data_index: 34112
// CHECK:               locale: "VPU_CMX_NN",
// CHECK:               locale_index: [
// CHECK:                 1
// CHECK:               ],
// CHECK:               data_dtype: "I32",
// CHECK:           variant: [
// CHECK:               mpe_mode: "VECTOR_FP16",
// CHECK:               workload_start_Y: 15,
// CHECK:               workload_end_X: 29,
// CHECK:               workload_end_Y: 29,
// CHECK:               workload_end_Z: 15
