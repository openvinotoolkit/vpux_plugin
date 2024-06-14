//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="function-outlining=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @Convolution
module @Convolution attributes {VPU.arch = #VPU.arch_kind<NPU30XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
    // CHECK-DAG: {{  }}module @UsedMemory
    // CHECK-DAG: {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR
    // CHECK-DAG: {{  }}IE.PipelineOptions @Options
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantSum : 512
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantCount : 512
    // CHECK-DAG: {{  }}IE.ExecutorResource 1 of @DMA_NN
    // CHECK-DAG: {{  }}IE.ExecutorResource 16 of @SHAVE_UPA
    // CHECK-DAG: {{  }}IE.TileResource 4 of @NCE at 7.000000e+02 MHz
    // CHECK-DAG: {{    }}IE.ExecutorResource 5 of @DPU
    // CHECK-DAG: {{    }}builtin.module @UsedMemory
    // CHECK-DAG: {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x62x62xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x48x60x60xf16, @DDR>) -> memref<1x48x60x60xf16, @DDR> {
    func.func @main(%arg0: memref<1x3x62x62xf16>, %arg1: memref<1x48x60x60xf16>) -> memref<1x48x60x60xf16> {
        %cst = const.Declare memref<48x16x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %cst_0 = const.Declare memref<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %alloc = memref.alloc() : memref<1x16x62x62xf16>
        %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x62x62xf16>) outputs(%alloc : memref<1x16x62x62xf16>) -> memref<1x16x62x62xf16>
        %alloc_1 = memref.alloc() : memref<1x16x62x62xf16, #NHWC>
        %1 = VPUIP.PermuteUPA {order_value = #NHWC} inputs(%0 : memref<1x16x62x62xf16>) outputs(%alloc_1 : memref<1x16x62x62xf16, #NHWC>) -> memref<1x16x62x62xf16, #NHWC>
        %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
        %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x16x62x62xf16, #NHWC>) outputs(%2 as %arg3: memref<1x16x62x62xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
            %13 = VPUIP.Copy inputs(%arg2 : memref<1x16x62x62xf16, #NHWC>) outputs(%arg3 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>) -> memref<1x16x62x62xf16, #NHWC, @CMX_NN>
        }
        %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<48x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
        %5 = VPUIP.NCEClusterTiling inputs(%cst as %arg2: memref<48x16x3x3xf16, #NHWC>) outputs(%4 as %arg3: memref<48x16x3x3xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<48x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
            %13 = VPUIP.Copy inputs(%arg2 : memref<48x16x3x3xf16, #NHWC>) outputs(%arg3 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>) -> memref<48x16x3x3xf16, #NHWC, @CMX_NN>
        }
        %6 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
        %7 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<48x1x1x4xsi32>) outputs(%6 as %arg3: memref<48x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
            %13 = VPUIP.Copy inputs(%arg2 : memref<48x1x1x4xsi32>) outputs(%arg3 : memref<48x1x1x4xsi32, @CMX_NN>) -> memref<48x1x1x4xsi32, @CMX_NN>
        }
        %8 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
        %9 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x16x62x62xf16, #NHWC, @CMX_NN>, %5 as %arg3: memref<48x16x3x3xf16, #NHWC, @CMX_NN>, %7 as %arg4: memref<48x1x1x4xsi32, @CMX_NN>) outputs(%8 as %arg5: memref<1x48x60x60xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
            %13 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 21126 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<48x16x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<48x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x16x62x62xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) -> memref<1x48x60x60xf16, #NHWC, @CMX_NN> variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 2, 47], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 2, 47], outStart = [16, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 2, 47], outStart = [32, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 2, 47], outStart = [48, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 5, 47], outStart = [0, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 5, 47], outStart = [16, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 5, 47], outStart = [32, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 5, 47], outStart = [48, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 8, 47], outStart = [0, 6, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 8, 47], outStart = [16, 6, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 8, 47], outStart = [32, 6, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 8, 47], outStart = [48, 6, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 11, 47], outStart = [0, 9, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 11, 47], outStart = [16, 9, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 11, 47], outStart = [32, 9, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 11, 47], outStart = [48, 9, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 14, 47], outStart = [0, 12, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 14, 47], outStart = [16, 12, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 14, 47], outStart = [32, 12, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 14, 47], outStart = [48, 12, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 17, 47], outStart = [0, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 17, 47], outStart = [16, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 17, 47], outStart = [32, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 17, 47], outStart = [48, 15, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 20, 47], outStart = [0, 18, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 20, 47], outStart = [16, 18, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 20, 47], outStart = [32, 18, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 20, 47], outStart = [48, 18, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 23, 47], outStart = [0, 21, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 23, 47], outStart = [16, 21, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 23, 47], outStart = [32, 21, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 23, 47], outStart = [48, 21, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 26, 47], outStart = [0, 24, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 26, 47], outStart = [16, 24, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 26, 47], outStart = [32, 24, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 26, 47], outStart = [48, 24, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 29, 47], outStart = [0, 27, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 29, 47], outStart = [16, 27, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 29, 47], outStart = [32, 27, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 29, 47], outStart = [48, 27, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 32, 47], outStart = [0, 30, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 32, 47], outStart = [16, 30, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 32, 47], outStart = [32, 30, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 32, 47], outStart = [48, 30, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 35, 47], outStart = [0, 33, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 35, 47], outStart = [16, 33, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 35, 47], outStart = [32, 33, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 35, 47], outStart = [48, 33, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 38, 47], outStart = [0, 36, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 38, 47], outStart = [16, 36, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 38, 47], outStart = [32, 36, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 38, 47], outStart = [48, 36, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 41, 47], outStart = [0, 39, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 41, 47], outStart = [16, 39, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 41, 47], outStart = [32, 39, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 41, 47], outStart = [48, 39, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 44, 47], outStart = [0, 42, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 44, 47], outStart = [16, 42, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 44, 47], outStart = [32, 42, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 44, 47], outStart = [48, 42, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 47, 47], outStart = [0, 45, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 47, 47], outStart = [16, 45, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 47, 47], outStart = [32, 45, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 47, 47], outStart = [48, 45, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 50, 47], outStart = [0, 48, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 50, 47], outStart = [16, 48, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 50, 47], outStart = [32, 48, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 50, 47], outStart = [48, 48, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 53, 47], outStart = [0, 51, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 53, 47], outStart = [16, 51, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 53, 47], outStart = [32, 51, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 53, 47], outStart = [48, 51, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 56, 47], outStart = [0, 54, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 56, 47], outStart = [16, 54, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 56, 47], outStart = [32, 54, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 56, 47], outStart = [48, 54, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 59, 47], outStart = [0, 57, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 59, 47], outStart = [16, 57, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [47, 59, 47], outStart = [32, 57, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [59, 59, 47], outStart = [48, 57, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            } PPE : {
            }
        }
        %alloc_2 = memref.alloc() : memref<1x48x60x60xf16, #NHWC>
        %10 = VPUIP.NCEClusterTiling inputs(%9 as %arg2: memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs(%alloc_2 as %arg3: memref<1x48x60x60xf16, #NHWC>) -> memref<1x48x60x60xf16, #NHWC> {
            %13 = VPUIP.Copy inputs(%arg2 : memref<1x48x60x60xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x48x60x60xf16, #NHWC>) -> memref<1x48x60x60xf16, #NHWC>
        }
        %alloc_3 = memref.alloc() : memref<1x48x60x60xf16>
        %11 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%10 : memref<1x48x60x60xf16, #NHWC>) outputs(%alloc_3 : memref<1x48x60x60xf16>) -> memref<1x48x60x60xf16>
        %12 = VPUIP.Copy inputs(%11 : memref<1x48x60x60xf16>) outputs(%arg1 : memref<1x48x60x60xf16>) -> memref<1x48x60x60xf16>
        return %12 : memref<1x48x60x60xf16>


        // CHECK-DAG:   const.Declare memref<1x1x1x7296xf16>

        // CHECK-DAG:   [[OUT0_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x48x60x60xf16, @DDR>
        // CHECK-DAG:   [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <123008> -> memref<1x48x15x60xf16, #NHWC, @DDR>
        // CHECK-DAG:   [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <209408> -> memref<1x48x15x60xf16, #NHWC, @DDR>
        // CHECK-DAG:   [[OUT3_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <295808> -> memref<1x48x15x60xf16, #NHWC, @DDR>
        // CHECK-DAG:   [[OUT4_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <382208> -> memref<1x48x15x60xf16, #NHWC, @DDR>

        // CHECK-DAG:   [[IN0_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 0]>
        // CHECK-DAG:   [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 1]>
        // CHECK-DAG:   [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 2]>
        // CHECK-DAG:   [[IN3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 3]>
        // CHECK-DAG:   [[IN0_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <123008> -> memref<1x48x60x60xf16, #NHWC, @DDR>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x16x62xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x16x62xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x16x62xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 2]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_3:%.*]] : memref<1x16x14x62xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK-SAME:      [[weight_3:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK-SAME:      [[weight_table_3:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 3]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_3:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN0_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      outputs([[OUT1_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN1_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      outputs([[OUT2_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN2_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK-SAME:      outputs([[OUT3_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task updates([[barrier_2:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN3_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK-SAME:      outputs([[OUT4_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task waits([[barrier_2:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.PermuteUPA {order_value = #NWCH} inputs([[IN0_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
        // CHECK-SAME:      outputs([[OUT0_DDR]] : memref<1x48x60x60xf16, @DDR>)

        // CHECK:   return %arg1 : memref<1x48x60x60xf16, @DDR>
    }
}

// -----

// CHECK-LABEL: @SoftMax
module @SoftMax attributes {VPU.arch = #VPU.arch_kind<NPU30XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
    // CHECK-DAG: {{  }}module @UsedMemory
    // CHECK-DAG: {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR
    // CHECK-DAG: {{  }}IE.PipelineOptions @Options
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantSum : 512
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantCount : 512
    // CHECK-DAG: {{  }}IE.ExecutorResource 1 of @DMA_NN
    // CHECK-DAG: {{  }}IE.ExecutorResource 16 of @SHAVE_UPA
    // CHECK-DAG: {{  }}IE.TileResource 4 of @NCE at 7.000000e+02 MHz
    // CHECK-DAG: {{    }}IE.ExecutorResource 5 of @DPU
    // CHECK-DAG: {{    }}builtin.module @UsedMemory
    // CHECK-DAG: {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x1000xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR> {
    func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
        %alloc = memref.alloc() : memref<1x1x1x1000xf16>
        %1 = VPUIP.SoftMaxUPA {axisInd = 3 : i64} inputs(%0 : memref<1x1x1x1000xf16>) outputs(%alloc : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        %2 = VPUIP.GenericReshape inputs(%1 : memref<1x1x1x1000xf16>) -> memref<1x1000xf16>
        %3 = VPUIP.Copy inputs(%2 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        return %3 : memref<1x1000xf16>

        // CHECK-DAG:   [[IN1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK-DAG:   [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>

        // CHECK-DAG:   [[VAR1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK-DAG:   [[VAR2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1000xf16, @DDR>

        // CHECK-DAG:   [[VAR3:%.+]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
        // CHECK-NEXT:  VPURT.Task
        // CHECK-SAME:              updates([[VAR3]] : !VPURT.Barrier)
        // CHECK-NEXT:  VPUIP.SoftMaxUPA
        // CHECK-SAME:              axisInd = 3
        // CHECK-SAME:              inputs([[IN1]] : memref<1x1x1x1000xf16, @DDR>
        // CHECK-SAME:              outputs([[VAR1]] : memref<1x1x1x1000xf16, @DDR>

        // CHECK:  VPURT.Task
        // CHECK-SAME:              waits([[VAR3]] : !VPURT.Barrier)
        // CHECK-NEXT:  VPUIP.NNDMA
        // CHECK-SAME:              inputs([[VAR2]] : memref<1x1000xf16, @DDR>)
        // CHECK-SAME:              outputs([[OUT1]] : memref<1x1000xf16, @DDR>)

        // CHECK:  return [[ARG1]] : memref<1x1000xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions attributes {VPU.arch = #VPU.arch_kind<NPU30XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
    // CHECK-DAG: {{  }}module @UsedMemory
    // CHECK-DAG: {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR

    // CHECK-DAG: {{  }}IE.PipelineOptions @Options
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantSum : 512
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantCount : 512

    // CHECK-DAG: {{  }}IE.TileResource
    // CHECK-DAG: {{    }}builtin.module @UsedMemory
    // CHECK-DAG: {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x6x6xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x4x4xf16>
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo1(%arg0: memref<1x16x6x6xf16>, %arg1: memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16> {
        %cst = const.Declare memref<32x16x3x3xf16, #NHWC>
        = dense<1.000000e+00> : tensor<32x16x3x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
        %cst_0 = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

        %alloc = memref.alloc() : memref<1x16x6x6xf16, #NHWC>
        %0 = VPUIP.PermuteUPA {order_value = #NHWC}
                                inputs(%arg0 : memref<1x16x6x6xf16>)
                                outputs(%alloc : memref<1x16x6x6xf16, #NHWC>) -> memref<1x16x6x6xf16, #NHWC>
        %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
        %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x16x6x6xf16, #NHWC>)
                                    outputs(%1 as %arg3: memref<1x16x6x6xf16, #NHWC, @CMX_NN>)
                                    -> !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN,
                                            {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
          %12 = VPUIP.Copy inputs(%arg2 : memref<1x16x6x6xf16, #NHWC>)
          outputs(%arg3 : memref<1x16x6x6xf16, #NHWC, @CMX_NN>)
          -> memref<1x16x6x6xf16, #NHWC, @CMX_NN>
        }
        %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN,
                                            {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>
        %4 = VPUIP.NCEClusterTiling inputs(%cst as %arg2: memref<32x16x3x3xf16, #NHWC>)
                                    outputs(%3 as %arg3: memref<32x16x3x3xf16, #NHWC, @CMX_NN>)
                                    -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
          %12 = VPUIP.Copy inputs(%arg2 : memref<32x16x3x3xf16, #NHWC>)
          outputs(%arg3 : memref<32x16x3x3xf16, #NHWC, @CMX_NN>)
          -> memref<32x16x3x3xf16, #NHWC, @CMX_NN>
        }
        %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
                                        {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>
        %6 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<32x1x1x4xsi32>) outputs(%5 as %arg3: memref<32x1x1x4xsi32, @CMX_NN>)
                                    -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
                                    {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}> {
          %12 = VPUIP.Copy inputs(%arg2 : memref<32x1x1x4xsi32>) outputs(%arg3 : memref<32x1x1x4xsi32, @CMX_NN>) -> memref<32x1x1x4xsi32, @CMX_NN>
        }
        %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NHWC, @CMX_NN,
                                        {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
        %8 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x16x6x6xf16, #NHWC, @CMX_NN>,
                                           %4 as %arg3: memref<32x16x3x3xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
                                    outputs(%7 as %arg5: memref<1x32x4x4xf16, #NHWC, @CMX_NN>)
                                        -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NHWC, @CMX_NN,
                                        {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
          %12 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                            kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1256 : i64, task_type = #VPUIP.nce_task_type<CONV>}
                                        input(%arg2 : memref<1x16x6x6xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<32x16x3x3xf16, #NHWC, @CMX_NN>)
                                        weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x16x6x6xf16, #NHWC, @CMX_NN>)
                                        parent_output(%arg5 : memref<1x32x4x4xf16, #NHWC, @CMX_NN>)
                                        outputs(%arg5 : memref<1x32x4x4xf16, #NHWC, @CMX_NN>) -> memref<1x32x4x4xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 0, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 1, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 2, 31], outStart = [0, 2, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 3, 31], outStart = [0, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 0, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 1, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 2, 31], outStart = [0, 2, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 3, 31], outStart = [0, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 0, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 1, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 2, 31], outStart = [0, 2, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 3, 31], outStart = [0, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 0, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 1, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 2, 31], outStart = [0, 2, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 3 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [3, 3, 31], outStart = [0, 3, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
          }
        }

        %alloc_1 = memref.alloc() : memref<1x32x4x4xf16, #NHWC>
        %9 = VPUIP.NCEClusterTiling inputs(%8 as %arg2: memref<1x32x4x4xf16, #NHWC, @CMX_NN>)
                                    outputs(%alloc_1 as %arg3: memref<1x32x4x4xf16, #NHWC>)
                                    -> memref<1x32x4x4xf16, #NHWC> {
          %12 = VPUIP.Copy inputs(%arg2 : memref<1x32x4x4xf16, #NHWC, @CMX_NN>)
          outputs(%arg3 : memref<1x32x4x4xf16, #NHWC>) -> memref<1x32x4x4xf16, #NHWC>
        }

        %alloc_2 = memref.alloc() : memref<1x32x4x4xf16>
        %10 = VPUIP.PermuteUPA {order_value = #NWCH}
                                inputs(%9 : memref<1x32x4x4xf16, #NHWC>) outputs(%alloc_2 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        %11 = VPUIP.Copy inputs(%10 : memref<1x32x4x4xf16>) outputs(%arg1 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        return %11 : memref<1x32x4x4xf16>
    }

    // CHECK-NOT: func.func private @foo2
    func.func private @foo2(%arg0: memref<1x32x4x4xf16>, %arg1: memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16> {
        %alloc = memref.alloc() : memref<1x32x4x4xf16>
        %0 = VPUIP.SoftMaxUPA {axisInd = 3 : i64} inputs(%arg0 : memref<1x32x4x4xf16>) outputs(%alloc : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        %1 = VPUIP.Copy inputs(%0 : memref<1x32x4x4xf16>) outputs(%arg1 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        return %1 : memref<1x32x4x4xf16>
    }

    func.func @main(%arg0: memref<1x16x6x6xf16>, %arg1: memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16> {
        %alloc = memref.alloc() : memref<1x32x4x4xf16>
        %0 = call @foo1(%arg0, %alloc) : (memref<1x16x6x6xf16>, memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        %alloc_0 = memref.alloc() : memref<1x32x4x4xf16>
        %1 = call @foo2(%0, %alloc_0) : (memref<1x32x4x4xf16>, memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        %2 = VPUIP.Copy inputs(%1 : memref<1x32x4x4xf16>) outputs(%arg1 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        return %2 : memref<1x32x4x4xf16>
    }

    // CHECK-LABEL: @main
        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.PermuteDMA


        // CHECK: VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:   VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK: VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:   VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK: VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:   VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK: VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:   VPUIP.NCEClusterTask
        // CHECK-SAME: CONV


        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.PermuteDMA

        // CHECK: VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:   VPUIP.SoftMaxUPA

        // CHECK: return {{[^:]+}} : memref<1x32x4x4xf16, @DDR>
}
