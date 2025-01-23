//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
func.func @ConvMulticlusterSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [80, 64, 3, 3],
        strides = [1, 1]}
      -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

// CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
// CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]


// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x28x28xf16, {order = #NHWC}>)
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<80x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<80x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[CONV:%.*]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>
// CHECK-SAME{LITERAL}:                                                -> tensor<1x80x28x28xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvDistributedTensorSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>

func.func @DepthConvDistributedTensorSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

// CHECK:    [[WEIGHTSTABLE:%.*]]  = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
// CHECK:    [[WEIGHTS:%.*]]  = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<32x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[DEPTH_CONV:%.*]] = VPU.NCE.DepthConvolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[DEPTH_CONV]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:                return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @MaxPoolMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
            } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[MAXPOOL:%.*]] = VPU.NCE.MaxPool([[IN_CP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolMulticlusterHKSwitch
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolMulticlusterHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
            } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[MAXPOOL:%.*]] = VPU.NCE.MaxPool([[IN_CP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @EltwiseAddMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[ARG1]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[IN_CP0]], [[IN_CP1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[ELTWISE]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                    compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                    memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @AvgPoolMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
            } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]}>
// CHECK:               [[AVGPOOL:%.*]] = VPU.NCE.AveragePool([[IN_CP0]]) {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[AVGPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SparseConvMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.*]]: tensor<1x64x28x28xi1, {order = #NHWC}>)
func.func @SparseConvMulticlusterSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
            } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>


// CHECK:      [[INPUT_SPARSE:%.*]] = VPU.GroupSparseTensor([[ARG0]], [[ARG1]]) -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>
// CHECK:      [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
// CHECK:      [[WEIGHTS_SM:%.*]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
// CHECK:      [[WEIGHTS_SPARSE:%.*]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights>
// CHECK:      [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[INPUT_SPARSE]] : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>
// CHECK-SAME{LITERAL}:                                  -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}>,
// CHECK-SAME{LITERAL}:                                  sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}>>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS_SPARSE]] : !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights>
// CHECK-SAME{LITERAL}:                                  -> !VPU.SparseTensor<data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                  sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, is_weights>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTS_TABLE]] : tensor<80x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                  -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[CONV:%.*]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                  -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>,
// CHECK-SAME{LITERAL}:                                  sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1],
// CHECK-SAME{LITERAL}:                                  num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>> {
// CHECK:                                                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
// CHECK:                                               }
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[CONV]] : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>,
// CHECK-SAME{LITERAL}:                                  sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                  compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                  memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
// CHECK-SAME{LITERAL}:                                  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]}>>
// CHECK-SAME{LITERAL}:                                  -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
// CHECK:               return [[OUT_CP]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVN6SOK4
module @MVN6SOK4 {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @MVN6SOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOK(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x32x15x64xf16>
// CHECK-SAME{LITERAL}:                                                          -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                          compute_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
// CHECK-SAME{LITERAL}:                                                          memory_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
// CHECK:               [[MVN6:%.*]] = VPU.MVN6([[IN_CP0]]) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} :
// CHECK-SAME{LITERAL}:                                                          !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                          compute_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
// CHECK-SAME{LITERAL}:                                                          memory_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                          -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                          compute_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
// CHECK-SAME{LITERAL}:                                                          memory_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
// CHECK:               [[CP_OUT:%.*]] = VPU.UnrolledType([[MVN6]] : !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                          compute_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
// CHECK-SAME{LITERAL}:                                                          memory_shapes = [[1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64], [1, 8, 15, 64]],
// CHECK-SAME{LITERAL}:                                                          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                          -> tensor<1x32x15x64xf16>
// CHECK:               return [[CP_OUT]] : tensor<1x32x15x64xf16>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKConvOutputSegmented
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x64x64x64xf16, {order = #NHWC}>
func.func @UnrollSOKConvOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]
            } -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %mvn = VPU.MVN(%conv) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NHWC}>

// CHECK: [[WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<64x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTS_TABLE]] : tensor<64x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>
// CHECK:               [[CONV:%.*]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK:               [[INTER_CP0:%.*]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK:               [[INTER_CP1:%.*]] = VPU.UnrolledType([[INTER_CP0]] : tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>
// CHECK:               [[MVN_OUT:%.*]] = VPU.MVN([[INTER_CP1]]) {across_channels = false, eps = 1.000000e-04 : f64, normalize_variance = true} :
// CHECK-SAME{LITERAL}:                                                     !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[MVN_OUT]] : !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x64x64x64xf16, {order = #NHWC}>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKDWConvInputOutputDuplicated
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x1x320x1xf16>
func.func @UnrollSOKDWConvInputOutputDuplicated(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<320x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x320xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 320]>, #const.Reshape<[1, 320, 1, 1]>, #const.Reshape<[320, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[320, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<320x1x1x4xsi32> = dense<10> : tensor<320x1x1x4xsi32>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true}
            : tensor<1x1x320x1xf16> -> tensor<1x1x320x1xf16>

    %reshape = VPU.AffineReshape(%mvn) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
            : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    %cast = VPU.PermuteCast(%reshape) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %dwconv = VPU.NCE.DepthConvolution(%cast, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%dwconv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

// CHECK:    [[WEIGHTS:%.*]] = const.Declare tensor<320x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x320xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 320]>, #const.Reshape<[1, 320, 1, 1]>, #const.Reshape<[320, 1, 1, 1]>,
// CHECK:                                                            #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[320, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
// CHECK:    [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<320x1x1x4xsi32> = dense<10> : tensor<320x1x1x4xsi32>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x320x1xf16>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_MVN:%.*]] = VPU.MVN([[IN_CP0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} :
// CHECK-SAME{LITERAL}:                                                     !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_MVN_CP:%.*]] = VPU.UnrolledType([[OUT_MVN]] : !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x1x320x1xf16>
// CHECK:               [[OUT_AFFINE_RESHAPE:%.*]] = VPU.AffineReshape([[OUT_MVN_CP]])
// CHECK{LITERAL}:                                                      {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>
// CHECK:               [[OUT_PERMUTE_CAST:%.*]] = VPU.PermuteCast([[OUT_AFFINE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>
// CHECK:               [[INTER_CP1:%.*]] = VPU.UnrolledType([[OUT_PERMUTE_CAST:%.*]] : tensor<1x320x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<320x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<320x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTS_TABLE]] : tensor<320x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<320x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]]}>
// CHECK:               [[OUT_DCONV:%.*]] = VPU.NCE.DepthConvolution([[INTER_CP1]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME{LITERAL}:                                                     ppe = #VPU.PPEStub<>, rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
// CHECK:               [[INTER_CP2:%.*]] = VPU.UnrolledType([[OUT_DCONV]] : !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x320x1x1xf16, {order = #NHWC}>
// CHECK:               [[INTER_CP3:%.*]] = VPU.UnrolledType([[INTER_CP2]] : tensor<1x320x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
// CHECK:               [[OUT_SIGMOID:%.*]] = VPU.Sigmoid([[INTER_CP3]]) : !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[OUT_SIGMOID:%.*]] : !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
// CHECK-SAME{LITERAL}:                                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
// CHECK-SAME{LITERAL}:                                                     memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x320x1x1xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x320x1x1xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEInterpolateMulticlusterClustering
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @NCEInterpolateMulticlusterClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

// CHECK: [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
// CHECK: [[INPUT_SM:%.*]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>
// CHECK: [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 1, 1], seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
// CHECK:                                       scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>, seDepth = 1 : i64, seSize = 16 : i64} -> tensor<1x1x2x2xi32, {order = #NHWC}>
// CHECK: [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %cst_1, %0) {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
// CHECK:                                       scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>}
// CHECK:                                       -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x2x2xi1>, storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>, #VPU.SEInterpolate<mode = <NEAREST>,
// CHECK:                                       coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

// CHECK:               [[INPUT_CMX:%.*]] = VPU.UnrolledType([[INPUT_SPARSE]] : !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x2x2xi1>,
// CHECK-SAME{LITERAL}:                                         storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>, #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
// CHECK-SAME{LITERAL}:                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>
// CHECK-SAME{LITERAL}:                                         -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                         sparsity_map=!VPU.DistributedTensor<1x16x2x2xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                         storage_element_table=!VPU.DistributedTensor<1x1x2x2xi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
// CHECK-SAME{LITERAL}:                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>
// CHECK:               [[INPUT_WEIGHTS:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<16x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                         -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[INPUT_WEIGHTS_TABLE:%.*]] = VPU.UnrolledType([[WEIGHTS_TABLE]] : tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                         -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_INTERP:%.*]] = VPU.NCE.Interpolate([[INPUT_CMX]], [[INPUT_WEIGHTS]], [[INPUT_WEIGHTS_TABLE]]) {mode = #VPU.nce_interpolate_mode<NEAREST>,
// CHECK-SAME{LITERAL}:                                         rawFilterShape = [16, 16, 1, 1], scales_attr = [2, 2], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                         -> !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[OUT_INTERP]] : !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                         compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
// CHECK-SAME{LITERAL}:                                         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                         memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
// CHECK-SAME{LITERAL}:                                         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                         -> tensor<1x16x2x2xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x16x2x2xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @Permute {
IE.TileResource 2 of @NCE at 1.300000e+03 MHz

// CHECK-LABEL: @NCEPermuteCompressConv
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x3x224x224xf16>
func.func @NCEPermuteCompressConv(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPEStub<>} -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.CompressConvolution(%0, %WEIGHTS, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        rawFilterShape = [16, 4, 3, 3],
        strides = [2, 2],
        ppe = #VPU.PPEStub<>} -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %1 : tensor<1x16x112x112xf16, {order = #NHWC}>

// CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
// CHECK:   [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

// CHECK:               [[INPUT_CP:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x3x224x224xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
// CHECK:               [[OUT_PERMUTE:%.*]] = VPU.NCE.Permute([[INPUT_CP]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
// CHECK:               [[PERMUTE_CP:%.*]] = VPU.UnrolledType([[OUT_PERMUTE]] : !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
// CHECK:               [[INPUT_CMX:%.*]] = VPU.UnrolledType([[PERMUTE_CP]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
// CHECK:               [[IN_WEIGHTS:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<16x1x1x48x!qElemType, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<16x1x1x48x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[16, 1, 1, 48], [16, 1, 1, 48]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[16, 1, 1, 48], [16, 1, 1, 48]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_WT_TABLE:%.*]] = VPU.UnrolledType([[WEIGHTS_TABLE]] : tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CCONV:%.*]] = VPU.NCE.CompressConvolution([[INPUT_CMX]], [[IN_WEIGHTS]], [[IN_WT_TABLE]]) {cm_sp_pattern = 7 : i64, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
// CHECK-SAME{LITERAL}:                                     ppe = #VPU.PPEStub<>, rawFilterShape = [16, 4, 3, 3], strides = [2, 2]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 56, 112], [1, 16, 56, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 56, 112], [1, 16, 56, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[OUT_CCONV]] : !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 56, 112], [1, 16, 56, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 56, 112], [1, 16, 56, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x16x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x16x112x112xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SubtractSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @SubtractSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

// CHECK:               [[INPUT0_CP:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x32x44x44xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[INPUT1_CP:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x1x1x44xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_SUBST:%.+]] = VPU.Subtract([[INPUT0_CP]], [[INPUT1_CP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>,
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_SUBST]] : !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x44x44xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x32x44x44xf16>

}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @AddSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @AddSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

// CHECK:               [[INPUT0_CP:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x32x44x44xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[INPUT1_CP:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x1x44x44xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[OUT_ADD:%.+]]   = VPU.Add([[INPUT0_CP]], [[INPUT1_CP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>,
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[OUT_CP:%.+]]    = VPU.UnrolledType([[OUT_ADD]] : !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x44x44xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x32x44x44xf16>

}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @AddSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @AddSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xf16>

    return %0 : tensor<1x32x44x44xf16>

// CHECK:               [[INPUT0_CP:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x32x44x44xf16>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[INPUT1_CP:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x1x1x44xf16>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_ADD:%.+]]   = VPU.Add([[INPUT0_CP]], [[INPUT1_CP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME{LITERAL}:                                                    !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>,
// CHECK-SAME{LITERAL}:                                                    !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK:               [[OUT_CP:%.+]]    = VPU.UnrolledType([[OUT_ADD]] : !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x32x44x44xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x32x44x44xf16>

}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @FloorSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @FloorSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

// CHECK:               [[INPUT_CP:%.+]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x16x16x512xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
// CHECK:               [[OUT_FLOOR:%.+]] = VPU.Floor([[INPUT_CP]]) : !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_FLOOR]] : !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x16x16x512xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x16x16x512xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @FloorSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @FloorSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

// CHECK:               [[INPUT_CP:%.+]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x1x1x513xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_FLOOR:%.+]] = VPU.Floor([[INPUT_CP]]) : !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_FLOOR]] : !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x1x1x513xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x1x1x513xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateClustering
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateClustering(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>


    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>

// CHECK:               [[LHS_CP:%.+]] = VPU.UnrolledType([[LHS]] : tensor<1x64x16x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[RHS_CP:%.+]] = VPU.UnrolledType([[RHS]] : tensor<1x64x16x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[LHS_SCALES_CP:%.+]] = VPU.UnrolledType([[LHS_SCALES]] : tensor<1x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[RHS_SCALES_CP:%.+]] = VPU.UnrolledType([[RHS_SCALES]] : tensor<1x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_ACC:%.+]] = VPU.Accumulate([[LHS_CP]], [[RHS_CP]], [[LHS_SCALES_CP]], [[RHS_SCALES_CP]]) : !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     emory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1], [1, 64, 1, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_ACC]] : !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1], [1, 64, 16, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x16x1xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadSOH4
module @PadSOH4 {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @PadSwSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x32x50xf16>)
func.func @PadSwSOH(%arg0: tensor<1x16x32x50xf16>) -> tensor<1x17x32x60xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 1, 0, 10]} : tensor<1x16x32x50xf16> -> tensor<1x17x32x60xf16>
    return %0 : tensor<1x17x32x60xf16>

// CHECK:               [[INPUT_CP:%.+]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x16x32x50xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x32x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>
// CHECK:               [[OUT_PAD:%.+]] = VPU.Pad([[INPUT_CP]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 1, 0, 10]} :
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x16x32x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x17x32x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_PAD]] : !VPU.DistributedTensor<1x17x32x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x17x32x60xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x17x32x60xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadSOK4
module @PadSOK4 {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @PadSwSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x30x50xf16>)
func.func @PadSwSOK(%arg0: tensor<1x16x30x50xf16>) -> tensor<1x16x33x53xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} : tensor<1x16x30x50xf16> -> tensor<1x16x33x53xf16>
    return %0 : tensor<1x16x33x53xf16>

// CHECK:               [[INPUT_CP:%.+]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x16x30x50xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>
// CHECK:               [[OUT_PAD:%.+]] = VPU.Pad([[INPUT_CP]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} :
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_PAD]] : !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x16x33x53xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x16x33x53xf16>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseInputsSameOffsets
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>)
func.func @EltwiseInputsSameOffsets(%arg0: tensor<1x128x72x72xf16, {order = #NHWC}>, %arg1: tensor<1x128x72x72xf16, {order = #NHWC}>) -> tensor<1x128x72x72xf16> {
    %cst = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%0, %cst_1, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>} -> tensor<1x128x72x72xf16>

    return %3 : tensor<1x128x72x72xf16>

// CHECK: [[WT0:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[WT_TABLE0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
// CHECK: [[WT1:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[WT_TABLE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

// CHECK:               [[INPUT0_CP:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]]}>
// CHECK:               [[WT0_CP:%.+]] = VPU.UnrolledType([[WT0]] : tensor<64x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[WT_TABLE0_CP:%.+]] = VPU.UnrolledType([[WT_TABLE0]] : tensor<64x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT0_CP]], [[WT0_CP]], [[WT_TABLE0_CP]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME{LITERAL}:                                     ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK:               [[CONV_CP0:%.+]] = VPU.UnrolledType([[OUT_CONV]] : !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK:               [[CONV_CP1:%.+]] = VPU.UnrolledType([[CONV_CP0]] : tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]]
// CHECK-SAME{LITERAL}:                                      memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK:               [[WT1_CP:%.+]] = VPU.UnrolledType([[WT1]] : tensor<64x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[WT_TABLE1_CP:%.+]] = VPU.UnrolledType([[WT_TABLE1]] : tensor<64x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_DCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV_CP1]], [[WT1_CP]], [[WT_TABLE1_CP]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
// CHECK-SAME{LITERAL}:                                     ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK:               [[DCONV_CP0:%.+]] = VPU.UnrolledType([[OUT_DCONV]] : !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK:               [[OUT_CONCAT:%.+]] = VPU.Concat([[CONV_CP0]], [[DCONV_CP0]])
// CHECK-SAME{LITERAL}:                                     {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK:               [[CONCAT_CP:%.+]] = VPU.UnrolledType([[OUT_CONCAT]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 128, 19, 72], [1, 128, 20, 72], [1, 128, 20, 72], [1, 128, 19, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK:               [[INPUT1_CP:%.+]] = VPU.UnrolledType([[ARG1]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 128, 19, 72], [1, 128, 20, 72], [1, 128, 20, 72], [1, 128, 19, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
// CHECK:               [[OUT_ELTW:%.+]] = VPU.NCE.Eltwise([[CONCAT_CP]], [[INPUT1_CP]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_ELTW]] : !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x128x72x72xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x128x72x72xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GatherSOH4
module @GatherSOH4 {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @GatherSwSOH
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x64x72xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x16x1x1xsi32>
func.func @GatherSwSOH(%arg0: tensor<1x1x64x72xf16>, %arg1: tensor<1x16x1x1xsi32>) -> tensor<1x1x16x72xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x1x64x72xf16>, tensor<1x16x1x1xsi32> -> tensor<1x1x16x72xf16>

    return %0 : tensor<1x1x16x72xf16>

// CHECK:       [[DATA:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x1x64x72xf16>
// CHECK-SAME{LITERAL}:     -> !VPU.DistributedTensor<1x1x64x72xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 64, 72], [1, 1, 64, 72], [1, 1, 64, 72], [1, 1, 64, 72]]
// CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
// CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 64, 72], [1, 1, 64, 72], [1, 1, 64, 72], [1, 1, 64, 72]]
// CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

// CHECK:       [[INDICES:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x16x1x1xsi32>
// CHECK-SAME{LITERAL}:     -> !VPU.DistributedTensor<1x16x1x1xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:             compute_shapes = [[1, 4, 1, 1], [1, 4, 1, 1], [1, 4, 1, 1], [1, 4, 1, 1]]
// CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]
// CHECK-SAME{LITERAL}:             memory_shapes = [[1, 4, 1, 1], [1, 4, 1, 1], [1, 4, 1, 1], [1, 4, 1, 1]]
// CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]

// CHECK:       [[GATHER:%.+]] = VPU.Gather([[DATA]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64}

// CHECK:       [[OUT:%.+]] = VPU.UnrolledType([[GATHER]] : !VPU.DistributedTensor<1x1x16x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 4, 72], [1, 1, 4, 72], [1, 1, 4, 72], [1, 1, 4, 72]]
// CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]
// CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 4, 72], [1, 1, 4, 72], [1, 1, 4, 72], [1, 1, 4, 72]]
// CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]
// CHECK-SAME{LITERAL}:     -> tensor<1x1x16x72xf16>
// CHECK:       return [[OUT]] : tensor<1x1x16x72xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GatherSOK4
module @GatherSOK4 {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @GatherSwSOK
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x128x64x72xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x16x1x1xsi32>
func.func @GatherSwSOK(%arg0: tensor<1x128x64x72xf16>, %arg1: tensor<1x16x1x1xsi32>) -> tensor<1x128x16x72xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x128x64x72xf16>, tensor<1x16x1x1xsi32> -> tensor<1x128x16x72xf16>

    return %0 : tensor<1x128x16x72xf16>

// CHECK:       [[DATA:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x128x64x72xf16>
// CHECK-SAME{LITERAL}:     -> !VPU.DistributedTensor<1x128x64x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 64, 72], [1, 32, 64, 72], [1, 32, 64, 72], [1, 32, 64, 72]]
// CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
// CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 64, 72], [1, 32, 64, 72], [1, 32, 64, 72], [1, 32, 64, 72]]
// CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]

// CHECK:       [[INDICES:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x16x1x1xsi32>
// CHECK-SAME{LITERAL}:     -> !VPU.DistributedTensor<1x16x1x1xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]]
// CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
// CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]]
// CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

// CHECK:       [[GATHER:%.+]] = VPU.Gather([[DATA]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64}

// CHECK:       [[OUT:%.+]] = VPU.UnrolledType([[GATHER]] : !VPU.DistributedTensor<1x128x16x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 16, 72], [1, 32, 16, 72], [1, 32, 16, 72], [1, 32, 16, 72]]
// CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
// CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 16, 72], [1, 32, 16, 72], [1, 32, 16, 72], [1, 32, 16, 72]]
// CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
// CHECK-SAME{LITERAL}:     -> tensor<1x128x16x72xf16>
// CHECK:       return [[OUT]] : tensor<1x128x16x72xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SubgraphWithInplaceEltwise
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64x208x208xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x128x104x104xf16, {order = #NHWC}>, [[ARG2:%.+]]: tensor<1x128x104x104xf16, {order = #NHWC}>)
func.func @SubgraphWithInplaceEltwise(%arg0: tensor<1x64x208x208xf16, {order = #NHWC}>, %arg1: tensor<1x128x104x104xf16, {order = #NHWC}>, %arg2: tensor<1x128x104x104xf16, {order = #NHWC}>) -> (tensor<1x64x104x104xf16, {order = #NHWC}>, tensor<1x64x104x104xf16, {order = #NHWC}>, tensor<1x256x52x52xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<128x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>

    %cst_2 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>

    %cst_4 = const.Declare tensor<256x128x3x3xf16, {order = #NHWC}> = dense<2.0> : tensor<256x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<256x1x1x4xsi32> = dense<3> : tensor<256x1x1x4xsi32>
    //      Conv0
    //     /    \
    //  Conv1  InplaceEltWise0
    //           /    \
    //        Conv2   InplaceEltWise1
    //                   |
    //                  Conv3
    %0= VPU.NCE.Convolution(%arg0, %cst_0, %cst_1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [128, 64, 3, 3], strides = [2, 2]
        } -> tensor<1x128x104x104xf16, {order = #NHWC}>

    %1= VPU.NCE.Convolution(%0, %cst_2, %cst_3) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
        } -> tensor<1x64x104x104xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %arg1) {
        is_inplace = true,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
        } -> tensor<1x128x104x104xf16, {order = #NHWC}>

    %3 = VPU.NCE.Convolution(%2, %cst_2, %cst_3) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
        } -> tensor<1x64x104x104xf16, {order = #NHWC}>

    %4 = VPU.NCE.Eltwise(%2, %arg2) {
        is_inplace = true,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
        } -> tensor<1x128x104x104xf16, {order = #NHWC}>

    %5 = VPU.NCE.Convolution(%4, %cst_4, %cst_5) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 128, 3, 3], strides = [2, 2]
        } -> tensor<1x256x52x52xf16, {order = #NHWC}>
    return %1, %3, %5 : tensor<1x64x104x104xf16, {order = #NHWC}>, tensor<1x64x104x104xf16, {order = #NHWC}>, tensor<1x256x52x52xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS0:%.+]] = const.Declare tensor<128x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x64x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[WEIGHT_TABLE0:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    // CHECK:   [[WEIGHTS1:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[WEIGHT_TABLE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>
    // CHECK:   [[WEIGHTS2:%.+]] = const.Declare tensor<256x128x3x3xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<256x128x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[WEIGHT_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<3> : tensor<256x1x1x4xsi32>

    // CHECK:   [[CONV0_INPUT_COPY:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x208x208xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x64x208x208xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 64, 35, 208], [1, 64, 35, 208], [1, 64, 35, 208], [1, 64, 35, 208], [1, 64, 34, 208], [1, 64, 34, 208]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 35, 0], [0, 0, 70, 0], [0, 0, 105, 0], [0, 0, 140, 0], [0, 0, 174, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 64, 36, 208], [1, 64, 37, 208], [1, 64, 36, 208], [1, 64, 35, 208], [1, 64, 35, 208], [1, 64, 35, 208]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 35, 0], [0, 0, 70, 0], [0, 0, 105, 0], [0, 0, 139, 0], [0, 0, 173, 0]]}>
    // CHECK:   [[CONV0_WEIGHTS_COPY:%.+]] = VPU.UnrolledType([[WEIGHTS0]] : tensor<128x64x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<128x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3], [128, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV0_WEIGHT_TABLE_COPY:%.+]] = VPU.UnrolledType([[WEIGHT_TABLE0]] : tensor<128x1x1x4xsi32>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV0:%.+]] = VPU.NCE.Convolution([[CONV0_INPUT_COPY]], [[CONV0_WEIGHTS_COPY]], [[CONV0_WEIGHT_TABLE_COPY]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 64, 3, 3], strides = [2, 2]}
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[CONV0_OUTPUT:%.+]] = VPU.UnrolledType([[CONV0]]
    // CHECK-SAME:                    -> tensor<1x128x104x104xf16, {order = #NHWC}>

    // CHECK:   [[CONV1_INPUT_COPY:%.+]] = VPU.UnrolledType([[CONV0_OUTPUT]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[CONV1_WEIGHTS_COPY:%.+]] = VPU.UnrolledType([[WEIGHTS1]] : tensor<64x128x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV1_WEIGHTS_TABLE_COPY:%.+]] = VPU.UnrolledType([[WEIGHT_TABLE1]] : tensor<64x1x1x4xsi32>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV1_INPUT_COPY]], [[CONV1_WEIGHTS_COPY]], [[CONV1_WEIGHTS_TABLE_COPY]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x64x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 64, 18, 104], [1, 64, 18, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 64, 18, 104], [1, 64, 18, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[CONV1_OUTPUT:%.+]] = VPU.UnrolledType([[CONV1]]
    // CHECK-SAME:                    -> tensor<1x64x104x104xf16, {order = #NHWC}>

    // CHECK:   [[ELTWISE0_INPUT_COPY0:%.+]] = VPU.UnrolledType([[CONV0_OUTPUT]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[ELTWISE0_INPUT_COPY1:%.+]] = VPU.UnrolledType([[ARG1]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[ELTWISE0_INPUT_COPY0]], [[ELTWISE0_INPUT_COPY1]]) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[ELTWISE0_OUTPUT:%.+]] = VPU.UnrolledType([[ELTWISE0]]
    // CHECK-SAME:                    -> tensor<1x128x104x104xf16, {order = #NHWC}>

    // CHECK:   [[CONV2_INPUT_COPY:%.+]] = VPU.UnrolledType([[ELTWISE0_OUTPUT]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[CONV2_WEIGHTS_COPY:%.+]] = VPU.UnrolledType([[WEIGHTS1]] : tensor<64x128x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV2_WEIGHTS_TABLE_COPY:%.+]] = VPU.UnrolledType([[WEIGHT_TABLE1]] : tensor<64x1x1x4xsi32>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV2_INPUT_COPY]], [[CONV2_WEIGHTS_COPY]], [[CONV2_WEIGHTS_TABLE_COPY]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x64x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 64, 18, 104], [1, 64, 18, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 64, 18, 104], [1, 64, 18, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104], [1, 64, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[CONV2_OUTPUT:%.+]] = VPU.UnrolledType([[CONV2]]
    // CHECK-SAME:                    -> tensor<1x64x104x104xf16, {order = #NHWC}>

    // CHECK:   [[ELTWISE1_INPUT_COPY0:%.+]] = VPU.UnrolledType([[ELTWISE0_OUTPUT]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[ELTWISE1_INPUT_COPY1:%.+]] = VPU.UnrolledType([[ARG2]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE1_INPUT_COPY0]], [[ELTWISE1_INPUT_COPY1]]) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[ELTWISE1_OUTPUT:%.+]] = VPU.UnrolledType([[ELTWISE1]]
    // CHECK-SAME:                    -> tensor<1x128x104x104xf16, {order = #NHWC}>

    // CHECK:   [[CONV3_INPUT_COPY:%.+]] = VPU.UnrolledType([[ELTWISE1_OUTPUT]] : tensor<1x128x104x104xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x128x104x104xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 128, 18, 104], [1, 128, 18, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 128, 18, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 19, 104], [1, 128, 18, 104], [1, 128, 17, 104]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0], [0, 0, 70, 0], [0, 0, 87, 0]]}>
    // CHECK:   [[CONV3_WEIGHTS_COPY:%.+]] = VPU.UnrolledType([[WEIGHTS2]] : tensor<256x128x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<256x128x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3], [256, 128, 3, 3]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV3_WEIGHT_TABLE_COPY:%.+]] = VPU.UnrolledType([[WEIGHT_TABLE2]] : tensor<256x1x1x4xsi32>
    // CHECK-SAME:                    -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[CONV3:%.+]] = VPU.NCE.Convolution([[CONV3_INPUT_COPY]], [[CONV3_WEIGHTS_COPY]], [[CONV3_WEIGHT_TABLE_COPY]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 128, 3, 3], strides = [2, 2]}
    // CHECK-SAME:                    -> !VPU.DistributedTensor<1x256x52x52xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 256, 9, 52], [1, 256, 9, 52], [1, 256, 9, 52], [1, 256, 9, 52], [1, 256, 8, 52], [1, 256, 8, 52]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 18, 0], [0, 0, 27, 0], [0, 0, 36, 0], [0, 0, 44, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 256, 9, 52], [1, 256, 9, 52], [1, 256, 9, 52], [1, 256, 9, 52], [1, 256, 8, 52], [1, 256, 8, 52]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 18, 0], [0, 0, 27, 0], [0, 0, 36, 0], [0, 0, 44, 0]]}>
    // CHECK:   [[CONV3_OUTPUT:%.+]] = VPU.UnrolledType([[CONV3]]
    // CHECK-SAME:                    -> tensor<1x256x52x52xf16, {order = #NHWC}>
    // CHECK:   return [[CONV1_OUTPUT]], [[CONV2_OUTPUT]], [[CONV3_OUTPUT]] : tensor<1x64x104x104xf16, {order = #NHWC}>, tensor<1x64x104x104xf16, {order = #NHWC}>, tensor<1x256x52x52xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SwOpOperand0IsCst
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x4x4xf16, {order = #NHWC}>)
func.func @SwOpOperand0IsCst(%arg0: tensor<1x64x4x4xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16> {
    %cst_0 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 1024 : i64>, #const.SubView<[0, 0, 0, 0], [64, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reshape<[64, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %depth_conv = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    rawFilterShape = [64, 1, 1, 1], strides = [1, 1]} -> tensor<1x64x4x4xf16, {order = #NHWC}>
    %shape_cast = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs(%depth_conv : tensor<1x64x4x4xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    %sqrt = VPU.Sqrt(%shape_cast) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    %permute_cast = VPU.PermuteCast(%sqrt) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16>
    %div = VPU.Divide(%cst_2, %permute_cast) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<1x1024x1x1xf16>

    return %div : tensor<1x1024x1x1xf16>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 1024 : i64>, #const.SubView<[0, 0, 0, 0], [64, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reshape<[64, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:   [[COPY_0:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x64x4x4xf16, {order = #NHWC}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x4x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1:%.+]] = VPU.UnrolledType([[CST_0]] : tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<64x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_2:%.+]] = VPU.UnrolledType([[CST]] : tensor<64x1x1x4xsi32>
    // CHECK-SAME:          -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[COPY_0]], [[COPY_1]], [[COPY_2]]) {
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 1, 1], strides = [1, 1]}
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x4x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_3:%.+]] = VPU.UnrolledType([[DEPTH_CONV]] : !VPU.DistributedTensor<1x64x4x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4], [1, 64, 4, 4]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> tensor<1x64x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs([[COPY_3]] : tensor<1x64x4x4xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    // CHECK:   [[COPY_4:%.+]] = VPU.UnrolledType([[SHAPE_CAST]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>
    // CHECK:   [[SQRT:%.+]] = VPU.Sqrt([[COPY_4]]) : !VPU.DistributedTensor<1x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>
    // CHECK:   [[COPY_5:%.+]] = VPU.UnrolledType([[SQRT]] : !VPU.DistributedTensor<1x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[COPY_5]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16>
    // CHECK:   [[COPY_6:%.+]] = VPU.UnrolledType([[CST_1]] : tensor<1x1x1x1xf16>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_7:%.+]] = VPU.UnrolledType([[PERMUTE_CAST]] : tensor<1x1024x1x1xf16>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>
    // CHECK:   [[DIV:%.+]] = VPU.Divide([[COPY_6]], [[COPY_7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:          !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:          !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>
    // CHECK:   [[COPY_8:%.+]] = VPU.UnrolledType([[DIV]] : !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 704, 0, 0], [0, 864, 0, 0]]}>) -> tensor<1x1024x1x1xf16>
    // CHECK:   return [[COPY_8]] : tensor<1x1024x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SubgraphWithMaxPoolAndEltwise
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @SubgraphWithMaxPoolAndEltwise(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %0) {
        is_inplace = true,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    return %1 : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[MAXPOOL:%.*]] = VPU.NCE.MaxPool([[IN_CP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[MAXPOOL_OUT_CP:%.*]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[ELTWISE_IN_CP:%.*]] = VPU.UnrolledType([[MAXPOOL_OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[ELTWISE_IN_CP]], [[ELTWISE_IN_CP]]) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK:               [[ELTWISE_OUT_CP:%.*]] = VPU.UnrolledType([[ELTWISE]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                     compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
// CHECK-SAME{LITERAL}:                                     memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
// CHECK-SAME{LITERAL}:                                     memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[ELTWISE_OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}
}

// -----

// CHECK-LABEL: func.func @RMSNormSOH
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x1x32x6xf16>
func.func @RMSNormSOH(%arg0: tensor<1x1x32x6xf16>) -> tensor<1x1x32x6xf16> {
  %cst = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00>: tensor<1x1x1x6xf16>
  %0 = VPU.RMS(%arg0, %cst) {epsilon = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x32x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x1x32x6xf16>
  return %0 : tensor<1x1x32x6xf16>

    // CHECK:    [[CST:%.*]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x1x6xf16>
    // CHECK:    [[DATA:%.*]] = VPU.UnrolledType(%arg0 : tensor<1x1x32x6xf16>) ->
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK:    [[GAMMA:%.*]] = VPU.UnrolledType([[CST]] : tensor<1x1x1x6xf16>) ->
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:    [[RMS:%.*]] = VPU.RMS([[DATA]], [[GAMMA]]) {epsilon = 9.9999997473787516E-6 : f64} :
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>,
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> ->
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>

    // CHECK:    [[OUT:%.*]] = VPU.UnrolledType([[RMS]]
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 6, 6], [1, 1, 6, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6], [1, 1, 5, 6]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>) -> tensor<1x1x32x6xf16>
    // CHECK:    return [[OUT]] : tensor<1x1x32x6xf16>
}

// -----

// CHECK-LABEL: func.func @RMSNormSOK
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x32x1x6xf16>
func.func @RMSNormSOK(%arg0: tensor<1x32x1x6xf16>) -> tensor<1x32x1x6xf16> {
  %cst = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00>: tensor<1x1x1x6xf16>
  %0 = VPU.RMS(%arg0, %cst) {epsilon = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x32x1x6xf16>
  return %0 : tensor<1x32x1x6xf16>

    // CHECK:    [[CST:%.*]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x1x6xf16>
    // CHECK:    [[DATA:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x1x6xf16>) ->
    // CHECK-SAME:             !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>
    // CHECK:    [[GAMMA:%.*]] = VPU.UnrolledType([[CST]] : tensor<1x1x1x6xf16>) ->
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:    [[RMS:%.*]] = VPU.RMS([[DATA]], [[GAMMA]]) {epsilon = 9.9999997473787516E-6 : f64} :
    // CHECK-SAME:             !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>,
    // CHECK-SAME:             !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> ->
    // CHECK-SAME:             !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:    [[OUT:%.*]] = VPU.UnrolledType([[RMS]]
    // CHECK-SAME:             !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6], [1, 5, 1, 6]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>) -> tensor<1x32x1x6xf16>
    // CHECK:    return [[OUT]] : tensor<1x32x1x6xf16>
}
