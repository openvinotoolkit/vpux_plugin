//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" --wrap-distributed-ops-in-nceclustertiling %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingSOHOverlapped
func.func @ConvToNCEClusterTilingSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [80, 64, 3, 3],
        strides = [1, 1]}
      -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}
    //CHECK:            VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingHKSwitch
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @ConvToNCEClusterTilingHKSwitch(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [80, 64, 3, 3],
        strides = [1, 1]}
      -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[INNER_ARG0:[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]}
    //CHECK:            VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<80x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<80x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<80x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingSOK
func.func @ConvToNCEClusterTilingSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 128, 1, 1], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x96x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<96x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<96x128x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<96x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x96x28x28xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingSOK4Clusters
func.func @ConvToNCEClusterTilingSOK4Clusters(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<64x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<64x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 28, 28], [1, 64, 28, 28], [1, 64, 28, 28], [1, 64, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingSOB3Batches
// CHECK-SAME:      [[INPUT:%.+]]: tensor<3x1024x14x14xf16, {order = #NHWC}>
func.func @ConvToNCEClusterTilingSOB3Batches(%arg0: tensor<3x1024x14x14xf16, {order = #NHWC}>) -> tensor<3x256x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} -> tensor<3x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<3x256x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:[^:]+]]: tensor<3x1024x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<3x1024x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<3x1024x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<3x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<256x1024x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<256x1024x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<256x1024x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<256x1024x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<256x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<3x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<256x1024x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<3x256x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:               rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]
    //CHECK-SAME:           } -> tensor<3x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<3x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<3x256x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<3x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<3x256x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<3x256x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingSOB
// CHECK-SAME:      [[INPUT:%.+]]: tensor<6x1024x14x14xf16, {order = #NHWC}>
func.func @ConvToNCEClusterTilingSOB(%arg0: tensor<6x1024x14x14xf16, {order = #NHWC}>) -> tensor<6x256x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} -> tensor<6x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<6x256x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_ARG0:[^:]+]]: tensor<6x1024x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<6x1024x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14], [1, 1024, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<6x1024x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<6x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<256x1024x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<256x1024x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1], [256, 1024, 1, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<256x1024x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<256x1024x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<256x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<6x1024x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<256x1024x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:           [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<6x256x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:       {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14], [1, 256, 14, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:               rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]
    //CHECK-SAME:           } -> tensor<6x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<6x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<6x256x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<6x256x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<6x256x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<6x256x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvToNCEClusterTilingClustering
func.func @ConvToNCEClusterTilingClustering(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x64x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14], [1, 64, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x64x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<48x64x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<48x64x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3], [48, 64, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<48x64x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<48x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<48x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<48x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x48x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [48, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x48x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x48x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOHOverlapped
func.func @DepthConvToNCEClusterTilingSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution([[INNER_ARG4]], [[INNER_ARG5]], [[INNER_ARG6]]) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:               rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy([[INNER_ARG0]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToNCEClusterTilingHKSwitch
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @DepthConvToNCEClusterTilingHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 1, 3, 3],
        strides = [1, 1]}
            -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution([[INNER_ARG4]], [[INNER_ARG5]], [[INNER_ARG6]]) {
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:               rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy([[INNER_ARG0]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5:%.*]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOHOverlappedNoAlign
// CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
func.func @DepthConvToNCEClusterTilingSOHOverlappedNoAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:      [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:      [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:      [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1],
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 4, 14], [1, 32, 5, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 3, 14]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 5, 0], [0, 0, 7, 0], [0, 0, 9, 0], [0, 0, 11, 0]]}> {
    //CHECK:            VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN}
    //CHECK-SAME:         : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:      [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[IN_ARG1:[^:]+]]: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:            VPU.Copy([[IN_ARG1]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:          -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG2:[^:]+]]: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:            VPU.Copy([[IN_ARG2]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:         [[INPUT_CMX]] as [[IN_ARG3:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:         [[WEIGHTS_CMX]] as [[IN_ARG4:[^:]+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:         [[WEIGHTSTABLE_CMX]] as [[IN_ARG5:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]}> {
    //CHECK:            VPU.NCE.DepthConvolution([[IN_ARG3]], [[IN_ARG4]], [[IN_ARG5]]) {
    //CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:          rawFilterShape = [32, 1, 3, 3],
    //CHECK-SAME:          strides = [1, 1]}
    //CHECK-SAME:         -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[IN_ARG6:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:          VPU.Copy([[IN_ARG6]]) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToNCEClusterTilingSOK
func.func @DepthConvToNCEClusterTilingSOK(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x128x56x56xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 56, 56], [1, 32, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 56, 56], [1, 32, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x128x56x56xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<128x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<128x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<128x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<128x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x56x56xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 56, 56], [1, 32, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56], [1, 16, 56, 56]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56], [1, 128, 56, 56]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution([[INNER_ARG4]], [[INNER_ARG5]], [[INNER_ARG6]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [128, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG8:[^:]+]]: tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy([[INNER_ARG8]]) : tensor<1x128x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x56x56xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x128x56x56xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvToNCEClusterTilingClustering
func.func @DepthConvToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<32x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.NCE.DepthConvolution([[INNER_ARG4]], [[INNER_ARG5]], [[INNER_ARG6]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [32, 1, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG8:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy([[INNER_ARG8]]) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOHOverlapped
func.func @MaxPoolToNCEClusterTilingSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:        [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool([[INNER_ARG3]])
    //CHECK-SAME:           {kernel_size = [1, 1],
    //CHECK-SAME:            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToNCEClusterTilingHKSwitch
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolToNCEClusterTilingHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:        [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool([[INNER_ARG3]])
    //CHECK-SAME:           {kernel_size = [1, 1],
    //CHECK-SAME:            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG6:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToNCEClusterTilingSOHOverlappedNoAlign
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
func.func @MaxPoolToNCEClusterTilingSOHOverlappedNoAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK:        VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[IN_ARG1:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]]
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK:            VPU.NCE.MaxPool([[IN_ARG1]]) {
    //CHECK-SAME:           kernel_size = [1, 1],
    //CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[OUT_CMX]] as [[IN_ARG2:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:         -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:          VPU.Copy([[IN_ARG2]]) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:         -> tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolToNCEClusterTilingClustering
func.func @MaxPoolToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.MaxPool([[INNER_ARG3]])
    //CHECK-SAME:       {kernel_size = [1, 1],
    //CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG4]]) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @MaxPoolToNCEClusterTilingSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<6x32x14x14xf16, {order = #NHWC}>)
func.func @MaxPoolToNCEClusterTilingSOB(%input: tensor<6x32x14x14xf16, {order = #NHWC}>) -> tensor<6x32x14x14xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x14x14xf16, {order = #NHWC}>
    return %maxpool : tensor<6x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%.+]]: tensor<6x32x14x14xf16, {order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES0:%.+]] = VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<6x32x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:               -> tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES0]]
    // CHECK:                }

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_CMX]] as [[INPUT_CMX_ARG:%.+]]: tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES1:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX_ARG]]) {
    // CHECK-SAME:                   kernel_size = [1, 1],
    // CHECK-SAME:                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]
    // CHECK-SAME:               } -> tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES1]]
    // CHECK:                }

    // CHECK:                [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:%.+]]: tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<6x32x14x14xf16, {order = #NHWC}> {
    // CHECK:                    [[RES2:%.+]] = VPU.Copy([[OUT_CMX_ARG]]) : tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:               -> tensor<6x32x14x14xf16, {order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES2]]
    // CHECK:                }

    // CHECK:                return [[OUT]] : tensor<6x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @MaxPoolToNCEClusterTilingSOB3Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x14x14xf16, {order = #NHWC}>)
func.func @MaxPoolToNCEClusterTilingSOB3Batches(%input: tensor<3x32x14x14xf16, {order = #NHWC}>) -> tensor<3x32x14x14xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<3x32x14x14xf16, {order = #NHWC}>
    return %maxpool : tensor<3x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%.+]]: tensor<3x32x14x14xf16, {order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES0:%.+]] = VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<3x32x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:               -> tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES0]]
    // CHECK:                }

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_CMX]] as [[INPUT_CMX_ARG:%.+]]: tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES1:%.+]] = VPU.NCE.MaxPool([[INPUT_CMX_ARG]]) {
    // CHECK-SAME:                   kernel_size = [1, 1],
    // CHECK-SAME:                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]
    // CHECK-SAME:               } -> tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES1]]
    // CHECK:                }

    // CHECK:                [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:%.+]]: tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<3x32x14x14xf16, {order = #NHWC}> {
    // CHECK:                    [[RES2:%.+]] = VPU.Copy([[OUT_CMX_ARG]]) : tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:               -> tensor<3x32x14x14xf16, {order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES2]]
    // CHECK:                }

    // CHECK:                return [[OUT]] : tensor<3x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingSOHOverlapped
func.func @EltwiseAddToNCEClusterTilingSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as [[INNER_ARG1:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[INPUT0_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG2]], [[INNER_ARG3]]) {op_type = #VPU.eltwise_type<ADD>}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG4]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingHKSwitch
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @EltwiseAddToNCEClusterTilingHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG1]] as [[INNER_ARG1:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[INPUT0_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG2]], [[INNER_ARG3]]) {op_type = #VPU.eltwise_type<ADD>}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG4]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddToNCEClusterTilingClustering
func.func @EltwiseAddToNCEClusterTilingClustering(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>, %arg1: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.eltwise_type<ADD> } :
         tensor<1x32x14x14xf16, {order = #NHWC}>, tensor<1x32x14x14xf16, {order = #NHWC}>
         -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0: tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg1 as [[INNER_ARG1:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[INPUT0_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG2]], [[INNER_ARG3]]) {op_type = #VPU.eltwise_type<ADD>}
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG4]]) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolToNCEClusterTilingSOHOverlapped
func.func @AvgPoolToNCEClusterTilingSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]
    //CHECK:        [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.AveragePool([[INNER_ARG1]])
    //CHECK-SAME:           {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG2]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolToNCEClusterTilingHKSwitch
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolToNCEClusterTilingHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 20, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 21, 112], [1, 32, 20, 112], [1, 32, 19, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 37, 0], [0, 0, 56, 0], [0, 0, 75, 0], [0, 0, 93, 0]]
    //CHECK:        [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        VPU.Yield [[RES0]]

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 19, 112], [1, 32, 18, 112], [1, 32, 18, 112]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0], [0, 0, 76, 0], [0, 0, 94, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112], [1, 32, 112, 112]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.AveragePool([[INNER_ARG1]])
    //CHECK-SAME:           {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1]}
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG2]]) : tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x32x112x112xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolToNCEClusterTilingSOHOverlappedNoAlign
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
func.func @AvgPoolToNCEClusterTilingSOHOverlappedNoAlign(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:      [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x32x14x14xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 4, 14], [1, 32, 5, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 4, 14], [1, 32, 3, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 5, 0], [0, 0, 7, 0], [0, 0, 9, 0], [0, 0, 11, 0]]
    //CHECK:        VPU.Copy([[IN_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x32x14x14xf16, {order = #NHWC}>
    //CHECK-SAME:     -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:          [[INPUT_CMX]] as [[IN_ARG1:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x32x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 3, 14], [1, 32, 3, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14], [1, 32, 2, 14]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK:            VPU.NCE.AveragePool([[IN_ARG1]]) {
    //CHECK-SAME:          kernel_size = [3, 3],
    //CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:          strides = [1, 1]} -> tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[IN_ARG2:[^:]+]]: tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
     //CHECK-SAME:        -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    //CHECK:            VPU.Copy([[IN_ARG2]]) : tensor<1x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
     //CHECK-SAME:          -> tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[OUT]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @AvgPoolToNCEClusterTilingSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<6x32x14x14xf16, {order = #NHWC}>)
func.func @AvgPoolToNCEClusterTilingSOB(%input: tensor<6x32x14x14xf16, {order = #NHWC}>) -> tensor<6x32x14x14xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x14x14xf16, {order = #NHWC}>
    return %avgpool : tensor<6x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%.+]]: tensor<6x32x14x14xf16, {order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES0:%.+]] = VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<6x32x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:               -> tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES0]]
    // CHECK:                }

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_CMX]] as [[INPUT_CMX_ARG:%.+]]: tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<6x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES1:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX_ARG]]) {
    // CHECK-SAME:                   kernel_size = [1, 1],
    // CHECK-SAME:                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]
    // CHECK-SAME:               } -> tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES1]]
    // CHECK:                }

    // CHECK:                [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:%.+]]: tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<6x32x14x14xf16, {order = #NHWC}> {
    // CHECK:                    [[RES2:%.+]] = VPU.Copy([[OUT_CMX_ARG]]) : tensor<6x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:               -> tensor<6x32x14x14xf16, {order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES2]]
    // CHECK:                }

    // CHECK:                return [[OUT]] : tensor<6x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:  func.func @AvgPoolToNCEClusterTilingSOB3Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x14x14xf16, {order = #NHWC}>)
func.func @AvgPoolToNCEClusterTilingSOB3Batches(%input: tensor<3x32x14x14xf16, {order = #NHWC}>) -> tensor<3x32x14x14xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<3x32x14x14xf16, {order = #NHWC}>
    return %avgpool : tensor<3x32x14x14xf16, {order = #NHWC}>

    // CHECK:                [[INPUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%.+]]: tensor<3x32x14x14xf16, {order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES0:%.+]] = VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<3x32x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:               -> tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES0]]
    // CHECK:                }

    // CHECK:                [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_CMX]] as [[INPUT_CMX_ARG:%.+]]: tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<3x32x14x14xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:               mode = "SEGMENTED",  num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]
    // CHECK-SAME:           }> {
    // CHECK:                    [[RES1:%.+]] = VPU.NCE.AveragePool([[INPUT_CMX_ARG]]) {
    // CHECK-SAME:                   kernel_size = [1, 1],
    // CHECK-SAME:                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]
    // CHECK-SAME:               } -> tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES1]]
    // CHECK:                }

    // CHECK:                [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:%.+]]: tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<3x32x14x14xf16, {order = #NHWC}> {
    // CHECK:                    [[RES2:%.+]] = VPU.Copy([[OUT_CMX_ARG]]) : tensor<3x32x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:               -> tensor<3x32x14x14xf16, {order = #NHWC}>
    // CHECK:                    VPU.Yield [[RES2]]
    // CHECK:                }

    // CHECK:                return [[OUT]] : tensor<3x32x14x14xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceL1SplitOverKernel
module @ReduceL1SplitOverKernel {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceL1SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL1SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceL1(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1024x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16>

    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceL2SplitOverKernel
module @ReduceL2SplitOverKernel {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceL2SplitOverKernel(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL2SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 171, 7, 7], [1, 170, 7, 7], [1, 170, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1024x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]], memory_shapes = {{\[\[}}1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 171, 1, 1], [1, 170, 1, 1], [1, 170, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 171, 0, 0], [0, 342, 0, 0], [0, 513, 0, 0], [0, 684, 0, 0], [0, 854, 0, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceL2(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1024x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1024x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1024x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1024x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceLogicalAndClustering
module @ReduceLogicalAndClustering {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceLogicalAndClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalAndClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceLogicalAnd(%[[VAL_5]]) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceLogicalOrClustering
module @ReduceLogicalOrClustering {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceLogicalOrClustering(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalOrClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7], [1, 1024, 7, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceLogicalOr(%[[VAL_5]]) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x1x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceMaxSplitOverHeight
module @ReduceMaxSplitOverHeight {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceMaxSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMaxSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceMax(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceMeanSplitOverHeight
module @ReduceMeanSplitOverHeight {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceMeanSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMeanSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceMean(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceProdSplitOverHeight
module @ReduceProdSplitOverHeight {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceProdSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceProdSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceProd(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReduceSumSplitOverHeight
module @ReduceSumSplitOverHeight {

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @ReduceSumSplitOverHeight(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceSumSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

    // CHECK:   %[[VAL_1:.*]] = VPU.NCE.ClusterTiling (%[[VAL_0]] as %[[VAL_2:.*]]: tensor<1x1024x7x7xf16>) -> !VPU.DistributedTensor<1x1024x7x7xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1024, 2, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7], [1, 1024, 1, 7]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_3:.*]] = VPU.Copy(%[[VAL_2]]) {out_mem_space = @CMX_NN} : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_4:.*]] = VPU.NCE.ClusterTiling (%[[VAL_1]] as %[[VAL_5:.*]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x1x7x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]], memory_shapes = {{\[\[}}1, 1, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0]]}> {
    // CHECK:     %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   %[[VAL_7:.*]] = VPU.NCE.ClusterTiling (%[[VAL_4]] as %[[VAL_8:.*]]: tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x7x1xf16> {
    // CHECK:     %[[VAL_9:.*]] = VPU.Copy(%[[VAL_8]]) : tensor<1x1x7x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x7x1xf16>
    // CHECK:   return %[[VAL_7]] : tensor<1x1x7x1xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SparseConvToNCEClusterTilingSOHOverlapped
func.func @SparseConvToNCEClusterTilingSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
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
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[INPUT_SPARSE]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK:         [[VAR0:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR0]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights> {
    // CHECK:         [[VAR1:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR1]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<80x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:         [[VAR2:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR2]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<80x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    // CHECK:         [[VAR3:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR3]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR4:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR4]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SparseConvToNCEClusterTilingHKSwitch
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x64x28x28xf16, {order = #NHWC}>
// CHECK-SAME:   [[ARG1:%.*]]: tensor<1x64x28x28xi1, {order = #NHWC}>
func.func @SparseConvToNCEClusterTilingHKSwitch(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
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
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[ARG1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[INPUT_SPARSE]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK:         [[VAR0:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR0]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights> {
    // CHECK:         [[VAR1:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR1]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<80x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4], [80, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:         [[VAR2:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR2]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<80x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 5, 28], [1, 80, 4, 28], [1, 80, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28], [1, 80, 28, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:         [[VAR3:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR3]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x80x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR4:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR4]]
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DontSetAlignmentForConvEltwiseChainCase1
func.func @DontSetAlignmentForConvEltwiseChainCase1(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%0, %1) {op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG1:[^:]+]]: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG2:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG2]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_0]] as [[INNER_ARG3:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG4:[^:]+]]: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG5:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as [[INNER_ARG6:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG6]]) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[INPUT0_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG7:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG7]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as [[INNER_ARG8:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG8]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[INPUT0_CMX_1]] as [[INNER_ARG9:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1_CMX_1]] as [[INNER_ARG10:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG9]], [[INNER_ARG10]]) {op_type = #VPU.eltwise_type<ADD>}
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as [[INNER_ARG0:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG0:[^:]+]]) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.Eltwise([[OUT_0]], [[OUT_1]]) {op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DontSetAlignmentForConvEltwiseChainCase2
func.func @DontSetAlignmentForConvEltwiseChainCase2(%arg0: tensor<1x16x22x22xf16, {order = #NHWC}>, %arg1: tensor<1x16x22x22xf16, {order = #NHWC}>) -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> tensor<1x16x22x22xf16, {order = #NHWC}>
    return %2 : tensor<1x16x22x22xf16, {order = #NHWC}>

    //CHECK:        [[INPUT0_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[INNER_ARG0:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg1 as [[INNER_ARG1:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[INPUT0_CMX_0]] as [[INNER_ARG2:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1_CMX_0]] as [[INNER_ARG3:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG2]], [[INNER_ARG3]]) {op_type = #VPU.eltwise_type<ADD>}
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_0]] as [[INNER_ARG4:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG4]]) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[INPUT0_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as [[INNER_ARG5:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG5]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[INPUT1_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg1 as [[INNER_ARG6:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG6]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:       [[INPUT0_CMX_1]] as [[INNER_ARG7:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1_CMX_1]] as [[INNER_ARG8:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG7]], [[INNER_ARG8]]) {op_type = #VPU.eltwise_type<ADD>}
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_1]] as [[INNER_ARG9:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = VPU.Copy([[INNER_ARG9]]) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1]] as [[INNER_ARG10:[^:]+]]: tensor<1x16x22x22xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 5, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 6, 22], [1, 16, 5, 22], [1, 16, 4, 22]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0], [0, 0, 15, 0], [0, 0, 18, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG10]]) {out_mem_space = @CMX_NN} : tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG11:[^:]+]]: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG11]]) {out_mem_space = @CMX_NN} : tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG12:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG12]]) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_2]] as [[INNER_ARG13:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG14:[^:]+]]: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG15:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x22x22xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 4, 22], [1, 16, 3, 22], [1, 16, 3, 22]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0], [0, 0, 16, 0], [0, 0, 19, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG13]], [[INNER_ARG14]], [[INNER_ARG15]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX_2]] as [[INNER_ARG16:[^:]+]]: tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x16x22x22xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG16]]) : tensor<1x16x22x22xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x16x22x22xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_2]] : tensor<1x16x22x22xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MVNToNCEClusterTilingDuplicateBuffer
func.func @MVNToNCEClusterTilingDuplicateBuffer(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK:            [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x4x512x1xf16, {order = #NCWH}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK:            VPU.Yield [[RCopy]]

    //CHECK:            [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1], [1, 4, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RMVN:%*]] = VPU.MVN([[ARG2]])
    //CHECK-SAME:           {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true}
    //CHECK-SAME:           : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK: [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>)
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {order = #NCWH}> {
    //CHECK:        [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MVNToNCEClusterTilingSegmentedBuffer
func.func @MVNToNCEClusterTilingSegmentedBuffer(%arg0: tensor<1x12x512x1xf16, {order = #NCWH}>) -> tensor<1x12x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x12x512x1xf16, {order = #NCWH}> -> tensor<1x12x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x12x512x1xf16, {order = #NCWH}>

    //CHECK:            [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x12x512x1xf16, {order = #NCWH}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x12x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]]
    //CHECK:            [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x12x512x1xf16, {order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x12x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy]]

    //CHECK:            [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x12x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x12x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1], [1, 2, 512, 1]],
    //CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0]]
    //CHECK:            [[RMVN:%*]] = VPU.MVN([[ARG2]])
    //CHECK-SAME:           {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true}
    //CHECK-SAME:           : tensor<1x12x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x12x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK:            [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x12x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>)
    //CHECK-SAME:           -> tensor<1x12x512x1xf16, {order = #NCWH}> {
    //CHECK:            [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x12x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x12x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x12x512x1xf16, {order = #NCWH}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MVNToNCEClusterTilingSegmentedBufferReducedClusters
func.func @MVNToNCEClusterTilingSegmentedBufferReducedClusters(%arg0: tensor<1x4x512x1xf16, {order = #NCWH}>) -> tensor<1x4x512x1xf16, {order = #NCWH}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x4x512x1xf16, {order = #NCWH}> -> tensor<1x4x512x1xf16, {order = #NCWH}>

    return %0: tensor<1x4x512x1xf16, {order = #NCWH}>

    //CHECK:        [[ClusterCopy:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG1:%.*]]: tensor<1x4x512x1xf16, {order = #NCWH}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]
    //CHECK:        [[RCopy:%*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy]]

    //CHECK:        [[RClusterMVN:%.*]] = VPU.NCE.ClusterTiling ([[ClusterCopy]] as [[ARG2:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>)
    //CHECK-SAME:           -> !VPU.DistributedTensor<1x4x512x1xf16, #NCWH, @CMX_NN,
    //CHECK-SAME:               {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:       compute_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
    //CHECK-SAME{LITERAL}:       memory_shapes = [[1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1], [1, 1, 512, 1]],
    //CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]
    //CHECK:        [[RMVN:%*]] = VPU.MVN([[ARG2]])
    //CHECK-SAME:           {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true}
    //CHECK-SAME:           : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK: VPU.Yield [[RMVN]]

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[RClusterMVN]] as [[ARG3:%.*]]: tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>)
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {order = #NCWH}> {
    //CHECK:        [[RCopy1:%*]] =  VPU.Copy([[ARG3]]) : tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NCWH}>
    //CHECK-SAME:           -> tensor<1x4x512x1xf16, {order = #NCWH}>
    //CHECK: VPU.Yield [[RCopy1]]

    //CHECK: return [[OUT]] : tensor<1x4x512x1xf16, {order = #NCWH}>
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
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x32x15x64xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK:         [[INNER_MVN:%.*]] = VPU.MVN6([[ARG2]]) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as [[ARG3:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x15x64xf16>
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVN6SOH4
module @MVN6SOH4 {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @MVN6SOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOH(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x32x15x64xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK:         [[INNER_MVN:%.*]] = VPU.MVN6([[ARG2]]) {axes = [1, 3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as [[ARG3:%.+]]: tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x15x64xf16>
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x32x15x64xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x15x64xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x15x64xf16>
}

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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x16x32x50xf16>)
    //CHECK-SAME:                     -> !VPU.DistributedTensor<1x16x32x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50], [1, 16, 8, 50]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}> {
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x50xf16> -> tensor<1x16x32x50xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[PAD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x16x32x50xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x17x32x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60], [1, 17, 8, 60]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}> {
    //CHECK:          [[INNER_PAD:%.*]] = VPU.Pad([[ARG2]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 1, 0, 10]} : tensor<1x16x32x50xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x17x32x60xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PAD]] as [[ARG3:%.+]]: tensor<1x17x32x60xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x17x32x60xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x17x32x60xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x17x32x60xf16>
    //CHECK:        }
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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:%.+]]: tensor<1x16x30x50xf16>)
    //CHECK-SAME:                     -> !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50], [1, 4, 30, 50]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}> {
    //CHECK:         [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x30x50xf16> -> tensor<1x16x30x50xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[PAD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG2:%.+]]: tensor<1x16x30x50xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53], [1, 4, 33, 53]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}> {
    //CHECK:          [[INNER_PAD:%.*]] = VPU.Pad([[ARG2]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} : tensor<1x16x30x50xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x33x53xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[PAD]] as [[ARG3:%.+]]: tensor<1x16x33x53xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x33x53xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG3]]) : tensor<1x16x33x53xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x33x53xf16>
    //CHECK:        }
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKConvOutputSegmented
func.func @UnrollSOKConvOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 64, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %mvn = VPU.MVN(%conv) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x64x64x64xf16, {order = #NHWC}>
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %mvn : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (DUP 4 CL) CONV (SEG 4 CL) -> (SEG 6 CL) MVN (SEG 6 CL)

    //CHECK:        [[CONV_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    //CHECK:        [[SOK_CONV:%.*]] = VPU.NCE.ClusterTiling ([[CONV_IN]] as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]

    //CHECK:        [[MVN_IN:%.*]] = VPU.NCE.ClusterTiling (%4 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]

    //CHECK:        [[SOK_MVN:%.*]] = VPU.NCE.ClusterTiling (%5 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKDWConvInputOutputDuplicated
func.func @UnrollSOKDWConvInputOutputDuplicated(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<320x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x320xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 320]>, #const.Reshape<[1, 320, 1, 1]>, #const.Reshape<[320, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[320, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
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
            ppe = #VPU.PPETask<mode = <NOOP>,
            clamp_low = -2147483648 : i64,
            clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%dwconv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

    // (DUP) MVN (DUP) -> (DUP) DWCONV (SEG) -> (SEG) Sigmoid (SEG)

    //CHECK:        [[MVN_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x1x320x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x320x1xf16>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_COPY_IN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:                    VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> tensor<1x1x320x1xf16> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[RESHAPE:%.*]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>
    //CHECK:        [[CAST:%.*]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[DWCONV_INPUT_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[CAST]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV_WEIGHTS_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%cst as %arg1: tensor<320x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x16x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<320x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<320x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV_WEIGHTS_TABLE_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%cst_0 as %arg1: tensor<320x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<320x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0], [128, 0, 0, 0], [176, 0, 0, 0], [224, 0, 0, 0], [272, 0, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<320x1x1x4xsi32>
    //CHECK-SAME:                   -> tensor<320x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV_INPUT_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_COPY_IN]] as %arg2: tensor<320x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                                           [[DWCONV_WEIGHTS_TABLE_COPY_IN]] as %arg3: tensor<320x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}> {
    //CHECK:                    VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [320, 1, 1, 1], strides = [1, 1]}
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[DWCONV_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                        -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[DWCONV_COPY_OUT]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}> {
    //CHECK:                    VPU.Sigmoid(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                        -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKConvOutputSegmentedWithSlice
func.func @UnrollSOKConvOutputSegmentedWithSlice(%input: tensor<1x80x1x3000xf16, {order = #NHWC}>) -> tensor<1x384x1x1500xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<384x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<384x1x1x4xsi32> = dense<10> : tensor<384x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [384, 80, 1, 3], strides = [1, 1]}
               -> tensor<1x384x1x3000xf16, {order = #NHWC}>
    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>
    %gelu =  VPU.Gelu(%slice) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x384x1x1500xf16, {order = #NHWC}> -> tensor<1x384x1x1500xf16, {order = #NHWC}>

    return %gelu : tensor<1x384x1x1500xf16, {order = #NHWC}>

    // (DUP) CONV (SEG) -> SLICE -> (SEG) GELU (SEG)

    // CHECK:       [[CONV_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x80x1x3000xf16, {order = #NHWC}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:       [[CONV_IN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK-SAME:     -> tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[SOK_CONV:%.*]] = VPU.NCE.ClusterTiling ([[CONV_IN]] as %arg1: tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>, %1 as %arg2: tensor<384x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, %2 as %arg3: tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:    -> !VPU.DistributedTensor<1x384x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000], [1, 64, 1, 3000]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]]
    // CHECK:       [[SOK_CONV_INNER:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:        {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [384, 80, 1, 3], strides = [1, 1]}
    // CHECK-SAME:    -> tensor<1x384x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[CONV_OUT:%.*]] = VPU.NCE.ClusterTiling ([[SOK_CONV]] as %arg1: tensor<1x384x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:     -> tensor<1x384x1x3000xf16, {order = #NHWC}> {
    // CHECK:          [[CONV_OUT_INNER:%.*]] = VPU.Copy(%arg1) : tensor<1x384x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x384x1x3000xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.*]] = VPU.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 384, 1, 1500] : tensor<1x384x1x3000xf16, {order = #NHWC}> to tensor<1x384x1x1500xf16, {order = #NHWC}>

    // CHECK:       [[GELU_IN:%.*]] = VPU.NCE.ClusterTiling ([[SLICE]]  as %arg1: tensor<1x384x1x1500xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]]
    // CHECK:       [[GELU_IN_INNER:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x384x1x1500xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[SOK_GELU:%.*]] = VPU.NCE.ClusterTiling ([[GELU_IN]] as %arg1: tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x384x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500], [1, 64, 1, 1500]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 256, 0, 0], [0, 320, 0, 0]]
    // CHECK:       [[SOK_GELU_INNER:%.*]] = VPU.Gelu(%arg1) : tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x384x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKDWConvInputOutputSegmented
func.func @UnrollSOKDWConvInputOutputSegmented(%input: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>

    %mvn = VPU.MVN(%input) {across_channels = false, eps = 1.0E-4 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true}
            : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
    %dwconv = VPU.NCE.DepthConvolution(%mvn, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [64, 1, 1, 1], strides = [1, 1]}
                -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %dwconv : tensor<1x64x64x64xf16, {order = #NHWC}>

    // (SEG 6 CL) MVN (SEG 6 CL) -> (SEG 4 CL) DWCONV (SEG|DUP 4 CL)
    // DW is SEG|DUP since only consequent SW layer is compatible with SEG, in all other cases it is SEG|DUP

    //CHECK:        [[MVN_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]

    //CHECK:        [[SOK_MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_IN]] as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 11, 64, 64], [1, 10, 64, 64], [1, 10, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]

    //CHECK:        [[DWCONV_IN:%.*]] = VPU.NCE.ClusterTiling (%2 as %arg1: tensor<1x64x64x64xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]

    //CHECK:        [[SOK_DWCONV:%.*]] = VPU.NCE.ClusterTiling (%3 as %arg1: tensor<1x64x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, %4 as %arg2: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, %5 as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringKHSwitch
func.func @ChainOpsToNCEClusteringKHSwitch(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]} -> tensor<1x96x14x14xf16, {order = #NHWC}>
    return %1 : tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as [[ARG_1:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28], [1, 128, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy([[ARG_1]]) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as [[ARG_1:%.+]]: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[ARG_1:%.+]]: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as [[ARG_1:%.+]]: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as [[ARG_2:%.+]]: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[ARG_3:%.+]]: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as [[ARG_1:%.+]]: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[ARG_1]]) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as [[ARG_1:%.+]]: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28], [1, 96, 28, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[ARG_1]]) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as [[ARG_1:%.+]]: tensor<96x96x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy([[ARG_1]]) {out_mem_space = @CMX_NN} : tensor<96x96x5x5xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[ARG_1:%.+]]: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy([[ARG_1]]) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as [[ARG_1:%.+]]: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as [[ARG_2:%.+]]: tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as [[ARG_3:%.+]]: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]]) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 5, 5], strides = [2, 2]
    //CHECK-SAME:             } -> tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as [[ARG_1:%.+]]: tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:        -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy([[ARG_1]]) : tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]] : tensor<1x96x14x14xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringSOHOverlapped
func.func @ChainOpsToNCEClusteringSOHOverlapped(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]} -> tensor<1x96x14x14xf16, {order = #NHWC}>
    return %1 : tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x5x5xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 5, 5], strides = [2, 2]
    //CHECK-SAME:             } -> tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:        -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]] : tensor<1x96x14x14xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainSparseOpsoNCEClusterTilingSOHOverlapped
func.func @ChainSparseOpsoNCEClusterTilingSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<64x1x1x640xi1> = dense<1.000000e+00> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<64x1x1x640xi1>, is_weights>

    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [64, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 32, 32] <left = 1 , right = 1, top = 1, bottom = 1> #VPU.mpe_mode<VECTOR_FP16>
        }

    %1 = VPU.NCE.Convolution(%0, %weights_sparse, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [64, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 32, 32] <left = 1 , right = 1, top = 1, bottom = 1> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %1 : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<64x1x1x640xi1> = dense<1.000000e+00>
    // CHECK-DAG:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<64x1x1x640xi1>, is_weights>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[INPUT_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[INPUT_SPARSE]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK:         [[VAR0:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR0]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<64x1x1x640xi1>, is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<64x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<64x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights> {
    // CHECK:         [[VAR1:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR1]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:         [[VAR2:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR2]]
    // CHECK:       }

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[INPUT_SPARSE_CMX]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<64x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK:         [[VAR3:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR3]]
    // CHECK:       }

    // CHECK:       [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR4:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR4]]
    // CHECK:       }

    // CHECK:       [[OUT_COPYBACK:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[OUT]] as [[INPUT_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                       sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 6, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 7, 28], [1, 64, 6, 28], [1, 64, 5, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    // CHECK:         [[VAR5:%.+]] = VPU.Copy([[INPUT_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR5]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_1_SPARSE_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[WEIGHTS_SPARSE]] as [[WEIGHTS_SPARSE_ARG:%.+]]: !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                                                           sparsity_map=tensor<64x1x1x640xi1>, is_weights>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<64x64x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3], [64, 64, 3, 3]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<64x1x1x640xi1, #NCHW, @CMX_NN,
    // CHECK-SAME:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640], [64, 1, 1, 640]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:          is_weights> {
    // CHECK:         [[VAR6:%.+]] = VPU.Copy([[WEIGHTS_SPARSE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR6]]
    // CHECK:       }

    // CHECK:       [[WEIGHTS_TABLE_1_CMX:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      ([[CST_WEIGHTS_TABLE]] as [[WEIGHTS_TABLE_ARG:%.+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:         [[VAR7:%.+]] = VPU.Copy([[WEIGHTS_TABLE_ARG]]) {out_mem_space = @CMX_NN}
    // CHECK:         VPU.Yield [[VAR7]]
    // CHECK:       }

    // CHECK:       [[OUT_1_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[OUT_COPYBACK]] as [[INPUT_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
    // CHECK-SAME:      [[WEIGHTS_1_SPARSE_CMX]] as [[WEIGHTS_SPARSE_CMX_ARG:[^:]+]]:
    // CHECK-SAME:          !VPU.SparseTensor<data=tensor<64x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                            sparsity_map=tensor<64x1x1x640xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>,
    // CHECK-SAME:      [[WEIGHTS_TABLE_1_CMX]] as [[WEIGHTS_TABLE_CMX_ARG:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.SparseTensor<
    // CHECK-SAME:          data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    // CHECK-SAME:          sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 5, 28], [1, 64, 4, 28], [1, 64, 4, 28]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    // CHECK:         [[VAR8:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_CMX_ARG]], [[WEIGHTS_SPARSE_CMX_ARG]], [[WEIGHTS_TABLE_CMX_ARG]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>
    // CHECK:         VPU.Yield [[VAR8]]
    // CHECK:       }

    // CHECK:       [[OUT_1:%.+]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as [[OUT_CMX_ARG:[^:]+]]:
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x64x28x28xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>> {
    // CHECK:         [[VAR9:%.+]] = VPU.Copy([[OUT_CMX_ARG]])
    // CHECK:         VPU.Yield [[VAR9]]
    // CHECK:       }

    // CHECK:       return [[OUT_1]] : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlapped
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlapped(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>)
    -> (tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x28x28xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x28x28xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.*]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 18, 0], [0, 0, 22, 0]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // First conv comsumer

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Second conv comsumer

    //CHECK:        [[OUT_0_COPYBACK_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 18, 0], [0, 0, 22, 0]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<96x96x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x5x5xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as %arg2: tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_2_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 5, 5], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x28x28xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion0
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion0(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>)
    -> (tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x14x14xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]} -> tensor<1x96x14x14xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x14x14xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.*]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // First conv consumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 6, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28], [1, 96, 5, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Second conv comsumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_0_COPYBACK_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<96x96x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5], [96, 96, 5, 5]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x5x5xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as %arg2: tensor<96x96x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_2_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 3, 14], [1, 96, 3, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14], [1, 96, 2, 14]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 5, 5], strides = [2, 2]
    //CHECK-SAME:             } -> tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x14x14xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x28x28xf16, {order = #NHWC}>, tensor<1x96x14x14xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion1
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedSiblingsMemViewUnion1(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>)
    -> (tensor<1x96x26x26xf16, {order = #NHWC}>, tensor<1x96x27x27xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x4x4xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x28x28xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} -> tensor<1x96x26x26xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 0 : i64, top = 2 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 4, 4], strides = [1, 1]} -> tensor<1x96x27x27xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x26x26xf16, {order = #NHWC}>, tensor<1x96x27x27xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.*]] = const.Declare tensor<96x96x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x4x4xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 5, 28], [1, 128, 4, 28], [1, 128, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 6, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 7, 28], [1, 128, 6, 28], [1, 128, 5, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // First conv consumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 6, 28], [1, 96, 6, 28], [1, 96, 6, 28], [1, 96, 6, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x26x26xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 26], [1, 96, 5, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 5, 26], [1, 96, 5, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26], [1, 96, 4, 26]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x26x26xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x26x26xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x26x26xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x26x26xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x26x26xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Second conv consumer
    //
    // Requirements for this consumer only, w/o sibling:
    //  memory_shapes = [[1, 96, 6, 28], [1, 96, 8, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]

    //CHECK:        [[OUT_0_COPYBACK_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 5, 28], [1, 96, 4, 28], [1, 96, 4, 28]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 7, 28], [1, 96, 9, 28], [1, 96, 8, 28], [1, 96, 7, 28], [1, 96, 7, 28], [1, 96, 7, 28]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<96x96x4x4xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x4x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4], [96, 96, 4, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x4x4xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]] as %arg1: tensor<1x96x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as %arg2: tensor<96x96x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_2_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x27x27xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 4, 27], [1, 96, 4, 27], [1, 96, 4, 27]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 5, 27], [1, 96, 4, 27], [1, 96, 4, 27], [1, 96, 4, 27]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 2 : i64, right = 0 : i64, top = 2 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 4, 4], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x27x27xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x96x27x27xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x27x27xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x27x27xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x27x27xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x26x26xf16, {order = #NHWC}>, tensor<1x96x27x27xf16, {order = #NHWC}>
}

}
// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedImproperSplitForOutputShape
// Between the two conv siblings, even though one has the bigger kernel, the inferred output shapes per cluster
// don't fully satisfy H >= 1 for each tile.
func.func @ChainOpsMultipleConsumersToNCEClusteringSOHOverlappedImproperSplitForOutputShape(%arg0: tensor<1x128x8x8xf16, {order = #NHWC}>)
    -> (tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 128, 1, 1], strides = [1, 1]} -> tensor<1x96x8x8xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 8, 8], strides = [8, 8]} -> tensor<1x96x1x1xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 1, 1], strides = [1, 1]} -> tensor<1x96x8x8xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x8x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x8x8xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_2:%.*]] = const.Declare tensor<96x96x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x1x1xf16>, [#const.Reorder<#NHWC>]

    // Conv producer

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 2, 8], [1, 128, 2, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 2, 8], [1, 128, 2, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8], [1, 128, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x8x8xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1], [96, 128, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x8x8xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // First conv comsumer
    //
    // Op is incompatible with SOH strategy, do not take into account when computin overlapped params for consumer or sibling.
    //

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8], [1, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x8x8xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8], [16, 96, 8, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x8x8xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1], [1, 96, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 8, 8], strides = [8, 8]
    //CHECK-SAME:             } -> tensor<1x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x1x1xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x1x1xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Second conv comsumer

    //CHECK:        [[OUT_0_COPYBACK_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x8x8xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x8x8xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<96x96x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1], [96, 96, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK_1]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as %arg2: tensor<96x96x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_2_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x8x8xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 2, 8], [1, 96, 2, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8], [1, 96, 1, 8]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x8x8xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x8x8xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]], [[OUT_2]] : tensor<1x96x1x1xf16, {order = #NHWC}>, tensor<1x96x8x8xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedStart
func.func @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedStart(%arg0: tensor<1x128x65x65xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x65x65xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [2, 2]} -> tensor<1x96x32x32xf16, {order = #NHWC}>
    return %1 : tensor<1x96x32x32xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x65x65xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 12, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 43, 0], [0, 0, 54, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x65x65xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 12, 65], [1, 96, 11, 65], [1, 96, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x65x65xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x65x65xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Requirements for this consumer only, w/o producer compute view:
    //  memory_shapes = [[1, 96, 13, 65], [1, 96, 13, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 24, 0], [0, 0, 34, 0], [0, 0, 44, 0], [0, 0, 54, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x65x65xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 12, 65], [1, 96, 11, 65], [1, 96, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x65x65xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3], [96, 96, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32>
    //CHECK-SAME:           -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x32x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 6, 32], [1, 96, 6, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 6, 32], [1, 96, 6, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32], [1, 96, 5, 32]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 3, 3], strides = [2, 2]
    //CHECK-SAME:             } -> tensor<1x96x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x32x32xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]] : tensor<1x96x32x32xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedEnd
func.func @ChainOpsToNCEClusteringSOHIncompatibleOutputOverlappedEnd(%arg0: tensor<1x128x65x65xf16, {order = #NHWC}>) -> tensor<1x96x20x20xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x7x7xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]} -> tensor<1x96x65x65xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [96, 96, 7, 7], strides = [3, 3]} -> tensor<1x96x20x20xf16, {order = #NHWC}>
    return %1 : tensor<1x96x20x20xf16, {order = #NHWC}>

    //CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<96x1x1x4xsi32> = dense<10> : tensor<96x1x1x4xsi32>
    //CHECK-DAG:    [[WEIGHTS_0:%.*]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS_1:%.*]] = const.Declare tensor<96x96x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x7x7xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x65x65xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x128x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 11, 65], [1, 128, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 128, 12, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 13, 65], [1, 128, 11, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0], [0, 0, 32, 0], [0, 0, 43, 0], [0, 0, 54, 0]]
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x65x65xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x128x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<96x128x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3], [96, 128, 3, 3]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_0_CMX]] as %arg2: tensor<96x128x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 15, 65], [1, 96, 16, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 15, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 41, 0], [0, 0, 50, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 128, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x65x65xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x65x65xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    // Requirements for this consumer only, w/o producer compute view:
    //  memory_shapes = [[1, 96, 15, 65], [1, 96, 16, 65], [1, 96, 13, 65], [1, 96, 13, 65], [1, 96, 13, 65], [1, 96, 13, 65]]
    //  memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 23, 0], [0, 0, 32, 0], [0, 0, 41, 0], [0, 0, 50, 0]]

    //CHECK:        [[OUT_0_COPYBACK:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x96x65x65xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x65x65xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 11, 65], [1, 96, 10, 65]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 15, 65], [1, 96, 16, 65], [1, 96, 14, 65], [1, 96, 13, 65], [1, 96, 14, 65], [1, 96, 15, 65]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 32, 0], [0, 0, 41, 0], [0, 0, 50, 0]]
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x65x65xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<96x96x7x7xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x96x7x7xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7], [96, 96, 7, 7]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x96x7x7xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<96x96x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<96x1x1x4xsi32>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<96x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4], [96, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[OUT_0_COPYBACK]] as %arg1: tensor<1x96x65x65xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as %arg2: tensor<96x96x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_1_CMX]] as %arg3: tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x96x20x20xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 96, 4, 20], [1, 96, 4, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 11, 0], [0, 0, 14, 0], [0, 0, 17, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 96, 4, 20], [1, 96, 4, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20], [1, 96, 3, 20]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 11, 0], [0, 0, 14, 0], [0, 0, 17, 0]]
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                 rawFilterShape = [96, 96, 7, 7], strides = [3, 3]
    //CHECK-SAME:             } -> tensor<1x96x20x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x96x20x20xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:           -> tensor<1x96x20x20xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x96x20x20xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x20x20xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT_1]] : tensor<1x96x20x20xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x28x28xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x48x28x28xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x26x26xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x27x27xf16, {order = #NHWC}>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConcatWithOverlappedInputsNCEConsumersMemViewUnion
func.func @ConcatWithOverlappedInputsNCEConsumersMemViewUnion(%arg0: !ProducerConvType) -> (!ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%2, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 0 : i64, top = 2 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 4, 4], strides = [1, 1]} -> !ConvConsumerOutput1

    return %3, %4, %5 : !ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1

    //CHECK:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.*]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:           [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 6, 28], [1, 16, 5, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN
    //CHECK-SAME:          tensor<1x16x28x28xf16, {order = #NHWC}> -> tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:             [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                  [[INPUT_CMX_0]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTS_0_CMX]] as %arg2: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:              VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CHECK:           [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:        -> tensor<1x16x28x28xf16, {order = #NHWC}>
    //CHECK:           VPU.Copy(%arg1) : tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x28x28xf16, {order = #NHWC}>

    // CONV 1

    //CHECK:           [[INPUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 6, 28], [1, 16, 5, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:           VPU.Copy(%arg1)

    //CHECK:           [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)


    //CHECK:           [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                [[INPUT_CMX_1]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                [[WEIGHTS_1_CMX]] as %arg2: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:        mode = "OVERLAPPED"
    //CHECK-SAME:        num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:        num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:           VPU.NCE.Convolution(%arg1, %arg2, %arg3)


    //CHECK:           [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:        -> tensor<1x16x28x28xf16, {order = #NHWC}>
    //CHECK:           VPU.Copy(%arg1) : tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x28x28xf16, {order = #NHWC}>

    // CONV 2

    //CHECK:           [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 6, 28], [1, 16, 5, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:           VPU.Copy(%arg1)

    //CHECK:           [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)


    //CHECK:             [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                  [[INPUT_CMX_2]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTS_2_CMX]] as %arg2: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:           VPU.NCE.Convolution(%arg1, %arg2, %arg3)


    //CHECK:           [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:        -> tensor<1x16x28x28xf16, {order = #NHWC}>
    //CHECK:           VPU.Copy(%arg1) : tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x28x28xf16, {order = #NHWC}>


    //CHECK:           [[CONCAT:%.*]] = VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
    //CHECK-SAME               tensor<1x16x28x28xf16, {order = #NHWC}>, tensor<1x16x28x28xf16, {order = #NHWC}>, tensor<1x16x28x28xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x48x28x28xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:           [[INPUT_CMX_3:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x16x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN
    //CHECK-SAME:          tensor<1x16x28x28xf16, {order = #NHWC}> -> tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTS_3_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:             [[OUT_3_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                  [[INPUT_CMX_3]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTS_3_CMX]] as %arg2: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x26x26xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 26], [1, 16, 5, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 5, 26], [1, 16, 5, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26], [1, 16, 4, 26]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 14, 0], [0, 0, 18, 0], [0, 0, 22, 0]]
    //CHECK:              VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CONV 4

    //CHECK:           [[INPUT_CMX_4:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2]] as %arg1: tensor<1x16x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x28x28xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 5, 28], [1, 16, 4, 28], [1, 16, 4, 28]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 24, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 7, 28], [1, 16, 9, 28], [1, 16, 8, 28], [1, 16, 7, 28], [1, 16, 7, 28], [1, 16, 7, 28]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 8, 0], [0, 0, 13, 0], [0, 0, 17, 0], [0, 0, 21, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN
    //CHECK-SAME:          tensor<1x16x28x28xf16, {order = #NHWC}> -> tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTS_4_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<16x16x4x4xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x4x4xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4], [16, 16, 4, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x16x4x4xf16, {order = #NHWC}> -> tensor<16x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:             [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                  [[INPUT_CMX_4]] as %arg1: tensor<1x16x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTS_4_CMX]] as %arg2: tensor<16x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x27x27xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 4, 27], [1, 16, 4, 27], [1, 16, 4, 27]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 5, 27], [1, 16, 4, 27], [1, 16, 4, 27], [1, 16, 4, 27]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 19, 0], [0, 0, 23, 0]]
    //CHECK:              VPU.NCE.Convolution(%arg1, %arg2, %arg3)
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x48x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x32x32xf16, {order = #NHWC}>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConcatWithOverlappedInputsCompatibleNCEConsumers
func.func @ConcatWithOverlappedInputsCompatibleNCEConsumers(%arg0: !ProducerConvType) -> (!ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%2, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} -> !ConvConsumerOutput1

    return %3, %4, %5 : !ConcatOutputType, !ConvConsumerOutput0, !ConvConsumerOutput1


    //CHECK:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.*]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:            [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:            VPU.Copy(%arg1)
    //CHECK-SAME:           out_mem_space = @CMX_NN
    //CHECK-SAME:           tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:            [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy(%arg1)
    //CHECK-SAME:           out_mem_space = @CMX_NN}
    //CHECK-SAME:           tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:            [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy(%arg1)
    //CHECK-SAME:           out_mem_space = @CMX_NN}
    //CHECK-SAME:           tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:              [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                   [[INPUT_CMX_0]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[WEIGHTS_0_CMX]] as %arg2: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:             VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy(%arg1) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CONV 1

    //CHECK:            [[INPUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:            VPU.Copy(%arg1)

    //CHECK:            [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy(%arg1)

    //CHECK:            [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy(%arg1)


    //CHECK:              [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                   [[INPUT_CMX_1]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[WEIGHTS_1_CMX]] as %arg2: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:             VPU.NCE.Convolution(%arg1, %arg2, %arg3)


    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy(%arg1) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CONV 2

    //CHECK:            [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:            VPU.Copy(%arg1)

    //CHECK:            [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<16x16x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy(%arg1)

    //CHECK:            [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:           mode = "DUPLICATED"
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:            VPU.Copy(%arg1)


    //CHECK:              [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                   [[INPUT_CMX_2]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[WEIGHTS_2_CMX]] as %arg2: tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:         -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:           mode = "OVERLAPPED"
    //CHECK-SAME:           num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:           num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:              VPU.NCE.Convolution(%arg1, %arg2, %arg3)


    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy(%arg1) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>


    //CHECK:                VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
    //CHECK-SAME               tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x48x32x32xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:           [[INPUT_CMX_3:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0]] as %arg1: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN
    //CHECK-SAME:          tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTS_3_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:             [[OUT_3_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                  [[INPUT_CMX_3]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTS_3_CMX]] as %arg2: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:              VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CONV 4

    //CHECK:           [[INPUT_CMX_4:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2]] as %arg1: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN
    //CHECK-SAME:          tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTS_4_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as %arg1: tensor<16x16x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x16x5x5xf16, {order = #NHWC}> -> tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:           [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:           VPU.Copy(%arg1)
    //CHECK-SAME:          out_mem_space = @CMX_NN}
    //CHECK-SAME:          tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:             [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                  [[INPUT_CMX_4]] as %arg1: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTS_4_CMX]] as %arg2: tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                  [[WEIGHTSTABLE_CMX]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:        -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:              VPU.NCE.Convolution(%arg1, %arg2, %arg3)
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingClustering
func.func @NCEInterpolateToNCEClusterTilingClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
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
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 1, 1],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:       -> tensor<1x1x2x2xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x16x2x2xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x2x2xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x2x2xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                      scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x16x2x2xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x2x2xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x2x2xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2], [1, 16, 2, 2]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2],
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x16x2x2xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x16x2x2xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:             -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @NCEInterpolateToNCEClusterTilingSOK
func.func @NCEInterpolateToNCEClusterTilingSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1, {order = #NHWC}> = dense<1> : tensor<1x64x10x20xi1, {order = #NHWC}>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1, {order = #NHWC}>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        scales_attr = [2, 2],
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x64x10x20xi1, {order = #NHWC}> = dense<true> : tensor<1x64x10x20xi1, {order = #NHWC}>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 64, 5, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 64 : i64}
    // CHECK-SAME:       -> tensor<1x1x10x20xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x64x10x20xi1, {order = #NHWC}>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x64x5x10xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10], [1, 64, 5, 10]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x64x10x20xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x10x20xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20], [1, 1, 10, 20]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<64x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<64x64x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1], [16, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<64x1x1x4xsi32>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x64x10x20xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 10, 20], [1, 16, 10, 20], [1, 16, 10, 20], [1, 16, 10, 20]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20], [1, 64, 10, 20]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:           ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:           rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:           scales_attr = [2, 2],
    // CHECK-SAME:           strides = [1, 1]
    // CHECK-SAME:           } -> tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x64x10x20xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @BilinearNCEInterpolateToNCEClusterTilingSOH
func.func @BilinearNCEInterpolateToNCEClusterTilingSOH(%arg0: tensor<1x16x5x5xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x22x22xi1> = dense<1> : tensor<1x16x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x16x5x5xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 4, 4],
        strides = [2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x16x10x10xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x10x10xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x16x22x22xi1> = dense<true> : tensor<1x16x22x22xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 5, 5],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:       -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x16x5x5xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x16x22x22xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x5x5xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 3, 5], [1, 16, 3, 5]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 3, 5], [1, 16, 3, 5]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x22x22xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 11, 22], [1, 16, 11, 22]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 12, 22], [1, 16, 12, 22]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x22x22xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 11, 22], [1, 1, 11, 22]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 12, 22], [1, 1, 12, 22]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<16x16x4x4xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x4x4xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x16x22x22xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x22x22xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 16, 22, 22]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<16x16x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x16x10x10xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:           rawFilterShape = [16, 16, 4, 4],
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:           strides = [2, 2]
    // CHECK-SAME:           } -> tensor<1x16x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x16x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x16x10x10xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x16x10x10xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @BilinearNCEInterpolateToNCEClusterTilingSOHWithTiling
func.func @BilinearNCEInterpolateToNCEClusterTilingSOHWithTiling(%arg0: tensor<1x16x42x320xf16, {order = #NHWC}>) -> tensor<1x16x80x320xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x4x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x4x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x162x320xi1> = dense<1> : tensor<1x16x162x320xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 42, 320],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 1.0], nearest_mode = <FLOOR>,
                                    offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320],
                                    initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>
    } -> tensor<1x1x162x320xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 1.0], nearest_mode = <FLOOR>,
            offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320],
            initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>
    } -> !VPU.SparseTensor<data=tensor<1x16x42x320xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x162x320xi1>,
                           storage_element_table=tensor<1x1x162x320xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 1.0], nearest_mode = <FLOOR>,
                                              offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320],
                                              initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 4, 1],
        strides = [2, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 1.0]
    } -> tensor<1x16x80x320xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x80x320xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<16x16x4x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x4x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x16x162x320xi1> = dense<true> : tensor<1x16x162x320xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 42, 320],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], nearest_mode = <FLOOR>,
    // CHECK-SAME:                                   offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320], initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>,
    // CHECK-SAME:                                   seDepth = 1 : i64, seSize = 16 : i64}
    // CHECK-SAME:       -> tensor<1x1x162x320xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x16x42x320xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x16x162x320xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x162x320xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320], initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x42x320xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 21, 320], [1, 16, 21, 320]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 22, 320], [1, 16, 22, 320]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x162x320xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 81, 320], [1, 16, 81, 320]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 81, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 16, 82, 320], [1, 16, 82, 320]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 80, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x162x320xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 81, 320], [1, 1, 81, 320]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 81, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 82, 320], [1, 1, 82, 320]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 80, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], nearest_mode = <FLOOR>,
    // CHECK-SAME:                     offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320],
    // CHECK-SAME:                     initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<16x16x4x1xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x16x4x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 16, 4, 1], [16, 16, 4, 1]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 16, 4, 1], [16, 16, 4, 1]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<
    // CHECK-SAME:                  data=tensor<1x16x42x320xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                  sparsity_map=tensor<1x16x162x320xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                  storage_element_table=tensor<1x1x162x320xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00],
    // CHECK-SAME:                                     nearest_mode = <FLOOR>,
    // CHECK-SAME:                                     offsets = [0, 0, 4, 0], sizes = [1, 16, 162, 320],
    // CHECK-SAME:                                     initial_input_shape = [1, 16, 160, 320], initial_output_shape = [1, 16, 320, 320]>>
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<16x16x4x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x16x80x320xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 40, 320], [1, 16, 40, 320]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 40, 320], [1, 16, 40, 320]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 40, 0]]
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:           rawFilterShape = [16, 16, 4, 1],
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00],
    // CHECK-SAME:           strides = [2, 1]
    // CHECK-SAME:           } -> tensor<1x16x80x320xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x16x80x320xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x16x80x320xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x16x80x320xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @BilinearNCEInterpolateToNCEClusterTilingSOK
func.func @BilinearNCEInterpolateToNCEClusterTilingSOK(%arg0: tensor<1x32x5x5xf16, {order = #NHWC}>) -> tensor<1x32x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x32x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x32x22x22xi1> = dense<1> : tensor<1x32x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 32, dataShape = [1, 32, 5, 5],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x32x5x5xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x32x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [32, 32, 4, 4],
        strides = [2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x32x10x10xf16, {order = #NHWC}>

    return %interpolate : tensor<1x32x10x10xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x32x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x32x22x22xi1> = dense<true> : tensor<1x32x22x22xi1>
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 32, 5, 5],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 32 : i64}
    // CHECK-SAME:       -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x32x5x5xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x32x22x22xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x32x5x5xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 5, 5], [1, 32, 5, 5]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 32, 5, 5], [1, 32, 5, 5]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x32x22x22xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 22, 22], [1, 32, 22, 22]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes =  [[1, 32, 22, 22], [1, 32, 22, 22]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x22x22xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 22, 22], [1, 1, 22, 22]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 22, 22], [1, 1, 22, 22]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>> {
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<32x32x4x4xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x32x4x4xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 32, 4, 4], [16, 32, 4, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 32, 4, 4], [16, 32, 4, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]]
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<data=tensor<1x32x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x32x22x22xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x22x22xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>>,
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<32x32x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x10x10xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 10, 10], [1, 16, 10, 10]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 10, 10], [1, 32, 10, 10]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES3:%.*]] = VPU.NCE.Interpolate([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:           rawFilterShape = [32, 32, 4, 4],
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:           strides = [2, 2]
    // CHECK-SAME:           } -> tensor<1x32x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x10x10xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x32x10x10xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x32x10x10xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// Different memory offsets and shapes are generated for the output of the Convolution and the input of the Interpolate,
// which would force a spill to be preserved between them

// CHECK:       func.func @OverlappedConvToOverlappedSEPOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x32x60x60xf16, {order = #NHWC}>
func.func @OverlappedConvToOverlappedSEPOp(%input: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x32x60x60xf16, {order = #NHWC}> {
    %conv_weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv_weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %conv_weights, %conv_weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]}
      -> tensor<1x32x30x30xf16, {order = #NHWC}>

    %input_sparsity_map = const.Declare tensor<1x32x122x122xi1> = dense<1> : tensor<1x32x122x122xi1>
    %input_storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 32, dataShape = [1, 32, 30, 30],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>
    } -> tensor<1x1x122x122xi32, {order = #NHWC}>
    %input_sparse = VPU.GroupSparseTensor(%conv, %input_sparsity_map, %input_storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>
    } -> !VPU.SparseTensor<data=tensor<1x32x30x30xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x32x122x122xi1>,
                           storage_element_table=tensor<1x1x122x122xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>>

    %weights_interp = const.Declare tensor<32x32x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table_interp = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %interpolate = VPU.NCE.Interpolate(%input_sparse, %weights_interp, %weights_table_interp) {
        rawFilterShape = [32, 32, 4, 4],
        strides = [2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        scales_attr = [1.0, 1.0, 2.0, 2.0]
    } -> tensor<1x32x60x60xf16, {order = #NHWC}>

    return %interpolate : tensor<1x32x60x60xf16, {order = #NHWC}>

    // CHECK-DAG:    [[CONV_WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[CONV_WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32>

    // CHECK:        [[CONV_INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INNER_INPUT:[^:]+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 16, 15, 30], [1, 16, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 16, 16, 30], [1, 16, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}
    // CHECK:            VPU.Copy([[INNER_INPUT]]) {out_mem_space = @CMX_NN}
    // CHECK:        }

    // CHECK:        [[CONV_WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[CONV_WEIGHTS]] as [[INNER_CONV_WEIGHTS:[^:]+]]: tensor<32x16x3x3xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}
    // CHECK:            VPU.Copy([[INNER_CONV_WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK:        }

    // CHECK:        [[CONV_WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[CONV_WEIGHTS_TABLE]] as [[INNER_CONV_WEIGHTS_TABLE:[^:]+]]: tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:      compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}
    // CHECK:            VPU.Copy([[INNER_CONV_WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN}
    // CHECK:        }

    // CHECK:        [[CONV_CMX:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:          [[CONV_INPUT_CMX]] as [[INNER_CONV_INPUT_CMX:[^:]+]]: tensor<1x16x30x30xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          [[CONV_WEIGHTS_CMX]] as [[INNER_CONV_WEIGHTS_CMX:[^:]+]]: tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:          [[CONV_WEIGHTS_TABLE_CMX]] as [[INNER_CONV_WEIGHTS_TABLE_CMX:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK-SAME:      ) -> !VPU.DistributedTensor<1x32x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]]}
    // CHECK:          VPU.NCE.Convolution([[INNER_CONV_INPUT_CMX]], [[INNER_CONV_WEIGHTS_CMX]], [[INNER_CONV_WEIGHTS_TABLE_CMX]])
    // CHECK:        }

    // CHECK:        [[CONV_DDR:%.*]] = VPU.NCE.ClusterTiling ([[CONV_CMX]] as [[INNER_CONV_CMX:[^:]+]]: tensor<1x32x30x30xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x32x30x30xf16, {order = #NHWC}>
    // CHECK:            VPU.Copy([[INNER_CONV_CMX]])
    // CHECK:        }

    // CHECK-DAG:    [[INTERP_INPUT_SM:%.*]] = const.Declare tensor<1x32x122x122xi1> = dense<true> : tensor<1x32x122x122xi1>
    // CHECK:        [[INTERP_INPUT_SE:%.*]] = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 32, 30, 30],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                   scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = 32 : i64}
    // CHECK-SAME:       -> tensor<1x1x122x122xi32, {order = #NHWC}>
    // CHECK:        [[INTERP_INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[CONV_DDR]], [[INTERP_INPUT_SM]], [[INTERP_INPUT_SE]])

    // CHECK-DAG:    [[INTERP_WEIGHTS:%.+]] = const.Declare tensor<32x32x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[INTERP_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK:        [[INTER_INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INTERP_INPUT_SPARSE]] as [[INNER_INTERP_INPUT_SPARSE:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x32x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x32x122x122xi1>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x122x122xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x32x30x30xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 15, 30], [1, 32, 15, 30]], compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 16, 30], [1, 32, 16, 30]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]}
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x32x122x122xi1, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 32, 61, 122], [1, 32, 61, 122]], compute_offsets = [[0, 0, 0, 0], [0, 0, 61, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 32, 62, 122], [1, 32, 62, 122]], memory_offsets = [[0, 0, 0, 0], [0, 0, 60, 0]]}
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x122x122xi32, #NHWC, @CMX_NN,
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 61, 122], [1, 1, 61, 122]], compute_offsets = [[0, 0, 0, 0], [0, 0, 61, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 62, 122], [1, 1, 62, 122]], memory_offsets = [[0, 0, 0, 0], [0, 0, 60, 0]]}
    // CHECK-SAME:               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>> {
    // CHECK:            VPU.Copy([[INNER_INTERP_INPUT_SPARSE]]) {out_mem_space = @CMX_NN}
    // CHECK:        }

    // CHECK:        [[INTERP_WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INTERP_WEIGHTS]] as [[INNER_INTERP_WEIGHTS:[^:]+]]: tensor<32x32x4x4xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x32x4x4xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 32, 4, 4], [32, 32, 4, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 32, 4, 4], [32, 32, 4, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}
    // CHECK:            VPU.Copy([[INNER_INTERP_WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK:        }

    // CHECK:        [[INTERP_WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INTERP_WEIGHTS_TABLE]] as [[INNER_INTERP_WEIGHTS_TABLE:[^:]+]]: tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}
    // CHECK:            VPU.Copy([[INNER_INTERP_WEIGHTS_TABLE]]) {out_mem_space = @CMX_NN}
    // CHECK:        }

    // CHECK:        [[INTERP_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INTER_INPUT_CMX]] as [[INNER_INTER_INPUT_CMX:[^:]+]]: !VPU.SparseTensor<data=tensor<1x32x30x30xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      sparsity_map=tensor<1x32x122x122xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      storage_element_table=tensor<1x1x122x122xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
    // CHECK-SAME:                                                                                         scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                                                         nearest_mode = <FLOOR>,
    // CHECK-SAME:                                                                                         offsets = [0, 0, 0, 0], sizes = [1, 32, 122, 122]>>,
    // CHECK-SAME:             [[INTERP_WEIGHTS_CMX]] as [[INNER_INTERP_WEIGHTS_CMX:[^:]+]]: tensor<32x32x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[INTERP_WEIGHTS_TABLE_CMX]] as [[INNER_INTERP_WEIGHTS_TABLE_CMX:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x60x60xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 30, 60], [1, 32, 30, 60]], compute_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 30, 60], [1, 32, 30, 60]], memory_offsets = [[0, 0, 0, 0], [0, 0, 30, 0]]}
    // CHECK:            VPU.NCE.Interpolate([[INNER_INTER_INPUT_CMX]], [[INNER_INTERP_WEIGHTS_CMX]], [[INNER_INTERP_WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:           mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:           rawFilterShape = [32, 32, 4, 4],
    // CHECK-SAME:           scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:           strides = [2, 2]
    // CHECK-SAME:           } -> tensor<1x32x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:        }

    // CHECK:        [[INTERP_DDR:%.*]] = VPU.NCE.ClusterTiling ([[INTERP_CMX]] as [[INNER_INTERP_CMX:[^:]+]]: tensor<1x32x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x32x60x60xf16, {order = #NHWC}> {
    // CHECK:            VPU.Copy([[INNER_INTERP_CMX]])
    // CHECK:        }

    // CHECK:        return [[INTERP_DDR]] : tensor<1x32x60x60xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SEPadToNCEClusterTilingSOH
func.func @SEPadToNCEClusterTilingSOH(%arg0: tensor<1x16x40x40xf16, {order = #NHWC}>) -> tensor<1x32x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x42x42xi1, {order = #NHWC}> = dense<1> : tensor<1x16x42x42xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]

    %storage_element = VPU.StorageElementTable {dataElemType = i32, dataShape = [1, 16, 40, 40],
            seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = 16 : i64
        } -> tensor<1x1x42x42xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
            seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>
        } -> !VPU.SparseTensor<data=tensor<1x16x40x40xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x16x42x42xi1, {order = #NHWC}>,
                               storage_element_table=tensor<1x1x42x42xi32, {order = #NHWC}>,
                               #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>

    %conv = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [32, 16, 3, 3], strides = [2, 2]
        } -> tensor<1x32x20x20xf16, {order = #NHWC}>

    return %conv : tensor<1x32x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK-DAG:    [[INPUT_SM:%.*]] = const.Declare tensor<1x16x42x42xi1, {order = #NHWC}> = dense<1> : tensor<1x16x42x42xi8>, [#const.Reorder<#NHWC>, #const.ConvertElemType<i1>]
    // CHECK:        [[INPUT_SE:%.*]] = VPU.StorageElementTable {
    // CHECK-SAME:                              dataElemType = i32, dataShape = [1, 16, 40, 40],
    // CHECK-SAME:                              seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>,
    // CHECK-SAME:                              seDepth = 1 : i64, seSize = 16 : i64} -> tensor<1x1x42x42xi32, {order = #NHWC}>
    // CHECK:        [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])
    // CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_SPARSE]] as [[INNER_ARG0:[^:]+]]:
    // CHECK-SAME:           !VPU.SparseTensor<data=tensor<1x16x40x40xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1x16x42x42xi1, {order = #NHWC}>,
    // CHECK-SAME:                             storage_element_table=tensor<1x1x42x42xi32, {order = #NHWC}>,
    // CHECK-SAME:                             #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>)
    // CHECK-SAME:           -> !VPU.SparseTensor<
    // CHECK-SAME:               data=!VPU.DistributedTensor<1x16x40x40xf16, #NHWC, @CMX_NN
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 20, 40], [1, 16, 20, 40]]
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 20, 40], [1, 16, 21, 40]]
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0]]
    // CHECK-SAME:               sparsity_map=!VPU.DistributedTensor<1x16x42x42xi1, #NHWC, @CMX_NN
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 16, 21, 42], [1, 16, 21, 42]]
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0]]
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 16, 21, 42], [1, 16, 21, 42]]
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:               storage_element_table=!VPU.DistributedTensor<1x1x42x42xi32, #NHWC, @CMX_NN
    // CHECK-SAME:                     {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 21, 42], [1, 1, 21, 42]]
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0]]
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 21, 42], [1, 1, 21, 42]]
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:               #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>
    // CHECK:            [[RES0:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES0]]
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as [[INNER_ARG0:[^:]+]]: tensor<32x16x3x3xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x16x3x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES1:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES1]]
    // CHECK:        }

    // CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[INNER_ARG0:[^:]+]]: tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.Copy([[INNER_ARG0]]) {out_mem_space = @CMX_NN}
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:             [[INPUT_CMX]] as [[INNER_ARG0:[^:]+]]: !VPU.SparseTensor<
    // CHECK-SAME:                                                          data=tensor<1x16x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                          sparsity_map=tensor<1x16x42x42xi1, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                          storage_element_table=tensor<1x1x42x42xi32, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                                                          #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>>
    // CHECK-SAME:             [[WEIGHTS_CMX]] as [[INNER_ARG1:[^:]+]]: tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1x32x20x20xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 10, 20], [1, 32, 10, 20]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 10, 20], [1, 32, 10, 20]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
    // CHECK:           [[RES3:%.*]] = VPU.NCE.Convolution([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]])
    // CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:           rawFilterShape = [32, 16, 3, 3], strides = [2, 2]
    // CHECK-SAME:          } -> tensor<1x32x20x20xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES3]]
    // CHECK:        }

    // CHECK:       [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x20x20xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x32x20x20xf16, {order = #NHWC}> {
    // CHECK:            [[RES4:%.*]] = VPU.Copy([[INNER_ARG0]])
    // CHECK:            VPU.Yield [[RES4]]
    // CHECK:        }

    // CHECK:        return [[OUT]] : tensor<1x32x20x20xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SliceConvConcatGeluSOK

func.func @SliceConvConcatGeluSOK(%arg0: tensor<1x80x1x3008xf16, {order = #NHWC}>) -> tensor<1x512x1x3000xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_0 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %weights_1 = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_1 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%0, %weights_1, %weights_table_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [256, 80, 1, 3], strides = [1, 1]} -> tensor<1x256x1x3000xf16, {order = #NHWC}>

    %3 = VPU.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x3000xf16, {order = #NHWC}>, tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    %4 = VPU.Slice %3 [0, 0, 0, 0] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %5 = VPU.Gelu(%4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %6 = VPU.Slice %3 [0, 0, 0, 1500] [1, 512, 1, 1500] : tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>
    %7 = VPU.Gelu(%6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>

    %8 = VPU.Concat(%5, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1500]]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>
    return %8 : tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<256x80x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x80x1x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    // CHECK: [[CONV_INPUT:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 80, 1, 3000] : tensor<1x80x1x3008xf16, {order = #NHWC}> to tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK: [[CONV0_INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[CONV_INPUT]] as %arg1: tensor<1x80x1x3000xf16, {order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:   [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES0]]

    // CHECK: [[CONV0_WEIGHT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as %arg1: tensor<256x80x1x3xf16, {order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]]
    // CHECK:   [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x80x1x3xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES1]]

    // CHECK: [[CONV0_WEIGHTTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE_0]] as %arg1: tensor<256x1x1x4xsi32>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]]
    // CHECK:   [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32> -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[RES2]]

    // CHECK: [[CONV0:%.*]] = VPU.NCE.ClusterTiling ([[CONV0_INPUT_CMX]] as %arg1: tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV0_WEIGHT_CMX]] as %arg2: tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV0_WEIGHTTABLE_CMX]] as %arg3: tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]]
    // CHECK:   [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK:   VPU.Yield [[RES3]]

    // CHECK: [[CONV0_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CONV0]] as %arg1: tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x256x1x3000xf16, {order = #NHWC}> {
    // CHECK:   [[RES4:%.*]] = VPU.Copy(%arg1) : tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES4]]


    // CHECK: [[CONV1_INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[CONV_INPUT]] as %arg1: tensor<1x80x1x3000xf16, {order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x80x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000], [1, 80, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK:   [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x80x1x3000xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES5]]

    // CHECK: [[CONV1_WEIGHT_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as %arg1: tensor<256x80x1x3xf16, {order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<256x80x1x3xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [48, 80, 1, 3], [32, 80, 1, 3], [32, 80, 1, 3]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]]
    // CHECK:   [[RES6:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x80x1x3xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES6]]

    // CHECK: [[CONV1_WEIGHTTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE_1]] as %arg1: tensor<256x1x1x4xsi32>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<256x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [224, 0, 0, 0]]
    // CHECK:   [[RES7:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<256x1x1x4xsi32> -> tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:   VPU.Yield [[RES7]]
    // CHECK:  }

    // CHECK: [[CONV1:%.*]] = VPU.NCE.ClusterTiling ([[CONV1_INPUT_CMX]] as %arg1: tensor<1x80x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV1_WEIGHT_CMX]] as %arg2: tensor<256x80x1x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK:               [[CONV1_WEIGHTTABLE_CMX]] as %arg3: tensor<256x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x256x1x3000xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 48, 1, 3000], [1, 32, 1, 3000], [1, 32, 1, 3000]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 224, 0, 0]]
    // CHECK:   [[RES8:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK:   VPU.Yield [[RES8]]

    // CHECK: [[CONV1_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[CONV1]] as %arg1: tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x256x1x3000xf16, {order = #NHWC}> {
    // CHECK:   [[RES9:%.*]] = VPU.Copy(%arg1) : tensor<1x256x1x3000xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x1x3000xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES9]]

    // CHECK: [[CONV_CONCAT:%.*]] = VPU.Concat([[CONV0_OUTPUT]], [[CONV1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:    [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:  ]} :
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}>,
    // CHECK-SAME: tensor<1x256x1x3000xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_SLICE:%.*]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 0] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_0_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_0_SLICE]] as %arg1: tensor<1x512x1x1500xf16, {order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    /// CHECK-SAME:             {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]
    // CHECK:   [[RES10:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES10]]

    // CHECK: [[GELU_0:%.*]] = VPU.NCE.ClusterTiling ([[GELU_0_INPUT]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    /// CHECK-SAME:             {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]
    // CHECK:   [[RES11:%.*]] = VPU.Gelu(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES11]]

    // CHECK: [[GELU_0_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_0]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {order = #NHWC}> {
    // CHECK:   [[RES12:%.*]] = VPU.Copy(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES12]]

    // CHECK: [[GELU_1_SLICE:%.*]] = VPU.Slice [[CONV_CONCAT]] [0, 0, 0, 1500] [1, 512, 1, 1500] :
    // CHECK-SAME:                  tensor<1x512x1x3000xf16, {order = #NHWC}> to tensor<1x512x1x1500xf16, {order = #NHWC}>

    // CHECK: [[GELU_1_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_1_SLICE]] as %arg1: tensor<1x512x1x1500xf16, {order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]
    // CHECK:   [[RES13:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES13]]

    // CHECK: [[GELU_1:%.*]] = VPU.NCE.ClusterTiling ([[GELU_1_INPUT]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x512x1x1500xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]],
    // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 96, 1, 1500], [1, 96, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500], [1, 80, 1, 1500]],
    // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0], [0, 352, 0, 0], [0, 432, 0, 0]]
    // CHECK:   [[RES14:%.*]] = VPU.Gelu(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   VPU.Yield [[RES14]]

    // CHECK: [[GELU_1_OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[GELU_1]] as %arg1: tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x512x1x1500xf16, {order = #NHWC}> {
    // CHECK:   [[RES15:%.*]] = VPU.Copy(%arg1) : tensor<1x512x1x1500xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x512x1x1500xf16, {order = #NHWC}>
    // CHECK:   VPU.Yield [[RES15]]

    // CHECK: [[GELU_CONCAT:%.*]] = VPU.Concat([[GELU_0_OUTPUT]], [[GELU_1_OUTPUT]]) {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 0, 1500]
    // CHECK:  ]} : tensor<1x512x1x1500xf16, {order = #NHWC}>, tensor<1x512x1x1500xf16, {order = #NHWC}> -> tensor<1x512x1x3000xf16, {order = #NHWC}>

    // CHECK: return [[GELU_CONCAT]] : tensor<1x512x1x3000xf16, {order = #NHWC}>
}

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x48x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x32x32xf16, {order = #NHWC}>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @OverlappedThroughConcatWithCompatibleNCEConsumers
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
func.func @OverlappedThroughConcatWithCompatibleNCEConsumers(%arg0: !ProducerConvType) -> (!ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x48x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x48x7x7xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x48x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x48x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%3, %cst_4, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, rawFilterShape = [16, 48, 7, 7], strides = [1, 1]} -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%3, %cst_5, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 48, 5, 5], strides = [1, 1]} -> !ConvConsumerOutput1

    return %4, %5 : !ConvConsumerOutput0, !ConvConsumerOutput1


    //CHECK:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.*]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_3:%.*]] = const.Declare tensor<16x48x7x7xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_4:%.*]] = const.Declare tensor<16x48x5x5xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:        [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:        VPU.Copy([[IN_ARG0]])
    //CHECK-SAME:       out_mem_space = @CMX_NN
    //CHECK-SAME:       tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as [[IN_ARG1:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG1]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG2:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG2]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:      [[INPUT_CMX_0]] as [[IN_ARG3:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:      [[WEIGHTS_0_CMX]] as [[IN_ARG4:[^:]+]]: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:      [[WEIGHTSTABLE_CMX]] as [[IN_ARG5:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 9, 32], [1, 16, 12, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG3]], [[IN_ARG4]], [[IN_ARG5]])

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as [[IN_ARG6:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG6]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CONV 1

    //CHECK:        [[INPUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG7:[^:]+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:        VPU.Copy([[IN_ARG7]])

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as [[IN_ARG8:[^:]+]]: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG8]])

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG9:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG9]])


    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_1]] as [[IN_ARG10:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as [[IN_ARG11:[^:]+]]: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG12:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 9, 32], [1, 16, 12, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG10]], [[IN_ARG11]], [[IN_ARG12]])


    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as [[IN_ARG13:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG13]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CONV 2

    //CHECK:        [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG14:[^:]+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:        VPU.Copy([[IN_ARG14]])

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as [[IN_ARG15:[^:]+]]: tensor<16x16x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG15]])

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG16:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG16]])


    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_2]] as [[IN_ARG17:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as [[IN_ARG18:[^:]+]]: tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG19:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 9, 32], [1, 16, 12, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 11, 32], [1, 16, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG17]], [[IN_ARG18]], [[IN_ARG19]])


    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as [[IN_ARG20:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG20]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>


    //CHECK:         [[CONCAT:%.*]] =  VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
    //CHECK-SAME               tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x48x32x32xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:        [[INPUT_CMX_3:%.*]] = VPU.NCE.ClusterTiling ([[CONCAT]] as [[IN_ARG21:[^:]+]]: tensor<1x48x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 48, 6, 32], [1, 48, 6, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 48, 9, 32], [1, 48, 12, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]
    //CHECK:        VPU.Copy([[IN_ARG21]])
    //CHECK-SAME:       out_mem_space = @CMX_NN
    //CHECK-SAME:       tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_3_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_3]] as [[IN_ARG22:[^:]+]]: tensor<16x48x7x7xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x48x7x7xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7], [16, 48, 7, 7]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG22]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x48x7x7xf16, {order = #NHWC}> -> tensor<16x48x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG23:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG23]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_3_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_3]] as [[IN_ARG24:[^:]+]]: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_3_CMX]] as [[IN_ARG25:[^:]+]]: tensor<16x48x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG26:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG24]], [[IN_ARG25]], [[IN_ARG26]])

    //CHECK:        [[OUT_3:%.*]] = VPU.NCE.ClusterTiling ([[OUT_3_CMX]] as [[IN_ARG27:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG27]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    //CONV 4

    //CHECK:        [[INPUT_CMX_4:%.*]] = VPU.NCE.ClusterTiling ([[CONCAT]] as [[IN_ARG28:[^:]+]]: tensor<1x48x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 48, 6, 32], [1, 48, 6, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32], [1, 48, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 48, 9, 32], [1, 48, 12, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 11, 32], [1, 48, 8, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 9, 0], [0, 0, 14, 0], [0, 0, 19, 0], [0, 0, 24, 0]]
    //CHECK:        VPU.Copy([[IN_ARG28]])
    //CHECK-SAME:       out_mem_space = @CMX_NN
    //CHECK-SAME:       tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_4_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_4]] as [[IN_ARG29:[^:]+]]: tensor<16x48x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x48x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5], [16, 48, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG29]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x48x5x5xf16, {order = #NHWC}> -> tensor<16x48x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG30:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG30]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_4_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_4]] as [[IN_ARG31:[^:]+]]: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_4_CMX]] as [[IN_ARG32:[^:]+]]: tensor<16x48x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG33:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG31]], [[IN_ARG32]], [[IN_ARG33]])

    //CHECK:        [[OUT_4:%.*]] = VPU.NCE.ClusterTiling ([[OUT_4_CMX]] as [[IN_ARG34:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG34]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ProducerConvType = tensor<1x16x32x32xf16, {order = #NHWC}>
!ConcatOutputType = tensor<1x16x96x32xf16, {order = #NHWC}>
!ConvConsumerOutput0 = tensor<1x16x96x32xf16, {order = #NHWC}>
!ConvConsumerOutput1 = tensor<1x16x96x32xf16, {order = #NHWC}>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @IncompatibleConcatOverlappedWithNCEConsumers
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
func.func @IncompatibleConcatOverlappedWithNCEConsumers(%arg0: !ProducerConvType) -> (!ConvConsumerOutput0, !ConvConsumerOutput1) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x16x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x7x7xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} -> !ProducerConvType
    %1 = VPU.NCE.Convolution(%arg0, %cst_1, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]} -> !ProducerConvType
    %2 = VPU.NCE.Convolution(%arg0, %cst_2, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} -> !ProducerConvType

    %3 = VPU.Concat(%0, %1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0]]} : !ProducerConvType, !ProducerConvType, !ProducerConvType -> !ConcatOutputType

    %4 = VPU.NCE.Convolution(%3, %cst_4, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, rawFilterShape = [16, 16, 7, 7], strides = [1, 1]} -> !ConvConsumerOutput0

    %5 = VPU.NCE.Convolution(%3, %cst_5, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} -> !ConvConsumerOutput1

    return %4, %5 : !ConvConsumerOutput0, !ConvConsumerOutput1


    //CHECK:    [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_1:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_2:%.*]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_3:%.*]] = const.Declare tensor<16x16x7x7xf16, {order = #NHWC}>
    //CHECK:    [[WEIGHTS_4:%.*]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}>

    //CONV 0

    //CHECK:        [[INPUT_CMX_0:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:        VPU.Copy([[IN_ARG0]])
    //CHECK-SAME:       out_mem_space = @CMX_NN
    //CHECK-SAME:       tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_0_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_0]] as [[IN_ARG1:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG1]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG2:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG2]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_0_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:      [[INPUT_CMX_0]] as [[IN_ARG3:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:      [[WEIGHTS_0_CMX]] as [[IN_ARG4:[^:]+]]: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:      [[WEIGHTSTABLE_CMX]] as [[IN_ARG5:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:    -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG3]], [[IN_ARG4]], [[IN_ARG5]])

    //CHECK:        [[OUT_0:%.*]] = VPU.NCE.ClusterTiling ([[OUT_0_CMX]] as [[IN_ARG6:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG6]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CONV 1

    //CHECK:        [[INPUT_CMX_1:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG7:[^:]+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:        VPU.Copy([[IN_ARG7]])

    //CHECK:        [[WEIGHTS_1_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_1]] as [[IN_ARG8:[^:]+]]: tensor<16x16x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3], [16, 16, 3, 3]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG8]])

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG9:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG9]]


    //CHECK:        [[OUT_1_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_1]] as [[IN_ARG10:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_1_CMX]] as [[IN_ARG11:[^:]+]]: tensor<16x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG12:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG10]], [[IN_ARG11]], [[IN_ARG12]])


    //CHECK:        [[OUT_1:%.*]] = VPU.NCE.ClusterTiling ([[OUT_1_CMX]] as [[IN_ARG13:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG13]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    // CONV 2

    //CHECK:        [[INPUT_CMX_2:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG14:[^:]+]]: tensor<1x16x32x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 8, 32], [1, 16, 10, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 9, 32], [1, 16, 7, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    //CHECK:        VPU.Copy([[IN_ARG14]])

    //CHECK:        [[WEIGHTS_2_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_2]] as [[IN_ARG15:[^:]+]]: tensor<16x16x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG15]])

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG16:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG16]])


    //CHECK:        [[OUT_2_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_2]] as [[IN_ARG17:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_2_CMX]] as [[IN_ARG18:[^:]+]]: tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG19:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG17]], [[IN_ARG18]], [[IN_ARG19]])


    //CHECK:        [[OUT_2:%.*]] = VPU.NCE.ClusterTiling ([[OUT_2_CMX]] as [[IN_ARG20:[^:]+]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG20]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>


    //CHECK:         [[CONCAT:%.*]] =  VPU.Concat([[OUT_0]], [[OUT_1]], [[OUT_2]])
    //CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0]]
    //CHECK-SAME               tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
    //CHECK-SAME:              -> tensor<1x16x96x32xf16, {order = #NHWC}>

    //CONV 3

    //CHECK:        [[INPUT_CMX_3:%.*]] = VPU.NCE.ClusterTiling ([[CONCAT]] as [[IN_ARG21:[^:]+]]: tensor<1x16x96x32xf16, {order = #NHWC}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 19, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 19, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 29, 0], [0, 0, 45, 0], [0, 0, 61, 0], [0, 0, 77, 0]]
    //CHECK:        VPU.Copy([[IN_ARG21]])
    //CHECK-SAME:       out_mem_space = @CMX_NN
    //CHECK-SAME:       tensor<1x16x96x32xf16, {order = #NHWC}> -> tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_3_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_3]] as [[IN_ARG22:[^:]+]]: tensor<16x16x7x7xf16, {order = #NHWC}>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x16x7x7xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7], [16, 16, 7, 7]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG22]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x16x7x7xf16, {order = #NHWC}> -> tensor<16x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG23:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG23]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_3_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_3]] as [[IN_ARG24:[^:]+]]: tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_3_CMX]] as [[IN_ARG25:[^:]+]]: tensor<16x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG26:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG24]], [[IN_ARG25]], [[IN_ARG26]])

    //CHECK:        [[OUT_3:%.*]] = VPU.NCE.ClusterTiling ([[OUT_3_CMX]] as %arg1: tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x96x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy(%arg1) : tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x96x32xf16, {order = #NHWC}>

    //CONV 4

    //CHECK:        [[INPUT_CMX_4:%.*]] = VPU.NCE.ClusterTiling ([[CONCAT]] as [[IN_ARG27:[^:]+]]: tensor<1x16x96x32xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 19, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 22, 32], [1, 16, 19, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 29, 0], [0, 0, 45, 0], [0, 0, 61, 0], [0, 0, 77, 0]]
    //CHECK:        VPU.Copy([[IN_ARG27]])
    //CHECK-SAME:       out_mem_space = @CMX_NN
    //CHECK-SAME:       tensor<1x16x96x32xf16, {order = #NHWC}> -> tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTS_4_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_4]] as [[IN_ARG28:[^:]+]]: tensor<16x16x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<16x16x5x5xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5], [16, 16, 5, 5]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG28]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x16x5x5xf16, {order = #NHWC}> -> tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as [[IN_ARG29:[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:     -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN
    //CHECK-SAME:          mode = "DUPLICATED"
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //CHECK:        VPU.Copy([[IN_ARG29]])
    //CHECK-SAME:       out_mem_space = @CMX_NN}
    //CHECK-SAME:       tensor<16x1x1x4xsi32> -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUT_4_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX_4]] as [[IN_ARG30:[^:]+]]: tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_4_CMX]] as [[IN_ARG31:[^:]+]]: tensor<16x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as [[IN_ARG32:[^:]+]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:      -> !VPU.DistributedTensor<1x16x96x32xf16, #NHWC, @CMX_NN
    //CHECK-SAME:          mode = "OVERLAPPED"
    //CHECK-SAME:          num_tiles = [1, 1, 6, 1]
    //CHECK-SAME:          num_clusters = 6 : i64, uniform_distributed_segments
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]],
    //CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32], [1, 16, 16, 32]],
    //CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 0, 64, 0], [0, 0, 80, 0]]
    //CHECK:        VPU.NCE.Convolution([[IN_ARG30]], [[IN_ARG31]], [[IN_ARG32]])

    //CHECK:        [[OUT_4:%.*]] = VPU.NCE.ClusterTiling ([[OUT_4_CMX]] as [[IN_ARG33:[^:]+]]: tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:     -> tensor<1x16x96x32xf16, {order = #NHWC}>
    //CHECK:        VPU.Copy([[IN_ARG33]]) : tensor<1x16x96x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:       -> tensor<1x16x96x32xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @Permute {
IE.TileResource 2 of @NCE at 1.300000e+03 MHz

func.func @NCEPermuteCompressConv(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.CompressConvolution(%0, %WEIGHTS, %WEIGHT_TABLE) {
        cm_sp_pattern = 7 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        rawFilterShape = [16, 4, 3, 3],
        strides = [2, 2]
    } -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %1 : tensor<1x16x112x112xf16, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}> {

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling ([[COPY_INPUT]] as %arg1: tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}> {

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:          -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}> {
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKAveragePoolInputDuplicatedOutputSegmented
func.func @UnrollSOKAveragePoolInputDuplicatedOutputSegmented(%input: tensor<1x1x320x1xf16>) -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    %mvn = VPU.MVN(%input) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true}
            : tensor<1x1x320x1xf16> -> tensor<1x1x320x1xf16>

    %reshape = VPU.AffineReshape(%mvn) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
            : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    %cast = VPU.PermuteCast(%reshape) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %averagePool = VPU.NCE.AveragePool(%cast) {kernel_size = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [2.500000e-01], fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x320x1x1xf16, {order = #NHWC}>

    %activation = VPU.Sigmoid(%averagePool) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x320x1x1xf16, {order = #NHWC}> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    return %activation : tensor<1x320x1x1xf16, {order = #NHWC}>

    // (DUP) MVN (DUP) -> (DUP) AveragePool (SEG) -> (SEG) Sigmoid

    //CHECK:        [[MVN_COPY_IN:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x1x320x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x1x320x1xf16>
    //CHECK-SAME:                    -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN:%.*]] = VPU.NCE.ClusterTiling ([[MVN_COPY_IN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       !VPU.DistributedTensor<1x1x320x1xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1], [1, 1, 320, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:                    VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[MVN_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[MVN]] as %arg1: tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:       -> tensor<1x1x320x1xf16> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x1x320x1xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-SAME:                   -> tensor<1x1x320x1xf16>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[RESHAPE:%.*]] = VPU.AffineReshape([[MVN_COPY_OUT]])
    //CHECK-SAME{{LITERAL}}:    {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 320, 1, 1]} : tensor<1x1x320x1xf16> -> tensor<1x320x1x1xf16>

    //CHECK:        [[CAST:%.*]] = VPU.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x320x1x1xf16> -> tensor<1x320x1x1xf16, {order = #NHWC}>

    //CHECK:        [[AVERAGEPOOL_INPUT_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[CAST]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1], [1, 320, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[AVERAGEPOOL:%.*]]  = VPU.NCE.ClusterTiling ([[AVERAGEPOOL_INPUT_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}> {
    //CHECK:                    VPU.NCE.AveragePool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [2.500000e-01], fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]}
    //CHECK-SAME:                    -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[AVERAGEPOOL_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[AVERAGEPOOL]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:                  -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_IN:%.*]] = VPU.NCE.ClusterTiling ([[AVERAGEPOOL_COPY_OUT]] as %arg1: tensor<1x320x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
    //CHECK:                    VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID_COPY_IN]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x320x1x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 64, 1, 1], [1, 64, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1], [1, 48, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 176, 0, 0], [0, 224, 0, 0], [0, 272, 0, 0]]}>
    //CHECK:                    VPU.Sigmoid(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:                   -> tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }

    //CHECK:        [[SIGMOID_COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[SIGMOID]] as %arg1: tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    //CHECK-SAME:       -> tensor<1x320x1x1xf16, {order = #NHWC}> {
    //CHECK:                    VPU.Copy(%arg1) : tensor<1x320x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                        -> tensor<1x320x1x1xf16, {order = #NHWC}>
    //CHECK:                    VPU.Yield
    //CHECK:        }
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermute
module @NCEPermute {

IE.TileResource 2 of @NCE at 1.700000e+03 MHz

func.func @NCEPermute3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPermuteWithSOK
module @NCEPermuteWithSOK {

IE.TileResource 4 of @NCE at 1.700000e+03 MHz

func.func @NCEPermuteSOK(%arg0: tensor<1x128x32x64xf16>) -> tensor<1x128x32x64xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 128 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPETask<
            mode = <NOOP>,
            clamp_low = -2147483648 : i64,
            clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64
        >
    } -> tensor<1x128x32x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x32x64xf16, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x32x64xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x128x32x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 4, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED",
    // CHECK-SAME:          num_tiles = [1, 4, 1, 1],
    // CHECK-SAME:          num_clusters = 4 : i64,
    // CHECK-SAME:          alignment = [1, 16, 1, 1],
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]
    // CHECK-SAME:      }


    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 128 : i64,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x128x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> tensor<1x128x32x64xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteDepthwiseConv
module @NCEPermuteDepthwiseConv {

IE.TileResource 2 of @NCE at 1.700000e+03 MHz

func.func @NCEPermuteDWCONV3x224x224(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224x!qElemType, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Reorder<#NHWC>
        ]
    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %WEIGHTS, %WEIGHT_TABLE, %cst) {
        activation_window_channel_length = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        rawFilterShape = [16, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x3x224x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 112, 224], [1, 16, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 112, 224], [1, 16, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }


    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 16 : i64,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      tensor<1x16x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NCEPermuteConv3x3
module @NCEPermuteConv3x3 {

IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// CHECK: @NCEPermuteCONV3x3([[ARG0:%.+]]: tensor<1x3x224x224xf16>)
func.func @NCEPermuteCONV3x3(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224x!qElemType, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x16x3x3x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Reorder<#NHWC>
        ]
    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 16 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %WEIGHTS, %WEIGHT_TABLE) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >,
        rawFilterShape = [16, 16, 3, 3],
        strides = [1, 1]
    } -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]] as [[IN_ARG0:[^:]+]]: tensor<1x3x224x224xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]
    // CHECK-SAME:      }

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "OVERLAPPED",
    // CHECK-SAME:          num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:          num_clusters = 2 : i64,
    // CHECK-SAME:          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 16, 112, 224], [1, 16, 112, 224]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 16, 113, 224], [1, 16, 113, 224]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 111, 0]]
    // CHECK-SAME:      }

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 16 : i64,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[COPY_OUTPUT:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:      tensor<1x16x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiDepthConv
func.func @MultiDepthConv(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x96x112x112xf16, {order = #NHWC}> {
    %wt_1 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_1 = VPU.NCE.DepthConvolution(%arg0, %weight_1, %wt_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %wt_2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_2 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_2 = VPU.NCE.DepthConvolution(%arg0, %weight_2, %wt_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %wt_3 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_3 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %dwconv_3 = VPU.NCE.DepthConvolution(%arg0, %weight_3, %wt_3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %concat = VPU.Concat(%dwconv_1, %dwconv_2, %dwconv_3) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]]} : tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}> -> tensor<1x96x112x112xf16, {order = #NHWC}>
    return %concat: tensor<1x96x112x112xf16, {order = #NHWC}>

    // CHECK: [[TILING_COPY_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_2:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_3:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[DWCONV_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_OUT_1:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[TILING_COPY_4:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_5:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_6:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[DWCONV_2:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_OUT_2:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[TILING_COPY_7:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_8:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_9:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[DWCONV_3:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[TILING_COPY_OUT_3:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[CONCAT:%.*]] = VPU.Concat([[TILING_COPY_OUT_1]], [[TILING_COPY_OUT_2]], [[TILING_COPY_OUT_3]])
    // CHECK:    {static_offsets = [
    // CHECK-SAME:   [0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]
    // CHECK-SAME:   ]} : tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x96x112x112xf16, {order = #NHWC}>
    // CHECK:   return [[CONCAT]] : tensor<1x96x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @InplaceEltwiseInferFromInput
func.func @InplaceEltwiseInferFromInput(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
            is_inplace = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>
        } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    return %2 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK: [[TILING_COPY_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK:       [[AVG_POOL_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 57, 112], [1, 32, 57, 112]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 55, 0]]
    // CHECK: [[TILING_COPY_2:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[TILING_COPY_3:%.*]] = VPU.NCE.ClusterTiling
    // CHECK:       [[AVG_POOL_2:%.*]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 57, 112], [1, 32, 57, 112]],
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 55, 0]]
    // CHECK: [[TILING_COPY_4:%.*]] = VPU.NCE.ClusterTiling

    // CHECK: [[INPUT0_CMX:%.*]] = VPU.NCE.ClusterTiling
    // CHECK: [[INPUT1_CMX:%.*]] = VPU.NCE.ClusterTiling

    // CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:       [[INPUT0_CMX]] as [[INNER_ARG2:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:       [[INPUT1_CMX]] as [[INNER_ARG3:[^:]+]]: tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:               {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]],
    // CHECK-NOT{LITERAL}:        memory_shapes = [[1, 32, 56, 112], [1, 32, 56, 112]],
    // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 32, 57, 112], [1, 32, 57, 112]],
    // CHECK-NOT{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]]
    // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 55, 0]]
    // CHECK:            [[RES2:%.*]] = VPU.NCE.Eltwise([[INNER_ARG2]], [[INNER_ARG3]]) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>}
    // CHECK-SAME:           -> tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:            VPU.Yield [[RES2]]
    // CHECK:        }

    // CHECK: [[TILING_COPY_OUT_2:%.*]] = VPU.NCE.ClusterTiling
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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SUBTRACT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_SUBTRACT:%.*]] = VPU.Subtract(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[SUBTRACT]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ADD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_ADD:%.*]] = VPU.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ADD]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
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

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as %arg2: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as %arg2: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ADD:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as %arg3: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_ADD:%.*]] = VPU.Add(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ADD]] as %arg2: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy(%arg2) : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @EqualSWSOHTileNotAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x44x44xf16>
func.func @EqualSWSOHTileNotAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x44x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x44x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as {{[^:]+}}: tensor<1x1x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 8, 44], [1, 1, 8, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44], [1, 1, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x1x44x44xf16> -> tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[EQUAL:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as {{[^:]+}}: tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_EQUAL:%.*]] = VPU.Equal({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[EQUAL]] as {{[^:]+}}: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @EqualSWSOHTileAtBroadcastAxis
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x32x44x44xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x44xf16>
func.func @EqualSWSOHTileAtBroadcastAxis(%arg0: tensor<1x32x44x44xf16>,
                %arg1: tensor<1x1x1x44xf16>) -> tensor<1x32x44x44xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
                tensor<1x32x44x44xf16>,
                tensor<1x1x1x44xf16> -> tensor<1x32x44x44xi8>

    return %0 : tensor<1x32x44x44xi8>

    //CHECK:        [[INPUT0:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_0]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_1]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[EQUAL:%.*]] = VPU.NCE.ClusterTiling ([[INPUT0]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK:                                                [[INPUT1]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xi8, #NCHW, @CMX_NN,
    // CHECK-SAME{LITERAL}:                {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}> {
    //CHECK:          [[INNER_EQUAL:%.*]] = VPU.Equal({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[EQUAL]] as {{[^:]+}}: tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xi8> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy({{[^:]+}}) : tensor<1x32x44x44xi8, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x32x44x44xi8>

    //CHECK:        return [[OUTPUT]] : tensor<1x32x44x44xi8>
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

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x16x512xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[FLOOR:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}> {
    //CHECK:          [[INNER_FLOOR:%.*]] = VPU.Floor([[ARG1]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[FLOOR]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x16x512xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @FloorSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @FloorSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.Floor(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ({{[^:]+}} as [[ARG1:%.+]]: tensor<1x16x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[FLOOR:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:          [[INNER_FLOOR:%.*]] = VPU.Floor([[ARG1]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[FLOOR]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
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

    //CHECK:        [[INPUT:%.*]]  = VPU.NCE.ClusterTiling ({{[^:]+}} as [[ARG1:%.+]]: tensor<1x1x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]]  = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[FLOOR:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_FLOOR:%.*]] = VPU.Floor([[ARG1]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[FLOOR]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RoundSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @RoundSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x16x512xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ROUND:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}> {
    //CHECK:          [[INNER_ROUND:%.*]] = VPU.Round([[ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ROUND]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x16x512xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RoundSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @RoundSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.*]] = VPU.NCE.ClusterTiling ({{[^:]+}} as [[ARG1:%.+]]: tensor<1x16x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ROUND:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:          [[INNER_ROUND:%.*]] = VPU.Round([[ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ROUND]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @RoundSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @RoundSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.Round(%arg0) {
            mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.*]]  = VPU.NCE.ClusterTiling ({{[^:]+}} as [[ARG1:%.+]]: tensor<1x1x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.*]]  = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[ROUND:%.*]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_ROUND:%.*]] = VPU.Round([[ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[ROUND]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.*]] = VPU.Copy([[ARG1]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
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

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverHeight
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverHeight(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, {{6|3}}, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, {{6|3}}, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, {{6|3}}, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverKernel
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverKernel(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, {{6|3}}, 1, 1],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AccumulateSplitOverWidth
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x32xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x32xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @AccumulateSplitOverWidth(
    %LHS: tensor<1x64x16x32xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x32xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    } : tensor<1x64x16x32xf16, {order = #NHWC}>,
        tensor<1x64x16x32xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK:   [[COPY_LHS:%.*]] = VPU.NCE.ClusterTiling ([[LHS]] as [[LHS_COPY_ARG:%.*]]: tensor<1x64x16x32xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3}}],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_COPY_RES:%.*]] = VPU.Copy([[LHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS:%.*]] = VPU.NCE.ClusterTiling ([[RHS]] as [[RHS_COPY_ARG:%.*]]: tensor<1x64x16x32xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3}}],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_COPY_RES:%.*]] = VPU.Copy([[RHS_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_LHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[LHS_SCALES]] as [[LHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[LHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[LHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[LHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_RHS_SCALES:%.*]] = VPU.NCE.ClusterTiling ([[RHS_SCALES]] as [[RHS_SCALES_COPY_ARG:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x64x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "DUPLICATED",
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[RHS_SCALES_COPY_RES:%.*]] = VPU.Copy([[RHS_SCALES_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[RHS_SCALES_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_LHS]] as [[LHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS]] as [[RHS_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_LHS_SCALES]] as [[LHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:      [[COPY_RHS_SCALES]] as [[RHS_SCALES_ARG:%arg[0-9]]]:
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x64x16x32xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "SEGMENTED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3}}],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[ACCUMULATE_RES:%.*]] = VPU.Accumulate([[LHS_ARG]], [[RHS_ARG]], [[LHS_SCALES_ARG]], [[RHS_SCALES_ARG]])
    // CHECK:       VPU.Yield [[ACCUMULATE_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[ACCUMULATE]] as [[OUT_COPY_ARG:%.*]]: tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x64x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

    return %ACCUMULATE : tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x64x16x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PoolingSplitOverWidth
// CHECK-SAME: ([[DATA:%arg[0-9]]]: tensor<1x16x32x64xf16, {order = #NHWC}>)
func.func @PoolingSplitOverWidth(
    %DATA: tensor<1x16x32x64xf16, {order = #NHWC}>
) -> tensor<1x16x32x64xf16, {order = #NHWC}> {
    %POOL = VPU.NCE.MaxPool(%DATA) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1],
        kernel_size = [3, 3]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    // CHECK:   [[COPY_INPUT:%.*]] = VPU.NCE.ClusterTiling ([[DATA]] as [[INPUT_COPY_ARG:%.*]]: tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3}}],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[INPUT_COPY_RES:%.*]] = VPU.Copy([[INPUT_COPY_ARG]]) {
    // CHECK-SAME:      out_mem_space = @CMX_NN
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       VPU.Yield [[INPUT_COPY_RES]]
    // CHECK:   }

    // CHECK:   [[POOL:%.*]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:      [[COPY_INPUT]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:  ) -> !VPU.DistributedTensor<1x16x32x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:      mode = "OVERLAPPED",
    // CHECK-SAME:      num_tiles = [1, 1, 1, {{6|3}}],
    // CHECK-SAME:      num_clusters = {{6|3}} : i64
    // CHECK-SAME:  }> {
    // CHECK:       [[POOL_RES:%.*]] = VPU.NCE.MaxPool([[INPUT_ARG]])
    // CHECK:       VPU.Yield [[POOL_RES]]
    // CHECK:   }

    // CHECK:   [[COPY_OUT:%.*]] = VPU.NCE.ClusterTiling ([[POOL]] as [[OUT_COPY_ARG:%.*]]: tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x16x32x64xf16, {order = #NHWC}> {
    // CHECK:       [[OUT_COPY_RES:%.*]] = VPU.Copy([[OUT_COPY_ARG]]) :
    // CHECK-SAME:      tensor<1x16x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:       VPU.Yield [[OUT_COPY_RES]]
    // CHECK:   }

    return %POOL : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   return [[COPY_OUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x3x384x640xf16>)
func.func @FakeQuantizeSWSOH(%arg0: tensor<1x3x384x640xf16>) -> tensor<1x3x384x640xf16> {
    %inLow = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    %inHigh = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    %outLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %outHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x3x384x640xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x384x640xf16>
    return %fq : tensor<1x3x384x640xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+01> : tensor<1x3x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:[^:]+]]: tensor<1x3x384x640xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, {{6|3}}, 1], num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x384x640xf16> -> tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_LOW]] as [[ARG1:[^:]+]]: tensor<1x3x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.NCE.ClusterTiling ([[IN_HIGH]] as [[ARG1:[^:]+]]: tensor<1x3x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as  [[ARG1:[^:]+]]: tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_LOW_COPY]] as  [[ARG2:[^:]+]]: tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_HIGH_COPY]] as  [[ARG3:[^:]+]]: tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_LOW_COPY]] as  [[ARG4:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_HIGH_COPY]] as  [[ARG5:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x3x384x640xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, {{6|3}}, 1], num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:   VPU.FakeQuantize([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x3x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[FQ_CLUSTER]] as [[ARG1:[^:]+]]: tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x3x384x640xf16> {
    //CHECK:     VPU.Copy([[ARG1]]) : tensor<1x3x384x640xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x3x384x640xf16>
    //CHECK: return [[OUTPUT]] : tensor<1x3x384x640xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x128x1x512xf16>)
func.func @FakeQuantizeSWSOK(%arg0: tensor<1x128x1x512xf16>) -> tensor<1x128x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %outHigh = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    return %fq : tensor<1x128x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:[^:]+]]: tensor<1x128x1x512xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3}}, 1, 1], num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x512xf16> -> tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_HIGH_COPY:%.+]]  = VPU.NCE.ClusterTiling ([[IN_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_LOW]] as [[ARG1:[^:]+]]: tensor<1x128x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3}}, 1, 1], num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_HIGH]] as [[ARG1:[^:]+]]: tensor<1x128x1x1xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3}}, 1, 1], num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:    VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as [[ARG1:[^:]+]]: tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_LOW_COPY]] as [[ARG2:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_HIGH_COPY]] as [[ARG3:[^:]+]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_LOW_COPY]] as [[ARG4:[^:]+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_HIGH_COPY]] as [[ARG5:[^:]+]]: tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x128x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, {{6|3}}, 1, 1], num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:   VPU.FakeQuantize([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x128x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[FQ_CLUSTER]] as [[ARG1:[^:]+]]: tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x128x1x512xf16> {
    //CHECK:     VPU.Copy([[ARG1]]) : tensor<1x128x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x128x1x512xf16>
    //CHECK: return [[OUTPUT]] : tensor<1x128x1x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FakeQuantizeSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x512xf16>)
func.func @FakeQuantizeSWClustering(%arg0: tensor<1x1x1x512xf16>) -> tensor<1x1x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %inHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    %outLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %outHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    return %fq : tensor<1x1x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    //CHECK: [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[IN_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[IN_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_LOW_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_LOW]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUT_HIGH_COPY:%.+]] = VPU.NCE.ClusterTiling ([[OUT_HIGH]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK: VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[FQ_CLUSTER:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_COPY]] as  [[ARG1:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_LOW_COPY]] as  [[ARG2:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[IN_HIGH_COPY]] as  [[ARG3:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_LOW_COPY]] as  [[ARG4:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[OUT_HIGH_COPY]] as  [[ARG5:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME: -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = {{6|3}} : i64, uniform_distributed_segments
    //CHECK:   VPU.FakeQuantize([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64} : tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[FQ_CLUSTER]] as [[ARG1:[^:]+]]: tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x512xf16> {
    //CHECK:     VPU.Copy([[ARG1]]) : tensor<1x1x1x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x512xf16>
    //CHECK: return [[OUTPUT]] : tensor<1x1x1x512xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @SelectSWSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x10x40x40xf16>, [[INPUT1:%.+]]: tensor<1x10x40x40xf16>)
func.func @SelectSWSplitOverHeight(%arg0: tensor<1x10x40x40xf16>, %arg1: tensor<1x10x40x40xf16>) -> tensor<1x10x40x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x10x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16>
    return %0 : tensor<1x10x40x40xf16>

    //CHECK-DAG:    [[INPUT0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]]}> {
    //CHECK:            VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:            VPU.Copy([[INPUT0_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x10x40x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 7, 40], [1, 10, 6, 40], [1, 10, 6, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0], [0, 0, 34, 0]]}> {
    //CHECK:            VPU.Copy([[INPUT1_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x10x40x40xf16> -> tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SELECT:%.+]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_COPY]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT0_COPY]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT1_COPY]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        VPU.Select([[INPUT_ARG]], [[INPUT0_ARG]], [[INPUT1_ARG]]) {
    //CHECK-SAME:       auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[SELECT]] as [[OUTPUT_ARG:%arg[0-9]]]: tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x10x40x40xf16> {
    //CHECK:            VPU.Copy([[OUTPUT_ARG]]) : tensor<1x10x40x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x10x40x40xf16>

    //CHECK: return [[OUTPUT]] : tensor<1x10x40x40xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @SelectSWSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x40xf16>, [[INPUT1:%.+]]: tensor<1x16x1x40xf16>)
func.func @SelectSWSplitOverKernel(%arg0: tensor<1x16x1x40xf16>, %arg1: tensor<1x16x1x40xf16>) -> tensor<1x16x1x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x16x1x40xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16>
    return %0 : tensor<1x16x1x40xf16>

    //CHECK-DAG:    [[INPUT0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]

    //CHECK:        [[INPUT_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:            VPU.Copy([[INPUT_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:            VPU.Copy([[INPUT0_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16>)
    //CHECK-SAME:       -> !VPU.DistributedTensor<1x16x1x40xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:   compute_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:   memory_shapes = [[1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 3, 1, 40], [1, 2, 1, 40], [1, 2, 1, 40]],
    //CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:            VPU.Copy([[INPUT1_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x40xf16> -> tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[SELECT:%.+]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:           [[INPUT_COPY]] as [[INPUT_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT0_COPY]] as [[INPUT0_ARG:%arg[0-9]]]: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:           [[INPUT1_COPY]] as [[INPUT1_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        VPU.Select([[INPUT_ARG]], [[INPUT0_ARG]], [[INPUT1_ARG]]) {
    //CHECK-SAME:       auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[SELECT]] as [[OUTPUT_ARG:%arg[0-9]]]: tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x40xf16> {
    //CHECK:            VPU.Copy([[OUTPUT_ARG]]) : tensor<1x16x1x40xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x40xf16>

    //CHECK: return [[OUTPUT]] : tensor<1x16x1x40xf16>
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

    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%0, %cst_1, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
        } -> tensor<1x128x72x72xf16>

    return %3 : tensor<1x128x72x72xf16>

    // CHECK:                   [[TILING_COPY_0:%.*]] = VPU.NCE.ClusterTiling ([[ARG0]]
    // CHECK:                   [[TILING_COPY_1:%.*]] = VPU.NCE.ClusterTiling
    // CHECK:                   [[TILING_COPY_2:%.*]] = VPU.NCE.ClusterTiling
    // CHECK:                   [[TILING_CONV:%.*]] = VPU.NCE.ClusterTiling ([[TILING_COPY_0]]
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
    // CHECK:                   [[TILING_COPY_3:%.*]] = VPU.NCE.ClusterTiling ([[TILING_CONV]]
    // CHECK:                   [[TILING_COPY_4:%.*]] = VPU.NCE.ClusterTiling ([[TILING_COPY_3]]
    // CHECK:                   [[TILING_COPY_5:%.*]] = VPU.NCE.ClusterTiling
    // CHECK:                   [[TILING_COPY_6:%.*]] = VPU.NCE.ClusterTiling
    // CHECK:                   [[TILING_DWCONV:%.*]] = VPU.NCE.ClusterTiling ([[TILING_COPY_4]]
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72], [1, 64, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 64, 19, 72], [1, 64, 20, 72], [1, 64, 20, 72], [1, 64, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}>
    // CHECK:                   [[TILING_COPY_7:%.*]] = VPU.NCE.ClusterTiling ([[TILING_DWCONV]]
    // CHECK:                   [[CONCAT:%.*]] = VPU.Concat([[TILING_COPY_3]], [[TILING_COPY_7]])
    // CHECK-SAME{LITERAL}:         {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>
    // CHECK:                   [[TILING_COPY_8:%.*]] = VPU.NCE.ClusterTiling ([[CONCAT]] as %arg2: tensor<1x128x72x72xf16, {order = #NHWC}>)
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 128, 19, 72], [1, 128, 20, 72], [1, 128, 20, 72], [1, 128, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}> {
    // CHECK:                       VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x128x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                   [[TILING_COPY_9:%.*]] = VPU.NCE.ClusterTiling ([[ARG1]] as %arg2: tensor<1x128x72x72xf16, {order = #NHWC}>)
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 128, 19, 72], [1, 128, 20, 72], [1, 128, 20, 72], [1, 128, 19, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 17, 0], [0, 0, 35, 0], [0, 0, 53, 0]]}> {
    // CHECK:                       VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x128x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:                   [[ELTWISE:%.*]] = VPU.NCE.ClusterTiling ([[TILING_COPY_8]] as %arg2: tensor<1x128x72x72xf16, {mem_space = @CMX_NN, order = #NHWC}>, [[TILING_COPY_9]] as %arg3: tensor<1x128x72x72xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    // CHECK-SAME{LITERAL}:         -> !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], compute_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]], memory_shapes = [[1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72], [1, 128, 18, 72]], memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 54, 0]]}> {
    // CHECK:                        VPU.NCE.Eltwise(%arg2, %arg3) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x128x72x72xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                   [[TILING_COPY_9:%.*]] = VPU.NCE.ClusterTiling ([[ELTWISE]]
    // CHECK:                   return  [[TILING_COPY_9]] : tensor<1x128x72x72xf16>
}

// -----

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AndClustering
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndClustering(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x16x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT0_COPY_RES:%.+]] = VPU.Copy([[INPUT0_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT0_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<1x1x32x32xf16>) -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT1_COPY_RES:%.+]] = VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x32x32xf16> -> tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT1_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[AND:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0_COPY]] as [[AND_LHS:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[INPUT1_COPY]] as [[AND_RHS:%arg[0-9]]]: tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32], [1, 16, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[AND_OUT:%.+]] = VPU.And([[AND_LHS]], [[AND_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[AND_OUT]]
    // CHECK:                   }

    // CHECK:               [[RES:%.+]] = VPU.NCE.ClusterTiling ([[AND]] as [[AND_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x32x32xf16> {
    // CHECK:                   [[COPY_OUT:%.+]] = VPU.Copy([[AND_COPY_ARG]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16>
    // CHECK:                   VPU.Yield [[COPY_OUT]]
    // CHECK:               }

    // CHECK:               return [[RES]] : tensor<1x16x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @LogSoftmaxSWSOH
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x16x512xf16>)
func.func @LogSoftmaxSWSOH(%arg0: tensor<1x16x16x512xf16>) -> tensor<1x16x16x512xf16> {

    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
          : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16>

    return %0 : tensor<1x16x16x512xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_DATA]] as [[ARG0:%.+]]: tensor<1x16x16x512xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x16x16x512xf16> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x16x512xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 3, 512], [1, 16, 2, 512], [1, 16, 2, 512]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0], [0, 0, 12, 0], [0, 0, 14, 0]]}> {
    //CHECK:          [[INNER_LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[ARG1]]) {axisInd = 3 : i64} : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[LOG_SOFTMAX]] as [[ARG1:%.+]]: tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x16x512xf16> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG1]]) : tensor<1x16x16x512xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x16x512xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x16x16x512xf16>
}

// -----

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AndSplitOverHeight
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x16x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndSplitOverHeight(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x16x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT0_COPY_RES:%.+]] = VPU.Copy([[INPUT0_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT0_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<1x1x32x32xf16>) -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 6, 32], [1, 1, 6, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = [[1, 1, 6, 32], [1, 1, 6, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32], [1, 1, 5, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT1_COPY_RES:%.+]] = VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x32x32xf16> -> tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT1_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[AND:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0_COPY]] as [[AND_LHS:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[INPUT1_COPY]] as [[AND_RHS:%arg[0-9]]]: tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x16x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]], memory_shapes = [[1, 16, 6, 32], [1, 16, 6, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32], [1, 16, 5, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 17, 0], [0, 0, 22, 0], [0, 0, 27, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[AND_OUT:%.+]] = VPU.And([[AND_LHS]], [[AND_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[AND_OUT]]
    // CHECK:                   }

    // CHECK:               [[RES:%.+]] = VPU.NCE.ClusterTiling ([[AND]] as [[AND_COPY_ARG:%arg[0-9]]]: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x32x32xf16> {
    // CHECK:                   [[COPY_OUT:%.+]] = VPU.Copy([[AND_COPY_ARG]]) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x32x32xf16>
    // CHECK:                   VPU.Yield [[COPY_OUT]]
    // CHECK:               }

    // CHECK:               return [[RES]] : tensor<1x16x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @LogSoftmaxSWSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x513xf16>)
func.func @LogSoftmaxSWSOK(%arg0: tensor<1x16x1x513xf16>) -> tensor<1x16x1x513xf16> {

    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
          : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16>

    return %0 : tensor<1x16x1x513xf16>

    //CHECK:        [[INPUT:%.+]] = VPU.NCE.ClusterTiling ({{[^:]+}} as [[ARG1:%.+]]: tensor<1x16x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x16x1x513xf16> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x16x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 3, 1, 513], [1, 2, 1, 513], [1, 2, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0]]}> {
    //CHECK:          [[INNER_LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[ARG1]]) {axisInd = 3 : i64} : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[LOG_SOFTMAX]] as [[ARG1:%.+]]: tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x16x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG1]]) : tensor<1x16x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x16x1x513xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x16x1x513xf16>
}

// -----

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AndSplitOverKernel
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x64x32x32xf16>, [[INPUT1:%.+]]: tensor<1x1x32x32xf16>)
func.func @AndSplitOverKernel(%arg0: tensor<1x64x32x32xf16>, %arg1: tensor<1x1x32x32xf16>) -> tensor<1x64x32x32xf16> {
    %0 = VPU.And(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } : tensor<1x64x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x64x32x32xf16>

    return %0 : tensor<1x64x32x32xf16>

    // CHECK:               [[INPUT0_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0]] as [[INPUT0_COPY_ARG:%arg[0-9]]]: tensor<1x64x32x32xf16>) -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]], memory_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT0_COPY_RES:%.+]] = VPU.Copy([[INPUT0_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x64x32x32xf16> -> tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT0_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[INPUT1_COPY:%.+]] = VPU.NCE.ClusterTiling ([[INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<1x1x32x32xf16>) -> !VPU.DistributedTensor<1x1x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[INPUT1_COPY_RES:%.+]] = VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<1x1x32x32xf16> -> tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[INPUT1_COPY_RES]]
    // CHECK:                   }

    // CHECK:               [[AND:%.+]] = VPU.NCE.ClusterTiling ([[INPUT0_COPY]] as [[AND_LHS:%arg[0-9]]]: tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, [[INPUT1_COPY]] as [[AND_RHS:%arg[0-9]]]: tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> !VPU.DistributedTensor<1x64x32x32xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME{LITERAL}:     mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]], memory_shapes = [[1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 11, 32, 32], [1, 10, 32, 32], [1, 10, 32, 32]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0], [0, 33, 0, 0], [0, 44, 0, 0], [0, 54, 0, 0]]}>
    // CHECK-SAME:              {
    // CHECK:                       [[AND_OUT:%.+]] = VPU.And([[AND_LHS]], [[AND_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>, tensor<1x1x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:                       VPU.Yield [[AND_OUT]]
    // CHECK:                   }

    // CHECK:               [[RES:%.+]] = VPU.NCE.ClusterTiling ([[AND]] as [[AND_COPY_ARG:%arg[0-9]]]: tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x64x32x32xf16> {
    // CHECK:                   [[COPY_OUT:%.+]] = VPU.Copy([[AND_COPY_ARG]]) : tensor<1x64x32x32xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x64x32x32xf16>
    // CHECK:                   VPU.Yield [[COPY_OUT]]
    // CHECK:               }

    // CHECK:               return [[RES]] : tensor<1x64x32x32xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @LogSoftmaxSWClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1x1x513xf16>)
func.func @LogSoftmaxSWClustering(%arg0: tensor<1x1x1x513xf16>) -> tensor<1x1x1x513xf16> {

    %0 = VPU.LogSoftmax(%arg0) {
            axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
          : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16>

    return %0 : tensor<1x1x1x513xf16>

    //CHECK:        [[INPUT:%.+]]  = VPU.NCE.ClusterTiling ({{[^:]+}} as [[ARG1:%.+]]: tensor<1x1x1x513xf16>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_COPY:%.+]]  = VPU.Copy([[ARG1]]) {out_mem_space = @CMX_NN} : tensor<1x1x1x513xf16> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[LOG_SOFTMAX:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x513xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                       {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513], [1, 1, 1, 513]],
    //CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
    //CHECK:          [[INNER_LOG_SOFTMAX:%.+]] = VPU.LogSoftmax([[ARG1]]) {axisInd = 3 : i64} : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[LOG_SOFTMAX]] as [[ARG1:%.+]]: tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x513xf16> {
    //CHECK:          [[INNER_COPY:%.+]] = VPU.Copy([[ARG1]]) : tensor<1x1x1x513xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<1x1x1x513xf16>

    //CHECK:        return [[OUTPUT]] : tensor<1x1x1x513xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SinSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @SinSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[SIN:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SIN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SinSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @SinSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[SIN:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SIN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @SinSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @SinSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[SIN:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[SIN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @CosSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @CosSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[COS:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[COS]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @CosSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @CosSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[COS:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[COS]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @CosSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @CosSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Cos(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[COS:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[COS]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @ExpSWWithSOH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x44x44xf16>
func.func @ExpSWWithSOH(%arg0: tensor<1x32x44x44xf16>) -> tensor<1x32x44x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x44x44xf16> -> tensor<1x32x44x44xf16>
    return %0 : tensor<1x32x44x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x44x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[EXP:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x44x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 32, 8, 44], [1, 32, 8, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44], [1, 32, 7, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0], [0, 0, 30, 0], [0, 0, 37, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[EXP]] as {{[^:]+}}: tensor<1x32x44x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x44x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x44x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @ExpSWWithSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x1x44xf16>
func.func @ExpSWWithSOK(%arg0: tensor<1x32x1x44xf16>) -> tensor<1x32x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x44xf16> -> tensor<1x32x1x44xf16>
    return %0 : tensor<1x32x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x32x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[EXP:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x32x1x44xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 6, 1, 44], [1, 6, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44], [1, 5, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[EXP]] as {{[^:]+}}: tensor<1x32x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x32x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x32x1x44xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @ExpSWWithClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x44xf16>
func.func @ExpSWWithClustering(%arg0: tensor<1x1x1x44xf16>) -> tensor<1x1x1x44xf16> {
    %0 = VPU.Exp(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x44xf16> -> tensor<1x1x1x44xf16>
    return %0 : tensor<1x1x1x44xf16>

    // CHECK:        [[IN:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as {{[^:]+}}: tensor<1x1x1x44xf16>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[EXP:%.+]] = VPU.NCE.ClusterTiling ([[IN]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:                       -> !VPU.DistributedTensor<1x1x1x44xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44], [1, 1, 1, 44]],
    // CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:        [[OUT:%.+]] = VPU.NCE.ClusterTiling ([[EXP]] as {{[^:]+}}: tensor<1x1x1x44xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<1x1x1x44xf16>
    // CHECK:        return [[OUT]] : tensor<1x1x1x44xf16>
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @executors {
IE.TileResource 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MatMulToNCEClusterTilingSOG
// CHECK-SAME:    [[FUNC_INPUT1:%.+]]:  tensor<4x1x32x64x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[FUNC_INPUT2:%.+]]:  tensor<4x64x32x1x1xf16, {order = #GNHWC}>
func.func @MatMulToNCEClusterTilingSOG4Clusters(%arg0:  tensor<4x1x32x64x1xf16, {order = #GNHWC}>, %arg1: tensor<4x64x32x1x1xf16, {order = #GNHWC}>)
    -> tensor<4x1x64x64x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<4x64x1x1x4xsi32> = dense<1> : tensor<4x64x1x1x4xsi32>
    // CHECK:   [[IN_WT_CONST:%.+]] = const.Declare tensor<4x64x1x1x4xsi32> = dense<1> : tensor<4x64x1x1x4xsi32>

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>,
        clamp_low = -2147483648 : i64,
        clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64,
        fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [4, 1, 64, 32, 1], strides = [1, 1]
    } -> tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:               [[IN1:%.+]] = VPU.NCE.ClusterTiling ([[FUNC_INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<4x1x32x64x1xf16, {order = #GNHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x1x32x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}>
    // CHECK:               VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<4x1x32x64x1xf16, {order = #GNHWC}> -> tensor<4x1x32x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:               [[IN2:%.+]] = VPU.NCE.ClusterTiling ([[FUNC_INPUT2]] as [[INPUT2_COPY_ARG:%arg[0-9]]]: tensor<4x64x32x1x1xf16, {order = #GNHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x64x32x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}> {
    // CHECK:               VPU.Copy([[INPUT2_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<4x64x32x1x1xf16, {order = #GNHWC}> -> tensor<4x64x32x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:               [[IN_WT:%.+]] = VPU.NCE.ClusterTiling ([[IN_WT_CONST]] as [[WT_COPY_ARG:%arg[0-9]]]: tensor<4x64x1x1x4xsi32>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x64x1x1x4xsi32, #NCDHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}> {
    // CHECK:               VPU.Copy([[WT_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<4x64x1x1x4xsi32> -> tensor<4x64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCDHW}>

    // CHECK:               [[MATMUL_OUT:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:                        [[IN1]] as [[MATMUL_ARG1:%arg[0-9]]]: tensor<4x1x32x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
    // CHECK-SAME:                        [[IN2]] as [[MATMUL_ARG2:%arg[0-9]]]: tensor<4x64x32x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
    // CHECK-SAME:                        [[IN_WT]] as [[MATMUL_ARG_WT:%arg[0-9]]]: tensor<4x64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCDHW}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<4x1x64x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]]}> {

    // CHECK:               VPU.NCE.MatMul([[MATMUL_ARG1]], [[MATMUL_ARG2]], [[MATMUL_ARG_WT]])
    // CHECK-SAME:              {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:              rawFilterShape = [4, 1, 64, 32, 1], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<4x1x64x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:               [[COPY_OUT:%.+]] = VPU.NCE.ClusterTiling ([[MATMUL_OUT]] as [[COPY_OUT_ARG:%arg[0-9]]]: tensor<4x1x64x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>)
    // CHECK-SAME:          -> tensor<4x1x64x64x1xf16, {order = #GNHWC}> {
    // CHECK:               VPU.Copy([[COPY_OUT_ARG]]) : tensor<4x1x64x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> -> tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK: return [[COPY_OUT]] : tensor<4x1x64x64x1xf16, {order = #GNHWC}>
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @executors {
IE.TileResource 6 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MatMulToNCEClusterTilingSOG
// CHECK-SAME:  [[FUNC_INPUT1:%.+]]:  tensor<8x1x32x64x1xf16, {order = #GNHWC}>
// CHECK-SAME:  [[FUNC_INPUT2:%.+]]:  tensor<8x64x32x1x1xf16, {order = #GNHWC}>
func.func @MatMulToNCEClusterTilingSOG6Clusters(%arg0:  tensor<8x1x32x64x1xf16, {order =#GNHWC}>, %arg1: tensor<8x64x32x1x1xf16, {order = #GNHWC}>)
    -> tensor<8x1x64x64x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<8x64x1x1x4xsi32> = dense<1> : tensor<8x64x1x1x4xsi32>
    // CHECK:   [[IN_WT_CONST:%.+]] = const.Declare tensor<8x64x1x1x4xsi32> = dense<1> : tensor<8x64x1x1x4xsi32>

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>,
        clamp_low = -2147483648 : i64,
        clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64,
        fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [8, 1, 64, 32, 1], strides = [1, 1]
    } -> tensor<8x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<8x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:               [[IN1:%.+]] = VPU.NCE.ClusterTiling ([[FUNC_INPUT1]] as [[INPUT1_COPY_ARG:%arg[0-9]]]: tensor<8x1x32x64x1xf16, {order = #GNHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x1x32x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 1, 32, 64, 1], [2, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 1, 32, 64, 1], [2, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1], [1, 1, 32, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}>
    // CHECK:               VPU.Copy([[INPUT1_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<8x1x32x64x1xf16, {order = #GNHWC}> -> tensor<8x1x32x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:               [[IN2:%.+]] = VPU.NCE.ClusterTiling ([[FUNC_INPUT2]] as [[INPUT2_COPY_ARG:%arg[0-9]]]: tensor<8x64x32x1x1xf16, {order = #GNHWC}>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x64x32x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 64, 32, 1, 1], [2, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 64, 32, 1, 1], [2, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1], [1, 64, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}> {
    // CHECK:               VPU.Copy([[INPUT2_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<8x64x32x1x1xf16, {order = #GNHWC}> -> tensor<8x64x32x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:               [[IN_WT:%.+]] = VPU.NCE.ClusterTiling ([[IN_WT_CONST]] as [[WT_COPY_ARG:%arg[0-9]]]: tensor<8x64x1x1x4xsi32>)
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x64x1x1x4xsi32, #NCDHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 64, 1, 1, 4], [2, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 64, 1, 1, 4], [2, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4], [1, 64, 1, 1, 4]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}> {
    // CHECK:               VPU.Copy([[WT_COPY_ARG]]) {out_mem_space = @CMX_NN} : tensor<8x64x1x1x4xsi32> -> tensor<8x64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCDHW}>

    // CHECK:               [[MATMUL_OUT:%.+]] = VPU.NCE.ClusterTiling
    // CHECK-SAME:                        [[IN1]] as [[MATMUL_ARG1:%arg[0-9]]]: tensor<8x1x32x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
    // CHECK-SAME:                        [[IN2]] as [[MATMUL_ARG2:%arg[0-9]]]: tensor<8x64x32x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
    // CHECK-SAME:                        [[IN_WT]] as [[MATMUL_ARG_WT:%arg[0-9]]]: tensor<8x64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCDHW}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<8x1x64x64x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:            compute_shapes = [[2, 1, 64, 64, 1], [2, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:            memory_shapes = [[2, 1, 64, 64, 1], [2, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1], [1, 1, 64, 64, 1]],
    // CHECK-SAME{LITERAL}:            memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0], [6, 0, 0, 0, 0], [7, 0, 0, 0, 0]]}> {

    // CHECK:               VPU.NCE.MatMul([[MATMUL_ARG1]], [[MATMUL_ARG2]], [[MATMUL_ARG_WT]])
    // CHECK-SAME:              {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:              rawFilterShape = [8, 1, 64, 32, 1], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<8x1x64x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:               [[COPY_OUT:%.+]] = VPU.NCE.ClusterTiling ([[MATMUL_OUT]] as [[COPY_OUT_ARG:%arg[0-9]]]: tensor<8x1x64x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>)
    // CHECK-SAME:          -> tensor<8x1x64x64x1xf16, {order = #GNHWC}> {
    // CHECK:               VPU.Copy([[COPY_OUT_ARG]]) : tensor<8x1x64x64x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> -> tensor<8x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK: return [[COPY_OUT]] : tensor<8x1x64x64x1xf16, {order = #GNHWC}>
}

}
