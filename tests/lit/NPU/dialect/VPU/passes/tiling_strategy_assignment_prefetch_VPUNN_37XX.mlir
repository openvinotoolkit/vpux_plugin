//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="vpunn-cost=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<576x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<576x1x1x4xsi32, {order = #NHWC}>
func.func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<576x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<576x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x576x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = #VPU.eltwise_type<ADD>} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [576, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<576x144x3x3xf16, {order = #NHWC}>, tensor<576x1x1x4xsi32, {order = #NHWC}> -> tensor<1x576x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = #VPU.eltwise_type<ADD>}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [576, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 3, 1, 1]}
    // CHECK-SAME:          -> tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverC(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>
func.func @SplitNCEEltwise(
        %arg0: tensor<1x512x28x28xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x28x28xf16, {order = #NHWC}>)
            -> tensor<1x512x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>
    } -> tensor<1x512x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x512x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

    // return [[ELTWISE_0]] : tensor<1x512x28x28xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x64x640xf16>
func.func @PermuteTiling(%arg0: tensor<1x512x64x640xf16>) -> tensor<1x512x64x640xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 512 : i64,
                                 multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                                 ppe = #VPU.PPETask<mode = <NOOP>,
                                 clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                 lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x512x64x640xf16, {order = #NHWC}>
    return %0 : tensor<1x512x64x640xf16, {order = #NHWC}>

    // CHECK:       [[RET:%.*]] = VPU.NCE.Permute([[INPUT]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 512 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME{LITERAL}              tilingStrategy = [1, 1, 32, 1]} -> tensor<1x512x64x640xf16, {order = #NHWC}>
    //CHECK:        return [[RET]] : tensor<1x512x64x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: func.func @PrefetchTilingSubgraphTest
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x256x9x39xf16, {order = #NHWC}>
func.func @PrefetchTilingSubgraphTest(
        %arg0: tensor<1x256x9x39xf16, {order = #NHWC}>
    ) -> tensor<1x512x1x4xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %weights = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv_1 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [256, 256, 3, 3],
        strides = [2, 2]}
            -> tensor<1x256x4x19xf16, {order = #NHCW}>
    %layout_cast = VPU.LayoutCast(%conv_1) {dst_order = #NCHW} : tensor<1x256x4x19xf16, {order = #NHCW}> -> tensor<1x256x4x19xf16>
    %shape_cast = VPU.ShapeCast {shape = [1, 4, 256, 19]} inputs(%layout_cast : tensor<1x256x4x19xf16>) -> tensor<1x4x256x19xf16>
    %affine_reshape = VPU.AffineReshape(%shape_cast) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4, 4864, 1, 1]} : tensor<1x4x256x19xf16> -> tensor<4x4864x1x1xf16>
    %permute_cast = VPU.PermuteCast(%affine_reshape) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} : tensor<4x4864x1x1xf16> -> tensor<1x4864x4x1xf16, {order = #NHWC}>
    %affine_reshape_2 = VPU.AffineReshape(%permute_cast) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 4864, 1, 4]} : tensor<1x4864x4x1xf16, {order = #NHWC}> -> tensor<1x4864x1x4xf16, {order = #NHWC}>

    %weights_table_1 = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
    %weights_1 = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x4864x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_2 = VPU.NCE.Convolution(%affine_reshape_2, %weights_1, %weights_table_1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [512, 4864, 1, 1],
        strides = [1, 1]}
            -> tensor<1x512x1x4xf16, {order = #NHWC}>

    return %conv_2 : tensor<1x512x1x4xf16, {order = #NHWC}>

    // The tiling strategy of conv_2 should be [1, 4, 1, 1] for prefetching
    // Prefetching check could skip the intermediate view ops
    // CHECK-DAG:   [[WT:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK:       [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[CONV_1]])
    // CHECK:       [[SHAPECAST_CAST:%.+]] = VPU.ShapeCast {shape = [1, 4, 256, 19]} inputs([[LAYOUT_CAST]]
    // CHECK:       [[AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[SHAPECAST_CAST]])
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[AFFINE_RESHAPE]])
    // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = VPU.AffineReshape([[PERMUTE_CAST]])
    // CHECK-DAG:   [[WT_1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2:%.+]] = VPU.NCE.Convolution([[AFFINE_RESHAPE_1]], [[WEIGHTS_1]], [[WT_1]])
    // CHECK-SAME:      tilingStrategy = [1, 4, 1, 1]
}
