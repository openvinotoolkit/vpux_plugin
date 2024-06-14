//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
// CHECK-LABEL: func.func @NoTilingConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<128x32x3x3xf16, {order = #YXOI}>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x128x1x1xf16>
func.func @NoTilingConv(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<128x32x3x3xf16, {order = #YXOI}>,
        %bias: tensor<1x128x1x1xf16>)
            -> tensor<1x128x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<128x32x3x3xf16, {order = #YXOI}>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-NOT:           tilingStrategy
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @NoTilingMaxPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x170x170xf16>
func.func @NoTilingMaxPool(
        %input: tensor<1x16x170x170xf16>)
            -> tensor<1x16x170x170xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x170x170xf16> -> tensor<1x16x170x170xf16>
    return %1 : tensor<1x16x170x170xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-NOT:           tilingStrategy
    // CHECK-SAME:      -> tensor<1x16x170x170xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x170x170xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<256x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<256x1x1x4xsi32, {order = #NHWC}>
func.func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<256x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<256x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = #VPU.eltwise_type<ADD>} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<256x144x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NHWC}> -> tensor<1x256x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = #VPU.eltwise_type<ADD>}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_2:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]]) {
    // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:        rawFilterShape = [256, 144, 3, 3],
    // CHECK-SAME:        strides = [1, 1], tilingStrategy = [1, 1, 5, 1]}
    // CHECK-SAME:          -> tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[CONV_2]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<128x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 3, 1]
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x170x170xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x170x170xf16, {order = #NHWC}>) -> tensor<1x16x170x170xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x170x170xf16, {order = #NHWC}>

    return %0 : tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:      = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]] , [[ACTIVATION_WINDOW]] ) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 4, 1]
    // CHECK-SAME:      } -> tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x16x170x170xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
func.func @NoTileWithSOH(
        %arg0: tensor<1x32x100x100xf16, {order = #NHWC}>)
            -> tensor<1x128x100x100xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1>
        : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-NOT:           tilingStrategy
    // CHECK-SAME:          tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x100x100xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>
func.func @TileWithSOH(
        %arg0: tensor<1x16x200x200xf16, {order = #NHWC}>)
            -> tensor<1x32x200x200xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1>
        : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]
    } -> tensor<1x32x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x32x200x200xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          tensor<1x32x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[CONV1]] : tensor<1x32x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x10x10xf16, {order = #NHWC}>
func.func @NoTileWithSOK(
        %arg0: tensor<1x32x10x10xf16, {order = #NHWC}>)
            -> tensor<1x240x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<240x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<240x1x1x4xsi32> = dense<1>
        : tensor<240x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x240x10x10xf16, {order = #NHWC}>

    return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>


    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
    // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-NOT:           tilingStrategy
    // CHECK-SAME:          tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x240x10x10xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func.func @TileWithSOK(
        %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
            -> tensor<1x384x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<384x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<384x1x1x4xsi32> = dense<1>
        : tensor<384x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [384, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x384x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:          : tensor<384x32x7x7xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<384x1x1x4xsi32> = dense<1>
    // CHECK-SAME:          : tensor<384x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
    // CHECK-SAME:          rawFilterShape = [384, 32, 7, 7]
    // CHECK-SAME:          strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x384x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeConstPipeliningSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func.func @LargeConstPipeliningSOK(
        %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
            -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1>
        : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:          : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<1>
    // CHECK-SAME:          : tensor<512x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [512, 256, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x20x20xf16, {order = #NHWC}>
func.func @SplitNCEEltwise(
        %arg0: tensor<1x512x20x20xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x20x20xf16, {order = #NHWC}>)
            -> tensor<1x512x20x20xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>
    } -> tensor<1x512x20x20xf16, {order = #NHWC}>

    return %0 : tensor<1x512x20x20xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x512x20x20xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x512x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NoPrefetchingForEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x48x48xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x48x48xf16, {order = #NHWC}>
func.func @NoPrefetchingForEltwise(
        %arg0: tensor<1x32x48x48xf16, {order = #NHWC}>,
        %arg1: tensor<1x64x48x48xf16, {order = #NHWC}>)
            -> tensor<1x64x48x48xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x48x48xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>
    } -> tensor<1x64x48x48xf16, {order = #NHWC}>

    return %1 : tensor<1x64x48x48xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x48x48xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]]) {op_type = #VPU.eltwise_type<ADD>}
    // CHECK-NOT:           tilingStrategy
    // CHECK-SAME:          -> tensor<1x64x48x48xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x64x48x48xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<128x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<128x1x1x384xi1>, is_weights

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<128x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 3, 1]}
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x3840xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x3840xf16, {order = #NHWC}>) -> tensor<1x16x1x3840xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>, quant_scale = [2.500000e-01]>, strides = [1, 1]} -> tensor<1x16x1x3840xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x3840xf16, {order = #NHWC}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]
    // CHECK-SAME:      -> tensor<1x16x1x3840xf16, {order = #NHWC}>

    // CHECK:       return [[AVGPOOL]] : tensor<1x16x1x3840xf16, {order = #NHWC}>
}

// -----


// CHECK-LABEL: @SplitAveragePoolNoTileOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x7680xf16>
func.func @SplitAveragePoolNoTileOverW(%arg0: tensor<1x16x7x7680xf16>) -> tensor<1x16x1x7680xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x7x7680xf16> -> tensor<1x16x1x7680xf16>

    return %0 : tensor<1x16x1x7680xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [7, 1],
    // CHECK-SAME:      pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-NOT        tilingStrategy
    // CHECK-SAME:      : tensor<1x16x7x7680xf16> -> tensor<1x16x1x7680xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x7680xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitOverWForSOHCompatibility
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x18x3000xf16, {order = #NHWC}>
func.func @SplitOverWForSOHCompatibility(%arg0: tensor<1x16x18x3000xf16, {order = #NHWC}>) -> tensor<1x16x18x3000xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } -> tensor<1x16x18x3000xf16, {order = #NHWC}>

    return %0 : tensor<1x16x18x3000xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[ACT_WIN:%.+]] = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]] , [[ACT_WIN]] )
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:       tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      -> tensor<1x16x18x3000xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x16x18x3000xf16, {order = #NHWC}>
}
