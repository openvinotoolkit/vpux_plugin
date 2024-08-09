//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitSwConvOverOC
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>,
    // CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x24x3x3xf16>,
    // CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
    func.func @SplitSwConvOverOC(
            %input: tensor<1x24x64x64xf16>,
            %filter: tensor<256x24x3x3xf16>,
            %bias: tensor<1x256x1x1xf16>)
                -> tensor<1x256x64x64xf16> {
        %1 = VPU.Convolution(%input, %filter, %bias) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x24x64x64xf16>, tensor<256x24x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
        return %1 : tensor<1x256x64x64xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
        // CHECK-SAME:          dilations = [1, 1]
        // CHECK-SAME:          pads_begin = [1, 1]
        // CHECK-SAME:          pads_end = [1, 1]
        // CHECK-SAME:          strides = [1, 1]
        // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
        // CHECK-SAME:      -> tensor<1x256x64x64xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitSwMaxPoolOverH
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x168xf16>
    func.func @SplitSwMaxPoolOverH(
            %input: tensor<1x16x244x168xf16>)
                -> tensor<1x16x244x168xf16> {
        %1 = VPU.MaxPool(%input) {
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x244x168xf16> -> tensor<1x16x244x168xf16>
        return %1 : tensor<1x16x244x168xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
        // CHECK-SAME:          kernel_size = [3, 3]
        // CHECK-SAME:          pads_begin = [1, 1]
        // CHECK-SAME:          pads_end = [1, 1]
        // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
        // CHECK-SAME:          strides = [1, 1]
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
        // CHECK-SAME:      : tensor<1x16x244x168xf16> -> tensor<1x16x244x168xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x244x168xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitSoftMaxWithSoK
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x4096x4096xf16>
    func.func @SplitSoftMaxWithSoK(%arg0: tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16>
        return %0 : tensor<1x8x4096x4096xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.SoftMax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 94, 1]}
        // CHECK-SAME:      : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x4096x4096xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitSoftMaxOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x384xf16>
    func.func @SplitSoftMaxOverW(%arg0: tensor<1x20x256x384xf16>) -> tensor<1x20x256x384xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x384xf16> -> tensor<1x20x256x384xf16>
        return %0 : tensor<1x20x256x384xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.SoftMax([[INPUT]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 1, 6]}
        // CHECK-SAME:      : tensor<1x20x256x384xf16> -> tensor<1x20x256x384xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x20x256x384xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @InterpSplitOverC
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>
    func.func @InterpSplitOverC(
            %input1: tensor<1x24x64x64xf16>)
                -> tensor<1x24x256x256xf16> {

        %0 = VPU.Interpolate(%input1) {
                attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
                axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
            tensor<1x24x64x64xf16> -> tensor<1x24x256x256xf16>

        return %0 : tensor<1x24x256x256xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT]]
        // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
        // CHECK-SAME:          tilingStrategy = [1, 1, 3, 1]
        // CHECK-SAME:      :  tensor<1x24x64x64xf16>
        // CHECK-SAME:      -> tensor<1x24x256x256xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @InterpSplitOverH
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
    func.func @InterpSplitOverH(
        %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
                -> tensor<1x64x192x320xf16, {order = #NHWC}> {
        %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            sizes_attr = [192, 320],
            tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
            tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
        return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>

        // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 7]
        // CHECH-SAME:  : tensor<1x64x48x80xf16, {order = #NHWC}>
        // CHECH-SAME:  -> tensor<1x64x192x320xf16, {order = #NHWC}>

        // CHECK:  return [[INTERP0]] : tensor<1x64x192x320xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @InterpSplitOverHW
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x35x35xf16, {order = #NHWC}>
    func.func @InterpSplitOverHW(
        %input1: tensor<1x128x35x35xf16, {order = #NHWC}>)
                -> tensor<1x128x168x335xf16, {order = #NHWC}> {
        %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            sizes_attr = [168, 335]} :
            tensor<1x128x35x35xf16, {order = #NHWC}> -> tensor<1x128x168x335xf16, {order = #NHWC}>
        return %0 : tensor<1x128x168x335xf16, {order = #NHWC}>

        // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 11]
        // CHECH-SAME:  : tensor<1x128x35x35xf16, {order = #NHWC}>
        // CHECH-SAME:  -> tensor<1x128x168x335xf16, {order = #NHWC}>

        // CHECK:  return [[INTERP0]] : tensor<1x128x168x335xf16, {order = #NHWC}>

    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @InterpSplitOverCNoCommonFactor
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x31x31xf16, {order = #NHWC}>
    func.func @InterpSplitOverCNoCommonFactor(
        %arg0: tensor<1x64x31x31xf16, {order = #NHWC}>)
                -> tensor<1x64x121x121xf16, {order = #NHWC}> {
        %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            sizes_attr = [121, 121],
            tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
            tensor<1x64x31x31xf16, {order = #NHWC}> -> tensor<1x64x121x121xf16, {order = #NHWC}>
        return %0 : tensor<1x64x121x121xf16, {order = #NHWC}>

        // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
        // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]
        // CHECH-SAME:  : tensor<1x64x31x31xf16, {order = #NHWC}>
        // CHECH-SAME:  -> tensor<1x64x121x121xf16, {order = #NHWC}>

        // CHECK:  return [[INTERP0]] : tensor<1x64x121x121xf16, {order = #NHWC}>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitPReluOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
    func.func @SplitPReluOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %cst = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 8, 1, 1]>]
        %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x960xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

        // CHECK:       [[OUTPUT:%.+]] = VPU.PRelu([[INPUT]], [[CST]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitLeakyReluOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
    func.func @SplitLeakyReluOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LeakyRelu([[INPUT]]) {
        // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64, tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

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
        // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [576, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 1, 1]}
        // CHECK-SAME:          -> tensor<1x576x20x20xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL:   @SplitNCEConvOverOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x48xf16, {order = #NHWC}>
    func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x48xf16, {order = #NHWC}>) -> tensor<1x256x64x48xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [256, 32, 3, 3],
            strides = [1, 1]
        } -> tensor<1x256x64x48xf16, {order = #NHWC}>

        return %0 : tensor<1x256x64x48xf16, {order = #NHWC}>

        // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

        // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1>
        // CHECK-SAME:      : tensor<256x1x1x4xsi32>

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
        // CHECK-SAME:          -> tensor<1x256x64x48xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x256x64x48xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitNCEPoolOverH
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x256xf16, {order = #NHWC}>)
    func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x256xf16, {order = #NHWC}>) -> tensor<1x16x340x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            strides = [1, 1]
        } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        return %0 : tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
        // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:      tilingStrategy = [1, 1, 7, 1]
        // CHECK-SAME:      } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x340x256xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

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

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @TileWithSOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x210x512xf16, {order = #NHWC}>
    func.func @TileWithSOH(
            %arg0: tensor<1x16x210x512xf16, {order = #NHWC}>)
                -> tensor<1x32x210x512xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1>
            : tensor<32x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [32, 16, 3, 3],
            strides = [1, 1]
        } -> tensor<1x32x210x512xf16, {order = #NHWC}>

        return %0 : tensor<1x32x210x512xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>
        // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
        // CHECK-SAME:          tensor<1x32x210x512xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x32x210x512xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

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

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @TileWithSOK
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
    func.func @TileWithSOK(
            %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
                -> tensor<1x768x30x30xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1>
            : tensor<768x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
            rawFilterShape = [768, 32, 7, 7],
            strides = [1, 1]
        } -> tensor<1x768x30x30xf16, {order = #NHWC}>

        return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:          : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
        // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1>
        // CHECK-SAME:          : tensor<768x1x1x4xsi32>

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
        // CHECK-SAME:          rawFilterShape = [768, 32, 7, 7],
        // CHECK-SAME:          strides = [1, 1],
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]}
        // CHECK-SAME:        -> tensor<1x768x30x30xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x768x30x30xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @LargeConstPipeliningSOKFor
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
    func.func @LargeConstPipeliningSOKFor(
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

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [512, 256, 3, 3]
        // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
        // CHECK-SAME:          -> tensor<1x512x14x14xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x512x14x14xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

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
        // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, tilingStrategy = [1, 2, 1, 1]}
        // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

        // return [[ELTWISE_0]] : tensor<1x512x28x28xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @NoPrefetchingForEltwise
    // CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x50xf16, {order = #NHWC}>,
    // CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x50xf16, {order = #NHWC}>
    func.func @NoPrefetchingForEltwise(
            %arg0: tensor<1x32x70x50xf16, {order = #NHWC}>,
            %arg1: tensor<1x64x70x50xf16, {order = #NHWC}>)
                -> tensor<1x64x70x50xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [64, 32, 3, 3],
            strides = [1, 1]
        } -> tensor<1x64x70x50xf16, {order = #NHWC}>

        %1 = VPU.NCE.Eltwise(%0, %arg1) {
            op_type = #VPU.eltwise_type<ADD>
        } -> tensor<1x64x70x50xf16, {order = #NHWC}>

        return %1 : tensor<1x64x70x50xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>

        // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          -> tensor<1x64x70x50xf16, {order = #NHWC}>

        // Eltwise is not tiled for prefetching
        // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]])
        // CHECK-SAME:              op_type = #VPU.eltwise_type<ADD>
        // CHECK-NOT:               tilingStrategy
        // CHECK-SAME:          -> tensor<1x64x70x50xf16, {order = #NHWC}>

        // return [[ELTWISE]] : tensor<1x64x70x50xf16, {order = #NHWC}>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL:   @SplitSparseNCEConvOverOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x60xf16, {order = #NHWC}>
    func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x60xf16, {order = #NHWC}>) -> tensor<1x160x80x60xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
            -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
        %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [160, 32, 3, 3],
            strides = [1, 1]
        } -> tensor<1x160x80x60xf16, {order = #NHWC}>

        return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

        // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

        // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor
        // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
        // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

        // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
        // CHECK-SAME:      : tensor<160x1x1x4xsi32>

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3]
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
        // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitNCEAveragePoolOverW
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x8640xf16, {order = #NHWC}>
    func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x8640xf16, {order = #NHWC}>) -> tensor<1x16x1x8640xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>, quant_scale = [2.500000e-01]>, strides = [1, 1]} -> tensor<1x16x1x8640xf16, {order = #NHWC}>
        return %0 : tensor<1x16x1x8640xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
        // CHECK-SAME:      tilingStrategy = [1, 1, 1, 4]
        // CHECK-SAME:      -> tensor<1x16x1x8640xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8640xf16, {order = #NHWC}>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitAveragePoolOverW
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x184320xf16>
    func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x184320xf16>) -> tensor<1x1x1x184320xf16> {
        %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x184320xf16> -> tensor<1x1x1x184320xf16>

        return %0 : tensor<1x1x1x184320xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]])
        // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]
        // CHECK-SAME:      -> tensor<1x1x1x184320xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x1x1x184320xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @ClampSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @ClampSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Clamp([[INPUT]]) {
        // CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @ReLUSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.ReLU(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @LogSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @LogSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Log(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Log([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @AbsSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @AbsSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Abs(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Abs([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitFloorModEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.FloorMod([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitModEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitModEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Mod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Mod([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitPowerEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Power([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitLogicalOrEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalOr([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitLogicalXorEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalXor([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitNotEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NotEqual([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitLessEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitLessEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LessEqual([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitGreaterEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitGreaterEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.GreaterEqual([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitErfOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @SplitErfOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Erf(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Erf([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitFloorOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @SplitFloorOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Floor(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Floor([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @TanSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @TanSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Tan(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Tan([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SwishSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @SwishSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Swish([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @HSigmoidSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @HSigmoidSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.HSigmoid([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitNegativeActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Negative(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Negative(%arg0) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitCeilingActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Ceiling([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitSignActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitSignActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Sign(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Sign([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitSelectEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_2:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
    func.func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>, %arg2: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Select([[INPUT_0]], [[INPUT_1]], [[INPUT_2]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 3, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitAndEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitAndEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitRoundActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitRoundActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Round([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitGeluActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitGeluActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Gelu(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Gelu([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitTopK
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x384xf16>) -> (tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>)
    func.func @SplitTopK(%arg0: tensor<1x5x512x384xf16>) -> (tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>) {
        %output_values, %target_shape = VPU.TopK(%arg0) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x512x384xf16> -> tensor<1x1x512x384xf16>, tensor<1x1x512x384xsi32>
        %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x384xf16> -> tensor<1x1x512x384xf32>
        return %0, %target_shape : tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>

        // CHECK: [[OUTPUT_VALUE:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK([[INPUT_0]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>, tilingStrategy = [1, 1, 3, 1]} : tensor<1x5x512x384xf16> -> tensor<1x1x512x384xf16>, tensor<1x1x512x384xsi32>
        // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x384xf16> -> tensor<1x1x512x384xf32>
        // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @SplitStridedSliceOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
    func.func @SplitStridedSliceOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x480xf16> {
        %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 960], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x480xf16>
        return %0 : tensor<1x8x80x480xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.StridedSlice([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x480xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x480xf16>
    }

}
    // -----

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: @SplitLogicalNotEltwiseSw
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalNot([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL:   @SplitNCECompressConv
    func.func @SplitNCECompressConv(
            %arg0: tensor<1x4x512x512xf16, {order = #NHWC}>,
            %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>,
            %arg2: tensor<64x1x1x4xsi32>)
            -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
            cm_sp_pattern = 15 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            ppe = #VPU.PPETask<
                    clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>
                  >,
            rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
        } -> tensor<1x64x256x256xf16, {order = #NHWC}>

        return %0 : tensor<1x64x256x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
        // CHECK-SAME:      cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        // CHECK-SAME:      pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        // CHECK-SAME:      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        // CHECK-SAME:      rawFilterShape = [64, 4, 7, 7], strides = [2, 2], tilingStrategy = [1, 1, 2, 1]}
        // CHECK-SAME:      -> tensor<1x64x256x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x64x256x256xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    !qElemType = !quant.uniform<u8:f16, 1.0>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @PrefetchTilingWithParentConsidered
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<512x1x1x4xsi32, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<2048x1x1x4xsi32, {order = #NHWC}>
    func.func @PrefetchTilingWithParentConsidered(
            %input: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
            %weights1: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
            %weights2: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>,
            %weights_table1: tensor<512x1x1x4xsi32, {order = #NHWC}>,
            %weights_table2: tensor<2048x1x1x4xsi32, {order = #NHWC}>)
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            rawFilterShape = [512, 512, 3, 3], strides = [2, 2]}
                -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
        %1 = VPU.NCE.Convolution(%0, %weights2, %weights_table2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [2048, 512, 1, 1], strides = [1, 1]}
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}>
        return %1 : tensor<1x2048x7x7x!qElemType, {order = #NHWC}>

        // Prefetching mode is triggered for the child conv
        // with tiled parent memory considered
        // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
        // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
        // CHECK-SAME:  tilingStrategy = [1, 64, 1, 1]
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    !qElemType = !quant.uniform<u8:f16, 1.0>

module @executors {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    // CHECK-LABEL: func.func @PrefetchTilingWithSOHParentConsidered
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<512x1x1x4xsi32, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<2048x1x1x4xsi32, {order = #NHWC}>
    func.func @PrefetchTilingWithSOHParentConsidered(
            %input: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
            %weights1: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
            %weights2: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>,
            %weights_table1: tensor<512x1x1x4xsi32, {order = #NHWC}>,
            %weights_table2: tensor<2048x1x1x4xsi32, {order = #NHWC}>)
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            rawFilterShape = [512, 512, 3, 3], strides = [2, 2]}
                -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
        %1 = VPU.NCE.Convolution(%0, %weights2, %weights_table2) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [2048, 512, 1, 1], strides = [1, 1]}
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}>
        return %1 : tensor<1x2048x7x7x!qElemType, {order = #NHWC}>

        // Prefetching mode is triggered for the child conv
        // with tiled parent memory considered
        // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
        // CHECK-SAME:  tilingStrategy = [1, 4, 1, 1]
        // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
    }
}
