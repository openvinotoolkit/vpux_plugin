//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scale-shift-depthwise %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @NotConvertScaleShiftToEnableCMajorWithoutConvert
func.func @NotConvertScaleShiftToEnableCMajorWithoutConvert(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x32x256x256xf16> {
    %weights_0 = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias_0 = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %weights_0, %bias_0) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>

    %weights_1 = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    %bias_1 = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    %1 = IE.Convolution(%0, %weights_1, %bias_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>

    return %1 : tensor<1x32x256x256xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:       %[[BIAS_0:.*]] = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_0:.*]] = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    // CHECK-DAG:       %[[BIAS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       %[[SCALESHIFT:.*]] =  IE.ScaleShift(%arg0, %[[WEIGHTS_1]], %[[BIAS_1]]) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[CONV:.*]] =  IE.Convolution(%[[SCALESHIFT]], %[[WEIGHTS_0]], %[[BIAS_0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @NotConvertScaleShiftToEnableCMajorWithConvert
func.func @NotConvertScaleShiftToEnableCMajorWithConvert(%arg0: tensor<1x3x512x512xui8>) -> tensor<1x32x256x256xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x512x512xui8> -> tensor<1x3x512x512xf16>

    %weights_0 = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias_0 = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %1 = IE.ScaleShift(%0, %weights_0, %bias_0) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>

    %weights_1 = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    %bias_1 = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    %2 = IE.Convolution(%1, %weights_1, %bias_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>

    return %2 : tensor<1x32x256x256xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:       %[[BIAS_0:.*]] = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_0:.*]] = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    // CHECK-DAG:       %[[BIAS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       %[[CONVERT:.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x512x512xui8> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[SCALESHIFT:.*]] =  IE.ScaleShift(%[[CONVERT]], %[[WEIGHTS_1]], %[[BIAS_1]]) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[CONV:.*]] =  IE.Convolution(%[[SCALESHIFT]], %[[WEIGHTS_0]], %[[BIAS_0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @NotConvertScaleShiftToEnableCMajorWithUPALayer
func.func @NotConvertScaleShiftToEnableCMajorWithUPALayer(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x32x256x256xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [512, 512]
         } : tensor<1x3x256x256xf16> -> tensor<1x3x512x512xf16>

    %weights_0 = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias_0 = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %1 = IE.ScaleShift(%0, %weights_0, %bias_0) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>

    %weights_1 = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    %bias_1 = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    %2 = IE.Convolution(%1, %weights_1, %bias_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>

    return %2 : tensor<1x32x256x256xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:       %[[BIAS_0:.*]] = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_0:.*]] = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    // CHECK-DAG:       %[[BIAS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       %[[INTERPOLATE:.*]] = IE.Interpolate(%arg0)
    // CHECK-SAME:      {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [512, 512]} : tensor<1x3x256x256xf16> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[SCALESHIFT:.*]] =  IE.ScaleShift(%[[INTERPOLATE]], %[[WEIGHTS_1]], %[[BIAS_1]]) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[CONV:.*]] =  IE.Convolution(%[[SCALESHIFT]], %[[WEIGHTS_0]], %[[BIAS_0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @ConvertScaleShiftAndDisableCMajorWithNCELayer
func.func @ConvertScaleShiftAndDisableCMajorWithNCELayer(%arg0: tensor<1x32x512x512xf16>) -> tensor<1x32x256x256xf16> {
    %weights = const.Declare tensor<3x32x1x1xf16> = dense<7.843020e-03> : tensor<3x32x1x1xf16>
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Clamp0", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [1, 1]} : tensor<1x32x512x512xf16>, tensor<3x32x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>

    %weights_0 = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias_0 = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %1 = IE.ScaleShift(%0, %weights_0, %bias_0) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x3x512x512xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>

    %weights_1 = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    %bias_1 = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    %2 = IE.Convolution(%1, %weights_1, %bias_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp1", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>

    return %2 : tensor<1x32x256x256xf16>

    // CHECK-NOT:   IE.ScaleShift
    // CHECK-DAG:       %[[BIAS_0:.*]] = const.Declare tensor<1x32x1x1xf16> = dense<-1.000000e+00> : tensor<1x32x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_0:.*]] = const.Declare tensor<32x3x3x3xf16> = dense<7.843020e-03> : tensor<32x3x3x3xf16>
    // CHECK-DAG:       %[[WEIGHTS_1:.*]] = const.Declare tensor<3x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>, [#const.Reshape<[3, 1, 1, 1]>]
    // CHECK-DAG:       %[[BIAS_1:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK-DAG:       %[[BIAS_2:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK-DAG:       %[[WEIGHTS_2:.*]] = const.Declare tensor<3x32x1x1xf16> = dense<7.843020e-03> : tensor<3x32x1x1xf16>

    // CHECK:       %[[CONV_0:.*]] = IE.Convolution(%arg0, %[[WEIGHTS_2]], %[[BIAS_2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Clamp0", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [1, 1]} : tensor<1x32x512x512xf16>, tensor<3x32x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[GROUPCONV:.*]] = IE.GroupConvolution(%[[CONV_0]], %[[WEIGHTS_1]], %[[BIAS_1]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x512x512xf16>, tensor<3x1x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x512x512xf16>
    // CHECK:       %[[CONV_1:.*]] = IE.Convolution(%[[GROUPCONV]], %[[WEIGHTS_0]], %[[BIAS_0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Clamp1", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, strides = [2, 2]} : tensor<1x3x512x512xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x256x256xf16>

    // CHECK:       return %[[CONV_1]]
}
