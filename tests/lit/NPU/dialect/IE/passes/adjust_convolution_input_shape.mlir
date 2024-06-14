//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-input-shape %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @ReshapeInputFor1x1Conv
func.func @ReshapeInputFor1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.*]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour
func.func @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour(%arg0: tensor<1x1280x77x1xf16>) -> tensor<1x320x77x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x77x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x77x1xf16>
    return %0 : tensor<1x320x77x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 11, 7]} : tensor<1x1280x77x1xf16> -> tensor<1x1280x11x7xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x11x7xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x11x7xf16>
    // CHECK:       [[RESHAPE1:%.*]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 77, 1]} : tensor<1x320x11x7xf16> -> tensor<1x320x77x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x77x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1ConvWithInputHeightBePrimeNumbers
func.func @NotReshapeInputFor1x1ConvWithInputHeightBePrimeNumbers(%arg0: tensor<1x1280x4091x1xf16>) -> tensor<1x320x4091x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4091x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    return %0 : tensor<1x320x4091x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4091x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4091x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment
func.func @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4095x1xf16>
}

// CHECK-LABEL: @NotReshapeInputForNon1x1Conv
func.func @NotReshapeInputForNon1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// CHECK-LABEL: @ReshapeInputFor1x1GroupConv
func.func @ReshapeInputFor1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1024, 4]} : tensor<1x320x4096x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[CONV:%.*]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1024x4xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.*]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// CHECK-LABEL: @NotReshapeInputForNon1x1GroupConv
func.func @NotReshapeInputForNon1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1GroupConvWithInputHeightBePrimeNumbers
func.func @NotReshapeInputFor1x1GroupConvWithInputHeightBePrimeNumbers(%arg0: tensor<1x320x4091x1xf16>) -> tensor<1x320x4091x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4091x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    return %0 : tensor<1x320x4091x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4091x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4091x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment
func.func @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.*]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4095x1xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeSingleConstGroupConv
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @ReshapeSingleConstGroupConv(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>


    // CHECK-DAG:   [[FILTER:%.*]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 1]>]
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 80, 1, 1]>]
    // CHECK:       [[SHAPECAST_IN:%.*]] = IE.ShapeCast {shape = [1, 80, 4, 4]} inputs([[INPUT]] : tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 80 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<80x1x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.*]] = IE.ShapeCast {shape = [1, 1280, 1, 1]} inputs([[GROUPCONV]] : tensor<1x80x4x4xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT:%.*]] : tensor<1x1280x1x1xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeSingleConstGroupConvForNCHWOut
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @NotReshapeSingleConstGroupConvForNCHWOut(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16>
    return %0 : tensor<1x1280x1x1xf16>

    // CHECK-DAG:   [[BIAS:%.*]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.*]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.*]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForBiasIsNotConst
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @NotReshapeForBiasIsNotConst(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x1280x1x1xf16>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %arg1) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTER:%.*]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.*]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForInputHAndWIsNotOne
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x2x3xf16, {order = #NHWC}>
func.func @NotReshapeForInputHAndWIsNotOne(%arg0: tensor<1x1280x2x3xf16, {order = #NHWC}>) -> tensor<1x1280x2x3xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x2x3xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x2x3xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x2x3xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.*]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.*]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.*]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForRemainingChannelNotAlign
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x1x1xf16, {order = #NHWC}>
func.func @NotReshapeForRemainingChannelNotAlign(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<64x1x1x1xf16> = dense<1.000000e+00> : tensor<64x1x1x1xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x1xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x64x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.*]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.*]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.*]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForKernelIsNot1X1
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x4x1xf16, {order = #NHWC}>
func.func @NotReshapeForKernelIsNot1X1(%arg0: tensor<1x1280x4x1xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x4x1xf16> = dense<1.000000e+00> : tensor<1280x1x4x1xf16>
    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4x1xf16, {order = #NHWC}>, tensor<1280x1x4x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.*]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.*]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.*]]
}
