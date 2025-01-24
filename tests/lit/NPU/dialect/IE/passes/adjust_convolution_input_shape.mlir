//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-input-shape %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ReshapeInputFor1x1Conv
func.func @ReshapeInputFor1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// CHECK-LABEL: @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour
func.func @ReshapeInputFor1x1ConvWithInputHeightNotDivisibleByFour(%arg0: tensor<1x1280x77x1xf16>) -> tensor<1x320x77x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x77x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x77x1xf16>
    return %0 : tensor<1x320x77x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 11, 7]} : tensor<1x1280x77x1xf16> -> tensor<1x1280x11x7xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x11x7xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x11x7xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 77, 1]} : tensor<1x320x11x7xf16> -> tensor<1x320x77x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x77x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1ConvWithInputHeightBePrimeNumbers
func.func @NotReshapeInputFor1x1ConvWithInputHeightBePrimeNumbers(%arg0: tensor<1x1280x4091x1xf16>) -> tensor<1x320x4091x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4091x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    return %0 : tensor<1x320x4091x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4091x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4091x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment
func.func @NotReshapeInputFor1x1ConvMismatchedFilterShapeAlignment(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x2x1xf16> = dense<1.000000e+00> : tensor<320x1280x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4095x1xf16>
}

// CHECK-LABEL: @NotReshapeInputForNon1x1Conv
func.func @NotReshapeInputForNon1x1Conv(%arg0: tensor<1x1280x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x1280x4096x1xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// CHECK-LABEL: @ReshapeInputFor1x1GroupConv
func.func @ReshapeInputFor1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4096x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4096x1xf16>
    return %0 : tensor<1x320x4096x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1024, 4]} : tensor<1x320x4096x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1024x4xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4096, 1]} : tensor<1x320x1024x4xf16> -> tensor<1x320x4096x1xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x4096x1xf16>
}

// CHECK-LABEL: @NotReshapeInputForNon1x1GroupConv
func.func @NotReshapeInputForNon1x1GroupConv(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x2048x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    return %0 : tensor<1x320x2048x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x320x4096x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x2048x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x2048x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1GroupConvWithInputHeightBePrimeNumbers
func.func @NotReshapeInputFor1x1GroupConvWithInputHeightBePrimeNumbers(%arg0: tensor<1x320x4091x1xf16>) -> tensor<1x320x4091x1xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4091x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    return %0 : tensor<1x320x4091x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4091x1xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4091x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x320x4091x1xf16>
}

// CHECK-LABEL: @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment
func.func @NotReshapeInputFor1x1GroupConvMismatchedFilterShapeAlignment(%arg0: tensor<1x320x4096x1xf16>) -> tensor<1x320x4095x1xf16> {
    %filter = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
    return %0 : tensor<1x320x4095x1xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x2x1xf16> = dense<1.000000e+00> : tensor<320x1x2x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<320x1x2x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4095x1xf16>
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


    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 1]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x1280x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 80, 1, 1]>]
    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 80, 4, 4]} inputs([[INPUT]] : tensor<1x1280x1x1xf16, {order = #NHWC}>) -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 80 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<80x1x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1280, 1, 1]} inputs([[GROUPCONV]] : tensor<1x80x4x4xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT:%.+]] : tensor<1x1280x1x1xf16, {order = #NHWC}>
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

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.+]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotReshapeForBiasIsNotConst
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1280x1x1xf16, {order = #NHWC}>
func.func @NotReshapeForBiasIsNotConst(%arg0: tensor<1x1280x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x1280x1x1xf16>) -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<1280x1x1x1xf16> = dense<1.000000e+00> : tensor<1280x1x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %arg1) {dilations = [1, 1], groups = 1280 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x1xf16, {order = #NHWC}>, tensor<1280x1x1x1xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.+]]
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

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.+]]
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

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.+]]
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

    // CHECK-DAG:   [[BIAS:%.+]]  = const.Declare
    // CHECK-DAG:   [[FILTER:%.+]]  = const.Declare
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution
    // CHECK:       return [[GROUPCONV:%.+]]
}

// -----

// CHECK: @ReshapeInputFor1x1ConvHeight1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1280x1x4096xf16>)
func.func @ReshapeInputFor1x1ConvHeight1(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1280x1x4096xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1280x1x1xf16> = dense<1.000000e+00> : tensor<320x1280x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 1024, 4]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x1024x4xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE0]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1280x1024x4xf16>, tensor<320x1280x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1024x4xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 1, 4096]} : tensor<1x320x1024x4xf16> -> tensor<1x320x1x4096xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x1x4096xf16>
}

// -----

// CHECK: @ReshapeInputFor1x1GroupConvHeight1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x320x1x4096xf16>)
func.func @ReshapeInputFor1x1GroupConvHeight1(%arg0: tensor<1x320x1x4096xf16>) -> tensor<1x320x1x4096xf16> {
    %filter = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    %0 = IE.GroupConvolution(%arg0, %filter, %bias) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x1x4096xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x1x4096xf16>
    return %0 : tensor<1x320x1x4096xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<1.000000e+00> : tensor<1x320x1x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 320, 4, 1024]} : tensor<1x320x1x4096xf16> -> tensor<1x320x4x1024xf16>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE0]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:        {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:     : tensor<1x320x4x1024xf16>, tensor<320x1x1x1xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x4x1024xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME{{LITERAL}}:  {dim_mapping = [[0], [1], [2, 3], [2]], shape_value = [1, 320, 1, 4096]} : tensor<1x320x4x1024xf16> -> tensor<1x320x1x4096xf16>
    // CHECK:       return [[RESHAPE1]] : tensor<1x320x1x4096xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.026685049019607842>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputForAddOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x245760x1x1xf16, {order = #NHWC}>
func.func @ReshapeInputForAddOp(%arg0: tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x245760x1x1xf16, {order = #NHWC}>, tensor<1x245760x1x1xf16, {order = #NHWC}> -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs([[INPUT]] : tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPECAST_IN]], [[SHAPECAST_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x7680x8x4xf16, {order = #NHWC}> -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs([[ADD]] : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
    // CHECK:       return [[SHAPECAST_OUT:%.+]] : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConv
func.func @ShapeCastToAlignExpandedDWConv(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640xf16, {order  = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.*]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs(%arg0 : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.*]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.*]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0039085829959196201>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConvQuant
func.func @ShapeCastToAlignExpandedDWConvQuant(%arg0: tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640x!qElemType, {order  = #NHWC}> -> tensor<1x16x640x640x!qElemType, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.*]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs(%arg0 : tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x48x640x40x!qElemType, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.*]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40x!qElemType, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.*]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}
