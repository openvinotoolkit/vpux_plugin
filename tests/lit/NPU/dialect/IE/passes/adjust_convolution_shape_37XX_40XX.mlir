//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-shape %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldStrideIntoKernel
func.func @FoldStrideIntoKernel(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x16x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 64]}
  // CHECK-SAME:      inputs(%arg0 : tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x64xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForDifferentKXGreaterStride
func.func @NotFoldStrideIntoKernelForDifferentKXGreaterStride(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x4xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x4xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForWmodStideNone0
func.func @NotFoldStrideIntoKernelForWmodStideNone0(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x3xf16, {order = #NHWC}> -> tensor<1x2x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelWhenChannelAligned
func.func @NotFoldStrideIntoKernelWhenChannelAligned(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<16x16x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}> -> tensor<1x16x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x16x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShape
func.func @AdjustConvolutionShape(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 45, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 42, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 39, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 9, 0, 0], [0, 36, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 33, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 15, 0, 0], [0, 30, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 27, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 21, 0, 0], [0, 24, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 21, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 27, 0, 0], [0, 18, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 15, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 33, 0, 0], [0, 12, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 9, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 39, 0, 0], [0, 6, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 3, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 0, 0, 0]>]
  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeNoneSplatBias
func.func @AdjustConvolutionShapeNoneSplatBias(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x3x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 45, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 42, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 39, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 9, 0, 0], [0, 36, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 33, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 15, 0, 0], [0, 30, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 27, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 21, 0, 0], [0, 24, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 21, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 27, 0, 0], [0, 18, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 15, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 33, 0, 0], [0, 12, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 9, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 39, 0, 0], [0, 6, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 3, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 0, 0, 0]>]
  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<1 : i64, 48 : i64>, #const.Reshape<[1, 48, 1, 1]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingRight
func.func @AdjustConvolutionShapeWithKXGreat1andPadingRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 90, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 87, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 84, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 9, 0, 0], [0, 81, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 78, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 15, 0, 0], [0, 75, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 72, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 21, 0, 0], [0, 69, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 66, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 27, 0, 0], [0, 63, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 60, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 33, 0, 0], [0, 57, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 54, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 39, 0, 0], [0, 51, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 48, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 45, 0, 0]>]
  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x96x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 48, 1, 2]} inputs([[FILTER_CST]] : tensor<48x96x1x1xf16, {order = #NHWC}>) -> tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingLeft
func.func @AdjustConvolutionShapeWithKXGreat1andPadingLeft(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 45, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 42, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 51, 0, 0], [0, 39, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 54, 0, 0], [0, 36, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 57, 0, 0], [0, 33, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 60, 0, 0], [0, 30, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 63, 0, 0], [0, 27, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 66, 0, 0], [0, 24, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 69, 0, 0], [0, 21, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 72, 0, 0], [0, 18, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 75, 0, 0], [0, 15, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 78, 0, 0], [0, 12, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 81, 0, 0], [0, 9, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 84, 0, 0], [0, 6, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 87, 0, 0], [0, 3, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 90, 0, 0], [0, 0, 0, 0]>]
  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x96x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 48, 1, 2]} inputs([[FILTER_CST]] : tensor<48x96x1x1xf16, {order = #NHWC}>) -> tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingLeftRight
func.func @AdjustConvolutionShapeWithKXGreat1andPadingLeftRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 90, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 87, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 51, 0, 0], [0, 84, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 54, 0, 0], [0, 81, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 57, 0, 0], [0, 78, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 60, 0, 0], [0, 75, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 63, 0, 0], [0, 72, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 66, 0, 0], [0, 69, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 69, 0, 0], [0, 66, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 72, 0, 0], [0, 63, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 75, 0, 0], [0, 60, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 78, 0, 0], [0, 57, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 81, 0, 0], [0, 54, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 84, 0, 0], [0, 51, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 87, 0, 0], [0, 48, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 90, 0, 0], [0, 45, 0, 0]>]
  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x144x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 48, 1, 3]} inputs([[FILTER_CST]] : tensor<48x144x1x1xf16, {order = #NHWC}>) -> tensor<48x48x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingStride
func.func @AdjustConvolutionShapeWithKXGreat1andPadingStride(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x960xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 93, 0, 0], [0, 90, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 99, 0, 0], [0, 84, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 105, 0, 0], [0, 78, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 111, 0, 0], [0, 72, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 117, 0, 0], [0, 66, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 123, 0, 0], [0, 60, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 129, 0, 0], [0, 54, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 135, 0, 0], [0, 48, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 141, 0, 0], [0, 42, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 147, 0, 0], [0, 36, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 153, 0, 0], [0, 30, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 159, 0, 0], [0, 24, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 165, 0, 0], [0, 18, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 171, 0, 0], [0, 12, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 177, 0, 0], [0, 6, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 183, 0, 0], [0, 0, 0, 0]>]

  // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x192x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 96, 1, 2]} inputs([[FILTER_CST]] : tensor<48x192x1x1xf16, {order = #NHWC}>) -> tensor<48x96x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 96, 1080, 60]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x96x1080x60xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 960]} inputs([[CONV_RET]] : tensor<1x48x1080x60xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotAdjustConvolutionShapeWhenTensorFitCMX
func.func @NotAdjustConvolutionShapeWhenTensorFitCMX(%arg0: tensor<1x12x2x8xf16, {order = #NHWC}>) -> tensor<1x2x2x8xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x12x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x12x2x8xf16, {order = #NHWC}>, tensor<2x12x1x2xf16, {order = #NHWC}> -> tensor<1x2x2x8xf16, {order = #NHWC}>
  return %0 : tensor<1x2x2x8xf16, {order = #NHWC}>

  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS_0]])
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPadingBeginStride
func.func @AdjustConvolutionShapeWithKXGreat3andPadingBeginStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 24, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 32, 0, 0], [0, 16, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 40, 0, 0], [0, 8, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 0, 0, 0]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}

  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [16, 32, 1, 2]} inputs([[FILTER_CST]] : tensor<16x64x1x1xf16, {order = #NHWC}>) -> tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPadingEndStride
func.func @AdjustConvolutionShapeWithKXGreat3andPadingEndStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 48, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 8, 0, 0], [0, 40, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 16, 0, 0], [0, 32, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}

  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [16, 32, 1, 2]} inputs([[FILTER_CST]] : tensor<16x64x1x1xf16, {order = #NHWC}>) -> tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPadingBeginEndStride
func.func @AdjustConvolutionShapeWithKXGreat3andPadingBeginEndStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 28, 0, 0], [0, 52, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 44, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 44, 0, 0], [0, 36, 0, 0]>]
  // CHECK-DAG:   [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 52, 0, 0], [0, 28, 0, 0]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}

  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [16, 32, 1, 3]} inputs([[FILTER_CST]] : tensor<16x96x1x1xf16, {order = #NHWC}>) -> tensor<16x32x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatOutputQuantInput([[INPUT_DATA:%.+]]: tensor<1x3x320x320x!qElemType, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatOutputQuantInput(%arg0: tensor<1x3x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>

  return %result : tensor<1x32x160x160xf16, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantWeights([[INPUT_DATA:%.+]]: tensor<1x3x320x320xf16, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantWeights(%arg0: tensor<1x3x320x320xf16, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>

  return %result : tensor<1x32x160x160xf16, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantOutput([[INPUT_DATA:%.+]]: tensor<1x3x320x320xf16, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantOutput(%arg0: tensor<1x3x320x320xf16, {order = #NHWC}>) -> tensor<1x32x160x160x!qElemType, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x160x160x!qElemType, {order = #NHWC}>

  return %result : tensor<1x32x160x160x!qElemType, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x160x160x!qElemType, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @Adjust1x1ConvUnalignedChannelByPadding([[INPUT_DATA:%.+]]: tensor<1x3x320x639xf16, {order = #NHWC}>)
func.func @Adjust1x1ConvUnalignedChannelByPadding(%arg0: tensor<1x3x320x639xf16, {order = #NHWC}>) -> tensor<1x3x320x320xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
            : tensor<1x3x320x639xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

  return %result : tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CST_WEIGHTS0:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 45, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS1:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 39, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS2:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 33, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS3:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 27, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS4:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 21, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS5:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 15, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS6:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 9, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS7:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 3, 0, 0]>]
    // CHECK-DAG:  [[CST_WEIGHTS8:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS9:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 54, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS10:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 60, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS11:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 66, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS12:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 72, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS13:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 78, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS14:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 84, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_WEIGHTS15:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 90, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:  [[CST_PAD:%.+]] = const.Declare tensor<1x3x320x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x320x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:    [[NEW_FILTER:%.+]] = IE.Concat([[CST_WEIGHTS0]], [[CST_WEIGHTS1]], [[CST_WEIGHTS2]], [[CST_WEIGHTS3]], [[CST_WEIGHTS4]], [[CST_WEIGHTS5]], [[CST_WEIGHTS6]], [[CST_WEIGHTS7]], [[CST_WEIGHTS8]], [[CST_WEIGHTS9]], [[CST_WEIGHTS10]], [[CST_WEIGHTS11]], [[CST_WEIGHTS12]], [[CST_WEIGHTS13]], [[CST_WEIGHTS14]], [[CST_WEIGHTS15]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:      : tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<48x48x1x1xf16, {order = #NHWC}>
    // CHECK:    [[CONCAT_PAD:%.+]] = IE.Concat([[INPUT_DATA]], [[CST_PAD]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x3x320x639xf16, {order = #NHWC}>, tensor<1x3x320x1xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
      // CHECK:    [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 48, 320, 40]} inputs([[CONCAT_PAD]] : tensor<1x3x320x640xf16, {order = #NHWC}>) -> tensor<1x48x320x40xf16, {order = #NHWC}>
      // CHECK:    [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[NEW_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x320x40xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
      // CHECK:    [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 320]} inputs([[CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x320xf16, {order = #NHWC}>
      // CHECK:    return [[SHAPE_CAST_OUT]] : tensor<1x3x320x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @Adjust1x1ConvUnalignedChannelByPadding2([[INPUT_DATA:%.+]]: tensor<1x3x320x638xf16, {order = #NHWC}>)
func.func @Adjust1x1ConvUnalignedChannelByPadding2(%arg0: tensor<1x3x320x638xf16, {order = #NHWC}>) -> tensor<1x3x320x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 4]}
            : tensor<1x3x320x638xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x160xf16, {order = #NHWC}>

  return %result : tensor<1x3x320x160xf16, {order = #NHWC}>

    // CHECK-DAG:    [[CST_WEIGHTS0:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 45, 0, 0]>]
    // CHECK-DAG:    [[CST_WEIGHTS1:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 33, 0, 0]>]
    // CHECK-DAG:    [[CST_WEIGHTS2:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 21, 0, 0]>]
    // CHECK-DAG:    [[CST_WEIGHTS3:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 9, 0, 0]>]
    // CHECK-DAG:    [[CST_WEIGHTS4:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS5:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 60, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS6:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 72, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS7:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 84, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS8:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 96, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS9:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 108, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS10:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 120, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS11:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 132, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS12:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 144, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS13:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 156, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS14:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 168, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_WEIGHTS15:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 180, 0, 0], [0, 0, 0, 0]>, #const.SubView<[0, 0, 0, 0], [3, 48, 1, 1]>]
    // CHECK-DAG:    [[CST_PAD:%.+]] = const.Declare tensor<1x3x320x2xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x320x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:    [[NEW_FILTER:%.+]] = IE.Concat([[CST_WEIGHTS0]], [[CST_WEIGHTS1]], [[CST_WEIGHTS2]], [[CST_WEIGHTS3]], [[CST_WEIGHTS4]], [[CST_WEIGHTS5]], [[CST_WEIGHTS6]], [[CST_WEIGHTS7]], [[CST_WEIGHTS8]], [[CST_WEIGHTS9]], [[CST_WEIGHTS10]], [[CST_WEIGHTS11]], [[CST_WEIGHTS12]], [[CST_WEIGHTS13]], [[CST_WEIGHTS14]], [[CST_WEIGHTS15]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:      : tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>, tensor<3x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<48x48x1x1xf16, {order = #NHWC}>
    // CHECK:    [[CONCAT_PAD:%.+]] = IE.Concat([[INPUT_DATA]], [[CST_PAD]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x3x320x638xf16, {order = #NHWC}>, tensor<1x3x320x2xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 48, 320, 40]} inputs([[CONCAT_PAD]] : tensor<1x3x320x640xf16, {order = #NHWC}>) -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[NEW_FILTER]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x320x40xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 320, 160]} inputs([[CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x160xf16, {order = #NHWC}>
    // CHECK:    return [[SHAPE_CAST_OUT]] : tensor<1x3x320x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotAdjust1x1ConvPadNumGreaterThanStride([[INPUT_DATA:%.+]]: tensor<1x3x320x635xf16, {order = #NHWC}>)
func.func @NotAdjust1x1ConvPadNumGreaterThanStride(%arg0: tensor<1x3x320x635xf16, {order = #NHWC}>) -> tensor<1x3x320x318xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
            : tensor<1x3x320x635xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x318xf16, {order = #NHWC}>

  return %result : tensor<1x3x320x318xf16, {order = #NHWC}>

    //CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK:     [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x320x635xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x320x318xf16, {order = #NHWC}>
    //CHECK:    return [[CONV]] : tensor<1x3x320x318xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotAdjustConvCaseWithExtraPadding([[INPUT_DATA:%.+]]: tensor<1x10x10x13xf16, {order = #NHWC}>)
func.func @NotAdjustConvCaseWithExtraPadding(%arg0: tensor<1x10x10x13xf16, {order = #NHWC}>) -> tensor<1x10x10x5xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<10x10x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<10x10x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 3]}
               : tensor<1x10x10x13xf16, {order = #NHWC}>, tensor<10x10x1x1xf16, {order = #NHWC}> -> tensor<1x10x10x5xf16, {order = #NHWC}>

  return %result : tensor<1x10x10x5xf16, {order = #NHWC}>

  //CHECK:    [[WEIGHTS:%.+]] = const.Declare tensor<10x10x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<10x10x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
  //CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 3]} : tensor<1x10x10x13xf16, {order = #NHWC}>, tensor<10x10x1x1xf16, {order = #NHWC}> -> tensor<1x10x10x5xf16, {order = #NHWC}>
  //CHECK:    return [[CONV]] : tensor<1x10x10x5xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @AdjustConvolutionShapeWithAlginedICButUnalignedOC([[INPUT_DATA:%.+]]: tensor<1x4x512x512xf16, {order = #NHWC}>)
func.func @AdjustConvolutionShapeWithAlginedICButUnalignedOC(%arg0: tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x4x256x256xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<4x4x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<4x4x2x2xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]}
                : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<4x4x2x2xf16, {order = #NHWC}> -> tensor<1x4x256x256xf16, {order = #NHWC}>


  return %result : tensor<1x4x256x256xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<4x32x1x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x4x2x2xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 8, 1, 2]>, #const.PadWithZero<[0, 24, 0, 0], [0, 0, 0, 0]>]
  // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<4x32x1x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x4x2x2xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 8, 1, 2]>, #const.PadWithZero<[0, 16, 0, 0], [0, 8, 0, 0]>]
  // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<4x32x1x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x4x2x2xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 8, 1, 2]>, #const.PadWithZero<[0, 8, 0, 0], [0, 16, 0, 0]>]
  // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<4x32x1x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x4x2x2xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 8, 1, 2]>, #const.PadWithZero<[0, 0, 0, 0], [0, 24, 0, 0]>]

  // CHECK:        [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 8, 512, 256]} inputs([[INPUT_DATA]] : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x8x512x256xf16, {order = #NHWC}>

  // CHECK:        [[CONCAT:%.+]] = IE.Concat([[CST_2]], [[CST_1]], [[CST_0]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
  // CHECK-SAME:      tensor<4x32x1x2xf16, {order = #NHWC}>,
  // CHECK-SAME:      tensor<4x32x1x2xf16, {order = #NHWC}>,
  // CHECK-SAME:      tensor<4x32x1x2xf16, {order = #NHWC}>,
  // CHECK-SAME:      tensor<4x32x1x2xf16, {order = #NHWC}> -> tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:        [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [16, 32, 2, 1]} inputs([[CONCAT]] : tensor<16x32x1x2xf16, {order = #NHWC}>) -> tensor<16x32x2x1xf16, {order = #NHWC}>

  // CHECK:        [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 32, 512, 64]} inputs([[SHAPE_CAST_0]] : tensor<1x8x512x256xf16, {order = #NHWC}>) -> tensor<1x32x512x64xf16, {order = #NHWC}>

  // CHECK:        [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_2]], [[SHAPE_CAST_1]]) {
  // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]}
  // CHECK-SAME:      : tensor<1x32x512x64xf16, {order = #NHWC}>, tensor<16x32x2x1xf16, {order = #NHWC}> -> tensor<1x16x256x64xf16, {order = #NHWC}>

  // CHECK:        [[SHAPE_CAST_3:%.+]] = IE.ShapeCast {shape = [1, 4, 256, 256]} inputs([[CONV]] : tensor<1x16x256x64xf16, {order = #NHWC}>) -> tensor<1x4x256x256xf16, {order = #NHWC}>

  // CHECK:        return [[SHAPE_CAST_3]] : tensor<1x4x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @SkipShapeAdjustmentWithConsideringAdjacentConvLayers([[INPUT_DATA:%.+]]: tensor<1x128x80x80xf16, {order = #NHWC}>)
func.func @SkipShapeAdjustmentWithConsideringAdjacentConvLayers(%arg0: tensor<1x128x80x80xf16, {order = #NHWC}>) -> tensor<1x128x80x80xf16, {order = #NHWC}> {
  %weights_1 = const.Declare tensor<88x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<88x128x1x1xf16>, [#const.Reorder<#NHWC>]
  %weights_2 = const.Declare tensor<88x88x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<88x88x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_3 = const.Declare tensor<128x88x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x88x1x1xf16>, [#const.Reorder<#NHWC>]

  %conv_1 = IE.Convolution(%arg0, %weights_1) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]}
                : tensor<1x128x80x80xf16, {order = #NHWC}>, tensor<88x128x1x1xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  %conv_2 = IE.Convolution(%conv_1, %weights_2) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]}
                : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<88x88x3x3xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  %conv_3 = IE.Convolution(%conv_2, %weights_3) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<128x88x1x1xf16, {order = #NHWC}> -> tensor<1x128x80x80xf16, {order = #NHWC}>

  return %conv_3 : tensor<1x128x80x80xf16, {order = #NHWC}>

  // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<88x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<88x128x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[WEIGHTS_2:%.+]] = const.Declare tensor<88x88x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<88x88x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG:   [[WEIGHTS_3:%.+]] = const.Declare tensor<128x88x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x88x1x1xf16>, [#const.Reorder<#NHWC>]

  // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT_DATA]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]} : tensor<1x128x80x80xf16, {order = #NHWC}>, tensor<88x128x1x1xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[CONV_1]], [[WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]} : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<88x88x3x3xf16, {order = #NHWC}> -> tensor<1x88x80x80xf16, {order = #NHWC}>
  // CHECK:       [[CONV_3:%.+]] = IE.Convolution([[CONV_2]], [[WEIGHTS_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x88x80x80xf16, {order = #NHWC}>, tensor<128x88x1x1xf16, {order = #NHWC}> -> tensor<1x128x80x80xf16, {order = #NHWC}>

  // CHECK:       return [[CONV_3]] : tensor<1x128x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @NotAdjustConvWithAsymmetricStrides
// CHECK-SAME:  [[INPUT_DATA:%.+]]: tensor<1x1x1x256xf16, {order = #NHWC}>
func.func @NotAdjustConvWithAsymmetricStrides(%arg0: tensor<1x1x1x256xf16, {order = #NHWC}>) -> tensor<1x64x1x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<64x1x1x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x1x1x4xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 4]}
            : tensor<1x1x1x256xf16, {order = #NHWC}>, tensor<64x1x1x4xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>

  return %result : tensor<1x64x1x64xf16, {order = #NHWC}>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<64x1x1x4xf16, {order = #NHWC}>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT_DATA]], [[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 4]} :
  // CHECK-SAME:  tensor<1x1x1x256xf16, {order = #NHWC}>,
  // CHECK-SAME:  tensor<64x1x1x4xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>
  // CHECK:       return [[CONV]]
}
