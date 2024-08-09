//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nce-ops-to-4d %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution
func.func @ConvertNceOpsTo4DConvolution(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x61xf16> {
    %FILTERS = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [2], pads_begin = [3], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x61xf16>
    return %RESULT : tensor<1x1x61xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 3]
    // CHECK-SAME:      pads_end = [0, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x61xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 61]} : tensor<1x1x1x61xf16> -> tensor<1x1x61xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolutionWithPostOp
func.func @ConvertNceOpsTo4DConvolutionWithPostOp(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x64xf16> {
    %cts = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1], pads_begin = [2], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64xf16> -> tensor<1x1x64xf16>

    return %1 : tensor<1x1x64xf16>

    // CHECK:       [[VAL0:%.*]] = IE.Convolution
    // CHECK-SAME:       {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 2],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:       tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x64xf16>

    // CHECK:       [[VAL1:%.*]] = IE.Reshape([[VAL0]]) {shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>

    // CHECK:       [[VAL2:%.*]] = IE.ReLU([[VAL1]]) : tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:       return [[VAL2]] : tensor<1x1x64xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution
func.func @ConvertNceOpsTo4DGroupConvolution(%arg0: tensor<1x16x30xf16>) -> tensor<1x8x28xf16>{
    %FILTERS = const.Declare tensor<8x8x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf32>, [#const.Reshape<[8, 8, 3]>, #const.ConvertElemType<f16>]
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1], groups = 2, pads_begin = [0], pads_end = [0], strides = [1]} : tensor<1x16x30xf16>, tensor<8x8x3xf16> -> tensor<1x8x28xf16>
    return %RESULT : tensor<1x8x28xf16>

    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x1x30xf16>, tensor<8x8x1x3xf16> -> tensor<1x8x1x28xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 8, 28]} : tensor<1x8x1x28xf16> -> tensor<1x8x28xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution
func.func @ConvertNceOpsTo4DTransposedConvolution(%arg0: tensor<1x384x1344xf16>) -> tensor<1x192x5380xf16> {
    %FILTERS = const.Declare tensor<192x384x8xf16> = dense<1.000000e+00> : tensor<192x384x8xf16>
    %RESULT = IE.TransposedConvolution(%arg0, %FILTERS) {
        dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0], pads_begin = [0], pads_end = [0], strides = [4]} : tensor<1x384x1344xf16>, tensor<192x384x8xf16> -> tensor<1x192x5380xf16>

    return %RESULT : tensor<1x192x5380xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 384, 1, 1344]} : tensor<1x384x1344xf16> -> tensor<1x384x1x1344xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<192x384x1x8xf16> = dense<1.000000e+00> : tensor<192x384x8xf16>, [#const.Reshape<[192, 384, 1, 8]>]

    // CHECK:       [[TRANSPOSED_CONV:%.*]] = IE.TransposedConvolution([[RESHAPE_INPUT]], [[CST_0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:      output_padding = [0, 0]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      tensor<1x384x1x1344xf16>, tensor<192x384x1x8xf16> -> tensor<1x192x1x5380xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[TRANSPOSED_CONV]]) {shape_value = [1, 192, 5380]} : tensor<1x192x1x5380xf16> -> tensor<1x192x5380xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x192x5380xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DMaxpool
func.func @ConvertNceOpsTo4DMaxpool(%arg0: tensor<1x512x16xf16>) -> tensor<1x512x1xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [16], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [16]} : tensor<1x512x16xf16> -> tensor<1x512x1xf16>
    return %RESULT : tensor<1x512x1xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 512, 1, 16]} : tensor<1x512x16xf16> -> tensor<1x512x1x16xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [1, 16]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 16]
    // CHECK-SAME:      tensor<1x512x1x16xf16> -> tensor<1x512x1x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[MAXPOOL]]) {shape_value = [1, 512, 1]} : tensor<1x512x1x1xf16> -> tensor<1x512x1xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x512x1xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DAvgpool
func.func @ConvertNceOpsTo4DAvgpool(%arg0: tensor<1x512x16xf16>) -> tensor<1x512x1xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [16], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [16]} : tensor<1x512x16xf16> -> tensor<1x512x1xf16>
    return %RESULT : tensor<1x512x1xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 512, 1, 16]} : tensor<1x512x16xf16> -> tensor<1x512x1x16xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [1, 16]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 16]
    // CHECK-SAME:      tensor<1x512x1x16xf16> -> tensor<1x512x1x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 512, 1]} : tensor<1x512x1x1xf16> -> tensor<1x512x1xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x512x1xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DMaxpoolwithDepthOne
func.func @ConvertNceOpsTo4D3DMaxpoolwithDepthOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x25x14x14xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [1, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x25x14x14xf16>
    return %RESULT : tensor<1x32x25x14x14xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 800, 28, 28]} : tensor<1x32x25x28x28xf16> -> tensor<1x800x28x28xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      tensor<1x800x28x28xf16> -> tensor<1x800x14x14xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[MAXPOOL]]) {shape_value = [1, 32, 25, 14, 14]} : tensor<1x800x14x14xf16> -> tensor<1x32x25x14x14xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x32x25x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DMaxpoolwithDepthNotOne
func.func @ConvertNceOpsTo4D3DMaxpoolwithDepthNotOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x23x14x14xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [3, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>
    return %RESULT : tensor<1x32x23x14x14xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool(%arg0)
    // CHECK-SAME:      kernel_size = [3, 3, 3]
    // CHECK-SAME:      pads_begin = [0, 1, 1]
    // CHECK-SAME:      pads_end = [0, 1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 2, 2]
    // CHECK-SAME:      tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>

    // CHECK: return [[MAXPOOL]] : tensor<1x32x23x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DAvgpoolwithDepthOne
func.func @ConvertNceOpsTo4D3DAvgpoolwithDepthOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x25x14x14xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [1, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x25x14x14xf16>
    return %RESULT : tensor<1x32x25x14x14xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 800, 28, 28]} : tensor<1x32x25x28x28xf16> -> tensor<1x800x28x28xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      tensor<1x800x28x28xf16> -> tensor<1x800x14x14xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 32, 25, 14, 14]} : tensor<1x800x14x14xf16> -> tensor<1x32x25x14x14xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x32x25x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DAvgpoolwithDepthNotOne
func.func @ConvertNceOpsTo4D3DAvgpoolwithDepthNotOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x23x14x14xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [3, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>
    return %RESULT : tensor<1x32x23x14x14xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:      kernel_size = [3, 3, 3]
    // CHECK-SAME:      pads_begin = [0, 1, 1]
    // CHECK-SAME:      pads_end = [0, 1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 2, 2]
    // CHECK-SAME:      tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>

    // CHECK: return [[AVGPOOL]] : tensor<1x32x23x14x14xf16>
}
