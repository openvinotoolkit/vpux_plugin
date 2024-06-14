//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --insert-identity-pool-before-op %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @InsertMaxPoolToConcatAndLRelu
func.func @InsertMaxPoolToConcatAndLRelu(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.MaxPool(%[[VAL_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.LeakyRelu(%[[VAL_1]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}

// -----

// CHECK-LABEL: @InsertMaxPoolToConcatAndClamp
func.func @InsertMaxPoolToConcatAndClamp(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.MaxPool(%[[VAL_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.Clamp(%[[VAL_1]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}

// -----

// CHECK-LABEL: @InsertMaxPoolToSplitAndLRelu
func.func @InsertMaxPoolToSplitAndLRelu(%arg0: tensor<1x128x2x32xf16>) -> (tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>) {
    %0:2 = IE.Split(%arg0) {axis_value = 1, num_splits = 2} : tensor<1x128x2x32xf16> -> tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>
    %1 = IE.LeakyRelu(%0#0) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    %2 = IE.LeakyRelu(%0#1) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>

    return %1, %2 : tensor<1x64x2x32xf16> , tensor<1x64x2x32xf16>

    // CHECK:   %[[VAL_0:.*]]:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x128x2x32xf16> -> tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.MaxPool(%[[VAL_0]]#0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.LeakyRelu(%[[VAL_1]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_3:.*]] = IE.MaxPool(%[[VAL_0]]#1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_4:.*]] = IE.LeakyRelu(%[[VAL_3]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>

    // CHECK:   return %[[VAL_2]], %[[VAL_4]] : tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>
}

// -----

// CHECK-LABEL: @InsertMaxPoolToReshapeAndClamp
func.func @InsertMaxPoolToReshapeAndClamp(%arg0: tensor<1x128x2x32xf16>) -> tensor<1x2x32x128xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 2, 32, 128]} : tensor<1x128x2x32xf16> -> tensor<1x2x32x128xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x2x32x128xf16> -> tensor<1x2x32x128xf16>

    return %1 : tensor<1x2x32x128xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 2, 32, 128]} : tensor<1x128x2x32xf16> -> tensor<1x2x32x128xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.MaxPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2x32x128xf16> -> tensor<1x2x32x128xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.Clamp(%[[VAL_1]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x2x32x128xf16> -> tensor<1x2x32x128xf16>

    // CHECK:   return %[[VAL_2]] : tensor<1x2x32x128xf16>
}

// -----

// CHECK-LABEL: @DoNotInsertMaxPoolToAddAndClamp
func.func @DoNotInsertMaxPoolToAddAndClamp(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x2x32xf16>) -> tensor<1x128x2x32xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    return %1 : tensor<1x128x2x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.Clamp(%[[VAL_0]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    // CHECK:   return %[[VAL_1]] : tensor<1x128x2x32xf16>
}

// -----

// CHECK-LABEL: @InsertMaxPoolToFakeQuantizeAndLeakyRelu
func.func @InsertMaxPoolToFakeQuantizeAndLeakyRelu(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<  1x16x8x8xf16>
    %2 = IE.LeakyRelu(%1) { negative_slope = 1.000000e-01 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>

    return %2 : tensor<1x16x8x8xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:     [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
    // CHECK:     [[MAX_POOL:%.*]] = IE.MaxPool([[FQ]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
    // CHECK:     [[LEAKY_RELU:%.*]] = IE.LeakyRelu([[MAX_POOL]]) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>

    //CHECK:  return [[LEAKY_RELU]] : tensor<1x16x8x8xf16>
}

// -----

// CHECK-LABEL: @SkipLReluWithNCEProducer
func.func @SkipLReluWithNCEProducer(%arg0: tensor<1x128x2x32xf16>) -> tensor<1x128x2x32xf16> {
     %AND_OP = IE.And(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    %LRELU = IE.LeakyRelu(%AND_OP) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    return %LRELU : tensor<1x128x2x32xf16>

    // CHECK:   [[AND_OP:%.*]] = IE.And(%arg0, %arg0) {
    // CHECK-SAME:       auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    // CHECK-SAME:   } : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>


    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[AND_OP]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   return [[LRELU]] : tensor<1x128x2x32xf16>
}

// -----

// CHECK-LABEL: @LReluWithNCEProducerAndTwoUsers
func.func @LReluWithNCEProducerAndTwoUsers(%arg0: tensor<1x128x2x32xf16>)
        -> (tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16>) {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    %LRELU = IE.LeakyRelu(%AVG_POOL) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    return %AVG_POOL, %LRELU : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16>

    // CHECK:   [[ORIG_POOL:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[NEW_POOL:%.*]] = IE.MaxPool([[ORIG_POOL]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[NEW_POOL]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   return [[ORIG_POOL]], [[LRELU]] : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16>
}

// -----

// CHECK-LABEL: @InsertMaxPoolToAddWithMultipleUsersAndRelu
func.func @InsertMaxPoolToAddWithMultipleUsersAndRelu(%arg0: tensor<1x128x4x4xf16>, %arg1: tensor<1x128x4x4xf16>) -> tensor<1x128x4x4xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %relu = IE.ReLU(%0) : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %1 = IE.Add(%0, %relu) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    return %1 : tensor<1x128x4x4xf16>

    // CHECK:   [[ADD0:%.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:   [[MAXPOOL:%.*]] = IE.MaxPool([[ADD0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:   [[RELU:%.*]] = IE.ReLU([[MAXPOOL]]) : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:   [[ADD1:%.*]] = IE.Add([[ADD0]], [[RELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    // CHECK:   return [[ADD1]] : tensor<1x128x4x4xf16>
}

// -----

// CHECK-LABEL: @NotInsertMaxPoolIfActivationHasBatch
func.func @NotInsertMaxPoolIfActivationHasBatch(%arg0: tensor<1x12x10x20xf16>) -> tensor<3x4x10x20xf16> {
    %reshape = IE.Reshape(%arg0) {shape_value = [3, 4, 10, 20]} : tensor<1x12x10x20xf16> -> tensor<3x4x10x20xf16>
    %relu = IE.ReLU(%reshape) : tensor<3x4x10x20xf16> -> tensor<3x4x10x20xf16>
    return %relu : tensor<3x4x10x20xf16>

    // CHECK:   [[RESHAPE:%.*]] = IE.Reshape(%arg0) {
    // CHECK-SAME:      shape_value = [3, 4, 10, 20]
    // CHECK-SAME:  } : tensor<1x12x10x20xf16> -> tensor<3x4x10x20xf16>

    // CHECK:   [[MAX_POOL:%.*]] = IE.MaxPool([[RESHAPE]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<3x4x10x20xf16> -> tensor<3x4x10x20xf16>

    // CHECK:   [[RELU:%.*]] = IE.ReLU([[MAX_POOL]]) : tensor<3x4x10x20xf16> -> tensor<3x4x10x20xf16>
    // CHECK:   return [[RELU]] : tensor<3x4x10x20xf16>
}
