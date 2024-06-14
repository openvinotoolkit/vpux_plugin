//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL: func.func @ConvertShapeTo4DFrom5D
func.func @ConvertShapeTo4DFrom5D(%arg0: tensor<1x3x9x16x1xf16>, %arg1: tensor<1x1x1x1x1xf16>) -> (tensor<1x3x9x16x1xf16>) {
    %0 = IE.Sigmoid(%arg0) : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16x1xf16>
    %1 = IE.Subtract(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x9x16x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x3x9x16x1xf16>
    return %1 : tensor<1x3x9x16x1xf16>
    // CHECK-DAG:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 9, 16]} : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16>
    // CHECK-DAG:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    %[[SIGMOID:.*]] = IE.Sigmoid(%[[Reshape_0]]) : tensor<1x3x9x16xf16> -> tensor<1x3x9x16xf16>
    // CHECK:    %[[SUB:.*]] = IE.Subtract(%[[SIGMOID]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x9x16xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x9x16xf16>
    // CHECK:    %[[Reshape_2:.*]] = IE.AffineReshape(%[[SUB]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 3, 9, 16, 1]} : tensor<1x3x9x16xf16> -> tensor<1x3x9x16x1xf16>
    // CHECK:    return %[[Reshape_2]]
}

// -----

// CHECK-LABEL: @ConvertScalarAdd
func.func @ConvertScalarAdd(%arg0: tensor<3614x4xf32>) -> tensor<f32> {
    %cst = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<3614x4xf32> -> tensor<3614x4xf16>
    %1 = IE.Reshape(%0) {shape_value = [1, 1, 14456, 1]} : tensor<3614x4xf16> -> tensor<1x1x14456x1xf16>
    %2 = IE.MaxPool(%1) {kernel_size = [14456, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x14456x1xf16> -> tensor<1x1x1x1xf16>
    %3 = IE.Reshape(%2) {shape_value = []} : tensor<1x1x1x1xf16> -> tensor<f16>
    %4 = IE.Add(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<f16>, tensor<f16> -> tensor<f16>
    %5 = IE.Cos(%4) : tensor<f16> -> tensor<f16>
    %6 = IE.Convert(%5) {dstElemType = f32} : tensor<f16> -> tensor<f32>

    return %6 : tensor<f32>

    //CHECK:            [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<f16>, [#const.Reshape<[1, 1, 1, 1]>]
    //CHECK:            [[CONVERT_0:%.*]] = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x3614x4xf32> -> tensor<1x1x3614x4xf16>
    //CHECK:            [[RESHAPE_0:%.*]] = IE.AffineReshape([[CONVERT_0]])
    //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1, 14456, 1]} : tensor<1x1x3614x4xf16> -> tensor<1x1x14456x1xf16>
    //CHECK:            [[MAXPOOL_0:%.*]] = IE.MaxPool([[RESHAPE_0]])
    //CHECK-SAME:           {kernel_size = [14456, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x14456x1xf16> -> tensor<1x1x1x1xf16>
    //CHECK:            [[ADD_0:%.*]] = IE.Add([[MAXPOOL_0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    //CHECK:            IE.Cos([[ADD_0]]) : tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

}

// -----

func.func @Convert2dTopKPositiveAxis(%arg0: tensor<80x77xsi32>) -> (tensor<80x1xsi32>, tensor<80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<80x1xsi32>, tensor<80x1xsi32>

    return %output_values, %target_shape : tensor<80x1xsi32>, tensor<80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [1, 1, 80, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x1x80x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>}
    // CHECK-SAME:         tensor<1x1x80x77xsi32> -> tensor<1x1x80x1xsi32>, tensor<1x1x80x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x80x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x80x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]

}

// -----

// CHECK-LABEL: @Convert2dTopKNegativeAxis
func.func @Convert2dTopKNegativeAxis(%arg0: tensor<80x77xsi32>) -> (tensor<1x77xsi32>, tensor<1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = -2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<1x77xsi32>, tensor<1x77xsi32>

    return %output_values, %target_shape : tensor<1x77xsi32>, tensor<1x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [1, 1, 80, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x1x80x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x80x77xsi32> -> tensor<1x1x1x77xsi32>, tensor<1x1x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKPositiveAxis
func.func @Convert3dTopKPositiveAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x1x77xsi32>, tensor<60x1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    return %output_values, %target_shape : tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 60, 80, 77]} : tensor<60x80x77xsi32> -> tensor<1x60x80x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x60x80x77xsi32> -> tensor<1x60x1x77xsi32>, tensor<1x60x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [60, 1, 77]} : tensor<1x60x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [60, 1, 77]} : tensor<1x60x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKFirstAxis
func.func @Convert3dTopKFirstAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<1x80x77xsi32>, tensor<1x80x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 0 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<1x80x77xsi32>, tensor<1x80x77xsi32>

    return %output_values, %target_shape : tensor<1x80x77xsi32>, tensor<1x80x77xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 1, 60, 6160]} : tensor<60x80x77xsi32> -> tensor<1x1x60x6160xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 2 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x60x6160xsi32> -> tensor<1x1x1x6160xsi32>, tensor<1x1x1x6160xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [1, 80, 77]} : tensor<1x1x1x6160xsi32> -> tensor<1x80x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [1, 80, 77]} : tensor<1x1x1x6160xsi32> -> tensor<1x80x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKLastAxis
func.func @Convert3dTopKLastAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x80x1xsi32>, tensor<60x80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 2 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x80x1xsi32>, tensor<60x80x1xsi32>

    return %output_values, %target_shape : tensor<60x80x1xsi32>, tensor<60x80x1xsi32>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {
    // CHECK-SAME{LITERAL}:   shape_value = [1, 1, 4800, 77]} : tensor<60x80x77xsi32> -> tensor<1x1x4800x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} :
    // CHECK-SAME:         tensor<1x1x4800x77xsi32> -> tensor<1x1x4800x1xsi32>, tensor<1x1x4800x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.Reshape([[VALUE]]) {
    // CHECK-SAME:         shape_value = [60, 80, 1]} : tensor<1x1x4800x1xsi32> -> tensor<60x80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.Reshape([[SHAPE]]) {
    // CHECK-SAME:         shape_value = [60, 80, 1]} : tensor<1x1x4800x1xsi32> -> tensor<60x80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}
