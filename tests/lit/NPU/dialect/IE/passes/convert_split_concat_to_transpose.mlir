//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-split-concat-to-transpose %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: @ConvertSplitAffineReshapeConcatToTranspose
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @ConvertSplitAffineReshapeConcatToTranspose(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x120x2x96x49xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x2x96x49xf16>

    return %3 : tensor<1x120x2x96x49xf16>

   // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #map} : tensor<1x2x120x96x49xf16> -> tensor<1x120x2x96x49xf16>
   // CHECK:       return [[TRANSPOSE]] : tensor<1x120x2x96x49xf16>
}

// -----

// CHECK-LABEL: @NotConvertSplitAffineReshapeConcatToTransposeAsSplitShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x96x2x48xf16>
func.func @NotConvertSplitAffineReshapeConcatToTransposeAsSplitShape(%arg0: tensor<1x96x2x48xf16>) -> tensor<1x2x48x96xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x96x2x48xf16> -> tensor<1x48x2x48xf16>, tensor<1x48x2x48xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x48x96xf16>, tensor<1x1x48x96xf16> -> tensor<1x2x48x96xf16>

    return %3 : tensor<1x2x48x96xf16>

    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x96x2x48xf16> -> tensor<1x48x2x48xf16>, tensor<1x48x2x48xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 48, 96]} : tensor<1x48x2x48xf16> -> tensor<1x1x48x96xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x48x96xf16>, tensor<1x1x48x96xf16> -> tensor<1x2x48x96xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x2x48x96xf16>
}

// -----

// CHECK-LABEL: @NotConvertSplitAffineReshapeConcatToTransposeAsDimMapping
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x120x96x48xf16>
func.func @NotConvertSplitAffineReshapeConcatToTransposeAsDimMapping(%arg0: tensor<1x2x120x96x48xf16>) -> tensor<1x120x2x96x48xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x48xf16> -> tensor<1x1x120x96x48xf16>, tensor<1x1x120x96x48xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 48, 0]]} : tensor<1x120x2x48x48xf16>, tensor<1x120x2x48x48xf16> -> tensor<1x120x2x96x48xf16>

    return %3 : tensor<1x120x2x96x48xf16>

    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x120x96x48xf16> -> tensor<1x1x120x96x48xf16>, tensor<1x1x120x96x48xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3], [4]], shape_value = [1, 120, 2, 48, 48]} : tensor<1x1x120x96x48xf16> -> tensor<1x120x2x48x48xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 48, 0]]} : tensor<1x120x2x48x48xf16>, tensor<1x120x2x48x48xf16> -> tensor<1x120x2x96x48xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x120x2x96x48xf16>
}

// -----

// CHECK-LABEL: @NotConvertSplitAffineReshapeConcatToTransposeAsDimSize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @NotConvertSplitAffineReshapeConcatToTransposeAsDimSize(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x120x3x96x49xf16> {
    %cst = const.Declare tensor<1x120x1x96x49xf16> = dense<0.000000e+00> : tensor<1x120x1x96x49xf16>
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.AffineReshape(%0#0) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %3 = IE.Concat(%1, %2, %cst) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x3x96x49xf16>

    return %3 : tensor<1x120x3x96x49xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]], [[CST]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x3x96x49xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x120x3x96x49xf16>
}

// -----

// CHECK-LABEL: @ConvertSplitConcatToAffineReshape
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @ConvertSplitConcatToAffineReshape(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x1x240x96x49xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.Concat(%0#0, %0#1) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 120, 0, 0]]} : tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16> -> tensor<1x1x240x96x49xf16>

    return %1 : tensor<1x1x240x96x49xf16>

   // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
   // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [2], [3], [4]], shape_value = [1, 1, 240, 96, 49]} : tensor<1x2x120x96x49xf16> -> tensor<1x1x240x96x49xf16>
   // CHECK:       return [[AFFINERESHAPE]] : tensor<1x1x240x96x49xf16>
}
