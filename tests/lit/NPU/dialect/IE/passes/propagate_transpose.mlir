//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-transpose %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithSoftmax
func.func @SwapWithSoftmax(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.SoftMax(%1) {axisInd = -1 : i64} : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[TRANSPOSE:%.*]] = IE.Transpose([[SOFTMAX]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK:        return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithGelu
func.func @SwapWithGelu(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.Gelu(%1) : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK: [[GELU:%.*]] = IE.Gelu(%arg0) : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[GELU]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithSwish
func.func @SwapWithSwish(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.Swish(%1) {beta_value = 1.000000e+00 : f64} : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>

    return %2 : tensor<1x1x16x24xf32>

    // CHECK: [[LAYER:%.*]] = IE.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[LAYER]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d2, d0)>

// CHECK-LABEL: @SwapWithConstAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x16x128x1xf16>
func.func @SwapWithConstAdd(%arg0: tensor<1x16x128x1xf16>) -> tensor<1x16x1x128xf16> {
    %cst = const.Declare tensor<1x16x1x128xf16> = dense<1.0> : tensor<1x16x1x128xf16>
    %0 = IE.Transpose(%arg0) {order_value = #map1} : tensor<1x16x128x1xf16> -> tensor<16x1x128x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [3, 3]], shape_value = [1, 16, 1, 128]} : tensor<16x1x128x1xf16> -> tensor<1x16x1x128xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x1x128xf16>

    return %2 : tensor<1x16x1x128xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x128x1xf16> = dense<1.000000e+00> : tensor<1x16x1x128xf16>, [#const.Reshape<[16, 1, 128, 1]>, #const.Transpose<#map>]
    // CHECK:           [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x128x1xf16>, tensor<1x16x128x1xf16> -> tensor<1x16x128x1xf16>
    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #map1} : tensor<1x16x128x1xf16> -> tensor<16x1x128x1xf16>
    // CHECK:           [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2], [3, 3]], shape_value = [1, 16, 1, 128]} : tensor<16x1x128x1xf16> -> tensor<1x16x1x128xf16>
    // CHECK:           return [[AFFINE_RESHAPE]] : tensor<1x16x1x128xf16>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotSwapWithConstAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x9x128x16xf16>
func.func @NotSwapWithConstAdd(%arg0: tensor<1x9x128x16xf16>) -> tensor<1x144x128x1xf16> {
    %cst = const.Declare tensor<1x144x128x1xf16> = dense<1.0> : tensor<1x144x128x1xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x9x128x16xf16> -> tensor<1x16x9x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 144, 128, 1]} : tensor<1x16x9x128xf16> -> tensor<1x144x128x1xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x128x1xf16>, tensor<1x144x128x1xf16> -> tensor<1x144x128x1xf16>

    return %2 : tensor<1x144x128x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x144x128x1xf16> = dense<1.000000e+00> : tensor<1x144x128x1xf16>
    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x9x128x16xf16> -> tensor<1x16x9x128xf16>
    // CHECK:           [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 144, 128, 1]} : tensor<1x16x9x128xf16> -> tensor<1x144x128x1xf16>
    // CHECK:           [[ADD:%.+]] = IE.Add([[AFFINE_RESHAPE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x128x1xf16>, tensor<1x144x128x1xf16> -> tensor<1x144x128x1xf16>
    // CHECK:           return [[ADD]] : tensor<1x144x128x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0031202451855528589>
!qElemType1 = !quant.uniform<u8:f16, 0.0062404903711057178>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapWithQuantizeAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x144x2304x1xf16>
func.func @SwapWithQuantizeAdd(%arg0: tensor<1x144x2304x1xf16>) -> tensor<1x1x2304x144x!qElemType> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : tensor<1x144x2304x1xf16> -> tensor<2304x144x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 2304, 144]} : tensor<2304x144x1x1xf16> -> tensor<1x1x2304x144xf16>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x2304x144xf16>, tensor<1x1x2304x144xf16> -> tensor<1x1x2304x144x!qElemType1>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType} : tensor<1x1x2304x144x!qElemType1> -> tensor<1x1x2304x144x!qElemType>

    return %3 : tensor<1x1x2304x144x!qElemType>

    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x2304x1xf16>, tensor<1x144x2304x1xf16> -> tensor<1x144x2304x1x!qElemType1>
    // CHECK:   [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x144x2304x1x!qElemType1> -> tensor<1x144x2304x1x!qElemType>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[QUANTIZE_CAST]]) {order_value = #map} : tensor<1x144x2304x1x!qElemType> -> tensor<2304x144x1x1x!qElemType>
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 2304, 144]} : tensor<2304x144x1x1x!qElemType> -> tensor<1x1x2304x144x!qElemType>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x1x2304x144x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0037092456630631989:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithDeQuantizeAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x512x2x1x!qElemType>
func.func @SwapWithDeQuantizeAdd(%arg0: tensor<1x512x2x1x!qElemType>) -> tensor<1x2x256x2xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x512x2x1x!qElemType> -> tensor<1x2x1x512x!qElemType>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 256, 2]} : tensor<1x2x1x512x!qElemType> -> tensor<1x2x256x2x!qElemType>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x256x2x!qElemType>, tensor<1x2x256x2x!qElemType> -> tensor<1x2x256x2xf16>

    return %2 : tensor<1x2x256x2xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x2x1x!qElemType>, tensor<1x512x2x1x!qElemType> -> tensor<1x512x2x1xf16>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #NHWC} : tensor<1x512x2x1xf16> -> tensor<1x2x1x512xf16>
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 256, 2]} : tensor<1x2x1x512xf16> -> tensor<1x2x256x2xf16>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x2x256x2xf16>
}
