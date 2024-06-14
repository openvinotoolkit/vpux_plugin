//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-batched-layer-to-1n %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @main(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x1x1xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x1x1xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1xf16>, tensor<4x16x1x1xf16> -> tensor<1x4x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf16>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @MixedPrecisionCase(%arg0: tensor<5x16x1x1x!qElemType>, %arg1: tensor<4x16x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1x!qElemType>, tensor<4x16x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1x!qElemType> -> tensor<1x16x5x1x!qElemType>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1x!qElemType>, tensor<4x16x1x1x!qElemType1> -> tensor<1x4x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertWithWNotEq1(%arg0: tensor<512x48x1x336xf16>, %arg1: tensor<6x48x1x3xf16>) -> tensor<512x6x1x336xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<512x48x1x336xf16>, tensor<6x48x1x3xf16> -> tensor<512x6x1x336xf16>
    return %0 : tensor<512x6x1x336xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<512x48x1x336xf16> -> tensor<1x48x512x336xf16>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x48x512x336xf16>, tensor<6x48x1x3xf16> -> tensor<1x6x512x336xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x6x512x336xf16> -> tensor<512x6x1x336xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<512x6x1x336xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>
!qElemType1 = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @NoChagesPerAxisQuantization(%arg0: tensor<5x8x1x1x!qElemType>, %arg1: tensor<4x8x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x8x1x1x!qElemType>, tensor<4x8x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}

// -----

func.func @NoChangesFilterPlaneNotEq1(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x2x2xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x2x2xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertAdd(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<5x16x1x1xf16>) -> tensor<5x16x1x1xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<5x16x1x1xf16>, tensor<5x16x1x1xf16> -> tensor<5x16x1x1xf16>
    return %0 : tensor<5x16x1x1xf16>

    // CHECK: [[IN_TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[IN_TRANSPOSE2:%.*]] = IE.Transpose(%arg1) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[ADD:%.*]] = IE.Add([[IN_TRANSPOSE1]], [[IN_TRANSPOSE2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x5x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[ADD]]) {order_value = #map} : tensor<1x16x5x1xf16> -> tensor<5x16x1x1xf16>
    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x16x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
func.func @NoChangesAddOp(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<5x16x1x1xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<5x16x1x1xf16>
    return %0 : tensor<5x16x1x1xf16>

    // CHECK: [[ADD:%.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<5x16x1x1xf16>
    // CHECK: return [[ADD]] : tensor<5x16x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertGroupConvolution(%arg0: tensor<16x16x1x32xf16>) -> tensor<16x16x1x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %2 = IE.GroupConvolution(%arg0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x16x1x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<16x16x1x32xf16>
    return %2 : tensor<16x16x1x32xf16>

    // CHECK: [[FILTER:%.*]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    // CHECK: [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<16x16x1x32xf16> -> tensor<1x16x16x32xf16>
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution([[IN_TRANSPOSE]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x16x32xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x16x16x32xf16> -> tensor<16x16x1x32xf16>
    // CHECK: return [[OUT_TRANSPOSE]] : tensor<16x16x1x32xf16>

}

// -----

func.func @NoChangesGroupConvolution(%arg0: tensor<16x16x2x32xf16>) -> tensor<16x16x2x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %2 = IE.GroupConvolution(%arg0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x16x2x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<16x16x2x32xf16>
    return %2 : tensor<16x16x2x32xf16>

    // CHECK: [[FILTER:%.*]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    // CHECK: [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK-NOT: IE.Transpose
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x16x2x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<16x16x2x32xf16>
    // CHECK: return [[CONV]] : tensor<16x16x2x32xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertMaxPool(%arg0: tensor<16x16x1x3xf16>) -> tensor<16x16x1x3xf16> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<16x16x1x3xf16> -> tensor<16x16x1x3xf16>

    return %0 : tensor<16x16x1x3xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<16x16x1x3xf16> -> tensor<1x16x16x3xf16>
    // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[IN_TRANSPOSE]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x16x3xf16> -> tensor<1x16x16x3xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[MAXPOOL]]) {order_value = #map} : tensor<1x16x16x3xf16> -> tensor<16x16x1x3xf16>
    // CHECK: return [[OUT_TRANSPOSE]]  : tensor<16x16x1x3xf16>
}

// -----

func.func @NoChangesMaxPool(%arg0: tensor<16x16x3x3xf16>) -> tensor<16x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<16x16x3x3xf16> -> tensor<16x16x3x3xf16>

    return %0 : tensor<16x16x3x3xf16>

    // CHECK-NOT: IE.Transpose
    // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<16x16x3x3xf16> -> tensor<16x16x3x3xf16>
    // CHECK: return [[MAXPOOL]]  : tensor<16x16x3x3xf16>
}
