//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-IE-to-VPU %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: func.func @SingleLayer([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
func.func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK:  [[VAR0:%.+]] = VPU.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK:  return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK: func.func @ConstantLayer() -> tensor<1x2x2x2xf16> {
func.func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf16>
    // CHECK:  return [[CST]] : tensor<1x2x2x2xf16>
}

// -----

// CHECK: func.func @Reshape([[ARG0:%.+]]: tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
func.func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK:  [[VAR0:%.*]] = VPU.Reshape([[ARG0]]) {shape_value = [1, 512]} : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    // CHECK:  return [[VAR0]] : tensor<1x512xf32>
}

// -----

// CHECK: func.func @ReshapeInGraph([[ARG0:%.*]]: tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32> {
func.func @ReshapeInGraph(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x512xf32> -> tensor<1x512xf32>
    %2 = IE.Reshape(%1) { shape_value = [1, 512, 1, 1] } : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    return %2 : tensor<1x512x1x1xf32>

    // CHECK:  [[VAR0:%.+]] = VPU.Reshape([[ARG0]]) {shape_value = [1, 512]} : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    // CHECK:  [[VAR1:%.+]] = VPU.SoftMax([[VAR0]]) {axisInd = 1 : i64} : tensor<1x512xf32> -> tensor<1x512xf32>
    // CHECK:  [[VAR2:%.+]] = VPU.Reshape([[VAR1]]) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    // CHECK:  return [[VAR2]] : tensor<1x512x1x1xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvToNCE([[ARG0:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
func.func @ConvToNCE(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16>
            -> tensor<1x64x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x64x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:      {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [64, 32, 1, 1], strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x64x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[OUT]] : tensor<1x64x16x16xf16, {order = #NHWC}>
}
