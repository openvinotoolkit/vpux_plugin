//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-reshape-mvn %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseReorderWithMVN1
module @FuseReorderWithMVN1 {

// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
    %2 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16>
    %3 = IE.Reshape(%2) {shape_value = [1, 32, 524288, 1]} : tensor<1x256x256x256xf16> -> tensor<1x32x524288x1xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x32x524288x1xf16> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %5 = IE.MVN(%4) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %6 = IE.Reorder(%5) {dstOrder = #NCHW} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16>
    %7 = IE.Reshape(%6) {shape_value = [1, 256, 256, 256]} : tensor<1x32x524288x1xf16> -> tensor<1x256x256x256xf16>
    %8 = IE.Reorder(%7) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    return %8 : tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK: [[VAR0:%.*]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 5.000000e-01 : f64, internal_reshape = [1, 32, 524288, 1], normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: return [[VAR0]] : tensor<1x256x256x256xf16, {order = #NHWC}>
  }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveGroupConvPostFuseReorderWithMVN
module @MoveGroupConvPostFuseReorderWithMVN {

// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x1xf16, {order = #NHWC}> = dense<9.000000e+00> : tensor<32x1x1x1xf16, {order = #NHWC}>
    %2 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16>
    %3 = IE.Reshape(%2) {shape_value = [1, 32, 524288, 1]} : tensor<1x256x256x256xf16> -> tensor<1x32x524288x1xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x32x524288x1xf16> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %5 = IE.MVN(%4) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16, {order = #NHWC}>
    %6 = IE.Reorder(%5) {dstOrder = #NCHW} : tensor<1x32x524288x1xf16, {order = #NHWC}> -> tensor<1x32x524288x1xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 524288]} : tensor<1x32x524288x1xf16> -> tensor<1x32x1x524288xf16>
    %8 = IE.Reorder(%7) {dstOrder = #NHWC} : tensor<1x32x1x524288xf16> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %9 = IE.GroupConvolution(%8, %cst_0) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x1x524288xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16, {order = #NHWC}>
    %10 = IE.Reorder(%9) {dstOrder = #NCHW} : tensor<1x32x1x524288xf16, {order = #NHWC}> -> tensor<1x32x1x524288xf16>
    %11 = IE.Reshape(%10) {shape_value = [1, 256, 256, 256]} : tensor<1x32x1x524288xf16> -> tensor<1x256x256x256xf16>
    %12 = IE.Reorder(%11) {dstOrder = #NHWC} : tensor<1x256x256x256xf16> -> tensor<1x256x256x256xf16, {order = #NHWC}>

    return %12 : tensor<1x256x256x256xf16, {order = #NHWC}>

    // CHECK: [[CST:%.*]] = const.Declare tensor<256x1x1x1xf16, {order = #NHWC}> = dense<9.000000e+00> : tensor<256x1x1x1xf16, {order = #NHWC}>
    // CHECK: [[VAR0:%.*]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 5.000000e-01 : f64, internal_reshape = [1, 32, 524288, 1], normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: [[VAR1:%.*]] = IE.GroupConvolution([[VAR0]], [[CST]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<256x1x1x1xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
    // CHECK: return [[VAR1]] : tensor<1x256x256x256xf16, {order = #NHWC}>
  }

}
