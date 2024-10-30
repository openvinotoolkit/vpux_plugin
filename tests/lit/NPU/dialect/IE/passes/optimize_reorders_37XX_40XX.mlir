//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-reorders %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNCHWLayoutForEfficiency
module @MVNInNCHWLayoutForEfficiency {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x64x64x192xf16> {
    %cst = const.Declare tensor<64x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<64x4x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16>
    return %4 : tensor<1x64x64x192xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK:      [[REORDER:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
    // CHECK:      [[MVN:%.*]] = IE.MVN([[REORDER]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNHWCLayoutForEfficiency
module @MVNInNHWCLayoutForEfficiency {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x384x16x48xf16> {
    %cst = const.Declare tensor<384x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [8, 8]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<384x4x3x3xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x384x16x48xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x384x16x48xf16> -> tensor<1x384x16x48xf16>
    return %4 : tensor<1x384x16x48xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK-NOT:  IE.Reorder
    // CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x384x16x48xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNHWCLayoutForAcrossChannels
module @MVNInNHWCLayoutForAcrossChannels {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x64x64x192xf16> {
    %cst = const.Declare tensor<64x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<64x4x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
    %4 = IE.MVN(%3) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16>
    return %4 : tensor<1x64x64x192xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK-NOT:  IE.Reorder
    // CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]]) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNHWCLayoutForWNotAligned
module @MVNInNHWCLayoutForWNotAligned {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x64x64x191xf16> {
    %cst = const.Declare tensor<64x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<64x4x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x191xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x191xf16, {order = #NHWC}> -> tensor<1x64x64x191xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x191xf16> -> tensor<1x64x64x191xf16>
    return %4 : tensor<1x64x64x191xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK-NOT:  IE.Reorder
    // CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x191xf16, {order = #NHWC}> -> tensor<1x64x64x191xf16, {order = #NHWC}>
}
}
