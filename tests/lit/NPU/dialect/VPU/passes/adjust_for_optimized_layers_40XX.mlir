//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-for-optimized-layers %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForSoftmaxMultiShaveOptNCHW
func.func @AdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 16, 2, 32]} inputs(%arg0 : tensor<1x2x16x32xf16>) -> tensor<1x16x2x32xf16>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x16x2x32xf16> -> tensor<1x16x2x32xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[SOFTMAX]] : tensor<1x16x2x32xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @AdjustForSoftmaxMultiShaveOptNHWC
func.func @AdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 32, 16, 2]} inputs(%arg0 : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x16x2xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x32x16x2xf16, {order = #NHWC}> -> tensor<1x32x16x2xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[SOFTMAX]] : tensor<1x32x16x2xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNHWC
func.func @NotAdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x64x7x7xf16, {order = #NHWC}>) -> tensor<1x64x7x7xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x64x7x7xf16, {order = #NHWC}> -> tensor<1x64x7x7xf16, {order = #NHWC}>
    return %0 : tensor<1x64x7x7xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x64x7x7xf16, {order = #NHWC}> -> tensor<1x64x7x7xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @AdjustForSoftmaxMultiShaveOptNCHWwithBatch
func.func @AdjustForSoftmaxMultiShaveOptNCHWwithBatch(%arg0: tensor<2x8x16x16xf16>) -> tensor<2x8x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<2x8x16x16xf16> -> tensor<2x8x16x16xf16>
    return %0 : tensor<2x8x16x16xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 16, 16, 16]} inputs(%arg0 : tensor<2x8x16x16xf16>) -> tensor<1x16x16x16xf16>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [2, 8, 16, 16]} inputs([[SOFTMAX]] : tensor<1x16x16x16xf16>) -> tensor<2x8x16x16xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @AdjustForSoftmaxMultiShaveOptNHWCwithBatch
func.func @AdjustForSoftmaxMultiShaveOptNHWCwithBatch(%arg0: tensor<2x8x16x16xf16, {order = #NHWC}>) -> tensor<2x8x16x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<2x8x16x16xf16, {order = #NHWC}> -> tensor<2x8x16x16xf16, {order = #NHWC}>
    return %0 : tensor<2x8x16x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 8, 32, 16]} inputs(%arg0 : tensor<2x8x16x16xf16, {order = #NHWC}>) -> tensor<1x8x32x16xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x8x32x16xf16, {order = #NHWC}> -> tensor<1x8x32x16xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [2, 8, 16, 16]} inputs([[SOFTMAX]] : tensor<1x8x32x16xf16, {order = #NHWC}>) -> tensor<2x8x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForNCEPermuteDueToSmallNewHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x3x64xf16>
func.func @NotAdjustForNCEPermuteDueToSmallNewHeight(%arg0: tensor<1x32x3x64xf16>) -> tensor<1x32x3x64xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {opaque_ppe = #VPU.PPEStub<>, dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64} -> tensor<1x32x3x64xf16, {order = #NHWC}>

    return %0 : tensor<1x32x3x64xf16, {order = #NHWC}>

    // CHECK:        [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 32 : i64, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x32x3x64xf16, {order = #NHWC}>
    // CHECK:        return [[NCE_PERMUTE]]
}
