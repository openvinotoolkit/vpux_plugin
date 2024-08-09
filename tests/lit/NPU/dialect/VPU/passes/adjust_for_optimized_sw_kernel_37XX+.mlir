//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-for-optimized-sw-kernel %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForSoftmaxAxisZeroOptNCHW
func.func @AdjustForSoftmaxAxisZeroOptNCHW(%arg0: tensor<1x64x16x1xf16>) -> tensor<1x64x16x1xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x1xf16> -> tensor<1x64x16x1xf16>
    return %0 : tensor<1x64x16x1xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 64, 1, 16]} inputs(%arg0 : tensor<1x64x16x1xf16>) -> tensor<1x64x1x16xf16>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x64x1x16xf16> -> tensor<1x64x1x16xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 64, 16, 1]} inputs([[SOFTMAX]] : tensor<1x64x1x16xf16>) -> tensor<1x64x16x1xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxAxisZeroOptNCHW
func.func @NotAdjustForSoftmaxAxisZeroOptNCHW(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    return %0 : tensor<1x64x16x16xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @AdjustForSoftmaxAxisZeroOptNHWC
func.func @AdjustForSoftmaxAxisZeroOptNHWC(%arg0: tensor<1x1x16x64xf16, {order = #NHWC}>) -> tensor<1x1x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x1x16x64xf16, {order = #NHWC}> -> tensor<1x1x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x1x16x64xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 64, 16, 1]} inputs(%arg0 : tensor<1x1x16x64xf16, {order = #NHWC}>) -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x64x16x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 16, 64]} inputs([[SOFTMAX]] : tensor<1x64x16x1xf16, {order = #NHWC}>) -> tensor<1x1x16x64xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxAxisZeroOptNHWC
func.func @NotAdjustForSoftmaxAxisZeroOptNHWC(%arg0: tensor<1x2x16x64xf16, {order = #NHWC}>) -> tensor<1x2x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNCHW
func.func @NotAdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    return %0 : tensor<1x2x16x16xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNHWC
func.func @NotAdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x2x16x64xf16, {order = #NHWC}>) -> tensor<1x2x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// -----

// CHECK-LABEL:   @AdjustForGeluMultiShaveOptNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x2x16x32xf16>)
func.func @AdjustForGeluMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs([[INPUT]] : tensor<1x2x16x32xf16>) -> tensor<1x1024x1x1xf16>
    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[SHAPECAST_IN]]) : tensor<1x1024x1x1xf16> -> tensor<1x1024x1x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[GELU]] : tensor<1x1024x1x1xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForGeluMultiShaveOptNHWC
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x32x2x16xf16, {order = #NHWC}>)
func.func @AdjustForGeluMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[SHAPECAST_IN]]) : tensor<1x1x1024x1xf16, {order = #NHWC}> -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[GELU]] : tensor<1x1x1024x1xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @AdjustForGeluMultiShaveOptForBatchSize
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1024x2560x1x1xf16>)
func.func @AdjustForGeluMultiShaveOptForBatchSize(%arg0: tensor<1024x2560x1x1xf16>) -> tensor<1024x2560x1x1xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1024x2560x1x1xf16> -> tensor<1024x2560x1x1xf16>
    return %0 : tensor<1024x2560x1x1xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 2621440, 1, 1]} inputs([[INPUT]] : tensor<1024x2560x1x1xf16>) -> tensor<1x2621440x1x1xf16>
    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[SHAPECAST_IN]]) : tensor<1x2621440x1x1xf16> -> tensor<1x2621440x1x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1024, 2560, 1, 1]} inputs([[GELU]] : tensor<1x2621440x1x1xf16>) -> tensor<1024x2560x1x1xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @NotAdjustForGeluMultiShaveOptNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16>)
func.func @NotAdjustForGeluMultiShaveOptNCHW(%arg0: tensor<1x1x1024x1xf16>) -> tensor<1x1x1024x1xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x1x1024x1xf16> -> tensor<1x1x1024x1xf16>
    return %0 : tensor<1x1x1024x1xf16>

    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[INPUT]]) : tensor<1x1x1024x1xf16> -> tensor<1x1x1024x1xf16>
    // CHECK:        return [[GELU]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForGeluMultiShaveOptNHWC
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForGeluMultiShaveOptNHWC(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    // CHECK:        [[GELU:%.*]] = VPU.Gelu([[INPUT]]) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    // CHECK:        return [[GELU]]
}

// -----

// CHECK-LABEL:   @AdjustForMultiplyMultiShaveOptNCHW
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x2x16x32xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x2x16x32xf16>)
func.func @AdjustForMultiplyMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>, %arg1: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x2x16x32xf16>, tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN_1:%.*]] = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs([[INPUT0]] : tensor<1x2x16x32xf16>) -> tensor<1x1024x1x1xf16>
    // CHECK:        [[SHAPECAST_IN_2:%.*]] = VPU.ShapeCast {shape = [1, 1024, 1, 1]} inputs([[INPUT1]] : tensor<1x2x16x32xf16>) -> tensor<1x1024x1x1xf16>
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[SHAPECAST_IN_1]], [[SHAPECAST_IN_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<1x1024x1x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[MULTIPLY]] : tensor<1x1024x1x1xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForMultiplyMultiShaveOptNHWC
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x32x2x16xf16, {order = #NHWC}>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x32x2x16xf16, {order = #NHWC}>)
func.func @AdjustForMultiplyMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>, %arg1: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x32x2x16xf16, {order = #NHWC}>, tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>

    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN_1:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT0]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_IN_2:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT1]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[SHAPECAST_IN_1]], [[SHAPECAST_IN_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1024x1xf16, {order = #NHWC}>, tensor<1x1x1024x1xf16, {order = #NHWC}> -> tensor<1x1x1024x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[MULTIPLY]] : tensor<1x1x1024x1xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @NotAdjustForMultiplyMultiShaveOptForBroadcast
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x3x384x2xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x3x1x2xf16>)
func.func @NotAdjustForMultiplyMultiShaveOptForBroadcast(%arg0: tensor<1x3x384x2xf16>, %arg1: tensor<1x3x1x2xf16>) -> tensor<1x3x384x2xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x3x384x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x384x2xf16>

    return %0 : tensor<1x3x384x2xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x384x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x384x2xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[MULTIPLY]]
}

// -----

// CHECK-LABEL:   @NotAdjustForMultiplyMultiShaveOptForConstInput
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x128x32xf16>)
func.func @NotAdjustForMultiplyMultiShaveOptForConstInput(%arg0: tensor<1x1x128x32xf16>) -> tensor<1x1x128x32xf16> {
    %cst = const.Declare tensor<1x1x128x32xf16> = dense<1.0> : tensor<1x128x32xf16>, [#const.Reshape<[1, 1, 128, 32]>]
    %0 = VPU.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x128x32xf16>, tensor<1x1x128x32xf16> -> tensor<1x1x128x32xf16>

    return %0 : tensor<1x1x128x32xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[CST:%.*]] = const.Declare tensor<1x1x128x32xf16> = dense<1.000000e+00> : tensor<1x128x32xf16>, [#const.Reshape<[1, 1, 128, 32]>]
    // CHECK:        [[MULTIPLY:%.*]] = VPU.Multiply([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x128x32xf16>, tensor<1x1x128x32xf16> -> tensor<1x1x128x32xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[MULTIPLY]]
}

// -----

// CHECK-LABEL:   @AdjustForMVNWithBatchAndAcrossChannels
// CHECK-SAME:    [[INPUT:%.*]]: tensor<8x16x2x8xf16>
func.func @AdjustForMVNWithBatchAndAcrossChannels(%arg0: tensor<8x16x2x8xf16>) -> tensor<8x16x2x8xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = true, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<8x16x2x8xf16> -> tensor<8x16x2x8xf16>

    return %0 : tensor<8x16x2x8xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 8, 256, 1]} inputs([[INPUT]] : tensor<8x16x2x8xf16>) -> tensor<1x8x256x1xf16>
    // CHECK:        [[MVN:%.*]] = VPU.MVN([[SHAPECAST_IN]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x8x256x1xf16> -> tensor<1x8x256x1xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [8, 16, 2, 8]} inputs([[MVN]] : tensor<1x8x256x1xf16>) -> tensor<8x16x2x8xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// CHECK-LABEL:   @AdjustForMVNWithBatchAndNotAcrossChannels
// CHECK-SAME:    [[INPUT:%.*]]: tensor<8x16x2x8xf16>
func.func @AdjustForMVNWithBatchAndNotAcrossChannels(%arg0: tensor<8x16x2x8xf16>) -> tensor<8x16x2x8xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<8x16x2x8xf16> -> tensor<8x16x2x8xf16>

    return %0 : tensor<8x16x2x8xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 128, 2, 8]} inputs([[INPUT]] : tensor<8x16x2x8xf16>) -> tensor<1x128x2x8xf16>
    // CHECK:        [[MVN:%.*]] = VPU.MVN([[SHAPECAST_IN]]) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x128x2x8xf16> -> tensor<1x128x2x8xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [8, 16, 2, 8]} inputs([[MVN]] : tensor<1x128x2x8xf16>) -> tensor<8x16x2x8xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}
