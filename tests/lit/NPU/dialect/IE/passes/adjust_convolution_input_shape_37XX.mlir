//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-input-shape --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeInputsForParallelAddOp
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<2x1x1x2048xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x2048x1x1xf16, {order = #NHWC}>, [[INPUT_2:%.+]]: tensor<1x2048x1x1xf16, {order = #NHWC}>)
func.func @ReshapeInputsForParallelAddOp(%arg0: tensor<2x1x1x2048xf16, {order = #NHWC}>, %arg1: tensor<1x2048x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x2048x1x1xf16, {order = #NHWC}>) -> (tensor<1x1x1x2048xf16, {order = #NHWC}>, tensor<1x1x1x2048xf16, {order = #NHWC}>) {
    %1 = IE.ShapeCast {shape = [2, 2048, 1, 1]} inputs(%arg0 : tensor<2x1x1x2048xf16, {order = #NHWC}>) -> tensor<2x2048x1x1xf16, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 2048, 1, 1] : tensor<2x2048x1x1xf16, {order = #NHWC}> to tensor<1x2048x1x1xf16, {order = #NHWC}>
    %3 = IE.Add(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2048x1x1xf16, {order = #NHWC}>, tensor<1x2048x1x1xf16, {order = #NHWC}> -> tensor<1x2048x1x1xf16, {order = #NHWC}>
    %4 = IE.Slice %1 [1, 0, 0, 0] [1, 2048, 1, 1] : tensor<2x2048x1x1xf16, {order = #NHWC}> to tensor<1x2048x1x1xf16, {order = #NHWC}>
    %5 = IE.Add(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2048x1x1xf16, {order = #NHWC}>, tensor<1x2048x1x1xf16, {order = #NHWC}> -> tensor<1x2048x1x1xf16, {order = #NHWC}>
    %6 = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs(%3 : tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x2048xf16, {order = #NHWC}>
    %7 = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs(%5 : tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x2048xf16, {order = #NHWC}>

    return %6, %7 : tensor<1x1x1x2048xf16, {order = #NHWC}>, tensor<1x1x1x2048xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [2, 2048, 1, 1]} inputs([[INPUT_0]] : tensor<2x1x1x2048xf16, {order = #NHWC}>) -> tensor<2x2048x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[SHAPECAST_0]] [0, 0, 0, 0] [1, 2048, 1, 1] : tensor<2x2048x1x1xf16, {order = #NHWC}> to tensor<1x2048x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 128, 4, 4]} inputs([[SLICE_0]] : tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x128x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 128, 4, 4]} inputs([[INPUT_1]] : tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x128x4x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[SHAPECAST_1]], [[SHAPECAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x128x4x4xf16, {order = #NHWC}>, tensor<1x128x4x4xf16, {order = #NHWC}> -> tensor<1x128x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[SHAPECAST_0]] [1, 0, 0, 0] [1, 2048, 1, 1] : tensor<2x2048x1x1xf16, {order = #NHWC}> to tensor<1x2048x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 128, 4, 4]} inputs([[SLICE_1]] : tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x128x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_4:%.+]] = IE.ShapeCast {shape = [1, 128, 4, 4]} inputs([[INPUT_2]] : tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x128x4x4xf16, {order = #NHWC}>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[SHAPECAST_3]], [[SHAPECAST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x128x4x4xf16, {order = #NHWC}>, tensor<1x128x4x4xf16, {order = #NHWC}> -> tensor<1x128x4x4xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT_0:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs([[ADD_0]] : tensor<1x128x4x4xf16, {order = #NHWC}>) -> tensor<1x1x1x2048xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT_1:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs([[ADD_1]] : tensor<1x128x4x4xf16, {order = #NHWC}>) -> tensor<1x1x1x2048xf16, {order = #NHWC}>

    // CHECK:       return [[SHAPECAST_OUT_0]], [[SHAPECAST_OUT_1]] : tensor<1x1x1x2048xf16, {order = #NHWC}>, tensor<1x1x1x2048xf16, {order = #NHWC}>
}
