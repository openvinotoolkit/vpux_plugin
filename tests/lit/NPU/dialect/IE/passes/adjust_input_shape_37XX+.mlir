//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-input-shape --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.5>

// CHECK-LABEL: @AdjustInputShapeForPermuteQuantizeDueToBigW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x245760xf16>
func.func @AdjustInputShapeForPermuteQuantizeDueToBigW(%arg0: tensor<1x1x1x245760xf16>) -> tensor<1x1x1x245760x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1x1x245760xf16> -> tensor<1x1x1x245760x!qElemType, {order = #NHWC}>
    return %0 : tensor<1x1x1x245760x!qElemType, {order = #NHWC}>

    // CHECK:       [[CAST:%.+]] = IE.ShapeCast {shape = [1, 1, 30, 8192]} inputs([[INPUT]] : tensor<1x1x1x245760xf16>) -> tensor<1x1x30x8192xf16>
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CAST]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1x30x8192xf16> -> tensor<1x1x30x8192x!qElemType, {order = #NHWC}>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 245760]} inputs([[PERMUTEQUANTIZE]] : tensor<1x1x30x8192x!qElemType, {order = #NHWC}>) -> tensor<1x1x1x245760x!qElemType, {order = #NHWC}>
    // CHECK:       return [[CAST_OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustInputShapeForPermuteQuantizeDueToBigH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x245760x16xf16>
func.func @AdjustInputShapeForPermuteQuantizeDueToBigH(%arg0: tensor<1x8x245760x16xf16>) -> tensor<1x8x245760x16xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x245760x16xf16> -> tensor<1x8x245760x16xf16, {order = #NHWC}>
    return %0 : tensor<1x8x245760x16xf16, {order = #NHWC}>

    // CHECK:       [[CAST:%.+]] = IE.ShapeCast {shape = [1, 8, 8192, 480]} inputs([[INPUT]] : tensor<1x8x245760x16xf16>) -> tensor<1x8x8192x480xf16>
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CAST]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8192x480xf16> -> tensor<1x8x8192x480xf16, {order = #NHWC}>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 8, 245760, 16]} inputs([[PERMUTEQUANTIZE]] : tensor<1x8x8192x480xf16, {order = #NHWC}>) -> tensor<1x8x245760x16xf16, {order = #NHWC}>
    // CHECK:       return [[CAST_OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeDueToNonePad
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x245760x16xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeDueToNonePad(%arg0: tensor<1x8x245760x16xf16>) -> tensor<1x16x245760x16xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x245760x16xf16> -> tensor<1x16x245760x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x245760x16xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x245760x16xf16> -> tensor<1x16x245760x16xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeDueToBigW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x245760x2048xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeDueToBigW(%arg0: tensor<1x8x245760x2048xf16>) -> tensor<1x8x245760x2048xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x245760x2048xf16> -> tensor<1x8x245760x2048xf16, {order = #NHWC}>
    return %0 : tensor<1x8x245760x2048xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x245760x2048xf16> -> tensor<1x8x245760x2048xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustInputShapeForPermuteQuantizeDueToBigH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x8x2048x245760xf16>
func.func @NotAdjustInputShapeForPermuteQuantizeDueToBigH(%arg0: tensor<1x8x2048x245760xf16>) -> tensor<1x8x2048x245760xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x2048x245760xf16> -> tensor<1x8x2048x245760xf16, {order = #NHWC}>
    return %0 : tensor<1x8x2048x245760xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x2048x245760xf16> -> tensor<1x8x2048x245760xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTEQUANTIZE]]
}
