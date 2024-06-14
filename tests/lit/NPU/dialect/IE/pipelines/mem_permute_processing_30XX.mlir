//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --mempermute-processing %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MemPermuteProcessingWithoutFusing
func.func @MemPermuteProcessingWithoutFusing(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x64x32xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x16x32x64xf16> -> tensor<1x16x64x32xf16>

    return %2 : tensor<1x16x64x32xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute([[CONV]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x64x32xf16>

    // CHECK:   return [[PERMUTE]] : tensor<1x16x64x32xf16>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteProcessingConvertPass
func.func @MemPermuteProcessingConvertPass(%arg0: tensor<1x2x3x4xf32>,
                            %arg1: tensor<1x2x3x4xf32, {order = #NHWC}>,
                            %arg2: tensor<1x2x3x4xf32>) ->
                        (tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>) {
    %0 = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>

    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>

    %2 = IE.Reorder(%arg2) {dstOrder = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    return %0, %1, %2 : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>

    // CHECK-NOT: IE.Transpose
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Reorder
    // CHECK:     %[[VAL0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>
    // CHECK:     %[[VAL1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    // CHECK:     %[[VAL2:.*]] = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    // CHECK:     return %[[VAL0]], %[[VAL1:.*]], %[[VAL2:.*]] : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @MemPermuteProcessingWithNDReorder
func.func @MemPermuteProcessingWithNDReorder(%arg0: tensor<6x10x10x4x1xf16, {order = #map}>) -> tensor<6x10x10x4x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<6x10x10x4x1xf16, {order = #map}> -> tensor<6x10x10x4x1xf16>
    return %0 : tensor<6x10x10x4x1xf16>

    // CHECK-NOT: IE.Reorder
    // CHECK: [[VAL0:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<6x10x10x4x1xf16, {order = #map}> -> tensor<1x10x10x4x6xf16>
    // CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 100, 4, 6]} : tensor<1x10x10x4x6xf16> -> tensor<1x100x4x6xf16>
    // CHECK: [[VAL2:%.*]] = IE.MemPermute([[VAL1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x100x4x6xf16> -> tensor<1x6x100x4xf16>
    // CHECK: [[VAL3:%.*]] = IE.AffineReshape([[VAL2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [6, 10, 10, 4, 1]} : tensor<1x6x100x4xf16> -> tensor<6x10x10x4x1xf16>

    // CHECK return [[VAL3]] : tensor<6x10x10x4x1xf16>
}
