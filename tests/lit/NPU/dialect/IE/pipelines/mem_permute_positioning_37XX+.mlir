//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --mempermute-positioning %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MemPemutePositioning
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x16x32x64xf16>)
func.func @MemPemutePositioning(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x64x32xi8> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.0, min = 0.0}>
        } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16>

    %3 = IE.Convert(%2) {dstElemType = i8} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xi8>

    %4 = IE.Transpose(%3) {order_value = #NCWH} : tensor<1x16x32x64xi8> -> tensor<1x16x64x32xi8>

    return %4 : tensor<1x16x64x32xi8>

    // CHECK:   [[MEMPERMUTE0:%.*]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   [[POOL:%.*]] = IE.MaxPool([[MEMPERMUTE0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   [[CONVERT:%.*]] = IE.Convert([[POOL]]) {dstElemType = i8} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xi8, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE1:%.*]] = IE.MemPermute([[CONVERT]]) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x16x32x64xi8, {order = #NHWC}> -> tensor<1x16x64x32xi8>
    // CHECK:   return [[MEMPERMUTE1]] : tensor<1x16x64x32xi8>
}
