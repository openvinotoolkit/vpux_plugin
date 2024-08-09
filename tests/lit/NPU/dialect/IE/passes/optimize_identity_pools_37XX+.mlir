//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-identity-pools %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseIdentityAvgPoolWithQuantizedAdd([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @FuseIdentityAvgPoolWithQuantizedAdd (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK: IE.Add([[ARG0]], [[ARG0]])
    //CHECK-SAME: post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>

    //CHECK-NOT: IE.AvgPool
}


// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @DoNotFuseIdentityAvgPoolWithQuantizedAdd([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @DoNotFuseIdentityAvgPoolWithQuantizedAdd (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    %2 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    %3 = IE.Add(%1, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK: IE.AvgPool([[ARG0]])
    //CHECK-SAME: post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>

    //CHECK: IE.Add
    //CHECK: IE.Add
    //CHECK: IE.Add

}
