//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-identity-pools %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @RemoveIdentityAvgPool
func.func @RemoveIdentityAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK-NOT:   IE.AvgPool
}

// CHECK-LABEL: @RemoveIdentityMaxPool
func.func @RemoveIdentityMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK-NOT:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPool
func.func @NotRemoveIdentityAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x9x12xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x9x12xf16>

    return %ave_pool : tensor<1x64x9x12xf16>
    // CHECK:   IE.AvgPool
}

// CHECK-LABEL: @NotRemoveIdentityMaxPool
func.func @NotRemoveIdentityMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x9x12xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x9x12xf16>

    return %max_pool : tensor<1x64x9x12xf16>
    // CHECK:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPoolPostOp
func.func @NotRemoveIdentityAvgPoolPostOp(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>,
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.AvgPool
}

// CHECK-LABEL: @NotRemoveIdentityMaxPoolPostOp
func.func @NotRemoveIdentityMaxPoolPostOp(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>,
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.MaxPool
}

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>
// CHECK-LABEL: @NotRemoveIdentityMaxPoolDiffType
func.func @NotRemoveIdentityMaxPoolDiffType(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13x!qElemType>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13x!qElemType>

    return %max_pool : tensor<1x64x10x13x!qElemType>
    // CHECK:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPoolDiffType
func.func @NotRemoveIdentityAvgPoolDiffType(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13x!qElemType>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13x!qElemType>

    return %ave_pool : tensor<1x64x10x13x!qElemType>
    // CHECK:   IE.AvgPool
}

// CHECK-LABEL: @FuseConvIdentityAvgPoolWithPostOp
func.func @FuseConvIdentityAvgPoolWithPostOp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
              {
                  dilations = [1, 1],
                  pads_begin = [0, 0],
                  pads_end = [0, 0],
                  strides = [1, 1]
              } :
              tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
                  {
                      exclude_pads,
                      kernel_size = [1, 1],
                      pads_begin = [0, 0],
                      pads_end = [0, 0],
                      post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}>,
                      rounding_type = #IE.rounding_type<FLOOR>,
                      strides = [1, 1]
                  } :
                  tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>
    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}>
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @NotFuseConvIdentityAvgPoolAsExistingPostOp
func.func @NotFuseConvIdentityAvgPoolAsExistingPostOp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
              {
                  dilations = [1, 1],
                  pads_begin = [0, 0],
                  pads_end = [0, 0],
                  post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
                  strides = [1, 1]
              } :
              tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
                  {
                      exclude_pads,
                      kernel_size = [1, 1],
                      pads_begin = [0, 0],
                      pads_end = [0, 0],
                      post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}>,
                      rounding_type = #IE.rounding_type<FLOOR>,
                      strides = [1, 1]
                  } :
                  tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>
    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>
    // CHECK-SAME:     strides = [1, 1]
    // CHECK:       IE.AvgPool
    // CHECK-SAME:     kernel_size = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}>
    // CHECK-SAME:     rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:     strides = [1, 1]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @NotFuseAddIdentityAvgPoolAsUnsupportedPostOp([[ARG0:%.+]]: tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
func.func @NotFuseAddIdentityAvgPoolAsUnsupportedPostOp(%arg0 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
    %add = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    %ave_pool = IE.AvgPool(%add) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}  :
                  tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x320x320xf16, {order = #NHWC}>
    // CHECK:       IE.Add([[ARG0]], [[ARG0]])
    // CHECK:       IE.AvgPool 
    //CHECK-SAME:       post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>
    //CHECK-SAME:       tensor<1x16x320x320xf16, {order = #NHWC}>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @FuseIdentityAvgPool (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NHWC}>) {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK: IE.AvgPool(%arg0)
    //CHECK-SAME: post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>
    //CHECK-SAME: tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK-NOT: IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
func.func @FuseIdentityAvgPoolDiffOrder (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NWCH}>) {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NWCH}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NWCH}>

    //CHECK: IE.AvgPool(%arg0)
    //CHECK-SAME: post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.0999755859375 : f64}>
    //CHECK-SAME: tensor<1x16x320x320x!qElemType, {order = #NWCH}>

    //CHECK-NOT: IE.AvgPool
}
