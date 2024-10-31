//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-shape-cast %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAbs
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithAbs(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.Abs(%0) : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[ABS:%.*]] = IE.Abs([[SHAPE_CAST_0]]) : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[ABS]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithGelu
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithGelu(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.Gelu(%0) : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GELU:%.*]] = IE.Gelu([[SHAPE_CAST_0]]) : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[GELU]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithSwish
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithSwish(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.Swish(%0) {beta_value = 1.0} : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SWISH:%.*]] = IE.Swish([[SHAPE_CAST_0]]) {beta_value = 1.000000e+00 : f64} : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[SWISH]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithHSwish
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithHSwish(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.HSwish(%0) : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[HSWISH:%.*]] = IE.HSwish([[SHAPE_CAST_0]]) : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[HSWISH]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithSigmoid
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithSigmoid(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.Sigmoid(%0) : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SIGMOID:%.*]] = IE.Sigmoid([[SHAPE_CAST_0]]) : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[SIGMOID]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithTanh
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithTanh(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[TANH:%.*]] = IE.Tanh([[SHAPE_CAST_0]]) : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[TANH]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAvgPool
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithAvgPool(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[POOL:%.*]] = IE.AvgPool([[SHAPE_CAST_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[POOL]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithMaxPool
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x24x112x224xf16, {order = #NHWC}>
func.func @SwapWithMaxPool(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 12, 224, 224]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x12x224x224xf16, {order = #NHWC}> -> tensor<1x12x224x224xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x12x224x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[POOL:%.*]] = IE.MaxPool([[SHAPE_CAST_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.*]] = IE.GroupConvolution([[POOL]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.*]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.026685049019607842>

// CHECK-LABEL: @SwapWithSubtract
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x7680x8x4xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @SwapWithSubtract(%arg0: tensor<1x7680x8x4xf16, {order = #NHWC}>,
                            %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
                            -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs(%arg0 : tensor<1x7680x8x4xf16, {order = #NHWC}>) -> tensor<1x245760x1x1xf16, {order = #NHWC}>
    %1 = IE.Subtract(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x245760x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x245760x1x1xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs(%1 : tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x7680x8x4xf16, {order = #NHWC}> -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs(%3 : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    return %4 : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>

    // CHECK:               [[SUBTRACT:%.+]] = IE.Subtract([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    // CHECK:               [[ADD:%.+]] = IE.Add([[SUBTRACT]], [[SUBTRACT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x7680x8x4xf16, {order = #NHWC}> -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    // CHECK:               [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs([[ADD]] : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>) -> tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
    // CHECK:               return [[SHAPECAST]] : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAdd
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x24x112x224xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @SwapWithAdd(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>,
                       %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
                       -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 48, 112, 112]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x112x112xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x48x112x112xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT0]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[ADD:%.+]] = IE.Add([[SHAPE_CAST_0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[ADD]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0078507965686274508>

// CHECK-LABEL: @NotSwapWithAdd
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x245760x1x1x!qElemType, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @NotSwapWithAdd(%arg0: tensor<1x245760x1x1x!qElemType, {order = #NHWC}>,
                            %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
                            -> tensor<1x7680x8x4xf16, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs(%arg0 : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>, tensor<1x7680x8x4x!qElemType, {order = #NHWC}> -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 245760, 1, 1]} inputs(%1 : tensor<1x7680x8x4xf16, {order = #NHWC}>) -> tensor<1x245760x1x1xf16, {order = #NHWC}>
    %3 = IE.Subtract(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x245760x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x245760x1x1xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs(%3 : tensor<1x245760x1x1xf16, {order = #NHWC}>) -> tensor<1x7680x8x4xf16, {order = #NHWC}>

    return %4 : tensor<1x7680x8x4xf16, {order = #NHWC}>

    // CHECK:               [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 7680, 8, 4]} inputs([[INPUT0]] : tensor<1x245760x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x7680x8x4x!qElemType, {order = #NHWC}>
    // CHECK:               [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[SHAPE_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x7680x8x4x!qElemType, {order = #NHWC}>, tensor<1x7680x8x4x!qElemType, {order = #NHWC}> -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    // CHECK:               [[SUBTRACT:%.+]] = IE.Subtract([[ADD]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x7680x8x4xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x7680x8x4xf16, {order = #NHWC}>
    // CHECK:               return [[SUBTRACT]] : tensor<1x7680x8x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithMultiply
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x24x112x224xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @SwapWithMultiply(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>,
                            %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
                            -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 48, 112, 112]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x112x112xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x48x112x112xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT0]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[MULTIPLY:%.+]] = IE.Multiply([[SHAPE_CAST_0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[MULTIPLY]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAnd
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x24x112x224xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @SwapWithAnd(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>,
                       %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
                       -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 48, 112, 112]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %1 = IE.And(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x112x112xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x48x112x112xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT0]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[AND:%.+]] = IE.And([[SHAPE_CAST_0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[AND]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAndForDifferentOperandsOrder
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x24x112x224xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @SwapWithAndForDifferentOperandsOrder(%arg0: tensor<1x24x112x224xf16, {order = #NHWC}>,
                                                %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
                                                -> tensor<1x192x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    %0 = IE.ShapeCast {shape = [1, 48, 112, 112]} inputs(%arg0 : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %1 = IE.And(%arg1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x48x112x112xf16, {order = #NHWC}> -> tensor<1x48x112x112xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs(%1 : tensor<1x48x112x112xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %3 = IE.GroupConvolution(%2, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs(%3 : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>

    return %4 : tensor<1x192x56x56xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
    // CHECK:               [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 196, 192]} inputs([[INPUT0]] : tensor<1x24x112x224xf16, {order = #NHWC}>) -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[AND:%.+]] = IE.And([[SHAPE_CAST_0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[AND]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x196x192xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x196x192xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 192, 56, 56]} inputs([[GROUP_CONVOLUTION]] : tensor<1x16x196x192xf16, {order = #NHWC}>) -> tensor<1x192x56x56xf16, {order = #NHWC}>
    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x192x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithMvn
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x10x8xf16, {order = #NHWC}>
func.func @SwapWithMvn(%arg0: tensor<1x16x10x8xf16, {order = #NHWC}>)
                       -> tensor<1x1x1280x1xf16, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 1, 1, 1280]} inputs(%arg0 : tensor<1x16x10x8xf16, {order = #NHWC}>) -> tensor<1x1x1x1280xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x1x1280xf16, {order = #NHWC}> -> tensor<1x1x1x1280xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 1, 1280, 1]} inputs(%1 : tensor<1x1x1x1280xf16, {order = #NHWC}>) -> tensor<1x1x1280x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1x1280x1xf16, {order = #NHWC}>

    // CHECK:               [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 1, 1280, 1]} inputs([[INPUT]] : tensor<1x16x10x8xf16, {order = #NHWC}>) -> tensor<1x1x1280x1xf16, {order = #NHWC}>
    // CHECK:               [[MVN:%.+]] = IE.MVN([[SHAPE_CAST]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x1280x1xf16, {order = #NHWC}> -> tensor<1x1x1280x1xf16, {order = #NHWC}>

    // CHECK:               return [[MVN]] : tensor<1x1x1280x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithMvnWhenChannelChanged
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x10x8xf16, {order = #NHWC}>
func.func @SwapWithMvnWhenChannelChanged(%arg0: tensor<1x16x10x8xf16, {order = #NHWC}>)
                                         -> tensor<1x1280x1x1xf16, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 1, 1, 1280]} inputs(%arg0 : tensor<1x16x10x8xf16, {order = #NHWC}>) -> tensor<1x1x1x1280xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x1x1280xf16, {order = #NHWC}> -> tensor<1x1x1x1280xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 1280, 1, 1]} inputs(%1 : tensor<1x1x1x1280xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK:               [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 1280, 1, 1]} inputs([[INPUT]] : tensor<1x16x10x8xf16, {order = #NHWC}>) -> tensor<1x1280x1x1xf16, {order = #NHWC}>
    // CHECK:               [[MVN:%.+]] = IE.MVN([[SHAPE_CAST]]) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1280x1x1xf16, {order = #NHWC}> -> tensor<1x1280x1x1xf16, {order = #NHWC}>

    // CHECK:               return [[MVN]] : tensor<1x1280x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapWithMvnWhenChannelChanged
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x20x8xf16, {order = #NHWC}>
func.func @NotSwapWithMvnWhenChannelChanged(%arg0: tensor<1x16x20x8xf16, {order = #NHWC}>)
                                            -> tensor<1x2560x1x1xf16, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 2, 1, 1280]} inputs(%arg0 : tensor<1x16x20x8xf16, {order = #NHWC}>) -> tensor<1x2x1x1280xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x2x1x1280xf16, {order = #NHWC}> -> tensor<1x2x1x1280xf16, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 2560, 1, 1]} inputs(%1 : tensor<1x2x1x1280xf16, {order = #NHWC}>) -> tensor<1x2560x1x1xf16, {order = #NHWC}>

    return %2 : tensor<1x2560x1x1xf16, {order = #NHWC}>

    // CHECK:               [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 2, 1, 1280]} inputs([[INPUT]] : tensor<1x16x20x8xf16, {order = #NHWC}>) -> tensor<1x2x1x1280xf16, {order = #NHWC}>
    // CHECK:               [[MVN:%.+]] = IE.MVN([[SHAPE_CAST_0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x2x1x1280xf16, {order = #NHWC}> -> tensor<1x2x1x1280xf16, {order = #NHWC}>
    // CHECK:               [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 2560, 1, 1]} inputs([[MVN]] : tensor<1x2x1x1280xf16, {order = #NHWC}>) -> tensor<1x2560x1x1xf16, {order = #NHWC}>

    // CHECK:               return [[SHAPE_CAST_1]] : tensor<1x2560x1x1xf16, {order = #NHWC}>
}
