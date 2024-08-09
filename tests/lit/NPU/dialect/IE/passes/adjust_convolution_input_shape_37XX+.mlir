//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-input-shape --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConv
func.func @ShapeCastToAlignExpandedDWConv(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640xf16, {order  = #NHWC}> -> tensor<1x16x640x640xf16, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.*]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs(%arg0 : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.*]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.*]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0039085829959196201>

// CHECK-LABEL: @ShapeCastToAlignExpandedDWConvQuant
func.func @ShapeCastToAlignExpandedDWConvQuant(%arg0: tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x640xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x640x640x!qElemType, {order  = #NHWC}> -> tensor<1x16x640x640x!qElemType, {order = #NHWC}>
    %conv = IE.GroupConvolution(%expand, %filter, %bias) {
        dilations = [1, 1], groups = 16, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]
    } : tensor<1x16x640x640x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>

    return %conv : tensor<1x16x320x640xf16, {order = #NHWC}>

    // CHECK-DAG:    [[FILTER:%.*]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:    [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>, #const.Reorder<#NHWC>]
    // CHECK:        [[IN_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs(%arg0 : tensor<1x3x640x640x!qElemType, {order = #NHWC}>) -> tensor<1x48x640x40x!qElemType, {order = #NHWC}>
    // CHECK:        [[GRP_CONV:%.*]] = IE.GroupConvolution([[IN_SHAPECAST]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x48x640x40x!qElemType, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x48x320x40xf16, {order = #NHWC}>
    // CHECK:        [[OUT_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 3, 320, 640]} inputs([[GRP_CONV]] : tensor<1x48x320x40xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK:        [[EXPAND:%.*]] = IE.Expand([[OUT_SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x16x320x640xf16, {order = #NHWC}>
    // CHECK:        return [[EXPAND]] : tensor<1x16x320x640xf16, {order = #NHWC}>
}
