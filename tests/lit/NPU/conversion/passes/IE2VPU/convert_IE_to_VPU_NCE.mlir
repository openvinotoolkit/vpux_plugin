//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE
func.func @ConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEWithBiasBroadCast
func.func @ConvToNCEWithBiasBroadCast(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithReluRewriter
func.func @ConvWithReluRewriter(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCE
func.func @EltwiseAddToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddSameInputsToNCE
func.func @EltwiseAddSameInputsToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg0)
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}
