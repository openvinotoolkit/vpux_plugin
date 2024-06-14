//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0024088541666666668:128>

// CHECK-LABEL: @ConvLeakyRelu01ToNCE
func.func @ConvLeakyRelu01ToNCE(%arg0: tensor<1x16x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x16x16x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %weights) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.1}>
        } : tensor<1x16x16x16x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK-DAG: [[INSTRUCTION_LIST:%.*]] = const.Declare tensor<1x1x1x32xsi32>
    // CHECK-SAME: [-67092478, -57130494, -47168510, -37730814, -28229630, -18791934, -9354238, 83458, 66732034, 672258, -375806, 148994, 212994, 213506, -310270, 6, -309758, -1818622, -62111230, 23348226, -22264318, -15908862, -9616894, 9782274, 5588482, 409602, 0, 6, 6, 6, 6, 6]
    // CHECK: ppe = #VPU.PPETask<mode = <FLEXARB>, clamp_low = -128 : i64, clamp_high = 127 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [1], quant_shift = [0], quant_post_shift = 4 : i64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0024088541666666668:45>

// CHECK-LABEL: @ConvLeakyRelu02ToNCE
func.func @ConvLeakyRelu02ToNCE(%arg0: tensor<1x16x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x16x16x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %weights) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.2}>
        } : tensor<1x16x16x16x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK-DAG: [[INSTRUCTION_LIST:%.*]] = const.Declare tensor<1x1x1x32xsi32>
    // CHECK-SAME: [-117948414, -100646398, -83344382, -66042366, -48676862, -31374846, -14072830, 83458, 110247938, 147970, 148482, 148994, 212994, 213506, 214018, 6, 214530, -770046, 22823426, 19153922, 16008706, 12402690, 9257474, 5587970, 2442754, 409602, 0, 6, 6, 6, 6, 6]
    // CHECK:  ppe = #VPU.PPETask<mode = <FLEXARB>, clamp_low = -225 : i64, clamp_high = 210 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [1], quant_shift = [0], quant_post_shift = 2 : i64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0024088541666666668:128>

// CHECK-LABEL: @ConvLeakyRelu025ToNCE
func.func @ConvLeakyRelu025ToNCE(%arg0: tensor<1x16x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x16x16x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %weights) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.25}>
        } : tensor<1x16x16x16x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK-DAG: [[INSTRUCTION_LIST:%.*]] = const.Declare tensor<1x1x1x32xsi32>
    // CHECK-SAME: [-2147467262, 16898, 76039170, 152061442, 228147202, 304169474, 380191746, 456214018, 532824066, 1196546, 148482, 148994, 212994, 213506, 214018, 6, 214530, 278530, 2376194, 2376706, 2377218, 2441218, 2441730, 2442242, 2442754, 2506754, 0, 6, 6, 6, 6, 6]
    // CHECK: ppe = #VPU.PPETask<mode = <FLEXARB>, clamp_low = -4096 : i64, clamp_high = 1016 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [1], quant_shift = [0], quant_post_shift = 3 : i64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0024088541666666668:0>

// CHECK-LABEL: @ConvLeakyRelu020ToNCE
func.func @ConvLeakyRelu020ToNCE(%arg0: tensor<1x16x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x16x16x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %weights) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.2}>
        } : tensor<1x16x16x16x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // CHECK-DAG: [[INSTRUCTION_LIST:%.*]] = const.Declare tensor<1x1x1x32xsi32>
    // CHECK-SAME: [16386, 133710338, 133710850, 133711362, 133775362, 133775874, 133776386, 133776898, 133840898, -2473470, -2472958, -2472446, -2408446, -2407934, -2407422, 6, -2406910, -2342910, 279042, -2341886, -2341374, -2277374, -2276862, -2276350, -2275838, -2211838, 0, 6, 6, 6, 6, 6]
    // CHECK: ppe = #VPU.PPETask<mode = <FLEXARB>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [1], quant_shift = [0], quant_post_shift = 5 : i64>
}
