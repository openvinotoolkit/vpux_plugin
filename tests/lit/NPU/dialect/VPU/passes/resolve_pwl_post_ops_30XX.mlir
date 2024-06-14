//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --resolve-pwl-post-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

!qElemType = !quant.uniform<ui8:f32, 1.000000e+00>
!qElemType1 = !quant.uniform<ui8:f32, 9.8455479662789983E-4>
!qElemType2 = !quant.uniform<ui8:f32, 0.0040160642570281121:3>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InsertQuantizeCastConvSigmoid
func.func @InsertQuantizeCastConvSigmoid(%arg0: tensor<1x16x4x4x!qElemType, {order = #NHWC}>) -> tensor<1x16x3x3xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<clamp_high = 1638 : i64,clamp_low = -1638 : i64,lrelu_mult = 1 : i64,lrelu_shift = 0 : i64,mode = <SIGMOID>>,
        rawFilterShape = [16, 16, 2, 2],
        strides = [1, 1]
        } -> tensor<1x16x3x3x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%0, %0){
        op_type = #VPU.eltwise_type<AND>
    } -> tensor<1x16x3x3xf16, {order = #NHWC}>
    return %1 : tensor<1x16x3x3xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:     ppe = #VPU.PPETask<mode = <SIGMOID>, clamp_low = -1638 : i64, clamp_high = 1638 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>
    // CHECK-SAME:     rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-SAME:     -> tensor<1x16x3x3x!qElemType1, {order = #NHWC}>

    // CHECK:       [[VAL0:%.*]] = VPU.QuantizeCast
    // CHECK-SAME:     {dstElemType = !qElemType2}
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType1, {order = #NHWC}> ->
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType2, {order = #NHWC}>

    // CHECK:       [[VAL1:%.*]] =  VPU.NCE.Eltwise([[VAL0]]
    // CHECK-SAME:      -> tensor<1x16x3x3xf16, {order = #NHWC}>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<ui8:f32, 1.000000e+00>
!qElemType1 = !quant.uniform<ui8:f32, 9.8455479662789983E-4>
!qElemType2 = !quant.uniform<ui8:f32, 0.0040160642570281121:3>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InsertRequantizeConvSigmoid
func.func @InsertRequantizeConvSigmoid(%arg0: tensor<1x16x4x4x!qElemType, {order = #NHWC}>) -> tensor<1x16x1x9x!qElemType> {
    %weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<clamp_high = 1638 : i64,clamp_low = -1638 : i64,lrelu_mult = 1 : i64,lrelu_shift = 0 : i64,mode = <SIGMOID>>,
        rawFilterShape = [16, 16, 2, 2],
        strides = [1, 1]
        } -> tensor<1x16x3x3x!qElemType, {order = #NHWC}>

    %1 = VPU.Reshape(%0) { shape_value = [1, 16, 1, 9] } : tensor<1x16x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x1x9x!qElemType>
    return %1 : tensor<1x16x1x9x!qElemType>

    // CHECK:       [[VAL0:%.*]] = VPU.NCE.Convolution
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:     ppe = #VPU.PPETask<mode = <SIGMOID>, clamp_low = -1638 : i64, clamp_high = 1638 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>
    // CHECK-SAME:     rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-SAME:     -> tensor<1x16x3x3x!qElemType1, {order = #NHWC}>

    // CHECK-NOT:   VPU.QuantizeCast

    // CHECK:       [[VAL1:%.*]] =  VPU.NCE.Eltwise([[VAL0]]
    // CHECK-SAME:      mode = <AND>

    // CHECK:       [[VAL2:%.*]] =  VPU.NCE.Eltwise([[VAL1]]
    // CHECK-SAME:     mode = <AND>

    // CHECK:       [[VAL3:%.*]] =  VPU.Reshape([[VAL2]]

    // CHECK:       return [[VAL3]]
}
