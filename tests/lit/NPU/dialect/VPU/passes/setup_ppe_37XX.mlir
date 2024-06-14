//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --setup-ppe %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882:128>

// CHECK-LABEL: @NoopCase
func.func @NoopCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 127 : i64,
            clamp_low = -128 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:         mode = <NOOP>,
    // CHECK-SAME:         clamp_low = 0 : i64,
    // CHECK-SAME:         clamp_high = 255 : i64,
    // CHECK-SAME:         lrelu_mult = 1 : i64,
    // CHECK-SAME:         lrelu_shift = 0 : i64,
    // CHECK-SAME:         quant_mult = [16822],
    // CHECK-SAME:         quant_shift = [13],
    // CHECK-SAME:         quant_post_shift = 0 : i64,
    // CHECK-SAME:         fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @ReLUCase
func.func @ReLUCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:       ppe = #VPU.PPETask<
    // CHECK-SAME:           mode = <LRELU>,
    // CHECK-SAME:           clamp_low = 0 : i64,
    // CHECK-SAME:           clamp_high = 255 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64,
    // CHECK-SAME:           lrelu_shift = 0 : i64,
    // CHECK-SAME:           quant_mult = [16822],
    // CHECK-SAME:           quant_shift = [13],
    // CHECK-SAME:           quant_post_shift = 0 : i64,
    // CHECK-SAME:           fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:       >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.027450980392156862>
!qElemType1 = !quant.uniform<u8:f16, 0.015686274509803921>
!qElemType2 = !quant.uniform<u8:f16, 0.070588235294117646:128>

// CHECK-LABEL: @ReLUQuantizedCaseWithZeroPointNotEqual0
func.func @ReLUQuantizedCaseWithZeroPointNotEqual0(%arg0: tensor<1x16x32x32x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x16x32x32x!qElemType1, {order = #NHWC}>) -> tensor<1x16x32x32x!qElemType2, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            mode = <ADD>,
            clamp_low = 0 : i64,
            clamp_high = 127 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            quant_mult = [29013],
            quant_shift = [31],
            quant_post_shift = 0 : i64,
            in1_quant_mult = [28784],
            in2_quant_mult = [16448],
            fp_prelu_alpha = 1.000000e+00 : f64
        >
    } -> tensor<1x16x32x32x!qElemType2, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:       ppe = #VPU.PPETask<
    // CHECK-SAME:           mode = <NOOP>,
    // CHECK-SAME:           clamp_low = 128 : i64,
    // CHECK-SAME:           clamp_high = 255 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64,
    // CHECK-SAME:           lrelu_shift = 0 : i64,
    // CHECK-SAME:           quant_mult = [29013],
    // CHECK-SAME:           quant_shift = [31],
    // CHECK-SAME:           quant_post_shift = 0 : i64,
    // CHECK-SAME:           in1_quant_mult = [28784],
    // CHECK-SAME:           in2_quant_mult = [16448],
    // CHECK-SAME:           fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:       >
    // CHECK-SAME:  } -> tensor<1x16x32x32x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x16x32x32x!qElemType2, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.027450980392156862>
!qElemType1 = !quant.uniform<u8:f16, 0.015686274509803921>
!qElemType2 = !quant.uniform<u8:f16, 0.070588235294117646>

// CHECK-LABEL: @ReLUQuantizedCaseWithZeroPointEqual0
func.func @ReLUQuantizedCaseWithZeroPointEqual0(%arg0: tensor<1x16x32x32x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x16x32x32x!qElemType1, {order = #NHWC}>) -> tensor<1x16x32x32x!qElemType2, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            mode = <ADD>,
            clamp_low = 0 : i64,
            clamp_high = 255 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            quant_mult = [29013],
            quant_shift = [31],
            quant_post_shift = 0 : i64,
            in1_quant_mult = [28784],
            in2_quant_mult = [16448],
            fp_prelu_alpha = 1.000000e+00 : f64
        >
    } -> tensor<1x16x32x32x!qElemType2, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:       ppe = #VPU.PPETask<
    // CHECK-SAME:           mode = <NOOP>,
    // CHECK-SAME:           clamp_low = 0 : i64,
    // CHECK-SAME:           clamp_high = 255 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64,
    // CHECK-SAME:           lrelu_shift = 0 : i64,
    // CHECK-SAME:           quant_mult = [29013],
    // CHECK-SAME:           quant_shift = [31],
    // CHECK-SAME:           quant_post_shift = 0 : i64,
    // CHECK-SAME:           in1_quant_mult = [28784],
    // CHECK-SAME:           in2_quant_mult = [16448],
    // CHECK-SAME:           fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:       >
    // CHECK-SAME:  } -> tensor<1x16x32x32x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x16x32x32x!qElemType2, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @ReLUXCase
func.func @ReLUXCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 6 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:       ppe = #VPU.PPETask<
    // CHECK-SAME:           mode = <LRELUX>,
    // CHECK-SAME:           clamp_low = 0 : i64,
    // CHECK-SAME:           clamp_high = 6 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64,
    // CHECK-SAME:           lrelu_shift = 0 : i64,
    // CHECK-SAME:           quant_mult = [16822],
    // CHECK-SAME:           quant_shift = [13],
    // CHECK-SAME:           quant_post_shift = 0 : i64,
    // CHECK-SAME:           fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:       >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @PReLUCase
func.func @PReLUCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 10 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 10 : i64,
    // CHECK-SAME:          quant_mult = [16822],
    // CHECK-SAME:          quant_shift = [13],
    // CHECK-SAME:          quant_post_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 9.765625E-4 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.061718750000000003:132>

// CHECK-LABEL: @MixedPrecision
func.func @MixedPrecision(%arg0: tensor<1x256x16x32xf16, {order =#NHWC}>) -> tensor<1x256x16x32x!qElemType, {order = #NHWC}> {
    %w = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<256x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<256x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %w, %wt) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [256, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x256x16x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution(%arg0, %cst, %cst_0) {
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 16.202531814575195 : f64
    // CHECK-SAME:      >

    return %0 : tensor<1x256x16x32x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.061718750000000003>

// CHECK-LABEL: @I8WeightsMixedPrecision
func.func @I8WeightsMixedPrecision(%arg0: tensor<1x16x16x32xf16, {order =#NHWC}>) -> tensor<1x16x16x32xf16, {order = #NHWC}> {
    %w = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %w, %wt) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x16x32xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          clamp_high = 2147483647 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 0.06171875074505806 : f64
    // CHECK-SAME:      >

    return %0 : tensor<1x16x16x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PReluConv
func.func @PReluConv(%arg0: tensor<1x1024x40x40xf16, {order =#NHWC}>) -> tensor<1x256x40x40xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> =
    dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 0 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} -> tensor<1x256x40x40xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          clamp_high = 2147483647 : i64,
    // CHECK-SAME:          lrelu_mult = 0 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 0.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x40x40xf16, {order = #NHWC}>

    return %0 : tensor<1x256x40x40xf16, {order = #NHWC}>

}
