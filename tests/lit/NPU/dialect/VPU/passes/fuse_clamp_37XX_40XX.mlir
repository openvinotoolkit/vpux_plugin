//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-clamp --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp32to128
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClamp32to128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 32 : i64,
    // CHECK-SAME:          clamp_high = 128 : i64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SkipClampWithoutNCE
func.func @SkipClampWithoutNCE(%arg0: tensor<1x255x56x56x!qElemType, {order = #NHWC}>)
                          -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x255x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.+]] = VPU.Expand
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[EXPAND]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:1, {0.25, 0.5, 0.75, 1.0,
                                            0.25, 0.5, 0.75, 1.0,
                                            0.25, 0.5, 0.75, 1.0,
                                            0.25, 0.5, 0.75, 1.0}>

// CHECK-LABEL: @SkipPerChannelQuant
func.func @SkipPerChannelQuant(%arg0: tensor<1x16x56x56x!qElemType, {order = #NHWC}>,
                          %arg1: tensor<16x16x1x1x!qElemType, {order = #NHWC}>)
                          -> tensor<1x16x56x56x!qElemType, {order = #NHWC}> {
    %WEIGHT_TABLE = const.Declare tensor<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %arg1, %WEIGHT_TABLE) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1638 : i64,
            lrelu_shift = 14 : i64,
            mode = <LPRELU>
        >,
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x16x56x56x!qElemType, {order = #NHWC}> -> tensor<1x16x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x16x56x56x!qElemType, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[CONV]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipFloat16WithNonZeroMin
func.func @SkipFloat16WithNonZeroMin(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                                %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                                -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK:   [[CLAMP:%.+]] = VPU.Clamp([[ELTWISE]])
    // CHECK:   return [[CLAMP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to120
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @FloatClamp0to120(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.200000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          clamp_high = 2147418111 : i64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to128
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @FloatClamp0to128(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<
            clamp_high = 22400 : i64,
            clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          clamp_high = 22400 : i64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:255>:f16, 0.5:127>

// CHECK-LABEL: @QuantClampNeg6to6
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClampNeg6to6(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 6.000000e+00 : f64,
        min = -6.000000e+00 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 115 : i64,
    // CHECK-SAME:          clamp_high = 139 : i64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp0To128
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClamp0To128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                       -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<
            clamp_high = 120 : i64,
            clamp_low = 32 : i64,
            fp_prelu_alpha = 1.250000e-01 : f64,
            lrelu_mult = 1024 : i64,
            lrelu_shift = 13 : i64,
            mode = <LPRELU>
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 32 : i64,
    // CHECK-SAME:          clamp_high = 120 : i64,
    // CHECK-SAME:          lrelu_mult = 1024 : i64,
    // CHECK-SAME:          lrelu_shift = 13 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.250000e-01 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 13.571496821384804:117>

// CHECK-LABEL: @ConvWithMultipleConsumers
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x32x16x16x!qElemType, {order = #NHWC}>)
func.func @ConvWithMultipleConsumers(%arg0: tensor<1x32x16x16x!qElemType, {order = #NHWC}>) -> (tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>, tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>) {
    %weights = const.Declare tensor<9216x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<clamp_high = 255 : i64,
               clamp_low = 0 : i64,
               fp_prelu_alpha = 1.000000e+00 : f64,
               lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = <NOOP> >,
               rawFilterShape = [9216, 32, 3, 3],
               strides = [1, 1]
    } -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 4608, 16, 16] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> to tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>
    %2 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    return %1, %2 : tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>, tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<9216x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:  [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
    // CHECK:      [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME: {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:  ppe = #VPU.PPEInt<
    // CHECK-SAME:         mode = <NOOP>,
    // CHECK-SAME:         clamp_low = 0 : i64,
    // CHECK-SAME:         clamp_high = 255 : i64,
    // CHECK-SAME:         lrelu_mult = 1 : i64,
    // CHECK-SAME:         lrelu_shift = 0 : i64,
    // CHECK-SAME:         fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME: rawFilterShape = [9216, 32, 3, 3],
    // CHECK-SAME: strides = [1, 1]} -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:      [[SLICE:%.+]] = VPU.Slice [[CONV_0]] [0, 0, 0, 0] [1, 4608, 16, 16] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> to tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:      [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME: {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:  ppe = #VPU.PPEInt<
    // CHECK-SAME:      mode = <NOOP>,
    // CHECK-SAME:      clamp_low = 117 : i64,
    // CHECK-SAME:      clamp_high = 126 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64,
    // CHECK-SAME:      lrelu_shift = 0 : i64,
    // CHECK-SAME:      fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:  rawFilterShape = [9216, 32, 3, 3],
    // CHECK-SAME:  strides = [1, 1]} -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:      return [[SLICE]], [[CONV_1]] : tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>, tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
}
