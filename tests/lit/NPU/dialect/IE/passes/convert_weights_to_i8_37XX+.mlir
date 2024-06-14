//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-i8 --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

!qElemType = !quant.uniform<u8:f16, 2.000000e+00:128>
// CHECK: !qElemType = !quant.uniform<i8:f16, 2.000000e+00>


// CHECK-LABEL: @ConvertFromU8ToI8
func.func @ConvertFromU8ToI8(%arg0: tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32> {
  %cst = const.Declare tensor<16x16x1x1x!qElemType> = dense<127.0> : tensor<16x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf16>
  %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1,1]}:  tensor<1x16x1x1xf16> , tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
  return %3 : tensor<1x16x1x1xf32>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:      dense<1.270000e+02> : tensor<16x16x1x1xf32>
    // CHECK-SAME:      [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.QuantCast<>,
    // CHECK-SAME:       #const.ConvertElemType<f16>, #const.Add<-1.280000e+02 : f64>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
    // CHECK:       [[VAL1:%.*]] = IE.Convert([[ARG0:%.*]]) {dstElemType = f16} : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
    // CHECK:       [[VAL3:%.*]] = IE.Convolution([[VAL1]], [[VAL2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[VAL4:%.*]] = IE.Convert([[VAL3]]) {dstElemType = f32} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf32>
    // CHECK:       return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// We don't convert u8 to i8 because of the zero point value of U8 which must be 128.
// Conversion converts from u8 ZP = 128 to i8 ZP = 0
// CHECK-LABEL: @NotConvertU8Weights
func.func @NotConvertU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<9.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.ConvertElemType<ui8>,
    // CHECK-SAME:      #const.QuantCast<!qElemType>
    // CHECK:       [[VAL0:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @KeepI8Weights
func.func @KeepI8Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.ConvertElemType<si8>,
    // CHECK-SAME:      #const.QuantCast<!qElemType>
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8<0:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
// CHECK: !qElemType = !quant.uniform<i8:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK-LABEL: @ConvertFromPerAxisTypeU8ToI8
func.func @ConvertFromPerAxisTypeU8ToI8(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<3.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:       [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>,
    // CHECK-SAME:        #const.QuantCast<>, #const.ConvertElemType<f16>, #const.Add<-1.280000e+02 : f64>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
    // CHECK:       [[VAL0:%.*]] = IE.Dequantize([[CST]])
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL1]]
}
