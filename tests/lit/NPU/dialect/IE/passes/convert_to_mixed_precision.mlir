//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

func.func @MixedPrecisionConv(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL2]]
}

// -----


!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @AvoidMixedPrecisionConvPerAxes(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionGroupConv(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x1x3x3x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @MixedPrecisionGroupConvPerAxes(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    //CHECK: [[VAL1:%.*]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType1>, tensor<16x1x3x3x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionAdd(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Add([[VAL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1x!qElemType>, tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0080899796503431654:-126>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @AvoidMixedPrecisionAddDifferentOperandType(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL3:%.*]] = IE.Add([[VAL1]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @AvoidMixedPrecisionAddPerAxes(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL3:%.*]] = IE.Add([[VAL1]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x1x1xf16>
}
