//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

func.func @FuseQuantParamsIntoAvgPoolAsymmetricKernel(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x15x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.AvgPool(%2) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x15x14xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x15x14xf16> -> tensor<1x3x15x14x!qElemType>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x15x14x!qElemType> -> tensor<1x3x15x14xf16>
  return %5 : tensor<1x3x15x14xf16>

  // CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  // CHECK: [[VAL1:%.*]] = IE.AvgPool([[VAL0]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x15x14x!qElemType>
  // CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x15x14x!qElemType> -> tensor<1x3x15x14xf16>
  // CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

func.func @FuseQuantParamsIntoAvgPoolSymmetricKernel(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.AvgPool(%2) {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x14x14xf16>
  %4 = IE.Quantize(%3) {dstElemType = !qElemType} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
  return %5 : tensor<1x3x14x14xf16>

  // CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  // CHECK: [[VAL1:%.*]] = IE.AvgPool([[VAL0]]) {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x14x14x!qElemType>
  // CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
  // CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039117518593283261>
!qElemType1 = !quant.uniform<u8:f16, 0.0039005478223164878>

func.func @DoNotFuseQuantParamsIntoAvgPoolWithExcludePadsAttr(%arg0: tensor<1x3x135x240x!qElemType>) -> tensor<1x3x68x120x!qElemType1> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x135x240x!qElemType> -> tensor<1x3x135x240xf16>
  %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType1} : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType1>
  return %2 : tensor<1x3x68x120x!qElemType1>

  // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {
  // CHECK-SAME:      dstElemType = f16
  // CHECK-SAME:      } : tensor<1x3x135x240x!qElemType> -> tensor<1x3x135x240xf16>
  // CHECK: [[AVGPOOL:%.*]] = IE.AvgPool([[DEQUANTIZE]]) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
  // CHECK: [[RESULT:%.*]] = IE.Quantize([[AVGPOOL]]) {
  // CHECK-SAME:      dstElemType = !qElemType1
  // CHECK-SAME:      } : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType1>
  // CHECK: return [[RESULT]] : tensor<1x3x68x120x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {1.010680671751968504:128,1.0081200787401574797:128,1.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.0534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 3.1368405211205576E-7>

// CHECK-LABEL: @NotFuseQuantParamsIntoConvForInvalidApproximation
func.func @NotFuseQuantParamsIntoConvForInvalidApproximation(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  //CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]]) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  //CHECK: return [[VAL6]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 3.1368405211205576E-7>

// CHECK-LABEL: @NotFuseQuantParamsIntoGroupConvForInvalidApproximation
func.func @NotFuseQuantParamsIntoGroupConvForInvalidApproximation(%arg0: tensor<1x3x10x10xf16>) -> tensor<1x3x10x10xf16> {
    %cst = const.Declare tensor<3x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>

    %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

    return %5 : tensor<1x3x10x10xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<3x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>
    //CHECK: [[VAL3:%.*]] = IE.GroupConvolution([[VAL1]], [[VAL2]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>
    //CHECK: [[VAL4:%.*]] = IE.Quantize([[VAL3]]) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
    //CHECK: [[VAL5:%.*]] = IE.Dequantize([[VAL4]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

    //CHECK: return [[VAL5]] : tensor<1x3x10x10xf16>
}
