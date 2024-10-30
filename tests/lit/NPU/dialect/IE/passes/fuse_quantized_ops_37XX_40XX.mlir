//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerChannelMaxPool
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @DoNotFusePerChannelMaxPool(%arg0: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %maxPool = IE.MaxPool(%dequantize) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%maxPool) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[MAXPOOL:%.+]] = IE.MaxPool([[DEQUANT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[MAXPOOL]])
    //CHECK:  return [[QUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotFusePerChannelAvgPool
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @DoNotFusePerChannelAvgPool(%arg0: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %avgPool = IE.AvgPool(%dequantize) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%avgPool) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[AVGPOOL:%.+]] = IE.AvgPool([[DEQUANT]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[AVGPOOL]])
    //CHECK:  return [[QUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @FusePerChannelEltwiseNoChanges
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x16x16x!qElemType>, [[ARG1:%.+]]: tensor<1x4x16x16x!qElemType>)
func.func @FusePerChannelEltwiseNoChanges(%arg0: tensor<1x4x16x16x!qElemType>, %arg1: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %dequantize0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %dequantize1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %add = IE.Add(%dequantize0, %dequantize1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %quantize = IE.Quantize(%add) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %quantize : tensor<1x4x16x16x!qElemType>

    //CHECK:  [[DEQUANT0:%.+]] = IE.Dequantize([[ARG0]])
    //CHECK:  [[DEQUANT1:%.+]] = IE.Dequantize([[ARG1]])
    //CHECK:  [[ADD:%.+]] = IE.Add([[DEQUANT0]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  [[QUANT:%.+]] = IE.Quantize([[ADD]])
    //CHECK:  return [[QUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.051102941176470587:121>
!qElemType1 = !quant.uniform<u8:f16, 0.034064797794117647:55>

// CHECK-LABEL: @LReluQuant
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x128x512x!qElemType>)
func.func @LReluQuant(%arg0: tensor<1x1x128x512x!qElemType>) -> tensor<1x1x128x512x!qElemType1> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
  %1 = IE.LeakyRelu(%0) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType1 } : tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>

  return %2 : tensor<1x1x128x512x!qElemType1>

  //CHECK: [[LEAKYRELU:%.+]] = IE.LeakyRelu([[INPUT]]) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512x!qElemType1>
  //CHECK: return [[LEAKYRELU]] : tensor<1x1x128x512x!qElemType1>
}
