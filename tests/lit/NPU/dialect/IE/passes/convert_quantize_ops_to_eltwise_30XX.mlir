//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-quantize-ops-to-nce-ops  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

module @ConvertQuantizeToEltwise {

func.func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>

    // CHECK:  %[[VAL0:.*]] = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!qElemType>
    // CHECK:  %[[VAL1:.*]] = IE.And(%[[VAL0]], %[[VAL0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x!qElemType>, tensor<1x4x!qElemType> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>

    // CHECK:  return %[[VAL1]] : tensor<1x4xf32>
}

}

// -----

!qElemType = !quant.uniform<u8:f32:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

func.func @PerChannelNoChanges(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>

    // CHECK:  IE.Quantize
    // CHECK:  IE.Dequantize
    // CHECK-NOT:  IE.And
    // CHECK-NOT:  IE.And

    return %1 : tensor<1x4xf32>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

module @NoConvertQuantizeToEltwise {

func.func @EnabelCMcovNoChanges(%arg0 : tensor<1x3x352x352xf32>) -> tensor<1x32x175x175x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x352x352xf32> -> tensor<1x3x352x352x!qElemType>
    %1 = const.Declare tensor<32x3x3x3xf32> = dense<1.0> : tensor<32x3x3x3xf32>
    %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} :
                    tensor<1x3x352x352x!qElemType>, tensor<32x3x3x3xf32> -> tensor<1x32x175x175x!qElemType>

    return %2 : tensor<1x32x175x175x!qElemType>

    // CHECK:  [[QUANT0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x352x352xf32> -> tensor<1x3x352x352x!qElemType>
    // CHECK-DAG:  [[CST0:%.*]] = const.Declare tensor<32x3x3x3xf32> = dense<1.000000e+00> : tensor<32x3x3x3xf32>
    // CHECK:  [[VAR0:%.*]] = IE.Convolution([[QUANT0]], [[CST0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]}
    // CHECK-SAME:  tensor<1x3x352x352x!qElemType>, tensor<32x3x3x3xf32> -> tensor<1x32x175x175x!qElemType>
    // CHECK:  return [[VAR0]] : tensor<1x32x175x175x!qElemType>
}

}
