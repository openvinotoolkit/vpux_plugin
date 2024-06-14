//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-reduce-sum-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @ConvertReduceSumToConv4D
func.func @ConvertReduceSumToConv4D(%arg0: tensor<1x4x32x32xf16>) -> tensor<1x1x32x32xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x32x32xf16> -> tensor<1x1x32x32xf16>
  return %1 : tensor<1x1x32x32xf16>

  // CHECK:       %cst = const.Declare tensor<1x4x1x1xf16> = dense<1.000000e+00> : tensor<1x4x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NCHW>]
  // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x32x32xf16>, tensor<1x4x1x1xf16> -> tensor<1x1x32x32xf16>

  // CHECK:       return [[CONV]] : tensor<1x1x32x32xf16>
}

// -----

// CHECK-LABEL: @NotConvertReduceSumToConv4DIfNotReduceChannel
func.func @NotConvertReduceSumToConv4DIfNotReduceChannel(%arg0: tensor<1x4x32x32xf16>) -> tensor<1x4x32x1xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x4x32x32xf16> -> tensor<1x4x32x1xf16>
  return %1 : tensor<1x4x32x1xf16>

  // CHECK-NOT:   Convolution
}

// -----

// CHECK-LABEL: @ConvertReduceSumToConvForChannelAlignedInput
func.func @ConvertReduceSumToConvForChannelAlignedInput(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x1x32x32xf16> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x32x32xf16> -> tensor<1x1x32x32xf16>
  return %1 : tensor<1x1x32x32xf16>

  // CHECK:       %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NCHW>]
  // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x32x32xf16>

  // CHECK:       return [[CONV]] : tensor<1x1x32x32xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.010857077205882353:127>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertReduceSumToConvU8
func.func @ConvertReduceSumToConvU8(%arg0: tensor<1x4x32x32x!qElemType>) -> tensor<1x1x32x32x!qElemType> {
  %1 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x32x32x!qElemType> -> tensor<1x1x32x32x!qElemType>
  return %1 : tensor<1x1x32x32x!qElemType>

  // CHECK:       %cst = const.Declare tensor<1x4x1x1x!qElemType1>
  // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x32x32x!qElemType>, tensor<1x4x1x1x!qElemType1> -> tensor<1x1x32x32x!qElemType>

  // CHECK:       return [[CONV]] : tensor<1x1x32x32x!qElemType>
}
