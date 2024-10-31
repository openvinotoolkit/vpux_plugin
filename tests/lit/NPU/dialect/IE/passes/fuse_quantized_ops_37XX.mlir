//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 0.39320635328105852:128>
!qElemType2 = !quant.uniform<u8:f16, 0.39320638320025275:128>

// CHECK-LABEL: @FuseQParamsIntoAddWithDiffInTypes
func.func @FuseQParamsIntoAddWithDiffInTypes(%arg0: tensor<1x16x180x320xf16>, %arg1: tensor<1x16x180x320xf16>) -> tensor<1x16x180x320xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x180x320x!qElemType> -> tensor<1x16x180x320xf16>

  %2 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>

  %4 = IE.Add(%1, %3) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x16x180x320xf16>

  %5 = IE.Quantize(%4) {dstElemType = !qElemType2} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  return %6 : tensor<1x16x180x320xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL0]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x180x320x!qElemType>, tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320x!qElemType2>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0034409466911764705>
!qElemType1 = !quant.uniform<u8:f16, 0.12503063725490196:128>
!qElemType2 = !quant.uniform<u8:f16, 0.067708337073232608:128>

// CHECK-LABEL: @DoNotFuseQuantParamsIntoEltwiseMul
func.func @DoNotFuseQuantParamsIntoEltwiseMul(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %5 = IE.Multiply(%2, %4) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %6 = IE.Quantize(%5) {dstElemType = !qElemType2}: tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
  %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x3x16x16x!qElemType2> -> tensor<1x3x16x16xf16>

  return %7 : tensor<1x3x16x16xf16>

  // CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} :
  // CHECK-SAME:    tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>

  // CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} :
  // CHECK-SAME:    tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

  // CHECK: [[VAL3:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType1} :
  // CHECK-SAME:    tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>

  // CHECK: [[VAL4:%.*]] = IE.Dequantize([[VAL3]]) {dstElemType = f16} :
  // CHECK-SAME:    tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>

  // CHECK: [[VAL5:%.*]] = IE.Multiply([[VAL2]], [[VAL4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:    tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

  // CHECK: [[VAL6:%.*]] = IE.Quantize([[VAL5]]) {dstElemType = !qElemType2} :
  // CHECK-SAME:    tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>

  // CHECK: [[VAL7:%.*]] = IE.Dequantize([[VAL6]]) {dstElemType = f16} :
  // CHECK-SAME:    tensor<1x3x16x16x!qElemType2> -> tensor<1x3x16x16xf16>

  // CHECK: return [[VAL7]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0057995634920456826:125>

// CHECK-LABEL: @FuseQParamIntoDepthToSpaceOp
func.func @FuseQParamIntoDepthToSpaceOp(%arg0: tensor<1x12x180x320xf16>, %arg1: tensor<1x3x360x640xf16>) -> tensor<1x3x360x640xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x12x180x320xf16> -> tensor<1x12x180x320x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x12x180x320x!qElemType> -> tensor<1x12x180x320xf16>
  %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x3x360x640xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x3x360x640xf16> -> tensor<1x3x360x640x!qElemType>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x360x640x!qElemType> -> tensor<1x3x360x640xf16>
  return %4 : tensor<1x3x360x640xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x12x180x320xf16> -> tensor<1x12x180x320x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.DepthToSpace([[VAL0]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x12x180x320x!qElemType> -> tensor<1x3x360x640x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x360x640x!qElemType> -> tensor<1x3x360x640xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FuseQuantParamsIntoConv
func.func @FuseQuantParamsIntoConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.3127431869506836 : f64}>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 2.3127431869506836 : f64}>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType1>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType2>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseQuantParamsIntoConv
func.func @DoNotFuseQuantParamsIntoConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = -2.3127431869506836 : f64}>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>

  //CHECK: [[VAL4:%.*]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = -2.3127431869506836 : f64}>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]]) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL6]]
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {7.8740158653634745E-4:128,7.8740158653634745E-4:128,7.8740158653634745E-4:128}>
!qElemType1 = !quant.uniform<u8:f16, 3.9215E-5:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseToConvWithInvalidRescaleBias
func.func @DoNotFuseToConvWithInvalidRescaleBias(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %bias = const.Declare tensor<1x3x1x1xf16> = dense<100.0> : tensor<1x3x1x1xf16>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK:   [[VAL0:%.*]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+02> : tensor<1x3x1x1xf16>
  //CHECK:   [[VAL1:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK:   [[VAL2:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK:   [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK:   [[VAL4:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  //CHECK:   [[VAL5:%.*]] = IE.Convolution([[VAL3]], [[VAL4]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  //CHECK:   [[VAL6:%.*]] = IE.Quantize([[VAL5]]) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  //CHECK:   [[VAL7:%.*]] = IE.Dequantize([[VAL6]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  //CHECK: return [[VAL7]]
}


// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {7.8740158653634745E-4:128,7.8740158653634745E-4:128,7.8740158653634745E-4:128}>
!qElemType1 = !quant.uniform<u8:f16, 3.9215E-5:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @DoNotFuseToGroupConvWithInvalidRescaleBias
func.func @DoNotFuseToGroupConvWithInvalidRescaleBias(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x1x3x3x!qElemType> = dense<1.0> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %bias = const.Declare tensor<1x3x1x1xf16> = dense<100.0> : tensor<1x3x1x1xf16>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>
  %4 = IE.GroupConvolution(%2, %3, %bias) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x1x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK:   [[VAL0:%.*]] = const.Declare tensor<1x3x1x1xf16> = dense<1.000000e+02> : tensor<1x3x1x1xf16>
  //CHECK:   [[VAL1:%.*]] = const.Declare tensor<3x1x3x3x!qElemType> = dense<1.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK:   [[VAL2:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK:   [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  //CHECK:   [[VAL4:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>
  //CHECK:   [[VAL5:%.*]] = IE.GroupConvolution([[VAL3]], [[VAL4]], [[VAL0]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x1x3x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x14x14xf16>
  //CHECK:   [[VAL6:%.*]] = IE.Quantize([[VAL5]]) {dstElemType = !qElemType2} : tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  //CHECK:   [[VAL7:%.*]] = IE.Dequantize([[VAL6]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  //CHECK: return [[VAL7]]
}
