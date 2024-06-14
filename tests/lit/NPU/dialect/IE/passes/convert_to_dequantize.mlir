//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-dequantize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertToDequantize
func.func @ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK:      ConvertToDequantizeWithQuantInput
// CHECK-SAME: ([[ARG0:%.*]]: tensor<1x64x64x100x[[Q_ELEM_TYPE1]], {order = #NHWC}>, [[ARG1:%.*]]: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @ConvertToDequantizeWithQuantInput(%arg0: tensor<1x64x64x100x!qElemType1, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType1, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x[[Q_ELEM_TYPE1]], {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertToDequantizeWithMiddleOp
func.func @ConvertToDequantizeWithMiddleOp(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1xsi8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1xsi8> -> tensor<64x64x1xf16>
  %1 = IE.Reshape(%0) { shape_value = [64, 64, 1, 1] } : tensor<64x64x1xf16> -> tensor<64x64x1x1xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %2 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64x1xsi8> -> tensor<64x64x1x!qElemType>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x!qElemType> -> tensor<64x64x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64x1xf16> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL3:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @KeepConvertIfTypeIsNotI8
func.func @KeepConvertIfTypeIsNotI8(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xui8>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xui8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:  [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xui8> -> tensor<64x64x1x1xf16>
  // CHECK:  [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:  return [[VAL1]]
}

// -----

// CHECK-LABEL: @KeepConvertIfNotFilter
func.func @KeepConvertIfNotFilter(%arg0: tensor<1x64x32x200xsi8>, %arg1: tensor<1x64x64x64xf16>) -> tensor<1x1x1x37xf16> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x64x32x200xsi8> -> tensor<1x64x32x200xf16>
  %1 = IE.Reshape(%0) { shape_value = [1, 64, 64, 100] } : tensor<1x64x32x200xf16> -> tensor<1x64x64x100xf16>
  %2 = IE.Convolution(%1, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16>, tensor<1x64x64x64xf16> -> tensor<1x1x1x37xf16>
  return %2 : tensor<1x1x1x37xf16>

  // CHECK:  [[VAL0:%.*]] = IE.Convert([[ARG0:%.*]]) {dstElemType = f16} : tensor<1x64x32x200xsi8> -> tensor<1x64x32x200xf16>
  // CHECK:  [[VAL1:%.*]] = IE.Reshape([[VAL0]]) {shape_value = [1, 64, 64, 100]} : tensor<1x64x32x200xf16> -> tensor<1x64x64x100xf16>
  // CHECK:  [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[ARG1:%.*]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16>, tensor<1x64x64x64xf16> -> tensor<1x1x1x37xf16>
  // CHECK:  return [[VAL2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertToDequantizeIfInputIsConstants
func.func @NotConvertToDequantizeIfInputIsConstants(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<64x64x1x1xsi8> = dense<1> : tensor<64x64x1x1xsi8>
  %0 = IE.Convert(%cst) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              [[CST:%.*]] = const.Declare tensor<64x64x1x1xf16> = dense<1> : tensor<64x64x1x1xsi8>, [#const.ConvertElemType<f16>]
  // CHECK-NOT:          IE.QuantizeCast
  // CHECK:              [[VAL0:%.*]] = IE.Convolution([[ARG0:%.*]], [[CST:%.*]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @I4ConvertToDequantize
func.func @I4ConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi4>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi4> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi4> -> tensor<64x64x1x1x!qElemType>
  // CHECK:              [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I1DontConvertToDequantize
func.func @I1DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xi1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xi1> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xi1> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I32DontConvertToDequantize
func.func @I32DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi32>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi32> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xsi32> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I16DontConvertToDequantize
func.func @I16DontConvertToDequantize(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x1xsi16>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<64x64x1x1xsi16> -> tensor<64x64x1x1xf16>
  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:              [[VAL0:%.*]] = IE.Convert([[ARG1:%.*]]) {dstElemType = f16} : tensor<64x64x1x1xsi16> -> tensor<64x64x1x1xf16>
  // CHECK:              [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:              return [[VAL1]]
}
