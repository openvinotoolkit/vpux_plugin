//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-quantize-dequantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantReshape
func.func @PropagateDequantReshape(%arg0: tensor<1x256x!qElemType>) -> tensor<1x256x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x!qElemType> -> tensor<1x256xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 256, 1, 1], dim_mapping = [[0], [1, 2, 3]]} : tensor<1x256xf16> -> tensor<1x256x1x1xf16>
  %3 = IE.AffineReshape(%1) {shape_value = [1, 256, 1, 1], dim_mapping = [[0], [1, 2, 3]]} : tensor<1x256xf16> -> tensor<1x256x1x1xf16>
  %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>

  return %4 : tensor<1x256x1x1xf16>

  //CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANT0:%.*]] = IE.Dequantize([[RESHAPE0]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  //CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANT1:%.*]] = IE.Dequantize([[RESHAPE1]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT0]], [[DEQUANT1]])
  //CHECK: return [[ADD]] : tensor<1x256x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantReshape
func.func @PropagateQuantReshape(%arg0: tensor<1x9x1x1xf32>) -> (tensor<1x9x!qElemType>, tensor<1x9x!qElemType>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1x1xf32> -> tensor<1x9x1x1xf16>
  %1 = IE.AffineReshape(%0) {shape_value = [1, 9], dim_mapping = [[0], [1], [1], [1]]} : tensor<1x9x1x1xf16> -> tensor<1x9xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9xf16> -> tensor<1x9x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9xf16> -> tensor<1x9x!qElemType>

  return %2, %3 : tensor<1x9x!qElemType>, tensor<1x9x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 9]} : tensor<1x9x1x1x!qElemType> -> tensor<1x9x!qElemType>
  //CHECK: return [[VAL1]], [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDequantTranspose
func.func @PropagateDequantTranspose(%arg0: tensor<1x256x2x2x!qElemType>) -> tensor<1x2x2x256xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x2x2x!qElemType> -> tensor<1x256x2x2xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x2x256xf16>, tensor<1x2x2x256xf16> -> tensor<1x2x2x256xf16>

  return %3 : tensor<1x2x2x256xf16>

  //CHECK: [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x2x2x256x!qElemType> -> tensor<1x2x2x256xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDequantBeforeOutput
func.func @PropagateDequantBeforeOutput(%arg0: tensor<1x256x2x2x!qElemType>) -> tensor<1x2x2x256xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x2x2x!qElemType> -> tensor<1x256x2x2xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>

  return %2 : tensor<1x2x2x256xf16>

  //CHECK: [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x2x2x256x!qElemType> -> tensor<1x2x2x256xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {0.0016649433210784313, 0.0026649433210784313, 0.0036649433210784313, 0.0046649433210784313}>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateDequantTransposeWithQuantizeDimChanged
func.func @PropagateDequantTransposeWithQuantizeDimChanged(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x4x1x1xf16> {
  %0 = const.Declare tensor<1x128x1x4x!qElemType> = dense<1.0> : tensor<128x4xf16>, [#const.Reshape<[1, 128, 1, 4]>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x128x1x4x!qElemType> -> tensor<1x128x1x4xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 4]} : tensor<1x128x1x4xf16> -> tensor<1x1x128x4xf16>
  %3 = IE.Transpose(%2) {order_value = #NCWH} : tensor<1x1x128x4xf16> -> tensor<1x1x4x128xf16>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 128, 1, 1]} : tensor<1x1x4x128xf16> -> tensor<4x128x1x1xf16>
  %5 = IE.Convolution(%arg0, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>
  return %5 : tensor<256x4x1x1xf16>

  //CHECK: [[CONST:%.*]] = const.Declare tensor<4x128x1x1x!qElemType> = dense<1.000000e+00>
  //CHECK-SAME: : tensor<128x4xf16>, [#const.Reshape<[1, 128, 1, 4]>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reshape<[1, 1, 128, 4]>, #const.Transpose<#NCWH>, #const.ChangeShapeAndElemType<[4, 128, 1, 1], !qElemType>]
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<4x128x1x1x!qElemType> -> tensor<4x128x1x1xf16>
  //CHECK: [[VAL2:%.*]] = IE.Convolution(%arg0, [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<256x128x1x1xf16>, tensor<4x128x1x1xf16> -> tensor<256x4x1x1xf16>
  //CHECK: return [[VAL2]] : tensor<256x4x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateQuantTranspose
func.func @PropagateQuantTranspose(%arg0: tensor<1x256x2x2xf32>) -> tensor<1x2x2x256x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x256x2x2xf32> -> tensor<1x256x2x2xf16>
  %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x2x2x256xf16> -> tensor<1x2x2x256x!qElemType>

  return %2 : tensor<1x2x2x256x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x256x2x2xf16> -> tensor<1x256x2x2x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantExpandDilated
func.func @PropagateDequantExpandDilated(%arg0: tensor<1x9x3x3x!qElemType>) -> tensor<1x9x5x5xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9x5x5xf16>, tensor<1x9x5x5xf16> -> tensor<1x9x5x5xf16>

  return %3 : tensor<1x9x5x5xf16>

  //CHECK: [[VAL0:%.*]] = IE.ExpandDilated(%arg0) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x9x5x5x!qElemType> -> tensor<1x9x5x5xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantConvertExpandDilated
func.func @PropagateDequantConvertExpandDilated(%arg0: tensor<1x9x3x3x!qElemType>) -> tensor<1x9x5x5xf32> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x9x5x5xf16> -> tensor<1x9x5x5xf32>

  return %3 : tensor<1x9x5x5xf32>

  //CHECK: [[VAL0:%.*]] = IE.ExpandDilated(%arg0) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x9x5x5x!qElemType> -> tensor<1x9x5x5xf16>
  //CHECK: [[VAL2:%.*]] = IE.Convert
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantExpandDilated
func.func @PropagateQuantExpandDilated(%arg0: tensor<1x9x3x3xf32>) -> tensor<1x9x5x5x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x3x3xf32> -> tensor<1x9x3x3xf16>
  %1 = IE.ExpandDilated(%0) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x5x5xf16> -> tensor<1x9x5x5x!qElemType>

  return %2 : tensor<1x9x5x5x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x3x3xf16> -> tensor<1x9x3x3x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.ExpandDilated([[VAL0]]) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagateDequantConstPerAxisReshape
func.func @PropagateDequantConstPerAxisReshape() -> tensor<3x1x1x1xf16> {
  %0 = const.Declare tensor<1x3x1x1x!qElemType1> = dense<1.0> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x1x1x!qElemType1> -> tensor<1x3x1x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [3, 1, 1, 1], dim_mapping = [[0], [0], [1], [2, 3]]} : tensor<1x3x1x1xf16> -> tensor<3x1x1x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>

  return %3 : tensor<3x1x1x1xf16>

  //CHECK: [[CONST:%.*]] =  const.Declare tensor<3x1x1x1x!qElemType> = dense<1.000000e+00> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.ChangeShapeAndElemType<[3, 1, 1, 1], !qElemType>]
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<3x1x1x1x!qElemType> -> tensor<3x1x1x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<3x1x1x1xf16>
}

// -----

!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagationDequantPerAxisReshapeOneToZeroAxis
func.func @PropagationDequantPerAxisReshapeOneToZeroAxis(%arg0: tensor<1x3x2x1x!qElemType>) -> tensor<3x2x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x2x1x!qElemType> -> tensor<1x3x2x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [3, 2, 1, 1], dim_mapping = [[0], [0], [1], [2, 3]]} : tensor<1x3x2x1xf16> -> tensor<3x2x1x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x2x1x1xf16>, tensor<3x2x1x1xf16> -> tensor<3x2x1x1xf16>

  return %3 : tensor<3x2x1x1xf16>

  //CHECK: [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [3, 2, 1, 1]} : tensor<1x3x2x1x!qElemType> -> tensor<3x2x1x1x!qElemType1>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[RESHAPE]]) {dstElemType = f16} : tensor<3x2x1x1x!qElemType1> -> tensor<3x2x1x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<3x2x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisReshape
func.func @NoPropagationDequantPerAxisReshape(%arg0: tensor<1x3x1x2x!qElemType>) -> tensor<1x2x1x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %2 = IE.Reshape(%1) {shape_value = [1, 2, 1, 3]} : tensor<1x3x1x2xf16> -> tensor<1x2x1x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x3xf16>, tensor<1x2x1x3xf16> -> tensor<1x2x1x3xf16>

  return %3 : tensor<1x2x1x3xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  //CHECK: [[RESHAPE:%.*]] = IE.Reshape([[DEQUANT]]) {shape_value = [1, 2, 1, 3]} : tensor<1x3x1x2xf16> -> tensor<1x2x1x3xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<1x2x1x3xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagationDequantPerAxisReshapeZeroToOneAxis
func.func @PropagationDequantPerAxisReshapeZeroToOneAxis(%arg0: tensor<3x2x1x1x!qElemType>) -> tensor<1x3x2x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<3x2x1x1x!qElemType> -> tensor<3x2x1x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 3, 2, 1], dim_mapping = [[0, 1], [2], [3], [3]]} : tensor<3x2x1x1xf16> -> tensor<1x3x2x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x2x1xf16>, tensor<1x3x2x1xf16> -> tensor<1x3x2x1xf16>

  return %3 : tensor<1x3x2x1xf16>

  //CHECK: [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 3, 2, 1]} : tensor<3x2x1x1x!qElemType> -> tensor<1x3x2x1x!qElemType1>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x2x1x!qElemType1> -> tensor<1x3x2x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<1x3x2x1xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagateDequantConstPerAxisExpandDilated
func.func @PropagateDequantConstPerAxisExpandDilated() -> tensor<1x3x5x5xf16> {
  %0 = const.Declare tensor<1x3x3x3x!qElemType> = dense<1.0> : tensor<1x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x3x3x!qElemType> -> tensor<1x3x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x5xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x5x5xf16>, tensor<1x3x5x5xf16> -> tensor<1x3x5x5xf16>

  return %3 : tensor<1x3x5x5xf16>

  //CHECK: [[CONST:%.*]] =  const.Declare tensor<1x3x5x5x!qElemType> = dense<1.000000e+00> : tensor<1x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.ExpandDilated<[2, 2]>]
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<1x3x5x5x!qElemType> -> tensor<1x3x5x5xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<1x3x5x5xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

// CHECK-LABEL: @PropagateQuantAffineReshapeTranspose
func.func @PropagateQuantAffineReshapeTranspose(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4x!qElemType> {
  %0 = IE.AffineReshape(%arg0) {shape_value = [1, 3, 4, 1, 4, 1], dim_mapping = [[0], [1], [2, 3], [4, 5]]} : tensor<1x3x4x4xf16> -> tensor<1x3x4x1x4x1xf16>
  %1 = IE.Transpose(%0) {order_value = #map} : tensor<1x3x4x1x4x1xf16> -> tensor<1x1x1x3x4x4xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 3, 4, 4], dim_mapping = [[0], [0], [0], [1], [2], [3]]} : tensor<1x1x1x3x4x4xf16> -> tensor<1x3x4x4xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4x!qElemType>

  return %3 : tensor<1x3x4x4x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4x!qElemType>
  //CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape([[QUANTIZE]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [4, 5]], shape_value = [1, 3, 4, 1, 4, 1]} : tensor<1x3x4x4x!qElemType> -> tensor<1x3x4x1x4x1x!qElemType>
  //CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map}
  //CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1], [2], [3]], shape_value = [1, 3, 4, 4]} : tensor<1x1x1x3x4x4x!qElemType> -> tensor<1x3x4x4x!qElemType>
  //CHECK: return [[RESHAPE1]] : tensor<1x3x4x4x!qElemType>
}

// ######## CONCAT TEST ########

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcat
func.func @PerTensorConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxisConcat
func.func @PerAxisQuantOtherAxisConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxisConcat
func.func @PerAxisQuantSameAxisConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    // expected-error@+1 {{Misaligned element types}}
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----
!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @NoPropagationPerTensorQuantConcat
func.func @NoPropagationPerTensorQuantConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @QuantizePropagationAndNoDequantizePropagationPerTensorQuantConcat
func.func @QuantizePropagationAndNoDequantizePropagationPerTensorQuantConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType>
    return %3 : tensor<1x4x3x4x!qElemType>

    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[QUANTIZE1:%.*]] = IE.Quantize([[DEQUANTIZE1]]) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, [[QUANTIZE1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x4x3x4x!qElemType>

    //CHECK: return [[CONCAT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantOtherAxisConcat
func.func @NoPropagationPerAxisQuantOtherAxisConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x2x6x4x!qElemType> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x2x6x4xf16> -> tensor<1x2x6x4x!qElemType>
    return %3 : tensor<1x2x6x4x!qElemType>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONCAT]])
    //CHECK: return [[QUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantSameAxisConcat
func.func @NoPropagationPerAxisQuantSameAxisConcat(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType2> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType2} : tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType2>
    return %3 : tensor<1x4x3x4x!qElemType2>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONCAT]])
    //CHECK: return [[QUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcatOffsets
func.func @PerTensorConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxisConcatOffsets
func.func @PerAxisQuantOtherAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxisConcatOffsets
func.func @PerAxisQuantSameAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @NoPropagationPerTensorConcatOffsets
func.func @NoPropagationPerTensorConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantOtherAxisConcatOffsets
func.func @NoPropagationPerAxisQuantOtherAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantSameAxisConcatOffsets
func.func @NoPropagationPerAxisQuantSameAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]])
    //CHECK: return [[ADD]]
}

// ######## CONCAT TEST ########

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantClamp
func.func @PropagateDequantClamp(%arg0: tensor<1x256x1x1x!qElemType>) -> tensor<1x256x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  %2 = IE.Clamp(%1) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>
  %3 = IE.Clamp(%1) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>
  %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>

  return %4 : tensor<1x256x1x1xf16>

  //CHECK: [[CLAMP1:%.*]] = IE.Clamp(%arg0) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize([[CLAMP1]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>

  //CHECK: [[CLAMP2:%.*]] = IE.Clamp(%arg0) {max = 5.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANTIZE2:%.*]] = IE.Dequantize([[CLAMP2]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>

  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE1]], [[DEQUANTIZE2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>

  //CHECK: return [[ADD]] : tensor<1x256x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantClamp
func.func @PropagateQuantClamp(%arg0: tensor<1x9x1x1xf32>) -> (tensor<1x9x1x1x!qElemType>, tensor<1x9x1x1x!qElemType>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1x1xf32> -> tensor<1x9x1x1xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>

  return %2, %3 : tensor<1x9x1x1x!qElemType>, tensor<1x9x1x1x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1x1xf32> -> tensor<1x9x1x1xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[QUANTIZE]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x9x1x1x!qElemType> -> tensor<1x9x1x1x!qElemType>

  //CHECK: return [[CLAMP]], [[CLAMP]] : tensor<1x9x1x1x!qElemType>, tensor<1x9x1x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisClamp
func.func @NoPropagationDequantPerAxisClamp(%arg0: tensor<1x3x1x2x!qElemType>) -> tensor<1x3x1x2xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>

  return %2 : tensor<1x3x1x2xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0)
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[DEQUANT]])
  //CHECK: [[ADD:%.*]] = IE.Add([[CLAMP]], [[CLAMP]])
  //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisClamp
func.func @NoPropagationDequantPerAxisClamp(%arg0: tensor<1x3x1x2xf32>) -> (tensor<1x3x1x2x!qElemType>, tensor<1x3x1x2x!qElemType>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1x2xf32> -> tensor<1x3x1x2xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2x!qElemType>

  return %2, %3 : tensor<1x3x1x2x!qElemType>, tensor<1x3x1x2x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert(%arg0)
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[CONVERT]])
  //CHECK: [[QUANTIZE0:%.*]] = IE.Quantize([[CLAMP]])
  //CHECK: [[QUANTIZE1:%.*]] = IE.Quantize([[CLAMP]])
  //CHECK: return [[QUANTIZE0]], [[QUANTIZE1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantSplit
func.func @PropagateDequantSplit(%arg0: tensor<1x3x30x30x!qElemType>) -> (tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>) {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30xf16>
  %1:3 = IE.Split(%0) {axis_value = 1, num_splits = 3} : tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
  %2 = IE.Add(%1#0, %1#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
  %3 = IE.Add(%1#1, %1#1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
  %4 = IE.Add(%1#2, %1#2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>

 return %2, %3, %4 : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

 //CHECK: [[SPLIT:%.*]]:3 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 3 : i64} : tensor<1x3x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>
 //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[SPLIT]]#2) {dstElemType = f16} : tensor<1x1x30x30x!qElemType> -> tensor<1x1x30x30xf16>
 //CHECK: [[DEQUANT1:%.*]] = IE.Dequantize([[SPLIT]]#1) {dstElemType = f16} : tensor<1x1x30x30x!qElemType> -> tensor<1x1x30x30xf16>
 //CHECK: [[DEQUANT2:%.*]] = IE.Dequantize([[SPLIT]]#0) {dstElemType = f16} : tensor<1x1x30x30x!qElemType> -> tensor<1x1x30x30xf16>
 //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT2]], [[DEQUANT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
 //CHECK: [[ADD1:%.*]] = IE.Add([[DEQUANT1]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
 //CHECK: [[ADD2:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
 //CHECK: return [[ADD]], [[ADD1]], [[ADD2]] : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisSplit
func.func @NoPropagationDequantPerAxisSplit(%arg0: tensor<1x3x1x2x!qElemType>) -> (tensor<1x3x1x2xf16>) {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %1:1 = IE.Split(%0) {axis_value = 1, num_splits = 1} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  %2 = IE.Add(%1#0, %1#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>

  return %2 : tensor<1x3x1x2xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  //CHECK: [[SPLIT:%.*]] = IE.Split([[DEQUANT]]) {axis_value = 1 : i64, num_splits = 1 : i64} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  //CHECK: [[ADD:%.*]] = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  //CHECK: return [[ADD]] : tensor<1x3x1x2xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantSplit
func.func @PropagateQuantSplit(%arg0: tensor<1x3x30x30xf16>) -> (tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>) {
  %0:3 = IE.Split(%arg0) {axis_value = 1, num_splits = 3} : tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
  %1 = IE.Quantize(%0#0) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
  %2 = IE.Quantize(%0#1) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
  %3 = IE.Quantize(%0#2) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>

 return %1, %2, %3 : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>

 //CHECK: [[QUANT:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30x!qElemType>
 //CHECK: [[SPLIT:%.*]]:3 = IE.Split([[QUANT]]) {axis_value = 1 : i64, num_splits = 3 : i64} : tensor<1x3x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>
 //CHECK: return [[SPLIT]]#0, [[SPLIT]]#1, [[SPLIT]]#2 : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @FuseQuantParamsIntoSplit
func.func @FuseQuantParamsIntoSplit(%arg0: tensor<1x2x3x4xf16>) -> (tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>) {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2:2 = IE.Split(%1) {num_splits = 2, axis_value = 1} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>

    %3 = IE.Quantize(%2#0) {dstElemType = !qElemType}: tensor<1x1x3x4xf16> -> tensor<1x1x3x4x!qElemType>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>

    %5 = IE.Quantize(%2#1) {dstElemType = !qElemType}: tensor<1x1x3x4xf16> -> tensor<1x1x3x4x!qElemType>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>

    return %4, %6 : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[VAL1:%.*]]:2 = IE.Split([[VAL0]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4x!qElemType> -> tensor<1x1x3x4x!qElemType>, tensor<1x1x3x4x!qElemType>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]#0) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>
    //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL1]]#1) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>
    //CHECK: return [[VAL2]], [[VAL3]] : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>
!qElemType1 = !quant.uniform<i8:f16, 0.0016544117647058823>

// CHECK-LABEL: @NoPropagateQuantSplitDifferentTypes
func.func @NoPropagateQuantSplitDifferentTypes(%arg0: tensor<1x2x30x30xf16>) -> (tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType1>) {
  %0:2 = IE.Split(%arg0) {axis_value = 1, num_splits = 2} : tensor<1x2x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
  %1 = IE.Quantize(%0#0) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
  %2 = IE.Quantize(%0#1) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>

 return %1, %2 : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType1>

 //CHECK: [[SPLIT:%.*]]:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
 //CHECK: [[QUANT0:%.*]] = IE.Quantize([[SPLIT]]#0) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
 //CHECK: [[QUANT1:%.*]] = IE.Quantize([[SPLIT]]#1) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
 //CHECK: return [[QUANT0]], [[QUANT1]] : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType1>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantTileShapeRepeatsEqualShapeData
func.func @PropagateDequantTileShapeRepeatsEqualShapeData(%arg0: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x9x16xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
  %1 = IE.Tile(%0) {repeats_values = [1, 2, 3, 4]} : tensor<1x2x3x4xf16> -> tensor<1x4x9x16xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x9x16xf16>, tensor<1x4x9x16xf16> -> tensor<1x4x9x16xf16>

 return %2 : tensor<1x4x9x16xf16>

  //CHECK: [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 2, 3, 4]} : tensor<1x2x3x4x!qElemType> -> tensor<1x4x9x16x!qElemType>
  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[TILE]]) {dstElemType = f16} : tensor<1x4x9x16x!qElemType> -> tensor<1x4x9x16xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x9x16xf16>, tensor<1x4x9x16xf16> -> tensor<1x4x9x16xf16>
  //CHECK: return [[ADD]] : tensor<1x4x9x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantTileShapeRepeatsLessThenShapeData
func.func @PropagateDequantTileShapeRepeatsLessThenShapeData(%arg0: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x12xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
  %1 = IE.Tile(%0) {repeats_values = [1, 2, 3]} : tensor<1x2x3x4xf16> -> tensor<1x2x6x12xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x12xf16>, tensor<1x2x6x12xf16> -> tensor<1x2x6x12xf16>

 return %2 : tensor<1x2x6x12xf16>

  //CHECK: [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 2, 3]} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x12x!qElemType>
  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[TILE]]) {dstElemType = f16} : tensor<1x2x6x12x!qElemType> -> tensor<1x2x6x12xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x12xf16>, tensor<1x2x6x12xf16> -> tensor<1x2x6x12xf16>
  //CHECK: return [[ADD]] : tensor<1x2x6x12xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantTileShapeRepeatsMoreThenShapeData
func.func @PropagateDequantTileShapeRepeatsMoreThenShapeData(%arg0: tensor<1x2x3x!qElemType>) -> tensor<1x2x6x12xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x!qElemType> -> tensor<1x2x3xf16>
  %1 = IE.Tile(%0) {repeats_values = [1, 2, 3, 4]} : tensor<1x2x3xf16> -> tensor<1x2x6x12xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x12xf16>, tensor<1x2x6x12xf16> -> tensor<1x2x6x12xf16>

 return %2 : tensor<1x2x6x12xf16>

  //CHECK: [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 2, 3, 4]} : tensor<1x2x3x!qElemType> -> tensor<1x2x6x12x!qElemType>
  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[TILE]]) {dstElemType = f16} : tensor<1x2x6x12x!qElemType> -> tensor<1x2x6x12xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x12xf16>, tensor<1x2x6x12xf16> -> tensor<1x2x6x12xf16>
  //CHECK: return [[ADD]] : tensor<1x2x6x12xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantTileShapeRepeatsEqualShapeData
func.func @PropagateQuantTileShapeRepeatsEqualShapeData(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x4x9x16x!qElemType> {
  %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 3, 4]}  : tensor<1x2x3x4xf16> -> tensor<1x4x9x16xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x4x9x16xf16> -> tensor<1x4x9x16x!qElemType>

  return %1 : tensor<1x4x9x16x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
  //CHECK: [[TILE:%.*]] = IE.Tile([[QUANTIZE]]) {repeats_values = [1, 2, 3, 4]} : tensor<1x2x3x4x!qElemType> -> tensor<1x4x9x16x!qElemType>
  //CHECK: return [[TILE]] : tensor<1x4x9x16x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantTileShapeRepeatsLessThenShapeData
func.func @PropagateQuantTileShapeRepeatsLessThenShapeData(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x6x12x!qElemType> {
  %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 3]} : tensor<1x2x3x4xf16> -> tensor<1x2x6x12xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x2x6x12xf16> -> tensor<1x2x6x12x!qElemType>

  return %1 : tensor<1x2x6x12x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
  //CHECK: [[TILE:%.*]] = IE.Tile([[QUANTIZE]]) {repeats_values = [1, 2, 3]} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x12x!qElemType>
  //CHECK: return [[TILE]] : tensor<1x2x6x12x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantTileShapeRepeatsMoreThenShapeData
func.func @PropagateQuantTileShapeRepeatsMoreThenShapeData(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x6x12x20x!qElemType> {
  %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 3, 4, 5]} : tensor<1x2x3x4xf16> -> tensor<1x2x6x12x20xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x2x6x12x20xf16> -> tensor<1x2x6x12x20x!qElemType>

  return %1 : tensor<1x2x6x12x20x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
  //CHECK: [[TILE:%.*]] = IE.Tile([[QUANTIZE]]) {repeats_values = [1, 2, 3, 4, 5]} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x12x20x!qElemType>
  //CHECK: return [[TILE]] : tensor<1x2x6x12x20x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotPropagateDequantTilePerChannel
func.func @DoNotPropagateDequantTilePerChannel(%arg0: tensor<1x4x16x16x!qElemType>) -> tensor<1x8x16x16xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
  %1 = IE.Tile(%0) {repeats_values = [1, 2, 1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x8x16x16xf16>
  %2 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x8x16x16xf16>, tensor<1x8x16x16xf16> -> tensor<1x8x16x16xf16>

  return %2 : tensor<1x8x16x16xf16>

  // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
  // CHECK: [[TILE:%.*]] = IE.Tile([[DEQUANTIZE]]) {repeats_values = [1, 2, 1, 1]} : tensor<1x4x16x16xf16> -> tensor<1x8x16x16xf16>
  // CHECK: [[ADD:%.*]] = IE.Add([[TILE]], [[TILE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x16x16xf16>, tensor<1x8x16x16xf16> -> tensor<1x8x16x16xf16>
  // CHECK: return [[ADD]] : tensor<1x8x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotPropagateQuantTilePerChannel
func.func @DoNotPropagateQuantTilePerChannel(%arg0: tensor<1x4x16x16xf16>) -> tensor<1x4x32x32x!qElemType> {
  %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 2, 2]} : tensor<1x4x16x16xf16> -> tensor<1x4x32x32xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x4x32x32xf16> -> tensor<1x4x32x32x!qElemType>

  return %1 : tensor<1x4x32x32x!qElemType>

  // CHECK: [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 1, 2, 2]} : tensor<1x4x16x16xf16> -> tensor<1x4x32x32xf16>
  // CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[TILE]]) {dstElemType = !qElemType} : tensor<1x4x32x32xf16> -> tensor<1x4x32x32x!qElemType>
  // CHECK: return [[QUANTIZE]] : tensor<1x4x32x32x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @PropagateDequantSqueeze
func.func @PropagateDequantSqueeze(%arg0: tensor<1x2x3x3x3x!qElemType>) -> tensor<2x3x3x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x3x3x!qElemType> -> tensor<1x2x3x3x3xf16>
  %2 = IE.Squeeze(%1) {axes_value = []} : tensor<1x2x3x3x3xf16> -> tensor<2x3x3x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x3xf16>, tensor<2x3x3x3xf16> -> tensor<2x3x3x3xf16>

  return %3 : tensor<2x3x3x3xf16>

  //CHECK: [[SQUEEZE:%.*]] = IE.Squeeze(%arg0) {axes_value = []} : tensor<1x2x3x3x3x!qElemType> -> tensor<2x3x3x3x!qElemType>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[SQUEEZE]]) {dstElemType = f16} : tensor<2x3x3x3x!qElemType> -> tensor<2x3x3x3xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x3xf16>, tensor<2x3x3x3xf16> -> tensor<2x3x3x3xf16>
  //CHECK: return [[ADD]] : tensor<2x3x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @PropagateQuantSqueeze
func.func @PropagateQuantSqueeze(%arg0: tensor<1x2x3x3xf16>) -> tensor<2x3x3x!qElemType> {
  %1 = IE.Squeeze(%arg0) {axes_value = []} : tensor<1x2x3x3xf16> -> tensor<2x3x3xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<2x3x3xf16> -> tensor<2x3x3x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x!qElemType>, tensor<2x3x3x!qElemType> -> tensor<2x3x3x!qElemType>

  return %3 : tensor<2x3x3x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x3xf16> -> tensor<1x2x3x3x!qElemType>
  //CHECK: [[SQUEEZE:%.*]] = IE.Squeeze([[QUANTIZE]]) {axes_value = []} : tensor<1x2x3x3x!qElemType> -> tensor<2x3x3x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[SQUEEZE]], [[SQUEEZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :  tensor<2x3x3x!qElemType>, tensor<2x3x3x!qElemType> -> tensor<2x3x3x!qElemType>
  //CHECK: return [[ADD]] : tensor<2x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @PropagateQuantUnsqueeze
func.func @PropagateQuantUnsqueeze(%arg0: tensor<1x2x3x3xf16>) -> tensor<1x1x2x3x3x!qElemType> {
  %1 = IE.Unsqueeze(%arg0) {axes_value = [0]} : tensor<1x2x3x3xf16> -> tensor<1x1x2x3x3xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x1x2x3x3xf16> -> tensor<1x1x2x3x3x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x3x3x!qElemType>, tensor<1x1x2x3x3x!qElemType> -> tensor<1x1x2x3x3x!qElemType>

  return %3 : tensor<1x1x2x3x3x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x3xf16> -> tensor<1x2x3x3x!qElemType>
  //CHECK: [[UNSQUEEZE:%.*]] = IE.Unsqueeze([[QUANTIZE]]) {axes_value = [0]} : tensor<1x2x3x3x!qElemType> -> tensor<1x1x2x3x3x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[UNSQUEEZE]], [[UNSQUEEZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x3x3x!qElemType>, tensor<1x1x2x3x3x!qElemType> -> tensor<1x1x2x3x3x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x1x2x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @PropagateDequantUnsqueeze
func.func @PropagateDequantUnsqueeze(%arg0: tensor<2x3x3x!qElemType>) -> tensor<1x2x3x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<2x3x3x!qElemType> -> tensor<2x3x3xf16>
  %2 = IE.Unsqueeze(%1) {axes_value = [0]} : tensor<2x3x3xf16> -> tensor<1x2x3x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x3xf16>, tensor<1x2x3x3xf16> -> tensor<1x2x3x3xf16>

  return %3 : tensor<1x2x3x3xf16>

   //CHECK: [[UNSQUEEZE:%.*]] = IE.Unsqueeze(%arg0) {axes_value = [0]} : tensor<2x3x3x!qElemType> -> tensor<1x2x3x3x!qElemType>
   //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[UNSQUEEZE]]) {dstElemType = f16} : tensor<1x2x3x3x!qElemType> -> tensor<1x2x3x3xf16>
   //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x3xf16>, tensor<1x2x3x3xf16> -> tensor<1x2x3x3xf16>
   //CHECK: return [[ADD]] : tensor<1x2x3x3xf16>
}


// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateDequantSqueezePerChannel
func.func @DoNotPropagateDequantSqueezePerChannel(%arg0: tensor<1x2x3x3x3x!qElemType>) -> tensor<2x3x3x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x3x3x!qElemType> -> tensor<1x2x3x3x3xf16>
  %2 = IE.Squeeze(%1) {axes_value = []} : tensor<1x2x3x3x3xf16> -> tensor<2x3x3x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x3xf16>, tensor<2x3x3x3xf16> -> tensor<2x3x3x3xf16>

  return %3 : tensor<2x3x3x3xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x3x3x!qElemType> -> tensor<1x2x3x3x3xf16>
  //CHECK: [[SQUEEZE:%.*]] = IE.Squeeze([[DEQUANT]]) {axes_value = []} : tensor<1x2x3x3x3xf16> -> tensor<2x3x3x3xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[SQUEEZE]], [[SQUEEZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x3xf16>, tensor<2x3x3x3xf16> -> tensor<2x3x3x3xf16>
  //CHECK: return [[ADD]] : tensor<2x3x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateQuantSqueezePerChannel
func.func @DoNotPropagateQuantSqueezePerChannel(%arg0: tensor<1x2x3x3xf16>) -> tensor<2x3x3x!qElemType> {
  %1 = IE.Squeeze(%arg0) {axes_value = []} : tensor<1x2x3x3xf16> -> tensor<2x3x3xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<2x3x3xf16> -> tensor<2x3x3x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x!qElemType>, tensor<2x3x3x!qElemType> -> tensor<2x3x3x!qElemType>

  return %3 : tensor<2x3x3x!qElemType>

  //CHECK: [[SQUEEZE:%.*]] = IE.Squeeze(%arg0) {axes_value = []} : tensor<1x2x3x3xf16> -> tensor<2x3x3xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[SQUEEZE]]) {dstElemType = !qElemType} : tensor<2x3x3xf16> -> tensor<2x3x3x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x3x!qElemType>, tensor<2x3x3x!qElemType> -> tensor<2x3x3x!qElemType>
  //CHECK: return [[ADD]] : tensor<2x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127}>

// CHECK-LABEL: @DoNotPropagateQuantUnsqueezePerChannel
func.func @DoNotPropagateQuantUnsqueezePerChannel(%arg0: tensor<1x2x3x3xf16>) -> tensor<1x1x2x3x3x!qElemType> {
  %1 = IE.Unsqueeze(%arg0) {axes_value = [0]} : tensor<1x2x3x3xf16> -> tensor<1x1x2x3x3xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x1x2x3x3xf16> -> tensor<1x1x2x3x3x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x3x3x!qElemType>, tensor<1x1x2x3x3x!qElemType> -> tensor<1x1x2x3x3x!qElemType>

  return %3 : tensor<1x1x2x3x3x!qElemType>

  //CHECK: [[UNSQUEEZE:%.*]] = IE.Unsqueeze(%arg0) {axes_value = [0]} : tensor<1x2x3x3xf16> -> tensor<1x1x2x3x3xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[UNSQUEEZE]]) {dstElemType = !qElemType} : tensor<1x1x2x3x3xf16> -> tensor<1x1x2x3x3x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x3x3x!qElemType>, tensor<1x1x2x3x3x!qElemType> -> tensor<1x1x2x3x3x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x1x2x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateDequantUnsqueezePerChannel
func.func @DoNotPropagateDequantUnsqueezePerChannel(%arg0: tensor<2x3x3x!qElemType>) -> tensor<1x2x3x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<2x3x3x!qElemType> -> tensor<2x3x3xf16>
  %2 = IE.Unsqueeze(%1) {axes_value = [0]} : tensor<2x3x3xf16> -> tensor<1x2x3x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x3xf16>, tensor<1x2x3x3xf16> -> tensor<1x2x3x3xf16>

  return %3 : tensor<1x2x3x3xf16>

   //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<2x3x3x!qElemType> -> tensor<2x3x3xf16>
   //CHECK: [[UNSQUEEZE:%.*]] = IE.Unsqueeze([[DEQUANT]]) {axes_value = [0]} : tensor<2x3x3xf16> -> tensor<1x2x3x3xf16>
   //CHECK: [[ADD:%.*]] = IE.Add([[UNSQUEEZE]], [[UNSQUEEZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x3xf16>, tensor<1x2x3x3xf16> -> tensor<1x2x3x3xf16>
   //CHECK: return [[ADD]] : tensor<1x2x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateQuantD2SPerChannel
func.func @DoNotPropagateQuantD2SPerChannel(%arg0: tensor<1x12x180x320xf16>) -> tensor<1x3x360x640x!qElemType> {
  %1 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x3x360x640xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x3x360x640xf16> -> tensor<1x3x360x640x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x360x640x!qElemType>, tensor<1x3x360x640x!qElemType> -> tensor<1x3x360x640x!qElemType>

  return %3 : tensor<1x3x360x640x!qElemType>

  //CHECK: [[VAL0:%.*]] = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x3x360x640xf16>
  //CHECK: [[VAL1:%.*]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x3x360x640xf16> -> tensor<1x3x360x640x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL1]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x360x640x!qElemType>, tensor<1x3x360x640x!qElemType> -> tensor<1x3x360x640x!qElemType>
  //CHECK: return [[VAL2]] : tensor<1x3x360x640x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateDequantD2SPerChannel
func.func @DoNotPropagateDequantD2SPerChannel(%arg0: tensor<1x12x180x320x!qElemType>) -> tensor<1x3x360x640xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x180x320x!qElemType> -> tensor<1x12x180x320xf16>
  %2 = IE.DepthToSpace(%1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x3x360x640xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x360x640xf16>, tensor<1x3x360x640xf16> -> tensor<1x3x360x640xf16>

  return %3 : tensor<1x3x360x640xf16>

  //CHECK: [[VAL0:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x180x320x!qElemType> -> tensor<1x12x180x320xf16>
  //CHECK: [[VAL1:%.*]] = IE.DepthToSpace([[VAL0]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x3x360x640xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL1]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x360x640xf16>, tensor<1x3x360x640xf16> -> tensor<1x3x360x640xf16>
  //CHECK: return [[VAL2]] : tensor<1x3x360x640xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantClampNon4DTensor
func.func @PropagateQuantClampNon4DTensor(%arg0: tensor<1x9x1xf32>) -> tensor<1x9x1x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1xf32> -> tensor<1x9x1xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x9x1xf16> -> tensor<1x9x1xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x1xf16> -> tensor<1x9x1x!qElemType>

  return %2 : tensor<1x9x1x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1xf32> -> tensor<1x9x1xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x1xf16> -> tensor<1x9x1x!qElemType>
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[QUANTIZE]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x9x1x!qElemType> -> tensor<1x9x1x!qElemType>

  //CHECK: return [[CLAMP]] : tensor<1x9x1x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDeQuantClampNon4DTensor
func.func @PropagateDeQuantClampNon4DTensor(%arg0: tensor<1x9x1x!qElemType>) -> tensor<1x9x1xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x1x!qElemType> -> tensor<1x9x1xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x9x1xf16> -> tensor<1x9x1xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9x1xf16>, tensor<1x9x1xf16> -> tensor<1x9x1xf16>

  return %2 : tensor<1x9x1xf16>

  //CHECK: [[CLAMP:%.*]] = IE.Clamp(%arg0) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x9x1x!qElemType> -> tensor<1x9x1x!qElemType>
  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CLAMP]]) {dstElemType = f16} : tensor<1x9x1x!qElemType> -> tensor<1x9x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9x1xf16>, tensor<1x9x1xf16> -> tensor<1x9x1xf16>

  //CHECK: return [[ADD]] : tensor<1x9x1xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128}>

// CHECK-LABEL: @DoNotPropagateDequantReduceMaxPerChannel
func.func @DoNotPropagateDequantReduceMaxPerChannel(%arg0: tensor<1x1x1x50x!qElemType>) -> tensor<1x1x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x1x50x!qElemType> -> tensor<1x1x1x50xf16>
  %2 = IE.ReduceMax(%1) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

  return %3 : tensor<1x1x1x1xf16>

  // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x1x50x!qElemType> -> tensor<1x1x1x50xf16>
  // CHECK: [[REDUCEMAX:%.*]] = IE.ReduceMax([[DEQUANTIZE]]) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK: [[ADD:%.*]] = IE.Add([[REDUCEMAX]], [[REDUCEMAX]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK: return [[ADD]] : tensor<1x1x1x1xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128}>

 // CHECK-LABEL: @DoNotPropagateQuantReduceMaxPerChannel
func.func @DoNotPropagateQuantReduceMaxPerChannel(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1x!qElemType> {
  %1 = IE.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType}: tensor<1x1x1x1xf16> -> tensor<1x1x1x1x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x1x1x1x!qElemType>, tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1x!qElemType>

  return %3 : tensor<1x1x1x1x!qElemType>

  // CHECK: [[REDUCEMAX:%.*]] = IE.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[REDUCEMAX]]) {dstElemType = !qElemType} : tensor<1x1x1x1xf16> -> tensor<1x1x1x1x!qElemType>
  // CHECK: [[ADD:%.*]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1x!qElemType>, tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1x!qElemType>
  // CHECK: return [[ADD]] : tensor<1x1x1x1x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantExpand
func.func @PropagateDequantExpand(%arg0: tensor<1x9x3x3x!qElemType>) -> tensor<1x9x5x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x3x3xf16>
  %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9x5x3xf16>, tensor<1x9x5x3xf16> -> tensor<1x9x5x3xf16>

  return %3 : tensor<1x9x5x3xf16>

  //CHECK: [[VAL0:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x3x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x9x5x3x!qElemType> -> tensor<1x9x5x3xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantExpand
func.func @PropagateQuantExpand(%arg0: tensor<1x9x3x3xf32>) -> tensor<1x9x5x3x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x3x3xf32> -> tensor<1x9x3x3xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x3xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x5x3xf16> -> tensor<1x9x5x3x!qElemType>

  return %2 : tensor<1x9x5x3x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x3x3xf16> -> tensor<1x9x3x3x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Expand([[VAL0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x3x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateDequantExpandPerChannel
func.func @DoNotPropagateDequantExpandPerChannel(%arg0: tensor<1x3x3x3x!qElemType>) -> tensor<1x3x5x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x3x3x!qElemType> -> tensor<1x3x3x3xf16>
  %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x5x3xf16>, tensor<1x3x5x3xf16> -> tensor<1x3x5x3xf16>

  return %3 : tensor<1x3x5x3xf16>

  //CHECK: [[VAL0:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x3x3x!qElemType> -> tensor<1x3x3x3xf16>
  //CHECK: [[VAL1:%.*]] = IE.Expand([[VAL0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x3xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @DoNotPropagateQuantExpandPerChannel
func.func @DoNotPropagateQuantExpandPerChannel(%arg0: tensor<1x3x3x3xf32>) -> tensor<1x3x5x3x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x3x3xf32> -> tensor<1x3x3x3xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x3xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x3x5x3xf16> -> tensor<1x3x5x3x!qElemType>

  return %2 : tensor<1x3x5x3x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Expand([[CONVERT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x3xf16>
  //CHECK: [[VAL1:%.*]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x3x5x3xf16> -> tensor<1x3x5x3x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateDequantInterpolate
func.func @DoNotPropagateDequantInterpolate(%arg0: tensor<1x1x48x80x!qElemType>) -> tensor<1x1x96x160xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x48x80x!qElemType> -> tensor<1x1x48x80xf16>
  %2 = IE.Interpolate(%1) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x1x96x160xf16>, tensor<1x1x96x160xf16> -> tensor<1x1x96x160xf16>

  return %3 : tensor<1x1x96x160xf16>

  //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x48x80x!qElemType> -> tensor<1x1x48x80xf16>
  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate([[DEQUANTIZE]]) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[INTERPOLATE]], [[INTERPOLATE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x96x160xf16>, tensor<1x1x96x160xf16> -> tensor<1x1x96x160xf16>
  //CHECK: return [[ADD]] : tensor<1x1x96x160xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateQuantInterpolate
func.func @DoNotPropagateQuantInterpolate(%arg0: tensor<1x1x48x80xf16>) -> tensor<1x1x96x160x!qElemType> {
  %1 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x1x96x160xf16> -> tensor<1x1x96x160x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x1x96x160x!qElemType>, tensor<1x1x96x160x!qElemType> -> tensor<1x1x96x160x!qElemType>

  return %3 : tensor<1x1x96x160x!qElemType>

  //CHECK: [[INTERPOLATE:%.*]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[INTERPOLATE]]) {dstElemType = !qElemType} : tensor<1x1x96x160xf16> -> tensor<1x1x96x160x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[QUANTIZE]], [[QUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x96x160x!qElemType>, tensor<1x1x96x160x!qElemType> -> tensor<1x1x96x160x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x1x96x160x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantStridedSlice
func.func @PropagateDequantStridedSlice(%arg0: tensor<1x2x32x64x!qElemType>) -> tensor<1x2x15x64xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x32x64x!qElemType> -> tensor<1x2x32x64xf16>
  %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 17, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 2, 32, 64], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x2x32x64xf16> -> tensor<1x2x15x64xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x15x64xf16>, tensor<1x2x15x64xf16> -> tensor<1x2x15x64xf16>

  return %2 : tensor<1x2x15x64xf16>

  //CHECK: [[SS0:%.*]] =  IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 17, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 2, 32, 64], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x2x32x64x!qElemType> -> tensor<1x2x15x64x!qElemType>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[SS0]]) {dstElemType = f16} : tensor<1x2x15x64x!qElemType> -> tensor<1x2x15x64xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x15x64xf16>, tensor<1x2x15x64xf16> -> tensor<1x2x15x64xf16>
  //CHECK: return [[ADD]] : tensor<1x2x15x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantStridedSlice
func.func @PropagateQuantStridedSlice(%arg0: tensor<1x32x64x4xf16>) -> tensor<1x32x17x4x!qElemType> {
  %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 64, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x32x64x4xf16> -> tensor<1x32x17x4xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType>

  return %1 : tensor<1x32x17x4x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x32x64x4xf16> -> tensor<1x32x64x4x!qElemType>
  //CHECK: [[SS0:%.*]] = IE.StridedSlice([[QUANTIZE]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 64, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x32x64x4x!qElemType> -> tensor<1x32x17x4x!qElemType>
  //CHECK: return [[SS0]] : tensor<1x32x17x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotPropagateDequantStridedSlicePerChannel
func.func @DoNotPropagateDequantStridedSlicePerChannel(%arg0: tensor<1x4x32x64x!qElemType>) -> tensor<1x4x15x64xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x32x64x!qElemType> -> tensor<1x4x32x64xf16>
  %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 17, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 4, 32, 64], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x4x32x64xf16> -> tensor<1x4x15x64xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x15x64xf16>, tensor<1x4x15x64xf16> -> tensor<1x4x15x64xf16>

  return %2 : tensor<1x4x15x64xf16>

  // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x32x64x!qElemType> -> tensor<1x4x32x64xf16>
  // CHECK: [[SS0:%.*]] = IE.StridedSlice([[DEQUANTIZE]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 17, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 4, 32, 64], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x4x32x64xf16> -> tensor<1x4x15x64xf16>
  // CHECK: [[ADD:%.*]] = IE.Add([[SS0]], [[SS0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x15x64xf16>, tensor<1x4x15x64xf16> -> tensor<1x4x15x64xf16>
  // CHECK: return [[ADD]] : tensor<1x4x15x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotPropagateQuantStridedSlicePerChannel
func.func @DoNotPropagateQuantStridedSlicePerChannel(%arg0: tensor<1x4x64x4xf16>) -> tensor<1x4x17x4x!qElemType> {
  %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 4, 64, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x4x64x4xf16> -> tensor<1x4x17x4xf16>
  %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x4x17x4xf16> -> tensor<1x4x17x4x!qElemType>

  return %1 : tensor<1x4x17x4x!qElemType>

  // CHECK: [[SS0:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 4, 64, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x4x64x4xf16> -> tensor<1x4x17x4xf16>
  // CHECK: [[QUANTIZE:%.*]] =  IE.Quantize([[SS0]]) {dstElemType = !qElemType} : tensor<1x4x17x4xf16> -> tensor<1x4x17x4x!qElemType>
  // CHECK: return [[QUANTIZE]] : tensor<1x4x17x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127
}>

// CHECK-LABEL-DAG: @PropagatePerAxisDequant
// CHECK-DAG:   [[Q_TYPE_AXIS_1:!.*]] = !quant.uniform<u8<0:254>:f16:1, {
// CHECK-DAG:   [[Q_TYPE_AXIS_3:!.*]] = !quant.uniform<u8<0:254>:f16:3, {

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @PropagatePerAxisDequant(%arg0: tensor<2x16x4x8x!qElemType>) -> tensor<2x4x8x16xf32> {
    %DEQUANT = IE.Dequantize(%arg0) {
        dstElemType = f16
    } : tensor<2x16x4x8x!qElemType> -> tensor<2x16x4x8xf16>

    // CHECK:   [[TRANSPOSE:%.*]] = IE.Transpose(%arg0) {
    // CHECK-SAME:      order_value = #NHWC
    // CHECK-SAME:  } : tensor<2x16x4x8x[[Q_TYPE_AXIS_1]]> -> tensor<2x4x8x16x[[Q_TYPE_AXIS_3]]>

    %TRANSPOSE = IE.Transpose(%DEQUANT) {
        order_value = #NHWC
    } : tensor<2x16x4x8xf16> -> tensor<2x4x8x16xf16>

    // CHECK:   [[DEQUANT:%.*]] = IE.Dequantize([[TRANSPOSE]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<2x4x8x16x[[Q_TYPE_AXIS_3]]> -> tensor<2x4x8x16xf16>

    %CONVERT = IE.Convert(%TRANSPOSE) {
        dstElemType = f32
    } : tensor<2x4x8x16xf16> -> tensor<2x4x8x16xf32>

    // CHECK:   [[CONVERT:%.*]] = IE.Convert([[DEQUANT]]) {
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:  } : tensor<2x4x8x16xf16> -> tensor<2x4x8x16xf32>

    return %CONVERT : tensor<2x4x8x16xf32>

    // CHECK:   return [[CONVERT]] : tensor<2x4x8x16xf32>
}

// -----
!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantMaxPoolNCE
func.func @PropagateDequantMaxPoolNCE(%arg0: tensor<1x8x1x4x!qElemType>) -> tensor<1x8x1x4xf16> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x8x1x4x!qElemType> -> tensor<1x8x1x4xf16>
    %2 = IE.MaxPool(%1) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x8x1x4xf16> -> tensor<1x8x1x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x8x1x4xf16>, tensor<1x8x1x4xf16> -> tensor<1x8x1x4xf16>

    return %3 : tensor<1x8x1x4xf16>

    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8x1x4x!qElemType> -> tensor<1x8x1x4x!qElemType>
    //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[MAXPOOL]]) {dstElemType = f16} : tensor<1x8x1x4x!qElemType> -> tensor<1x8x1x4xf16>
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x1x4xf16>, tensor<1x8x1x4xf16> -> tensor<1x8x1x4xf16>
    //CHECK: return [[ADD]] : tensor<1x8x1x4xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @QuantizePropagationMaxPoolNCE
func.func @QuantizePropagationMaxPoolNCE(%arg0: tensor<1x8x1x4xf16>) -> tensor<1x8x1x4x!qElemType> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x8x1x4xf16> -> tensor<1x8x1x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x8x1x4xf16> -> tensor<1x8x1x4x!qElemType>
    return %1 : tensor<1x8x1x4x!qElemType>

    //CHECK: [[QUANTIZE:%.*]]  = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x1x4xf16> -> tensor<1x8x1x4x!qElemType>
    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[QUANTIZE]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x8x1x4x!qElemType> -> tensor<1x8x1x4x!qElemType>
    //CHECK: return [[MAXPOOL]] : tensor<1x8x1x4x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DequantizePropagationReorder
func.func @DequantizePropagationReorder(%arg0: tensor<1x3x24x24x!qElemType, {order = #NCHW}>) -> tensor<1x3x24x24xf16, {order = #NHWC}> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x24x24x!qElemType, {order = #NCHW}> -> tensor<1x3x24x24xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x3x24x24xf16> -> tensor<1x3x24x24xf16, {order = #NHWC}>
  return %1 : tensor<1x3x24x24xf16, {order = #NHWC}>
  // CHECK: [[REORDER:%.*]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x24x24x!qElemType, {order = #NCHW}> -> tensor<1x3x24x24x!qElemType, {order = #NHWC}>
  // CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[REORDER]]) {dstElemType = f16} : tensor<1x3x24x24x!qElemType, {order = #NHWC}> -> tensor<1x3x24x24xf16, {order = #NHWC}>
  // CHECK: return [[DEQUANT]] : tensor<1x3x24x24xf16, {order = #NHWC}>

}
