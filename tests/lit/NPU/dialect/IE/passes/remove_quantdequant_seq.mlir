//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --remove-quantdequant-seq %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @RemoveQuantDequantSequence
func.func @RemoveQuantDequantSequence(%arg0: tensor<1x3x16x16xf16>) -> (tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>)  {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6, %4 : tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType1> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]
  //CHECK: [[VAL2:%.*]] = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType1> -> tensor<3x3x3x3xf16>

  //CHECK: [[VAL3:%.*]] = IE.Convolution(%arg0, [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]], [[VAL3]]
}

// CHECK-LABEL: @RemoveQuantReshapeDequantSequence
func.func @RemoveQuantReshapeDequantSequence(%arg0: tensor<1x4420x1x2xf16>, %arg1: tensor<1x4420x1x2xf16>) -> tensor<1x4420x1x2xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2x!qElemType1>
  %2 = IE.AffineReshape(%1) { dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 4420, 2] } : tensor<1x4420x1x2x!qElemType1> -> tensor<1x1x4420x2x!qElemType1>
  %3 = IE.AffineReshape(%2) { dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 4420, 1, 2] } : tensor<1x1x4420x2x!qElemType1> -> tensor<1x4420x1x2x!qElemType1>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x4420x1x2x!qElemType1> -> tensor<1x4420x1x2xf16>
  %5 = IE.Add(%4, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x4420x1x2xf16>, tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2xf16>
  return %5 : tensor<1x4420x1x2xf16>

  //CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 4420, 2]} : tensor<1x4420x1x2xf16> -> tensor<1x1x4420x2xf16>
  //CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 4420, 1, 2]} : tensor<1x1x4420x2xf16> -> tensor<1x4420x1x2xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL1]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4420x1x2xf16>, tensor<1x4420x1x2xf16> -> tensor<1x4420x1x2xf16>
  //CHECK: return [[VAL2]]
}

// CHECK-LABEL: @RemoveQuantReshapeDequantMaxPool
func.func @RemoveQuantReshapeDequantMaxPool(%arg0: tensor<1x64x40x112x112xf16>) -> tensor<1x64x40x112x112xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112x!qElemType1>
  %1 = IE.MaxPool(%0) {
        kernel_size = [3, 3, 3],
        pads_begin = [1, 1, 1],
        pads_end = [1, 1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1, 1]} : tensor<1x64x40x112x112x!qElemType1> -> tensor<1x64x40x112x112x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x64x40x112x112x!qElemType1> -> tensor<1x64x40x112x112xf16>
  return %2 : tensor<1x64x40x112x112xf16>

    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [3, 3, 3], pads_begin = [1, 1, 1], pads_end = [1, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1, 1]} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>
    //CHECK: return [[MAXPOOL]] : tensor<1x64x40x112x112xf16>
}

