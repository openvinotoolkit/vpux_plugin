//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --propagate-quantize-dequantize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// -----
!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @DoNotPropagateQuantD2S
func.func @DoNotPropagateQuantD2S(%arg0: tensor<1x12x180x320xf16>) -> tensor<1x3x360x640x!qElemType> {
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
!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @DoNotPropagateDequantD2S
func.func @DoNotPropagateDequantD2S(%arg0: tensor<1x12x180x320x!qElemType>) -> tensor<1x3x360x640xf16> {
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
//Test to be removed once E-80362 is solved

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DoNotPropagateDequantReduceMax
func.func @DoNotPropagateDequantReduceMax(%arg0: tensor<1x1x1x50x!qElemType>) -> tensor<1x1x1x1xf16> {
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
//Test to be removed once E-80362 is solved

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

 // CHECK-LABEL: @DoNotPropagateQuantReduceMax
func.func @DoNotPropagateQuantReduceMax(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1x!qElemType> {
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @DoNotPropagateDequantS2D
func.func @DoNotPropagateDequantS2D(%arg0: tensor<1x12x180x320x!qElemType>) -> tensor<1x192x45x80xf16, {order = #NHWC}> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x180x320x!qElemType> -> tensor<1x12x180x320xf16, {order = #NHWC}>
    %2 = IE.SpaceToDepthOp(%1) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x12x180x320xf16, {order = #NHWC}> -> tensor<1x192x45x80xf16, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192x45x80xf16, {order = #NHWC}>, tensor<1x192x45x80xf16, {order = #NHWC}> -> tensor<1x192x45x80xf16, {order = #NHWC}>

   return %3 : tensor<1x192x45x80xf16, {order = #NHWC}>

  //CHECK: [[VAL0:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x12x180x320x!qElemType> -> tensor<1x12x180x320xf16, {order = #NHWC}>
  //CHECK: [[VAL1:%.*]] = IE.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x12x180x320xf16, {order = #NHWC}> -> tensor<1x192x45x80xf16, {order = #NHWC}>
  //CHECK: [[VAL2:%.*]] =  IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192x45x80xf16, {order = #NHWC}>, tensor<1x192x45x80xf16, {order = #NHWC}> -> tensor<1x192x45x80xf16, {order = #NHWC}>
  //CHECK: return [[VAL2]] : tensor<1x192x45x80xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>

// CHECK-LABEL: @DoNotPropagateQuantS2D
func.func @DoNotPropagateQuantS2D(%arg0: tensor<1x12x180x320xf16>) -> tensor<1x192x45x80x!qElemType, {order = #NHWC}> {
    %1 = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x192x45x80xf16, {order = #NHWC}>
    %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x192x45x80xf16, {order = #NHWC}> -> tensor<1x192x45x80x!qElemType, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192x45x80x!qElemType, {order = #NHWC}>, tensor<1x192x45x80x!qElemType, {order = #NHWC}> -> tensor<1x192x45x80x!qElemType, {order = #NHWC}>
   return %3 : tensor<1x192x45x80x!qElemType, {order = #NHWC}>

  //CHECK: [[VAL1:%.*]] = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x12x180x320xf16> -> tensor<1x192x45x80xf16, {order = #NHWC}>
  //CHECK: [[VAL1:%.*]] =  IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x192x45x80xf16, {order = #NHWC}> -> tensor<1x192x45x80x!qElemType, {order = #NHWC}>
  //CHECK: [[VAL2:%.*]] =  IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192x45x80x!qElemType, {order = #NHWC}>, tensor<1x192x45x80x!qElemType, {order = #NHWC}> -> tensor<1x192x45x80x!qElemType, {order = #NHWC}>
  //CHECK: return [[VAL2]] : tensor<1x192x45x80x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @DontPropagateDequantMaxPoolWithLargeKernel
func.func @DontPropagateDequantMaxPoolWithLargeKernel(%arg0: tensor<1x512x19x19x!qElemType>) -> tensor<1x512x19x19xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19xf16>
  %2 = IE.MaxPool(%1) {
        kernel_size = [13, 13],
        pads_begin = [6, 6],
        pads_end = [6, 6],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x512x19x19xf16>, tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>

  return %3 : tensor<1x512x19x19xf16>

    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [13, 13], pads_begin = [6, 6], pads_end = [6, 6], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>
    //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[MAXPOOL]]) {dstElemType = f16} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19xf16>
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x19x19xf16>, tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
    //CHECK: return [[ADD]] : tensor<1x512x19x19xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

 // CHECK-LABEL: @DontPropagateQuantMaxPoolWithLargeKernel
func.func @DontPropagateQuantMaxPoolWithLargeKernel(%arg0: tensor<1x512x19x19xf16>) -> tensor<1x512x19x19x!qElemType> {
  %1 = IE.MaxPool(%arg0) {
        kernel_size = [13, 13],
        pads_begin = [6, 6],
        pads_end = [6, 6],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x512x19x19x!qElemType>, tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>

  return %3 : tensor<1x512x19x19x!qElemType>

    //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19x!qElemType>
    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[QUANTIZE]]) {kernel_size = [13, 13], pads_begin = [6, 6], pads_end = [6, 6], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>
    //CHECK: [[ADD:%.*]] = IE.Add([[MAXPOOL]], [[MAXPOOL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x19x19x!qElemType>, tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>
    //CHECK: return [[ADD]] : tensor<1x512x19x19x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @MaxPool5DNoQuantizePropagationMaxPool5D
func.func @MaxPool5DNoQuantizePropagationMaxPool5D(%arg0: tensor<1x64x40x112x112xf16>) -> tensor<1x64x40x112x112x!qElemType> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3, 3],
        pads_begin = [1, 1, 1],
        pads_end = [1, 1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1, 1]} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112x!qElemType>
    return %1 : tensor<1x64x40x112x112x!qElemType>

    //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112x!qElemType>
    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[QUANTIZE]]) {kernel_size = [3, 3, 3], pads_begin = [1, 1, 1], pads_end = [1, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1, 1]} : tensor<1x64x40x112x112x!qElemType> -> tensor<1x64x40x112x112x!qElemType>
    //CHECK: return [[MAXPOOL]] : tensor<1x64x40x112x112x!qElemType>
    
}

// -----
!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @MaxPool5DNoDequantPropagation
func.func @MaxPool5DNoDequantPropagation(%arg0: tensor<1x64x40x112x112x!qElemType>) -> tensor<1x64x40x112x112xf16> {
    %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x64x40x112x112x!qElemType> -> tensor<1x64x40x112x112xf16>
    %2 = IE.MaxPool(%1) {
        kernel_size = [3, 3, 3],
        pads_begin = [1, 1, 1],
        pads_end = [1, 1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1, 1]} : tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x64x40x112x112xf16>, tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>

    return %3 : tensor<1x64x40x112x112xf16>

    //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [3, 3, 3], pads_begin = [1, 1, 1], pads_end = [1, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1, 1]} : tensor<1x64x40x112x112x!qElemType> -> tensor<1x64x40x112x112x!qElemType>
    //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[MAXPOOL]]) {dstElemType = f16} : tensor<1x64x40x112x112x!qElemType> -> tensor<1x64x40x112x112xf16>
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x40x112x112xf16>, tensor<1x64x40x112x112xf16> -> tensor<1x64x40x112x112xf16>
    //CHECK: return [[ADD]] : tensor<1x64x40x112x112xf16

}
