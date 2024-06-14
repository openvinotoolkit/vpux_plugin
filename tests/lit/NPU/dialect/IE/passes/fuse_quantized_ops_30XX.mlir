//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

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
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL2:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL4:%.*]] = IE.Add([[VAL1]], [[VAL3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]]) {dstElemType = !qElemType2} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType2>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  //CHECK: return [[VAL6]] : tensor<1x16x180x320xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0034409466911764705>
!qElemType1 = !quant.uniform<u8:f16, 0.12503063725490196:128>
!qElemType2 = !quant.uniform<u8:f16, 0.067708337073232608:128>

// CHECK-LABEL: @FuseQuantParamsIntoEltwiseMul
func.func @FuseQuantParamsIntoEltwiseMul(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %5 = IE.Multiply(%2, %4) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %6 = IE.Quantize(%5) {dstElemType = !qElemType2}: tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
  %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x3x16x16x!qElemType2> -> tensor<1x3x16x16xf16>

  return %7 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.Multiply([[VAL0]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16x!qElemType>, tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16x!qElemType2>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType2> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL3]]
}
