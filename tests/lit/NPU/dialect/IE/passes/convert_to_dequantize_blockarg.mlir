//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-dequantize="enable-wd-blockarg-input=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: @ConvertToDequantize
// CHECK-SAME: ([[ARG0:%.+]]: tensor<64x64x1x1xsi8>)
func.func @ConvertToDequantize(%arg0: tensor<64x64x1x1xsi8>) -> tensor<64x64x1x1xf16> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1xf16>
  return %0 : tensor<64x64x1x1xf16>

  // CHECK:  [[VAL0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = !qElemType} : tensor<64x64x1x1xsi8> -> tensor<64x64x1x1x!qElemType>
  // CHECK:  [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1xf16>
  // CHECK:  return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE2:!.+]] = !quant.uniform<u4:f16, 0.0057189941406250002:8>
!qElemType = !quant.uniform<u4:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u4:f16, 0.0057189941406250002:8>

// CHECK:      ConvertToDequantizeForU4WeightsWithBranchingSlices
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<128x64x1x1xui4>)
func.func @ConvertToDequantizeForU4WeightsWithBranchingSlices(
    %arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<128x64x1x1xui4>)
    -> (tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<1x64x64x100xf16, {order = #NHWC}>) {
  %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<128x64x1x1xui4> -> tensor<128x64x1x1xf16>
  %1 = IE.Slice %0 [0, 0, 0, 0] [64, 64, 1, 1] : tensor<128x64x1x1xf16> to tensor<64x64x1x1xf16>
  %2 = IE.Slice %0 [64, 0, 0, 0] [64, 64, 1, 1] : tensor<128x64x1x1xf16> to tensor<64x64x1x1xf16>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<64x64x1x1xf16> -> tensor<64x64x1x1x!qElemType>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType1} : tensor<64x64x1x1x!qElemType> -> tensor<64x64x1x1x!qElemType1>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<64x64x1x1x!qElemType1> -> tensor<64x64x1x1xf16>
  %6 = IE.Convolution(%arg0, %5)
      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  %7 = IE.Convolution(%arg0, %2)
      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %6, %7 : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK-NOT: IE.Convert

  // CHECK:       [[VAL0:%.+]] = IE.QuantizeCast([[ARG1]])
  // CHECK-SAME:      {dstElemType = [[Q_ELEM_TYPE0]]}
  // CHECK-SAME:        : tensor<128x64x1x1xui4> -> tensor<128x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:       [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16}
  // CHECK-SAME:        : tensor<128x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<128x64x1x1xf16>

  // CHECK:       [[VAL2:%.+]] = IE.Slice [[VAL1]] [0, 0, 0, 0] [64, 64, 1, 1]
  // CHECK:       [[VAL3:%.+]] = IE.Slice [[VAL1]] [64, 0, 0, 0] [64, 64, 1, 1]

  // CHECK:       [[VAL4:%.+]] = IE.Quantize([[VAL2]]) {dstElemType = [[Q_ELEM_TYPE1]]}
  // CHECK:       [[VAL5:%.+]] = IE.QuantizeCast([[VAL4]]) {dstElemType = [[Q_ELEM_TYPE2]]}
  // CHECK:       [[VAL6:%.+]] = IE.Dequantize([[VAL5]]) {dstElemType = f16}

  // CHECK:       [[VAL7:%.+]] = IE.Convolution([[ARG0]], [[VAL6]])
  // CHECK-SAME:       : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:       [[VAL8:%.+]] = IE.Convolution([[ARG0]], [[VAL3]])
  // CHECK-SAME:       : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>

}
