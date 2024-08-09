//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --optimize-slice-expand %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSoftmaxExpand
module @OptimizeSliceSoftmaxExpand {
// CHECK-LABEL:       @fuseSliceSoftmaxExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}>
func.func @fuseSliceSoftmaxExpand(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x80x8x16xf16, {order = #NHWC}>

  // CHECK:       [[OUTPUT:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64, padSize = 3 : i64} :
  // CHECK-SAME:  tensor<1x80x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  // CHECK:       return [[OUTPUT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpand
module @notOptimizeSliceSoftmaxExpand {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x81x8x16xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpand(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x81x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x81x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x81x8x16xf16, {order = #NHWC}>

  // CHECK:    [[CUT_INPUT:%.+]] = IE.Slice [[INPUT]]
  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[CUT_INPUT]]) {axisInd = 1 : i64}
  // CHECK:    [[OUTPUT:%.+]] = IE.Expand([[OUT_SOFTMAX]])
  // CHECK:    return [[OUTPUT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpandAxisNotC
module @notOptimizeSliceSoftmaxExpandAxisNotC {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpandAxisNotC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpandAxisNotC(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x80x8x16xf16, {order = #NHWC}>

  // CHECK:    [[CUT_INPUT:%.+]] = IE.Slice [[INPUT]]
  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[CUT_INPUT]]) {axisInd = 2 : i64}
  // CHECK:    [[OUTPUT:%.+]] = IE.Expand([[OUT_SOFTMAX]])
  // CHECK:    return [[OUTPUT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpandWithOffsetOfSliceNotAllZero
module @notOptimizeSliceSoftmaxExpandWithOffsetOfSliceNotAllZero {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpandWithOffsetOfSliceNotAllZero
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpandWithOffsetOfSliceNotAllZero(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 3, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x80x8x16xf16, {order = #NHWC}>

  // CHECK:    [[CUT_INPUT:%.+]] = IE.Slice [[INPUT]]
  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[CUT_INPUT]]) {axisInd = 1 : i64}
  // CHECK:    [[OUTPUT:%.+]] = IE.Expand([[OUT_SOFTMAX]])
  // CHECK:    return [[OUTPUT]]
}
}
