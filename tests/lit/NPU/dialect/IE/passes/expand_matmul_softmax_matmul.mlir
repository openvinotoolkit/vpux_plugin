//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --expand-matmul-softmax-matmul %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: @ExpandMatmulSoftmaxMatmul
// CHECK-SAME:    ([[INPUT1:%.+]]: tensor<1x256x49x32xf16>, [[INPUT2:%.+]]: tensor<1x256x32x49xf16>, [[INPUT3:%.+]]: tensor<1x256x49x32xf16>)
func.func @ExpandMatmulSoftmaxMatmul(%arg0: tensor<1x256x49x32xf16>, %arg1: tensor<1x256x32x49xf16>, %arg2: tensor<1x256x49x32xf16>) -> tensor<1x256x49x32xf16> {
  %0 = IE.MatMul(%arg0, %arg1) : tensor<1x256x49x32xf16>, tensor<1x256x32x49xf16> -> tensor<1x256x49x49xf16>
  %1 = IE.Reshape(%0) {shape_value = [64, 4, 49, 49]} : tensor<1x256x49x49xf16> -> tensor<64x4x49x49xf16>
  %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<64x4x49x49xf16> -> tensor<64x4x49x49xf16>
  %3 = IE.Reshape(%2) {shape_value = [1, 256, 49, 49]} : tensor<64x4x49x49xf16> -> tensor<1x256x49x49xf16>
  %4 = IE.MatMul(%3, %arg2) : tensor<1x256x49x49xf16>, tensor<1x256x49x32xf16> -> tensor<1x256x49x32xf16>
  return %4 : tensor<1x256x49x32xf16>
}
    // CHECK: [[EXPAND1:%.+]] = IE.Expand([[INPUT2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]} : tensor<1x256x32x49xf16> -> tensor<1x256x32x64xf16>
    // CHECK: [[MATMUL1:%.+]] = IE.MatMul([[INPUT1]], [[EXPAND1]]) : tensor<1x256x49x32xf16>, tensor<1x256x32x64xf16> -> tensor<1x256x49x64xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.Reshape([[MATMUL1]]) {shape_value = [64, 4, 49, 64]} : tensor<1x256x49x64xf16> -> tensor<64x4x49x64xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE1]]) {axisInd = 3 : i64, padSize = 15 : i64} : tensor<64x4x49x64xf16> -> tensor<64x4x49x64xf16>
    // CHECK: [[EXPAND2:%.+]] = IE.Expand([[INPUT3]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 32]} : tensor<1x256x49x32xf16> -> tensor<1x256x49x64xf16>
    // CHECK: [[RESHAPE2:%.+]] = IE.Reshape([[SOFTMAX]]) {shape_value = [1, 256, 49, 64]} : tensor<64x4x49x64xf16> -> tensor<1x256x49x64xf16>
    // CHECK: [[MATMUL2:%.+]] = IE.MatMul([[RESHAPE2]], [[EXPAND2]]) : tensor<1x256x49x64xf16>, tensor<1x256x49x64xf16> -> tensor<1x256x49x64xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[MATMUL2]] [0, 0, 0, 0] [1, 256, 49, 32] : tensor<1x256x49x64xf16> to tensor<1x256x49x32xf16>
    // CHECK: return [[SLICE]] : tensor<1x256x49x32xf16>
