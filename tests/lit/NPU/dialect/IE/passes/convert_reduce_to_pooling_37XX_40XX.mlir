//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL:    func.func @ConvertReduceMeanToPooling3D
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<256x7x7xf16>)
func.func @ConvertReduceMeanToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<256x7x7xf16> -> tensor<256x1x7xf16>
  return %0 : tensor<256x1x7xf16>

  // CHECK:       IE.Reshape([[INPUT]]) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceMean
  // CHECK:       %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       %2 = IE.Reshape(%1) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
}
