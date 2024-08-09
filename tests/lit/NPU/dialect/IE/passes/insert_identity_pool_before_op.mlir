//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --insert-identity-pool-before-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @SkipLReluWithNCEProducer
func.func @SkipLReluWithNCEProducer(%arg0: tensor<1x128x2x32xf16>) -> tensor<1x128x2x32xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    %LRELU = IE.LeakyRelu(%MAX_POOL) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    return %LRELU : tensor<1x128x2x32xf16>

    // CHECK:   [[MAX_POOL:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[MAX_POOL]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   return [[LRELU]] : tensor<1x128x2x32xf16>
}
