//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-batch %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @DoNotUnrollAveragePoolingBatch
func.func @DoNotUnrollAveragePoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %AVG_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[AVG_POOL:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[AVG_POOL]] : tensor<2x128x32x64xf16>
}
