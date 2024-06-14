//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-batch %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @UnrollAveragePoolingBatch
func.func @UnrollAveragePoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %AVG_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[SLICE_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[AVG_POOL_0:%.*]] = IE.AvgPool([[SLICE_0]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[SLICE_1:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[AVG_POOL_1:%.*]] = IE.AvgPool([[SLICE_1]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[AVG_POOL_0]], [[AVG_POOL_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x128x32x64xf16>, tensor<1x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[CONCAT]] : tensor<2x128x32x64xf16>
}
