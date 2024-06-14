//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @NotConvertLargeKernelsXAvgPoolForBetterSWPerf
func.func @NotConvertLargeKernelsXAvgPoolForBetterSWPerf(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 13]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 13]
    // CHECK-SAME:      : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>
}

// -----

// CHECK-LABEL: @NotConvertLargeKernelsYAvgPoolForBetterSWPerf
func.func @NotConvertLargeKernelsYAvgPoolForBetterSWPerf(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>
    return %ave_pool : tensor<1x64x1x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [13, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [13, 1]
    // CHECK-SAME:      : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>
}
