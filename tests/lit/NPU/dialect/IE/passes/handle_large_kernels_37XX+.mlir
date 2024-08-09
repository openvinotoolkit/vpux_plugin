//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @HandleLargeKernelsXAvgPool
func.func @HandleLargeKernelsXAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>

    // CHECK:       const.Declare
    // CHECK-SAME:      tensor<64x1x1x7xf16> = dense<1.538090e-01>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 64 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 7]
    // CHECK-SAME:      : tensor<1x64x10x13xf16>, tensor<64x1x1x7xf16> -> tensor<1x64x10x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x10x2xf16> -> tensor<1x64x10x1xf16>

}

// -----

// CHECK-LABEL: @HandleLargeKernelsYAvgPool
func.func @HandleLargeKernelsYAvgPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>

    return %ave_pool : tensor<1x64x1x10xf16>

    // CHECK:       const.Declare
    // CHECK-SAME:      tensor<64x1x7x1xf16> = dense<1.538090e-01>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 64 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      strides = [7, 1]
    // CHECK-SAME:      : tensor<1x64x13x10xf16>, tensor<64x1x7x1xf16> -> tensor<1x64x2x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x2x10xf16> -> tensor<1x64x1x10xf16>

}

// -----

// CHECK-LABEL: @HandleLargerKernelsAvgPoolPaddingNeededOneDim
func.func @HandleLargerKernelsAvgPoolPaddingNeededOneDim(%arg0 : tensor<1x128x1x75076xf16>) -> (tensor<1x128x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 75076],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x1x75076xf16> -> tensor<1x128x1x1xf16>

    return %ave_pool : tensor<1x128x1x1xf16>

    // CHECK-DAG:   const.Declare tensor<128x1x1x5xf16> = dense<1.999510e-01> : tensor<128x1x1x5xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x6xf16> = dense<1.667480e-01> : tensor<128x1x1x6xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x2xf16> = dense<5.014650e-01> : tensor<128x1x1x2xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x2xf16> = dense<5.034180e-01> : tensor<128x1x1x2xf16>
    // CHECK-DAG:   const.Declare tensor<128x1x1x8xf16> = dense<1.265870e-01> : tensor<128x1x1x8xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      : tensor<1x128x1x75076xf16> -> tensor<1x128x1x18769xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 5]
    // CHECK-SAME:      : tensor<1x128x1x18769xf16>, tensor<128x1x1x5xf16> -> tensor<1x128x1x3754xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x3754xf16> -> tensor<1x128x1x1877xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x128x1x1877xf16>, tensor<128x1x1x6xf16> -> tensor<1x128x1x313xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x313xf16>, tensor<128x1x1x2xf16> -> tensor<1x128x1x157xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x157xf16>, tensor<128x1x1x2xf16> -> tensor<1x128x1x79xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 128 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 8]
    // CHECK-SAME:      : tensor<1x128x1x79xf16>, tensor<128x1x1x8xf16> -> tensor<1x128x1x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x10xf16> -> tensor<1x128x1x1xf16>
}
