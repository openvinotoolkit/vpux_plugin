//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @HandleLargeKernelsXAvgPool
func.func @HandleLargeKernelsXAvgPool(%arg0 : tensor<1x64x10x17xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 17],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 17]
    } : tensor<1x64x10x17xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>

    // CHECK:       const.Declare
    // CHECK-SAME:      tensor<64x1x1x6xf16> = dense<1.765140e-01>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 64 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x64x10x17xf16>, tensor<64x1x1x6xf16> -> tensor<1x64x10x3xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x10x3xf16> -> tensor<1x64x10x1xf16>

}

// -----

// CHECK-LABEL: @HandleLargeKernelsYAvgPool
func.func @HandleLargeKernelsYAvgPool(%arg0 : tensor<1x64x17x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [17, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [17, 1]
    } : tensor<1x64x17x10xf16> -> tensor<1x64x1x10xf16>

    return %ave_pool : tensor<1x64x1x10xf16>

    // CHECK:       const.Declare
    // CHECK-SAME:      tensor<64x1x6x1xf16> = dense<1.765140e-01>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 64 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      strides = [6, 1]
    // CHECK-SAME:      : tensor<1x64x17x10xf16>, tensor<64x1x6x1xf16> -> tensor<1x64x3x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x3x10xf16> -> tensor<1x64x1x10xf16>

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

// -----

// CHECK-LABEL: @HandleLargerKernelsAvgPoolWithStaticScalePaddingNeededOneDim
func.func @HandleLargerKernelsAvgPoolWithStaticScalePaddingNeededOneDim(%arg0 : tensor<1x128x1x75076xf16>) -> (tensor<1x128x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 75076],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        static_scale = 0.135327876 : f32,
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
    // CHECK-NOT:       static_scale
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
    // CHECK-NOT:       static_scale
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
    // CHECK-SAME:      static_scale = 0.135327876 : f32,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x10xf16> -> tensor<1x128x1x1xf16>
}
