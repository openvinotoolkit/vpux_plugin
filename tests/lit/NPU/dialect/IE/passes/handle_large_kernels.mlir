//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithSameKernelSize
func.func @HandleLargeKernelsAvgPoolWithSameKernelSize(%arg0 : tensor<1x128x16x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [16, 16]
    } : tensor<1x128x16x16xf16> -> tensor<1x128x1x1xf16>

    return %avg_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x128x16x16xf16> -> tensor<1x128x2x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x2x2xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithMultiSplit
func.func @HandleLargeKernelsAvgPoolWithMultiSplit(%arg0 : tensor<1x16x128x128xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [128, 128],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [128, 128]
    } : tensor<1x16x128x128xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x16x128x128xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x16x16x16xf16> -> tensor<1x16x2x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x2x2xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolAvoidLargerStride
func.func @HandleLargeKernelsAvgPoolAvoidLargerStride(%arg0 : tensor<1x16x176x176xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [176, 176],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [176, 176]
    } : tensor<1x16x176x176xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x16x176x176xf16> -> tensor<1x16x22x22xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      : tensor<1x16x22x22xf16> -> tensor<1x16x11x11xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x11x11xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithAsymmetricStride
func.func @HandleLargeKernelsAvgPoolWithAsymmetricStride(%arg0 : tensor<1x1024x32x64xf16>) -> (tensor<1x1024x2x2xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 32],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<CEIL>,
        strides = [16, 32]
    } : tensor<1x1024x32x64xf16> -> tensor<1x1024x2x2xf16>

    return %ave_pool : tensor<1x1024x2x2xf16>

    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<CEIL>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x1024x32x64xf16> -> tensor<1x1024x4x8xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<CEIL>,
    // CHECK-SAME:      strides = [2, 4]
    // CHECK-SAME:      : tensor<1x1024x4x8xf16> -> tensor<1x1024x2x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithDiffKernelSize
func.func @HandleLargeKernelsAvgPoolWithDiffKernelSize(%arg0 : tensor<1x128x9x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [9, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [9, 16]
    } : tensor<1x128x9x16xf16> -> tensor<1x128x1x1xf16>

    return %avg_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [9, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 8]
    // CHECK-SAME:      : tensor<1x128x9x16xf16> -> tensor<1x128x1x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x2xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargerKernelsAsymmetricAvgPool
func.func @HandleLargerKernelsAsymmetricAvgPool(%arg0 : tensor<1x16x144x99xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [144, 99],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [144, 99]
    } : tensor<1x16x144x99xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 3]
    // CHECK-SAME:      : tensor<1x16x144x99xf16> -> tensor<1x16x18x33xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [6, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [6, 3]
    // CHECK-SAME:      : tensor<1x16x18x33xf16> -> tensor<1x16x3x11xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x3x11xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargerKernelsAsymmetricAvgPoolWithLargeStride
func.func @HandleLargerKernelsAsymmetricAvgPoolWithLargeStride(%arg0 : tensor<1x16x40x20xf16>) -> (tensor<1x16x2x2xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [20, 10],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [20, 10]
    } : tensor<1x16x40x20xf16> -> tensor<1x16x2x2xf16>

    return %ave_pool : tensor<1x16x2x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [5, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [5, 10]
    // CHECK-SAME:      : tensor<1x16x40x20xf16> -> tensor<1x16x8x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [4, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [4, 1]
    // CHECK-SAME:      : tensor<1x16x8x2xf16> -> tensor<1x16x2x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithPadNeed
func.func @HandleLargeKernelsAvgPoolWithPadNeed(%arg0 : tensor<1x2048x23x30xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [23, 30],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [23, 30]
    } : tensor<1x2048x23x30xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK-DAG:       const.Declare
    // CHECK-SAME:      tensor<2048x1x8x6xf16> = dense<2.174380e-02> : tensor<2048x1x8x6xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2048 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      strides = [8, 6]
    // CHECK-SAME:      : tensor<1x2048x23x30xf16>, tensor<2048x1x8x6xf16> -> tensor<1x2048x3x5xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 5]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x3x5xf16> -> tensor<1x2048x1x1xf16>
}

// -----

// CHECK-LABEL: @CanNotHandleLargeKernelsAvgPoolWithPadNeed
func.func @CanNotHandleLargeKernelsAvgPoolWithPadNeed(%arg0 : tensor<1x2048x46x46xf16>) -> (tensor<1x2048x2x2xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [23, 23],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [23, 23]
    } : tensor<1x2048x46x46xf16> -> tensor<1x2048x2x2xf16>

    return %ave_pool : tensor<1x2048x2x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [23, 23]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [23, 23]
    // CHECK-SAME:      : tensor<1x2048x46x46xf16> -> tensor<1x2048x2x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithPadNeedIsGlobalPool
func.func @HandleLargeKernelsAvgPoolWithPadNeedIsGlobalPool(%arg0 : tensor<1x2048x48x23xf16>) -> (tensor<1x2048x2x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [24, 23],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [24, 23]
    } : tensor<1x2048x48x23xf16> -> tensor<1x2048x2x1xf16>

    return %ave_pool : tensor<1x2048x2x1xf16>
    // CHECK-DAG:       const.Declare
    // CHECK-SAME:      tensor<2048x1x8x8xf16> = dense<1.631160e-02> : tensor<2048x1x8x8xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2048 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x2048x48x23xf16>, tensor<2048x1x8x8xf16> -> tensor<1x2048x6x3xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 1]
    // CHECK-SAME:      : tensor<1x2048x6x3xf16> -> tensor<1x2048x2x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplit
func.func @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplit(%arg0 : tensor<1x2048x65x65xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [65, 65],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [65, 65]
    } : tensor<1x2048x65x65xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK-DAG:       const.Declare
    // CHECK-SAME:      tensor<2048x1x7x7xf16> = dense<2.366640e-02> : tensor<2048x1x7x7xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [5, 5]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [5, 5]
    // CHECK-SAME:      : tensor<1x2048x65x65xf16> -> tensor<1x2048x13x13xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2048 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [7, 7]
    // CHECK-SAME:      : tensor<1x2048x13x13xf16>, tensor<2048x1x7x7xf16> -> tensor<1x2048x2x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x2x2xf16> -> tensor<1x2048x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplitAsymmetric
func.func @HandleLargeKernelsAvgPoolWithPadNeedAndMultiSplitAsymmetric(%arg0 : tensor<1x16x258x257xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [258, 257],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x258x257xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK-DAG:       const.Declare
    // CHECK-SAME:      tensor<16x1x6x6xf16> = dense<2.789310e-02> : tensor<16x1x6x6xf16>
    // CHECK-DAG:       const.Declare
    // CHECK-SAME:      tensor<16x1x4x4xf16> = dense<6.542960e-02> : tensor<16x1x4x4xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      strides = [6, 6]
    // CHECK-SAME:      : tensor<1x16x258x257xf16>, tensor<16x1x6x6xf16> -> tensor<1x16x43x43xf16>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [4, 4]
    // CHECK-SAME:      : tensor<1x16x43x43xf16>, tensor<16x1x4x4xf16> -> tensor<1x16x11x11xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x11x11xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsXMaxPool
func.func @HandleLargeKernelsXMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %max_pool : tensor<1x64x10x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 10, 1] : tensor<1x64x10x13xf16> to tensor<1x64x10x1xf16>
    // CHECK:       IE.Concat(%arg0, [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x64x10x13xf16>, tensor<1x64x10x1xf16> -> tensor<1x64x10x14xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 7]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 7]
    // CHECK-SAME:      : tensor<1x64x10x14xf16> -> tensor<1x64x10x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x10x2xf16> -> tensor<1x64x10x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsYMaxPool
func.func @HandleLargeKernelsYMaxPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>

    return %max_pool : tensor<1x64x1x10xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 1, 10] : tensor<1x64x13x10xf16> to tensor<1x64x1x10xf16>
    // CHECK:       IE.Concat(%arg0, [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x64x13x10xf16>, tensor<1x64x1x10xf16> -> tensor<1x64x14x10xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [7, 1]
    // CHECK-SAME:      : tensor<1x64x14x10xf16> -> tensor<1x64x2x10xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [2, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x2x10xf16> -> tensor<1x64x1x10xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithSameKernelSize
func.func @HandleLargeKernelsMaxPoolWithSameKernelSize(%arg0 : tensor<1x16x32x32xf16>) -> (tensor<1x16x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [32, 32],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x32x32xf16> -> tensor<1x16x1x1xf16>

    return %max_pool : tensor<1x16x1x1xf16>

    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x16x32x32xf16> -> tensor<1x16x4x4xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [4, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x4x4xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithDiffKernelSize
func.func @HandleLargeKernelsMaxPoolWithDiffKernelSize(%arg0 : tensor<1x128x9x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [9, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [9, 16]
    } : tensor<1x128x9x16xf16> -> tensor<1x128x1x1xf16>

    return %max_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 8]
    // CHECK-SAME:      : tensor<1x128x9x16xf16> -> tensor<1x128x1x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x2xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeed
func.func @HandleLargeKernelsMaxPoolWithPadNeed(%arg0 : tensor<1x1x71x2xf16>) -> (tensor<1x1x1x2xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [71, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x1x71x2xf16> -> tensor<1x1x1x2xf16>

    return %max_pool : tensor<1x1x1x2xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 1, 1, 2] : tensor<1x1x71x2xf16> to tensor<1x1x1x2xf16>
    // CHECK:       IE.Concat(%arg0, [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x1x71x2xf16>, tensor<1x1x1x2xf16> -> tensor<1x1x72x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 1]
    // CHECK-SAME:      : tensor<1x1x72x2xf16> -> tensor<1x1x9x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x1x9x2xf16> -> tensor<1x1x1x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplit
func.func @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplit(%arg0 : tensor<1x2048x65x65xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [65, 65],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [65, 65]
    } : tensor<1x2048x65x65xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-SAME:      kernel_size = [5, 5]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [5, 5]
    // CHECK-SAME:      : tensor<1x2048x65x65xf16> -> tensor<1x2048x13x13xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[MAXPOOL]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 2048, 13, 1] : tensor<1x2048x13x13xf16> to tensor<1x2048x13x1xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[MAXPOOL]], [[SLICE0]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x2048x13x13xf16>, tensor<1x2048x13x1xf16> -> tensor<1x2048x13x14xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONCAT]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 2048, 1, 13] : tensor<1x2048x13x14xf16> to tensor<1x2048x1x13xf16>
    // CHECK:       IE.Concat([[CONCAT]], [[SLICE1]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x2048x13x14xf16>, tensor<1x2048x1x13xf16> -> tensor<1x2048x14x14xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 7]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [7, 7]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> -> tensor<1x2048x2x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [2, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x2x2xf16> -> tensor<1x2048x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplitAsymmetric
func.func @HandleLargeKernelsMaxPoolWithPadNeedAndMultiSplitAsymmetric(%arg0 : tensor<1x16x258x257xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [258, 257],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x258x257xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 16, 258, 1] : tensor<1x16x258x257xf16> to tensor<1x16x258x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat(%arg0, [[SLICE0]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x16x258x257xf16>, tensor<1x16x258x1xf16> -> tensor<1x16x258x258xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-SAME:      kernel_size = [6, 6]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [6, 6]
    // CHECK-SAME:      : tensor<1x16x258x258xf16> -> tensor<1x16x43x43xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[MAXPOOL]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 16, 43, 1] : tensor<1x16x43x43xf16> to tensor<1x16x43x1xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[MAXPOOL]], [[SLICE1]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x16x43x43xf16>, tensor<1x16x43x1xf16> -> tensor<1x16x43x44xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT1]]
    // CHECK-SAME:       [0, 0, 0, 0] [1, 16, 1, 43] : tensor<1x16x43x44xf16> to tensor<1x16x1x43xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[CONCAT1]], [[SLICE2]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:      : tensor<1x16x43x44xf16>, tensor<1x16x1x43xf16> -> tensor<1x16x44x44xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [4, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [4, 4]
    // CHECK-SAME:      : tensor<1x16x44x44xf16> -> tensor<1x16x11x11xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x11x11xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @CanNotHandleLargeKernelsMaxPoolWithPadNeed
func.func @CanNotHandleLargeKernelsMaxPoolWithPadNeed(%arg0 : tensor<1x2048x46x46xf16>) -> (tensor<1x2048x2x2xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [23, 23],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [23, 23]
    } : tensor<1x2048x46x46xf16> -> tensor<1x2048x2x2xf16>

    return %ave_pool : tensor<1x2048x2x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [23, 23]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [23, 23]
    // CHECK-SAME:      : tensor<1x2048x46x46xf16> -> tensor<1x2048x2x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPoolWithPadNeedIsGlobalPool
func.func @HandleLargeKernelsMaxPoolWithPadNeedIsGlobalPool(%arg0 : tensor<1x2048x48x23xf16>) -> (tensor<1x2048x2x1xf16>) {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [24, 23],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [24, 23]
    } : tensor<1x2048x48x23xf16> -> tensor<1x2048x2x1xf16>

    return %ave_pool : tensor<1x2048x2x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 2048, 48, 1] : tensor<1x2048x48x23xf16> to tensor<1x2048x48x1xf16>
    // CHECK:       IE.Concat(%arg0, [[SLICE]])
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:      : tensor<1x2048x48x23xf16>, tensor<1x2048x48x1xf16> -> tensor<1x2048x48x24xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x2048x48x24xf16> -> tensor<1x2048x6x3xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [3, 1]
    // CHECK-SAME:      : tensor<1x2048x6x3xf16> -> tensor<1x2048x2x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsOverlappedMaxPool
func.func @HandleLargeKernelsOverlappedMaxPool(%arg0 : tensor<1x512x19x19xf16>) -> (tensor<1x512x19x19xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [13, 13],
        pads_begin = [6, 6],
        pads_end = [6, 6],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>

    return %max_pool : tensor<1x512x19x19xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 7]
    // CHECK-SAME:      pads_begin = [3, 3]
    // CHECK-SAME:      pads_end = [3, 3]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 7]
    // CHECK-SAME:      pads_begin = [3, 3]
    // CHECK-SAME:      pads_end = [3, 3]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsOvSerlappedMaxPoolWithOneAxis
func.func @HandleLargeKernelsOvSerlappedMaxPoolWithOneAxis(%arg0 : tensor<1x512x19x19xf16>) -> (tensor<1x512x19x19xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [6, 0],
        pads_end = [6, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>

    return %max_pool : tensor<1x512x19x19xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      pads_begin = [3, 0]
    // CHECK-SAME:      pads_end = [3, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      pads_begin = [3, 0]
    // CHECK-SAME:      pads_end = [3, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv
func.func @HandleLargeKernelsConv(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 22], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x16xf16> = dense<0.000000e+00> : tensor<1x1x1x16xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], %arg0, [[CST]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x16xf16>, tensor<1x1x1x32000xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 22] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>

    // CHECK: return [[Add1]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithPostOp
func.func @HandleLargeKernelsConvWithPostOp(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]]  = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_0:%.+]]  = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_1:%.+]]  = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 22], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_2:%.+]]  = const.Declare tensor<1x1x1x16xf16> = dense<0.000000e+00> : tensor<1x1x1x16xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_2]], %arg0, [[CST_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x16xf16>, tensor<1x1x1x32000xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 22] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>

    // CHECK: return [[Add1]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithBias
func.func @HandleLargeKernelsConvWithBias(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    %conv = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 22], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x16xf16> = dense<0.000000e+00> : tensor<1x1x1x16xf16>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_2]], %arg0, [[CST_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x16xf16>, tensor<1x1x1x32000xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 22] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]], [[CST_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>

    // CHECK: return [[Add1]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @NotConvertDilatedConv
func.func @NotConvertDilatedConv(%arg0: tensor<1x256x1x512xf16>) -> tensor<1x256x1x512xf16> {
    %cst = const.Declare tensor<256x256x1x13xf16> = dense<1.000000e+00> : tensor<256x256x13xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[256, 256, 1, 13]>]
    %conv = IE.Convolution(%arg0, %cst) {
        dilations = [1, 2],
        pads_begin = [0, 12],
        pads_end = [0, 12],
        strides = [1, 1]
    } : tensor<1x256x1x512xf16>, tensor<256x256x1x13xf16> -> tensor<1x256x1x512xf16>

    return %conv : tensor<1x256x1x512xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<256x256x1x13xf16> = dense<1.000000e+00> : tensor<256x256x13xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[256, 256, 1, 13]>]
    // CHECK:           [[CONV:%.+]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:          dilations = [1, 2],
    // CHECK-SAME:          pads_begin = [0, 12],
    // CHECK-SAME:          pads_end = [0, 12],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x256x1x512xf16>, tensor<256x256x1x13xf16> -> tensor<1x256x1x512xf16>

    // CHECK:           return [[CONV]] : tensor<1x256x1x512xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithOneDimOnW
func.func @HandleLargeKernelsConvWithOneDimOnW(%arg0 : tensor<1x1x1x160000xf16>) -> tensor<1x257x1x1247xf16> {
    %cst = const.Declare tensor<257x1x1x512xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 128]} : tensor<1x1x1x160000xf16>, tensor<257x1x1x512xf16> -> tensor<1x257x1x1247xf16>
    return %conv : tensor<1x257x1x1247xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<257x64x1x8xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>, [#const.Reshape<[257, 8, 1, 64]>, #const.Transpose<#NWHC>]
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape(%arg0) {shape_value = [1, 2500, 1, 64]} : tensor<1x1x1x160000xf16> -> tensor<1x2500x1x64xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NWHC} : tensor<1x2500x1x64xf16> -> tensor<1x64x1x2500xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x64x1x2500xf16>, tensor<257x64x1x8xf16> -> tensor<1x257x1x1247xf16>
    // CHECK: return [[CONV]] : tensor<1x257x1x1247xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithOneDimOnH
func.func @HandleLargeKernelsConvWithOneDimOnH(%arg0 : tensor<1x1x2176x1xf16>) -> tensor<1x258x16x1xf16> {
    %cst = const.Declare tensor<258x1x512x1xf16> = dense<1.000000e+00> : tensor<258x1x512x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [128, 0], pads_end = [128, 0], strides = [128, 128]} : tensor<1x1x2176x1xf16>, tensor<258x1x512x1xf16> -> tensor<1x258x16x1xf16>
    return %conv : tensor<1x258x16x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<258x64x8x1xf16> = dense<1.000000e+00> : tensor<258x1x512x1xf16>, [#const.Reshape<[258, 8, 64, 1]>, #const.Transpose<#NHCW>]
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape(%arg0) {shape_value = [1, 34, 64, 1]} : tensor<1x1x2176x1xf16> -> tensor<1x34x64x1xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<1x34x64x1xf16> -> tensor<1x64x34x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE]], [[CST]]) {dilations = [1, 1], pads_begin = [2, 0], pads_end = [2, 0], strides = [2, 1]} : tensor<1x64x34x1xf16>, tensor<258x64x8x1xf16> -> tensor<1x258x16x1xf16>
    // CHECK: return [[CONV]] : tensor<1x258x16x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv2DimsSplit
func.func @HandleLargeKernelsConv2DimsSplit(%arg0 : tensor<1x1x32000x32000xf16>) -> tensor<1x64x2001x2001xf16> {
    %cst = const.Declare tensor<64x1x22x22xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [16, 16], pads_end = [16, 16], strides = [16, 16]} : tensor<1x1x32000x32000xf16>, tensor<64x1x22x22xf16> -> tensor<1x64x2001x2001xf16>

    return %conv : tensor<1x64x2001x2001xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 11, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 11, 11], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x32000x16xf16> = dense<0.000000e+00> : tensor<1x1x32000x16xf16>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x16x32032xf16> = dense<0.000000e+00> : tensor<1x1x16x32032xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[CST_3]], %arg0, [[CST_3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x32000x16xf16>, tensor<1x1x32000x32000xf16>, tensor<1x1x32000x16xf16> -> tensor<1x1x32000x32032xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], %0, [[CST_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x16x32032xf16>, tensor<1x1x32000x32032xf16>, tensor<1x1x16x32032xf16> -> tensor<1x1x32032x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 11] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add2:%.+]] = IE.Add([[Add1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>

    // CHECK: return [[Add2]] : tensor<1x64x2001x2001xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv2DimsUnevenSplit
func.func @HandleLargeKernelsConv2DimsUnevenSplit(%arg0 : tensor<1x1x32000x32000xf16>) -> tensor<1x64x2001x2001xf16> {
    %cst = const.Declare tensor<64x1x18x18xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [16, 16], pads_end = [16, 16], strides = [16, 16]} : tensor<1x1x32000x32000xf16>, tensor<64x1x18x18xf16> -> tensor<1x64x2001x2001xf16>

    return %conv : tensor<1x64x2001x2001xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x11x7xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 11, 7]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x7x11xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 11, 0], [64, 1, 7, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x7x7xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 11, 11], [64, 1, 7, 7]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x32000x16xf16> = dense<0.000000e+00> : tensor<1x1x32000x16xf16>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x16x32032xf16> = dense<0.000000e+00> : tensor<1x1x16x32032xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[CST_3]], %arg0, [[CST_3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x32000x16xf16>, tensor<1x1x32000x32000xf16>, tensor<1x1x32000x16xf16> -> tensor<1x1x32000x32032xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], [[CONCAT0]], [[CST_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x16x32032xf16>, tensor<1x1x32000x32032xf16>, tensor<1x1x16x32032xf16> -> tensor<1x1x32032x32032xf16>
    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 32011, 32007] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32007xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32007xf16>, tensor<64x1x11x7xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 0] [1, 1, 32007, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32007x32011xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32007x32011xf16>, tensor<64x1x7x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 11] [1, 1, 32007, 32007] : tensor<1x1x32032x32032xf16> to tensor<1x1x32007x32007xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32007x32007xf16>, tensor<64x1x7x7xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add2:%.+]] = IE.Add([[Add1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>

    // CHECK: return [[Add2]] : tensor<1x64x2001x2001xf16>
}

// -----

// CHECK-LABEL: @AvgPoolWithPadding
func.func @AvgPoolWithPadding(%arg0 : tensor<1x16x36x64xf16>) -> (tensor<1x16x3x3xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [12, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 22]
    } : tensor<1x16x36x64xf16> -> tensor<1x16x3x3xf16>

    return %avg_pool : tensor<1x16x3x3xf16>
    // CHECK:           [[AVGPOOL1:%.+]] = IE.AvgPool(%arg0) {kernel_size = [6, 2], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [6, 2]} : tensor<1x16x36x64xf16> -> tensor<1x16x6x33xf16>
    // CHECK:           [[AVGPOOL2:%.+]] = IE.AvgPool([[AVGPOOL1]]) {kernel_size = [2, 11], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 11]} : tensor<1x16x6x33xf16> -> tensor<1x16x3x3xf16>

    // CHECK:           return [[AVGPOOL2]] : tensor<1x16x3x3xf16>
}

// -----

// CHECK-LABEL: @MaxPoolWithPadding
func.func @MaxPoolWithPadding(%arg0 : tensor<1x16x36x64xf16>) -> (tensor<1x16x3x3xf16>) {
    %avg_pool = IE.MaxPool(%arg0) {
        kernel_size = [12, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 22]
    } : tensor<1x16x36x64xf16> -> tensor<1x16x3x3xf16>

    return %avg_pool : tensor<1x16x3x3xf16>
    // CHECK:           [[MAXPOOL1:%.+]] = IE.MaxPool(%arg0) {kernel_size = [6, 2], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [6, 2]} : tensor<1x16x36x64xf16> -> tensor<1x16x6x33xf16>
    // CHECK:           [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [2, 11], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 11]} : tensor<1x16x6x33xf16> -> tensor<1x16x3x3xf16>

    // CHECK:           return [[MAXPOOL2]] : tensor<1x16x3x3xf16>
}


// -----

// CHECK-LABEL: @AvgPoolWithExcludePadding
func.func @AvgPoolWithExcludePadding(%arg0 : tensor<1x16x36x64xf16>) -> (tensor<1x16x3x3xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [12, 22],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 22]
    } : tensor<1x16x36x64xf16> -> tensor<1x16x3x3xf16>

    return %avg_pool : tensor<1x16x3x3xf16>
    // CHECK:           [[AVGPOOL1:%.+]] = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [6, 2], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [6, 2]} : tensor<1x16x36x64xf16> -> tensor<1x16x6x33xf16>
    // CHECK:           [[AVGPOOL2:%.+]] = IE.AvgPool([[AVGPOOL1]]) {exclude_pads, kernel_size = [2, 11], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 11]} : tensor<1x16x6x33xf16> -> tensor<1x16x3x3xf16>

    // CHECK:           return [[AVGPOOL2]] : tensor<1x16x3x3xf16>
}


// -----

// CHECK-LABEL: @AvgPoolWithPaddingAndKernelPadding
func.func @AvgPoolWithPaddingAndKernelPadding(%arg0 : tensor<1x16x12x21xf16>) -> (tensor<1x16x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [12, 23],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 23]
    } : tensor<1x16x12x21xf16> -> tensor<1x16x1x1xf16>

    return %avg_pool : tensor<1x16x1x1xf16>
    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x6x8xf16> = dense<2.174380e-02> : tensor<16x1x6x8xf16>
    // CHECK:           [[GCONV:%.+]] = IE.GroupConvolution(%arg0, [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 1], pads_end = [0, 2],
    // CHECK-SAME:              strides = [6, 8]} : tensor<1x16x12x21xf16>, tensor<16x1x6x8xf16> -> tensor<1x16x2x3xf16>
    // CHECK:           [[AVGPOOL:%.+]] = IE.AvgPool([[GCONV]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x3xf16> -> tensor<1x16x1x1xf16>

    // CHECK:           return [[AVGPOOL]] : tensor<1x16x1x1xf16>
}


// -----

// CHECK-LABEL: @AvgPoolWithExcludePaddingAndKernelPadding
func.func @AvgPoolWithExcludePaddingAndKernelPadding(%arg0 : tensor<1x16x12x21xf16>) -> (tensor<1x16x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [12, 23],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 23]
    } : tensor<1x16x12x21xf16> -> tensor<1x16x1x1xf16>

    return %avg_pool : tensor<1x16x1x1xf16>
    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x6x8xf16> = dense<2.380370e-02> : tensor<16x1x6x8xf16>
    // CHECK:           [[GCONV:%.+]] = IE.GroupConvolution(%arg0, [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 1], pads_end = [0, 2],
    // CHECK-SAME:              strides = [6, 8]} : tensor<1x16x12x21xf16>, tensor<16x1x6x8xf16> -> tensor<1x16x2x3xf16>
    // CHECK:           [[AVGPOOL:%.+]] = IE.AvgPool([[GCONV]]) {exclude_pads, kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:              rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x3xf16> -> tensor<1x16x1x1xf16>

    // CHECK:           return [[AVGPOOL]] : tensor<1x16x1x1xf16>
}


// -----

// CHECK-LABEL: @MaxPoolWithPaddingAndKernelPadding
func.func @MaxPoolWithPaddingAndKernelPadding(%arg0 : tensor<1x16x12x21xf16>) -> (tensor<1x16x1x1xf16>) {
    %avg_pool = IE.MaxPool(%arg0) {
        kernel_size = [12, 23],
        pads_begin = [0, 1],
        pads_end = [0, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [12, 23]
    } : tensor<1x16x12x21xf16> -> tensor<1x16x1x1xf16>

    return %avg_pool : tensor<1x16x1x1xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 12, 1] : tensor<1x16x12x21xf16> to tensor<1x16x12x1xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat(%arg0, [[SLICE]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x16x12x21xf16>, tensor<1x16x12x1xf16> -> tensor<1x16x12x22xf16>
    // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool(%1) {kernel_size = [6, 8], pads_begin = [0, 1], pads_end = [0, 1],
    // CHECK-SAME:           rounding_type = #IE.rounding_type<FLOOR>, strides = [6, 8]} : tensor<1x16x12x22xf16> -> tensor<1x16x2x3xf16>
    // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool(%2) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x3xf16> -> tensor<1x16x1x1xf16>

    // CHECK:           return [[MAXPOOL2]] : tensor<1x16x1x1xf16>
}
