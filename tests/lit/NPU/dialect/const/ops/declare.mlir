//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: FuseSubViewTransformations
func.func @FuseSubViewTransformations() -> tensor<320x2816x1x1xf16> {
    %cst = const.Declare tensor<320x2816x1x1xf16> = dense<0.0> : tensor<4096x11008x1x1xf16>,
        [#const.SubView<[0, 8192, 0, 0], [4096, 2816, 1, 1]>, #const.SubView<[0, 0, 0, 0], [320, 2816, 1, 1]>]
    return %cst : tensor<320x2816x1x1xf16>

    // CHECK: [#const.SubView<[0, 8192, 0, 0], [320, 2816, 1, 1]>]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SwapReorderAndSubView
func.func @SwapReorderAndSubView() -> tensor<2x1x1x1xf16, { order = #NHWC }> {
    %cst = const.Declare tensor<2x1x1x1xf16, { order = #NHWC }> = dense<0.0> : tensor<4x3x2x1xf16>,
        [#const.Reorder<#NHWC>, #const.SubView<[2, 2, 1, 0], [2, 1, 1, 1]>]
    return %cst : tensor<2x1x1x1xf16, { order = #NHWC }>

    // CHECK: [#const.SubView<[2, 2, 1, 0], [2, 1, 1, 1]>, #const.Reorder<#NHWC>]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: SwapTransposeAndSubView
func.func @SwapTransposeAndSubView() -> tensor<4x3x1x2xf16, { order = #HCNW }> {
    %cst = const.Declare tensor<4x3x1x2xf16, { order = #HCNW }> = dense<0.0> : tensor<8x6x4x2xf16, { order = #HCNW }>,
        [#const.Transpose<#NHWC>, #const.SubView<[4, 1, 1, 4], [4, 3, 1, 2]>]
    return %cst : tensor<4x3x1x2xf16, { order = #HCNW }>

    // CHECK: [#const.SubView<[4, 4, 1, 1], [4, 2, 3, 1]>, #const.Transpose<#NHWC>]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: SwapMemPermAndSubViewMultipleInstances
func.func @SwapMemPermAndSubViewMultipleInstances() -> tensor<10x100x32x10xf16, { order = #NHWC }> {
    %cst = const.Declare tensor<10x100x32x10xf16, { order = #NHWC }> = dense<0.0> : tensor<4096x11008x1x1xf16>,
        [
            #const.Transpose<#HCNW>,
            #const.SubView<[0, 8192, 0, 0], [1, 2816, 4096, 1]>,
            #const.SubView<[0, 0, 0, 0], [1, 2816, 320, 1]>,
            #const.Reorder<#NHWC>,
            #const.SubView<[0, 1000, 0, 0], [1, 1816, 320, 1]>,
            #const.SubView<[0, 0, 0, 0], [1, 1000, 320, 1]>,
            #const.Reshape<[10, 100, 32, 10]>
        ]
    return %cst : tensor<10x100x32x10xf16, { order = #NHWC }>

    // CHECK: [#const.SubView<[0, 9192, 0, 0], [320, 1000, 1, 1]>, #const.Transpose<#map>, #const.Reorder<#NHWC>, #const.Reshape<[10, 100, 32, 10]>]
}

// -----

// CHECK-LABEL: DoNotOptimizeWithoutTransformations
func.func @DoNotOptimizeWithoutTransformations() -> tensor<320x2816x1x1xf16> {
    %cst = const.Declare tensor<320x2816x1x1xf16> = dense<0.0> : tensor<320x2816x1x1xf16>
    return %cst : tensor<320x2816x1x1xf16>

    // CHECK: const.Declare tensor<320x2816x1x1xf16> = dense<0.000000e+00> : tensor<320x2816x1x1xf16>
}
