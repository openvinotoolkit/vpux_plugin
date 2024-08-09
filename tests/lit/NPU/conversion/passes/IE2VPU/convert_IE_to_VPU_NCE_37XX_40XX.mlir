//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipEltwiseSubToNCE
func.func @SkipEltwiseSubToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Subtract(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Eltwise

    // CHECK:       [[OUT:%.+]] = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipEltwiseMulToNCE
func.func @SkipEltwiseMulToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Eltwise

    // CHECK:       [[OUT:%.+]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>

// CHECK: func.func @LowerMatMulToNCE([[INPUT:%.+]]: tensor<1x128x64x32xf16>)
func.func @LowerMatMulToNCE(%input : tensor<1x128x64x32xf16>) -> tensor<1x128x64x64xf16> {
    %matmul = IE.MatMul(%input, %input) { transpose_b } : tensor<1x128x64x32xf16>, tensor<1x128x64x32xf16> -> tensor<1x128x64x64xf16>
    return %matmul : tensor<1x128x64x64xf16>

    // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:      tensor<1x128x64x32xf16> -> tensor<128x64x32x1x1xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_0]])
    // CHECK-SAME:      tensor<128x64x32x1x1xf16> -> tensor<128x1x32x64x1xf16, {order = #GNHWC}>

    // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:      tensor<1x128x64x32xf16> -> tensor<128x64x32x1x1xf16>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_1]])
    // CHECK-SAME:      tensor<128x64x32x1x1xf16> -> tensor<128x64x32x1x1xf16, {order = #GNHWC}>


    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x64x1x1x4xsi32>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<
    // CHECK-SAME:          left = 0 : i64,
    // CHECK-SAME:          right = 0 : i64,
    // CHECK-SAME:          top = 0 : i64,
    // CHECK-SAME:          bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [128, 64, 32, 1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<128x1x64x64x1xf16, {order = #GNHWC}>


    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[MATMUL]])
    // CHECK-SAME:      tensor<128x1x64x64x1xf16, {order = #GNHWC}> -> tensor<128x64x64x1x1xf16>

    // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[MEMPERM]])
    // CHECK-SAME:      tensor<128x64x64x1x1xf16> -> tensor<1x128x64x64xf16>


    // CHECK:       return [[AFFINE_RESHAPE_2]] : tensor<1x128x64x64xf16>
}
