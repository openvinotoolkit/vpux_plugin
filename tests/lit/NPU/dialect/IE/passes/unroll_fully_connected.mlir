//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-fully-connected %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @UnrollMatMul
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x96xf32>,
// CHECK-SAME:   [[RHS_1:%arg1]]: tensor<1x32x64xf32>,
// CHECK-SAME:   [[RHS_2:%arg2]]: tensor<1x32x64xf32>,
// CHECK-SAME:   [[RHS_3:%arg3]]: tensor<1x32x64xf32>
func.func @UnrollMatMul(%LHS_1: tensor<16x96xf32>,
                        %RHS_1: tensor<1x32x64xf32>,
                        %RHS_2: tensor<1x32x64xf32>,
                        %RHS_3: tensor<1x32x64xf32>) -> tensor<16x64xf32> {
    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x32x64xf32>, tensor<1x32x64xf32>, tensor<1x32x64xf32> -> tensor<3x32x64xf32>
    // CHECK-NOT:   IE.Concat

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.*]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.*]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.*]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[LHS_SLICE_1:%.*]] = IE.Slice [[LHS_1]] [0, 0] [16, 32]
    // CHECK:   [[LHS_SLICE_2:%.*]] = IE.Slice [[LHS_1]] [0, 32] [16, 32]
    // CHECK:   [[LHS_SLICE_3:%.*]] = IE.Slice [[LHS_1]] [0, 64] [16, 32]

    // CHECK:   [[TRANSPOSE_1:%.*]] = IE.Transpose([[RESHAPE_RHS_1]])
    // CHECK:   [[GEMM_1:%.*]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]])
    // CHECK:   [[TRANSPOSE_2:%.*]] = IE.Transpose([[RESHAPE_RHS_2]])
    // CHECK:   [[GEMM_2:%.*]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]])
    // CHECK:   [[TRANSPOSE_3:%.*]] = IE.Transpose([[RESHAPE_RHS_3]])
    // CHECK:   [[GEMM_3:%.*]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]])

    // CHECK:   [[ADD_1:%.*]] = IE.Accumulate([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.*]] = IE.Accumulate([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithoutTranspose
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<64x96xf32>
func.func @SkipMatMulWithoutTranspose(%LHS: tensor<16x96xf32>, %RHS: tensor<64x96xf32>) -> tensor<16x64xf32> {
    %GEMM = IE.FullyConnected(%LHS, %RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.FullyConnected([[LHS]], [[RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithoutReshape
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<96x64xf32>
func.func @SkipMatMulWithoutReshape(%LHS: tensor<16x96xf32>, %RHS: tensor<96x64xf32>) -> tensor<16x64xf32> {
    %TRANSPOSE_RHS = IE.Transpose(%RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.*]] = IE.Transpose([[RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithUnsupportedReshape
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<2x96x32xf32>
func.func @SkipMatMulWithUnsupportedReshape(%LHS: tensor<16x96xf32>, %RHS: tensor<2x96x32xf32>) -> tensor<16x64xf32> {
    %RESHAPE_RHS = IE.AffineReshape(%RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<2x96x32xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.*]] = IE.AffineReshape([[RHS]])
    // CHECK-SAME:      shape_value = [96, 64]
    // CHECK-SAME:  tensor<2x96x32xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.*]] = IE.Transpose([[RESHAPE_RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithoutConcat
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<3x32x64xf32>
func.func @SkipMatMulWithoutConcat(%LHS: tensor<16x96xf32>, %RHS: tensor<3x32x64xf32>) -> tensor<16x64xf32> {
    %RESHAPE_RHS = IE.AffineReshape(%RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.*]] = IE.AffineReshape([[RHS]])
    // CHECK-SAME:      shape_value = [96, 64]
    // CHECK-SAME:  tensor<3x32x64xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.*]] = IE.Transpose([[RESHAPE_RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @SkipMatMulWithUnsupportedConcat
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS_0:%.*]]: tensor<3x16x64xf32>,
// CHECK-SAME:   [[RHS_1:%.*]]: tensor<3x16x64xf32>
func.func @SkipMatMulWithUnsupportedConcat(%LHS: tensor<16x96xf32>,
                                           %RHS_0: tensor<3x16x64xf32>,
                                           %RHS_1: tensor<3x16x64xf32>) -> tensor<16x64xf32> {
    %CONCAT_RHS = IE.Concat(%RHS_0, %RHS_1) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : tensor<3x16x64xf32>, tensor<3x16x64xf32> -> tensor<3x32x64xf32>
    // CHECK:   [[CONCAT_RHS:%.*]] = IE.Concat([[RHS_0]], [[RHS_1]])

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.*]] = IE.AffineReshape([[CONCAT_RHS]])
    // CHECK-SAME:      shape_value = [96, 64]
    // CHECK-SAME:  tensor<3x32x64xf32> -> tensor<96x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>
    // CHECK:   [[TRANSPOSE_RHS:%.*]] = IE.Transpose([[RESHAPE_RHS]])

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.FullyConnected([[LHS]], [[TRANSPOSE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @UnrollMatMulReshapeTranspose
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x96xf32>,
// CHECK-SAME:   [[RHS_1:%arg1]]: tensor<1x32x64xf32>,
// CHECK-SAME:   [[RHS_2:%arg2]]: tensor<1x32x64xf32>,
// CHECK-SAME:   [[RHS_3:%arg3]]: tensor<1x32x64xf32>
func.func @UnrollMatMulReshapeTranspose(%LHS_1: tensor<16x96xf32>,
                                        %RHS_1: tensor<1x32x64xf32>,
                                        %RHS_2: tensor<1x32x64xf32>,
                                        %RHS_3: tensor<1x32x64xf32>) -> tensor<16x64xf32> {
    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x32x64xf32>, tensor<1x32x64xf32>, tensor<1x32x64xf32> -> tensor<3x32x64xf32>
    // CHECK-NOT:   IE.Concat

    %TRANSPOSE_RHS = IE.Transpose(%CONCAT_RHS) {
        order_value = #map
    } : tensor<3x32x64xf32> -> tensor<64x3x32xf32>

    %RESHAPE_RHS = IE.AffineReshape(%TRANSPOSE_RHS) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [64, 96]
    } : tensor<64x3x32xf32> -> tensor<64x96xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.*]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.*]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.*]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>

    %GEMM = IE.FullyConnected(%LHS_1, %RESHAPE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[LHS_SLICE_1:%.*]] = IE.Slice [[LHS_1]] [0, 0] [16, 32]
    // CHECK:   [[LHS_SLICE_2:%.*]] = IE.Slice [[LHS_1]] [0, 32] [16, 32]
    // CHECK:   [[LHS_SLICE_3:%.*]] = IE.Slice [[LHS_1]] [0, 64] [16, 32]

    // CHECK:   [[TRANSPOSE_1:%.*]] = IE.Transpose([[RESHAPE_RHS_1]])
    // CHECK:   [[GEMM_1:%.*]] = IE.FullyConnected([[LHS_SLICE_1]], [[TRANSPOSE_1]])
    // CHECK:   [[TRANSPOSE_2:%.*]] = IE.Transpose([[RESHAPE_RHS_2]])
    // CHECK:   [[GEMM_2:%.*]] = IE.FullyConnected([[LHS_SLICE_2]], [[TRANSPOSE_2]])
    // CHECK:   [[TRANSPOSE_3:%.*]] = IE.Transpose([[RESHAPE_RHS_3]])
    // CHECK:   [[GEMM_3:%.*]] = IE.FullyConnected([[LHS_SLICE_3]], [[TRANSPOSE_3]])

    // CHECK:   [[ADD_1:%.*]] = IE.Accumulate([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.*]] = IE.Accumulate([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWith2dReshape
// CHECK-SAME:   [[LHS:%.+]]: tensor<32xf16>, [[RHS:%.+]]: tensor<32xf16>
func.func @SkipMatMulWith2dReshape(%LHS: tensor<32xf16>, %RHS: tensor<32xf16>) -> tensor<1x1xf16> {
    %LHS_TO_2D = IE.AffineReshape(%LHS) {
        dim_mapping = [[0, 1]],
        shape_value = [1, 32]
    } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[LHS_TO_2D:%.+]] = IE.AffineReshape([[LHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %RHS_TO_2D = IE.AffineReshape(%RHS) {
        dim_mapping = [[0, 1]],
        shape_value = [1, 32]
    } : tensor<32xf16> -> tensor<1x32xf16>

    // CHECK:   [[RHS_TO_2D:%.+]] = IE.AffineReshape([[RHS]]) {
    // CHECK-SAME:      shape_value = [1, 32]
    // CHECK-SAME:  } : tensor<32xf16> -> tensor<1x32xf16>

    %FC = IE.FullyConnected(%LHS_TO_2D, %RHS_TO_2D) : tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x1xf16>

    // CHECK:   [[FC:%.+]] = IE.FullyConnected([[LHS_TO_2D]], [[RHS_TO_2D]])

    return %FC : tensor<1x1xf16>
    // CHECK:   return [[FC]] : tensor<1x1xf16>
}
