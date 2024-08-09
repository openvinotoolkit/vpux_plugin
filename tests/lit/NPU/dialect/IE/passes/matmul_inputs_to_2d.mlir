//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --matmul-inputs-to-2d %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @MatMulInputsTo2d
func.func @MatMulInputsTo2d(%arg0: tensor<2x1x512xf32>) -> tensor<2x1x40xf32> {
    %cst = const.Declare tensor<2x512x40xf32> = dense<1.0> : tensor<2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x1x512xf32>, tensor<2x512x40xf32> -> tensor<2x1x40xf32>

    return %0 : tensor<2x1x40xf32>

    // CHECK-DAG:   %[[CST_1_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>, [#const.SubView<[0, 0, 0], [1, 512, 40]>, #const.Reshape<[512, 40]>]
    // CHECK-DAG:   %[[CST_2_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>, [#const.SubView<[1, 0, 0], [1, 512, 40]>, #const.Reshape<[512, 40]>]
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.Reshape(%[[IN_1]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [1, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.Reshape(%[[IN_2]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 1, 40]} : tensor<2x40xf32> -> tensor<2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<2x1x40xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:   %[[CST_1_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>]
    // CHECK-DAG:   %[[CST_2_2D:.*]] = const.Declare tensor<512x40xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>]
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.Reshape(%[[IN_1]]) {shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.Reshape(%[[IN_2]]) {shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [1, 2, 1, 40]} : tensor<2x40xf32> -> tensor<1x2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<1x2x1x40xf32>
}

// -----

// CHECK-LABEL: @MatMul3dInputsBatch1To2d
func.func @MatMul3dInputsBatch1To2d(%arg0: tensor<1x1x1024xf32>) -> tensor<1x1x512xf32> {
    %cst = const.Declare tensor<1x1024x512xf32> = dense<1.0> : tensor<1x1024x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x1x1024xf32>, tensor<1x1024x512xf32> -> tensor<1x1x512xf32>

    return %0 : tensor<1x1x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<1024x512xf32> = dense<1.000000e+00> : tensor<1x1024x512xf32>, [#const.Reshape<[1024, 512]>]
    // CHECK: %[[RESHAPE0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 1024]} : tensor<1x1x1024xf32> -> tensor<1x1024xf32>
    // CHECK: %[[MTML:.*]] = IE.MatMul(%[[RESHAPE0]], %[[CST]]) : tensor<1x1024xf32>, tensor<1024x512xf32> -> tensor<1x512xf32>
    // CHECK: %[[OUT:.*]] = IE.Reshape(%[[MTML]]) {shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x1x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInput2DWeightsTo2d
func.func @NoChangesMatMul3dInput2DWeightsTo2d(%arg0: tensor<1x4x9728xf32>) -> tensor<1x4x512xf32> {
    %cst = const.Declare tensor<9728x512xf32> = dense<1.0> : tensor<9728x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x4x9728xf32>, tensor<9728x512xf32> -> tensor<1x4x512xf32>

    return %0 : tensor<1x4x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<9728x512xf32> = dense<1.000000e+00> : tensor<9728x512xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [4, 9728]} : tensor<1x4x9728xf32> -> tensor<4x9728xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<4x9728xf32>, tensor<9728x512xf32> -> tensor<4x512xf32>
    // CHECK: %[[OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1, 4, 512]} : tensor<4x512xf32> -> tensor<1x4x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x4x512xf32>
}

// -----

// CHECK-LABEL: @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d
func.func @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x64xf32> {
    %cst = const.Declare tensor<128x64xf32> = dense<1.0> : tensor<128x64xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x64x128xf32>, tensor<128x64xf32> -> tensor<2x64x64xf32>

    return %0 : tensor<2x64x64xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<128x64xf32> = dense<1.000000e+00> : tensor<128x64xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [128, 128]} : tensor<2x64x128xf32> -> tensor<128x128xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<128x128xf32>, tensor<128x64xf32> -> tensor<128x64xf32>
    // CHECK: %[[OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [2, 64, 64]} : tensor<128x64xf32> -> tensor<2x64x64xf32>
    // CHECK: return %[[OUT]] : tensor<2x64x64xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputWithMulChannel3dWeightsTo2d
func.func @MatMul4dInputWithMulChannel3dWeightsTo2d(%arg0: tensor<1x2x16x2xf32>) -> tensor<1x2x16x2xf32> {
    %cst = const.Declare tensor<1x2x2xf32> = dense<1.0> : tensor<1x2x2xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x16x2xf32>, tensor<1x2x2xf32> -> tensor<1x2x16x2xf32>

    return %0 : tensor<1x2x16x2xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<1x2x2xf32>, [#const.Reshape<[2, 2]>]
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.Reshape(%arg0) {shape_value = [32, 2]} : tensor<1x2x16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<32x2xf32>, tensor<2x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1, 2, 16, 2]} : tensor<32x2xf32> -> tensor<1x2x16x2xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x2x16x2xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInput2dWeightsNBatchTo2d
func.func @MatMul4dInput2dWeightsNBatchTo2d(%arg0: tensor<16x2x16x2xf32>) -> tensor<16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<16x2x16x2xf32>, tensor<4x2xf32> -> tensor<16x2x16x4xf32>

    return %0 : tensor<16x2x16x4xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.Reshape(%arg0) {shape_value = [512, 2]} : tensor<16x2x16x2xf32> -> tensor<512x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_b} : tensor<512x2xf32>, tensor<4x2xf32> -> tensor<512x4xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [16, 2, 16, 4]} : tensor<512x4xf32> -> tensor<16x2x16x4xf32
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul6dInput2dWeights1BatchTo2d
func.func @MatMul6dInput2dWeights1BatchTo2d(%arg0: tensor<1x8x16x2x16x2xf32>) -> tensor<1x8x16x2x16x4xf32> {
    %cst = const.Declare tensor<4x2xf32> = dense<1.0> : tensor<4x2xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x8x16x2x16x2xf32>, tensor<4x2xf32> -> tensor<1x8x16x2x16x4xf32>

    return %0 : tensor<1x8x16x2x16x4xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<4x2xf32> = dense<1.000000e+00> : tensor<4x2xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.Reshape(%arg0) {shape_value = [4096, 2]} : tensor<1x8x16x2x16x2xf32> -> tensor<4096x2xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_b} : tensor<4096x2xf32>, tensor<4x2xf32> -> tensor<4096x4xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1, 8, 16, 2, 16, 4]} : tensor<4096x4xf32> -> tensor<1x8x16x2x16x4xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x8x16x2x16x4xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTo2dNoTranspose
func.func @MatMul5dInput2dWeightsTo2dNoTranspose(%arg0: tensor<5x6x7x8x16xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<5x6x7x8x16xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.Reshape(%arg0) {shape_value = [1680, 16]} : tensor<5x6x7x8x16xf32> -> tensor<1680x16xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<1680x16xf32>, tensor<16x32xf32> -> tensor<1680x32xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [5, 6, 7, 8, 32]} : tensor<1680x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInput2dWeightsTransposeATo3d
func.func @MatMul5dInput2dWeightsTransposeATo3d(%arg0: tensor<5x6x7x16x8xf32>) -> tensor<5x6x7x8x32xf32> {
    %cst = const.Declare tensor<16x32xf32> = dense<1.0> : tensor<16x32xf32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_a} : tensor<5x6x7x16x8xf32>, tensor<16x32xf32> -> tensor<5x6x7x8x32xf32>

    return %0 : tensor<5x6x7x8x32xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<16x32xf32> = dense<1.000000e+00> : tensor<16x32xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.Reshape(%arg0) {shape_value = [210, 16, 8]} : tensor<5x6x7x16x8xf32> -> tensor<210x16x8xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) {transpose_a} : tensor<210x16x8xf32>, tensor<16x32xf32> -> tensor<210x8x32xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [5, 6, 7, 8, 32]} : tensor<210x8x32xf32> -> tensor<5x6x7x8x32xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<5x6x7x8x32xf32>
}

// -----

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2d
func.func @MatMul4dInputs4dWeightsTo2d(%arg0: tensor<2x2x10x3xf32>, %arg1: tensor<2x2x10x3xf32>) -> tensor<2x2x10x10xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<2x2x10x3xf32>, tensor<2x2x10x3xf32> -> tensor<2x2x10x10xf32>

    return %0 : tensor<2x2x10x10xf32>

    // CHECK: %[[RESHAPE_IN1:.*]] = IE.Reshape(%arg0) {shape_value = [4, 10, 3]} : tensor<2x2x10x3xf32> -> tensor<4x10x3xf32>
    // CHECK: %[[RESHAPE_IN2:.*]] = IE.Reshape(%arg1) {shape_value = [4, 10, 3]} : tensor<2x2x10x3xf32> -> tensor<4x10x3xf32>
    // CHECK: %[[SLICE1:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [0, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_1:.*]] = IE.Reshape(%[[SLICE1:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE2:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [1, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_2:.*]] = IE.Reshape(%[[SLICE2:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE3:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [2, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_3:.*]] = IE.Reshape(%[[SLICE3:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE4:.*]] = IE.Slice %[[RESHAPE_IN1:.*]] [3, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_4:.*]] = IE.Reshape(%[[SLICE4:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE5:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [0, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_5:.*]] = IE.Reshape(%[[SLICE5:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE6:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [1, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_6:.*]] = IE.Reshape(%[[SLICE6:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE7:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [2, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_7:.*]] = IE.Reshape(%[[SLICE7:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[SLICE8:.*]] = IE.Slice %[[RESHAPE_IN2:.*]] [3, 0, 0] [1, 10, 3] : tensor<4x10x3xf32> to tensor<1x10x3xf32>
    // CHECK: %[[RESHAPE_8:.*]] = IE.Reshape(%[[SLICE8:.*]]) {shape_value = [10, 3]} : tensor<1x10x3xf32> -> tensor<10x3xf32>
    // CHECK: %[[MATMUL_1:.*]] = IE.MatMul(%[[RESHAPE_1:.*]], %[[RESHAPE_5:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[MATMUL_2:.*]] = IE.MatMul(%[[RESHAPE_2:.*]], %[[RESHAPE_6:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[MATMUL_3:.*]] = IE.MatMul(%[[RESHAPE_3:.*]], %[[RESHAPE_7:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[MATMUL_4:.*]] = IE.MatMul(%[[RESHAPE_4:.*]], %[[RESHAPE_8:.*]]) {transpose_b} : tensor<10x3xf32>, tensor<10x3xf32> -> tensor<10x10xf32>
    // CHECK: %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1:.*]], %[[MATMUL_2:.*]], %[[MATMUL_3:.*]], %[[MATMUL_4:.*]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32> -> tensor<40x10xf32>
    // CHECK: %[[RESHAPE_9:.*]] = IE.Reshape(%[[CONCAT:.*]]) {shape_value = [4, 10, 10]} : tensor<40x10xf32> -> tensor<4x10x10xf32>
    // CHECK: %[[RESHAPE_10:.*]] = IE.Reshape(%[[RESHAPE_9:.*]]) {shape_value = [2, 2, 10, 10]} : tensor<4x10x10xf32> -> tensor<2x2x10x10xf32>
    // CHECK: return %[[RESHAPE_10:.*]] : tensor<2x2x10x10xf32>
}

// -----

// CHECK-LABEL: @MatMulVectorMatrixTo2D
func.func @MatMulVectorMatrixTo2D(%arg0: tensor<1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024x1000xf32> = dense<1.0> : tensor<1024x1000xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1024xf32>, tensor<1024x1000xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1024x1000xf32> = dense<1.000000e+00> : tensor<1024x1000xf32>
    // CHECK:       %[[RESHAPE_IN:.*]] = IE.Reshape(%arg0) {shape_value = [1, 1024]} : tensor<1024xf32> -> tensor<1x1024xf32>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_IN]], %[[CST]]) : tensor<1x1024xf32>, tensor<1024x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1000]} : tensor<1x1000xf32> -> tensor<1000xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMulMatrixVectorTo2D
func.func @MatMulMatrixVectorTo2D(%arg0: tensor<1000x1024xf32>) -> tensor<1000xf32> {
    %cst = const.Declare tensor<1024xf32> = dense<1.0> : tensor<1024xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1000x1024xf32>, tensor<1024xf32> -> tensor<1000xf32>

    return %0 : tensor<1000xf32>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1024x1xf32> = dense<1.000000e+00> : tensor<1024xf32>, [#const.Reshape<[1024, 1]>]
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%arg0, %[[CST]]) : tensor<1000x1024xf32>, tensor<1024x1xf32> -> tensor<1000x1xf32>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1000]} : tensor<1000x1xf32> -> tensor<1000xf32>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1000xf32>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeBTo2d
func.func @MatMul5dInputs3dWeightsTransposeBTo2d(%arg0: tensor<1x1x1x1x2400xf16>, %arg1: tensor<1x256x2400xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x1x1x1x2400xf16>, tensor<1x256x2400xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 2400]} : tensor<1x1x1x1x2400xf16> -> tensor<1x2400xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape(%arg1) {shape_value = [256, 2400]} : tensor<1x256x2400xf16> -> tensor<256x2400xf16>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_0]], %[[RESHAPE_1]]) {transpose_b} : tensor<1x2400xf16>, tensor<256x2400xf16> -> tensor<1x256xf16>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}

// -----

// CHECK-LABEL: @MatMul5dInputs3dWeightsTransposeATo2d
func.func @MatMul5dInputs3dWeightsTransposeATo2d(%arg0: tensor<1x1x1x2400x1xf16>, %arg1: tensor<1x2400x256xf16>) -> tensor<1x1x1x1x256xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<1x1x1x2400x1xf16>, tensor<1x2400x256xf16> -> tensor<1x1x1x1x256xf16>

    return %0 : tensor<1x1x1x1x256xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.Reshape(%arg0) {shape_value = [2400, 1]} : tensor<1x1x1x2400x1xf16> -> tensor<2400x1xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape(%arg1) {shape_value = [2400, 256]} : tensor<1x2400x256xf16> -> tensor<2400x256xf16>
    // CHECK:       %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE_0]], %[[RESHAPE_1]]) {transpose_a} : tensor<2400x1xf16>, tensor<2400x256xf16> -> tensor<1x256xf16>
    // CHECK:       %[[RESHAPE_OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1, 1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x1x256xf16>
    // CHECK:       return %[[RESHAPE_OUT]] : tensor<1x1x1x1x256xf16>
}
