//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @EmbeddingBagOffsetsSumWithWeights
func.func @EmbeddingBagOffsetsSumWithWeights(%arg0: tensor<5x6xui8>) -> tensor<2x6xui8> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %cst_1 = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %cst_2 = const.Declare tensor<5xui8> = dense<[1, 5, 10, 8, 10]> : tensor<5xui8>
    %0 = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0, %cst_1, %cst_2) {default_index_value = 0 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6xui8>, tensor<5xsi32>, tensor<2xsi32>, tensor<si32>, tensor<5xui8> -> tensor<2x6xui8>
    return %0 : tensor<2x6xui8>

    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 0 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>, per_sample_weights_value = [1.000000e+00, 5.000000e+00, 1.000000e+01, 8.000000e+00, 1.000000e+01]} : tensor<5x6xui8> -> tensor<2x6xui8>
    // CHECK: return [[VAR0]] : tensor<2x6xui8>
}

// -----

// CHECK-LABEL: @EmbeddingBagOffsetsSumNoWeights
func.func @EmbeddingBagOffsetsSumNoWeights(%arg0: tensor<5x6xui8>) -> tensor<2x6xui8> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %cst_1 = const.Declare tensor<si32> = dense<0> : tensor<si32>
    %0 = IE.EmbeddingBagOffsetsSum(%arg0, %cst, %cst_0, %cst_1) {default_index_value = 0 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, per_sample_weights_value = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<5x6xui8>, tensor<5xsi32>, tensor<2xsi32>, tensor<si32> -> tensor<2x6xui8>
    return %0 : tensor<2x6xui8>

    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagOffsetsSum(%arg0) {default_index_value = 0 : i32, indices_value = [0, 1, 2, 2, 3], offsets_value = [0, 2], operandSegmentSizes = array<i32: 1, 0, 0, 0>, per_sample_weights_value = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]} : tensor<5x6xui8> -> tensor<2x6xui8>
    // CHECK: return [[VAR0]] : tensor<2x6xui8>
}

// -----

// CHECK-LABEL: @EmbeddingSegmentsSumWithWeights
func.func @EmbeddingSegmentsSumWithWeights(%arg0: tensor<5x6x4xui8>) -> tensor<7x6x4xui8> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    %cst_1 = const.Declare tensor<5xui8> = dense<[1, 5, 10, 8, 10]> : tensor<5xui8>
    %0 = IE.EmbeddingSegmentsSum(%arg0, %cst, %cst_0, %cst_1) {default_index_value = 0 : i32, num_segments_value = 7 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>} : tensor<5x6x4xui8>, tensor<5xsi32>, tensor<5xsi32>, tensor<5xui8> -> tensor<7x6x4xui8>
    return %0 : tensor<7x6x4xui8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<5xui8> = dense<[1, 5, 10, 8, 10]> : tensor<5xui8>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingSegmentsSum(%arg0) {default_index_value = 0 : i32, num_segments_value = 7 : i32, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<5x6x4xui8> -> tensor<7x6x4xui8>
    // CHECK: return [[VAR0]] : tensor<7x6x4xui8>
}

// -----

// CHECK-LABEL: @EmbeddingSegmentsSumNoWeights
func.func @EmbeddingSegmentsSumNoWeights(%arg0: tensor<5x6x4xui8>) -> tensor<7x6x4xui8> {
    %cst = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %cst_0 = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    %0 = IE.EmbeddingSegmentsSum(%arg0, %cst, %cst_0) {default_index_value = -1 : i32, num_segments_value = 7 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : tensor<5x6x4xui8>, tensor<5xsi32>, tensor<5xsi32> -> tensor<7x6x4xui8>
    return %0 : tensor<7x6x4xui8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 3, 4]> : tensor<5xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingSegmentsSum(%arg0) {default_index_value = -1 : i32, num_segments_value = 7 : i32, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<5x6x4xui8> -> tensor<7x6x4xui8>
    // CHECK: return [[VAR0]] : tensor<7x6x4xui8>
}
