//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-tile-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @FoldTileBeforeMultiply(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 16]} : tensor<1x1x1x1xf32> -> tensor<1x1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Multiply(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Multiply
}

func.func @FoldTileBeforeMultiplyWith3DInput(%arg0: tensor<1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 16]} : tensor<1x1x1xf32> -> tensor<1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Multiply(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Reshape
    // CHECK-SAME:       {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:        IE.Multiply
}

func.func @FoldTileBeforeAdd(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 16]} : tensor<1x1x1x1xf32> -> tensor<1x1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Add(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Add
}

func.func @FoldTileBeforeAddWith3DInput(%arg0: tensor<1x1x1xf32>) -> tensor<1x1x4x4xf32> {
    %cst_0 = const.Declare tensor<1x1x4x4xf32> = dense<1.0> : tensor<1x1x4x4xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 16]} : tensor<1x1x1xf32> -> tensor<1x1x16xf32>
    %1 = IE.Reshape(%0) { shape_value = [1, 1, 4, 4] } : tensor<1x1x16xf32> -> tensor<1x1x4x4xf32>
    %2 = IE.Add(%1, %cst_0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf32>

    return %2 : tensor<1x1x4x4xf32>

    // CHECK-NOT:    IE.Tile(
    // CHECK:        IE.Reshape
    // CHECK-SAME:       {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:        IE.Add
}
