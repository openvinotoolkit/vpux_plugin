//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @FoldTile
func.func @FoldTile(%arg0: tensor<3x4x2xf32>) -> tensor<3x4x2xf32> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1]} : tensor<3x4x2xf32> -> tensor<3x4x2xf32>
    // CHECK-NOT:   IE.Tile
    return %0 : tensor<3x4x2xf32>
    // CHECK:       return %arg0
}

// CHECK-LABEL: @FoldTileWithBroadcast
func.func @FoldTileWithBroadcast() -> tensor<4x6xsi64> {
    %0 = const.Declare tensor<2x2xsi64> = dense<[[0, 1], [2, 3]]> : tensor<2x2xsi64>
    // CHECK-NOT:   IE.Tile
    %1 = IE.Tile(%0) {repeats_values = [2, 3]} : tensor<2x2xsi64> -> tensor<4x6xsi64>
    // CHECK:       %[[VAL0:.*]] = const.Declare tensor<4x6xsi64> =
    // CHECK-SAME{LITERAL}:          dense<[[0, 1], [2, 3]]> : tensor<2x2xsi64>,
    // CHECK-SAME:                  [#const.Broadcast<0 : i64, 4 : i64>, #const.Broadcast<1 : i64, 6 : i64>]
    return %1 : tensor<4x6xsi64>
    // CHECK:       return %[[VAL0]]
}

// CHECK-LABEL: @InsertUnsqueezeBeforedTile
func.func @InsertUnsqueezeBeforedTile(%arg0: tensor<2x3xf32>) -> tensor<1x6x15xf32> {
    // CHECK:       %[[VAL0:.*]] = IE.Unsqueeze(%arg0) {axes_value = [0]} : tensor<2x3xf32> -> tensor<1x2x3xf32>
    %0 = IE.Tile(%arg0) {repeats_values = [1, 3, 5]} : tensor<2x3xf32> -> tensor<1x6x15xf32>
    // CHECK:       %[[VAL1:.*]] = IE.Tile(%[[VAL0]]) {repeats_values = [1, 3, 5]} : tensor<1x2x3xf32> -> tensor<1x6x15xf32>

    return %0 : tensor<1x6x15xf32>
    // CHECK:       return %[[VAL1]]
}
