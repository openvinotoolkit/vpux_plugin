//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-depthToSpace %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @Depth2SpaceCanConvertToNNDMAs_BLOCKS_FIRST
func.func @Depth2SpaceCanConvertToNNDMAs_BLOCKS_FIRST(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x256x256xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    // CHECK:       [[DepthToSpace:%.*]] = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>
    // CHECK:       return [[DepthToSpace]] : tensor<1x1x256x256xf16>
}

//
// -----
//

// CHECK-LABEL: @Depth2SpaceCanConvertToNNDMAs_DEPTH_FIRST
func.func @Depth2SpaceCanConvertToNNDMAs_DEPTH_FIRST(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x256x256xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    // CHECK:       [[DepthToSpace:%.*]] = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>
    // CHECK:       return [[DepthToSpace]] : tensor<1x1x256x256xf16>
}

//
// -----
//

// CHECK-LABEL: @Depth2SpaceCannotConvertToNNDMAsWithLargeSize_DEPTH_FIRST
func.func @Depth2SpaceCannotConvertToNNDMAsWithLargeSize_DEPTH_FIRST(%arg0: tensor<1x64x256x256xf16>) -> tensor<1x4x1024x1024xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x64x256x256xf16> -> tensor<1x4x1024x1024xf16>

    return %0 : tensor<1x4x1024x1024xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 4, 4, 4, 256, 256]} : tensor<1x64x256x256xf16> -> tensor<1x4x4x4x256x256xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #map} : tensor<1x4x4x4x256x256xf16> -> tensor<1x4x256x4x256x4xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [1, 4, 1024, 1024]} : tensor<1x4x256x4x256x4xf16> -> tensor<1x4x1024x1024xf16>
    // CHECK:       return [[VAL2]] : tensor<1x4x1024x1024xf16>
}
