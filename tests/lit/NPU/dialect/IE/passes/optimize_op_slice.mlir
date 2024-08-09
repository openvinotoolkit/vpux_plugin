//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-op-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @NoChangesAcrossInputsAtHeight
func.func @NoChangesAcrossInputsAtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

    // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL}    {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} :
    // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK:        [[VAR1:%.+]] = IE.Slice [[VAR0]]
    // CHECK-SAME:          [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    // CHECK: return [[VAR1]] : tensor<1x16x3x4xf16>
}

// CHECK-LABEL: @ChangesInInput0AtHeight
func.func @ChangesInInput0AtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x4x4xf16> to tensor<1x16x3x4xf16>
   // CHECK: return [[VAR0]] : tensor<1x16x3x4xf16>
}

// CHECK-LABEL: @ChangesInInput1SameShapeAtHeight
func.func @ChangesInInput1SameShapeAtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: return %arg1 : tensor<1x16x3x4xf16>
}

// CHECK-LABEL: @ChangesAtHeightTwoUsers
func.func @ChangesAtHeightTwoUsers(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> (tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    %2 = IE.Slice %0 [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1, %2 : tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} :
   // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
   // CHECK: [[VAR1:%.+]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x4x4xf16> to tensor<1x16x3x4xf16>
   // CHECK: [[VAR2:%.+]] = IE.Slice [[VAR0]]
   // CHECK-SAME:           [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
   // CHECK:  return [[VAR1]], [[VAR2:%.+]] : tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>

}

// CHECK-LABEL: @NoChangesNoOffsetAttribute
func.func @NoChangesNoOffsetAttribute(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) { per_axis = #IE.Concat<axis = 2 : i64> } : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

    // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK: [[VAR1:%.+]] = IE.Slice [[VAR0]] [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    // CHECK: return [[VAR1:%.+]] : tensor<1x16x3x4xf16>

}

// CHECK-LABEL: @NoChangesAtHeightWidth
func.func @NoChangesAtHeightWidth(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>, %arg2 : tensor<1x16x7x3xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
    %1 = IE.Slice %0 [0, 0, 2, 2] [1, 16, 3, 4] : tensor<1x16x7x7xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1, %arg2)
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} :
   // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
   // CHECK: [[VAR1:%.+]] = IE.Slice [[VAR0]]
   // CHECK-SAME:           [0, 0, 2, 2] [1, 16, 3, 4] : tensor<1x16x7x7xf16> to tensor<1x16x3x4xf16>
   // CHECK:  return [[VAR1:%.+]] : tensor<1x16x3x4xf16>

}

// CHECK-LABEL: @ChangesAtHeightWidth
func.func @ChangesAtHeightWidth(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>, %arg2 : tensor<1x16x7x3xf16>) -> tensor<1x16x2x2xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
    %1 = IE.Slice %0 [0, 0, 5, 1] [1, 16, 2, 2] : tensor<1x16x7x7xf16> to tensor<1x16x2x2xf16>
    return %1 : tensor<1x16x2x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Slice %arg1 [0, 0, 1, 1] [1, 16, 2, 2] : tensor<1x16x3x4xf16> to tensor<1x16x2x2xf16>
    // CHECK: return [[VAR0]] : tensor<1x16x2x2xf16>

}

// CHECK-LABEL: @ChangesAtChannelHeight
func.func @ChangesAtChannelHeight(%arg0: tensor<1x3x4x4xf16>, %arg1 : tensor<1x3x3x4xf16>, %arg2 : tensor<1x2x7x4xf16>) -> tensor<1x2x2x2xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 3, 0, 0]]} : tensor<1x3x4x4xf16>, tensor<1x3x3x4xf16>, tensor<1x2x7x4xf16> -> tensor<1x5x7x4xf16>
    %1 = IE.Slice %0 [0, 1, 5, 1] [1, 2, 2, 2] : tensor<1x5x7x4xf16> to tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Slice %arg1 [0, 1, 1, 1] [1, 2, 2, 2] : tensor<1x3x3x4xf16> to tensor<1x2x2x2xf16>
    // CHECK: return [[VAR0]] : tensor<1x2x2x2xf16>

}

// CHECK-LABEL: @NoChangesAcrossInputsAtHeightWithAffinReshape
func.func @NoChangesAcrossInputsAtHeightWithAffinReshape(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x1x12xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 16, 1, 28]} : tensor<1x16x7x4xf16> -> tensor<1x16x1x28xf16>
    %2 = IE.Slice %1 [0, 0, 0, 8] [1, 16, 1, 12] : tensor<1x16x1x28xf16> to tensor<1x16x1x12xf16>
    return %2 : tensor<1x16x1x12xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    // CHECK-SAME{LITERAL}    {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK:        [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}    {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 16, 1, 28]} : tensor<1x16x7x4xf16> -> tensor<1x16x1x28xf16>
    // CHECK:        [[SLICE:%.*]] = IE.Slice [[AFFINERESHAPE]] [0, 0, 0, 8] [1, 16, 1, 12] : tensor<1x16x1x28xf16> to tensor<1x16x1x12xf16>
    // CHECK:        return [[SLICE]] : tensor<1x16x1x12xf16>

}

// CHECK-LABEL: @ChangesInInput0AtHeightWithAffinReshape
func.func @ChangesInInput0AtHeightWithAffinReshape(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>, %arg2: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16> {
    %3 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x3xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 3, 1, 1]} : tensor<1x1x1x3xf16> -> tensor<1x3x1x1xf16>
    %5 = IE.Slice %4 [0, 1, 0, 0] [1, 1, 1, 1] : tensor<1x3x1x1xf16> to tensor<1x1x1x1xf16>
    return %5 : tensor<1x1x1x1xf16>

    // CHECK: return {{[^:]+}} : tensor<1x1x1x1xf16>

}

// CHECK-LABEL: @TileAffineReshapeSliceOpt
func.func @TileAffineReshapeSliceOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x249xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 300]} : tensor<1x128x3x100xf16> -> tensor<1x128x300xf16>
    %2 = IE.Slice %1 [0, 0, 0] [1, 128, 249] : tensor<1x128x300xf16> to tensor<1x128x249xf16>

    return %2 : tensor<1x128x249xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:        [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}    {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 249]} : tensor<1x128x3x83xf16> -> tensor<1x128x249xf16>
    // CHECK:        return [[AFFINERESHAPE]] : tensor<1x128x249xf16>

}

// The SliceOp output shape is 1x128x248, the data on dim 2 with shape "248" is sliced from TileOp output 3x100. 248 is not divisible by 3.
// So the pattern should not be optimized.
// CHECK-LABEL: @TileAffineReshapeSliceNotOpt
func.func @TileAffineReshapeSliceNotOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x248xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 300]} : tensor<1x128x3x100xf16> -> tensor<1x128x300xf16>
    %2 = IE.Slice %1 [0, 0, 0] [1, 128, 248] : tensor<1x128x300xf16> to tensor<1x128x248xf16>

    return %2 : tensor<1x128x248xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    // CHECK:        [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[TILE]])
    // CHECK-SAME{LITERAL}    {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 128, 300]} : tensor<1x128x3x100xf16> -> tensor<1x128x300xf16>
    // CHECK:        [[SLICE:%.*]] = IE.Slice [[AFFINERESHAPE]] [0, 0, 0] [1, 128, 248] : tensor<1x128x300xf16> to tensor<1x128x248xf16>
    // CHECK:        return [[SLICE]] : tensor<1x128x248xf16>

}

// CHECK-LABEL: @TileSliceOpt
func.func @TileSliceOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x3x83xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 128, 3, 83] : tensor<1x128x3x100xf16> to tensor<1x128x3x83xf16>

    return %1 : tensor<1x128x3x83xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 83]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x83xf16>
    // CHECK:        return [[TILE]] : tensor<1x128x3x83xf16>

}

// The slice performs on 2 Axes, so the pattern should not be optimized.
// CHECK-LABEL: @TileSliceNotOpt
func.func @TileSliceNotOpt(%arg0: tensor<1x128x3x1xf16>) -> tensor<1x128x2x83xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 128, 2, 83] : tensor<1x128x3x100xf16> to tensor<1x128x2x83xf16>

    return %1 : tensor<1x128x2x83xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 1, 100]} : tensor<1x128x3x1xf16> -> tensor<1x128x3x100xf16>
    // CHECK:        [[SLICE:%.*]] = IE.Slice [[TILE]] [0, 0, 0, 0] [1, 128, 2, 83] : tensor<1x128x3x100xf16> to tensor<1x128x2x83xf16>
    // CHECK:        return [[SLICE]] : tensor<1x128x2x83xf16>

}
