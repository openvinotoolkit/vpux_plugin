//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-mem-permute-around-op %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiply
func.func @AdjustMemPermutesAroundMultiply(%arg0: tensor<1x1x51x1xf16, {order = #NCWH}>, %arg1: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_CAST:%.*]] = IE.PermuteCast(%arg0)
    // CHECK:            {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x51x1xf16, {order = #NCWH}> -> tensor<1x1x51x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[PERMUTE_CAST]], %arg1)
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NHWC}>, tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithConstInput
func.func @AdjustMemPermutesAroundMultiplyWithConstInput(%arg0: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x51x1xf16, {order = #NCWH}> = dense<2.0> : tensor<1x1x51x1xf16>, [#const.Reorder<#NCWH>]
    %0 = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[CST:%.*]] = const.Declare tensor<1x1x51x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x1x51x1xf16>,
    // CHECK-SAME:            [#const.Reorder<#NCWH>, #const.MemPermute<#NHWC, #NWHC>]
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<1x1x51x1xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithPermuteQuantizeInput
func.func @AdjustMemPermutesAroundMultiplyWithPermuteQuantizeInput(%arg0: tensor<1x1x51x1xf16, {order = #NCWH}>, %arg1: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NCWH, mem_perm = #NWHC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_CAST:%.*]] = IE.PermuteCast(%arg0)
    // CHECK:            {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x51x1xf16, {order = #NCWH}> -> tensor<1x1x51x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[PERMUTE_CAST]], %arg1)
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NHWC}>, tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithoutDeadLoop
func.func @AdjustMemPermutesAroundMultiplyWithoutDeadLoop(%arg0: tensor<1x128x16x64xf16, {order = #NHWC}>, %arg1: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x128x16x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NWCH, mem_perm = #NWCH} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16, {order = #NWCH}>
    %1 = IE.PermuteCast(%arg1) {dst_order = #NWCH, mem_perm = #NHWC} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16, {order = #NWCH}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x128xf16, {order = #NWCH}>, tensor<1x16x1x128xf16, {order = #NWCH}> -> tensor<1x16x64x128xf16, {order = #NWCH}>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x64x128xf16, {order = #NWCH}> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    return %3 : tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:        [[IN_PERMUTE_CAST:%.*]] = IE.PermuteCast(%arg0)
    // CHECK:            {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    // CHECK:        [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1)
    // CHECK:            {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[IN_PERMUTE_CAST]], [[MEM_PERMUTE]])
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x64x128xf16>
    // CHECK:        [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[MULTIPLY]])
    // CHECK:            {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x64x128xf16> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    // CHECK:        return [[OUT_PERMUTE_CAST]] : tensor<1x128x16x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotAdjustMemPermutesAroundMultiply
func.func @NotAdjustMemPermutesAroundMultiply(%arg0: tensor<1x1x51x1xf16>, %arg1: tensor<1x128x51x64xf16>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16>, tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    return %1 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16>, tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16>
    // CHECK:        [[PERMUTE:%.*]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[PERMUTE]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustInputMemPermutesToOutput
func.func @AdjustInputMemPermutesToOutput(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16>, tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK:        [[PERMUTE:%.*]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[PERMUTE]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotAdjustInputMemPermutesToOutput
func.func @NotAdjustInputMemPermutesToOutput(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16, {order = #NHWC}>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_L:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE_R:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[PERMUTE_L]], [[PERMUTE_R]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustInputMemPermutesWithMultipleMultiplyUsers
func.func @AdjustInputMemPermutesWithMultipleMultiplyUsers(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16>, tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK:        [[PERMUTE_0:%.*]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE_1:%.*]] = IE.MemPermute([[PERMUTE_0]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[ADD:%.*]] = IE.Add([[PERMUTE_0]], [[PERMUTE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[ADD]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}


// CHECK-LABEL: @AdjustMemPermutesAfterTile
func.func @AdjustMemPermutesAfterTile(%arg0: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    return %1 : tensor<1x2x512x512xf16>

    // CHECK:        [[PERMUTE:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x1x1x512xf16>
    // CHECK:        [[TILE:%.*]] = IE.Tile([[PERMUTE]]) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16> -> tensor<1x2x512x512xf16>
    // CHECK:        return [[TILE]] : tensor<1x2x512x512xf16>
}

// CHECK-LABEL: @NotAdjustMemPermutesAfterTile
func.func @NotAdjustMemPermutesAfterTile(%arg0: tensor<1x2x256x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 2, 1]} : tensor<1x2x256x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    return %1 : tensor<1x2x512x512xf16>

    // CHECK:        [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 1, 2, 1]} : tensor<1x2x256x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE:%.*]] = IE.MemPermute([[TILE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:        return [[PERMUTE]] : tensor<1x2x512x512xf16>
}

// CHECK-LABEL: @NotAdjustMemPermutesLayoutNotSupport
func.func @NotAdjustMemPermutesLayoutNotSupport(%arg0: tensor<1x32x16x16xf16>, %arg1: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE0:%.*]] = IE.MemPermute(%arg0)
    // CHECK:        [[PERMUTE1:%.*]] = IE.MemPermute(%arg1)
    // CHECK:        [[ADD:%.*]] = IE.Add([[PERMUTE0]], [[PERMUTE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:        return [[ADD]] : tensor<1x32x16x16xf16, {order = #NHWC}>
}
