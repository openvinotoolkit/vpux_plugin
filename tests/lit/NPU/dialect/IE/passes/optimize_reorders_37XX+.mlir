//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-reorders %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @ReorderWithConcatNoSwap
module @ReorderWithConcatNoSwap {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x768x64x64xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x768x64x64xf16, {order = #NHWC}>) ->  tensor<1x64x70x768xf16> {
    %cst = const.Declare tensor<768x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[768, 1, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1x768x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>]
    %cst_1 = const.Declare tensor<1x64x6x768xf16> = dense<0.000000e+00> : tensor<1x64x6x768xf16>
    %0 = IE.GroupConvolution(%arg0, %cst, %cst_0) {dilations = [1, 1], groups = 768 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<768x1x1x1xf16, {order = #NHWC}>, tensor<1x768x1x1xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x768x64x64xf16> -> tensor<1x64x64x768xf16, {order = #NWCH}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x768xf16, {order = #NWCH}> -> tensor<1x64x64x768xf16>
    %4 = IE.Concat(%3, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]} : tensor<1x64x64x768xf16>, tensor<1x64x6x768xf16> -> tensor<1x64x70x768xf16>
    return %4 : tensor<1x64x70x768xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<768x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[768, 1, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x768x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>]

    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x64x6x768xf16> = dense<0.000000e+00> : tensor<1x64x6x768xf16>

    // CHECK:       [[GROUP_CONV:%.*]] = IE.GroupConvolution([[ARG0]], [[CST]], [[CST_0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 768 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<768x1x1x1xf16, {order = #NHWC}>, tensor<1x768x1x1xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK:       [[REORDER_0:%.*]] = IE.Reorder([[GROUP_CONV]])
    // CHECK-SAME:      {dstOrder = #NCHW} : tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16>
    // CHECK:       [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[REORDER_0]])
    // CHECK-SAME:      {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x768x64x64xf16> -> tensor<1x64x64x768xf16, {order = #NWCH}>
    // CHECK:       [[REORDER_1:%.*]] = IE.Reorder([[PERMUTE_CAST]])
    // CHECK-SAME:      {dstOrder = #NCHW} : tensor<1x64x64x768xf16, {order = #NWCH}> -> tensor<1x64x64x768xf16>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[REORDER_1]], [[CST_1]])
    // CHECK-SAME:      tensor<1x64x64x768xf16>, tensor<1x64x6x768xf16> -> tensor<1x64x70x768xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x64x70x768xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwitchReorderWithTileForBetterTileDMAPerfCase2
func.func @SwitchReorderWithTileForBetterTileDMAPerfCase2(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x1x1xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    return %1 : tensor<1x64x11x11xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[TILE]]) {dstOrder = #NCHW} : tensor<1x64x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16>
    // CHECK:       return [[REORDER]] : tensor<1x64x11x11xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @FusingNonTrivialReorderAroundTile
func.func @FusingNonTrivialReorderAroundTile(%arg0: tensor<1x1x11x11xf16, {order = #NCWH}> ) -> tensor<1x64x11x11xf16, {order = #NCWH}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x11x11xf16, {order = #NCWH}> -> tensor<1x1x11x11xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16> -> tensor<1x64x11x11xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11xf16, {order = #NCWH}>
    return %2 : tensor<1x64x11x11xf16, {order = #NCWH}>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16, {order = #NCWH}> -> tensor<1x64x11x11xf16, {order = #NCWH}>
    // CHECK        return [[TILE]] : tensor<1x64x11x11xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubtract
module @ReorderWithSubtract {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
func.func @main(%arg0: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16>, tensor<1x32x28x1xf16> -> tensor<1x32x28x1xf16>

    return %2 : tensor<1x32x28x1xf16>

    // CHECK:       [[SUB:%.+]] = IE.Subtract([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16, {order = #NHWC}>, tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[SUB]]) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    // CHECK:       return [[REORDER]] : tensor<1x32x28x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwDivideHasChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>
func.func @ReorderWithSwDivideHasChange(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x30x30xf16> = dense<0.1> : tensor<1x1x30x30xf16>
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.Divide(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare tensor<1x1x30x30xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x30x30xf16>, [#const.Reorder<#NHWC>]
    // CHECK:           [[VAR1:%.+]] = IE.Divide([[INPUT]], [[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK            return [[VAR1]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}
