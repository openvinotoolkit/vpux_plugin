//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --strategy-manager="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOKTiling
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func.func @TileWithSOKTiling(%arg0 : tensor<1x32x30x30xf16, {order = #NHWC}>)->tensor<1x768x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1> :
            tensor<768x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [ 768, 32, 7, 7 ], strides = [ 1, 1 ]
    } ->tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:          : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1>
    // CHECK-SAME:          : tensor<768x1x1x4xsi32>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
    // CHECK-SAME:          rawFilterShape = [768, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:        -> tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[CONV1]] : tensor<1x768x30x30xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @GridSampleSplit
func.func @GridSampleSplit(%arg0: tensor<1x3x272x480xf16>, %arg1: tensor<1x272x480x2xf16>) -> tensor<1x3x272x480xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>
    return %0 : tensor<1x3x272x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GridSample(%arg0, %arg1)
    // CHECK-NOT:           multiClusterStrategy = #VPU.multi_cluster_strategy
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:     : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>

    // CHECK:       return [[OUTPUT]] :  tensor<1x3x272x480xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x273x40x40xf16>) -> tensor<1x40x40x273xf16>
func.func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x273x40x40xf16>) -> tensor<1x40x40x273xf16> {
    %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x273x40x40xf16> -> tensor<1x40x40x273xf16>
    return %0 : tensor<1x40x40x273xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MemPermute([[INPUT]]) {
    // CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC
    // CHECK-NOT:   tilingStrategy
    // CHECK-SAME:  } : tensor<1x273x40x40xf16> -> tensor<1x40x40x273xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x40x40x273xf16>
}
