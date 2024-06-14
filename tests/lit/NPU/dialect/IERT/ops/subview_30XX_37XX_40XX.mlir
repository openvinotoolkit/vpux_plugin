//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: memref<1x3x8x4xf32, #NHWC>) -> memref<1x3x8x4xf32, #NHWC> {
    %0 = const.Declare memref<1x3x16x16xf32, #NHWC> =
        dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    %1 = IERT.SubView %0 [0, 0, 8, 12] [1, 3, 8, 4] :
        memref<1x3x16x16xf32, #NHWC> to
        memref<1x3x8x4xf32, {order = #NHWC, strides = [768, 1, 48, 3]}>

    %2 = IERT.Copy
        inputs(%1 : memref<1x3x8x4xf32, {order = #NHWC, strides = [768, 1, 48, 3]}>)
        outputs(%arg0 : memref<1x3x8x4xf32, #NHWC>)
        -> memref<1x3x8x4xf32, #NHWC>

    return %2 : memref<1x3x8x4xf32, #NHWC>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare memref<1x3x8x4xf32, #NHWC> =
    // CHECK-SAME:      [#const.SubView<[0, 0, 8, 12], [1, 3, 8, 4]>, #const.Reorder<#NHWC>]
    // CHECK-NOT:   IERT.SubView

    // CHECK:       [[VAR1:%.+]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x3x8x4xf32, #NHWC>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x3x8x4xf32, #NHWC>)

    // CHECK:       return [[VAR1]] : memref<1x3x8x4xf32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ComposeSubView
func.func @ComposeSubView(%arg0: memref<1x3x8x4xf32>) -> memref<1x3x8x4xf32> {
    %0 = memref.alloc() : memref<1x3x16x16xf32>

    %1 = IERT.SubView %0 [0, 0, 0, 8] [1, 3, 16, 8] :
        memref<1x3x16x16xf32> to
        memref<1x3x16x8xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>

    %2 = IERT.SubView %1 [0, 0, 8, 4] [1, 3, 8, 4] :
        memref<1x3x16x8xf32, {order = #NCHW, strides = [768, 256, 16, 1]}> to
        memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>

    %3 = IERT.ReLU
        inputs(%2 : memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>)
        outputs(%arg0 : memref<1x3x8x4xf32>)
        -> memref<1x3x8x4xf32>

    return %3 : memref<1x3x8x4xf32>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x3x16x16xf32>

    // CHECK:       [[VAR1:%.*]] = IERT.SubView [[VAR0]] [0, 0, 8, 12] [1, 3, 8, 4] :
    // CHECK-SAME:      memref<1x3x16x16xf32> to memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>

    // CHECK:       [[VAR2:%.*]] = IERT.ReLU
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x3x8x4xf32>)
    // CHECK-SAME:      -> memref<1x3x8x4xf32>

    // CHECK:       return [[VAR2]] : memref<1x3x8x4xf32>
}
