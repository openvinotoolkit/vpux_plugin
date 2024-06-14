
//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#C = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @StridedSliceWithNonConstEnds
func.func @StridedSliceWithNonConstEnds(%arg0: tensor<1xsi64>) -> tensor<?xf32, {bounds = [9], order = #C}> {
// CHECK:  ([[ARG0:[^:]+]]: tensor<1xsi64>)
    %cst = const.Declare tensor<9xf32> = dense<[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.000000e+00]> : tensor<9xf32>
    %cst_0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
    %cst_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    %0 = IE.StridedSlice(%cst, %cst_0, %arg0, %cst_1) {begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []}
       : tensor<9xf32>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<?xf32, {bounds = [9], order = #C}>

    return %0 : tensor<?xf32, {bounds = [9], order = #C}>
    // CHECK: [[CONST0:%.*]] = const.Declare tensor<9xf32>
    // CHECK: [[CONST1:%.*]] = const.Declare tensor<1xsi64>
    // CHECK: [[CONST2:%.*]] = const.Declare tensor<1xsi64>
    // CHECK: [[VAR0:%.+]] = VPU.StridedSlice([[CONST0]], [[CONST1]], [[ARG0]], [[CONST2]]) {
    // CHECK-SAME:      begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []
    // CHECK-SAME: } : tensor<9xf32>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<?xf32, {bounds = [9], order = #C}>

    // CHECK: return [[VAR0]]
}


// -----

// CHECK-LABEL: @GatherNDDynamicIndices
// CHECK:           ([[INPUT:%.*]]: tensor<1x88xsi32>, [[INDICES:%.*]]: tensor<?x2xsi32, {bounds = [88, 2], order = #NC}>) -> tensor<?xsi32, {bounds = [88], order = #C}>
func.func @GatherNDDynamicIndices(%arg0: tensor<1x88xsi32>, %arg1: tensor<?x2xsi32, {bounds = [88, 2], order = affine_map<(d0, d1) -> (d0, d1)>}>)
        -> tensor<?xsi32, {bounds = [88], order = affine_map<(d0) -> (d0)>}> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 0 : i64} : tensor<1x88xsi32>, tensor<?x2xsi32, {bounds = [88, 2], order = affine_map<(d0, d1) -> (d0, d1)>}>
        -> tensor<?xsi32, {bounds = [88], order = affine_map<(d0) -> (d0)>}>
    return %0 : tensor<?xsi32, {bounds = [88], order = affine_map<(d0) -> (d0)>}>

    // CHECK-NOT:   IE.GatherND
    // CHECK:       [[VAR0:%.+]] = VPU.GatherND([[INPUT]], [[INDICES]]) {batch_dims = 0 : i64} : tensor<1x88xsi32>, tensor<?x2xsi32, {bounds = [88, 2], order = #NC}> -> tensor<?xsi32, {bounds = [88], order = #C}>
    // CHECK:       return [[VAR0]] : tensor<?xsi32, {bounds = [88], order = #C}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAccumulateWithScales
func.func @ConvertAccumulateWithScales(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALE: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALE: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:   ([[LHS:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>, [[RHS:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  [[LHS_SCALE:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>, [[RHS_SCALE:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    %ACCUMULATE = IE.Accumulate(%LHS, %RHS, %LHS_SCALE, %RHS_SCALE) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.Accumulate([[LHS]], [[RHS]], [[LHS_SCALE]], [[RHS_SCALE]]) :
    // CHECK-SAME:  tensor<1x64x16x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x64x16x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x64x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[ACCUMULATE]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}
