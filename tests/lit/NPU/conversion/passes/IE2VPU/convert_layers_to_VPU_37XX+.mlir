
//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

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

// -----

!qElemType = !quant.uniform<i4:f16:1, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>
!qElemType1 = !quant.uniform<i4:f16:0, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>

// CHECK-LABEL: @ConvertAffineReshapeOnQuantAxis
func.func @ConvertAffineReshapeOnQuantAxis(%arg0: tensor<1x4x1x4096x!qElemType>) -> tensor<4x4096x1x1x!qElemType1> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 4096, 1, 1]} :
        tensor<1x4x1x4096x!qElemType> -> tensor<4x4096x1x1x!qElemType1>
    // CHECK:   [[RESHAPE:%.*]] = VPU.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 4096, 1, 1]} :
    // CHECK-SAME:    tensor<1x4x1x4096x!qElemType> -> tensor<4x4096x1x1x!qElemType1>
    return %0 : tensor<4x4096x1x1x!qElemType1>
    // CHECK:    return [[RESHAPE]] : tensor<4x4096x1x1x!qElemType1>
}

// -----

// CHECK-LABEL: @MaxPool8
// CHECK:           ([[ARG0:%.+]]: tensor<1x3x30x30xf16>)
func.func @MaxPool8(%arg0: tensor<1x3x30x30xf16>) -> (tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>) {
    %output, %output_index = IE.MaxPool8(%arg0) {axis = 0 : i64, dilations = [2, 2], index_element_type = si32, kernel_size = [3, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
    return %output, %output_index : tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>

    // CHECK:   [[MAXPOOL8:%.+]], [[INDICES:%.+]] = VPU.MaxPool8([[ARG0]]) {axis = 0 : i64, dilations = [2, 2], index_element_type = si32, kernel_size = [3, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
    // CHECK:   return [[MAXPOOL8]], [[INDICES]] : tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#OYXI = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDilatedGroupConv
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x80x56x56xf16, {order = #NHWC}>
func.func @ConvertDilatedGroupConv(%arg: tensor<1x80x56x56xf16, {order = #NHWC}>) ->  tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x80x1x1xf16>
    %cst_0 = const.Declare tensor<80x1x3x3xf16, {order = #OYXI}> = dense<1.000000e+00> : tensor<80x1x3x3xf16>, [#const.Reorder<#OYXI>]
    %1 = IE.GroupConvolution(%arg, %cst_0, %cst)
    {dilations = [3, 3], groups = 80 : i64, pads_begin = [3, 3], pads_end = [3, 3], strides = [2, 2]}
     : tensor<1x80x56x56xf16, {order = #NHWC}>,
      tensor<80x1x3x3xf16, {order = #OYXI}>,
        tensor<1x80x1x1xf16> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 :  tensor<1x80x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x80x1x1xf16>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x1x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[GROUPCONV:%.+]] = VPU.GroupConvolution([[ARG0]], [[CST0]], [[CST]]) {dilations = [3, 3], groups = 80 : i64, pads_begin = [3, 3],
    // CHECK-SAME:  pads_end = [3, 3], strides = [2, 2]} : tensor<1x80x56x56xf16, {order = #NHWC}>, tensor<80x1x3x3xf16, {order = #NHWC}>, tensor<1x80x1x1xf16>
    // CHECK-SAME:  -> tensor<1x80x28x28xf16, {order = #NHWC}>
    // CHECK: return [[GROUPCONV]] :  tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DynamicUnsqueeze
func.func @DynamicUnsqueeze(%arg0: tensor<1x1x?xf16, {bounds = [1, 1, 10], order = #CHW}>) -> tensor<1x1x?x1xf16, {bounds = [1, 1, 10, 1], order = #NCHW}> {
    %0 = IE.Unsqueeze(%arg0) {axes_value = [3]} : tensor<1x1x?xf16, {bounds = [1, 1, 10], order = #CHW}> -> tensor<1x1x?x1xf16, {bounds = [1, 1, 10, 1], order = #NCHW}>
    return %0 : tensor<1x1x?x1xf16, {bounds = [1, 1, 10, 1], order = #NCHW}>
    // CHECK:       VPU.Unsqueeze
    // CHECK-SAME:      tensor<1x1x?xf16, {bounds = [1, 1, 10], order = #CHW}>
    // CHECK-SAME:      -> tensor<1x1x?x1xf16, {bounds = [1, 1, 10, 1], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicTileFromBroadcast_case0 {
IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_1" : tensor<1x1x1x1xsi64>
    DataInfo "input_0" : tensor<1x1x10x5xsi64>
  } outputsInfo : {
    DataInfo "Broadcast_63" friendlyName = "Result_67" : tensor<1x1x10x5xsi64>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1xsi64>, [[ARG1:%.+]]: tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>) -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> {
  func.func @main(%arg0: tensor<1x1x1x1xsi64>, %arg1: tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>) -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> {
    %0 = IE.Convert(%arg1) {dstElemType = si32} : tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> -> tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}>
    %1 = IE.ShapeOf(%0) {dstElemType = si32} : tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}> -> tensor<4xsi32>
    %2 = IE.Convert(%arg0) {dstElemType = si32} : tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi32>
    %3 = IE.DynamicTile(%2, %1) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x1xsi32>, tensor<4xsi32> -> tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}>
    %4 = IE.Convert(%3) {dstElemType = si64} : tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}> -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>
    return %4 : tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>

    // CHECK-NOT:   IE.DynamicTile

    // CHECK:       [[CONVERT_0:%.+]] = VPU.Convert([[ARG1]]) {dstElemType = si32} : tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> -> tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}>
    // CHECK:       [[SHAPEOF:%.+]] = VPU.ShapeOf([[CONVERT_0]]) : tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}> -> tensor<4xsi32>
    // CHECK:       [[CONVERT_1:%.+]] = VPU.Convert([[ARG0]]) {dstElemType = si32} : tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[TILE:%.+]] = VPU.DynamicTile([[CONVERT_1]], [[SHAPEOF]]) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x1xsi32>, tensor<4xsi32> -> tensor<1x1x?x?xsi32, {bounds = [1, 1, 10, 5], order = #NCHW}>
  }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func @DynamicTileFromBroadcast_case1([[ARG0:%.+]]: tensor<1x1x?xsi64, {bounds = [1, 1, 10], order = #CHW}>, [[ARG1:%.+]]: tensor<4xsi32>) -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> {
func.func @DynamicTileFromBroadcast_case1(%arg0: tensor<1x1x?xsi64, {bounds = [1, 1, 10], order = #CHW}>, %arg1: tensor<4xsi32>) -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> {
    %0 = IE.DynamicTile(%arg0, %arg1) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x?xsi64, {bounds = [1, 1, 10], order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>
    return %0 : tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>

    // CHECK-NOT:   IE.DynamicTile
    // CHECK:       VPU.DynamicTile([[ARG0]], [[ARG1]]) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x?xsi64, {bounds = [1, 1, 10], order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>
}
