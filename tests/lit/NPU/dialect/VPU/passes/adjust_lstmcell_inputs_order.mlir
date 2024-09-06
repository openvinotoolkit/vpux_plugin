//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-lstmcell-inputs-order %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: @adjustLstmCellInputOrderTest
func.func @adjustLstmCellInputOrderTest(%arg0: tensor<1x2x64xf16>, %arg1: tensor<1x2x128xf16>, %arg2: tensor<1x2x128xf16>) -> (tensor<1x2x2x128xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>) {
  %cst = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<512xf16>, [#const.Reshape<[1, 1, 1, 512]>]
  %cst_0 = const.Declare tensor<1x1x512x128xf16> = dense<1.000000e+00> : tensor<512x128xf16>, [#const.Reshape<[1, 1, 512, 128]>]
  %cst_1 = const.Declare tensor<1x1x512x64xf16> = dense<1.000000e+00> : tensor<512x64xf16>, [#const.Reshape<[1, 1, 512, 64]>]
  %cst_2 = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<512xf16>, [#const.Reshape<[1, 1, 1, 512]>]
  %cst_3 = const.Declare tensor<1x1x512x128xf16> = dense<1.000000e+00> : tensor<512x128xf16>, [#const.Reshape<[1, 1, 512, 128]>]
  %cst_4 = const.Declare tensor<1x1x512x64xf16> = dense<1.000000e+00> : tensor<512x64xf16>, [#const.Reshape<[1, 1, 512, 64]>]
  %0 = VPU.Slice %arg0 [0, 1, 0] [1, 1, 64] : tensor<1x2x64xf16> to tensor<1x1x64xf16>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
  %2 = VPU.Slice %arg0 [0, 0, 0] [1, 1, 64] : tensor<1x2x64xf16> to tensor<1x1x64xf16>
  %3 = VPU.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
  %4:2 = VPU.Split(%arg1) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x128xf16> -> tensor<1x1x128xf16>, tensor<1x1x128xf16>
  %5:2 = VPU.Split(%arg2) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x128xf16> -> tensor<1x1x128xf16>, tensor<1x1x128xf16>
  %6 = VPU.AffineReshape(%4#0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
  %7 = VPU.AffineReshape(%5#0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
  %outputHiddenState, %outputCellState = VPU.LSTMCell(%3, %6, %7, %cst_4, %cst_3, %cst_2) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x1x512x128xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
  %outputHiddenState_5, %outputCellState_6 = VPU.LSTMCell(%1, %outputHiddenState, %outputCellState, %cst_4, %cst_3, %cst_2) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x1x512x128xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
  %8 = VPU.AffineReshape(%4#1) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
  %9 = VPU.AffineReshape(%5#1) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
  %outputHiddenState_7, %outputCellState_8 = VPU.LSTMCell(%1, %8, %9, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x1x512x128xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
  %outputHiddenState_9, %outputCellState_10 = VPU.LSTMCell(%3, %outputHiddenState_7, %outputCellState_8, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x1x512x128xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
  %10 = VPU.Concat(%outputCellState_6, %outputCellState_10) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x1x2x128xf16>
  %11 = VPU.AffineReshape(%10) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 2, 128]} : tensor<1x1x2x128xf16> -> tensor<1x2x128xf16>
  %12 = VPU.Concat(%outputHiddenState_5, %outputHiddenState_9) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x1x2x128xf16>
  %13 = VPU.AffineReshape(%12) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 2, 128]} : tensor<1x1x2x128xf16> -> tensor<1x2x128xf16>
  %14 = VPU.Concat(%outputHiddenState, %outputHiddenState_5, %outputHiddenState_9, %outputHiddenState_7) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]} : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x2x2x128xf16>
  return %14, %13, %11 : tensor<1x2x2x128xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>

    // CHECK:        %cst = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<512xf16>, [#const.Reshape<[1, 1, 1, 512]>, #const.Reshape<[1, 1, 4, 128]>, #const.Reorder<#NCWH>]
    // CHECK:        %cst_0 = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<1.000000e+00> : tensor<512x128xf16>, [#const.Reshape<[1, 1, 512, 128]>, #const.Reshape<[1, 4, 128, 128]>, #const.Reorder<#NWHC>]
    // CHECK:        %cst_1 = const.Declare tensor<1x4x128x64xf16, {order = #NWHC}> = dense<1.000000e+00> : tensor<512x64xf16>, [#const.Reshape<[1, 1, 512, 64]>, #const.Reshape<[1, 4, 128, 64]>, #const.Reorder<#NWHC>]
    // CHECK:        %0 = VPU.Slice %arg0 [0, 1, 0] [1, 1, 64] : tensor<1x2x64xf16> to tensor<1x1x64xf16>
    // CHECK:        %1 = VPU.AffineReshape(%0)
    //CHECK-LITERAL: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 64]} :
    // CHECK-SAME:   tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:        %2 = VPU.Slice %arg0 [0, 0, 0] [1, 1, 64] : tensor<1x2x64xf16> to tensor<1x1x64xf16>
    // CHECK:        %3 = VPU.AffineReshape(%2)
    //CHECK-LITERAL: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 64]} :
    // CHECK-SAME:   tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:        %4:2 = VPU.Split(%arg1) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x128xf16> -> tensor<1x1x128xf16>, tensor<1x1x128xf16>
    // CHECK:        %5:2 = VPU.Split(%arg2) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x128xf16> -> tensor<1x1x128xf16>, tensor<1x1x128xf16>
    // CHECK:        %6 = VPU.AffineReshape(%4#0)
    //CHECK-LITERAL: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} :
    // CHECK-SAME:   tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:        %7 = VPU.AffineReshape(%5#0)
    //CHECK-LITERAL: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} :
    // CHECK-SAME:   tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:        %outputHiddenState, %outputCellState = VPU.LSTMCell(%3, %6, %7, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x64xf16, {order = #NWHC}>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK:        %outputHiddenState_2, %outputCellState_3 = VPU.LSTMCell(%1, %outputHiddenState, %outputCellState, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x64xf16, {order = #NWHC}>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK:        %8 = VPU.AffineReshape(%4#1)
    //CHECK-LITERAL: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} :
    // CHECK-SAME:   tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:        %9 = VPU.AffineReshape(%5#1)
    //CHECK-LITERAL: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 128]} :
    // CHECK-SAME:   tensor<1x1x128xf16> -> tensor<1x1x1x128xf16>
    // CHECK:        %outputHiddenState_4, %outputCellState_5 = VPU.LSTMCell(%1, %8, %9, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x64xf16, {order = #NWHC}>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK:        %outputHiddenState_6, %outputCellState_7 = VPU.LSTMCell(%3, %outputHiddenState_4, %outputCellState_5, %cst_1, %cst_0, %cst) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x64xf16, {order = #NWHC}>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
    // CHECK:        %10 = VPU.Concat(%outputCellState_3, %outputCellState_7)
    //CHECK-LITERAL: {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} :
    // CHECK-SAME:   tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x1x2x128xf16>
    // CHECK:        %11 = VPU.AffineReshape(%10)
    //CHECK-LITERAL: {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 2, 128]} :
    // CHECK-SAME:   tensor<1x1x2x128xf16> -> tensor<1x2x128xf16>
    // CHECK:        %12 = VPU.Concat(%outputHiddenState_2, %outputHiddenState_6)
    //CHECK-LITERAL: {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} :
    // CHECK-SAME:   tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x1x2x128xf16>
    // CHECK:        %13 = VPU.AffineReshape(%12)
    //CHECK-LITERAL: {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 2, 128]} :
    // CHECK-SAME:   tensor<1x1x2x128xf16> -> tensor<1x2x128xf16>
    // CHECK:        %14 = VPU.Concat(%outputHiddenState, %outputHiddenState_2, %outputHiddenState_6, %outputHiddenState_4)
    //CHECK-LITERAL: {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]} :
    // CHECK-SAME:   tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x2x2x128xf16>
    // CHECK:        return %14, %13, %11 : tensor<1x2x2x128xf16>, tensor<1x2x128xf16>, tensor<1x2x128xf16>
}
