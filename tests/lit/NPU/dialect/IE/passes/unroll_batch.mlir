//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-batch %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @UnrollFullyConnectedBatch
func.func @UnrollFullyConnectedBatch(%arg0: tensor<2x16xf32>) -> tensor<2x64xf32> {
    %cst = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf32>, [#const.CastElemType<f16>]
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    %1 = IE.FullyConnected(%0, %cst) : tensor<2x16xf16>, tensor<64x16xf16> -> tensor<2x64xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>

    return %2 : tensor<2x64xf32>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16xf32>, [#const.CastElemType<f16>]
    // CHECK:       %[[INPUT:.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    // CHECK:       %[[INPUT_SLICE_1:.*]] = IE.Slice %[[INPUT]] [0, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_1:.*]] = IE.FullyConnected(%[[INPUT_SLICE_1]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[INPUT_SLICE_2:.*]] = IE.Slice %[[INPUT]] [1, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_2:.*]] = IE.FullyConnected(%[[INPUT_SLICE_2]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[FC_CONCAT:.*]] = IE.Concat(%[[FC_1]], %[[FC_2]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x64xf16>, tensor<1x64xf16> -> tensor<2x64xf16>
    // CHECK:       %[[OUT:.*]] = IE.Convert(%[[FC_CONCAT]]) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>
    // CHECK:       return %[[OUT]] : tensor<2x64xf32>
}

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @UnrollEltwiseAndBatch
func.func @UnrollEltwiseAndBatch(%arg0: tensor<2x128x40x8xf16>) -> tensor<2x128x40x8xf16> {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x128x40x8xf16>, tensor<2x128x40x8xf16> -> tensor<2x128x40x8x!qElemType>
    %1 = IE.And(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x128x40x8x!qElemType>, tensor<2x128x40x8x!qElemType> -> tensor<2x128x40x8xf16>
    return %1 : tensor<2x128x40x8xf16>

    // CHECK: [[SLICE0_ARG0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: [[AND0:%.*]] = IE.And([[SLICE0_ARG0]], [[SLICE0_ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<1x128x40x8x!qElemType>
    // CHECK: [[SLICE1_ARG0:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8xf16> to tensor<1x128x40x8xf16>
    // CHECK: [[AND1:%.*]] = IE.And([[SLICE1_ARG0]], [[SLICE1_ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<1x128x40x8x!qElemType>
    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[AND0]], [[AND1]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<2x128x40x8x!qElemType>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: [[AND2:%.*]] = IE.And([[SLICE2]], [[SLICE2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<1x128x40x8xf16>
    // CHECK: [[SLICE3:%.*]] = IE.Slice [[CONCAT0]] [1, 0, 0, 0] [1, 128, 40, 8] : tensor<2x128x40x8x!qElemType> to tensor<1x128x40x8x!qElemType>
    // CHECK: [[AND3:%.*]] = IE.And([[SLICE3]], [[SLICE3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME: tensor<1x128x40x8x!qElemType>, tensor<1x128x40x8x!qElemType> -> tensor<1x128x40x8xf16>
    // CHECK: [[CONCAT1:%.*]] = IE.Concat([[AND2]], [[AND3]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x128x40x8xf16>, tensor<1x128x40x8xf16> -> tensor<2x128x40x8xf16>
    // CHECK: return [[CONCAT1]] : tensor<2x128x40x8xf16>
}

// -----

func.func @UnrollSigmoidBatch(%arg0: tensor<3x9x16x1xf16>) -> tensor<3x9x16x1xf16> {
    %0 = IE.Sigmoid(%arg0) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
    return %0 : tensor<3x9x16x1xf16>
    // CHECK: [[SLICE0_ARG0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[SIGMOID0:%.*]] = IE.Sigmoid([[SLICE0_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE1_ARG0:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[SIGMOID1:%.*]] = IE.Sigmoid([[SLICE1_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE2_ARG0:%.*]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[SIGMOID2:%.*]] = IE.Sigmoid([[SLICE2_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[SIGMOID0]], [[SIGMOID1]], [[SIGMOID2]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK: return [[CONCAT0]] : tensor<3x9x16x1xf16>
}

// -----

func.func @UnrollExpBatch(%arg0: tensor<3x9x16x1xf16>) -> tensor<3x9x16x1xf16> {
    %0 = IE.Exp(%arg0) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
    return %0 : tensor<3x9x16x1xf16>
    // CHECK: [[SLICE0_ARG0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[EXP0:%.*]] = IE.Exp([[SLICE0_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE1_ARG0:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[EXP1:%.*]] = IE.Exp([[SLICE1_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE2_ARG0:%.*]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[EXP2:%.*]] = IE.Exp([[SLICE2_ARG0]]) :
    // CHECK-SAME: tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[EXP0]], [[EXP1]], [[EXP2]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK: return [[CONCAT0]] : tensor<3x9x16x1xf16>
}

// -----

func.func @UnrollGroupConvolutionBatch(%arg0: tensor<3x9x16x1xf16>, %arg1: tensor<9x1x1x1xf16>) -> tensor<3x9x16x1xf16> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<3x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<3x9x16x1xf16>
    return %0 : tensor<3x9x16x1xf16>
    // CHECK: [[SLICE0_ARG0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[GROUPCONVOLUTION0:%.*]] = IE.GroupConvolution([[SLICE0_ARG0]], %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE1_ARG0:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[GROUPCONVOLUTION1:%.*]] = IE.GroupConvolution([[SLICE1_ARG0]], %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[SLICE2_ARG0:%.*]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 9, 16, 1] : tensor<3x9x16x1xf16> to tensor<1x9x16x1xf16>
    // CHECK: [[GROUPCONVOLUTION2:%.*]] = IE.GroupConvolution([[SLICE2_ARG0]], %arg1) {dilations = [1, 1], groups = 9 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<9x1x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[GROUPCONVOLUTION0]], [[GROUPCONVOLUTION1]], [[GROUPCONVOLUTION2]]) {per_axis = #IE.Concat<axis = 0 : i64>} :
    // CHECK-SAME: tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK: return [[CONCAT0]] : tensor<3x9x16x1xf16>
}

// -----

func.func @UnrollMaxPoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %MAX_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[SLICE_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[MAX_POOL_0:%.*]] = IE.MaxPool([[SLICE_0]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[SLICE_1:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 128, 32, 64] :
    // CHECK-SAME:      tensor<2x128x32x64xf16> to tensor<1x128x32x64xf16>

    // CHECK:   [[MAX_POOL_1:%.*]] = IE.MaxPool([[SLICE_1]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x32x64xf16> -> tensor<1x128x32x64xf16>

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[MAX_POOL_0]], [[MAX_POOL_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 0 : i64>
    // CHECK-SAME:  } : tensor<1x128x32x64xf16>, tensor<1x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[CONCAT]] : tensor<2x128x32x64xf16>
}
