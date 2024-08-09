//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --align-scales %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @AlignAllFQ
func.func @AlignAllFQ(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>, %arg2: tensor<1x2x3x4xf16>) -> (tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>) {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.5000e+00> : tensor<1x1x1x1xf16>
  %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<4.0000e+00> : tensor<1x1x1x1xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.4000e+00> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.5000e+00> : tensor<1x1x1x1xf16>
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-1.7500e+00> : tensor<1x1x1x1xf16>
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<3.7500e+00> : tensor<1x1x1x1xf16>
  %cst_5 = const.Declare tensor<1x1x1x1xf16> = dense<-6.2500e+00> : tensor<1x1x1x1xf16>
  %cst_6 = const.Declare tensor<1x1x1x1xf16> = dense<5.3000e+00> : tensor<1x1x1x1xf16>
  %cst_7 = const.Declare tensor<1x1x1x1xf16> = dense<-2.4500e+00> : tensor<1x1x1x1xf16>
  %cst_8 = const.Declare tensor<1x1x1x1xf16> = dense<6.7000e+00> : tensor<1x1x1x1xf16>
  %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %1:2 = IE.Split(%0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
  %2 = IE.MaxPool(%1#0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2] } : tensor<1x1x3x4xf16> -> tensor<1x1x3x2xf16>
  %3 = IE.FakeQuantize(%2, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x2xf16>
  %4 = IE.FakeQuantize(%arg1, %cst_3, %cst_4, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %5 = IE.FakeQuantize(%arg2, %cst_5, %cst_6, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  %7 = IE.ReduceMax(%6) {axes_value = [1], keep_dims} : tensor<1x4x3x4xf16> -> tensor<1x1x3x4xf16>
  %8 = IE.Concat(%7, %1#1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x2x3x4xf16>
  %9 = IE.FakeQuantize(%8, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>

  return %3, %9 : tensor<1x1x3x2xf16>, tensor<1x2x3x4xf16>

  // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-6.25101137> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
  // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<5.29977036> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
  // CHECK: [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP:%.*]] = IE.Clamp([[FQ]]) {max = 4.000000e+00 : f64, min = -1.500000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[SPLIT:%.*]]:2 = IE.Split([[CLAMP]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
  // CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[SPLIT]]#0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x1x3x4xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[MAXPOOL]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[CLAMP_0:%.*]] = IE.Clamp([[FQ_0]]) {max = 2.500000e+00 : f64, min = -2.400390625 : f64} : tensor<1x1x3x2xf16> -> tensor<1x1x3x2xf16>
  // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize(%arg1, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_1:%.*]] = IE.Clamp([[FQ_1]]) {max = 3.750000e+00 : f64, min = -1.750000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize(%arg2, [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_2:%.*]] = IE.Clamp([[FQ_2]]) {max = 5.30078125 : f64, min = -6.250000e+00 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CONCAT:%.*]] = IE.Concat([[CLAMP_1]], [[CLAMP_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
  // CHECK: [[REDUCEMAX:%.*]] = IE.ReduceMax([[CONCAT]]) {axes_value = [1], keep_dims} : tensor<1x4x3x4xf16> -> tensor<1x1x3x4xf16>
  // CHECK: [[CONCAT_0:%.*]] = IE.Concat([[REDUCEMAX]], [[SPLIT]]#1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[FQ_3:%.*]] = IE.FakeQuantize([[CONCAT_0]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x3x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x3x4xf16>
  // CHECK: [[CLAMP_3:%.*]] = IE.Clamp([[FQ_3]]) {max = 4.000000e+00 : f64, min = -2.400390625 : f64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>

  // CHECK: return [[CLAMP_0]], [[CLAMP_3]]
}
