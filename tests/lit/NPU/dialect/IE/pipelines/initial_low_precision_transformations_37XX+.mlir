//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --initial-low-precision-transformations %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @ScalarArgumentsWeightsDequantizeToFakeQuantize
func.func @ScalarArgumentsWeightsDequantizeToFakeQuantize(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<16x1x1x1xf32> = dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]> : tensor<16x1x1x1xsi8>, [#const.ConvertElemType<f32>]
  %cst_3 = const.Declare tensor<f32> = dense<2.500000e+01> : tensor<f32>
  %cst_4 = const.Declare tensor<f32> = dense<0.0566197559> : tensor<f32>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<f32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<f32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-8.60620307> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<5.77521514> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[2.700000e+01]]], [[[2.500000e+01]]], [[[3.900000e+01]]], [[[2.200000e+01]]], [[[2.700000e+01]]], [[[2.500000e+01]]], [[[2.100000e+01]]], [[[2.700000e+01]]], [[[3.100000e+01]]], [[[2.900000e+01]]], [[[4.200000e+01]]], [[[2.700000e+01]]], [[[2.700000e+01]]], [[[2.800000e+01]]], [[[3.300000e+01]]], [[[3.300000e+01]]]]> : tensor<16x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_3]], [[CST]], [[CST_0]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  // CHECK-DAG:   [[ACT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  // CHECK:   [[GRUP_CONV:%.*]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>

  // CHECK:   return [[GRUP_CONV]] : tensor<1x16x32x32xf32>
}
