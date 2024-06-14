//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --weights-dequantize-to-fake-quantize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @WeightsMultToFakeQuantize
func.func @WeightsMultToFakeQuantize(%arg0: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, -70], [36, 72, -82]],[[31, -67, 22], [-70, -55, 12], [-99, 42, 90]],[[6, -18, 95], [-8, -37, -64], [40, 31, -41]],[[35, -2, -98], [-94, -60, -68], [-3, -39, 88]]],[[[-43, -95, 64], [46, -125, -63], [-21, -25, -25]],[[-118, -103, -12], [84, 67, 55], [-105, 13, -10]],[[97, -124, 39], [-28, -112, 116], [74, 104, 72]],[[14, 58, 0], [37, -48, 26], [33, -64, 53]]],[[[-124, 104, 105], [-14, 0, -25], [104, -46, -87]],[[87, -105, 69], [94, 88, 47], [53, 93, -34]],[[-62, -44, -10], [81, 110, 32], [10, 72, 30]],[[117, 64, 41], [0, -50, -39], [-108, 7, -12]]],[[[-73, -47, 7], [72, -17, 90], [-113, 44, 80]],[[-60, -102, -79], [-111, -43, 68], [-21, 53, 120]],[[-109, -69, 30], [120, -7, 107], [-30, 42, 66]],[[43, 16, -57], [95, 125, -99], [-30, 1, 126]]]]> : tensor<4x4x3x3xsi8>, [#const.ConvertElemType<f32>]
  %cst_1 = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]> : tensor<4x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst_3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[-0.37437284]]], [[[-0.397086918]]], [[[-0.33067891]]], [[[-0.342519492]]]]> : tensor<4x1x1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[0.37437284]]], [[[0.397086918]]], [[[0.33067891]]], [[[0.342519492]]]]> : tensor<4x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.300000e+01, 6.900000e+01, 9.500000e+01], [4.700000e+01, 8.500000e+01, -7.000000e+01], [3.600000e+01, 7.200000e+01, -8.200000e+01]], [[3.100000e+01, -6.700000e+01, 2.200000e+01], [-7.000000e+01, -5.500000e+01, 1.200000e+01], [-9.900000e+01, 4.200000e+01, 9.000000e+01]], [[6.000000e+00, -1.800000e+01, 9.500000e+01], [-8.000000e+00, -3.700000e+01, -6.400000e+01], [4.000000e+01, 3.100000e+01, -4.100000e+01]], [[3.500000e+01, -2.000000e+00, -9.800000e+01], [-9.400000e+01, -6.000000e+01, -6.800000e+01], [-3.000000e+00, -3.900000e+01, 8.800000e+01]]], [[[-4.300000e+01, -9.500000e+01, 6.400000e+01], [4.600000e+01, -1.250000e+02, -6.300000e+01], [-2.100000e+01, -2.500000e+01, -2.500000e+01]], [[-1.180000e+02, -1.030000e+02, -1.200000e+01], [8.400000e+01, 6.700000e+01, 5.500000e+01], [-1.050000e+02, 1.300000e+01, -1.000000e+01]], [[9.700000e+01, -1.240000e+02, 3.900000e+01], [-2.800000e+01, -1.120000e+02, 1.160000e+02], [7.400000e+01, 1.040000e+02, 7.200000e+01]], [[1.400000e+01, 5.800000e+01, 0.000000e+00], [3.700000e+01, -4.800000e+01, 2.600000e+01], [3.300000e+01, -6.400000e+01, 5.300000e+01]]], [[[-1.240000e+02, 1.040000e+02, 1.050000e+02], [-1.400000e+01, 0.000000e+00, -2.500000e+01], [1.040000e+02, -4.600000e+01, -8.700000e+01]], [[8.700000e+01, -1.050000e+02, 6.900000e+01], [9.400000e+01, 8.800000e+01, 4.700000e+01], [5.300000e+01, 9.300000e+01, -3.400000e+01]], [[-6.200000e+01, -4.400000e+01, -1.000000e+01], [8.100000e+01, 1.100000e+02, 3.200000e+01], [1.000000e+01, 7.200000e+01, 3.000000e+01]], [[1.170000e+02, 6.400000e+01, 4.100000e+01], [0.000000e+00, -5.000000e+01, -3.900000e+01], [-1.080000e+02, 7.000000e+00, -1.200000e+01]]], [[[-7.300000e+01, -4.700000e+01, 7.000000e+00], [7.200000e+01, -1.700000e+01, 9.000000e+01], [-1.130000e+02, 4.400000e+01, 8.000000e+01]], [[-6.000000e+01, -1.020000e+02, -7.900000e+01], [-1.110000e+02, -4.300000e+01, 6.800000e+01], [-2.100000e+01, 5.300000e+01, 1.200000e+02]], [[-1.090000e+02, -6.900000e+01, 3.000000e+01], [1.200000e+02, -7.000000e+00, 1.070000e+02], [-3.000000e+01, 4.200000e+01, 6.600000e+01]], [[4.300000e+01, 1.600000e+01, -5.700000e+01], [9.500000e+01, 1.250000e+02, -9.900000e+01], [-3.000000e+01, 1.000000e+00, 1.260000e+02]]]]> : tensor<4x4x3x3xf32>
  // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_6:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[ACT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_6]], [[CST_5]], [[CST_6]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK:   [[CONV:%.*]] = IE.Convolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK:   return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @WeightsMultSubToFakeQuantize
func.func @WeightsMultSubToFakeQuantize(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<16x1x1x1xf32> = dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]> : tensor<16x1x1x1xsi8>, [#const.ConvertElemType<f32>]
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.500000e+01> : tensor<1x1x1x1xf32>
  %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<0.0566197559> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-8.60620307> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<5.77521514> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[2.700000e+01]]], [[[2.500000e+01]]], [[[3.900000e+01]]], [[[2.200000e+01]]], [[[2.700000e+01]]], [[[2.500000e+01]]], [[[2.100000e+01]]], [[[2.700000e+01]]], [[[3.100000e+01]]], [[[2.900000e+01]]], [[[4.200000e+01]]], [[[2.700000e+01]]], [[[2.700000e+01]]], [[[2.800000e+01]]], [[[3.300000e+01]]], [[[3.300000e+01]]]]> : tensor<16x1x1x1xf32>
  // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_6:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[ACT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_6]], [[CST_5]], [[CST_6]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  // CHECK:   [[GRUP_CONV:%.*]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>

  // CHECK:   return [[GRUP_CONV]] : tensor<1x16x32x32xf32>
}

// -----

// CHECK-LABEL: @WeightsMultSubToNon4DFakeQuantize
func.func @WeightsMultSubToNon4DFakeQuantize(%arg0: tensor<1x4x48xf16>) -> tensor<1x4x48xf16> {
  %cst_0 = const.Declare tensor<48xf16> = dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]> : tensor<48xsi8>, [#const.ConvertElemType<f16>]
  %cst_1 = const.Declare tensor<1xf16> = dense<8.800000e+01> : tensor<1xf16>
  %cst_2 = const.Declare tensor<1xf16> = dense<9.88533836E-4> : tensor<1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %2 = IE.Add(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  return %2 : tensor<1x4x48xf16>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1xf16> = dense<-1.270000e+02> : tensor<1xf16>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1xf16> = dense<1.270000e+02> : tensor<1xf16>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1xf16> = dense<-2.126460e-01> : tensor<1xf16>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1xf16> = dense<3.857420e-02> : tensor<1xf16>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<48xf16>
  // CHECK-SAME{LITERAL}  = dense<[9.000000e+00, -5.000000e+00, -2.500000e+01, 5.200000e+01, -7.700000e+01, -1.230000e+02, 2.400000e+01, 6.700000e+01, -3.200000e+01, 1.100000e+01, -2.400000e+01, 9.300000e+01, -1.700000e+01, -1.270000e+02, -4.600000e+01, -3.800000e+01, 5.300000e+01, -8.800000e+01, -1.080000e+02, -6.000000e+01, 9.000000e+00, -8.000000e+00, -7.800000e+01, 1.060000e+02, -3.300000e+01, 1.400000e+01, -1.100000e+01, -2.100000e+01, -9.400000e+01, -7.200000e+01, 4.900000e+01, 1.250000e+02, -5.800000e+01, -9.300000e+01, 9.100000e+01, 4.400000e+01, 1.230000e+02, -9.900000e+01, -5.900000e+01, 1.500000e+01, -1.240000e+02, -1.300000e+01, 8.900000e+01, -9.200000e+01, -9.700000e+01, 1.000000e+01, -1.600000e+01, 3.800000e+01]> : tensor<48xf16>
  // CHECK:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<48xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<48xf16>
  // CHECK:   [[ADD:%.*]] = IE.Add(%arg0, [[WT_FQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  // CHECK:   return [[ADD]] : tensor<1x4x48xf16>
}

// -----

// CHECK-LABEL: @WeightsMultScalarSubFakeQuantize
func.func @WeightsMultScalarSubFakeQuantize(%arg0: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.ConvertElemType<f32>]
  %cst_1 = const.Declare tensor<1xf32> = dense<-22> : tensor<si8>, [#const.ConvertElemType<f32>, #const.Reshape<[1]>]
  %cst_2 = const.Declare tensor<1x1xf32> = dense<0.00704713073> : tensor<1x1xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1x1xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%arg0, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>

  return %3 : tensor<1x6x12x12xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1xf32> = dense<-1.270000e+02> : tensor<1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02> : tensor<1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1xf32> = dense<-0.739948749> : tensor<1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1xf32> = dense<1.05002248> : tensor<1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<6x6xf32>
  // CHECK-SAME{LITERAL}  = dense<[[-6.300000e+01, -6.000000e+00, 6.700000e+01, -6.200000e+01, 4.600000e+01, 4.000000e+01], [9.500000e+01, 5.600000e+01, -2.400000e+01, 2.000000e+01, -5.300000e+01, -4.300000e+01], [-4.100000e+01, -7.600000e+01, 1.130000e+02, 0.000000e+00, 8.700000e+01, -1.070000e+02], [-1.210000e+02, 1.050000e+02, -8.900000e+01, 6.400000e+01, -9.100000e+01, -3.900000e+01], [9.200000e+01, -1.600000e+01, 8.900000e+01, 5.000000e+00, 9.200000e+01, 2.700000e+01], [-1.120000e+02, 1.120000e+02, -1.010000e+02, 6.200000e+01, 6.100000e+01, -2.900000e+01]]> : tensor<6x6xf32>
  // CHECK:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<6x6xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<6x6xf32>
  // CHECK:   [[RESHAPE:%.*]] = IE.Reshape([[WT_FQ]]) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>

  // CHECK:   return [[CONV]] : tensor<1x6x12x12xf32>
}

// -----

// CHECK-LABEL: @WeightsMultToFakeQuantizeI4WithU8Storage
func.func @WeightsMultToFakeQuantizeI4WithU8Storage(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xsi4> = dense<[[[[7, 6, 5], [4, 5, -7], [3, 2, -2]]]]> : tensor<1x1x3x3xsi8>, [#const.ConvertElemType<si4>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Convert(%cst_0) { dstElemType = f16 } : tensor<1x1x3x3xsi4> -> tensor<1x1x3x3xf16>
  %1 = IE.Multiply(%0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-7.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.064510e-02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.064510e-02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.000000e+00, 6.000000e+00, 5.000000e+00], [4.000000e+00, 5.000000e+00, -7.000000e+00], [3.000000e+00, 2.000000e+00, -2.000000e+00]]]]> : tensor<1x1x3x3xf16>
  // CHECK:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 15 : i64} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  // CHECK:   return [[CONV]] : tensor<1x1x28x28xf16>
}

// -----

// CHECK-LABEL: @WeightsMultToFakeQuantizeU4WithFP16Storage
func.func @WeightsMultToFakeQuantizeU4WithFP16Storage(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xui4> = dense<[[[[7.0, 6.0, 5.0], [4.0, 5.0, 8.0], [3.0, 2.0, 15.0]]]]> : tensor<1x1x3x3xf16>, [#const.ConvertElemType<ui4>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Convert(%cst_0) { dstElemType = f16 } : tensor<1x1x3x3xui4> -> tensor<1x1x3x3xf16>
  %1 = IE.Multiply(%0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.500000e+01> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.422000e-02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.000000e+00, 6.000000e+00, 5.000000e+00], [4.000000e+00, 5.000000e+00, 8.000000e+00], [3.000000e+00, 2.000000e+00, 1.500000e+01]]]]> : tensor<1x1x3x3xf16>
  // CHECK:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_3]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  // CHECK:   return [[CONV]] : tensor<1x1x28x28xf16>
}

// -----

// CHECK-LABEL: @DontConvertFP16WeightsMultToFakeQuantize
func.func @DontConvertFP16WeightsMultToFakeQuantize(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xf16> = dense<[[[[7.0, 6.0, 5.0], [4.0, 5.0, -8.0], [3.0, 2.0, -2.0]]]]> : tensor<1x1x3x3xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03> : tensor<1x1x1x1xf16>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.000000e+00, 6.000000e+00, 5.000000e+00], [4.000000e+00, 5.000000e+00, -7.000000e+00], [3.000000e+00, 2.000000e+00, -2.000000e+00]]]]> : tensor<1x1x3x3xf16>
  // CHECK:   [[MULTI:%.*]] = IE.Multiply([[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[MULTI]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  // CHECK:   return [[CONV]] : tensor<1x1x28x28xf16>
}

// -----

// CHECK-LABEL: @WeightsUI8MultToFakeQuantize
func.func @WeightsUI8MultToFakeQuantize(%arg0: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]],[[31, 67, 22], [70, 55, 12], [99, 42, 90]],[[6, 18, 95], [8, 37, 64], [40, 31, 41]],[[35, 2, 98], [94, 60, 68], [3, 39, 88]]],[[[43, 95, 64], [46, 125, 63], [21, 25, 25]],[[118, 103, 12], [84, 67, 55], [105, 13, 10]],[[97, 124, 39], [28, 112, 116], [74, 104, 72]],[[14, 58, 0], [37, 48, 26], [33, 64, 53]]],[[[124, 104, 105], [14, 0, 25], [104, 46, 87]],[[87, 105, 69], [94, 88, 47], [53, 93, 34]],[[62, 44, 10], [81, 110, 32], [10, 72, 30]],[[117, 64, 41], [0, 50, 39], [108, 7, 12]]],[[[73, 47, 7], [72, 17, 90], [113, 44, 80]],[[60, 102, 79], [111, 43, 68], [21, 53, 120]],[[109, 69, 30], [120, 7, 107], [30, 42, 66]],[[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.ConvertElemType<f32>]
  %cst_1 = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]> : tensor<4x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst_3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<4x1x1x1xf32> = dense<0.000000e+00> : tensor<4x1x1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[0.751693487]]], [[[0.797300517]]], [[[0.663961589]]], [[[0.687735974]]]]> : tensor<4x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.300000e+01, 6.900000e+01, 9.500000e+01], [4.700000e+01, 8.500000e+01, -7.000000e+01], [3.600000e+01, 7.200000e+01, -8.200000e+01]], [[3.100000e+01, -6.700000e+01, 2.200000e+01], [-7.000000e+01, -5.500000e+01, 1.200000e+01], [-9.900000e+01, 4.200000e+01, 9.000000e+01]], [[6.000000e+00, -1.800000e+01, 9.500000e+01], [-8.000000e+00, -3.700000e+01, -6.400000e+01], [4.000000e+01, 3.100000e+01, -4.100000e+01]], [[3.500000e+01, -2.000000e+00, -9.800000e+01], [-9.400000e+01, -6.000000e+01, -6.800000e+01], [-3.000000e+00, -3.900000e+01, 8.800000e+01]]], [[[-4.300000e+01, -9.500000e+01, 6.400000e+01], [4.600000e+01, -1.250000e+02, -6.300000e+01], [-2.100000e+01, -2.500000e+01, -2.500000e+01]], [[-1.180000e+02, -1.030000e+02, -1.200000e+01], [8.400000e+01, 6.700000e+01, 5.500000e+01], [-1.050000e+02, 1.300000e+01, -1.000000e+01]], [[9.700000e+01, -1.240000e+02, 3.900000e+01], [-2.800000e+01, -1.120000e+02, 1.160000e+02], [7.400000e+01, 1.040000e+02, 7.200000e+01]], [[1.400000e+01, 5.800000e+01, 0.000000e+00], [3.700000e+01, -4.800000e+01, 2.600000e+01], [3.300000e+01, -6.400000e+01, 5.300000e+01]]], [[[-1.240000e+02, 1.040000e+02, 1.050000e+02], [-1.400000e+01, 0.000000e+00, -2.500000e+01], [1.040000e+02, -4.600000e+01, -8.700000e+01]], [[8.700000e+01, -1.050000e+02, 6.900000e+01], [9.400000e+01, 8.800000e+01, 4.700000e+01], [5.300000e+01, 9.300000e+01, -3.400000e+01]], [[-6.200000e+01, -4.400000e+01, -1.000000e+01], [8.100000e+01, 1.100000e+02, 3.200000e+01], [1.000000e+01, 7.200000e+01, 3.000000e+01]], [[1.170000e+02, 6.400000e+01, 4.100000e+01], [0.000000e+00, -5.000000e+01, -3.900000e+01], [-1.080000e+02, 7.000000e+00, -1.200000e+01]]], [[[-7.300000e+01, -4.700000e+01, 7.000000e+00], [7.200000e+01, -1.700000e+01, 9.000000e+01], [-1.130000e+02, 4.400000e+01, 8.000000e+01]], [[-6.000000e+01, -1.020000e+02, -7.900000e+01], [-1.110000e+02, -4.300000e+01, 6.800000e+01], [-2.100000e+01, 5.300000e+01, 1.200000e+02]], [[-1.090000e+02, -6.900000e+01, 3.000000e+01], [1.200000e+02, -7.000000e+00, 1.070000e+02], [-3.000000e+01, 4.200000e+01, 6.600000e+01]], [[4.300000e+01, 1.600000e+01, -5.700000e+01], [9.500000e+01, 1.250000e+02, -9.900000e+01], [-3.000000e+01, 1.000000e+00, 1.260000e+02]]]]> : tensor<4x4x3x3xf32>
  // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_6:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[ACT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_6]], [[CST_5]], [[CST_6]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK:   [[CONV:%.*]] = IE.Convolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK:   return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @WeightsUI8SubToFakeQuantize
func.func @WeightsUI8SubToFakeQuantize(%arg0: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]],[[31, 67, 22], [70, 55, 12], [99, 42, 90]],[[6, 18, 95], [8, 37, 64], [40, 31, 41]],[[35, 2, 98], [94, 60, 68], [3, 39, 88]]],[[[43, 95, 64], [46, 125, 63], [21, 25, 25]],[[118, 103, 12], [84, 67, 55], [105, 13, 10]],[[97, 124, 39], [28, 112, 116], [74, 104, 72]],[[14, 58, 0], [37, 48, 26], [33, 64, 53]]],[[[124, 104, 105], [14, 0, 25], [104, 46, 87]],[[87, 105, 69], [94, 88, 47], [53, 93, 34]],[[62, 44, 10], [81, 110, 32], [10, 72, 30]],[[117, 64, 41], [0, 50, 39], [108, 7, 12]]],[[[73, 47, 7], [72, 17, 90], [113, 44, 80]],[[60, 102, 79], [111, 43, 68], [21, 53, 120]],[[109, 69, 30], [120, 7, 107], [30, 42, 66]],[[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.ConvertElemType<f32>]
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<128.0> : tensor<1x1x1x1xf32>
  %1 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.300000e+01, 6.900000e+01, 9.500000e+01], [4.700000e+01, 8.500000e+01, -7.000000e+01], [3.600000e+01, 7.200000e+01, -8.200000e+01]], [[3.100000e+01, -6.700000e+01, 2.200000e+01], [-7.000000e+01, -5.500000e+01, 1.200000e+01], [-9.900000e+01, 4.200000e+01, 9.000000e+01]], [[6.000000e+00, -1.800000e+01, 9.500000e+01], [-8.000000e+00, -3.700000e+01, -6.400000e+01], [4.000000e+01, 3.100000e+01, -4.100000e+01]], [[3.500000e+01, -2.000000e+00, -9.800000e+01], [-9.400000e+01, -6.000000e+01, -6.800000e+01], [-3.000000e+00, -3.900000e+01, 8.800000e+01]]], [[[-4.300000e+01, -9.500000e+01, 6.400000e+01], [4.600000e+01, -1.250000e+02, -6.300000e+01], [-2.100000e+01, -2.500000e+01, -2.500000e+01]], [[-1.180000e+02, -1.030000e+02, -1.200000e+01], [8.400000e+01, 6.700000e+01, 5.500000e+01], [-1.050000e+02, 1.300000e+01, -1.000000e+01]], [[9.700000e+01, -1.240000e+02, 3.900000e+01], [-2.800000e+01, -1.120000e+02, 1.160000e+02], [7.400000e+01, 1.040000e+02, 7.200000e+01]], [[1.400000e+01, 5.800000e+01, 0.000000e+00], [3.700000e+01, -4.800000e+01, 2.600000e+01], [3.300000e+01, -6.400000e+01, 5.300000e+01]]], [[[-1.240000e+02, 1.040000e+02, 1.050000e+02], [-1.400000e+01, 0.000000e+00, -2.500000e+01], [1.040000e+02, -4.600000e+01, -8.700000e+01]], [[8.700000e+01, -1.050000e+02, 6.900000e+01], [9.400000e+01, 8.800000e+01, 4.700000e+01], [5.300000e+01, 9.300000e+01, -3.400000e+01]], [[-6.200000e+01, -4.400000e+01, -1.000000e+01], [8.100000e+01, 1.100000e+02, 3.200000e+01], [1.000000e+01, 7.200000e+01, 3.000000e+01]], [[1.170000e+02, 6.400000e+01, 4.100000e+01], [0.000000e+00, -5.000000e+01, -3.900000e+01], [-1.080000e+02, 7.000000e+00, -1.200000e+01]]], [[[-7.300000e+01, -4.700000e+01, 7.000000e+00], [7.200000e+01, -1.700000e+01, 9.000000e+01], [-1.130000e+02, 4.400000e+01, 8.000000e+01]], [[-6.000000e+01, -1.020000e+02, -7.900000e+01], [-1.110000e+02, -4.300000e+01, 6.800000e+01], [-2.100000e+01, 5.300000e+01, 1.200000e+02]], [[-1.090000e+02, -6.900000e+01, 3.000000e+01], [1.200000e+02, -7.000000e+00, 1.070000e+02], [-3.000000e+01, 4.200000e+01, 6.600000e+01]], [[4.300000e+01, 1.600000e+01, -5.700000e+01], [9.500000e+01, 1.250000e+02, -9.900000e+01], [-3.000000e+01, 1.000000e+00, 1.260000e+02]]]]> : tensor<4x4x3x3xf32>
  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK:   return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @WeightsUI8ToFakeQuantizeNoSubNoMult
func.func @WeightsUI8ToFakeQuantizeNoSubNoMult(%arg0: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]],[[31, 67, 22], [70, 55, 12], [99, 42, 90]],[[6, 18, 95], [8, 37, 64], [40, 31, 41]],[[35, 2, 98], [94, 60, 68], [3, 39, 88]]],[[[43, 95, 64], [46, 125, 63], [21, 25, 25]],[[118, 103, 12], [84, 67, 55], [105, 13, 10]],[[97, 124, 39], [28, 112, 116], [74, 104, 72]],[[14, 58, 0], [37, 48, 26], [33, 64, 53]]],[[[124, 104, 105], [14, 0, 25], [104, 46, 87]],[[87, 105, 69], [94, 88, 47], [53, 93, 34]],[[62, 44, 10], [81, 110, 32], [10, 72, 30]],[[117, 64, 41], [0, 50, 39], [108, 7, 12]]],[[[73, 47, 7], [72, 17, 90], [113, 44, 80]],[[60, 102, 79], [111, 43, 68], [21, 53, 120]],[[109, 69, 30], [120, 7, 107], [30, 42, 66]],[[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.ConvertElemType<f32>]
  %2 = IE.Convolution(%arg0, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[7.300000e+01, 6.900000e+01, 9.500000e+01], [4.700000e+01, 8.500000e+01, -7.000000e+01], [3.600000e+01, 7.200000e+01, -8.200000e+01]], [[3.100000e+01, -6.700000e+01, 2.200000e+01], [-7.000000e+01, -5.500000e+01, 1.200000e+01], [-9.900000e+01, 4.200000e+01, 9.000000e+01]], [[6.000000e+00, -1.800000e+01, 9.500000e+01], [-8.000000e+00, -3.700000e+01, -6.400000e+01], [4.000000e+01, 3.100000e+01, -4.100000e+01]], [[3.500000e+01, -2.000000e+00, -9.800000e+01], [-9.400000e+01, -6.000000e+01, -6.800000e+01], [-3.000000e+00, -3.900000e+01, 8.800000e+01]]], [[[-4.300000e+01, -9.500000e+01, 6.400000e+01], [4.600000e+01, -1.250000e+02, -6.300000e+01], [-2.100000e+01, -2.500000e+01, -2.500000e+01]], [[-1.180000e+02, -1.030000e+02, -1.200000e+01], [8.400000e+01, 6.700000e+01, 5.500000e+01], [-1.050000e+02, 1.300000e+01, -1.000000e+01]], [[9.700000e+01, -1.240000e+02, 3.900000e+01], [-2.800000e+01, -1.120000e+02, 1.160000e+02], [7.400000e+01, 1.040000e+02, 7.200000e+01]], [[1.400000e+01, 5.800000e+01, 0.000000e+00], [3.700000e+01, -4.800000e+01, 2.600000e+01], [3.300000e+01, -6.400000e+01, 5.300000e+01]]], [[[-1.240000e+02, 1.040000e+02, 1.050000e+02], [-1.400000e+01, 0.000000e+00, -2.500000e+01], [1.040000e+02, -4.600000e+01, -8.700000e+01]], [[8.700000e+01, -1.050000e+02, 6.900000e+01], [9.400000e+01, 8.800000e+01, 4.700000e+01], [5.300000e+01, 9.300000e+01, -3.400000e+01]], [[-6.200000e+01, -4.400000e+01, -1.000000e+01], [8.100000e+01, 1.100000e+02, 3.200000e+01], [1.000000e+01, 7.200000e+01, 3.000000e+01]], [[1.170000e+02, 6.400000e+01, 4.100000e+01], [0.000000e+00, -5.000000e+01, -3.900000e+01], [-1.080000e+02, 7.000000e+00, -1.200000e+01]]], [[[-7.300000e+01, -4.700000e+01, 7.000000e+00], [7.200000e+01, -1.700000e+01, 9.000000e+01], [-1.130000e+02, 4.400000e+01, 8.000000e+01]], [[-6.000000e+01, -1.020000e+02, -7.900000e+01], [-1.110000e+02, -4.300000e+01, 6.800000e+01], [-2.100000e+01, 5.300000e+01, 1.200000e+02]], [[-1.090000e+02, -6.900000e+01, 3.000000e+01], [1.200000e+02, -7.000000e+00, 1.070000e+02], [-3.000000e+01, 4.200000e+01, 6.600000e+01]], [[4.300000e+01, 1.600000e+01, -5.700000e+01], [9.500000e+01, 1.250000e+02, -9.900000e+01], [-3.000000e+01, 1.000000e+00, 1.260000e+02]]]]> : tensor<4x4x3x3xf32>
  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  // CHECK:   return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @WeightsUI8MultSubToFakeQuantize
func.func @WeightsUI8MultSubToFakeQuantize(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<16x1x1x1xf32> = dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]> : tensor<16x1x1x1xui8>, [#const.ConvertElemType<f32>]
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.500000e+01> : tensor<1x1x1x1xf32>
  %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<0.0566197559> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.41549385> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<13.0225439> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}  = dense<[[[[2.700000e+01]]], [[[2.500000e+01]]], [[[3.900000e+01]]], [[[2.200000e+01]]], [[[2.700000e+01]]], [[[2.500000e+01]]], [[[2.100000e+01]]], [[[2.700000e+01]]], [[[3.100000e+01]]], [[[2.900000e+01]]], [[[4.200000e+01]]], [[[2.700000e+01]]], [[[2.700000e+01]]], [[[2.800000e+01]]], [[[3.300000e+01]]], [[[3.300000e+01]]]]> : tensor<16x1x1x1xf32>
  // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  // CHECK-DAG:   [[ACT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST_5]], [[CST_0]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  // CHECK-DAG:   [[WT_FQ:%.*]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  // CHECK:   [[GRUP_CONV:%.*]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>

  // CHECK:   return [[GRUP_CONV]] : tensor<1x16x32x32xf32>
}


// -----

// CHECK-LABEL: @MultipleConsumerConstSubMult
func.func @MultipleConsumerConstSubMult(%arg0: tensor<1x4x48xf16>) -> tensor<1x4x48xf16> {
  %cst_0 = const.Declare tensor<48xf16> = dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]> : tensor<48xsi8>, [#const.ConvertElemType<f16>]
  %cst_1 = const.Declare tensor<1xf16> = dense<8.800000e+01> : tensor<1xf16>
  %cst_2 = const.Declare tensor<1xf16> = dense<9.88533836E-4> : tensor<1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %2 = IE.Add(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %3 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %4 = IE.Multiply(%3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %5 = IE.Add(%2, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %6 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %7 = IE.Add(%5, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %8 = IE.Add(%7, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %9 = IE.Multiply(%cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>  
  %10 = IE.Add(%8, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %11 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>  
  %12 = IE.Add(%10, %11) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  return %12 : tensor<1x4x48xf16>

  // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1xf16> = dense<1.256100e-01> : tensor<1xf16>
  // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1xf16> = dense<-1.256100e-01> : tensor<1xf16>
  // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1xf16> = dense<-1.270000e+02> : tensor<1xf16>
  // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1xf16> = dense<1.270000e+02> : tensor<1xf16>
  // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<1xf16> = dense<-2.126460e-01> : tensor<1xf16>
  // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1xf16> = dense<3.857420e-02> : tensor<1xf16>
  // CHECK-DAG:   [[CST_6:%.*]] = const.Declare tensor<48xf16>
  // CHECK-SAME{LITERAL}  = dense<[9.000000e+00, -5.000000e+00, -2.500000e+01, 5.200000e+01, -7.700000e+01, -1.230000e+02, 2.400000e+01, 6.700000e+01, -3.200000e+01, 1.100000e+01, -2.400000e+01, 9.300000e+01, -1.700000e+01, -1.270000e+02, -4.600000e+01, -3.800000e+01, 5.300000e+01, -8.800000e+01, -1.080000e+02, -6.000000e+01, 9.000000e+00, -8.000000e+00, -7.800000e+01, 1.060000e+02, -3.300000e+01, 1.400000e+01, -1.100000e+01, -2.100000e+01, -9.400000e+01, -7.200000e+01, 4.900000e+01, 1.250000e+02, -5.800000e+01, -9.300000e+01, 9.100000e+01, 4.400000e+01, 1.230000e+02, -9.900000e+01, -5.900000e+01, 1.500000e+01, -1.240000e+02, -1.300000e+01, 8.900000e+01, -9.200000e+01, -9.700000e+01, 1.000000e+01, -1.600000e+01, 3.800000e+01]> : tensor<48xf16>
  // CHECK-DAG:   [[CST_7:%.*]] = const.Declare tensor<1xf16> = dense<3.900000e+01> : tensor<1xf16>
  // CHECK-DAG:   [[CST_8:%.*]] = const.Declare tensor<1xf16> = dense<-2.150000e+02> : tensor<1xf16>
  // CHECK-DAG:   [[WT_FQ_0:%.*]] = IE.FakeQuantize([[CST_6]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<48xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<48xf16>
  // CHECK-DAG:   [[WT_FQ_1:%.*]] = IE.FakeQuantize([[CST_6]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<48xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<48xf16>
  // CHECK-DAG:   [[WT_FQ_2:%.*]] = IE.FakeQuantize([[CST_6]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<48xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<48xf16>
  // CHECK-DAG:   [[WT_FQ_3:%.*]] = IE.FakeQuantize([[CST_6]], [[CST_2]], [[CST_3]], [[CST_8]], [[CST_7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<48xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<48xf16>
  // CHECK-DAG:   [[WT_FQ_4:%.*]] = IE.FakeQuantize([[CST_6]], [[CST_2]], [[CST_3]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<48xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<48xf16>
  // CHECK:   [[ADD_0:%.*]] = IE.Add(%arg0, [[WT_FQ_0]])
  // CHECK:   [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[WT_FQ_2]])
  // CHECK:   [[ADD_2:%.*]] = IE.Add([[ADD_1]], [[WT_FQ_3]])
  // CHECK:   [[ADD_3:%.*]] = IE.Add([[ADD_2]], [[WT_FQ_0]])
  // CHECK:   [[ADD_4:%.*]] = IE.Add([[ADD_3]], [[WT_FQ_4]])
  // CHECK:   [[ADD_5:%.*]] = IE.Add([[ADD_4]], [[WT_FQ_1]])

  // CHECK:   return [[ADD_5]] : tensor<1x4x48xf16>
}

