//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-fq-and-mul %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @FuseFakeQuantizeAndMultiplyLhsIsActivation
func.func @FuseFakeQuantizeAndMultiplyLhsIsActivation(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst_0 = const.Declare tensor<288x16x3x3xf32> = dense<1.0> : tensor<288x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>
    %cst_4 = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>
    %0 = IE.FakeQuantize(%cst_0, %cst_1, %cst_2, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    %cst_5 = const.Declare tensor<288x1x1x1xf32> = dense<2.0> : tensor<288x1x1x1xf32>
    %1 = IE.Multiply(%0, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<288x16x3x3xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    %cst_6 = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    %2 = IE.Reshape(%1, %cst_6) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    %3 = IE.GroupConvolution(%arg0, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %3 : tensor<1x288x20x20xf32>

    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CST0:%.*]] = const.Declare tensor<288x1x1x1xf32> = dense<2.540000e+02> : tensor<288x1x1x1xf32>
    // CHECK:       [[CST1:%.*]] = const.Declare tensor<288x1x1x1xf32> = dense<-2.540000e+02> : tensor<288x1x1x1xf32>
    // CHECK:       [[CST2:%.*]] = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    // CHECK:       [[CST3:%.*]] = const.Declare tensor<288x16x3x3xf32> = dense<1.000000e+00> : tensor<288x16x3x3xf32>
    // CHECK:       [[CST4:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[CST5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize([[CST3]], [[CST4]], [[CST5]], [[CST1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[FQ]], [[CST2]]) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    // CHECK:       [[CONV:%.*]]  = IE.GroupConvolution(%arg0, [[RESHAPE]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}

// -----

// CHECK-LABEL: @FuseFakeQuantizeAndMultiplyRhsIsActivation
func.func @FuseFakeQuantizeAndMultiplyRhsIsActivation(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst_0 = const.Declare tensor<288x16x3x3xf32> = dense<1.0> : tensor<288x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>
    %cst_4 = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>
    %0 = IE.FakeQuantize(%cst_0, %cst_1, %cst_2, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    %cst_5 = const.Declare tensor<288x1x1x1xf32> = dense<2.0> : tensor<288x1x1x1xf32>
    %1 = IE.Multiply(%cst_5, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<288x1x1x1xf32>, tensor<288x16x3x3xf32> -> tensor<288x16x3x3xf32>
    %cst_6 = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    %2 = IE.Reshape(%1, %cst_6) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    %3 = IE.GroupConvolution(%arg0, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %3 : tensor<1x288x20x20xf32>

    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CST0:%.*]] = const.Declare tensor<288x1x1x1xf32> = dense<2.540000e+02> : tensor<288x1x1x1xf32>
    // CHECK:       [[CST1:%.*]] = const.Declare tensor<288x1x1x1xf32> = dense<-2.540000e+02> : tensor<288x1x1x1xf32>
    // CHECK:       [[CST2:%.*]] = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    // CHECK:       [[CST3:%.*]] = const.Declare tensor<288x16x3x3xf32> = dense<1.000000e+00> : tensor<288x16x3x3xf32>
    // CHECK:       [[CST4:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[CST5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize([[CST3]], [[CST4]], [[CST5]], [[CST1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[FQ]], [[CST2]]) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    // CHECK:       [[CONV:%.*]]  = IE.GroupConvolution(%arg0, [[RESHAPE]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}

// -----

// CHECK-LABEL: @NotFuseFakeQuantizeAndMultiply
func.func @NotFuseFakeQuantizeAndMultiply(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst = const.Declare tensor<1x288x1x1xf32> = dense<2.000000e+00> : tensor<1x288x1x1xf32>
    %cst_0 = const.Declare tensor<18x16x16x3x3xf32> = dense<1.000000e+00> : tensor<18x16x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x288x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x288x20x20xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x288x20x20xf32>, tensor<1x288x1x1xf32> -> tensor<1x288x20x20xf32>
    %2 = IE.GroupConvolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %2 : tensor<1x288x20x20xf32>

    // CHECK:       [[CST0:%.*]] = const.Declare tensor<1x288x1x1xf32> = dense<2.000000e+00> : tensor<1x288x1x1xf32>
    // CHECK:       [[CST1:%.*]] = const.Declare tensor<18x16x16x3x3xf32> = dense<1.000000e+00> : tensor<18x16x16x3x3xf32>
    // CHECK:       [[CST2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[CST3:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST2]], [[CST3]], [[CST2]], [[CST3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x288x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x288x20x20xf32>
    // CHECK:       [[MUL:%.*]] = IE.Multiply([[FQ]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x288x20x20xf32>, tensor<1x288x1x1xf32> -> tensor<1x288x20x20xf32>
    // CHECK:       [[CONV:%.*]]  = IE.GroupConvolution([[MUL]], [[CST1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}
