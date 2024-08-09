//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --matmul-inputs-to-2d="enable-grouped-matmul=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// Two test below show that CMX calculations are quantization sensitive. 
// Tests are valid only for NPU37XX due to dependency on CMX memory otherwise 
// batched matmul related changes are not specific to NPU37XX.

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dQuantized
func.func @MatMul4dInputs4dWeightsTo2dQuantized(%arg0: tensor<1x2x650x750xf16>) -> tensor<1x2x650x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_4 = const.Declare tensor<1x2x512x750xf16> = dense<1.0> : tensor<1x2x512x750xf32>, [#const.ConvertElemType<f16>]
    %0 = IE.FakeQuantize(%cst_4, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    %2 = IE.MatMul(%1, %0) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_3:%.*]] = const.Declare tensor<1x2x512x750xf16> = dense<1.000000e+00> : tensor<1x2x512x750xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[FQ0:%.*]] = IE.FakeQuantize([[CST_3]], [[CST_2]], [[CST_1]], [[CST]], [[CST_0]]) 
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    // CHECK: [[FQ1:%.*]] = IE.FakeQuantize(%arg0, [[CST_2]], [[CST_1]], [[CST]], [[CST_0]]) 
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    // CHECK: [[MATMUL:%.*]] = IE.MatMul([[FQ1]], [[FQ0]]) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>
    // CHECK: return [[MATMUL]] : tensor<1x2x650x512xf16>

    return %2 : tensor<1x2x650x512xf16>

}

// -----
// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dNoQuantizeDontFitCMX
func.func @MatMul4dInputs4dWeightsTo2dNoQuantizeDontFitCMX(%arg0: tensor<1x2x650x750xf16>, %arg1: tensor<1x2x512x750xf16>) -> tensor<1x2x650x512xf16> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>

    return %0 : tensor<1x2x650x512xf16>

    
    // CHECK: [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK: [[RESHAPE0:%.*]] = IE.Reshape([[SLICE0]]) {shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK: [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 650, 750] : tensor<1x2x650x750xf16> to tensor<1x1x650x750xf16>
    // CHECK: [[RESHAPE1:%.*]] = IE.Reshape([[SLICE1]]) {shape_value = [650, 750]} : tensor<1x1x650x750xf16> -> tensor<650x750xf16>
    // CHECK: [[SLICE2:%.*]] = IE.Slice %arg1 [0, 0, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK: [[RESHAPE2:%.*]] = IE.Reshape([[SLICE2]]) {shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK: [[SLICE3:%.*]] = IE.Slice %arg1 [0, 1, 0, 0] [1, 1, 512, 750] : tensor<1x2x512x750xf16> to tensor<1x1x512x750xf16>
    // CHECK: [[RESHAPE3:%.*]] = IE.Reshape([[SLICE3]]) {shape_value = [512, 750]} : tensor<1x1x512x750xf16> -> tensor<512x750xf16>
    // CHECK: [[MATMUL0:%.*]] = IE.MatMul([[RESHAPE0]], [[RESHAPE2]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK: [[MATMUL1:%.*]] = IE.MatMul([[RESHAPE1]], [[RESHAPE3]]) {transpose_b} : tensor<650x750xf16>, tensor<512x750xf16> -> tensor<650x512xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[MATMUL0]], [[MATMUL1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<650x512xf16>, tensor<650x512xf16> -> tensor<1300x512xf16>
    // CHECK: [[RESHAPE4:%.*]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 2, 650, 512]} : tensor<1300x512xf16> -> tensor<1x2x650x512xf16>
    // CHECK: return [[RESHAPE4]] : tensor<1x2x650x512xf16>
}

// -----

// CHECK-LABEL: @MatMul4dInputs4dWeightsTo2dOutputQuantized
func.func @MatMul4dInputs4dWeightsTo2dOutputQuantized(%arg0: tensor<1x2x650x750xf16>) -> tensor<1x2x650x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_4 = const.Declare tensor<1x2x512x750xf16> = dense<1.0> : tensor<1x2x512x750xf32>, [#const.ConvertElemType<f16>]
    %0 = IE.FakeQuantize(%cst_4, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    %2 = IE.MatMul(%1, %0) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>
    %3 = IE.FakeQuantize(%2, %cst_3, %cst_2, %cst, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x650x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x512xf16>
    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.285366833> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.435113788> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[CST_3:%.*]] = const.Declare tensor<1x2x512x750xf16> = dense<1.000000e+00> : tensor<1x2x512x750xf32>, [#const.ConvertElemType<f16>]
    // CHECK: [[FQ0:%.*]] = IE.FakeQuantize([[CST_3]], [[CST_2]], [[CST_1]], [[CST]], [[CST_0]]) 
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x512x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x512x750xf16>
    // CHECK: [[FQ1:%.*]] = IE.FakeQuantize(%arg0, [[CST_2]], [[CST_1]], [[CST]], [[CST_0]]) 
    // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x650x750xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x650x750xf16>
    // CHECK: [[MATMUL:%.*]] = IE.MatMul([[FQ1]], [[FQ0]]) {transpose_b} : tensor<1x2x650x750xf16>, tensor<1x2x512x750xf16> -> tensor<1x2x650x512xf16>
    // CHECK: [[FQ2:%.*]] = IE.FakeQuantize([[MATMUL]], [[CST_2]], [[CST_1]], [[CST]], [[CST_0]]) 
    // CHECK: return [[FQ2]] : tensor<1x2x650x512xf16>

    return %3 : tensor<1x2x650x512xf16>

}
