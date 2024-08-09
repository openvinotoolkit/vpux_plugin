//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-multiply-with-matmul %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// -----

// CHECK-LABEL: @SwapMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @SwapMultiplyWithMatmul(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x1024xf32>

    // CHECK:       return  [[MULTIPLY]] : tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @SwapTwoMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x80x1024xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x80x1024xf32>
func.func @SwapTwoMultiplyWithMatmul(%arg0: tensor<1x32x80x1024xf32>, %arg1: tensor<1x32x80x1024xf32>) -> tensor<1x32x80x80xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>
    %cst0 = const.Declare tensor<1x1x1x1xf32> = dense<0.2> : tensor<1x1x1x1xf32>
    %1 = IE.Multiply(%arg1, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>

    %2 = IE.MatMul(%0, %1) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>

    return %2 : tensor<1x32x80x80xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.*]]  = IE.MatMul([[INPUT1]], [[INPUT2]]) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY_0:%.*]] = IE.Multiply([[MULTIPLY]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf3

    // CHECK:       return  [[MULTIPLY_0]] : tensor<1x32x80x80xf32>
}


// -----

// CHECK-LABEL: @NotSwapWithPostOp
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NotSwapWithPostOp(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NotSwapForNotSplatConst
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x2xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x2xf32>
func.func @NotSwapForNotSplatConst(%arg0: tensor<1x32x1024x2xf32>, %arg1: tensor<1x32x1x2xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x2xf32> = dense<[[[[1.0,1.6]]]]> : tensor<1x1x1x2xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x32x1024x2xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x2xf32>, tensor<1x32x1024x2xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NoSwapForMultiUser
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NoSwapForMultiUser(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> (tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %0, %1 : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MULTIPLY:%.*]], [[MATMUL]] : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @NotBeneficialForSwap
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1025x80xf32>
func.func @NotBeneficialForSwap(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1025x80xf32>) -> tensor<1x32x1025x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1025x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1025x1024xf32>

    return %1 : tensor<1x32x1025x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1025x1024xf32>
}
