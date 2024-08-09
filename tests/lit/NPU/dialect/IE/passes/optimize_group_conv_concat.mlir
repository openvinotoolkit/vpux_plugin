//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-group-conv-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @OptimizeGroupConvConcat
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcat(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.*]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcat_SwapConcatInput
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcat_SwapConcatInput(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%1, %0) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.*]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcatWithReLu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcatWithReLu(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.*]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfRootTensorIsNotReLu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @NotConvertIfRootTensorIsNotReLu(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.*]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[GROUP_CONV:%.*]] = IE.GroupConvolution([[CONV]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>

    // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16

    // CHECK:           return [[CONCAT]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfPostOpIsNotReLu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @NotConvertIfPostOpIsNotReLu(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>,
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.*]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[GROUP_CONV:%.*]] = IE.GroupConvolution([[CONV]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.Sigmoid", attrs = {}>, strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>

    // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16

    // CHECK:           return [[CONCAT]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfConcatInputsDoNotHaveSameRoot
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x16x144x144xf16>,
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<1x16x144x144xf16>
func.func @NotConvertIfConcatInputsDoNotHaveSameRoot(%arg0: tensor<1x16x144x144xf16>, %arg1: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%arg1, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2 : tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.*]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT0]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[GROUP_CONV:%.*]] = IE.GroupConvolution([[INPUT1]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>

    // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16

    // CHECK:           return [[CONCAT]] : tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @OptimizeGroupConvConcatWhenRootTensorHasExtraUser
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x144x144xf16>
func.func @OptimizeGroupConvConcatWhenRootTensorHasExtraUser(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x16x144x144xf16>, tensor<1x32x144x144xf16>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %cst_1 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x16x144x144xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x144x144xf16>
    %2 = IE.Tanh(%0) : tensor<1x16x144x144xf16> -> tensor<1x16x144x144xf16>
    %3 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>

    return %2, %3 : tensor<1x16x144x144xf16>, tensor<1x32x144x144xf16>

    // CHECK-DAG:       [[NEW_WEIGHTS:%.*]] = const.Declare tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[TANH:%.*]] = IE.Tanh([[CONV]]) : tensor<1x16x144x144xf16> -> tensor<1x16x144x144xf16>
    // CHECK:           [[NEW_CONV:%.*]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x144x144xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x144x144xf16>

    // CHECK:           return [[TANH]], [[NEW_CONV]] : tensor<1x16x144x144xf16>, tensor<1x32x144x144xf16>
}

// -----

// CHECK-LABEL: @NotConvertForConcatWithLargeChannelNum
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x256x36x36xf16>
func.func @NotConvertForConcatWithLargeChannelNum(%arg0: tensor<1x256x36x36xf16>) -> (tensor<1x512x36x36xf16>) {
    %cst_0 = const.Declare tensor<256x256x1x1xf16> = dense<1.0> : tensor<256x256x1x1xf16>
    %cst_1 = const.Declare tensor<256x1x3x3xf16> = dense<1.0> : tensor<256x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst_0) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x256x36x36xf16>, tensor<256x256x1x1xf16> -> tensor<1x256x36x36xf16>
    %1 = IE.GroupConvolution(%0, %cst_1) {
        dilations = [1, 1],
        groups = 256 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x36x36xf16>, tensor<256x1x3x3xf16> -> tensor<1x256x36x36xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x36x36xf16>, tensor<1x256x36x36xf16> -> tensor<1x512x36x36xf16>

    return %2 : tensor<1x512x36x36xf16>

    // CHECK-DAG:       [[WEIGHTS_0:%.*]] = const.Declare tensor<256x256x1x1xf16> = dense<1.000000e+00> : tensor<256x256x1x1xf16>
    // CHECK-DAG:       [[WEIGHTS_1:%.*]] = const.Declare tensor<256x1x3x3xf16> = dense<1.000000e+00> : tensor<256x1x3x3xf16>

    // CHECK:           [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x36x36xf16>, tensor<256x256x1x1xf16> -> tensor<1x256x36x36xf16>
    // CHECK:           [[GROUP_CONV:%.*]] = IE.GroupConvolution([[CONV]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x256x36x36xf16>, tensor<256x1x3x3xf16> -> tensor<1x256x36x36xf16>

    // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CONV]], [[GROUP_CONV]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x36x36xf16>, tensor<1x256x36x36xf16> -> tensor<1x512x36x36xf16>

    // CHECK:           return [[CONCAT]] : tensor<1x512x36x36xf16>
}
