//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-subtract-to-add --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @SubtractWithConstAtSecondInputDiffShapes
func.func @SubtractWithConstAtSecondInputDiffShapes(%arg0: tensor<1x16x32x1xf32>) -> tensor<1x16x32x1xf32> {
    %cst = const.Declare tensor<1x16x1x1xf32> = dense<2.0> : tensor<1x16x1x1xf32>
    %0 = IE.Subtract(%arg0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x1xf32>

    return %0 : tensor<1x16x32x1xf32>

    // CHECK-DAG:       [[CST1:%.*]] = const.Declare tensor<1x16x1x1xf32> = dense<2.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK-SAME:      #const.Rescale<-1.000000e+00 : f64>
    // CHECK:       [[ADD:%.*]] = IE.Add(%arg0, [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x1xf32>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithConstInputsSameShapes
func.func @SubtractWithConstInputsSameShapes(%arg0: tensor<1x16x32x1xf32>) -> tensor<1x16x32x1xf32> {
    %cst = const.Declare tensor<1x16x32x1xf32> = dense<2.0> : tensor<1x16x32x1xf32>
    %0 = IE.Subtract(%arg0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf32>, tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>

    return %0 : tensor<1x16x32x1xf32>

    // CHECK-DAG:       [[CST1:%.*]] = const.Declare tensor<1x16x32x1xf32> = dense<2.000000e+00> : tensor<1x16x32x1xf32>
    // CHECK-SAME:      #const.Rescale<-1.000000e+00 : f64>
    // CHECK:       [[ADD:%.*]] = IE.Add(%arg0, [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf32>, tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithInputNeedsBroadcast
func.func @NotConvertSubtractWithInputNeedsBroadcast(%arg0: tensor<1x16x32x1xf32>) -> tensor<1x16x32x1xf32> {
    %cst = const.Declare tensor<1x16x1x1xf32> = dense<2.0> : tensor<1x16x1x1xf32>
    %input_2 = IE.ReLU(%cst) : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf32>
    %0 = IE.Subtract(%arg0, %input_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x1xf32>

    return %0 : tensor<1x16x32x1xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x16x1x1xf32> = dense<2.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       [[RELU:%.*]] = IE.ReLU([[CST]]) : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       [[SUBTRACT:%.*]] = IE.Subtract(%arg0, [[RELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x1xf32>
}

// -----

// CHECK-LABEL: @SubtractWithActivationInputsSameShapes
func.func @SubtractWithActivationInputsSameShapes(%arg0: tensor<1x16x32x1xf32>) -> tensor<1x16x32x1xf32> {
    %input_1 = IE.ReLU(%arg0) : tensor <1x16x32x1xf32> -> tensor<1x16x32x1xf32>
    %cst = const.Declare tensor<1x16x32x1xf32> = dense<2.0> : tensor<1x16x32x1xf32>
    %input_2 = IE.ReLU(%cst) : tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>
    %0 = IE.Subtract(%input_1, %input_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf32>, tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>

    return %0 : tensor<1x16x32x1xf32>

    // CHECK-DAG:       [[FILTER:%.*]] = const.Declare tensor<16x1x1x1xf32> = dense<-1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:       [[CST2:%.*]] = const.Declare tensor<1x16x32x1xf32> = dense<2.000000e+00> : tensor<1x16x32x1xf32>
    // CHECK-DAG:       [[RELU_1:%.*]] = IE.ReLU(%arg0) : tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>
    // CHECK-DAG:       [[RELU_2:%.*]] = IE.ReLU([[CST2]]) : tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>
    // CHECK:       [[GROUP_CONV:%.*]] = IE.GroupConvolution([[RELU_2]], [[FILTER]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x16x32x1xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x1xf32>
    // CHECK:       [[ADD:%.*]] = IE.Add([[RELU_1]], [[GROUP_CONV]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf32>, tensor<1x16x32x1xf32> -> tensor<1x16x32x1xf32>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithFQConstInputsSameShape
func.func @SubtractWithFQConstInputsSameShape(%arg0: tensor<1x1x1x64xf16>) -> tensor<1x1x1x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %2 = IE.FakeQuantize(%cst, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    return %3 : tensor<1x1x1x64xf16>

    // CHECK-DAG:       [[CONST:%.*]] = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK-DAG:       [[VAL_LOW_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_LOW_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_HIGH_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[ADD_INPUT_1:%.*]] = IE.FakeQuantize(%arg0, [[VAL_LOW_2]], [[VAL_HIGH_1]], [[VAL_LOW_2]], [[VAL_HIGH_1]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[ADD_INPUT_2:%.*]] = IE.FakeQuantize([[CONST]], [[VAL_LOW_1]], [[VAL_LOW_2]], [[VAL_LOW_1]], [[VAL_LOW_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[ADD:%.*]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:          tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithFQConstInputsDiffShape
func.func @SubtractWithFQConstInputsDiffShape(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x1xf16>
    %2 = IE.FakeQuantize(%cst, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x1xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>

    return %3 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[CONST:%.*]] = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK-DAG:       [[VAL_LOW_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_HIGH_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_HIGH_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[ADD_INPUT_1:%.*]] = IE.FakeQuantize(%arg0, [[VAL_HIGH_2]], [[VAL_HIGH_1]], [[VAL_HIGH_2]], [[VAL_HIGH_1]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x16x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK-DAG:       [[ADD_INPUT_2:%.*]] = IE.FakeQuantize([[CONST]], [[VAL_LOW_2]], [[VAL_HIGH_2]], [[VAL_LOW_2]], [[VAL_HIGH_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x1x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x1xf16>
    // CHECK-DAG:       [[ADD:%.*]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:          tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithFQActInputNeedsBroadcast
func.func @NotConvertSubtractWithFQActInputNeedsBroadcast(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>
    %act_1 = IE.ReLU(%arg0) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    %act_2 = IE.ReLU(%cst) : tensor<1x1x32x1xf16> -> tensor<1x1x32x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%act_1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x1xf16>
    %2 = IE.FakeQuantize(%act_2, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x1xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>

    return %3 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CST_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CST_3:%.*]] = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>
    // CHECK-DAG:       [[RELU_0:%.*]] = IE.ReLU(%arg0) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK-DAG:       [[RELU_1:%.*]] = IE.ReLU([[CST_3]]) : tensor<1x1x32x1xf16> -> tensor<1x1x32x1xf16>
    // CHECK:           [[FQ_0:%.*]] = IE.FakeQuantize([[RELU_0]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]])
    // CHECK:           [[FQ_1:%.*]] = IE.FakeQuantize([[RELU_1]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST_0]])
    // CHECK:           [[SUB:%.*]] = IE.Subtract([[FQ_0]], [[FQ_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>
}

// -----

// CHECK-LABEL: @SubtractWithFQActInputsSameShape
func.func @SubtractWithFQActInputsSameShape(%arg0: tensor<1x1x1x64xf16>) -> tensor<1x1x1x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    %act_1 = IE.ReLU(%arg0) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    %act_2 = IE.ReLU(%cst) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%act_1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %2 = IE.FakeQuantize(%act_2, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    return %3 : tensor<1x1x1x64xf16>

    // CHECK-DAG:       [[FILTER:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2_NEGATED:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MIN:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[RELU_2_INPUT:%.*]] = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[RELU_1:%.*]] = IE.ReLU(%arg0) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[RELU_2:%.*]] = IE.ReLU([[RELU_2_INPUT]]) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[ADD_INPUT_1:%.*]] = IE.FakeQuantize([[RELU_1]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX_1]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX_1]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[CONV_FQ_INPUT:%.*]] = IE.FakeQuantize([[RELU_2]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX_2]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[CONV_FQ_FILTER:%.*]] = IE.FakeQuantize([[FILTER]], [[FILTER]], [[FILTER]], [[FILTER]], [[FILTER]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[GROUP_CONV:%.*]] = IE.GroupConvolution([[CONV_FQ_INPUT]], [[CONV_FQ_FILTER]])
    // CHECK-SAME:      dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[ADD_INPUT_2:%.*]] = IE.FakeQuantize([[GROUP_CONV]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[ADD:%.*]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithInputNeedsBroadcast
func.func @NotConvertSubtractWithInputNeedsBroadcast(%arg0: tensor<1x1x1x8192xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x8192xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x8192xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x8192xf16>
    return %0 : tensor<1x1x1x8192xf16>

    // CHECK:  [[SUB:%.*]] = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x8192xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x8192xf16>
    // CHECK:  return [[SUB]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithInputNeedsExpansion
func.func @NotConvertSubtractWithInputNeedsExpansion(%arg0: tensor<1x10x1x57xf16>, %arg1: tensor<1x10x1x57xf16>) -> tensor<1x10x1x57xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1x57xf16>, tensor<1x10x1x57xf16> -> tensor<1x10x1x57xf16>
    return %0 : tensor<1x10x1x57xf16>

    // CHECK:  [[SUB:%.*]] = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1x57xf16>, tensor<1x10x1x57xf16> -> tensor<1x10x1x57xf16>
    // CHECK:  return [[SUB]]
}

// -----

// CHECK-LABEL: @SubtractIntInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x128x128xsi32>
func.func @SubtractIntInput(%INPUT: tensor<1x1x128x128xsi32>) -> tensor<1x1x128x128xsi32> {
    %cst_0 = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi64>, [#const.ConvertElemType<si32>]
    %3 = IE.Subtract(%cst_0, %INPUT) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x128x128xsi32> -> tensor<1x1x128x128xsi32>

    return %3 : tensor<1x1x128x128xsi32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[CST]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x128x128xsi32> -> tensor<1x1x128x128xsi32>
    // CHECK:       return [[SUBTRACT]]
}
