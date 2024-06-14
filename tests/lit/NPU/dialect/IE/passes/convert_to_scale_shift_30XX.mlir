//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-scale-shift %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @ConvertAddToScaleShift
func.func @ConvertAddToScaleShift(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftWithReshape
func.func @ConvertAddToScaleShiftWithReshape(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x16x64x64xf16> {
    %bias = const.Declare tensor<1x1x1x16xf16> = dense<2.0> : tensor<1x1x1x16xf16>
    %reshape = IE.Reshape(%bias) { shape_value = [1, 16, 1, 1] } : tensor<1x1x1x16xf16> -> tensor<1x16x1x1xf16>
    %0 = IE.Add(%arg0, %reshape)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x64x64xf16>

    return %0 : tensor<1x16x64x64xf16>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x16xf16>, [#const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x64x64xf16>
    // CHECK:       return %[[VAL0]]
}


// CHECK-LABEL: @ConvertAddWithNegativeConstToScaleShift
func.func @ConvertAddWithNegativeConstToScaleShift(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %cst = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Negative(%cst) : tensor<1x3x1x1xf16> -> tensor<1x3x1x1xf16>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %1 : tensor<1x3x224x224xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[CST]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftBroadcastChannels
func.func @ConvertAddToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddWithConstFQToScaleShiftBroadcastChannels
func.func @ConvertAddWithConstFQToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%bias, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    %1 = IE.Multiply(%arg0, %0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %1 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK-DAG:       %[[CONST_HIGH:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       %[[CONST_LOW:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.FakeQuantize(%[[BIAS]], %[[CONST_LOW]], %[[CONST_HIGH]], %[[CONST_LOW]], %[[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    // CHECK:       %[[VAL1:.*]] = IE.ScaleShift(%arg0, %[[VAL0]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL1]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftReversedInputs
func.func @ConvertAddToScaleShiftReversedInputs(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<2.0> : tensor<1x3x1x1xf16>
    %0 = IE.Add(%bias, %arg0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<2.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @cannotConvertAddToScaleShift
func.func @cannotConvertAddToScaleShift(%arg0: tensor<1x3x1x1xf16>, %arg1: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %0 = IE.Add(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShift
func.func @ConvertMultiplyToScaleShift(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<3.0> : tensor<1x3x1x1xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x3x1x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftBroadcastChannels
func.func @ConvertMultiplyToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftReversedInputs
func.func @ConvertMultiplyToScaleShiftReversedInputs(%arg0: tensor<1x3x1x1xf16>, %arg1: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x1x1xf16>, tensor<1x3x300x300xf16> -> tensor<1x3x300x300xf16>

    return %0 : tensor<1x3x300x300xf16>

    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg1, %arg0) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyWithConstFQToScaleShiftBroadcastChannels
func.func @ConvertMultiplyWithConstFQToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf16>) -> tensor<1x3x300x300xf16> {
    %weights = const.Declare tensor<1x1x1x1xf16> = dense<3.0> : tensor<1x1x1x1xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    %1 = IE.Reshape(%0) { shape_value = [1, 1, 1] } : tensor<1x1x1x1xf16> -> tensor<1x1x1xf16>
    %2 = IE.Reshape(%1) { shape_value = [1, 1, 1, 1] } : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>

    %3 = IE.Multiply(%arg0, %2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>

    return %3 : tensor<1x3x300x300xf16>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK-DAG:       %[[CONST_HIGH:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       %[[CONST_LOW:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.FakeQuantize(%[[WEIGHTS]], %[[CONST_LOW]], %[[CONST_HIGH]], %[[CONST_LOW]], %[[CONST_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    // CHECK:       %[[VAL1:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 1]} : tensor<1x3x1x1xf16> -> tensor<1x1x1xf16>
    // CHECK:       %[[VAL2:.*]] = IE.Reshape(%[[VAL1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       %[[VAL3:.*]] = IE.ScaleShift(%arg0, %[[VAL2]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x3x300x300xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x300x300xf16>
    // CHECK:       return %[[VAL3]]
}

// -----

// CHECK-LABEL: @NoConvertAddToScaleShift
func.func @NoConvertAddToScaleShift(%arg0: tensor<1x3x300x300xsi32>) -> tensor<1x3x300x300xsi32> {
    %bias = const.Declare tensor<1x3x1x1xsi32> = dense<2> : tensor<1x3x1x1xsi32>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>

    return %0 : tensor<1x3x300x300xsi32>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xsi32> = dense<2> : tensor<1x3x1x1xsi32>
    // CHECK:       %[[VAL0:.*]] = IE.Add(%arg0, %[[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @NoConvertAddToScaleShiftF32
func.func @NoConvertAddToScaleShiftF32(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK-DAG:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Add(%arg0, %[[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}


// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShift
func.func @NoConvertMultiplyToScaleShift(%arg0: tensor<1x3x300x300xsi32>) -> tensor<1x3x300x300xsi32> {
    %weights = const.Declare tensor<1x3x1x1xsi32> = dense<3> : tensor<1x3x1x1xsi32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>

    return %0 : tensor<1x3x300x300xsi32>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xsi32> = dense<3> : tensor<1x3x1x1xsi32>
    // CHECK:       %[[VAL0:.*]] = IE.Multiply(%arg0, %[[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xsi32>, tensor<1x3x1x1xsi32> -> tensor<1x3x300x300xsi32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShiftF32
func.func @NoConvertMultiplyToScaleShiftF32(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Multiply(%arg0, %[[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @NoConvertMultiplyToScaleShiftWithInconsistentActShape
func.func @NoConvertMultiplyToScaleShiftWithInconsistentActShape(%arg0: tensor<1x256x1x1xf16>) -> tensor<1x256x1x768xf16> {
    %weights = const.Declare tensor<1x1x1x768xf16> = dense<3.0> : tensor<1x1x1x768xf16>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x256x1x1xf16>, tensor<1x1x1x768xf16> -> tensor<1x256x1x768xf16>

    return %0 : tensor<1x256x1x768xf16>

    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<1x1x1x768xf16> = dense<3.000000e+00> : tensor<1x1x1x768xf16>
    // CHECK:       %[[VAL0:.*]] = IE.Multiply(%arg0, %[[WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x1x1x768xf16> -> tensor<1x256x1x768xf16>
    // CHECK:       return %[[VAL0]]
}
