//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-i4 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<i4<-8:7>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>
!qElemType1 = !quant.uniform<u4<0:15>:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
!qElemType2 = !quant.uniform<i4:f16, 1.1534313725490195>
!qElemType3 = !quant.uniform<i4:f16, 2.4627450980392158>

// CHECK-LABEL: @ConvertU4WeightsToI4
func.func @ConvertU4WeightsToI4(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType1> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>]
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    %2 = IE.Convolution(%1, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType1> -> tensor<1x3x14x14x!qElemType3>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>

    return %3 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType1>,
    // CHECK-SAME:      #const.ConvertElemType<!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    // CHECK:       [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType3>
    // CHECK:       [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<u4:f16, 1.1534313725490195>
// We don't convert u4 to i4 because of the bad zero point value of U4.
// CHECK-LABEL: @NotConvertU4Weights
func.func @NotConvertU4Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @KeepI4Weights
func.func @KeepI4Weights(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    return %2 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<si4>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType>
    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u4<0:15>:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
// CHECK: !qElemType = !quant.uniform<i4:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK-LABEL: @ConvertFromPerAxisTypeU4ToI4
func.func @ConvertFromPerAxisTypeU4ToI4(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:       [#const.CastElemType<ui4>, #const.CastElemType<!qElemType1>,
    // CHECK-SAME:        #const.ConvertElemType<!qElemType>]
    // CHECK:       [[VAL0:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u4<0:15>:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:7}>
// CHECK: !qElemType = !quant.uniform<u4:f16:0, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:7}>

// CHECK-LABEL: @DontConvertFromPerAxisTypeU4ToI4NotAllZeroPointAre8
func.func @DontConvertFromPerAxisTypeU4ToI4NotAllZeroPointAre8(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<3.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:       dense<3.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>]
    // CHECK:       [[VAL0:%.*]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u4:f16:1, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
!qElemType1 = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<i4:f16:1, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK: @ConvertQuantizeCastPerAxisU4ToI4AllZeroPoints8
func.func @ConvertQuantizeCastPerAxisU4ToI4AllZeroPoints8(%arg0: tensor<16x3x1x1x!qElemType1>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1x!qElemType1> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[VAL0:%.*]] = IE.QuantizeCast([[ARG0:%.*]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1x[[Q_ELEM_TYPE0]]> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

!qElemType = !quant.uniform<u4:f16:1, {0.010680671751968504:8,0.0081200787401574797:8,0.010596087598425197:8}>
// CHECK: [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<i4:f16:1, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

// CHECK: @ConvertQuantizePerAxisU4ToI4AllZeroPoints8
func.func @ConvertQuantizePerAxisU4ToI4AllZeroPoints8(%arg0: tensor<16x3x1x1xf16>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[ARG0:%.*]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

!qElemType = !quant.uniform<u4:f16, 0.010680671751968504:8>
!qElemType1 = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<i4:f16, 0.010680671751968504>

// CHECK: @ConvertQuantizeCastUniformQuantU4ToI4
func.func @ConvertQuantizeCastUniformQuantU4ToI4(%arg0: tensor<16x3x1x1x!qElemType1>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1x!qElemType1> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[VAL0:%.*]] = IE.QuantizeCast([[ARG0:%.*]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1x[[Q_ELEM_TYPE0]]> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}

// -----

!qElemType = !quant.uniform<u4:f16, 0.0081200787401574797:8>
// CHECK: [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<i4:f16, 0.0081200787401574797>

// CHECK: @ConvertQuantizeUniformQuantU4ToI4
func.func @ConvertQuantizeUniformQuantU4ToI4(%arg0: tensor<16x3x1x1xf16>) -> tensor<16x3x1x1x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x!qElemType>
    return %0 : tensor<16x3x1x1x!qElemType>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[ARG0:%.*]]) {dstElemType = [[Q_ELEM_TYPE1]]}
    // CHECK-SAME:      : tensor<16x3x1x1xf16> -> tensor<16x3x1x1x[[Q_ELEM_TYPE1]]>
}
