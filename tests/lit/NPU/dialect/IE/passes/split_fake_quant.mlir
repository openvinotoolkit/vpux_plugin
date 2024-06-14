//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK: !qElemType = !quant.uniform<u8:f32, 1.000000e+00>
// CHECK-LABEL: @SingleQuantParams
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @SingleQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<f8E5M2:f32, 1.000000e+00>
// CHECK-LABEL: @SingleQuantParamsF8E5M2
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @SingleQuantParamsF8E5M2(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-57344.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<57344.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-57344.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<57344.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK:       !qElemType = !quant.uniform<f8E4M3FN:f32, 5.000000e-01>
// CHECK-LABEL: @SingleQuantParamsF8E4M3FN
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @SingleQuantParamsF8E4M3FN(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-224.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<224.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-224.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<224.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @DoNotSingleQuantParamsF8E4M3FNNonZeroZeroPoint
func.func @DoNotSingleQuantParamsF8E4M3FNNonZeroZeroPoint(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<896.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<896.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       {dstElemType = !qElemType}
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.078431372549019607:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @DifferentQuantParams
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @DifferentQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.039215686274509803>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @OneDifferentQuantParam
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @OneDifferentQuantParam(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.011764705882352941:85>

// CHECK-LABEL: @BroadcastQuantParam
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x2x30x30xf32>)
func.func @BroadcastQuantParam(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x2x30x30xf32> ->
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x2x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.011764705882352941:85>

// CHECK-LABEL: @BroadcastQuantParamDiffRanks
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x2x30x30xf32>)
func.func @BroadcastQuantParamDiffRanks(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1xf32> = dense<-1.0> : tensor<1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x2x30x30xf32> ->
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x2x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f32, 1.000000e+00:-128>

// CHECK-LABEL: @UseDequantize
func.func @UseDequantize() -> tensor<1x3x30x30xf32> {
    %input = const.Declare tensor<1x3x30x30xf32> =
        dense<10> : tensor<1x3x30x30xui8>, [#const.ConvertElemType<f32>]

    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1x3x30x30x!qElemType> =
    // CHECK-SAME:      dense<10> : tensor<1x3x30x30xui8
    // CHECK-SAME:      #const.ConvertElemType<f32>
    // CHECK-SAME:      #const.ConvertElemType<si8>
    // CHECK-SAME:      #const.QuantCast<!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>
    // CHECK-SAME:      -> tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @UseRescale
func.func @UseRescale() -> tensor<1x2x30x30xf32> {
    %input = const.Declare tensor<1x2x30x30xf32> = dense<1.0> : tensor<1x2x30x30xf32>
    %input_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-2.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[1.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-0.5]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[1.0]],[[0.5]]]]> : tensor<1x2x1x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x30x30xf32>

    // CHECK-DAG:       [[VAL1:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x30x30xf32>, [#const.Rescale<2.000000e+00 : f64>]

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8<0:1>:f32, 1.000000e+00>

// CHECK-LABEL: @Level2Quantization
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @Level2Quantization(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 2 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @Scalar0AsInputRange
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @Scalar0AsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 7.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @Scalar7AsInputRange
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @Scalar7AsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<7.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<7.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>


    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 4.000000e+00:2>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NegativeScalarAsInputRange
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @NegativeScalarAsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-4.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<-4.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>


    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK-LABEL: @PerChannelQuant
func.func @PerChannelQuant(%arg0: tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32> {
    %input_low = const.Declare tensor<1x16x1x1xf32> =
        dense<[[[[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[-3.750000e-01]],
        [[0.000000e+00]], [[-0.297849566]], [[-0.382785916]], [[-0.399385154]], [[-3.750000e-01]],
        [[0.000000e+00]], [[-0.381789744]], [[0.000000e+00]], [[-3.750000e-01]], [[-3.750000e-01]], [[-0.389199734]]]]> : tensor<1x16x1x1xf32>
    %input_high = const.Declare tensor<1x16x1x1xf32> = dense<[[[[2.88798332]], [[17.5819988]], [[23.3847122]],
        [[6.1077733]], [[1.875000e+01]], [[3.81057024]], [[0.192161009]], [[13.5615578]], [[12.3310165]],
        [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]], [[2.70967746]], [[8.63315773]]]]> : tensor<1x16x1x1xf32>
    %output_low = const.Declare tensor<1x16x1x1xf32> = dense<[[[[0.000000e+00]], [[0.000000e+00]],
        [[0.000000e+00]], [[0.000000e+00]], [[-3.750000e-01]], [[0.000000e+00]], [[-0.297849566]],
        [[-0.382785916]], [[-0.399385154]], [[-3.750000e-01]], [[0.000000e+00]], [[-0.381789744]],
        [[0.000000e+00]], [[-3.750000e-01]], [[-3.750000e-01]], [[-0.389199734]]]]> : tensor<1x16x1x1xf32>
    %output_high =  const.Declare tensor<1x16x1x1xf32> = dense<[[[[2.88798332]], [[17.5819988]],
        [[23.3847122]], [[6.1077733]], [[1.875000e+01]], [[3.81057024]], [[0.192161009]], [[13.5615578]],
        [[12.3310165]], [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]],
        [[2.70967746]], [[8.63315773]]]]> : tensor<1x16x1x1xf32>

    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x112x112xf32>

    return %fq : tensor<1x16x112x112xf32>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----

// CHECK-LABEL: @PerChannelQuantInput
func.func @PerChannelQuantInput(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299xf16> {
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<2.63867188> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.65820313> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[2.551250e+02]], [[2.670000e+02]], [[2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_low =  const.Declare tensor<1x3x1x1xf16> = dense<[[[[-49.28125]], [[-35.65625]], [[-31.828125]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x299x299xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x299x299xf16>

    return %1 : tensor<1x3x299x299xf16>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16:1, {3.906250e-03,0.001953125,7.812500e-03,9.765625E-4}>
// CHECK: !qElemType1 = !quant.uniform<f8E5M2:f16, 0.001953125>

// CHECK-LABEL: @PerChannelQuantInputF8E4M3FN
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x4x15x15xf16>)
func.func @PerChannelQuantInputF8E4M3FN(%arg0: tensor<1x4x15x15xf16>) -> tensor<1x4x15x15xf16> {
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-112.0> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<112.0> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_low = const.Declare tensor<1x4x1x1xf16> = dense<[[[[-224.0]], [[-112.0]], [[-448.0]], [[-56.0]]]]> : tensor<1x4x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high =  const.Declare tensor<1x4x1x1xf16> = dense<[[[[224.0]], [[112.0]], [[448.0]], [[56.0]]]]> : tensor<1x4x1x1xf32>, [#const.ConvertElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2 } : tensor<1x4x15x15xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x15x15xf16>

    return %1 : tensor<1x4x15x15xf16>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]]) 
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x4x15x15xf16> ->
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType> ->
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f16}
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType1> ->
    // CHECK-SAME:      tensor<1x4x15x15xf16>

    // CHECK:       return [[VAL2]] : tensor<1x4x15x15xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16:1, {1.000000e+00,2.000000e+00,8.000000e+00,5.000000e-01}>
// CHECK: !qElemType1 = !quant.uniform<f8E4M3FN:f16, 8.000000e+00>

// CHECK-LABEL: @PerChannelQuantInputF8E5M2
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x4x15x15xf16>)
func.func @PerChannelQuantInputF8E5M2(%arg0: tensor<1x4x15x15xf16>) -> tensor<1x4x15x15xf16> {
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-3584.0> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<3584.0> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_low = const.Declare tensor<1x4x1x1xf16> = dense<[[[[-448.0]], [[-896.0]], [[-3584.0]], [[-224.0]]]]> : tensor<1x4x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high =  const.Declare tensor<1x4x1x1xf16> = dense<[[[[448.0]], [[896.0]], [[3584.0]], [[224.0]]]]> : tensor<1x4x1x1xf32>, [#const.ConvertElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } : tensor<1x4x15x15xf16>, tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x15x15xf16>

    return %1 : tensor<1x4x15x15xf16>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x4x15x15xf16>
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType> ->
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f16}
    // CHECK-SAME:      tensor<1x4x15x15x!qElemType1> ->
    // CHECK-SAME:      tensor<1x4x15x15xf16>

    // CHECK:       return [[VAL2]] : tensor<1x4x15x15xf16>
}

// -----

// CHECK-LABEL: @DoNotPerChannelQuantInputF8E5M2NonZeroZeroPoint
func.func @DoNotPerChannelQuantInputF8E5M2NonZeroZeroPoint(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299xf16> {
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<2.63867188> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.65820313> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[-2.551250e+02]], [[2.670000e+02]], [[2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_low =  const.Declare tensor<1x3x1x1xf16> = dense<[[[[2.551250e+02]], [[-2.670000e+02]], [[-2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2 } : tensor<1x3x299x299xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x299x299xf16>

    return %1 : tensor<1x3x299x299xf16>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----

// CHECK-LABEL: @DoNotPerChannelQuantInputF8E4M3FNNonZeroZeroPoint
func.func @DoNotPerChannelQuantInputF8E4M3FNNonZeroZeroPoint(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299xf16> {
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<2.63867188> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.65820313> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[-2.551250e+02]], [[2.670000e+02]], [[2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_low =  const.Declare tensor<1x3x1x1xf16> = dense<[[[[2.551250e+02]], [[-2.670000e+02]], [[-2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } : tensor<1x3x299x299xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x299x299xf16>

    return %1 : tensor<1x3x299x299xf16>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----
// CHECK: !qElemType = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>
// CHECK-LABEL: @ConstantsSplitFakeQuant
func.func @ConstantsSplitFakeQuant(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %1 : tensor<1x16x16x16xf16>
    // CHECK-NOT:       IE.FakeQuantize
    // CHECK:       [[Q_CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.270000e+02> : tensor<16x16x1x1xf16>,
    // CHECK-SAME:      [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
    // CHECK:       [[DQ_CST:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16}
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[ARG0:%.*]], [[DQ_CST]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<f8E5M2:f16, 1.743861607142857E-5>
// CHECK-LABEL: @ConstantsSplitFakeQuantF8E5M2
func.func @ConstantsSplitFakeQuantF8E5M2(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2 } : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x1x1xf16> 
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %1 : tensor<1x16x16x16xf16>
    // CHECK-NOT:       IE.FakeQuantize
    // CHECK:       [[Q_CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x16x1x1xf16>,
    // CHECK-SAME:      [#const.ConvertElemType<f8E5M2>, #const.QuantCast<!qElemType>]
    // CHECK:       [[DQ_CST:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16}
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[ARG0:%.*]], [[DQ_CST]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<f8E4M3FN:f16, 0.002232142857142857>
// CHECK-LABEL: @ConstantsSplitFakeQuantF8E4M3FN
func.func @ConstantsSplitFakeQuantF8E4M3FN(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x1x1xf16> 
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %1 : tensor<1x16x16x16xf16>
    // CHECK-NOT:       IE.FakeQuantize
    // CHECK:       [[Q_CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x16x1x1xf16>,
    // CHECK-SAME:      [#const.ConvertElemType<f8E4M3FN>, #const.QuantCast<!qElemType>]
    // CHECK:       [[DQ_CST:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16}
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[ARG0:%.*]], [[DQ_CST]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<i4:f16, 1.3385416666666667>
// CHECK-LABEL: @I4ConstantsSplitFakeQuant
func.func @I4ConstantsSplitFakeQuant(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<16x1x1x1xf16> = dense<-1.007810e+01> : tensor<16x1x1x1xf16>
    %cst_3 = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+01> : tensor<16x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %1 : tensor<1x16x16x16xf16>
    // CHECK-NOT:       IE.FakeQuantize
    // CHECK:       [[Q_CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<0.000000e+00> : tensor<16x16x1x1xf16>,
    // CHECK-SAME:      [#const.ConvertElemType<si4>, #const.QuantCast<!qElemType>]
    // CHECK:       [[DQ_CST:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16}
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[ARG0:%.*]], [[DQ_CST]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<i8:f16, 0.39215686274509803:127>
// CHECK-LABEL: @BringConstIn8BitRepresentableFormat
func.func @BringConstIn8BitRepresentableFormat() -> tensor<1x3x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+02> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x3x1x1xf16> = dense<[[[-6.90000e+01]], [[-2.500000e+02]], [[-2.300000e+02]]]> : tensor<3x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 3, 1, 1]>]
    %0 = IE.FakeQuantize(%cst_2, %cst_1, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    return %0 : tensor<1x3x1x1xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[Q_CST:%.*]] = const.Declare tensor<1x3x1x1x!qElemType> =
    // CHECK-SAME{LITERAL}:      dense<[[[[5.800000e+01]], [[-1.230000e+02]], [[-1.030000e+02]]]]> : tensor<1x3x1x1xf16>,
    // CHECK-SAME:      [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
    // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    //CHECK: return [[DEQUANTIZE]] : tensor<1x3x1x1xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<i8:f16, 0.0028435202205882352:127>
// CHECK-LABEL: @BringConstIn8BitRepresentableFormatWithBroadcast
func.func @BringConstIn8BitRepresentableFormatWithBroadcast() -> tensor<1x12x1x1xf16> {
    %cst_0 = const.Declare tensor<1x12x1x1xf16> = dense<2.300000e+02> : tensor<1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 1]>, #const.Rescale<-1.000000e+00 : f64>, #const.Broadcast<1 : i64, 12 : i64>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.252680e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst_0, %cst_3, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x1x1xf16>
    return %0 : tensor<1x12x1x1xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[Q_CST:%.*]] = const.Declare tensor<1x12x1x1x!qElemType> =
    // CHECK-SAME:      dense<-1.030000e+02> : tensor<1x12x1x1xf16>, 
    // CHECK-SAME:      [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>]
    // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16} : tensor<1x12x1x1x!qElemType> -> tensor<1x12x1x1xf16>
    //CHECK: return [[DEQUANTIZE]] : tensor<1x12x1x1xf16>
}
