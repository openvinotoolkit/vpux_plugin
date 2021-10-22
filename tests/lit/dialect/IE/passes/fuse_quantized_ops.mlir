// RUN: vpux-opt --split-input-file --fuse-quantized-ops %s | FileCheck %s

// CHECK-LABEL: @FuseQuantParamsIntoConv
func @FuseQuantParamsIntoConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> = #const.Content<dense<1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>>]>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !quant.uniform<u8:f16, 2.4627450980392158>}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> =
  //CHECK-SAME:                 #const.Content<dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>, tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]]
}

// -----

// CHECK-LABEL: @FuseQuantParamsIntoEltwise
func @FuseQuantParamsIntoEltwise(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16xf16>
  %3 = IE.Quantize(%arg1) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16xf16>
  %5 = IE.Add(%2, %4) { auto_broadcast = "NUMPY" } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %6 = IE.Quantize(%5) {dstElemType = !quant.uniform<u8:f16, 2.4627450980392158>}: tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 2.4627450980392158>>
  %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 2.4627450980392158>> -> tensor<1x3x16x16xf16>

  return %7 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType0} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType0} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  //CHECK: [[VAL2:%.*]] = IE.Add([[VAL0]], [[VAL1]]) {auto_broadcast = "NUMPY"} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>, tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 2.4627450980392158>>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 2.4627450980392158>> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL3]]
}

// -----

!qElemType = type !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @FusePerChannelEltwiseNoChanges
func @FusePerChannelEltwiseNoChanges(%arg0: tensor<1x4x16x16x!qElemType>, %arg1: tensor<1x4x16x16x!qElemType>) -> tensor<1x4x16x16x!qElemType> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x4x16x16x!qElemType> -> tensor<1x4x16x16xf16>
    %2 = IE.Add(%0, %1) { auto_broadcast = "NUMPY" } : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType}: tensor<1x4x16x16xf16> -> tensor<1x4x16x16x!qElemType>

    return %3 : tensor<1x4x16x16x!qElemType>

    //CHECK:  %0 = IE.Dequantize(%arg0)
    //CHECK:  %1 = IE.Dequantize(%arg1)
    //CHECK:  %2 = IE.Add(%0, %1) {auto_broadcast = "NUMPY"} : tensor<1x4x16x16xf16>, tensor<1x4x16x16xf16> -> tensor<1x4x16x16xf16>
    //CHECK:  %3 = IE.Quantize(%2)
    //CHECK:  return %3
}

// -----

// CHECK-LABEL: @FuseQuantParamsIntoSlice
func @FuseQuantParamsIntoSlice(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x8xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16xf16>
    %2 = IE.Slice %1 [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    %3 = IE.Quantize(%2) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>}: tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x8xf16>

    return %4 : tensor<1x3x16x8xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL1:%.*]] = IE.Slice [[VAL0]] [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> to tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x8xf16>
    //CHECK: return [[VAL2]] : tensor<1x3x16x8xf16>
}

// -----

// CHECK-LABEL: @FuseQuantParamsIntoPool
func @FuseQuantParamsIntoPool(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 0.57450980392156858>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>> -> tensor<1x3x16x16xf16>
  %3 = IE.MaxPool(%2) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %4 = IE.Quantize(%3) {dstElemType = !quant.uniform<u8:f16, 0.57450980392156858>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>> -> tensor<1x3x16x16xf16>
  return %5 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  //CHECK: [[VAL1:%.*]] = IE.MaxPool([[VAL0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>> -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL2]]
}

// -----

// CHECK-LABEL: @FuseQuantParamsIntoConcat
func @FuseQuantParamsIntoConcat(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x2x3x4xf16>

    %2 = IE.Quantize(%arg1) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x2x3x4xf16>

    %4 = IE.Concat (%1, %3) {axis = 1} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %5 = IE.Quantize(%4) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>}: tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x4x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x4x3x4xf16>

    return %6 : tensor<1x4x3x4xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL2:%.*]] = IE.Concat([[VAL0]], [[VAL1]]) {axis = 1 : i64, offset = 0 : i64, stride = 1 : i64} : tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>, tensor<1x2x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x4x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x4x3x4x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x4x3x4xf16>
    //CHECK: return [[VAL3]] : tensor<1x4x3x4xf16>
}
