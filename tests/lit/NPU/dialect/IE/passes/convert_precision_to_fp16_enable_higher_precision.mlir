//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-fp16="compute-layers-with-higher-precision=SoftMax,ReLU" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @NotConvertSoftMaxToFP16
module @NotConvertSoftMaxToFP16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf32>
        DataInfo "data" : tensor<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK-NEXT: %[[VAL0:.*]] = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1000xf16> -> tensor<1x1000xf32>
    // CHECK-NEXT: %[[VAL1:.*]] = IE.SoftMax(%[[VAL0]]) {axisInd = 1 : i64} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK-NEXT: %[[OUT:.*]] = IE.Convert(%[[VAL1]]) {dstElemType = f16} : tensor<1x1000xf32> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf32>
    // CHECK-NEXT: return %[[OUT]] : tensor<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @NotConvertReLUToFP16
module @NotConvertReLUToFP16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x8x128x128xf32>
        DataInfo "data" : tensor<1x8x128x128xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x8x128x128xf32>
        DataInfo "prob" : tensor<1x8x128x128xf32>
    }

// CHECK: func.func @main(%arg0: tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16> {
func.func @main(%arg0: tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32> {
    %cst_0 = const.Declare tensor<1x8x1x1xf32> = dense<1.0> : tensor<1x8x1x1xf32>
    %0 = IE.Add(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x128x128xf32>
    // CHECK: %[[VAL0:.*]] = IE.Add(%arg0, %[[CST0:.*]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x128x128xf16>
    %1 = IE.ReLU(%0) : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
    // CHECK: %[[VAL1:.*]] = IE.Convert(%[[VAL0]]) {dstElemType = f32} : tensor<1x8x128x128xf16> -> tensor<1x8x128x128xf32>
    // CHECK: %[[VAL2:.*]] = IE.ReLU(%[[VAL1]]) : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
    // CHECK: %[[OUT:.*]] = IE.Convert(%[[VAL2]]) {dstElemType = f16} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf16>
    return %1 : tensor<1x8x128x128xf32>
    // CHECK-NEXT: return %[[OUT]] : tensor<1x8x128x128xf16>
}

}
