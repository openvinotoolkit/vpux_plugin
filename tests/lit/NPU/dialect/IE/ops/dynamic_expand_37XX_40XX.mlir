//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: func.func @SingleLayerDynamicExpandOp([[ARG0:%.+]]: tensor<1x3x?x?xf16, {bounds = [1, 3, 20, 20]}>) -> tensor<1x3x20x20xf16>
func.func @SingleLayerDynamicExpandOp(%arg0: tensor<1x3x?x?xf16, {bounds = [1, 3, 20, 20]}>) -> tensor<1x3x20x20xf16> {
    %0 = IE.DynamicExpand(%arg0) : tensor<1x3x?x?xf16, {bounds = [1, 3, 20, 20]}> -> tensor<1x3x20x20xf16>
    return %0 : tensor<1x3x20x20xf16>

    // CHECK:       [[DynamicExpand:%.+]] = IE.DynamicExpand([[ARG0]]) : tensor<1x3x?x?xf16, {bounds = [1, 3, 20, 20]}> -> tensor<1x3x20x20xf16>
    // CHECK:       return [[DynamicExpand]] : tensor<1x3x20x20xf16>
}

// -----

// CHECK: func.func @SingleLayerDynamicExpandOpStaticInput([[ARG0:%.+]]: tensor<1x3x20x20xf16>) -> tensor<1x3x20x20xf16>
func.func @SingleLayerDynamicExpandOpStaticInput(%arg0: tensor<1x3x20x20xf16>) -> tensor<1x3x20x20xf16> {
    %0 = IE.DynamicExpand(%arg0) : tensor<1x3x20x20xf16> -> tensor<1x3x20x20xf16>
    return %0 : tensor<1x3x20x20xf16>

    // CHECK-NOT:       IE.DynamicExpand
    // CHECK:       return [[ARG0]] : tensor<1x3x20x20xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: func.func @SingleLayerDynamicExpandOpQuantized([[ARG0:%.+]]: tensor<1x3x?x?x!qElemType, {bounds = [1, 3, 20, 20]}>) -> tensor<1x3x20x20x!qElemType>
func.func @SingleLayerDynamicExpandOpQuantized(%arg0: tensor<1x3x?x?x!qElemType, {bounds = [1, 3, 20, 20]}>) -> tensor<1x3x20x20x!qElemType> {
    %0 = IE.DynamicExpand(%arg0) : tensor<1x3x?x?x!qElemType, {bounds = [1, 3, 20, 20]}> -> tensor<1x3x20x20x!qElemType>
    return %0 : tensor<1x3x20x20x!qElemType>

    // CHECK:       [[DynamicExpand:%.+]] = IE.DynamicExpand([[ARG0]]) : tensor<1x3x?x?x!qElemType, {bounds = [1, 3, 20, 20]}> -> tensor<1x3x20x20x!qElemType>
    // CHECK:       return [[DynamicExpand]] : tensor<1x3x20x20x!qElemType>
}
