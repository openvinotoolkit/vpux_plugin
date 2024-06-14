//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-divide-to-multiply --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

!qElemType = !quant.uniform<u8:f16, 0.01013327205882353>

// CHECK-LABEL: @DoNotConvertNonConstDivide
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @DoNotConvertNonConstDivide(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x1x1x1x!qElemType> = dense<2> : tensor<1x1x1x1xui8>, [#const.QuantCast<!qElemType>]
    %nonCst = IE.Dequantize(%divisor) {dstElemType = f16} : tensor<1x1x1x1x!qElemType> -> tensor<1x1x1x1xf16>
    %0 = IE.Divide(%arg, %nonCst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1x!qElemType>
    // CHECK: [[NON_CONST:%.+]] = IE.Dequantize([[CONST]])
    // CHECK: [[RES:%.+]] = IE.Divide([[ARG]], [[NON_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @DoNotConvertIntegerDivide
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xsi32>) -> tensor<1x12x512x512xsi32>
func.func @DoNotConvertIntegerDivide(%arg: tensor<1x12x512x512xsi32>) -> tensor<1x12x512x512xsi32> {
    %divisor = const.Declare tensor<1x1x1x1xsi32> = dense<2> : tensor<1x1x1x1xsi32>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xsi32>, tensor<1x1x1x1xsi32> -> tensor<1x12x512x512xsi32>
    return %0 : tensor<1x12x512x512xsi32>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xsi32>
    // CHECK: [[RES:%.+]] = IE.Divide([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @DoNotConvertConstDividend
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @DoNotConvertConstDividend(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Divide(%cst, %arg) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x1x1xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[RES:%.+]] = IE.Divide([[CONST]], [[ARG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @ConvertConstDivisor
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @ConvertConstDivisor(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x12x512x512xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x1x1x1xf16> {{.*}} [#const.ScalarMultInverse]
    // CHECK: [[RES:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @ConvertConstDivisor_NonScalar
// CHECK-SAME: ([[ARG:%.+]]: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16>
func.func @ConvertConstDivisor_NonScalar(%arg: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16> {
    %divisor = const.Declare tensor<1x12x512x512xf16> = dense<2.0> : tensor<1x12x512x512xf16>
    %0 = IE.Divide(%arg, %divisor) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x12x512x512xf16>, tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16>
    return %0 : tensor<1x12x512x512xf16>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<1x12x512x512xf16> {{.*}} [#const.ScalarMultInverse]
    // CHECK: [[RES:%.+]] = IE.Multiply([[ARG]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK: return [[RES]]
}
