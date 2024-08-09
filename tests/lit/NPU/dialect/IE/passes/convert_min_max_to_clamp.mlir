//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-min-max-to-clamp %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @EltwiseMinToClamp
func.func @EltwiseMinToClamp(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Minimum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[OUT:%.+]] = IE.Clamp(%arg0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// CHECK-LABEL: @EltwiseMaxToClamp
func.func @EltwiseMaxToClamp(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Maximum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[OUT:%.+]] = IE.Clamp(%arg0) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// CHECK-LABEL: @MaximumNotScalarInput
func.func @MaximumNotScalarInput(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    %0 = IE.Maximum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    // CHECK:       [[OUT:%.+]] = IE.Maximum([[CST]], %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// CHECK-LABEL: @MinimumNotScalarInput
func.func @MinimumNotScalarInput(%arg0: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
    %cst = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    %0 = IE.Minimum(%cst, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

    return %0 : tensor<1x64x28x28xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x64x28x28xf16> = dense<0.000000e+00> : tensor<1x64x28x28xf16>
    // CHECK:       [[OUT:%.+]] = IE.Minimum([[CST]], %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16>
    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16>
}

// CHECK-LABEL: @MinimumWithSecondInputScalar
func.func @MinimumWithSecondInputScalar(%arg0: tensor<1x1x1x3xf16>)
        -> tensor<1x1x1x3xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Minimum(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x3xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x1x1x3xf16>

    return %0 : tensor<1x1x1x3xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[OUT:%.+]] = IE.Clamp(%arg0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1x1x1x3xf16> -> tensor<1x1x1x3xf16>
    // CHECK:       return [[OUT]] : tensor<1x1x1x3xf16>
}
