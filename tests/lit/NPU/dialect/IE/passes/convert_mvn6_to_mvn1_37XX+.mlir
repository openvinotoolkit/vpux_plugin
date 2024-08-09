//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-mvn6-to-mvn1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case2D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<5x17xf16>)
func.func @ConvertMVN6ToMVN1Case2D(%arg0: tensor<5x17xf16>) -> tensor<5x17xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [1], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<5x17xf16> -> tensor<5x17xf16>
    return %0 : tensor<5x17xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 5, 17, 1]} : tensor<5x17xf16> -> tensor<1x5x17x1xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-5 : f64, normalize_variance = true} : tensor<1x5x17x1xf16> -> tensor<1x5x17x1xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [5, 17]} : tensor<1x5x17x1xf16> -> tensor<5x17xf16>
    // CHECK:       return [[OUTPUT]] : tensor<5x17xf16>
}

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case3D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x48x48xf16>)
func.func @ConvertMVN6ToMVN1Case3D(%arg0: tensor<1x48x48xf16>) -> tensor<1x48x48xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [1, 2], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<1x48x48xf16> -> tensor<1x48x48xf16>
    return %0 : tensor<1x48x48xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 48, 48]} : tensor<1x48x48xf16> -> tensor<1x1x48x48xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-5 : f64, normalize_variance = true} : tensor<1x1x48x48xf16> -> tensor<1x1x48x48xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [1, 48, 48]} : tensor<1x1x48x48xf16> -> tensor<1x48x48xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x48x48xf16>
}

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case4D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<4x10x5x17xf16>)
func.func @ConvertMVN6ToMVN1Case4D(%arg0: tensor<4x10x5x17xf16>) -> tensor<4x10x5x17xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [3], eps = 5.000000e-01 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<4x10x5x17xf16> -> tensor<4x10x5x17xf16>
    return %0 : tensor<4x10x5x17xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [4, 50, 1, 17]} : tensor<4x10x5x17xf16> -> tensor<4x50x1x17xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 5.000000e-01 : f64, normalize_variance = true} : tensor<4x50x1x17xf16> -> tensor<4x50x1x17xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [4, 10, 5, 17]} : tensor<4x50x1x17xf16> -> tensor<4x10x5x17xf16>
    // CHECK:       return [[OUTPUT]] : tensor<4x10x5x17xf16>
}

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1Case5D
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x32x20x20x20xf16>)
func.func @ConvertMVN6ToMVN1Case5D(%arg0: tensor<1x32x20x20x20xf16>) -> tensor<1x32x20x20x20xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3, 4], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<1x32x20x20x20xf16> -> tensor<1x32x20x20x20xf16>
    return %0 : tensor<1x32x20x20x20xf16>

    // CHECK:       [[INPUT_RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 32, 20, 400]} : tensor<1x32x20x20x20xf16> -> tensor<1x32x20x400xf16>
    // CHECK:       [[MVN:%.+]] = IE.MVN([[INPUT_RESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-5 : f64, normalize_variance = true} : tensor<1x32x20x400xf16> -> tensor<1x32x20x400xf16>
    // CHECK:       [[OUTPUT:%.+]] = IE.Reshape([[MVN]]) {shape_value = [1, 32, 20, 20, 20]} : tensor<1x32x20x400xf16> -> tensor<1x32x20x20x20xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x32x20x20x20xf16>
}

// CHECK-LABEL: func.func @ConvertMVN6ToMVN1NotApplied
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<4x10x5x17xf16>)
func.func @ConvertMVN6ToMVN1NotApplied(%arg0: tensor<4x10x5x17xf16>) -> tensor<4x10x5x17xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [0], eps = 5.000000e-01 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<4x10x5x17xf16> -> tensor<4x10x5x17xf16>
    return %0 : tensor<4x10x5x17xf16>

    // CHECK:       [[MVN:%.+]] = IE.MVN6([[INPUT]]) {axes_value = [0], eps = 5.000000e-01 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true} : tensor<4x10x5x17xf16> -> tensor<4x10x5x17xf16>
    // CHECK:       return [[MVN]] : tensor<4x10x5x17xf16>
}
