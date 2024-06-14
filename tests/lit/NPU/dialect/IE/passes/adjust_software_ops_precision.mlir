//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-software-ops-precision --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @TopK_SI32toFP16
func.func @TopK_SI32toFP16(%arg0: tensor<1x77xsi32>) -> (tensor<1x1xsi32>, tensor<1x1xsi32>) {
    %cst_K = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x77xsi32>, tensor<si64> -> tensor<1x1xsi32>, tensor<1x1xsi32>
    return %output_values, %target_shape : tensor<1x1xsi32>, tensor<1x1xsi32>

    // CHECK: [[INPUT_CVT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x77xsi32> -> tensor<1x77xf16>
    // CHECK: [[VALUES:%.*]], [[SHAPE:%.*]] = IE.TopK([[INPUT_CVT]]
    // CHECK: [[OUT_CVT:%.*]] = IE.Convert([[VALUES]]) {dstElemType = si32} : tensor<1x1xf16> -> tensor<1x1xsi32>
    // CHECK: return [[OUT_CVT]], [[SHAPE]] : tensor<1x1xsi32>, tensor<1x1xsi32>
}

// -----

// CHECK-LABEL: @TopK_SI64toFP16
func.func @TopK_SI64toFP16(%arg0: tensor<1x77xsi64>) -> (tensor<1x1xsi64>, tensor<1x1xsi32>) {
    %cst_K = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>} : tensor<1x77xsi64>, tensor<si64> -> tensor<1x1xsi64>, tensor<1x1xsi32>
    return %output_values, %target_shape : tensor<1x1xsi64>, tensor<1x1xsi32>

    // CHECK: [[INPUT_CVT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x77xsi64> -> tensor<1x77xf16>
    // CHECK: [[VALUES:%.*]], [[SHAPE:%.*]] = IE.TopK([[INPUT_CVT]]
    // CHECK: [[OUT_CVT:%.*]] = IE.Convert([[VALUES]]) {dstElemType = si64} : tensor<1x1xf16> -> tensor<1x1xsi64>
    // CHECK: return [[OUT_CVT]], [[SHAPE]] : tensor<1x1xsi64>, tensor<1x1xsi32>
}
