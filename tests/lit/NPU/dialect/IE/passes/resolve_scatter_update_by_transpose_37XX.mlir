//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-scatter-update-by-transpose  %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>

// CHECK-LABEL: @ResolveScatterUpdateByTranspose
func.func @ResolveScatterUpdateByTranspose (%arg0: tensor<10x16x12x15xf16>, %arg1: tensor<8xsi32>, %arg2: tensor<10x16x8x15xf16>)  -> tensor<10x16x12x15xf16> {
    %0 = IE.ScatterUpdate(%arg0, %arg1, %arg2) {axis_value = 2 : i64} : tensor<10x16x12x15xf16>, tensor<8xsi32>, tensor<10x16x8x15xf16> -> tensor<10x16x12x15xf16>
    return %0 : tensor<10x16x12x15xf16>

    // CHECK:  [[TRANSPOSE_1:%.*]] = IE.Transpose(%arg2) {order_value = #map} : tensor<10x16x8x15xf16> -> tensor<8x10x16x15xf16>
    // CHECK:  [[TRANSPOSE_2:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<10x16x12x15xf16> -> tensor<12x10x16x15xf16>
    // CHECK:  [[VAL1:%.*]] = IE.ScatterUpdate([[TRANSPOSE_2]], %arg1, [[TRANSPOSE_1]]) {axis_value = 0 : i64} : tensor<12x10x16x15xf16>, tensor<8xsi32>, tensor<8x10x16x15xf16> -> tensor<12x10x16x15xf16>
    // CHECK:  [[TRANSPOSE_3:%.*]] = IE.Transpose([[VAL1]]) {order_value = #map1} : tensor<12x10x16x15xf16> -> tensor<10x16x12x15xf16>
    // CHECK:  return [[TRANSPOSE_3]] : tensor<10x16x12x15xf16>
}
