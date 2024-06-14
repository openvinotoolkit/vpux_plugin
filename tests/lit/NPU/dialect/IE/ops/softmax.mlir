//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @LegalizeAxisInd
func.func @LegalizeAxisInd(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %softmax = IE.SoftMax(%arg0) {axisInd = -1} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %softmax : tensor<1x8x4x4xf32>

    // CHECK:       %[[VAL0:.*]] = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    // CHECK-NOT:   IE.SoftMax
    // CHECK:       return %[[VAL0]] : tensor<1x8x4x4xf32>
}
