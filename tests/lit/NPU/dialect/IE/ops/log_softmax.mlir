//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @ConstFold
func.func @ConstFold(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %prob = IE.LogSoftmax(%arg0) {axisInd = 0} : tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %prob : tensor<1x8x4x4xf32>

    // CHECK-DAG:       %[[VAL0:.*]] = const.Declare tensor<1x8x4x4xf32> = dense<0.000000e+00> : tensor<1x8x4x4xf32>
    // CHECK-NOT:   IE.LogSoftmax
    // CHECK:       return %[[VAL0]] : tensor<1x8x4x4xf32>
}
