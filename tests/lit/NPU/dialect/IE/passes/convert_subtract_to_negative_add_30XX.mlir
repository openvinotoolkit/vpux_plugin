//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-subtract-to-add %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @ConvertSubtractToAddWithNegative
func.func @ConvertSubtractToAddWithNegative(%arg0: tensor<1x16x32xf32>) -> tensor<1x16x32xf32> {
    %cst = const.Declare tensor<1x16x1xf32> = dense<2.0> : tensor<1x16x1xf32>
    %0 = IE.Subtract(%arg0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32xf32>, tensor<1x16x1xf32> -> tensor<1x16x32xf32>

    return %0 : tensor<1x16x32xf32>

    // CHECK-DAG:       %[[CST:.*]] = const.Declare tensor<1x16x1xf32> = dense<2.000000e+00> : tensor<1x16x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Negative(%[[CST]]) : tensor<1x16x1xf32> -> tensor<1x16x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.Add(%arg0, %[[VAL0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32xf32>, tensor<1x16x1xf32> -> tensor<1x16x32xf32>
    // CHECK:       return %[[VAL1]]
}
