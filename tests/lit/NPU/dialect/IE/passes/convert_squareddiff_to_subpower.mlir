//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-squareddiff-to-subpower %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertSquaredDiffToSubPower
func.func @ConvertSquaredDiffToSubPower(%arg0: tensor<1x64x128x64xf16>, %arg1: tensor<1x1x1x64xf16>) -> tensor<1x64x128x64xf16> {
    %0 = IE.SquaredDiff(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    return %0 : tensor<1x64x128x64xf16>


    // CHECK-NOT: IE.SquaredDiff
    // CHECK: [[SUBTRACT:%.*]] = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[POWER:%.*]] = IE.Power([[SUBTRACT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x128x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x128x64xf16>
    // CHECK: return [[POWER]] : tensor<1x64x128x64xf16>
}
