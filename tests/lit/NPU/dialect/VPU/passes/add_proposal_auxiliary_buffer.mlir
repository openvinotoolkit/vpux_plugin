//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --add-proposal-auxiliary-buffer %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: func.func @Proposal
// CHECK-SAME:        [[ARG0:%arg[0-9]]]: tensor<1x2x4x4xf16>
// CHECK-SAME:        [[ARG1:%arg[0-9]]]: tensor<1x4x4x4xf16>
func.func @Proposal(%arg0: tensor<1x2x4x4xf16>, %arg1: tensor<1x4x4x4xf16>) -> (tensor<300x5xf16>, tensor<300xf16>) {
    %cst = const.Declare tensor<3xf16> = dense<[2.250000e+02, 2.250000e+02, 1.000000e+00]> : tensor<3xf16>
    %output, %probs = VPU.Proposal(%arg0, %arg1, %cst) {proposal_attrs = #IE.Proposal<baseSize = 4 : i64, preNmsTopN = 6000 : i64, postNmsTopN = 300 : i64, nmsThresh = 0.69999998807907104 : f64, featStride = 1 : i64, minSize = 4 : i64, ratio = [5.000000e-01], scale = [1.2000000476837158], clipBeforeNms = true, clipAfterNms = false, normalize = true, boxSizeScale = 2.000000e+00 : f64, boxCoordinateScale = 2.000000e+00 : f64, framework = "", inferProbs = true>} : tensor<1x2x4x4xf16>, tensor<1x4x4x4xf16>, tensor<3xf16> -> tensor<300x5xf16>, tensor<300xf16>
    return %output, %probs : tensor<300x5xf16>, tensor<300xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3xf16> = dense<[2.250000e+02, 2.250000e+02, 1.000000e+00]> : tensor<3xf16>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<182xui8> = dense<0> : tensor<182xui8>
    // CHECK:       [[OUTPUT:%.+]], [[PROBS:%.+]] = VPU.Proposal([[ARG0]], [[ARG1]], [[CST]], [[CST_0]]) {proposal_attrs = #IE.Proposal<baseSize = 4 : i64, preNmsTopN = 6000 : i64, postNmsTopN = 300 : i64, nmsThresh = 0.69999998807907104 : f64, featStride = 1 : i64, minSize = 4 : i64, ratio = [5.000000e-01], scale = [1.2000000476837158], clipBeforeNms = true, clipAfterNms = false, normalize = true, boxSizeScale = 2.000000e+00 : f64, boxCoordinateScale = 2.000000e+00 : f64, framework = "", inferProbs = true>} : tensor<1x2x4x4xf16>, tensor<1x4x4x4xf16>, tensor<3xf16>, tensor<182xui8> -> tensor<300x5xf16>, tensor<300xf16>
    // CHECK:       return [[OUTPUT]], [[PROBS]] : tensor<300x5xf16>, tensor<300xf16>
}
