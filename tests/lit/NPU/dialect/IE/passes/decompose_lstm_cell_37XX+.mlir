//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-lstm-cell %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @DecomposeLSTMCell
func.func @DecomposeLSTMCell(%arg0: tensor<5x67xf16>, %arg1: tensor<5x64xf16>, %arg2: tensor<5x64xf16>, %arg3: tensor<256x67xf16>, %arg4: tensor<256x64xf16>, %arg5: tensor<256xf16>) -> (tensor<5x64xf16>, tensor<5x64xf16>) {
    %cst = const.Declare tensor<256x67xf16> = dense<1.0> : tensor<256x67xf16>
    %cst_0 = const.Declare tensor<256x64xf16> = dense<2.0> : tensor<256x64xf16>
    %cst_1 = const.Declare tensor<256xf16> = dense<3.0> : tensor<256xf16>
    %outputHiddenState, %outputCellState = IE.LSTMCell(%arg0, %arg1, %arg2, %cst, %cst_0, %cst_1) {hiddenSize = 64 : i64} : tensor<5x67xf16>, tensor<5x64xf16>, tensor<5x64xf16>, tensor<256x67xf16>, tensor<256x64xf16>, tensor<256xf16> -> tensor<5x64xf16>, tensor<5x64xf16>
    return %outputHiddenState, %outputCellState : tensor<5x64xf16>, tensor<5x64xf16>

    // CHECK-NOT: IE.LSTMCell

    // CHECK:   [[CST_0:%.*]] = const.Declare tensor<256x67xf16> = dense<1.000000e+00> : tensor<256x67xf16>
    // CHECK:   [[CST_1:%.*]] = const.Declare tensor<256x64xf16> = dense<2.000000e+00> : tensor<256x64xf16>
    // CHECK:   [[CST_2:%.*]] = const.Declare tensor<256xf16> = dense<3.000000e+00> : tensor<256xf16>

    // CHECK:   [[MATMUL_0:%.*]] = IE.MatMul(%arg0, [[CST_0]]) {transpose_b} : tensor<5x67xf16>, tensor<256x67xf16> -> tensor<5x256xf16>
    // CHECK:   [[MATMUL_1:%.*]] = IE.MatMul(%arg1, [[CST_1]]) {transpose_b} : tensor<5x64xf16>, tensor<256x64xf16> -> tensor<5x256xf16>
    // CHECK:   [[ADD_0:%.*]] = IE.Add([[MATMUL_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x256xf16>, tensor<256xf16> -> tensor<5x256xf16>
    // CHECK:   [[ADD_1:%.*]] = IE.Add([[ADD_0]], [[MATMUL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<5x256xf16>, tensor<5x256xf16> -> tensor<5x256xf16>

    // CHECK:   [[LSTMGATES_0:%.*]], [[LSTMGATES_1:%.*]] = IE.LSTMGates([[ADD_1]], %arg2) : tensor<5x256xf16>, tensor<5x64xf16> -> tensor<5x64xf16>, tensor<5x64xf16>
    // CHECK:   return [[LSTMGATES_0]], [[LSTMGATES_1]] : tensor<5x64xf16>, tensor<5x64xf16>
}
