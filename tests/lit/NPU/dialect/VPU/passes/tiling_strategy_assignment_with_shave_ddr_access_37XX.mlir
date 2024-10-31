//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="tiling-mode=ISOLATED enable-shave-ddr-access-optimization=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: func.func @GatherDDRAccessWithoutTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<51865x512xf16>
func.func @GatherDDRAccessWithoutTiling(%arg0: tensor<51865x512xf16>) -> tensor<1x16x512xf16> {
    %cst = const.Declare tensor<1x16xsi32> = dense<1> : tensor<1x16xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<51865x512xf16>, tensor<1x16xsi32> -> tensor<1x16x512xf16>
    return %0 : tensor<1x16x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x16xsi32> = dense<1> : tensor<1x16xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<51865x512xf16>, tensor<1x16xsi32> -> tensor<1x16x512xf16>

    // CHECK:     return [[GATHER]] : tensor<1x16x512xf16>
}

// -----

// CHECK-LABEL: func.func @GatherAssignTilingStrategy
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<51865x512xf16>
func.func @GatherAssignTilingStrategy(%arg0: tensor<51865x512xf16>) -> tensor<1x2000x512xf16> {
    %cst = const.Declare tensor<1x2000xsi32> = dense<1> : tensor<1x2000xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<51865x512xf16>, tensor<1x2000xsi32> -> tensor<1x2000x512xf16>
    return %0 : tensor<1x2000x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x2000xsi32> = dense<1> : tensor<1x2000xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 1, 29]} : tensor<51865x512xf16>, tensor<1x2000xsi32> -> tensor<1x2000x512xf16>

    // CHECK:     return [[GATHER]] : tensor<1x2000x512xf16>
}

// -----

// CHECK-LABEL: func.func @GRUSequenceDDRAccessWithoutTiling
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1x200xf16>
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1x1024xf16>
func.func @GRUSequenceDDRAccessWithoutTiling(%arg0: tensor<1x1x200xf16>, %arg1: tensor<1x1x1024xf16>) -> (tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>) {
    %cst = const.Declare tensor<1x3072x200xf16> = dense<1.000000e+00> : tensor<1x3072x200xf16>
    %cst_0 = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    %cst_1 = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x200xf16>, tensor<1x1x1024xf16>, tensor<1x3072x200xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    return %middle_hidden_state, %output_hidden_state : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x3072x200xf16> = dense<1.000000e+00> : tensor<1x3072x200xf16>
    // CHECK:     [[CST0:%.+]] = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    // CHECK:     [[CST1:%.+]] = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    // CHECK:     [[OUT_0:%.+]], [[OUT_1:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]]) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x200xf16>, tensor<1x1x1024xf16>, tensor<1x3072x200xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    // CHECK:     return [[OUT_0]], [[OUT_1]] : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
}

// -----

// CHECK-LABEL: func.func @GRUSequenceLastPartDDRAccessWithoutTiling
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1x1x3072xf16>
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1x1024xf16>
func.func @GRUSequenceLastPartDDRAccessWithoutTiling(%arg0: tensor<1x1x1x3072xf16>, %arg1: tensor<1x1x1024xf16>) -> (tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>) {
    %cst = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    %cst_0 = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    %middle_hidden_state, %output_hidden_state = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst, %cst_0) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x1x3072xf16>, tensor<1x1x1024xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    return %middle_hidden_state, %output_hidden_state : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    // CHECK: [[CST0:%.+]] = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    // CHECK: [[OUT0:%.+]], [[OUT1:%.+]] = VPU.GRUSequenceLastPart([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]]) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x1x3072xf16>, tensor<1x1x1024xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    // CHECK: return [[OUT0]], [[OUT1]] : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
}
