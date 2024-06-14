//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-gru-sequence %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: func.func @SplitGRUSequence
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x157x512xf16>
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x1x384xf16>
func.func @SplitGRUSequence(%arg0: tensor<1x157x512xf16>, %arg1: tensor<1x1x384xf16>) -> (tensor<1x1x157x384xf16>, tensor<1x1x384xf16>) {
      %cst = const.Declare tensor<1x1536xf16> = dense<1.000000e+00> : tensor<1x1536xf16>
      %cst_0 = const.Declare tensor<1x1152x384xf16> = dense<1.000000e+00> : tensor<1x1152x384xf16>
      %cst_1 = const.Declare tensor<1x1152x512xf16> = dense<1.000000e+00> : tensor<1x1152x512xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 384 : i64, seq_length = 157 : i64, should_linear_before_reset} : tensor<1x157x512xf16>, tensor<1x1x384xf16>, tensor<1x1152x512xf16>, tensor<1x1152x384xf16>, tensor<1x1536xf16> -> tensor<1x1x157x384xf16>, tensor<1x1x384xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<1x1x157x384xf16>, tensor<1x1x384xf16>

      // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1536xf16> = dense<1.000000e+00> : tensor<1x1536xf16>
      // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1152x384xf16> = dense<1.000000e+00> : tensor<1x1152x384xf16>
      // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1152x512xf16> = dense<1.000000e+00> : tensor<1x1152x512xf16

      // CHECK:     [[FIRST_PART:%.+]] = VPU.GRUSequenceFirstPart([[INPUT_0]], [[CST_1]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, hidden_size = 384 : i64, seq_length = 157 : i64} : tensor<1x157x512xf16>, tensor<1x1152x512xf16> -> tensor<1x1x157x1152xf16>
      // CHECK:     [[OUT_0:%.+]], [[OUT_1:%.+]] = VPU.GRUSequenceLastPart([[FIRST_PART]], [[INPUT_1]], [[CST_0]], [[CST]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 384 : i64, seq_length = 157 : i64, should_linear_before_reset}
      // CHECK-SAME:tensor<1x1x157x1152xf16>, tensor<1x1x384xf16>, tensor<1x1152x384xf16>, tensor<1x1536xf16> -> tensor<1x1x157x384xf16>, tensor<1x1x384xf16>
      // CHECK:     return [[OUT_0]], [[OUT_1]] : tensor<1x1x157x384xf16>, tensor<1x1x384xf16>
}

// -----

// CHECK-LABEL: func.func @NotSplitGRUSequence
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x157x384xf16>
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x1x256xf16>
func.func @NotSplitGRUSequence(%arg0: tensor<1x157x384xf16>, %arg1: tensor<1x1x256xf16>) -> (tensor<1x1x157x256xf16>, tensor<1x1x256xf16>) {
      %cst = const.Declare tensor<1x1024xf16> = dense<1.000000e+00> : tensor<1x1024xf16>
      %cst_0 = const.Declare tensor<1x768x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>
      %cst_1 = const.Declare tensor<1x768x384xf16> = dense<1.000000e+00> : tensor<1x768x384xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 256 : i64, seq_length = 157 : i64, should_linear_before_reset} : tensor<1x157x384xf16>, tensor<1x1x256xf16>, tensor<1x768x384xf16>, tensor<1x768x256xf16>, tensor<1x1024xf16> -> tensor<1x1x157x256xf16>, tensor<1x1x256xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<1x1x157x256xf16>, tensor<1x1x256xf16>

      // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1024xf16> = dense<1.000000e+00> : tensor<1x1024xf16>
      // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x768x256xf16> = dense<1.000000e+00> : tensor<1x768x256xf16>
      // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x768x384xf16> = dense<1.000000e+00> : tensor<1x768x384xf16

      // CHECK:     [[OUT_0:%.+]], [[OUT_1:%.+]] = VPU.GRUSequence([[INPUT_0]], [[INPUT_1]], [[CST_1]], [[CST_0]], [[CST]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 256 : i64, seq_length = 157 : i64, should_linear_before_reset}
      // CHECK-SAME tensor<1x157x384xf16>, tensor<1x1x256xf16>, tensor<1x768x384xf16>, tensor<1x768x256xf16>, tensor<1x1024xf16> -> tensor<1x1x157x256xf16>, tensor<1x1x256xf16>

      // CHECK:     return [[OUT_0]], [[OUT_1]] : tensor<1x1x157x256xf16>, tensor<1x1x256xf16>
}
