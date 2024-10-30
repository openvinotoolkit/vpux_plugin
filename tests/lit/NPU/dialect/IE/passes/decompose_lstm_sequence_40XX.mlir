//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-lstm-sequence %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: func.func @DecomposeLSTMSequence(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x64xf32>) -> (tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>) {
func.func @DecomposeLSTMSequence(%arg0: tensor<1x3x64xf32>) -> (tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>) {
    %cst_0 = const.Declare tensor<1x2x128xf32> = dense<1.000000e+00> : tensor<1x2x128xf32>
    %cst_1 = const.Declare tensor<2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>
    %cst_2 = const.Declare tensor<2x512x128xf32> = dense<3.000000e+00> : tensor<2x512x128xf32>
    %cst_3 = const.Declare tensor<2x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_1, %cst_2, %cst_3) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 3 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x3x64xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>, tensor<2x512x64xf32>, tensor<2x512x128xf32>, tensor<2x512xf32> -> tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>

// CHECK:   %[[VAL_1:.*]] = const.Declare tensor<2x1x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[2, 1, 512]>]
// CHECK:   %[[VAL_2:.*]] = const.Declare tensor<4xsi32> = dense<[1, 2, 3, 64]> : tensor<4xsi64>, [#const.CastElemType<si32>]
// CHECK:   %[[VAL_3:.*]] = const.Declare tensor<1x2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>, [#const.Reshape<[1, 2, 512, 64]>]
// CHECK:   %[[VAL_4:.*]] = const.Declare tensor<1x2x128xf32> = dense<1.000000e+00> : tensor<1x2x128xf32>
// CHECK:   %[[VAL_5:.*]] = const.Declare tensor<2x512x128xf32> = dense<3.000000e+00> : tensor<2x512x128xf32>
// CHECK:   %[[VAL_6:.*]] = IE.Unsqueeze(%[[VAL_0]]) {axes_value = [1]} : tensor<1x3x64xf32> -> tensor<1x1x3x64xf32>
// CHECK:   %[[VAL_7:.*]] = IE.Broadcast(%[[VAL_6]], %[[VAL_2]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x3x64xf32>, tensor<4xsi32> -> tensor<1x2x3x64xf32>
// CHECK:   %[[VAL_8:.*]] = IE.MatMul(%[[VAL_7]], %[[VAL_3]]) {transpose_b} : tensor<1x2x3x64xf32>, tensor<1x2x512x64xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_9:.*]] = IE.Add(%[[VAL_8]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x512xf32>, tensor<2x1x512xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]] = IE.LSTMSequence(%[[VAL_9]], %[[VAL_4]], %[[VAL_4]], %[[VAL_5]]) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>, sequenceLength = 3 : i64} : tensor<1x2x3x512xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>, tensor<2x512x128xf32> -> tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>
// CHECK:   return %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : tensor<1x2x3x128xf32>, tensor<1x2x128xf32>, tensor<1x2x128xf32>
}

// -----

// CHECK-LABEL: func.func @DecomposeUnsupportedLSTMSequenceToLSTMCells(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x64xf32>) -> (tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>) {
func.func @DecomposeUnsupportedLSTMSequenceToLSTMCells(%arg0: tensor<1x3x64xf32>) -> (tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>) {
    %cst_0 = const.Declare tensor<1x2x300xf32> = dense<1.000000e+00> : tensor<1x2x300xf32>
    %cst_1 = const.Declare tensor<2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>
    %cst_2 = const.Declare tensor<2x512x300xf32> = dense<3.000000e+00> : tensor<2x512x300xf32>
    %cst_3 = const.Declare tensor<2x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = IE.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_1, %cst_2, %cst_3) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 3 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x3x64xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>, tensor<2x512x64xf32>, tensor<2x512x300xf32>, tensor<2x512xf32> -> tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>

// CHECK:   %[[VAL_1:.*]] = const.Declare tensor<512x300xf32> = dense<3.000000e+00> : tensor<2x512x300xf32>, [#const.SubView<[1, 0, 0], [1, 512, 300]>, #const.Reshape<[512, 300]>]
// CHECK:   %[[VAL_2:.*]] = const.Declare tensor<1x300xf32> = dense<1.000000e+00> : tensor<1x2x300xf32>, [#const.SubView<[0, 1, 0], [1, 1, 300]>, #const.Reshape<[1, 300]>]
// CHECK:   %[[VAL_3:.*]] = const.Declare tensor<512x300xf32> = dense<3.000000e+00> : tensor<2x512x300xf32>, [#const.SubView<[0, 0, 0], [1, 512, 300]>, #const.Reshape<[512, 300]>]
// CHECK:   %[[VAL_4:.*]] = const.Declare tensor<1x300xf32> = dense<1.000000e+00> : tensor<1x2x300xf32>, [#const.SubView<[0, 0, 0], [1, 1, 300]>, #const.Reshape<[1, 300]>]
// CHECK:   %[[VAL_5:.*]] = const.Declare tensor<2x1x512xf32> = dense<4.000000e+00> : tensor<2x512xf32>, [#const.Reshape<[2, 1, 512]>]
// CHECK:   %[[VAL_6:.*]] = const.Declare tensor<4xsi32> = dense<[1, 2, 3, 64]> : tensor<4xsi64>, [#const.CastElemType<si32>]
// CHECK:   %[[VAL_7:.*]] = const.Declare tensor<1x2x512x64xf32> = dense<2.000000e+00> : tensor<2x512x64xf32>, [#const.Reshape<[1, 2, 512, 64]>]
// CHECK:   %[[VAL_8:.*]] = IE.Unsqueeze(%[[VAL_0]]) {axes_value = [1]} : tensor<1x3x64xf32> -> tensor<1x1x3x64xf32>
// CHECK:   %[[VAL_9:.*]] = IE.Broadcast(%[[VAL_8]], %[[VAL_6]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x3x64xf32>, tensor<4xsi32> -> tensor<1x2x3x64xf32>
// CHECK:   %[[VAL_10:.*]] = IE.MatMul(%[[VAL_9]], %[[VAL_7]]) {transpose_b} : tensor<1x2x3x64xf32>, tensor<1x2x512x64xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_11:.*]] = IE.Add(%[[VAL_10]], %[[VAL_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x3x512xf32>, tensor<2x1x512xf32> -> tensor<1x2x3x512xf32>
// CHECK:   %[[VAL_12:.*]] = IE.Slice %[[VAL_11]] [0, 0, 0, 0] [1, 1, 3, 512] : tensor<1x2x3x512xf32> to tensor<1x1x3x512xf32>
// CHECK:   %[[VAL_13:.*]] = IE.Slice %[[VAL_11]] [0, 1, 0, 0] [1, 1, 3, 512] : tensor<1x2x3x512xf32> to tensor<1x1x3x512xf32>
// CHECK:   %[[VAL_14:.*]] = IE.Squeeze(%[[VAL_12]]) {axes_value = [1]} : tensor<1x1x3x512xf32> -> tensor<1x3x512xf32>
// CHECK:   %[[VAL_15:.*]] = IE.Slice %[[VAL_14]] [0, 0, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_16:.*]] = IE.Squeeze(%[[VAL_15]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_17:.*]], %[[VAL_18:.*]] = IE.LSTMCell(%[[VAL_16]], %[[VAL_4]], %[[VAL_4]], %[[VAL_3]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_19:.*]] = IE.Unsqueeze(%[[VAL_17]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_20:.*]] = IE.Slice %[[VAL_14]] [0, 1, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_21:.*]] = IE.Squeeze(%[[VAL_20]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_22:.*]], %[[VAL_23:.*]] = IE.LSTMCell(%[[VAL_21]], %[[VAL_17]], %[[VAL_18]], %[[VAL_3]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_24:.*]] = IE.Unsqueeze(%[[VAL_22]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_25:.*]] = IE.Slice %[[VAL_14]] [0, 2, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_26:.*]] = IE.Squeeze(%[[VAL_25]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_27:.*]], %[[VAL_28:.*]] = IE.LSTMCell(%[[VAL_26]], %[[VAL_22]], %[[VAL_23]], %[[VAL_3]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_29:.*]] = IE.Unsqueeze(%[[VAL_27]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_30:.*]] = IE.Concat(%[[VAL_19]], %[[VAL_24]], %[[VAL_29]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x3x300xf32>
// CHECK:   %[[VAL_31:.*]] = IE.Unsqueeze(%[[VAL_30]]) {axes_value = [1]} : tensor<1x3x300xf32> -> tensor<1x1x3x300xf32>
// CHECK:   %[[VAL_32:.*]] = IE.Unsqueeze(%[[VAL_27]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_33:.*]] = IE.Unsqueeze(%[[VAL_28]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_34:.*]] = IE.Squeeze(%[[VAL_13]]) {axes_value = [1]} : tensor<1x1x3x512xf32> -> tensor<1x3x512xf32>
// CHECK:   %[[VAL_35:.*]] = IE.Slice %[[VAL_34]] [0, 2, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_36:.*]] = IE.Squeeze(%[[VAL_35]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_37:.*]], %[[VAL_38:.*]] = IE.LSTMCell(%[[VAL_36]], %[[VAL_2]], %[[VAL_2]], %[[VAL_1]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_39:.*]] = IE.Unsqueeze(%[[VAL_37]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_40:.*]] = IE.Slice %[[VAL_34]] [0, 1, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_41:.*]] = IE.Squeeze(%[[VAL_40]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_42:.*]], %[[VAL_43:.*]] = IE.LSTMCell(%[[VAL_41]], %[[VAL_37]], %[[VAL_38]], %[[VAL_1]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_44:.*]] = IE.Unsqueeze(%[[VAL_42]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_45:.*]] = IE.Slice %[[VAL_34]] [0, 0, 0] [1, 1, 512] : tensor<1x3x512xf32> to tensor<1x1x512xf32>
// CHECK:   %[[VAL_46:.*]] = IE.Squeeze(%[[VAL_45]]) {axes_value = [1]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
// CHECK:   %[[VAL_47:.*]], %[[VAL_48:.*]] = IE.LSTMCell(%[[VAL_46]], %[[VAL_42]], %[[VAL_43]], %[[VAL_1]]) {hiddenSize = 300 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<1x512xf32>, tensor<1x300xf32>, tensor<1x300xf32>, tensor<512x300xf32> -> tensor<1x300xf32>, tensor<1x300xf32>
// CHECK:   %[[VAL_49:.*]] = IE.Unsqueeze(%[[VAL_47]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_50:.*]] = IE.Concat(%[[VAL_49]], %[[VAL_44]], %[[VAL_39]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x3x300xf32>
// CHECK:   %[[VAL_51:.*]] = IE.Unsqueeze(%[[VAL_50]]) {axes_value = [1]} : tensor<1x3x300xf32> -> tensor<1x1x3x300xf32>
// CHECK:   %[[VAL_52:.*]] = IE.Unsqueeze(%[[VAL_47]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_53:.*]] = IE.Unsqueeze(%[[VAL_48]]) {axes_value = [1]} : tensor<1x300xf32> -> tensor<1x1x300xf32>
// CHECK:   %[[VAL_54:.*]] = IE.Concat(%[[VAL_31]], %[[VAL_51]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x3x300xf32>, tensor<1x1x3x300xf32> -> tensor<1x2x3x300xf32>
// CHECK:   %[[VAL_55:.*]] = IE.Concat(%[[VAL_32]], %[[VAL_52]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x2x300xf32>
// CHECK:   %[[VAL_56:.*]] = IE.Concat(%[[VAL_33]], %[[VAL_53]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x300xf32>, tensor<1x1x300xf32> -> tensor<1x2x300xf32>
// CHECK:   return %[[VAL_54]], %[[VAL_55]], %[[VAL_56]] : tensor<1x2x3x300xf32>, tensor<1x2x300xf32>, tensor<1x2x300xf32>
}

