//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE --mlir-print-debuginfo ./tensor_iterator_rename.xml | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK:       func.func @main
// CHECK:           [[ARG0:%arg[0-9]+]]: tensor<1x3x32xf32>

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x64xf32> = dense<0.000000e+00> : tensor<1x64xf32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x64xf32> = dense<0.000000e+00> : tensor<1x64xf32>
// CHECK-DAG:       [[CST1:%.*]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>
// CHECK:           [[TI0:%.*]]:3 = IE.TensorIterator body_module : {

// CHECK:           ^bb0(%arg1: tensor<1x1x32xf32> loc(fused<{name = "[[LSTMSeq_1:LSTMSequence_[0-9]+]]"{{.*}}
// CHECK:           [[TI0CST7:%.*]] = const.Declare tensor<2x256x32xsi8> = dense<1> : tensor<2x256x32xsi8> {{.*}}
// CHECK:           [[TI0VAR12:%.*]] = IE.Convert([[TI0CST7]]) {dstElemType = f32} : tensor<2x256x32xsi8> -> tensor<2x256x32xf32> {{.*}}
// CHECK:           [[TI0CST8:%.*]] = const.Declare tensor<1x256x1xf32> = dense<1.000000e+00> : tensor<1x256x1xf32> {{.*}}
// CHECK:           [[TI0VAR13:%.*]] = IE.Multiply([[TI0VAR12]], [[TI0CST8]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:      : tensor<2x256x32xf32>, tensor<1x256x1xf32> -> tensor<2x256x32xf32> loc([[TI0MUL0LOC:#.*]])

// CHECK:           [[TI0CST11:%.*]] = const.Declare tensor<2x256x64xsi8> = dense<1> : tensor<2x256x64xsi8> {{.*}}
// CHECK:           [[TI0VAR16:%.*]] = IE.Convert([[TI0CST11]]) {dstElemType = f32} : tensor<2x256x64xsi8> -> tensor<2x256x64xf32> {{.*}}
// CHECK:           [[TI0CST12:%.*]] = const.Declare tensor<1x256x1xf32> = dense<1.000000e+00> : tensor<1x256x1xf32> {{.*}}
// CHECK:           [[TI0VAR17:%.*]] = IE.Multiply([[TI0VAR16]], [[TI0CST12]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:      : tensor<2x256x64xf32>, tensor<1x256x1xf32> -> tensor<2x256x64xf32> loc([[TI0MUL1LOC:#.*]])

// CHECK:           num_iterations : {{.*}}
// CHECK:           slice_input_descs : {{.*}}
// CHECK:           invariant_input_descs : {{.*}}
// CHECK:           feedback_input_descs : {{.*}}
// CHECK:           concat_output_descs : {{.*}}
// CHECK:           invariant_output_descs : {{.*}}
// CHECK:           ([[ARG0]], [[CST]], [[CST0]], [[CST1]])
// CHECK:           : tensor<1x3x32xf32>, tensor<1x64xf32>, tensor<1x64xf32>, tensor<1xsi64> -> tensor<1x3x64xf32>, tensor<1x64xf32>, tensor<1x64xf32> {{.*}}

// CHECK-DAG:       [[CST3:%.+]] = const.Declare tensor<1x64xf32> = dense<0.000000e+00> : tensor<1x64xf32>
// CHECK-DAG:       [[CST4:%.+]] = const.Declare tensor<1x64xf32> = dense<0.000000e+00> : tensor<1x64xf32>
// CHECK:           [[TI1:%.*]]:3 = IE.TensorIterator body_module : {

// CHECK:           ^bb0(%arg1: tensor<1x1x32xf32> loc(fused<{name = "[[LSTMSeq_2:LSTMSequence_[0-9]+]]"{{.*}}
// CHECK:           [[TI1CST7:%.*]] = const.Declare tensor<2x256x32xsi8> = dense<1> : tensor<2x256x32xsi8> {{.*}}
// CHECK:           [[TI1VAR12:%.*]] = IE.Convert([[TI1CST7]]) {dstElemType = f32} : tensor<2x256x32xsi8> -> tensor<2x256x32xf32> {{.*}}
// CHECK:           [[TI1CST8:%.*]] = const.Declare tensor<1x256x1xf32> = dense<1.000000e+00> : tensor<1x256x1xf32> {{.*}}
// CHECK:           [[TI1VAR13:%.*]] = IE.Multiply([[TI1VAR12]], [[TI1CST8]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:      : tensor<2x256x32xf32>, tensor<1x256x1xf32> -> tensor<2x256x32xf32> loc([[TI1MUL0LOC:#.*]])

// CHECK:           [[TI1CST11:%.*]] = const.Declare tensor<2x256x64xsi8> = dense<1> : tensor<2x256x64xsi8> {{.*}}
// CHECK:           [[TI1VAR16:%.*]] = IE.Convert([[TI1CST11]]) {dstElemType = f32} : tensor<2x256x64xsi8> -> tensor<2x256x64xf32> {{.*}}
// CHECK:           [[TI1CST12:%.*]] = const.Declare tensor<1x256x1xf32> = dense<1.000000e+00> : tensor<1x256x1xf32> {{.*}}
// CHECK:           [[TI1VAR17:%.*]] = IE.Multiply([[TI1VAR16]], [[TI1CST12]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:      : tensor<2x256x64xf32>, tensor<1x256x1xf32> -> tensor<2x256x64xf32> loc([[TI1MUL1LOC:#.*]])

// CHECK:           num_iterations : {{.*}}
// CHECK:           slice_input_descs : {{.*}}
// CHECK:           invariant_input_descs : {{.*}}
// CHECK:           feedback_input_descs : {{.*}}
// CHECK:           concat_output_descs : {{.*}}
// CHECK:           invariant_output_descs : {{.*}}
// CHECK:           ([[ARG0]], [[CST3]], [[CST4]], [[CST1]])
// CHECK:           : tensor<1x3x32xf32>, tensor<1x64xf32>, tensor<1x64xf32>, tensor<1xsi64> -> tensor<1x3x64xf32>, tensor<1x64xf32>, tensor<1x64xf32> {{.*}}

// CHECK:           return {{.*}}
// CHECK:           [[TI0MUL0LOC]] = loc(fused<{name = "[[LSTMSeq_1]]?t_TensorIterator/body/Multiply_8", type = "Multiply"}>{{.*}}
// CHECK:           [[TI0MUL1LOC]] = loc(fused<{name = "[[LSTMSeq_1]]?t_TensorIterator/body/Multiply_12", type = "Multiply"}>{{.*}}
// CHECK:           [[TI1MUL0LOC]] = loc(fused<{name = "[[LSTMSeq_2]]?t_TensorIterator/body/Multiply_8", type = "Multiply"}>{{.*}}
// CHECK:           [[TI1MUL1LOC]] = loc(fused<{name = "[[LSTMSeq_2]]?t_TensorIterator/body/Multiply_12", type = "Multiply"}>{{.*}}
