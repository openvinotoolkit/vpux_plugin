//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-shaped-type-result-dims %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BoundedType = tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>

// CHECK-LABEL: @ReifyReLUShape
func.func @ReifyReLUShape(%IN: !BoundedType) -> (!BoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
    %IDX_3 = arith.constant 3 : index
    // CHECK: [[IDX_3:%.+]] = arith.constant 3 : index

    %RELU = IE.ReLU(%IN) : !BoundedType -> !BoundedType
    // CHECK: [[RELU:%.+]] = IE.ReLU([[IN]])

    %DIM_3 = tensor.dim %RELU, %IDX_3 : !BoundedType
    // CHECK: [[DIM_3:%.+]] = tensor.dim [[IN]], [[IDX_3]]
    // Note that the first operand of tensor.dim comes from the block argument now

    return %RELU, %DIM_3 : !BoundedType, index
    // CHECK: return [[RELU]], [[DIM_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BoundedType = tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>

// CHECK-LABEL: @ReifyAddShape
// CHECK-SAME: [[IN1:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>,
// CHECK-SAME: [[IN2:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
func.func @ReifyAddShape(%IN1: !BoundedType, %IN2: !BoundedType) -> (!BoundedType, index) {
    %IDX_3 = arith.constant 3 : index
    // CHECK: [[IDX_3:%.+]] = arith.constant 3 : index

    %ADD = IE.Add(%IN1, %IN2) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : !BoundedType, !BoundedType -> !BoundedType
    // CHECK: [[ADD:%.+]] = IE.Add([[IN1]], [[IN2]])

    %DIM_3 = tensor.dim %ADD, %IDX_3 : !BoundedType
    // CHECK: [[DIM_3:%.+]] = tensor.dim [[IN1]], [[IDX_3]]

    return %ADD, %DIM_3 : !BoundedType, index
    // CHECK: return [[ADD]], [[DIM_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Type1 = tensor<1x64xf16, {order = #NCHW}>
!Type2 = tensor<1x?x32x1xf16, {bounds = [1, 16, 32, 1], order = #NCHW}>
!OutType = tensor<1x?x32x64xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>

// CHECK-LABEL: @ReifyBroadcastAddShape
// CHECK-SAME: [[IN1:%.+]]: tensor<1x64xf16, {order = #NCHW}>,
// CHECK-SAME: [[IN2:%.+]]: tensor<1x?x32x1xf16, {bounds = [1, 16, 32, 1], order = #NCHW}>
func.func @ReifyBroadcastAddShape(%IN1: !Type1, %IN2: !Type2) -> (!OutType, index) {
    %IDX_1 = arith.constant 1 : index
    // CHECK: [[IDX_1:%.+]] = arith.constant 1 : index

    %ADD = IE.Add(%IN1, %IN2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : !Type1, !Type2 -> !OutType
    // CHECK: [[ADD:%.+]] = IE.Add([[IN1]], [[IN2]])

    %DIM_1 = tensor.dim %ADD, %IDX_1 : !OutType
    // CHECK: [[DIM_1:%.+]] = tensor.dim [[IN2]], [[IDX_1]]

    return %ADD, %DIM_1 : !OutType, index
    // CHECK: return [[ADD]], [[DIM_1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BoundedType = tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>

// CHECK-LABEL: @ReifySoftmaxShape
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
func.func @ReifySoftmaxShape(%IN: !BoundedType) -> (!BoundedType, index) {
    %IDX_3 = arith.constant 3 : index
    // CHECK: [[IDX_3:%.+]] = arith.constant 3 : index

    %SOFTMAX = IE.SoftMax(%IN) {axisInd = 3 : i64} : !BoundedType -> !BoundedType
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[IN]])

    %DIM_3 = tensor.dim %SOFTMAX, %IDX_3 : !BoundedType
    // CHECK: [[DIM_3:%.+]] = tensor.dim [[IN]], [[IDX_3]]

    return %SOFTMAX, %DIM_3 : !BoundedType, index
    // CHECK: return [[SOFTMAX]], [[DIM_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!BoundedType = tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
// CHECK-LABEL: @ReifyMaxPoolShape
func.func @ReifyMaxPoolShape(%IN: !BoundedType) -> (tensor<1x16x30x?xf16, {bounds = [1, 16, 30, 64], order = #NCHW}>, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
    %C3 = arith.constant 3 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index

    %MAXPOOL = IE.MaxPool(%IN) {
            kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : !BoundedType -> tensor<1x16x30x?xf16, {bounds = [1, 16, 30, 64], order = #NCHW}>
    // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[IN]])

    %DIM = tensor.dim %MAXPOOL, %C3 : tensor<1x16x30x?xf16, {bounds = [1, 16, 30, 64], order = #NCHW}>
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN]], [[C3]]
    // CHECK: [[DIM_SUB:%.+]] = arith.subi [[DIM]], [[C2]]

    return %MAXPOOL, %DIM : tensor<1x16x30x?xf16, {bounds = [1, 16, 30, 64], order = #NCHW}>, index
    // CHECK: return [[MAXPOOL]], [[DIM_SUB]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!InBoundedType = tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
!OutBoundedType = tensor<1x16x16x?xf16, {bounds = [1, 16, 16, 64], order = #NCHW}>

// CHECK-LABEL: @ReifyMaxPoolShape
func.func @ReifyMaxPoolShape(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
    %C3 = arith.constant 3 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index

    %MAXPOOL = IE.MaxPool(%IN) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : !InBoundedType -> !OutBoundedType
    // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[IN]])

    %DIM = tensor.dim %MAXPOOL, %C3 : !OutBoundedType
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN]], [[C3]]
    // CHECK: [[DIM_DIV:%.+]] = arith.divsi [[DIM]], [[C2]]

    return %MAXPOOL, %DIM : !OutBoundedType, index
    // CHECK: return [[MAXPOOL]], [[DIM_DIV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!InBoundedType = tensor<1x16x?x32xf16, {bounds = [1, 16, 64, 32], order = #NCHW}>
!OutBoundedType = tensor<1x16x?x16xf16, {bounds = [1, 16, 64, 16], order = #NCHW}>

// CHECK-LABEL: @ReifyMaxPoolShape
func.func @ReifyMaxPoolShape(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x?x32xf16, {bounds = [1, 16, 64, 32], order = #NCHW}>
    %C2 = arith.constant 2 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index

    %MAXPOOL = IE.MaxPool(%IN) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : !InBoundedType -> !OutBoundedType
    // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[IN]])

    %DIM = tensor.dim %MAXPOOL, %C2 : !OutBoundedType
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN]], [[C2]]
    // CHECK: [[DIM_DIV:%.+]] = arith.divsi [[DIM]], [[C2]]

    return %MAXPOOL, %DIM : !OutBoundedType, index
    // CHECK: return [[MAXPOOL]], [[DIM_DIV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReifyConvShape
func.func @ReifyConvShape(%IN: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>)
    -> (tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
    %C3 = arith.constant 3 : index
    // CHECK: [[C3:%.+]] = arith.constant 3 : index
    %CST = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK: [[CST:%.+]] = const.Declare

    %CONV = IE.Convolution(%IN, %CST) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
                tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>,
                tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[IN]], [[CST]])

    %DIM = tensor.dim %CONV, %C3 : tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN]], [[C3]]

    return %CONV, %DIM : tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>, index
    // CHECK: return [[CONV]], [[DIM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReifyMaxPoolConvReLUMaxPoolConvShape
func.func @ReifyMaxPoolConvReLUMaxPoolConvShape(%IN: tensor<1x16x64x?xf16, {bounds = [1, 16, 64, 64], order = #NCHW}>)
    -> (tensor<1x16x16x?xf16, {bounds = [1, 32, 16, 64], order = #NCHW}>, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x64x?xf16, {bounds = [1, 16, 64, 64], order = #NCHW}>
    %C3 = arith.constant 3 : index
    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
    %CST1 = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    %CST2 = const.Declare tensor<16x32x1x1xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare
    // CHECK-SAME: tensor<32x16x3x3xf16, {order = #NCHW}>

    // CHECK-DAG: [[CST2:%.+]] = const.Declare
    // CHECK-SAME: tensor<16x32x1x1xf16, {order = #NCHW}>

    %MAXPOOL1 = IE.MaxPool(%IN) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x16x64x?xf16, {bounds = [1, 16, 64, 64], order = #NCHW}>
      -> tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>

    %CONV1 = IE.Convolution(%MAXPOOL1, %CST1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
                tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>,
                tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>

    %RELU = IE.ReLU(%CONV1) : tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>
                           -> tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>

    %MAXPOOL2 = IE.MaxPool(%RELU) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x32x32x?xf16, {bounds = [1, 32, 32, 64], order = #NCHW}>
      -> tensor<1x32x16x?xf16, {bounds = [1, 32, 16, 64], order = #NCHW}>

    %CONV2 = IE.Convolution(%MAXPOOL2, %CST2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
                tensor<1x32x16x?xf16, {bounds = [1, 32, 16, 64], order = #NCHW}>, tensor<16x32x1x1xf16, {order = #NCHW}>
                -> tensor<1x16x16x?xf16, {bounds = [1, 32, 16, 64], order = #NCHW}>

    %DIM = tensor.dim %CONV2, %C3 : tensor<1x16x16x?xf16, {bounds = [1, 32, 16, 64], order = #NCHW}>
    // CHECK: [[MAXPOOL1:%.+]] = IE.MaxPool([[IN]])
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[MAXPOOL1]], [[CST1]])
    // CHECK: [[RELU:%.+]] = IE.ReLU([[CONV1]])
    // CHECK: [[MAXPOOL2:%.+]] = IE.MaxPool([[RELU]])
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[MAXPOOL2]], [[CST2]])

    // CHECK: [[DIM:%.+]] = tensor.dim [[IN]], [[C3]]
    // CHECK: [[DIM_DIV_1:%.+]] = arith.divsi [[DIM]], [[C2]]
    // CHECK: [[DIM_DIV_2:%.+]] = arith.divsi [[DIM_DIV_1]], [[C2]]

    return %CONV2, %DIM : tensor<1x16x16x?xf16, {bounds = [1, 32, 16, 64], order = #NCHW}>, index
    // CHECK: return [[CONV2]], [[DIM_DIV_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType1 = tensor<1x2x1x?xf32, {bounds = [1, 2, 1, 512], order = #NCHW}>
!InBoundedType2 = tensor<1x2x512x?xf32, {bounds = [1, 2, 512, 40], order = #NCHW}>
!OutBoundedType = tensor<1x2x1x?xf32, {bounds = [1, 2, 1, 40], order = #NCHW}>

// CHECK-LABEL: @ReifyMatMulShape0
func.func @ReifyMatMulShape0(%IN1: !InBoundedType1, %IN2: !InBoundedType2) -> (!OutBoundedType, index) {
    // CHECK: [[IN1:%.+]]: tensor<1x2x1x?xf32, {bounds = [1, 2, 1, 512], order = #NCHW}>
    // CHECK: [[IN2:%.+]]: tensor<1x2x512x?xf32, {bounds = [1, 2, 512, 40], order = #NCHW}>
    %C3 = arith.constant 3 : index
    // CHECK: [[C3:%.+]] = arith.constant 3 : index

    %MATMUL = IE.MatMul(%IN1, %IN2) : !InBoundedType1, !InBoundedType2 -> !OutBoundedType
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN1]], [[IN2]])

    %DIM = tensor.dim %MATMUL, %C3 : !OutBoundedType
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN2]], [[C3]]

    return %MATMUL, %DIM : !OutBoundedType, index
    // CHECK: return [[MATMUL]], [[DIM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType1 = tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 512], order = #NCHW}>
!InBoundedType2 = tensor<1x2x?x512xf32, {bounds = [1, 2, 128, 512], order = #NCHW}>
!OutBoundedType = tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 128], order = #NCHW}>

// CHECK-LABEL: @ReifyMatMulShape1
func.func @ReifyMatMulShape1(%IN1: !InBoundedType1, %IN2: !InBoundedType2) -> (!OutBoundedType, index) {
    // CHECK: [[IN1:%.+]]: tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 512], order = #NCHW}>
    // CHECK: [[IN2:%.+]]: tensor<1x2x?x512xf32, {bounds = [1, 2, 128, 512], order = #NCHW}>
    %IDX_3 = arith.constant 3 : index
    // CHECK: [[IDX_2:%.+]] = arith.constant 2 : index

    %MATMUL = IE.MatMul(%IN1, %IN2) {transpose_b} : !InBoundedType1, !InBoundedType2 -> !OutBoundedType
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN1]], [[IN2]]) {transpose_b}

    %DIM = tensor.dim %MATMUL, %IDX_3 : !OutBoundedType
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN2]], [[IDX_2]]

    return %MATMUL, %DIM : !OutBoundedType, index
    // CHECK: return [[MATMUL]], [[DIM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType1 = tensor<1x2x256x512xf32>
!InBoundedType2 = tensor<1x2x?x512xf32, {bounds = [1, 2, 128, 512], order = #NCHW}>
!OutBoundedType = tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 128], order = #NCHW}>

// CHECK-LABEL: @ReifyMatMulShape2
func.func @ReifyMatMulShape2(%IN1: !InBoundedType1, %IN2: !InBoundedType2) -> (!OutBoundedType, index) {
    // CHECK: [[IN1:%.+]]: tensor<1x2x256x512xf32>
    // CHECK: [[IN2:%.+]]: tensor<1x2x?x512xf32, {bounds = [1, 2, 128, 512], order = #NCHW}>
    %IDX_3 = arith.constant 3 : index
    // CHECK: [[IDX_2:%.+]] = arith.constant 2 : index

    %MATMUL = IE.MatMul(%IN1, %IN2) {transpose_b} : !InBoundedType1, !InBoundedType2 -> !OutBoundedType
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN1]], [[IN2]]) {transpose_b}

    %DIM = tensor.dim %MATMUL, %IDX_3 : !OutBoundedType
    // CHECK: [[DIM:%.+]] = tensor.dim [[IN2]], [[IDX_2]]

    return %MATMUL, %DIM : !OutBoundedType, index
    // CHECK: return [[MATMUL]], [[DIM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType1 = tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 512], order = #NCHW}>
!InBoundedType2 = tensor<1x2x?x256xf32, {bounds = [1, 2, 128, 256], order = #NCHW}>
!OutBoundedType = tensor<1x2x?x?xf32, {bounds = [1, 2, 512, 128], order = #NCHW}>

// CHECK-LABEL: @ReifyMatMulShape3
func.func @ReifyMatMulShape3(%IN1: !InBoundedType1, %IN2: !InBoundedType2) -> (!OutBoundedType, index, index) {
    // CHECK: [[IN1:%.+]]: tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 512], order = #NCHW}>
    // CHECK: [[IN2:%.+]]: tensor<1x2x?x256xf32, {bounds = [1, 2, 128, 256], order = #NCHW}>
    %IDX_2 = arith.constant 2 : index
    %IDX_3 = arith.constant 3 : index

    // CHECK: [[IDX_2:%.+]] = arith.constant 2 : index
    // CHECK: [[IDX_3:%.+]] = arith.constant 3 : index

    %MATMUL = IE.MatMul(%IN1, %IN2) {transpose_a, transpose_b} : !InBoundedType1, !InBoundedType2 -> !OutBoundedType
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN1]], [[IN2]]) {transpose_a, transpose_b}

    %DIM_H = tensor.dim %MATMUL, %IDX_2 : !OutBoundedType
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[IN1]], [[IDX_3]]

    %DIM_W = tensor.dim %MATMUL, %IDX_3 : !OutBoundedType
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[IN2]], [[IDX_2]]

    return %MATMUL, %DIM_H, %DIM_W : !OutBoundedType, index, index
    // CHECK: return [[MATMUL]], [[DIM_H]], [[DIM_W]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType1 = tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 512], order = #NCHW}>
!InBoundedType2 = tensor<2x?x256xf32, {bounds = [2, 128, 256], order = #NCHW}>
!OutBoundedType = tensor<1x2x?x?xf32, {bounds = [1, 2, 512, 128], order = #NCHW}>

// CHECK-LABEL: @ReifyMatMulShape4
func.func @ReifyMatMulShape4(%IN1: !InBoundedType1, %IN2: !InBoundedType2) -> (!OutBoundedType, index, index) {
    // CHECK: [[IN1:%.+]]: tensor<1x2x256x?xf32, {bounds = [1, 2, 256, 512], order = #NCHW}>
    // CHECK: [[IN2:%.+]]: tensor<2x?x256xf32, {bounds = [2, 128, 256], order = #NCHW}>
    %IDX_2 = arith.constant 2 : index
    %IDX_3 = arith.constant 3 : index

    // CHECK: [[IDX_1:%.+]] = arith.constant 1 : index
    // CHECK: [[IDX_3:%.+]] = arith.constant 3 : index

    %MATMUL = IE.MatMul(%IN1, %IN2) {transpose_a, transpose_b} : !InBoundedType1, !InBoundedType2 -> !OutBoundedType
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN1]], [[IN2]]) {transpose_a, transpose_b}

    %DIM_H = tensor.dim %MATMUL, %IDX_2 : !OutBoundedType
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[IN1]], [[IDX_3]]

    %DIM_W = tensor.dim %MATMUL, %IDX_3 : !OutBoundedType
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[IN2]], [[IDX_1]]

    return %MATMUL, %DIM_H, %DIM_W : !OutBoundedType, index, index
    // CHECK: return [[MATMUL]], [[DIM_H]], [[DIM_W]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType1 = tensor<2x256x?xf32, {bounds = [2, 256, 512], order = #NCHW}>
!InBoundedType2 = tensor<1x2x?x256xf32, {bounds = [1, 2, 128, 256], order = #NCHW}>
!OutBoundedType = tensor<1x2x?x?xf32, {bounds = [1, 2, 512, 128], order = #NCHW}>

// CHECK-LABEL: @ReifyMatMulShape5
func.func @ReifyMatMulShape5(%IN1: !InBoundedType1, %IN2: !InBoundedType2) -> (!OutBoundedType, index, index) {
    // CHECK: [[IN1:%.+]]: tensor<2x256x?xf32, {bounds = [2, 256, 512], order = #NCHW}>
    // CHECK: [[IN2:%.+]]: tensor<1x2x?x256xf32, {bounds = [1, 2, 128, 256], order = #NCHW}>
    %IDX_2 = arith.constant 2 : index
    %IDX_3 = arith.constant 3 : index

    // CHECK: [[IDX_2:%.+]] = arith.constant 2 : index

    %MATMUL = IE.MatMul(%IN1, %IN2) {transpose_a, transpose_b} : !InBoundedType1, !InBoundedType2 -> !OutBoundedType
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[IN1]], [[IN2]]) {transpose_a, transpose_b}

    %DIM_H = tensor.dim %MATMUL, %IDX_2 : !OutBoundedType
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[IN1]], [[IDX_2]]

    %DIM_W = tensor.dim %MATMUL, %IDX_3 : !OutBoundedType
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[IN2]], [[IDX_2]]

    return %MATMUL, %DIM_H, %DIM_W : !OutBoundedType, index, index
    // CHECK: return [[MATMUL]], [[DIM_H]], [[DIM_W]]
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

!InBoundedType = tensor<?x2xf32, {bounds = [3, 2], order = #NC}>
!OutBoundedType = tensor<2x?xf32, {bounds = [2, 3], order = #NC}>

// CHECK-LABEL: @Transpose2dFirstDim
func.func @Transpose2dFirstDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<?x2xf32

    %IDX_1 = arith.constant 1 : index
    // CHECK:   [[IDX_0:%.+]] = arith.constant 0 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #CN
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_1 = tensor.dim %TRANSPOSE, %IDX_1 : !OutBoundedType
    // CHECK:   [[DIM_0:%.+]] = tensor.dim [[IN]], [[IDX_0]]

    return %TRANSPOSE, %DIM_1 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_0]]
}


// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

!InBoundedType = tensor<2x?xf32, {bounds = [2, 3], order = #NC}>
!OutBoundedType = tensor<?x2xf32, {bounds = [3, 2], order = #NC}>

// CHECK-LABEL: @Transpose2dSecondDim
func.func @Transpose2dSecondDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<2x?xf32

    %IDX_0 = arith.constant 0 : index
    // CHECK:   [[IDX_1:%.+]] = arith.constant 1 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #CN
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_0 = tensor.dim %TRANSPOSE, %IDX_0 : !OutBoundedType
    // CHECK:   [[DIM_1:%.+]] = tensor.dim [[IN]], [[IDX_1]]

    return %TRANSPOSE, %DIM_0 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_1]]
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

!InBoundedType = tensor<?x?xf32, {bounds = [2, 3], order = #NC}>
!OutBoundedType = tensor<?x?xf32, {bounds = [3, 2], order = #NC}>

// CHECK-LABEL: @Transpose2dTwoDims
func.func @Transpose2dTwoDims(%IN: !InBoundedType) -> (!OutBoundedType, index, index) {
    // CHECK: [[IN:%.+]]: tensor<?x?xf32

    %IDX_0 = arith.constant 0 : index
    // CHECK:   [[IDX_0:%.+]] = arith.constant 0 : index

    %IDX_1 = arith.constant 1 : index
    // CHECK:   [[IDX_1:%.+]] = arith.constant 1 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #CN
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_0 = tensor.dim %TRANSPOSE, %IDX_0 : !OutBoundedType
    // CHECK:   [[DIM_1:%.+]] = tensor.dim [[IN]], [[IDX_1]]

    %DIM_1 = tensor.dim %TRANSPOSE, %IDX_1 : !OutBoundedType
    // CHECK:   [[DIM_0:%.+]] = tensor.dim [[IN]], [[IDX_0]]

    return %TRANSPOSE, %DIM_0, %DIM_1 : !OutBoundedType, index, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_1]], [[DIM_0]]
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

!InBoundedType = tensor<?x3x4xf32, {bounds = [2, 3, 4], order = #CHW}>
!OutBoundedType = tensor<3x4x?xf32, {bounds = [3, 4, 2], order = #CHW}>

// CHECK-LABEL: @Transpose3dFirstDim
func.func @Transpose3dFirstDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<?x3x4xf32

    %IDX_2 = arith.constant 2 : index
    // CHECK:   [[IDX_0:%.+]] = arith.constant 0 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #HWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_2 = tensor.dim %TRANSPOSE, %IDX_2 : !OutBoundedType
    // CHECK:   [[DIM_0:%.+]] = tensor.dim [[IN]], [[IDX_0]]

    return %TRANSPOSE, %DIM_2 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_0]]
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

!InBoundedType = tensor<2x?x4xf32, {bounds = [2, 3, 4], order = #CHW}>
!OutBoundedType = tensor<?x4x2xf32, {bounds = [3, 4, 2], order = #CHW}>

// CHECK-LABEL: @Transpose3dSecondDim
func.func @Transpose3dSecondDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<2x?x4xf32

    %IDX_0 = arith.constant 0 : index
    // CHECK:   [[IDX_1:%.+]] = arith.constant 1 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #HWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_0 = tensor.dim %TRANSPOSE, %IDX_0 : !OutBoundedType
    // CHECK:   [[DIM_1:%.+]] = tensor.dim [[IN]], [[IDX_1]]

    return %TRANSPOSE, %DIM_0 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_1]]
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

!InBoundedType = tensor<2x3x?xf32, {bounds = [2, 3, 4], order = #CHW}>
!OutBoundedType = tensor<3x?x2xf32, {bounds = [3, 4, 2], order = #CHW}>

// CHECK-LABEL: @Transpose3dThirdDim
func.func @Transpose3dThirdDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<2x3x?xf32

    %IDX_1 = arith.constant 1 : index
    // CHECK:   [[IDX_2:%.+]] = arith.constant 2 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #HWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_1 = tensor.dim %TRANSPOSE, %IDX_1 : !OutBoundedType
    // CHECK:   [[DIM_2:%.+]] = tensor.dim [[IN]], [[IDX_2]]

    return %TRANSPOSE, %DIM_1 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InBoundedType = tensor<?x3x4x5xf32, {bounds = [2, 3, 4, 5], order = #NCHW}>
!OutBoundedType = tensor<?x4x5x3xf32, {bounds = [2, 4, 5, 3], order = #NCHW}>

// CHECK-LABEL: @Transpose4dFirstDim
func.func @Transpose4dFirstDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<?x3x4x5xf32

    %IDX_0 = arith.constant 0 : index
    // CHECK:   [[IDX_0:%.+]] = arith.constant 0 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #NHWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_0 = tensor.dim %TRANSPOSE, %IDX_0 : !OutBoundedType
    // CHECK:   [[DIM_0:%.+]] = tensor.dim [[IN]], [[IDX_0]]

    return %TRANSPOSE, %DIM_0 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InBoundedType = tensor<2x?x4x5xf32, {bounds = [2, 3, 4, 5], order = #NCHW}>
!OutBoundedType = tensor<2x4x5x?xf32, {bounds = [2, 4, 5, 3], order = #NCHW}>

// CHECK-LABEL: @Transpose4dSecondDim
func.func @Transpose4dSecondDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<2x?x4x5xf32

    %IDX_3 = arith.constant 3 : index
    // CHECK:   [[IDX_1:%.+]] = arith.constant 1 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #NHWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_3 = tensor.dim %TRANSPOSE, %IDX_3 : !OutBoundedType
    // CHECK:   [[DIM_1:%.+]] = tensor.dim [[IN]], [[IDX_1]]

    return %TRANSPOSE, %DIM_3 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InBoundedType = tensor<2x3x?x5xf32, {bounds = [2, 3, 4, 5], order = #NCHW}>
!OutBoundedType = tensor<2x?x5x3xf32, {bounds = [2, 4, 5, 3], order = #NCHW}>

// CHECK-LABEL: @Transpose4dThirdDim
func.func @Transpose4dThirdDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<2x3x?x5xf32

    %IDX_1 = arith.constant 1 : index
    // CHECK:   [[IDX_2:%.+]] = arith.constant 2 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #NHWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_1 = tensor.dim %TRANSPOSE, %IDX_1 : !OutBoundedType
    // CHECK:   [[DIM_2:%.+]] = tensor.dim [[IN]], [[IDX_2]]

    return %TRANSPOSE, %DIM_1 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InBoundedType = tensor<2x3x4x?xf32, {bounds = [2, 3, 4, 5], order = #NCHW}>
!OutBoundedType = tensor<2x4x?x3xf32, {bounds = [2, 4, 5, 3], order = #NCHW}>

// CHECK-LABEL: @Transpose4dFourthDim
func.func @Transpose4dFourthDim(%IN: !InBoundedType) -> (!OutBoundedType, index) {
    // CHECK: [[IN:%.+]]: tensor<2x3x4x?xf32

    %IDX_2 = arith.constant 2 : index
    // CHECK:   [[IDX_3:%.+]] = arith.constant 3 : index

    %TRANSPOSE = IE.Transpose(%IN) {
        order_value = #NHWC
    } : !InBoundedType -> !OutBoundedType
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[IN]])

    %DIM_2 = tensor.dim %TRANSPOSE, %IDX_2 : !OutBoundedType
    // CHECK:   [[DIM_3:%.+]] = tensor.dim [[IN]], [[IDX_3]]

    return %TRANSPOSE, %DIM_2 : !OutBoundedType, index
    // CHECK:   return [[TRANSPOSE]], [[DIM_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x1x?x128xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>
!OutBoundedType = tensor<1x2x?x128xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>

// CHECK-LABEL: @ConcatOverChannelsDynamicHeight
func.func @ConcatOverChannelsDynamicHeight(
    %IN0: !InBoundedType,
    %IN1: !InBoundedType
) -> (!OutBoundedType, index) {
    // CHECK: [[IN0:%.+]]: tensor<1x1x?x128xf16, {{.*}}>, [[IN1:%.+]]: tensor<1x1x?x128xf16, {{.*}}>

    %IDX_2 = arith.constant 2 : index
    // CHECK:   [[IDX_2:%.+]] = arith.constant 2 : index

    %CONCAT = IE.Concat(%IN0, %IN1) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : !InBoundedType, !InBoundedType -> !OutBoundedType
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[IN0]], [[IN1]])

    %DIM_2 = tensor.dim %CONCAT, %IDX_2 : !OutBoundedType
    // CHECK:   [[DIM_2:%.+]] = tensor.dim [[IN0]], [[IDX_2]]

    return %CONCAT, %DIM_2 : !OutBoundedType, index
    // CHECK:   return [[CONCAT]], [[DIM_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x1x?x128xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>
!OutBoundedType = tensor<1x3x?x128xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>

// CHECK-LABEL: @ConcatOverChannelsDynamicHeightThreeInputs
func.func @ConcatOverChannelsDynamicHeightThreeInputs(
    %IN0: !InBoundedType,
    %IN1: !InBoundedType,
    %IN2: !InBoundedType
) -> (!OutBoundedType, index) {
    // CHECK: [[IN0:%.+]]: tensor<1x1x?x128xf16, {{.*}}>, [[IN1:%.+]]: tensor<1x1x?x128xf16, {{.*}}>, [[IN2:%.+]]: tensor<1x1x?x128xf16, {{.*}}>

    %IDX_2 = arith.constant 2 : index
    // CHECK:   [[IDX_2:%.+]] = arith.constant 2 : index

    %CONCAT = IE.Concat(%IN0, %IN1, %IN2) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : !InBoundedType, !InBoundedType, !InBoundedType -> !OutBoundedType
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[IN0]], [[IN1]], [[IN2]])

    %DIM_2 = tensor.dim %CONCAT, %IDX_2 : !OutBoundedType
    // CHECK:   [[DIM_2:%.+]] = tensor.dim [[IN0]], [[IDX_2]]

    return %CONCAT, %DIM_2 : !OutBoundedType, index
    // CHECK:   return [[CONCAT]], [[DIM_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x1x64x?xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>
!OutBoundedType = tensor<1x2x64x?xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>

// CHECK-LABEL: @ConcatOverChannelsDynamicWidth
func.func @ConcatOverChannelsDynamicWidth(
    %IN0: !InBoundedType,
    %IN1: !InBoundedType
) -> (!OutBoundedType, index) {
    // CHECK: [[IN0:%.+]]: tensor<1x1x64x?xf16, {{.*}}>, [[IN1:%.+]]: tensor<1x1x64x?xf16, {{.*}}>

    %IDX_3 = arith.constant 3 : index
    // CHECK:   [[IDX_3:%.+]] = arith.constant 3 : index

    %CONCAT = IE.Concat(%IN0, %IN1) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : !InBoundedType, !InBoundedType -> !OutBoundedType
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[IN0]], [[IN1]])

    %DIM_3 = tensor.dim %CONCAT, %IDX_3 : !OutBoundedType
    // CHECK:   [[DIM_3:%.+]] = tensor.dim [[IN0]], [[IDX_3]]

    return %CONCAT, %DIM_3 : !OutBoundedType, index
    // CHECK:   return [[CONCAT]], [[DIM_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x1x?x?xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>
!OutBoundedType = tensor<1x2x?x?xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>

// CHECK-LABEL: @ConcatOverChannelsDynamicHW
func.func @ConcatOverChannelsDynamicHW(
    %IN0: !InBoundedType,
    %IN1: !InBoundedType
) -> (!OutBoundedType, index, index) {
    // CHECK: [[IN0:%.+]]: tensor<1x1x?x?xf16, {{.*}}>, [[IN1:%.+]]: tensor<1x1x?x?xf16, {{.*}}>

    %IDX_2 = arith.constant 2 : index
    // CHECK-DAG:   [[IDX_2:%.+]] = arith.constant 2 : index

    %IDX_3 = arith.constant 3 : index
    // CHECK-DAG:   [[IDX_3:%.+]] = arith.constant 3 : index

    %CONCAT = IE.Concat(%IN0, %IN1) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : !InBoundedType, !InBoundedType -> !OutBoundedType
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[IN0]], [[IN1]])

    %DIM_2 = tensor.dim %CONCAT, %IDX_2 : !OutBoundedType
    // CHECK-DAG:   [[DIM_2:%.+]] = tensor.dim [[IN0]], [[IDX_2]]

    %DIM_3 = tensor.dim %CONCAT, %IDX_3 : !OutBoundedType
    // CHECK-DAG:   [[DIM_3:%.+]] = tensor.dim [[IN0]], [[IDX_3]]

    return %CONCAT, %DIM_2, %DIM_3 : !OutBoundedType, index, index
    // CHECK:   return [[CONCAT]], [[DIM_2]], [[DIM_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InBoundedType = tensor<1x1x?x128xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>
!OutBoundedType = tensor<1x2x?x128xf16, {bounds = [1, 1, 64, 128], order = #NCHW}>

// CHECK-LABEL: @ConcatWithOffsets
func.func @ConcatWithOffsets(
    %IN0: !InBoundedType,
    %IN1: !InBoundedType
) -> (!OutBoundedType, index) {
    // CHECK: [[IN0:%.+]]: tensor<1x1x?x128xf16, {{.*}}>, [[IN1:%.+]]: tensor<1x1x?x128xf16, {{.*}}>

    %IDX_2 = arith.constant 2 : index
    // CHECK:   [[IDX_2:%.+]] = arith.constant 2 : index

    %CONCAT = IE.Concat(%IN0, %IN1) {
        static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
    } : !InBoundedType, !InBoundedType -> !OutBoundedType
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[IN0]], [[IN1]])

    %DIM_2 = tensor.dim %CONCAT, %IDX_2 : !OutBoundedType
    // CHECK:   [[DIM_2:%.+]] = tensor.dim [[IN0]], [[IDX_2]]

    return %CONCAT, %DIM_2 : !OutBoundedType, index
    // CHECK:   return [[CONCAT]], [[DIM_2]]
}
