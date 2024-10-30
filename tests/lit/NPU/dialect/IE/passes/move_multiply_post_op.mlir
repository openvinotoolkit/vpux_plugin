//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-multiply-post-op="move-multiply-post-fc-for-dynamic-quant=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// -----

// CHECK-LABEL: @SwapMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @SwapMultiplyWithMatmul(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[INPUT1]]) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1x1024xf32>

    // CHECK:       return  [[MULTIPLY]] : tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @SwapTwoMultiplyWithMatmul
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x80x1024xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x80x1024xf32>
func.func @SwapTwoMultiplyWithMatmul(%arg0: tensor<1x32x80x1024xf32>, %arg1: tensor<1x32x80x1024xf32>) -> tensor<1x32x80x80xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>
    %cst0 = const.Declare tensor<1x1x1x1xf32> = dense<0.2> : tensor<1x1x1x1xf32>
    %1 = IE.Multiply(%arg1, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x1024xf32>

    %2 = IE.MatMul(%0, %1) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>

    return %2 : tensor<1x32x80x80xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-01> : tensor<1x1x1x1xf32>
    // CHECK:       [[MATMUL:%.*]]  = IE.MatMul([[INPUT1]], [[INPUT2]]) {transpose_b} : tensor<1x32x80x1024xf32>, tensor<1x32x80x1024xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[MATMUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf32>
    // CHECK:       [[MULTIPLY_0:%.*]] = IE.Multiply([[MULTIPLY]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x80x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x80x80xf3

    // CHECK:       return  [[MULTIPLY_0]] : tensor<1x32x80x80xf32>
}


// -----

// CHECK-LABEL: @NotSwapWithPostOp
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NotSwapWithPostOp(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NotSwapForNotSplatConst
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x2xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x2xf32>
func.func @NotSwapForNotSplatConst(%arg0: tensor<1x32x1024x2xf32>, %arg1: tensor<1x32x1x2xf32>) -> tensor<1x32x1x1024xf32> {
    %cst = const.Declare tensor<1x1x1x2xf32> = dense<[[[[1.0,1.6]]]]> : tensor<1x1x1x2xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x32x1024x2xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x2xf32>, tensor<1x32x1024x2xf32> -> tensor<1x32x1x1024xf32>

    return %1 : tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1x1024xf32>
}

// -----

// CHECK-LABEL: @NoSwapForMultiUser
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1x80xf32>
func.func @NoSwapForMultiUser(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1x80xf32>) -> (tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1x1024xf32>

    return %0, %1 : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MULTIPLY:%.*]], [[MATMUL]] : tensor<1x32x1024x80xf32>, tensor<1x32x1x1024xf32>
}


// -----

// CHECK-LABEL: @NotBeneficialForSwap
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x1024x80xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x32x1025x80xf32>
func.func @NotBeneficialForSwap(%arg0: tensor<1x32x1024x80xf32>, %arg1: tensor<1x32x1025x80xf32>) -> tensor<1x32x1025x1024xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.1> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x1x1x1xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1024x80xf32>
    %1 = IE.MatMul(%arg1, %0) {transpose_b} : tensor<1x32x1025x80xf32>, tensor<1x32x1024x80xf32> -> tensor<1x32x1025x1024xf32>

    return %1 : tensor<1x32x1025x1024xf32>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare
    // CHECK:       [[MULTIPLY:%.*]] = IE.Multiply([[INPUT1]], [[CST]])
    // CHECK:       [[MATMUL:%.*]] = IE.MatMul([[INPUT2]], [[MULTIPLY]]) {transpose_b}

    // CHECK:       return  [[MATMUL]] : tensor<1x32x1025x1024xf32>
}


// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MoveMultiplyPostFCChannelWise
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<4095x4096xsi4>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<4095x1xf16>,
// CHECK-SAME:      [[INPUT3:%.+]]: tensor<1x1x4096xf32>
func.func @MoveMultiplyPostFCChannelWise(%arg0: tensor<4095x4096xsi4>, %arg1: tensor<4095x1xf16>, %arg2: tensor<1x1x4096xf32>) -> tensor<1x4095xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<4095x4096xsi4> -> tensor<4095x4096xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4095x4096xf16>, tensor<4095x1xf16> -> tensor<4095x4096xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<4095x4096xf16> -> tensor<4095x4096xf32>
    %3 = IE.Reshape(%arg2) {shape_value = [1, 4096]} : tensor<1x1x4096xf32> -> tensor<1x4096xf32>
    %4 = IE.FullyConnected(%3, %2) : tensor<1x4096xf32>, tensor<4095x4096xf32> -> tensor<1x4095xf32>
    return %4 : tensor<1x4095xf32>

    // CHECK:       [[CONVERT_1:%.+]] = IE.Convert([[INPUT1]]) {dstElemType = f16} : tensor<4095x4096xsi4> -> tensor<4095x4096xf16>
    // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT2]]) {order_value = #CN} : tensor<4095x1xf16> -> tensor<1x4095xf16>
    // CHECK:       [[CONVERT_2:%.+]] = IE.Convert([[CONVERT_1]]) {dstElemType = f32} : tensor<4095x4096xf16> -> tensor<4095x4096xf32>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[INPUT3]]) {shape_value = [1, 4096]} : tensor<1x1x4096xf32> -> tensor<1x4096xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[RESHAPE_1]], [[CONVERT_2]]) : tensor<1x4096xf32>, tensor<4095x4096xf32> -> tensor<1x4095xf32>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[TRANSPOSE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4095xf32>, tensor<1x4095xf16> -> tensor<1x4095xf32>
    // CHECK:       return [[MULTIPLY]] : tensor<1x4095xf32>
}


// -----

// CHECK-LABEL: @NotMoveMultiplyForUnmatchedShape
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<4095x4096xsi4>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x4096xf16>,
// CHECK-SAME:      [[INPUT3:%.+]]: tensor<1x1x4096xf32>
func.func @NotMoveMultiplyForUnmatchedShape(%arg0: tensor<4095x4096xsi4>, %arg1: tensor<1x4096xf16>, %arg2: tensor<1x1x4096xf32>) -> tensor<1x4095xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<4095x4096xsi4> -> tensor<4095x4096xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4095x4096xf16>, tensor<1x4096xf16> -> tensor<4095x4096xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<4095x4096xf16> -> tensor<4095x4096xf32>
    %3 = IE.Reshape(%arg2) {shape_value = [1, 4096]} : tensor<1x1x4096xf32> -> tensor<1x4096xf32>
    %4 = IE.FullyConnected(%3, %2) : tensor<1x4096xf32>, tensor<4095x4096xf32> -> tensor<1x4095xf32>
    return %4 : tensor<1x4095xf32>

    // CHECK:       [[CONVERT_1:%.+]] = IE.Convert([[INPUT1]])
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT_1]], [[INPUT2]])
    // CHECK:       [[CONVERT_2:%.+]] = IE.Convert([[MULTIPLY]])
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT3]])
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[RESHAPE]], [[CONVERT_2]])
    // CHECK:       return [[FC]] : tensor<1x4095xf32>
}


// -----

// CHECK-LABEL: @NotMoveForMultiplyNotUsedForQuant
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<4096x4096xf32>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT3:%.+]]: tensor<1x1x4096xf32>
func.func @NotMoveForMultiplyNotUsedForQuant(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x1xf16>, %arg2: tensor<1x1x4096xf32>) -> tensor<1x4096xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<4096x4096xf32> -> tensor<4096x4096xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096xf16>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<4096x4096xf16> -> tensor<4096x4096xf32>
    %3 = IE.Reshape(%arg2) {shape_value = [1, 4096]} : tensor<1x1x4096xf32> -> tensor<1x4096xf32>
    %4 = IE.FullyConnected(%3, %2) : tensor<1x4096xf32>, tensor<4096x4096xf32> -> tensor<1x4096xf32>
    return %4 : tensor<1x4096xf32>

    // CHECK:       [[CONVERT_1:%.+]] = IE.Convert([[INPUT1]])
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT_1]], [[INPUT2]])
    // CHECK:       [[CONVERT_2:%.+]] = IE.Convert([[MULTIPLY]])
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT3]])
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[RESHAPE]], [[CONVERT_2]])
    // CHECK:       return [[FC]] : tensor<1x4096xf32>
}


// -----

// CHECK-LABEL: @MoveMultiplySubtractPostGather
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x10xsi32>, [[INPUT2:%.+]]: tensor<10000x1xf16>, [[INPUT3:%.+]]: tensor<10000x1xf16>, [[INPUT4:%.+]]: tensor<10000x3584xui8>
func.func @MoveMultiplySubtractPostGather(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x1xf16>, %arg3: tensor<10000x3584xui8>) -> tensor<1x10x3584xf32> {
    %0 = IE.Convert(%arg3) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Subtract(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Multiply(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<10000x3584xf16> -> tensor<10000x3584xf32>
    %4 = IE.Gather(%3, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf32>, tensor<1x10xsi32> -> tensor<1x10x3584xf32>
    return %4 : tensor<1x10x3584xf32>

    // CHECK:       [[GATHER_IN:%.+]] = IE.Gather([[INPUT4]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xui8>, tensor<1x10xsi32> -> tensor<1x10x3584xui8>
    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[GATHER_IN]]) {dstElemType = f16} : tensor<1x10x3584xui8> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_SUB:%.+]]  = IE.Gather([[INPUT2]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[SUBTRACT:%.+]]  = IE.Subtract([[CONVERT_IN]], [[GATHER_SUB]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_MUL:%.+]]  = IE.Gather([[INPUT3]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[MULTIPLY:%.+]]  = IE.Multiply([[SUBTRACT]], [[GATHER_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       [[CONVERT_OUT:%.+]]  = IE.Convert([[MULTIPLY]]) {dstElemType = f32} : tensor<1x10x3584xf16> -> tensor<1x10x3584xf32>
    // CHECK:       return [[CONVERT_OUT]] : tensor<1x10x3584xf32>
}


// -----

// CHECK-LABEL: @MoveMultiplyPostGather
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x10xsi32>, [[INPUT2:%.+]]: tensor<10000x1xf16>, [[INPUT3:%.+]]: tensor<10000x3584xui8>
func.func @MoveMultiplyPostGather(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xui8>) -> tensor<1x10x3584xf32> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<10000x3584xf16> -> tensor<10000x3584xf32>
    %3 = IE.Gather(%2, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf32>, tensor<1x10xsi32> -> tensor<1x10x3584xf32>
    return %3 : tensor<1x10x3584xf32>

    // CHECK:       [[GATHER_IN:%.+]] = IE.Gather([[INPUT3]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xui8>, tensor<1x10xsi32> -> tensor<1x10x3584xui8>
    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[GATHER_IN]]) {dstElemType = f16} : tensor<1x10x3584xui8> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_MUL:%.+]]  = IE.Gather([[INPUT2]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[MULTIPLY:%.+]]  = IE.Multiply([[CONVERT_IN]], [[GATHER_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       [[CONVERT_OUT:%.+]]  = IE.Convert([[MULTIPLY]]) {dstElemType = f32} : tensor<1x10x3584xf16> -> tensor<1x10x3584xf32>
    // CHECK:       return [[CONVERT_OUT]] : tensor<1x10x3584xf32>
}


// -----

// CHECK-LABEL: @MoveMultiplyPostGatherWithOutConvert
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x10xsi32>, [[INPUT2:%.+]]: tensor<10000x1xf16>, [[INPUT3:%.+]]: tensor<10000x3584xui8>
func.func @MoveMultiplyPostGatherWithOutConvert(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xui8>) -> tensor<1x10x3584xf16> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Gather(%1, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf16>, tensor<1x10xsi32> -> tensor<1x10x3584xf16>
    return %2 : tensor<1x10x3584xf16>

    // CHECK:       [[GATHER_IN:%.+]] = IE.Gather([[INPUT3]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xui8>, tensor<1x10xsi32> -> tensor<1x10x3584xui8>
    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[GATHER_IN]]) {dstElemType = f16} : tensor<1x10x3584xui8> -> tensor<1x10x3584xf16>
    // CHECK:       [[GATHER_MUL:%.+]]  = IE.Gather([[INPUT2]], [[INPUT1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x1xf16>, tensor<1x10xsi32> -> tensor<1x10x1xf16>
    // CHECK:       [[MULTIPLY:%.+]]  = IE.Multiply([[CONVERT_IN]], [[GATHER_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x3584xf16>, tensor<1x10x1xf16> -> tensor<1x10x3584xf16>
    // CHECK:       return [[MULTIPLY]] : tensor<1x10x3584xf16>
}


// -----

// CHECK-LABEL: @NotConvertIfNotUsedForQuant
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x10xsi32>, [[INPUT2:%.+]]: tensor<10000x1xf16>, [[INPUT3:%.+]]: tensor<10000x3584xf32>
func.func @NotConvertIfNotUsedForQuant(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xf32>) -> tensor<1x10x3584xf16> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xf32> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Gather(%1, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf16>, tensor<1x10xsi32> -> tensor<1x10x3584xf16>
    return %2 : tensor<1x10x3584xf16>

    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK:       [[GATHER:%.+]] = IE.Gather
    // CHECK:       return [[GATHER]] : tensor<1x10x3584xf16>
}


// -----

// CHECK-LABEL: @NotConvertForAxisNotZero
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x10xsi32>, [[INPUT2:%.+]]: tensor<10000x1xf16>, [[INPUT3:%.+]]: tensor<10000x3584xui8>
func.func @NotConvertForAxisNotZero(%arg0: tensor<1x10xsi32>, %arg1: tensor<10000x1xf16>, %arg2: tensor<10000x3584xui8>) -> tensor<10000x1x10xf16> {
    %0 = IE.Convert(%arg2) {dstElemType = f16} : tensor<10000x3584xui8> -> tensor<10000x3584xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<10000x3584xf16>, tensor<10000x1xf16> -> tensor<10000x3584xf16>
    %2 = IE.Gather(%1, %arg0) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<10000x3584xf16>, tensor<1x10xsi32> -> tensor<10000x1x10xf16>
    return %2 : tensor<10000x1x10xf16>

    // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK:       [[GATHER:%.+]] = IE.Gather
    // CHECK:       return [[GATHER]] : tensor<10000x1x10xf16>
}
