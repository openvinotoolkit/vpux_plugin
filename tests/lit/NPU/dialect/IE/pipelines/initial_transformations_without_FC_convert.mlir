//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --initial-transformations="convert-fc-to-conv=false" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @TransformationsWithoutFC
func.func @TransformationsWithoutFC(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.Convolution
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[FC:%.*]] = IE.FullyConnected(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK:       return [[FC]]
}

// -----

// CHECK-LABEL: @UnrollMatMulAndPropagate
func.func @UnrollMatMulAndPropagate(%arg0: tensor<1x8x4096x40xf32>, %arg1: tensor<1x8x4096x40xf32>) -> tensor<8x4096x4096xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x8x4096x40xf32>, tensor<1x8x4096x40xf32> -> tensor<1x8x4096x4096xf32>
    %1 = IE.Reshape(%0) {shape_value = [8, 4096, 4096]} : tensor<1x8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<8x4096x4096xf32> -> tensor<8x4096x4096xf32>
    return %2 : tensor<8x4096x4096xf32>

    // CHECK:       %[[FC:.+]] = IE.FullyConnected
    // CHECK:       %[[AFFINE:.+]] = IE.AffineReshape(%[[FC]])
    // CHECK-SAME{LITERAL}  {dim_mapping = [[0, 1], [2]], shape_value = [1, 4096, 4096]} : tensor<4096x4096xf32> -> tensor<1x4096x4096xf32>
    // CHECK:       %[[SOFTMAX:.+]] = IE.SoftMax(%[[AFFINE]])
    // CHECK:       %[[CONCAT:.+]] = IE.Concat(%[[SOFTMAX]],
    // CHECK:       return %[[CONCAT]] : tensor<8x4096x4096xf32>
}
