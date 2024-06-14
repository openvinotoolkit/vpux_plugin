//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --initial-transformations="convert-fc-to-conv=true" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @TransformPassesWithFC
func.func @TransformPassesWithFC(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 1, 1]} : tensor<1x16xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       [[VAL1:%.*]] = IE.Reshape([[WEIGHTS]]) {shape_value = [64, 16, 1, 1]} : tensor<64x16xf32> -> tensor<64x16x1x1xf32>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[BIAS]]) {shape_value = [1, 64, 1, 1]} : tensor<1x64xf32> -> tensor<1x64x1x1xf32>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[VAL0]], [[VAL1]], [[VAL2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       [[VAL3:%.*]] = IE.Reshape([[CONV]]) {shape_value = [1, 64]} : tensor<1x64x1x1xf32> -> tensor<1x64xf32>
    // CHECK:       return [[VAL3]]
}

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:      [[CST_0:%.*]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>, #const.Transpose<#CN>]
    // CHECK-DAG:      [[CST_1:%.*]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>, #const.Transpose<#CN>]
    // CHECK:          [[IN_1:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_1_2D:%.*]] = IE.AffineReshape([[IN_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:          [[IN_2:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_2_2D:%.*]] = IE.AffineReshape([[IN_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:          [[IN_1_4D:%.*]] = IE.Reshape([[IN_1_2D]]) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    // CHECK:          [[WEIGHTS_1:%.*]] = IE.Reshape([[CST_1]]) {shape_value = [40, 512, 1, 1]} : tensor<40x512xf32> -> tensor<40x512x1x1xf32>
    // CHECK:          [[CONV_1:%.*]] = IE.Convolution([[IN_1_4D]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf32>, tensor<40x512x1x1xf32> -> tensor<1x40x1x1xf32>
    // CHECK:          [[OUT_1_2D:%.*]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 40]} : tensor<1x40x1x1xf32> -> tensor<1x40xf32>

    // CHECK:          [[IN_2_4D:%.*]] = IE.Reshape([[IN_2_2D]]) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    // CHECK:          [[WEIGHTS_2:%.*]] = IE.Reshape([[CST_0]]) {shape_value = [40, 512, 1, 1]} : tensor<40x512xf32> -> tensor<40x512x1x1xf32>
    // CHECK:          [[CONV_2:%.*]] = IE.Convolution([[IN_2_4D]], [[WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf32>, tensor<40x512x1x1xf32> -> tensor<1x40x1x1xf32>
    // CHECK:          [[OUT_2_2D:%.*]] = IE.Reshape([[CONV_2]]) {shape_value = [1, 40]} : tensor<1x40x1x1xf32> -> tensor<1x40xf32>

    // CHECK:          [[OUT_1_4D:%.*]] = IE.AffineReshape([[OUT_1_2D]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf32> -> tensor<1x1x1x40xf32>
    // CHECK:          [[OUT_2_4D:%.*]] = IE.AffineReshape([[OUT_2_2D]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf32> -> tensor<1x1x1x40xf32>

    // CHECK:          [[CONCAT:%.*]] = IE.Concat([[OUT_1_4D]], [[OUT_2_4D]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32> -> tensor<1x2x1x40xf32>

    // CHECK return [[OUT]] : tensor<1x2x1x40xf32>
}

// CHECK-LABEL: @MatMulWithGroupQuant
func.func @MatMulWithGroupQuant(%arg0: tensor<16x96xf32>) -> tensor<16x64xf32> {
    %WEIGHTS = const.Declare tensor<3x32x64xf32> = dense<1.0> : tensor<3x32x64xf32>
    // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<1x32x64xf32> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x32x64xf32>, [#const.SubView<[0, 0, 0], [1, 32, 64]>]
    // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<1x32x64xf32> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x32x64xf32>, [#const.SubView<[1, 0, 0], [1, 32, 64]>]
    // CHECK-DAG:   [[WEIGHTS_2:%.*]] = const.Declare tensor<1x32x64xf32> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x32x64xf32>, [#const.SubView<[2, 0, 0], [1, 32, 64]>]

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<3x1x64xf32> = dense<-1.0> : tensor<3x1x64xf32>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x64xf32> = dense<-1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x64xf32>, [#const.SubView<[0, 0, 0], [1, 1, 64]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x64xf32> = dense<-1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x64xf32>, [#const.SubView<[1, 0, 0], [1, 1, 64]>]
    // CHECK-DAG:   [[OUT_LOW_2:%.*]] = const.Declare tensor<1x1x64xf32> = dense<-1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x64xf32>, [#const.SubView<[2, 0, 0], [1, 1, 64]>]

    %OUT_HIGH = const.Declare tensor<3x1x64xf32> = dense<1.0> : tensor<3x1x64xf32>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x64xf32> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x64xf32>, [#const.SubView<[0, 0, 0], [1, 1, 64]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x64xf32> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x64xf32>, [#const.SubView<[1, 0, 0], [1, 1, 64]>]
    // CHECK-DAG:   [[OUT_HIGH_2:%.*]] = const.Declare tensor<1x1x64xf32> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x64xf32>, [#const.SubView<[2, 0, 0], [1, 1, 64]>]

    %FQ = IE.FakeQuantize(%WEIGHTS, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 255 : i64
    } : tensor<3x32x64xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<3x1x64xf32>,
        tensor<3x1x64xf32>
            -> tensor<3x32x64xf32>

    // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize([[WEIGHTS_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_0]], [[OUT_HIGH_0]])
    // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[WEIGHTS_1]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_1]], [[OUT_HIGH_1]])
    // CHECK: [[FQ_2:%.*]] = IE.FakeQuantize([[WEIGHTS_2]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_2]], [[OUT_HIGH_2]])

    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [96, 64]
    } : tensor<3x32x64xf32> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_FQ_0:%.*]] = IE.AffineReshape([[FQ_0]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>

    // CHECK:   [[RESHAPE_FQ_1:%.*]] = IE.AffineReshape([[FQ_1]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>

    // CHECK:   [[RESHAPE_FQ_2:%.*]] = IE.AffineReshape([[FQ_2]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE) {
        order_value = #CN
    } : tensor<96x64xf32> -> tensor<64x96xf32>

    %GEMM = IE.FullyConnected(%arg0, %TRANSPOSE_RHS) : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[SLICE_0:%.*]] = IE.Slice %arg0 [0, 0] [16, 32] : tensor<16x96xf32> to tensor<16x32xf32>
    // CHECK:   [[SLICE_1:%.*]] = IE.Slice %arg0 [0, 32] [16, 32] : tensor<16x96xf32> to tensor<16x32xf32>
    // CHECK:   [[SLICE_2:%.*]] = IE.Slice %arg0 [0, 64] [16, 32] : tensor<16x96xf32> to tensor<16x32xf32>

    // CHECK:   [[TRANSPOSE_FQ_0:%.*]] = IE.Transpose([[RESHAPE_FQ_0]])
    // CHECK-SAME:  tensor<32x64xf32> -> tensor<64x32xf32>

    // CHECK:   [[SLICE_0_4D:%.*]] = IE.Reshape([[SLICE_0]]) {shape_value = [16, 32, 1, 1]} : tensor<16x32xf32> -> tensor<16x32x1x1xf32>
    // CHECK:   [[FQ_0_4D:%.*]] = IE.Reshape([[TRANSPOSE_FQ_0]]) {shape_value = [64, 32, 1, 1]} : tensor<64x32xf32> -> tensor<64x32x1x1xf32>
    // CHECK:   [[GEMM_0:%.*]] = IE.Convolution([[SLICE_0_4D]], [[FQ_0_4D]])
    // CHECK-SAME:  tensor<16x32x1x1xf32>, tensor<64x32x1x1xf32> -> tensor<16x64x1x1xf32>
    // CHECK:   [[GEMM_0_2D:%.*]] = IE.Reshape([[GEMM_0]]) {shape_value = [16, 64]} : tensor<16x64x1x1xf32> -> tensor<16x64xf32>

    // CHECK:   [[TRANSPOSE_FQ_1:%.*]] = IE.Transpose([[RESHAPE_FQ_1]])
    // CHECK-SAME:  tensor<32x64xf32> -> tensor<64x32xf32>

    // CHECK:   [[SLICE_1_4D:%.*]] = IE.Reshape([[SLICE_1]]) {shape_value = [16, 32, 1, 1]} : tensor<16x32xf32> -> tensor<16x32x1x1xf32>
    // CHECK:   [[FQ_1_4D:%.*]] = IE.Reshape([[TRANSPOSE_FQ_1]]) {shape_value = [64, 32, 1, 1]} : tensor<64x32xf32> -> tensor<64x32x1x1xf32>
    // CHECK:   [[GEMM_1:%.*]] = IE.Convolution([[SLICE_1_4D]], [[FQ_1_4D]])
    // CHECK-SAME:  tensor<16x32x1x1xf32>, tensor<64x32x1x1xf32> -> tensor<16x64x1x1xf32>
    // CHECK:   [[GEMM_1_2D:%.*]] = IE.Reshape([[GEMM_1]]) {shape_value = [16, 64]} : tensor<16x64x1x1xf32> -> tensor<16x64xf32>

    // CHECK:   [[TRANSPOSE_FQ_2:%.*]] = IE.Transpose([[RESHAPE_FQ_2]])
    // CHECK-SAME:  tensor<32x64xf32> -> tensor<64x32xf32>

    // CHECK:   [[SLICE_2_4D:%.*]] = IE.Reshape([[SLICE_2]]) {shape_value = [16, 32, 1, 1]} : tensor<16x32xf32> -> tensor<16x32x1x1xf32>
    // CHECK:   [[FQ_2_4D:%.*]] = IE.Reshape([[TRANSPOSE_FQ_2]]) {shape_value = [64, 32, 1, 1]} : tensor<64x32xf32> -> tensor<64x32x1x1xf32>
    // CHECK:   [[GEMM_2:%.*]] = IE.Convolution([[SLICE_2_4D]], [[FQ_2_4D]])
    // CHECK-SAME:  tensor<16x32x1x1xf32>, tensor<64x32x1x1xf32> -> tensor<16x64x1x1xf32>
    // CHECK:   [[GEMM_2_2D:%.*]] = IE.Reshape([[GEMM_2]]) {shape_value = [16, 64]} : tensor<16x64x1x1xf32> -> tensor<16x64xf32>

    // CHECK:   [[ADD_0:%.*]] = IE.Accumulate([[GEMM_0_2D]], [[GEMM_1_2D]])
    // CHECK:   [[ADD_1:%.*]] = IE.Accumulate([[ADD_0]], [[GEMM_2_2D]])

    return %GEMM : tensor<16x64xf32>
}
