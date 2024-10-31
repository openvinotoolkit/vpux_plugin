//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @HandleLargeKernelsConv

func.func @HandleLargeKernelsConv(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 22], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x16xf16> = dense<0.000000e+00> : tensor<1x1x1x16xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], %arg0, [[CST]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x16xf16>, tensor<1x1x1x32000xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 22] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>

    // CHECK: return [[Add1]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithPostOp
func.func @HandleLargeKernelsConvWithPostOp(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]]  = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_0:%.+]]  = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_1:%.+]]  = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 22], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_2:%.+]]  = const.Declare tensor<1x1x1x16xf16> = dense<0.000000e+00> : tensor<1x1x1x16xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_2]], %arg0, [[CST_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x16xf16>, tensor<1x1x1x32000xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 22] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>

    // CHECK: return [[Add1]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConvWithBias
func.func @HandleLargeKernelsConvWithBias(%arg0 : tensor<1x1x1x32000xf16>) -> tensor<1x64x1x2000xf16> {
    %cst = const.Declare tensor<64x1x1x33xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    %conv = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 16]} : tensor<1x1x1x32000xf16>, tensor<64x1x1x33xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x2000xf16>

    return %conv : tensor<1x64x1x2000xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x1x11xf16> = dense<1.000000e+00> : tensor<64x1x1x33xf16>, [#const.SubView<[0, 0, 0, 22], [64, 1, 1, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x16xf16> = dense<0.000000e+00> : tensor<1x1x1x16xf16>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_2]], %arg0, [[CST_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x16xf16>, tensor<1x1x1x32000xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 22] [1, 1, 1, 31995] : tensor<1x1x1x32032xf16> to tensor<1x1x1x31995xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]], [[CST_3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 16]} : tensor<1x1x1x31995xf16>, tensor<64x1x1x11xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x2000xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x2000xf16>, tensor<1x64x1x2000xf16> -> tensor<1x64x1x2000xf16>

    // CHECK: return [[Add1]] : tensor<1x64x1x2000xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv2DimsSplit
func.func @HandleLargeKernelsConv2DimsSplit(%arg0 : tensor<1x1x32000x32000xf16>) -> tensor<1x64x2001x2001xf16> {
    %cst = const.Declare tensor<64x1x22x22xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [16, 16], pads_end = [16, 16], strides = [16, 16]} : tensor<1x1x32000x32000xf16>, tensor<64x1x22x22xf16> -> tensor<1x64x2001x2001xf16>

    return %conv : tensor<1x64x2001x2001xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 11, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x22x22xf16>, [#const.SubView<[0, 0, 11, 11], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x32000x16xf16> = dense<0.000000e+00> : tensor<1x1x32000x16xf16>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x16x32032xf16> = dense<0.000000e+00> : tensor<1x1x16x32032xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[CST_3]], %arg0, [[CST_3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x32000x16xf16>, tensor<1x1x32000x32000xf16>, tensor<1x1x32000x16xf16> -> tensor<1x1x32000x32032xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], %0, [[CST_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x16x32032xf16>, tensor<1x1x32000x32032xf16>, tensor<1x1x16x32032xf16> -> tensor<1x1x32032x32032xf16>

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 11] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add2:%.+]] = IE.Add([[Add1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>

    // CHECK: return [[Add2]] : tensor<1x64x2001x2001xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsConv2DimsUnevenSplit
func.func @HandleLargeKernelsConv2DimsUnevenSplit(%arg0 : tensor<1x1x32000x32000xf16>) -> tensor<1x64x2001x2001xf16> {
    %cst = const.Declare tensor<64x1x18x18xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [16, 16], pads_end = [16, 16], strides = [16, 16]} : tensor<1x1x32000x32000xf16>, tensor<64x1x18x18xf16> -> tensor<1x64x2001x2001xf16>

    return %conv : tensor<1x64x2001x2001xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x11x11xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 0, 0], [64, 1, 11, 11]>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<64x1x11x7xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 0, 11], [64, 1, 11, 7]>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<64x1x7x11xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 11, 0], [64, 1, 7, 11]>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<64x1x7x7xf16> = dense<1.000000e+00> : tensor<64x1x18x18xf16>, [#const.SubView<[0, 0, 11, 11], [64, 1, 7, 7]>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x32000x16xf16> = dense<0.000000e+00> : tensor<1x1x32000x16xf16>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x16x32032xf16> = dense<0.000000e+00> : tensor<1x1x16x32032xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[CST_3]], %arg0, [[CST_3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x32000x16xf16>, tensor<1x1x32000x32000xf16>, tensor<1x1x32000x16xf16> -> tensor<1x1x32000x32032xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST_4]], [[CONCAT0]], [[CST_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x16x32032xf16>, tensor<1x1x32000x32032xf16>, tensor<1x1x16x32032xf16> -> tensor<1x1x32032x32032xf16>
    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 1, 32011, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32011xf16>
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[SLICEACT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32011xf16>, tensor<64x1x11x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 1, 32011, 32007] : tensor<1x1x32032x32032xf16> to tensor<1x1x32011x32007xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32011x32007xf16>, tensor<64x1x11x7xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add0:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 0] [1, 1, 32007, 32011] : tensor<1x1x32032x32032xf16> to tensor<1x1x32007x32011xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32007x32011xf16>, tensor<64x1x7x11xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add1:%.+]] = IE.Add([[Add0]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 11, 11] [1, 1, 32007, 32007] : tensor<1x1x32032x32032xf16> to tensor<1x1x32007x32007xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x1x32007x32007xf16>, tensor<64x1x7x7xf16> -> tensor<1x64x2001x2001xf16>
    // CHECK: [[Add2:%.+]] = IE.Add([[Add1]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x2001x2001xf16>, tensor<1x64x2001x2001xf16> -> tensor<1x64x2001x2001xf16>

    // CHECK: return [[Add2]] : tensor<1x64x2001x2001xf16>
}

// -----

// CHECK-LABEL: @HandleLargePrimeKernelsConvWithOneDimOnH
// CHECK-SAME: [[INPUT:%.+]]: tensor<32x1x80000x1xf16>
func.func @HandleLargePrimeKernelsConvWithOneDimOnH(%arg0 : tensor<32x1x80000x1xf16>) -> tensor<32x80x7975x1xf16> {
    %cst = const.Declare tensor<80x1x251x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [10, 1]} : tensor<32x1x80000x1xf16>, tensor<80x1x251x1xf16> -> tensor<32x80x7975x1xf16>
    return %conv : tensor<32x80x7975x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<80x10x3x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 250, 1]>, #const.Reshape<[80, 25, 10, 1]>, #const.SubView<[0, 22, 0, 0], [80, 3, 10, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<80x10x11x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 250, 1]>, #const.Reshape<[80, 25, 10, 1]>, #const.SubView<[0, 11, 0, 0], [80, 11, 10, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<80x10x11x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 250, 1]>, #const.Reshape<[80, 25, 10, 1]>, #const.SubView<[0, 0, 0, 0], [80, 11, 10, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<80x1x251x1xf16>, [#const.SubView<[0, 0, 250, 0], [80, 1, 1, 1]>]

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [32, 1, 79990, 1] : tensor<32x1x80000x1xf16> to tensor<32x1x79990x1xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[SLICEACT0]]) {shape_value = [32, 7999, 10, 1]} : tensor<32x1x79990x1xf16> -> tensor<32x7999x10x1xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<32x7999x10x1xf16> -> tensor<32x10x7999x1xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [32, 10, 7985, 1] : tensor<32x10x7999x1xf16> to tensor<32x10x7985x1xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x7985x1xf16>, tensor<80x10x11x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 11, 0] [32, 10, 7985, 1] : tensor<32x10x7999x1xf16> to tensor<32x10x7985x1xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x7985x1xf16>, tensor<80x10x11x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[ADD0:%.+]] = IE.Add([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x7975x1xf16>, tensor<32x80x7975x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 22, 0] [32, 10, 7977, 1] : tensor<32x10x7999x1xf16> to tensor<32x10x7977x1xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x7977x1xf16>, tensor<80x10x3x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[ADD1:%.+]] = IE.Add([[ADD0]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x7975x1xf16>, tensor<32x80x7975x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[SLICEACT4:%.+]] = IE.Slice [[INPUT]] [0, 0, 250, 0] [32, 1, 79741, 1] : tensor<32x1x80000x1xf16> to tensor<32x1x79741x1xf16>
    // CHECK: [[CONV4:%.+]] = IE.Convolution([[SLICEACT4]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [10, 1]} : tensor<32x1x79741x1xf16>, tensor<80x1x1x1xf16> -> tensor<32x80x7975x1xf16>
    // CHECK: [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x7975x1xf16>, tensor<32x80x7975x1xf16> -> tensor<32x80x7975x1xf16>

    // CHECK: return [[ADD2]] : tensor<32x80x7975x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargePrimeKernelsConvWithOneDimOnW
// CHECK-SAME: [[INPUT:%.+]]: tensor<32x1x1x80000xf16>
func.func @HandleLargePrimeKernelsConvWithOneDimOnW(%arg0 : tensor<32x1x1x80000xf16>) -> tensor<32x80x1x7975xf16> {
    %cst = const.Declare tensor<80x1x1x251xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 10]} : tensor<32x1x1x80000xf16>, tensor<80x1x1x251xf16> -> tensor<32x80x1x7975xf16>
    return %conv : tensor<32x80x1x7975xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<80x10x1x3xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 250]>, #const.Reshape<[80, 25, 1, 10]>, #const.SubView<[0, 22, 0, 0], [80, 3, 1, 10]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<80x10x1x11xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 250]>, #const.Reshape<[80, 25, 1, 10]>, #const.SubView<[0, 11, 0, 0], [80, 11, 1, 10]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<80x10x1x11xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 0], [80, 1, 1, 250]>, #const.Reshape<[80, 25, 1, 10]>, #const.SubView<[0, 0, 0, 0], [80, 11, 1, 10]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<80x1x1x1xf16> = dense<1.000000e+00> : tensor<80x1x1x251xf16>, [#const.SubView<[0, 0, 0, 250], [80, 1, 1, 1]>]

    // CHECK: [[SLICEACT0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [32, 1, 1, 79990] : tensor<32x1x1x80000xf16> to tensor<32x1x1x79990xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[SLICEACT0]]) {shape_value = [32, 7999, 1, 10]} : tensor<32x1x1x79990xf16> -> tensor<32x7999x1x10xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NWHC} : tensor<32x7999x1x10xf16> -> tensor<32x10x1x7999xf16>
    // CHECK: [[SLICEACT1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [32, 10, 1, 7985] : tensor<32x10x1x7999xf16> to tensor<32x10x1x7985xf16>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[SLICEACT1]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x1x7985xf16>, tensor<80x10x1x11xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[SLICEACT2:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 11] [32, 10, 1, 7985] : tensor<32x10x1x7999xf16> to tensor<32x10x1x7985xf16>
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[SLICEACT2]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x1x7985xf16>, tensor<80x10x1x11xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[ADD0:%.+]] = IE.Add([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x1x7975xf16>, tensor<32x80x1x7975xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[SLICEACT3:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 22] [32, 10, 1, 7977] : tensor<32x10x1x7999xf16> to tensor<32x10x1x7977xf16>
    // CHECK: [[CONV3:%.+]] = IE.Convolution([[SLICEACT3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x10x1x7977xf16>, tensor<80x10x1x3xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[ADD1:%.+]] = IE.Add([[ADD0]], [[CONV3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x1x7975xf16>, tensor<32x80x1x7975xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[SLICEACT4:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 250] [32, 1, 1, 79741] : tensor<32x1x1x80000xf16> to tensor<32x1x1x79741xf16>
    // CHECK: [[CONV4:%.+]] = IE.Convolution([[SLICEACT4]], [[CST_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 10]} : tensor<32x1x1x79741xf16>, tensor<80x1x1x1xf16> -> tensor<32x80x1x7975xf16>
    // CHECK: [[ADD2:%.+]] = IE.Add([[ADD1]], [[CONV4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<32x80x1x7975xf16>, tensor<32x80x1x7975xf16> -> tensor<32x80x1x7975xf16>

    // CHECK: return [[ADD2]] : tensor<32x80x1x7975xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelConvWithGCDWhenOneDimOnW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x80000xf16>
func.func @HandleLargeKernelConvWithGCDWhenOneDimOnW(%arg0 : tensor<1x1x1x80000xf16>) -> tensor<1x257x1x497xf16> {
    %cst = const.Declare tensor<257x1x1x512xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 160]} : tensor<1x1x1x80000xf16>, tensor<257x1x1x512xf16> -> tensor<1x257x1x497xf16>
    return %conv : tensor<1x257x1x497xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<257x32x1x5xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>, [#const.Reshape<[257, 16, 1, 32]>, #const.SubView<[0, 11, 0, 0], [257, 5, 1, 32]>, #const.Transpose<#NWHC>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<257x32x1x11xf16> = dense<1.000000e+00> : tensor<257x1x1x512xf16>, [#const.Reshape<[257, 16, 1, 32]>, #const.SubView<[0, 0, 0, 0], [257, 11, 1, 32]>, #const.Transpose<#NWHC>]
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 2500, 1, 32]} : tensor<1x1x1x80000xf16> -> tensor<1x2500x1x32xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NWHC} : tensor<1x2500x1x32xf16> -> tensor<1x32x1x2500xf16>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 32, 1, 2491] : tensor<1x32x1x2500xf16> to tensor<1x32x1x2491xf16>
    // CHECK: [[CONV_0:%.+]] = IE.Convolution([[SLICE_0]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 5]} : tensor<1x32x1x2491xf16>, tensor<257x32x1x11xf16> -> tensor<1x257x1x497xf16>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 11] [1, 32, 1, 2485] : tensor<1x32x1x2500xf16> to tensor<1x32x1x2485xf16>
    // CHECK: [[CONV_1:%.+]] = IE.Convolution([[SLICE_1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 5]} : tensor<1x32x1x2485xf16>, tensor<257x32x1x5xf16> -> tensor<1x257x1x497xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x257x1x497xf16>, tensor<1x257x1x497xf16> -> tensor<1x257x1x497xf16>
    // CHECK: return [[ADD]] : tensor<1x257x1x497xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelConvWithGCDWhenOneDimOnH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x80000x1xf16>
func.func @HandleLargeKernelConvWithGCDWhenOneDimOnH(%arg0 : tensor<1x1x80000x1xf16>) -> tensor<1x257x497x1xf16> {
    %cst = const.Declare tensor<257x1x512x1xf16> = dense<1.000000e+00> : tensor<257x1x512x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [160, 1]} : tensor<1x1x80000x1xf16>, tensor<257x1x512x1xf16> -> tensor<1x257x497x1xf16>
    return %conv : tensor<1x257x497x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<257x32x5x1xf16> = dense<1.000000e+00> : tensor<257x1x512x1xf16>, [#const.Reshape<[257, 16, 32, 1]>, #const.SubView<[0, 11, 0, 0], [257, 5, 32, 1]>, #const.Transpose<#NHCW>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<257x32x11x1xf16> = dense<1.000000e+00> : tensor<257x1x512x1xf16>, [#const.Reshape<[257, 16, 32, 1]>, #const.SubView<[0, 0, 0, 0], [257, 11, 32, 1]>, #const.Transpose<#NHCW>]
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 2500, 32, 1]} : tensor<1x1x80000x1xf16> -> tensor<1x2500x32x1xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<1x2500x32x1xf16> -> tensor<1x32x2500x1xf16>
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 32, 2491, 1] : tensor<1x32x2500x1xf16> to tensor<1x32x2491x1xf16>
    // CHECK: [[CONV_0:%.+]] = IE.Convolution([[SLICE_0]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [5, 1]} : tensor<1x32x2491x1xf16>, tensor<257x32x11x1xf16> -> tensor<1x257x497x1xf16>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 11, 0] [1, 32, 2485, 1] : tensor<1x32x2500x1xf16> to tensor<1x32x2485x1xf16>
    // CHECK: [[CONV_1:%.+]] = IE.Convolution([[SLICE_1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [5, 1]} : tensor<1x32x2485x1xf16>, tensor<257x32x5x1xf16> -> tensor<1x257x497x1xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x257x497x1xf16>, tensor<1x257x497x1xf16> -> tensor<1x257x497x1xf16>
    // CHECK: return [[ADD]] : tensor<1x257x497x1xf16>
}
