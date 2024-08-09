//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --recompute-sparsity-ptrs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RecomputePtrsForSparseNCEConv
func.func @RecomputePtrsForSparseNCEConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %sparse_map_cst = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %sparse_weights_cst = VPU.GroupSparseTensor(%weights_cst, %sparse_map_cst) {is_weights} -> !VPU.SparseTensor<data=tensor<16x16x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights>
    %weights_table_cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %1 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst, %weights_table_cst) {
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-DAG:                [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[1, 0, 1, 1]]], [[[1, 32, 1, 1]]], [[[1, 64, 1, 1]]], [[[1, 96, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 128, 1, 1]]], [[[1, 160, 1, 1]]], [[[1, 192, 1, 1]]], [[[1, 224, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 256, 1, 1]]], [[[1, 288, 1, 1]]], [[[1, 320, 1, 1]]], [[[1, 352, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 384, 1, 1]]], [[[1, 416, 1, 1]]], [[[1, 448, 1, 1]]], [[[1, 480, 1, 1]]]]> : tensor<16x1x1x4xsi32>
    // CHECK:                 %{{.+}} = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WEIGHTS_TABLE]]) {{{.+}}} -> tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DontChangePtrsForDenseNCEConv
func.func @DontChangePtrsForDenseNCEConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %1 = VPU.NCE.Convolution(%arg0, %weights_cst, %weights_table_cst) {
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:       %{{.+}} = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WEIGHTS_TABLE]]) {{{.+}}} -> tensor<1x16x16x16xf16, {order = #NHWC}>
}
