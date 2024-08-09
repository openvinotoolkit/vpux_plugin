//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" --optimize-reorders %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwMultiplyHasChange
module @ReorderWithSwMultiplyHasChange {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
func.func @main(%arg0: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16>, tensor<1x32x28x1xf16> -> tensor<1x32x28x1xf16>

    return %2 : tensor<1x32x28x1xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16, {order = #NHWC}>, tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    // CHECK:       return [[VAR1]] : tensor<1x32x28x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ReorderInNHWCPowerOp
module @ReorderInNHWCPowerOp {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x4x5x10xf16>) -> tensor<1x4x5x10xf16> {
func.func @main(%arg0: tensor<1x4x5x10xf16>) -> tensor<1x4x5x10xf16> {
    %cst = const.Declare tensor<1x10x1x1xf16> = dense<[[[[9.997550e-02]], [[3.000490e-01]], [[5.000000e-01]], [[7.001950e-01]], [[8.999020e-01]], [[1.099610e+00]], [[1.299800e+00]], [[1.500000e+00]], [[1.700200e+00]], [[1.900390e+00]]]]> : tensor<1x10x1x1xf16>
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x5x10xf16> -> tensor<1x10x4x5xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x10x4x5xf16, {order = #NHWC}> -> tensor<1x10x4x5xf16>
    %2 = IE.Power(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x4x5xf16>, tensor<1x10x1x1xf16> -> tensor<1x10x4x5xf16>
    %3 = IE.PermuteCast(%2) {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x10x4x5xf16> -> tensor<1x4x5x10xf16, {order = #NWCH}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x4x5x10xf16, {order = #NWCH}> -> tensor<1x4x5x10xf16>
    return %4 : tensor<1x4x5x10xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x10x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      tensor<1x10x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:           [[VAR0:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x5x10xf16> -> tensor<1x10x4x5xf16, {order = #NHWC}>
    // CHECK:           [[VAR1:%.+]] = IE.Power([[VAR0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x4x5xf16, {order = #NHWC}>, tensor<1x10x1x1xf16, {order = #NHWC}> -> tensor<1x10x4x5xf16, {order = #NHWC}>
    // CHECK:           [[VAR2:%.+]] = IE.PermuteCast([[VAR1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x10x4x5xf16, {order = #NHWC}> -> tensor<1x4x5x10xf16>
    // CHECK:           return [[VAR2]] : tensor<1x4x5x10xf16>
}

}
