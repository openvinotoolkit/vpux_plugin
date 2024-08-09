//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-mem-permute-around-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @AdjustInputMemPermutesToOutput
func.func @AdjustInputMemPermutesToOutput(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16>, tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK:        [[PERMUTE:%.*]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[PERMUTE]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustInputMemPermutesWithMultipleMultiplyUsers
func.func @AdjustInputMemPermutesWithMultipleMultiplyUsers(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16>, tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK:        [[PERMUTE_0:%.*]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[PERMUTE_1:%.*]] = IE.MemPermute([[PERMUTE_0]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        [[ADD:%.*]] = IE.Add([[PERMUTE_0]], [[PERMUTE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[ADD]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}
