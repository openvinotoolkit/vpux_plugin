//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-mem-permute-around-op %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @AdjustMemPermutesAroundMultiply
func.func @AdjustMemPermutesAroundMultiply(%arg0: tensor<1x1x51x1xf16, {order = #NCWH}>, %arg1: tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NCWH}>, tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NCWH}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x128x51x64xf16, {order = #NCWH}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    return %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE_CAST:%.*]] = IE.PermuteCast(%arg0)
    // CHECK:            {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x51x1xf16, {order = #NCWH}> -> tensor<1x1x51x1xf16, {order = #NHWC}>
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[PERMUTE_CAST]], %arg1)
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16, {order = #NHWC}>, tensor<1x128x51x64xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[MULTIPLY]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @AdjustMemPermutesAroundMultiplyWithoutDeadLoop
func.func @AdjustMemPermutesAroundMultiplyWithoutDeadLoop(%arg0: tensor<1x128x16x64xf16, {order = #NHWC}>, %arg1: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x128x16x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NWCH, mem_perm = #NWCH} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16, {order = #NWCH}>
    %1 = IE.PermuteCast(%arg1) {dst_order = #NWCH, mem_perm = #NHWC} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16, {order = #NWCH}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x128xf16, {order = #NWCH}>, tensor<1x16x1x128xf16, {order = #NWCH}> -> tensor<1x16x64x128xf16, {order = #NWCH}>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x64x128xf16, {order = #NWCH}> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    return %3 : tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:        [[IN_PERMUTE_CAST:%.*]] = IE.PermuteCast(%arg0)
    // CHECK:            {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    // CHECK:        [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1)
    // CHECK:            {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[IN_PERMUTE_CAST]], [[MEM_PERMUTE]])
    // CHECK:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x64x128xf16>
    // CHECK:        [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[MULTIPLY]])
    // CHECK:            {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x64x128xf16> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    // CHECK:        return [[OUT_PERMUTE_CAST]] : tensor<1x128x16x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotAdjustMemPermutesAroundMultiply
func.func @NotAdjustMemPermutesAroundMultiply(%arg0: tensor<1x1x51x1xf16>, %arg1: tensor<1x128x51x64xf16>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16>, tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    return %1 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x51x1xf16>, tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16>
    // CHECK:        [[PERMUTE:%.*]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
    // CHECK:        return [[PERMUTE]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// CHECK-LABEL: @NotAdjustMemPermutesLayoutNotSupport
func.func @NotAdjustMemPermutesLayoutNotSupport(%arg0: tensor<1x2x16x16xf16>, %arg1: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x16x16xf16, {order = #NHWC}>, tensor<1x2x16x16xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x2x16x16xf16, {order = #NHWC}>

    // CHECK:        [[PERMUTE0:%.*]] = IE.MemPermute(%arg0)
    // CHECK:        [[PERMUTE1:%.*]] = IE.MemPermute(%arg1)
    // CHECK:        [[MULTIPLY:%.*]] = IE.Multiply([[PERMUTE0]], [[PERMUTE1]])
    // CHECK:        return [[MULTIPLY]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}
