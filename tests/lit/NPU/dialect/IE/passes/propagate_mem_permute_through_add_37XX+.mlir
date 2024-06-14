//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-through-add %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertPermuteAddWithODUPermute
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x16x16x16xf16>, [[INPUT1:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertPermuteAddWithODUPermute(%arg0 : tensor<1x16x16x16xf16>, %arg1 : tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    return %ADD: tensor<1x16x16x16xf16>

    // CHECK: [[LHS_MEM_PERMUTE0:%.*]] = IE.MemPermute([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[RHS_MEM_PERMUTE0:%.*]] = IE.MemPermute([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[LHS_MEM_PERMUTE1:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[RHS_MEM_PERMUTE1:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[ADD:%.*]] = IE.Add([[LHS_MEM_PERMUTE1]], [[RHS_MEM_PERMUTE1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK: return [[PERMUTE_CAST]] : tensor<1x16x16x16xf16>

}
