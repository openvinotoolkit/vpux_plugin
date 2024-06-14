//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @FoldPermuteCast
func.func @FoldPermuteCast(%arg0: tensor<1x1000x1x1xf32>) -> tensor<1x1000x1x1xf32> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x1000x1x1xf32> -> tensor<1x1000x1x1xf32>
    return %0 : tensor<1x1000x1x1xf32>

    // CHECK-NOT: IE.PermuteCast
    // CHECK:     return %arg0 : tensor<1x1000x1x1xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @FusePermuteCasts
func.func @FusePermuteCasts(%arg0: tensor<1x1000x1x1xf32>, %arg1: tensor<1x1000x1x1xf32, {order = #NHWC}>) ->
        (tensor<1x1x1x1000xf32>, tensor<1x1x1000x1xf32>) {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1000x1x1xf32> -> tensor<1x1x1000x1xf32>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCWH} :
        tensor<1x1x1000x1xf32> -> tensor<1x1x1x1000xf32>

    %2 = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NWCH} :
        tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32, {order = #NHWC}>
    %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1x1000x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    return %1, %3 : tensor<1x1x1x1000xf32>, tensor<1x1x1000x1xf32>

    // CHECK-NOT: IE.PermuteCast
    // CHECK-NOT: IE.PermuteCast
    // CHECK:     %[[VAL_0:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x1000x1x1xf32> -> tensor<1x1x1x1000xf32>
    // CHECK:     %[[VAL_1:.*]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    // CHECK:     return %[[VAL_0]], %[[VAL_1]] : tensor<1x1x1x1000xf32>, tensor<1x1x1000x1xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @FuseMemPermAndPermCast
func.func @FuseMemPermAndPermCast(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1000x1xf32> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} :
            tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32, {order = #NHWC}>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1x1000x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    return %1 : tensor<1x1x1000x1xf32>

    // CHECK:     %[[VAL_0:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    // CHECK:     return %[[VAL_0]] : tensor<1x1x1000x1xf32>
}
