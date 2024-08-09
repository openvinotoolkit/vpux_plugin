//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-sparsity-ops="fuse-sparsify=false" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8:f16, 0.034255280214197492:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparsifyNCEPermute
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x1568x32xf16>
func.func @SparsifyNCEPermute(%arg0: tensor<1x3x1568x32xf16>) -> tensor<1x4x1568x32x!qElemType, {order = #NHWC}> {
    %sparsify = VPU.Sparsify(%arg0) : tensor<1x3x1568x32xf16> -> !VPU.SparseTensor<data=tensor<1x3x1568x32xf16>>
    %desparsify = VPU.Desparsify(%sparsify) : !VPU.SparseTensor<data=tensor<1x3x1568x32xf16>> -> tensor<1x3x1568x32xf16>

    %nce_permute = VPU.NCE.Permute(%desparsify) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64
    } -> tensor<1x4x1568x32x!qElemType, {order = #NHWC}>


    return %nce_permute : tensor<1x4x1568x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[SPARSIFY:%.+]] = VPU.Sparsify([[INPUT]]) : tensor<1x3x1568x32xf16>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x3x1568x32xf16>>

    // CHECK:       [[DESPARSIFY:%.+]] = VPU.Desparsify([[SPARSIFY]])

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[DESPARSIFY]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x4x1568x32x!qElemType, {order = #NHWC}>

    // CHECK:       return [[NCE_PERMUTE]] : tensor<1x4x1568x32x!qElemType, {order = #NHWC}>
}
