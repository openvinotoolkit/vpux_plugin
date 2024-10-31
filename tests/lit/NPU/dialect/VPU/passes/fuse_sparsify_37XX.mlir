//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-sparsity-ops="fuse-sparsify=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

!qElemType = !quant.uniform<u8:f16, 0.034255280214197492:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotFuseNCEPermute
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x1568x32xf16>
func.func @DoNotFuseNCEPermute(%arg0: tensor<1x3x1568x32xf16>) -> tensor<1x4x1568x32x!qElemType, {order = #NHWC}> {
    %nce_permute = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        opaque_ppe = #VPU.PPEStub<>
    } -> tensor<1x4x1568x32x!qElemType, {order = #NHWC}>

    %sparsify1 = VPU.Sparsify(%nce_permute) : tensor<1x4x1568x32x!qElemType, {order = #NHWC}>
        -> !VPU.SparseTensor<data=tensor<1x4x1568x32x!qElemType, {order = #NHWC}>>

    %desparsify = VPU.Desparsify(%sparsify1) : !VPU.SparseTensor<data=tensor<1x4x1568x32x!qElemType, {order = #NHWC}>>
        -> tensor<1x4x1568x32x!qElemType, {order = #NHWC}>

    return %desparsify : tensor<1x4x1568x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x4x1568x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[SPARSIFY:%.+]] = VPU.Sparsify([[NCE_PERMUTE]])
    // CHECK:       [[DESPARSIFY:%.+]] = VPU.Desparsify([[SPARSIFY]])
    // CHECK:       return [[DESPARSIFY]]
}
