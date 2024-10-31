//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW ppe-version=IntPPE" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertNCEPermute
func.func @ConvertNCEPermute(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize

    // CHECK: [[PERMUTE_QUANTIZE:%.*]] = VPU.NCE.Permute(%arg0) {
    // CHECK-SAME: dstElemType = !qElemType,
    // CHECK-SAME: dstOrder = #NHWC,
    // CHECK-SAME: expandedChannels = 4 : i64
    // CHECK-SAME: } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK: return [[PERMUTE_QUANTIZE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}
