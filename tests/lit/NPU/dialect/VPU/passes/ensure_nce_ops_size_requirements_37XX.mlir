//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NCEPermuteLargeWidth
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x32x8208xf16>
func.func @NCEPermuteLargeWidth(%arg0: tensor<1x3x32x8208xf16>) -> tensor<1x4x32x8208x!qElemType, {order = #NHWC}> {
    %nce_permute = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64
    } -> tensor<1x4x32x8208x!qElemType, {order = #NHWC}>

    return %nce_permute : tensor<1x4x32x8208x!qElemType, {order = #NHWC}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 3, 32, 4112] :
    // CHECK-SAME:      tensor<1x3x32x8208xf16> to tensor<1x3x32x4112xf16>

    // CHECK:       [[FIRST_NCE_PERM:%.*]] = VPU.NCE.Permute([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x4x32x4112x!qElemType, {order = #NHWC}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 4112] [1, 3, 32, 4096] :
    // CHECK-SAME:      tensor<1x3x32x8208xf16> to tensor<1x3x32x4096xf16>

    // CHECK:       [[SECOND_NCE_PERM:%.*]] = VPU.NCE.Permute([[SECOND_SLICE]])
    // CHECK-SAME:      -> tensor<1x4x32x4096x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_NCE_PERM]], [[SECOND_NCE_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4112]]}
    // CHECK-SAME:      : tensor<1x4x32x4112x!qElemType, {order = #NHWC}>, tensor<1x4x32x4096x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x32x8208x!qElemType, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x4x32x8208x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NCEPermuteLargeChannels
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x8204x32x32xf16>
func.func @NCEPermuteLargeChannels(%arg0: tensor<1x8204x32x32xf16>) -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}> {
    %nce_permute = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 8208 : i64
    } -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    return %nce_permute : tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 4104, 32, 32] :
    // CHECK-SAME:      tensor<1x8204x32x32xf16> to tensor<1x4104x32x32xf16>

    // CHECK:       [[FIRST_NCE_PERM:%.*]] = VPU.NCE.Permute([[FIRST_SLICE]])
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4104 : i64}
    // CHECK-SAME:      -> tensor<1x4104x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 4104, 0, 0] [1, 4100, 32, 32] :
    // CHECK-SAME:      tensor<1x8204x32x32xf16> to tensor<1x4100x32x32xf16>

    // CHECK:       [[SECOND_NCE_PERM:%.*]] = VPU.NCE.Permute([[SECOND_SLICE]])
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4104 : i64}
    // CHECK-SAME:      -> tensor<1x4104x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_NCE_PERM]], [[SECOND_NCE_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 4104, 0, 0]]}
    // CHECK-SAME:      : tensor<1x4104x32x32x!qElemType, {order = #NHWC}>, tensor<1x4104x32x32x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x8208x32x32x!qElemType, {order = #NHWC}>
}
