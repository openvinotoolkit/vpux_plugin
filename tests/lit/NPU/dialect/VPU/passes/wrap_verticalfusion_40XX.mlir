//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-in-vertical-fusion %s | FileCheck %s
// REQUIRES: arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontWrapD2SDMA
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>
func.func @DontWrapD2SDMA(%arg0: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %2 = VPU.DepthToSpace(%arg0) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 1, 1]}
        : tensor<1x64x128x128x!qElemType, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %2 : tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[INPUT]])
    // CHECK:       return [[D2S]]
}
