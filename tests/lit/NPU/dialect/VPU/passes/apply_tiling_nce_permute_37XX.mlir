//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @SplitNCEPermute(%arg0: tensor<1x31x224x224xf16>) -> tensor<1x32x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 32 : i64,
        opaque_ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 2, 1, 1]
    } -> tensor<1x32x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x32x224x224x!qElemType, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 224, 224]
    // CHECK-SAME:      : tensor<1x31x224x224xf16> to tensor<1x16x224x224xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Permute([[INPUT_TILE0]])
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 16 : i64
    // CHECK-SAME:          opaque_ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice %arg0 [0, 16, 0, 0] [1, 15, 224, 224]
    // CHECK-SAME:      : tensor<1x31x224x224xf16> to tensor<1x15x224x224xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Permute([[INPUT_TILE1]])
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 16 : i64,
    // CHECK-SAME:          opaque_ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 16, 0, 0]
    // CHECK-SAME:      -> tensor<1x32x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x224x224x!qElemType, {order = #NHWC}>

}
