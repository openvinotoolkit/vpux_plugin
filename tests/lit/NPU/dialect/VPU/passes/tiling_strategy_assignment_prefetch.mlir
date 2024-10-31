//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true"  --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: DISABLED
// TODO: #-81889 restore arch-NPU37XX and arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:3, {0.1:127, 0.2:127, 0.3:127, 0.4:127, 0.5:127, 0.6:127, 0.7:127, 0.8:127}>

// // 1x16x4x8xf16 + weights_table + profiling buffer
IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 4800 bytes of @CMX_NN
}

// CHECK-LABEL: func.func @MultiAxesAndPerAxisQuant
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x8x8x!qElemType, {order = #NHWC}>
func.func @MultiAxesAndPerAxisQuant(
        %input: tensor<1x32x8x8x!qElemType, {order = #NHWC}>)
            -> tensor<1x32x8x8x!qElemType, {order = #NHWC}> {
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%input, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-SAME:      = dense<1> : tensor<32x1x1x4xsi32>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:           kernel_size = [3, 3],
    // CHECK-SAME:           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:           strides = [1, 1],
    // CHECK-SAME:           tilingStrategy = [1, 1, 8, 1]}
    // CHECK-SAME:      -> tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x32x8x8x!qElemType, {order = #NHWC}>
}
